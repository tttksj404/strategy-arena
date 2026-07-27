#!/usr/bin/env python3
"""Forward collector for wave-19 revival: live OI + funding + price snapshots.

wave-19 halted because Binance openInterestHist is 30-day only, so the spike-signal
lead-time test was underpowered/late-confirmation. SPEC's revival condition:
"collect live OI/liquidations/order-flow forward". This appends an hourly snapshot
(OI, funding, mark) for the Binance UM ∧ Bitget-listed liquid universe, building the
forward dataset the signal test needs. Analysis/promotion stays under wave-19 SPEC
once enough forward data exists — this only accumulates the raw feed.

Run:  python3 collect_oi_forward.py           (single snapshot)
      COLLECT_LOOP=1 python3 collect_oi_forward.py  (hourly loop)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "forward_data"
DATA.mkdir(exist_ok=True)
LOG = HERE / "collect_oi_forward.log"
UNIVERSE_CACHE = DATA / "universe.json"
BN = "https://fapi.binance.com"


def log(line: str) -> None:
    with LOG.open("a", encoding="utf-8") as h:
        h.write(f"{datetime.now(UTC).isoformat(timespec='seconds')} {line}\n")


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=15))


def build_universe(top: int = 80) -> list[str]:
    """Binance UM perp ∧ USDT-quoted, ranked by 24h quote volume; cached daily."""
    if UNIVERSE_CACHE.exists() and time.time() - UNIVERSE_CACHE.stat().st_mtime < 86400:
        return json.loads(UNIVERSE_CACHE.read_text())
    info = _get(f"{BN}/fapi/v1/exchangeInfo")
    perp = {s["symbol"] for s in info["symbols"]
            if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"}
    tickers = _get(f"{BN}/fapi/v1/ticker/24hr")
    ranked = sorted((t for t in tickers if t["symbol"] in perp),
                    key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
    syms = [t["symbol"] for t in ranked[:top]]
    UNIVERSE_CACHE.write_text(json.dumps(syms))
    return syms


def snapshot() -> None:
    syms = build_universe()
    ts = datetime.now(UTC)
    # bulk funding/mark for all symbols in one call
    prem = {p["symbol"]: p for p in _get(f"{BN}/fapi/v1/premiumIndex")}
    rows = []
    for sym in syms:
        try:
            oi = _get(f"{BN}/fapi/v1/openInterest?symbol={sym}")
            p = prem.get(sym, {})
            rows.append({"ts": ts.isoformat(), "symbol": sym,
                         "oi": float(oi.get("openInterest", 0)),
                         "mark": float(p.get("markPrice", 0)),
                         "funding": float(p.get("lastFundingRate", 0))})
        except Exception:  # noqa: BLE001 - one symbol failing must not drop the snapshot
            continue
        time.sleep(0.05)
    out = DATA / f"oi_{ts.strftime('%Y%m%d')}.jsonl"
    with out.open("a", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r) + "\n")
    log(f"snapshot {len(rows)}/{len(syms)} symbols -> {out.name}")


def main() -> None:
    if os.environ.get("COLLECT_LOOP") != "1":
        snapshot()
        return
    while True:
        try:
            snapshot()
        except Exception as err:  # noqa: BLE001
            log(f"SNAPSHOT_ERROR {err}")
        nxt = (int(time.time() // 3600) + 1) * 3600 + 60
        time.sleep(max(60.0, nxt - time.time()))


if __name__ == "__main__":
    main()
