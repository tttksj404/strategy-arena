#!/usr/bin/env python3
"""Bitget wave1-rwa paper runner — hourly tick, no orders, no keys, stdlib only.

Slots mirror the backtest exactly: donchian(20h) both-ways on XAU (L5) and TQQQ
(L2), Wednesday-US-morning short on ETH (L10, ET buckets 82-84 held). Fills at
the live mid right after the hourly close, charged taker+spread+slippage per
side; funding events apply while a position is open; liquidation is checked
against each closed bar's extremes from an entry-anchored isolated-margin line.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"
TRADES = HERE / "trades.jsonl"
LOG = HERE / "paper.log"
API = "https://api.bitget.com"
TAKER, SLIP = 0.0006, 0.0001
MM = 0.005
SLOTS = [
    {"name": "xau_dch_L5", "symbol": "XAUUSDT", "kind": "donchian", "L": 5, "capital": 100.0, "spread": 0.0001},
    {"name": "tqqq_dch_L2", "symbol": "TQQQUSDT", "kind": "donchian", "L": 2, "capital": 50.0, "spread": 0.0004},
    {"name": "eth_wed_L10", "symbol": "ETHUSDT", "kind": "wed_short", "L": 10, "capital": 50.0, "spread": 0.0001},
]


def api(path: str, params: dict) -> list | dict:
    query = "&".join(f"{key}={value}" for key, value in params.items())
    with urllib.request.urlopen(f"{API}{path}?{query}", timeout=15) as response:
        payload = json.load(response)
    if payload.get("code") not in ("00000", 0):
        raise RuntimeError(f"bitget {path}: {payload.get('msg')}")
    return payload["data"]


def closed_candles(symbol: str, limit: int = 30) -> list[dict]:
    rows = api("/api/v2/mix/market/candles", {"symbol": symbol, "productType": "usdt-futures", "granularity": "1H", "limit": str(limit)})
    bars = [{"ts": int(row[0]), "open": float(row[1]), "high": float(row[2]), "low": float(row[3]), "close": float(row[4])} for row in rows]
    bars.sort(key=lambda bar: bar["ts"])
    now_hour = int(datetime.now(UTC).timestamp() // 3600 * 3600 * 1000)
    return [bar for bar in bars if bar["ts"] < now_hour]


def mid_price(symbol: str) -> float:
    row = api("/api/v2/mix/market/ticker", {"symbol": symbol, "productType": "usdt-futures"})[0]
    return (float(row["bidPr"]) + float(row["askPr"])) / 2


def funding_since(symbol: str, since_ms: int) -> float:
    rows = api("/api/v2/mix/market/history-fund-rate", {"symbol": symbol, "productType": "usdt-futures", "pageSize": "20"})
    return sum(float(row["fundingRate"]) for row in rows if int(row["fundingTime"]) > since_ms)


def desired_position(slot: dict, bars: list[dict]) -> int:
    if slot["kind"] == "donchian":
        if len(bars) < 21:
            return 0
        last, window = bars[-1], bars[-21:-1]
        upper, lower = max(bar["high"] for bar in window), min(bar["low"] for bar in window)
        if last["close"] > upper:
            return 1
        if last["close"] < lower:
            return -1
        return 9  # keep current side (donchian holds until the opposite break)
    hour = datetime.now(ZoneInfo("America/New_York"))
    bucket = hour.weekday() * 24 + hour.hour
    return -1 if bucket in (82, 83, 84) else 0


def side_cost(slot: dict) -> float:
    return TAKER + max(0.0001, slot["spread"]) + SLIP


def log(line: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {line}\n")


def tick() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"slots": {}, "started": datetime.now(UTC).isoformat()}
    for slot in SLOTS:
        entry = state["slots"].setdefault(slot["name"], {"equity": slot["capital"], "pos": 0, "entry_px": 0.0, "entry_ts": 0, "peak": slot["capital"], "liquidated": False})
        if entry["liquidated"] or entry["equity"] <= 1:
            continue
        try:
            bars = closed_candles(slot["symbol"])
            price = mid_price(slot["symbol"])
        except Exception as error:  # noqa: BLE001 - keep the loop alive on API hiccups
            log(f"{slot['name']} API_ERROR {error}")
            continue
        lev, pos = slot["L"], entry["pos"]
        if pos:
            liq = entry["entry_px"] * (1 - 1 / lev + MM) if pos > 0 else entry["entry_px"] * (1 + 1 / lev - MM)
            breached = [bar for bar in bars if bar["ts"] > entry["entry_ts"] and ((pos > 0 and bar["low"] <= liq) or (pos < 0 and bar["high"] >= liq))]
            if breached:
                entry.update(equity=0.0, pos=0, liquidated=True)
                log(f"{slot['name']} LIQUIDATED at {liq:.4g}")
                TRADES.open("a").write(json.dumps({"slot": slot["name"], "event": "liquidation", "ts": datetime.now(UTC).isoformat()}) + "\n")
                continue
        want = desired_position(slot, bars)
        if want == 9:
            want = pos if pos else 0
        if want != pos:
            if pos:
                gross = pos * lev * (price / entry["entry_px"] - 1)
                fund = 0.0
                try:
                    fund = funding_since(slot["symbol"], entry["entry_ts"]) * lev * pos
                except Exception:  # noqa: BLE001
                    pass
                entry["equity"] = max(0.0, entry["equity"] * (1 + gross) - entry["equity"] * lev * side_cost(slot) - entry["equity"] * fund)
                TRADES.open("a").write(json.dumps({"slot": slot["name"], "event": "close", "side": pos, "entry": entry["entry_px"], "exit": price, "equity": round(entry["equity"], 2), "ts": datetime.now(UTC).isoformat()}) + "\n")
            if want:
                entry["equity"] = max(0.0, entry["equity"] * (1 - lev * side_cost(slot)))
                entry.update(pos=want, entry_px=price, entry_ts=int(time.time() * 1000))
                TRADES.open("a").write(json.dumps({"slot": slot["name"], "event": "open", "side": want, "price": price, "equity": round(entry["equity"], 2), "ts": datetime.now(UTC).isoformat()}) + "\n")
            else:
                entry.update(pos=0, entry_px=0.0, entry_ts=0)
        mark = entry["equity"] * (1 + entry["pos"] * slot["L"] * (price / entry["entry_px"] - 1)) if entry["pos"] else entry["equity"]
        entry["mark"] = round(mark, 2)
        entry["peak"] = max(entry.get("peak", mark), mark)
        log(f"{slot['name']} pos={entry['pos']} px={price:.4g} equity={entry['equity']:.2f} mark={mark:.2f}")
    state["total_mark"] = round(sum(slot_state.get("mark", slot_state["equity"]) for slot_state in state["slots"].values()), 2)
    state["updated"] = datetime.now(UTC).isoformat()
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    import os
    if os.environ.get("PAPER_ONCE"):
        tick()
        return
    while True:
        try:
            tick()
        except Exception as error:  # noqa: BLE001 - a bad tick must not kill the service
            log(f"TICK_ERROR {error}")
        now = time.time()
        next_run = (int(now // 3600) + 1) * 3600 + 90
        time.sleep(max(60.0, next_run - now))


if __name__ == "__main__":
    main()
