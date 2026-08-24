"""
PB104 — Liquidation cascade proxy fetcher.

Reality: Binance public `allForceOrders` endpoint is in maintenance, and
Coinglass / paid feeds require keys. We instead build a *proxy* using the
fully-public Binance futures statistics endpoints:

  - takerlongshortRatio  (5m taker buy vs sell volume)        → cascade flow
  - topLongShortAccountRatio (5m top-trader L/S position ratio) → position purge
  - globalLongShortAccountRatio (5m all-account L/S ratio)     → cross-check

Cascade hypothesis: a *long liquidation cascade* shows up as
  (a) sharp drop in long-ratio over a short window AND
  (b) heavy taker-sell dominance in the same window.
The mirror signal flags short-cascade reversals (LONG entry candidate).

Window: Binance only serves ~30d of 5m history → 30d backtest only.
Output: JSONL, one row per 5m bar, per symbol.

Usage:  python fetch_liquidations.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

BASE = "https://fapi.binance.com/futures/data"
OUTROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo\quant_runtime\liquidations")


def _get(endpoint: str, params: dict, retries: int = 3, sleep: float = 0.4) -> list[dict]:
    last = None
    for _ in range(retries):
        try:
            r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            last = (r.status_code, r.text[:200])
        except Exception as e:
            last = ("ERR", str(e))
        time.sleep(sleep)
    raise RuntimeError(f"{endpoint} failed: {last}")


def fetch_all(symbol: str, period: str = "5m", days: int = 30) -> list[dict]:
    """Fetch all three series and join on timestamp."""
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    # window = 500 candles per call → 5m * 500 = ~41h
    step = 500 * 5 * 60 * 1000
    by_ts: dict[int, dict] = {}

    for endpoint, key_map in [
        (
            "takerlongshortRatio",
            {"buyVol": "taker_buy_vol", "sellVol": "taker_sell_vol", "buySellRatio": "taker_bs_ratio"},
        ),
        (
            "topLongShortAccountRatio",
            {"longAccount": "top_long_acct", "shortAccount": "top_short_acct", "longShortRatio": "top_ls_ratio"},
        ),
        (
            "globalLongShortAccountRatio",
            {"longAccount": "g_long_acct", "shortAccount": "g_short_acct", "longShortRatio": "g_ls_ratio"},
        ),
    ]:
        cur = start
        while cur < end:
            chunk_end = min(cur + step, end)
            data = _get(
                endpoint,
                {"symbol": symbol, "period": period, "startTime": cur, "endTime": chunk_end, "limit": 500},
            )
            for row in data:
                ts = int(row["timestamp"])
                rec = by_ts.setdefault(ts, {"timestamp": ts, "symbol": symbol})
                for src, dst in key_map.items():
                    if src in row:
                        try:
                            rec[dst] = float(row[src])
                        except (TypeError, ValueError):
                            rec[dst] = None
            cur = chunk_end
            time.sleep(0.15)

    rows = sorted(by_ts.values(), key=lambda r: r["timestamp"])
    # Keep only fully-joined rows
    needed = {"taker_buy_vol", "taker_sell_vol", "top_ls_ratio", "g_ls_ratio"}
    rows = [r for r in rows if needed.issubset(r.keys())]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--period", default="5m")
    args = ap.parse_args()

    OUTROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    for sym in args.symbols.split(","):
        sym = sym.strip().upper()
        rows = fetch_all(sym, period=args.period, days=args.days)
        out = OUTROOT / f"{sym}_{args.period}_{args.days}d.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        first = rows[0]["timestamp"] if rows else None
        last = rows[-1]["timestamp"] if rows else None
        summary.append((sym, len(rows), first, last, str(out)))
        print(f"  {sym}: {len(rows)} rows -> {out.name}")

    print("\nSummary:")
    for sym, n, a, b, p in summary:
        print(f"  {sym:10s} n={n:5d}  {a} → {b}")


if __name__ == "__main__":
    main()
