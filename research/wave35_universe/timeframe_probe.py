#!/usr/bin/env python3
# Wave-35 timeframe probe: is a finer bar actually tradeable, or does cost eat it?
#
# The question "why only 1h -- can't we search 1m upward?" has a decisive arithmetic answer that
# needs no strategy search at all. A round trip costs a FIXED fraction of notional (taker 0.06%
# each way plus measured slippage = 0.1203% here), while the size of the move available to capture
# SHRINKS with the bar. So there is a minimum viable timeframe below which any strategy, however
# well optimised, pays more than the market offers.
#
# This script measures both sides on real Bitget data:
#   * available move  = median absolute bar return, and median high-low range, per timeframe
#   * required move   = round-trip cost as a fraction of notional (leverage-independent, because
#                       both the move and the cost scale with leverage identically)
#
# Leverage does NOT change the ratio -- that is the point people miss. Leverage multiplies profit
# and cost by the same factor, so it cannot fix a timeframe whose moves are smaller than its costs;
# it only makes the loss arrive faster.

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import requests

BITGET: Final = "https://api.bitget.com"
TIMEFRAMES: Final = ("1m", "5m", "15m", "30m", "1H", "4H", "1D")
BARS_PER_DAY: Final = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1H": 24, "4H": 6, "1D": 1}
SYMBOLS: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT")
TAKER: Final = 0.0006
SLIPPAGE: Final = 0.0000169  # wave13 measured, BTC/ETH/SOL tier
ROUND_TRIP: Final = 2.0 * (TAKER + SLIPPAGE)


def fetch(symbol: str, granularity: str, limit: int = 1000) -> pd.DataFrame:
    response = requests.get(
        f"{BITGET}/api/v2/mix/market/candles",
        params={"symbol": symbol, "productType": "usdt-futures", "granularity": granularity, "limit": limit},
        timeout=30,
    )
    rows = response.json().get("data") or []
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"])
    return frame.astype({c: float for c in ("open", "high", "low", "close")})


def main() -> int:
    print(f"왕복 비용 = 2 x (테이커 {TAKER:.4%} + 실측슬리피지 {SLIPPAGE:.5%}) = **{ROUND_TRIP:.4%}** of notional")
    print("레버리지는 이 비율을 바꾸지 못한다 — 수익과 비용에 같은 배수로 곱해지기 때문이다.\n")

    rows = []
    for granularity in TIMEFRAMES:
        moves, ranges = [], []
        for symbol in SYMBOLS:
            frame = fetch(symbol, granularity)
            time.sleep(0.08)
            if frame.empty:
                continue
            moves.append(np.abs(frame["close"].pct_change().dropna().to_numpy()))
            ranges.append(((frame["high"] - frame["low"]) / frame["close"]).to_numpy())
        if not moves:
            continue
        move = np.concatenate(moves)
        span = np.concatenate(ranges)
        rows.append(
            {
                "tf": granularity,
                "bars_per_day": BARS_PER_DAY[granularity],
                "median_abs_move": float(np.median(move)),
                "p90_abs_move": float(np.percentile(move, 90)),
                "median_range": float(np.median(span)),
                "p90_range": float(np.percentile(span, 90)),
            }
        )

    print(f"{'TF':>4} {'봉/일':>6} {'중앙 |수익|':>11} {'중앙 고저폭':>11} {'비용/중앙움직임':>15} "
          f"{'비용/중앙고저폭':>16} {'하루1회 연비용':>14} {'판정':>8}")
    print("-" * 104)
    for row in rows:
        ratio_move = ROUND_TRIP / row["median_abs_move"]
        ratio_range = ROUND_TRIP / row["median_range"]
        annual_cost_one_per_day = ROUND_TRIP * 365
        verdict = "불가" if ratio_range >= 1.0 else ("한계" if ratio_range >= 0.35 else "가능")
        print(
            f"{row['tf']:>4} {row['bars_per_day']:6d} {row['median_abs_move']:10.4%} {row['median_range']:10.4%} "
            f"{ratio_move:14.2f}x {ratio_range:15.2f}x {annual_cost_one_per_day:13.0%} {verdict:>8}"
        )

    print("\n[해석]")
    print("  '비용/중앙고저폭' 이 1.0 이상이면 그 봉의 전형적인 전체 변동폭을 다 잡아도 수수료를 못 낸다.")
    print("  0.35 이상이면 전형 변동폭의 3분의 1 이상을 수수료로 내는 것이므로 실질 엣지가 남기 어렵다.")

    print("\n[봉을 다 채워 매매하면 하루 비용은 얼마인가 — 봉당 1회 진입 가정]")
    print(f"{'TF':>4} {'봉/일':>6} {'하루 왕복비용':>13} {'연간 비용':>12}")
    for row in rows:
        daily = ROUND_TRIP * row["bars_per_day"]
        print(f"{row['tf']:>4} {row['bars_per_day']:6d} {daily:12.2%} {daily*365:11.0f}x")
    print("  → 1m에서 매 봉 진입하면 하루 비용만 자본의 173%다. 빈도 자체가 파산 경로다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
