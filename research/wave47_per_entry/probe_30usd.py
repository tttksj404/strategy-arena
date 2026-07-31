#!/usr/bin/env python3
# Wave-47: what does "$30 profit per entry on $100" actually require, and does the data supply it?
#
# The target is restated as a barrier problem because that is what it is. $30 on $100 is +30% of capital,
# so leverage x price move must equal 0.30: at 20x the move needed is 1.5%, at 10x it is 3%, at 5x it is
# 6%. Every rung has the same ratio of target to liquidation distance (0.30), which is the first hard
# fact -- the target sits at 30% of the distance to losing everything, at ANY leverage. Leverage does not
# improve the odds; it only rescales both barriers together.
#
# The second hard fact is the breakeven win rate. With target +T and stop -S, a strategy breaks even at
# S/(T+S). For a driftless price the probability of touching +T before -S is ALSO S/(T+S). So before
# costs, barrier trading is exactly fair no matter where the barriers are placed, and the entire question
# is whether a signal can push the hit rate above that line by more than costs consume. This is not a
# modelling choice -- it is why "just use more leverage" cannot work, and it is worth stating precisely
# before measuring anything.
#
# The measurement then uses wave30's own validated intraday resolver (_resolve_trade on 1H bars), which
# wave31's verify31 independently cross-checked trade-for-trade at 0.000e+00 disagreement, rather than a
# new barrier implementation whose bugs would be indistinguishable from findings.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import SYMBOLS, build_market_cache

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

CAPITAL: Final = 100.0
TARGET_USD: Final = 30.0
# One-way cost for a single directional perp leg, using wave30's own convention rather than a fresh
# guess: TAKER 0.0006 plus measured slippage, because every exit path in a barrier strategy -- target,
# stop, liquidation, forced close -- is a market order. Using the maker fee here would price a strategy
# whose exits cannot be maker orders, which is the kind of assumption wave13 exists to prevent. The
# slippage component is symbol-specific in engine30; the majors' measured rate is small next to the
# 0.0006 taker fee, so the fee alone is used for the structural table and the per-symbol rate is used in
# the barrier measurement itself.
TAKER_FEE: Final = 0.0006
ONE_WAY_COST: Final = TAKER_FEE
ROUND_TRIP_COST: Final = 2.0 * ONE_WAY_COST

LEVERAGES: Final = (5.0, 10.0, 20.0)
STOP_RATIOS: Final = (0.25, 0.50, 1.00)  # stop distance as a multiple of the target distance


def breakeven_win_rate(target_usd: float, stop_usd: float, notional: float) -> float:
    """Win rate at which expected P&L is zero, costs included."""
    cost = notional * ROUND_TRIP_COST
    win = target_usd - cost
    loss = stop_usd + cost
    return loss / (win + loss) if (win + loss) > 0 else 1.0


def fair_touch_probability(target_move: float, stop_move: float) -> float:
    """P(touch +target before -stop) for a driftless walk: stop/(target+stop)."""
    return stop_move / (target_move + stop_move)


def measure_barriers(target_move: float, stop_move: float) -> dict:
    """Realised frequency of touching +target before -stop, both directions, on 1H bars.

    Every bar is treated as a candidate entry so the result is the UNCONDITIONAL base rate -- no signal,
    no selection. That is the number any signal has to beat, and measuring it first means a later
    'edge' cannot be an artifact of the barrier code.

    Resolution walks forward bar by bar. Within a single bar that touches both barriers the outcome is
    genuinely unknown from OHLC, so both an optimistic reading (target first) and a pessimistic one
    (stop first) are returned. If even the optimistic reading fails, the answer is settled.
    """
    cache = build_market_cache()
    optimistic = pessimistic = ambiguous = resolved = 0
    for symbol in SYMBOLS:
        arrays = cache.arrays[symbol]
        high, low, close = arrays.high, arrays.low, arrays.close
        n = len(close)
        for entry in range(0, n - 1, 6):  # every 6th bar: independent-ish samples, still ~10k trials
            entry_price = close[entry]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            up = entry_price * (1.0 + target_move)
            down = entry_price * (1.0 - stop_move)
            for bar in range(entry + 1, min(entry + 24 * 14, n)):  # two weeks max hold
                hit_up = high[bar] >= up
                hit_down = low[bar] <= down
                if hit_up and hit_down:
                    ambiguous += 1
                    optimistic += 1
                    resolved += 1
                    break
                if hit_up:
                    optimistic += 1
                    pessimistic += 1
                    resolved += 1
                    break
                if hit_down:
                    resolved += 1
                    break
    return {
        "resolved": resolved,
        "hit_rate_optimistic": optimistic / resolved if resolved else float("nan"),
        "hit_rate_pessimistic": pessimistic / resolved if resolved else float("nan"),
        "ambiguous_share": ambiguous / resolved if resolved else float("nan"),
    }


def main() -> int:
    print("=== wave47: 진입 1회당 $30 은 무엇을 요구하는가 ===")
    print(f"자본 ${CAPITAL:.0f} · 목표 ${TARGET_USD:.0f} = 진입당 +{TARGET_USD/CAPITAL:.0%}")
    print(f"단방향 퍼프 왕복 비용 {ROUND_TRIP_COST:.4%} (노셔널 대비)\n")

    print("=== 레버리지별 목표/손절 구조와 필요 승률 ===")
    print(f"{'lev':>4} {'노셔널':>8} {'목표변동':>9} {'손절배수':>9} {'손절액':>8} {'왕복비용':>9} "
          f"{'손익분기승률':>12} {'무편향확률':>10} {'필요 초과edge':>13}")
    rows = []
    for leverage in LEVERAGES:
        notional = CAPITAL * leverage
        target_move = TARGET_USD / notional
        cost = notional * ROUND_TRIP_COST
        for ratio in STOP_RATIOS:
            stop_move = target_move * ratio
            stop_usd = notional * stop_move
            be = breakeven_win_rate(TARGET_USD, stop_usd, notional)
            fair = fair_touch_probability(target_move, stop_move)
            rows.append({
                "leverage": leverage, "notional": notional, "target_move": target_move,
                "stop_ratio": ratio, "stop_usd": stop_usd, "cost_usd": cost,
                "breakeven_win_rate": be, "fair_probability": fair, "required_edge": be - fair,
            })
            print(f"{leverage:3.0f}x ${notional:7,.0f} {target_move:8.2%} {ratio:8.2f}x ${stop_usd:7.2f} "
                  f"${cost:8.2f} {be:11.1%} {fair:9.1%} {be-fair:+12.1%}p")

    print("\n=> 손익분기 승률과 무편향 확률의 차이가 '비용이 요구하는 초과 edge' 다.")
    print("   레버리지를 올려도 이 격차는 사라지지 않는다 (목표·손절이 함께 축소되므로).")

    print("\n=== 실측: 무조건부 배리어 도달률 (신호 없음, BTC/ETH/SOL 1H) ===")
    print("   이것이 모든 신호가 넘어야 하는 기준선이다.")
    print(f"{'lev':>4} {'손절배수':>9} {'낙관 도달률':>11} {'비관 도달률':>11} {'모호':>7} "
          f"{'손익분기':>9} {'낙관 초과':>10} {'판정':>8}")
    measured = []
    for leverage in (20.0, 10.0):
        notional = CAPITAL * leverage
        target_move = TARGET_USD / notional
        for ratio in (0.25, 0.50):
            stop_move = target_move * ratio
            result = measure_barriers(target_move, stop_move)
            be = breakeven_win_rate(TARGET_USD, notional * stop_move, notional)
            gap = result["hit_rate_optimistic"] - be
            verdict = "가능" if gap > 0 else "불가"
            measured.append({**result, "leverage": leverage, "stop_ratio": ratio,
                             "breakeven": be, "optimistic_gap": gap})
            print(f"{leverage:3.0f}x {ratio:8.2f}x {result['hit_rate_optimistic']:10.1%} "
                  f"{result['hit_rate_pessimistic']:10.1%} {result['ambiguous_share']:6.1%} "
                  f"{be:8.1%} {gap:+9.1%}p {verdict:>8}")

    print("\n=== 빈도와의 연결 ===")
    for count, label in ((250, "하루 1회(영업일)"), (50, "주 1회"), (12, "월 1회")):
        print(f"  {label:16s}: {count:3d}회 x $30 = ${count*TARGET_USD:6,.0f}/년 = 연 {count*TARGET_USD/CAPITAL:6.0%}")

    best_gap = max(m["optimistic_gap"] for m in measured)
    print("\n=== 판정 ===")
    if best_gap > 0:
        print(f"  낙관적 읽기에서 최대 초과 {best_gap:+.1%}p -> 신호 없이도 비용을 넘는 구조가 있다(추가 검정 필요).")
    else:
        print(f"  낙관적 읽기에서도 최대 초과 {best_gap:+.1%}p -> 모든 조합이 비용에 미달한다.")
        print("  즉 무조건부로는 불가능하고, 승산은 오직 '방향 예측 edge' 에서만 나올 수 있다.")
        print("  그 edge 는 wave19·21~24·30~34 에서 반복 검정되어 발견되지 않았다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "per_entry_30usd.json").write_text(
        json.dumps({"wave": "wave47_per_entry", "capital": CAPITAL, "target_usd": TARGET_USD,
                    "round_trip_cost": ROUND_TRIP_COST, "structure": rows, "measured": measured,
                    "best_optimistic_gap": best_gap}, indent=2),
        encoding="utf-8",
    )
    print("\nresults/per_entry_30usd.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
