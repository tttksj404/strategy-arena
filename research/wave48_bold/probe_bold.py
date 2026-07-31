#!/usr/bin/env python3
# Wave-48: accept ruin risk, and the objective changes shape.
#
# Every gate in this campaign has scored expected value subject to survival constraints -- wave31's
# candidate failed on exactly one gate, Q4, at a 22.52% chance of dropping below $50 while its MEDIAN
# outcome was $1,287.97 on $100. With ruin risk accepted, Q4 stops being a disqualifier and the question
# becomes a different one: not "what maximises expected return" but "what maximises the probability of
# reaching a target before going broke". Those have different answers, and the difference is not a matter
# of taste.
#
# The relevant classical result is Dubins & Savage's: in a SUBFAIR game, timid play (many small bets)
# drives the probability of ever reaching a target toward zero, while bold play (bet as much as the goal
# requires, as few times as possible) maximises it. wave47 measured that barrier trading here is subfair
# after costs -- hit rates fell 2.0 to 3.9 points short of breakeven. So if the goal is a fixed $30 rather
# than a rate, the mathematically correct approach is the opposite of what every previous wave searched
# for: fewer, larger, more decisive bets.
#
# The single most consequential lever turns out to be where the stop sits. wave47 tested tight stops
# (0.25x and 0.50x of the target distance) because those are what a risk-managed strategy uses. Widening
# the stop all the way to the liquidation price inverts the odds: at 20x, target +1.5% against a -5%
# liquidation gives a driftless hit probability of 5/(1.5+5) = 76.9%. That is a 77% chance of +$30 paid
# for with a 23% chance of losing the entire $100 -- still roughly fair before costs, but a completely
# different shape of bet, and the shape the stated objective actually asks for.
#
# This probe measures those probabilities on real data instead of assuming the driftless value, and then
# states plainly what repetition does to them.

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
TAKER_FEE: Final = 0.0006
ROUND_TRIP_COST: Final = 2.0 * TAKER_FEE
# Maintenance margin means liquidation arrives slightly before 1/leverage; 0.9 is a deliberately
# conservative haircut on the distance, so the measured win probability is understated rather than
# flattered.
LIQUIDATION_SAFETY: Final = 0.9
MAX_HOLD_BARS: Final = 24 * 30  # a month; a bet that has not resolved by then is recorded as unresolved


def measure(target_move: float, stop_move: float, max_hold: int = MAX_HOLD_BARS) -> dict:
    """Measured P(touch +target before -stop), long side, every 6th bar as an entry.

    Bars that touch both barriers are counted pessimistically -- as losses -- which is the opposite of
    wave47's optimistic reading. When the conclusion is "this gamble is available", the conservative
    reading is the honest one.
    """
    cache = build_market_cache()
    wins = losses = unresolved = 0
    for symbol in SYMBOLS:
        arrays = cache.arrays[symbol]
        high, low, close = arrays.high, arrays.low, arrays.close
        n = len(close)
        for entry in range(0, n - 1, 6):
            price = close[entry]
            if not np.isfinite(price) or price <= 0:
                continue
            up = price * (1.0 + target_move)
            down = price * (1.0 - stop_move)
            outcome = None
            for bar in range(entry + 1, min(entry + max_hold, n)):
                hit_down = low[bar] <= down
                hit_up = high[bar] >= up
                if hit_down:  # pessimistic: a bar touching both is a loss
                    outcome = "loss"
                    break
                if hit_up:
                    outcome = "win"
                    break
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                unresolved += 1
    resolved = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "win_probability": wins / resolved if resolved else float("nan"),
    }


def main() -> int:
    print("=== wave48: 파산 위험을 감수할 때의 최적 구조 ===")
    print("목적함수 변경: 기대값 최대화 -> 목표 도달 확률 최대화")
    print("근거: 열세 게임(subfair)에서 목표 도달 확률은 '소액 반복'이 아니라 '대담한 단발'로 최대화된다")
    print("      (Dubins-Savage). wave47이 이 게임이 비용 후 열세임을 실측했다.\n")

    print("=== 구조: 손절을 청산선까지 넓히면 확률이 역전된다 ===")
    print(f"{'lev':>4} {'노셔널':>8} {'목표변동':>9} {'청산거리':>9} {'무편향 승률':>11} "
          f"{'이기면':>8} {'지면':>8} {'기대값(비용전)':>14}")
    ladder = []
    for leverage in (5.0, 10.0, 20.0):
        notional = CAPITAL * leverage
        target_move = 30.0 / notional
        liquidation_move = LIQUIDATION_SAFETY / leverage
        fair = liquidation_move / (target_move + liquidation_move)
        cost = notional * ROUND_TRIP_COST
        win_amount = 30.0 - cost
        loss_amount = CAPITAL
        ev = fair * win_amount - (1.0 - fair) * loss_amount
        ladder.append({
            "leverage": leverage, "notional": notional, "target_move": target_move,
            "liquidation_move": liquidation_move, "fair_win_probability": fair,
            "cost_usd": cost, "win_usd": win_amount, "loss_usd": loss_amount, "ev_usd": ev,
        })
        print(f"{leverage:3.0f}x ${notional:7,.0f} {target_move:8.2%} {liquidation_move:8.2%} "
              f"{fair:10.1%} +${win_amount:6.2f} -${loss_amount:6.2f} ${ev:13.2f}")

    print("\n=> 손절을 청산선에 두면 승률이 77%(20x)까지 올라간다. 목표가 청산거리보다 훨씬 가깝기 때문이다.")
    print("   대가는 '지면 전액'이다. 비용 전 기대값은 거의 0 (공평), 비용 후에는 음수.")

    print("\n=== 실측 (BTC/ETH/SOL 1H, 양 배리어 동시 접촉은 '패배'로 보수 처리) ===")
    print(f"{'lev':>4} {'목표':>7} {'청산':>7} {'무편향':>8} {'실측 승률':>10} {'표본':>7} "
          f"{'미해결':>7} {'실측 기대값':>12}")
    measured = []
    for entry in ladder:
        result = measure(entry["target_move"], entry["liquidation_move"])
        probability = result["win_probability"]
        ev = probability * entry["win_usd"] - (1.0 - probability) * entry["loss_usd"]
        measured.append({**entry, **result, "measured_ev_usd": ev})
        print(f"{entry['leverage']:3.0f}x {entry['target_move']:6.2%} {entry['liquidation_move']:6.2%} "
              f"{entry['fair_win_probability']:7.1%} {probability:9.1%} "
              f"{result['wins']+result['losses']:6,} {result['unresolved']:6,} ${ev:11.2f}")

    best = max(measured, key=lambda row: row["win_probability"])
    print(f"\n=== 단발 승부: 최고 승률 조합 ===")
    print(f"  {best['leverage']:.0f}x · 목표 +{best['target_move']:.2%}(=$30) · 청산 -{best['liquidation_move']:.2%}")
    print(f"  실측 승률 {best['win_probability']:.1%} -> $130 · 실패 {1-best['win_probability']:.1%} -> $0")
    print(f"  기대값 ${best['measured_ev_usd']:+.2f} (비용 ${best['cost_usd']:.2f} 포함)")

    print("\n=== 반복하면 어떻게 되는가 (같은 승부를 n번) ===")
    probability = best["win_probability"]
    print(f"{'n회':>5} {'전부 생존 확률':>13} {'누적 수익(전승시)':>17} {'파산 확률':>10}")
    for n in (1, 3, 5, 10, 20, 50):
        survive = probability**n
        print(f"{n:4d}회 {survive:12.1%} ${30*n:16,} {1-survive:9.1%}")
    print("\n=> 반복은 승률을 지수적으로 깎는다. 10회면 생존 확률이 한 자리로 떨어진다.")
    print("   이것이 열세 게임에서 '소액 반복'이 파산으로 수렴하는 이유이고,")
    print("   목표가 정해져 있다면 '한 번에 끝내는' 편이 확률적으로 우월한 이유다.")

    print("\n=== 비교: 이미 검정된 고위험 후보 (wave31, Q4만 FAIL) ===")
    wave31 = json.loads(
        (Path("research/wave31_sprint/results/final.json")).read_text(encoding="utf-8")
    )
    q4 = wave31["gates"]["Q4_ruin"]
    q6 = wave31["gates"]["Q6_executability"]
    q7 = wave31["gates"]["Q7_short_horizon_consistency"]
    print(f"  레버리지 {q6['leverage']:.2f}x · 손절 {q6['stop_pct']:.2%} · 거래 {q6['n_trades']}회")
    print(f"  중앙 결과 ${q4['median_usdt']:,.2f} · 하위5% ${q4['p05_usdt']:.2f}")
    print(f"  파산($50 미달) 확률 {q4['ruin_probability']:.2%}  <- 이 게이트 하나로만 기각됐다")
    print(f"  30일 창 승률 {q7['oos_positive_share']:.1%} (OOS {q7['oos_windows_counted']}창)")
    print(f"  Q1·Q2·Q3·Q5·Q6·Q7 전부 PASS (deflated Sharpe {wave31['gates']['Q5_deflated_sharpe']['observed_sharpe']:.4f})")

    print("\n=== 두 선택지의 성격 차이 ===")
    print(f"  A) 단발 대담: {best['win_probability']:.0%} 확률로 +$30, {1-best['win_probability']:.0%} 확률로 -$100. 즉시 종결.")
    print(f"  B) wave31 반복: 중앙 ${q4['median_usdt']:,.0f}, 파산 {q4['ruin_probability']:.0%}, 수년 소요.")
    print("  A는 기대값이 음수이고 B는 양수다. A의 장점은 속도뿐이다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "bold.json").write_text(
        json.dumps({
            "wave": "wave48_bold",
            "objective": "maximise P(reach target) rather than expected value",
            "capital": CAPITAL,
            "target_usd": 30.0,
            "round_trip_cost": ROUND_TRIP_COST,
            "liquidation_safety": LIQUIDATION_SAFETY,
            "ladder": measured,
            "best_single_bet": best,
            "wave31_reference": {"ruin_probability": q4["ruin_probability"],
                                 "median_usdt": q4["median_usdt"], "p05_usdt": q4["p05_usdt"],
                                 "leverage": q6["leverage"], "n_trades": q6["n_trades"]},
        }, indent=2),
        encoding="utf-8",
    )
    print("\nresults/bold.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
