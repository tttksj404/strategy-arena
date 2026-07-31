#!/usr/bin/env python3
# Wave-37 addendum: is $100 the binding constraint, or is the signal simply bad?
#
# The walk-forward forced 608 of 1250 days flat because legs fell below the $5 exchange minimum, and
# the arithmetic showed the affordable cross-sectional width k collapses as capital shrinks. That
# raises a question with a definite answer: run the SAME causal walk-forward at larger capital, where
# the minimum-order constraint never binds, and see whether the strategy works.
#
# The two outcomes are both useful and neither is a judgement call:
#   - If it works at $1,000+, the finding is "this strategy needs capital, not a better signal", which
#     is actionable information about the $100 mandate rather than another rejection.
#   - If it fails at every size, the minimum-order constraint was a symptom and the signal itself has
#     no edge. That closes the cross-sectional funding family for good, at any capital.
#
# Nothing about selection changes: same grid, same causal schedule, same frozen score. Only the
# starting capital moves, so the comparison isolates capital.

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

import research.wave37_walkforward.engine37 as engine37
from research.wave37_walkforward.engine37 import MIN_LEG_USDT

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
# $10,000 already drives blocked_days to zero, so the constraint is fully relaxed there and a larger
# rung would add runtime without adding information.
CAPITAL_LADDER: Final = (100.0, 1_000.0, 10_000.0)


def run_at_capital(capital: float) -> dict:
    """Re-run the frozen causal walk-forward with a different starting capital.

    engine37.TOTAL_CAPITAL is a module constant used both by simulate() for training segments and by
    run_wave37.walk_forward() for the live sleeve, so it is patched for the duration of the call and
    restored afterwards. Everything else -- grid, schedule, score, cost model -- is untouched.
    """
    import research.wave37_walkforward.run_wave37 as runner

    original_engine = engine37.TOTAL_CAPITAL
    original_runner = runner.TOTAL_CAPITAL
    engine37.TOTAL_CAPITAL = capital
    runner.TOTAL_CAPITAL = capital
    try:
        return runner.walk_forward(cost_multiplier=1.0, verbose=False)
    finally:
        engine37.TOTAL_CAPITAL = original_engine
        runner.TOTAL_CAPITAL = original_runner


def main(argv: list[str] | None = None) -> int:
    # One rung takes ~160s and the sandbox aborts a shell call well before three of them finish, so
    # the probe is designed to be driven one rung per invocation:
    #     capital_probe.py --only 100      # runs (or reuses) a single rung, then exits
    #     capital_probe.py                 # runs whatever is missing, reusing cached rungs
    # Combined with the incremental JSON write below, a run can never lose completed work and no
    # single invocation is long enough to be killed.
    argv = sys.argv[1:] if argv is None else argv
    only: float | None = None
    if "--only" in argv:
        only = float(argv[argv.index("--only") + 1])

    started = time.time()
    print("=== 자본 규모 의존성: 최소주문 제약이 병목인가, 신호가 없는 것인가 ===")
    print(f"동일한 인과적 워크포워드(훈련365→적용90, 288조합 전수)를 자본만 바꿔 재실행")
    print(f"최소주문 ${MIN_LEG_USDT:.0f} · 레그당 = 자본 x 슬리브 x 레버리지 / (2k)\n")
    print(f"{'자본':>10} {'최종':>12} {'연환산':>9} {'MDD':>8} {'강제관망':>9} {'마진콜':>7} "
          f"{'펀딩':>10} {'가격':>11}")
    print("-" * 84)

    # Each rung takes ~160s and the sandbox caps a call at 8 minutes, so results are persisted after
    # every rung and completed rungs are reloaded on a re-run. That way an interrupted probe resumes
    # instead of restarting, and no rung is ever computed twice.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = RESULTS_DIR / "capital_probe.json"
    done: dict[float, dict] = {}
    if partial_path.exists():
        try:
            for row in json.loads(partial_path.read_text(encoding="utf-8")).get("ladder", []):
                done[float(row["capital"])] = row
        except (json.JSONDecodeError, KeyError):
            done = {}

    rows = []
    for capital in CAPITAL_LADDER:
        if only is not None and capital != only and capital not in done:
            continue
        if capital in done:
            row = done[capital]
            rows.append(row)
            print(
                f"${capital:>9,.0f} ${row['final']:>11,.2f} {row['annualised']:>+8.2%} "
                f"{row['mdd']:>7.2%} {row['blocked_days']:>8d}일 {row['margin_calls']:>6d}건 "
                f"${row['funding_usdt']:>+9,.2f} ${row['price_pnl_usdt']:>+10,.2f}   (캐시)",
                flush=True,
            )
            continue
        result = run_at_capital(capital)
        multiple = result["final_usdt"] / capital
        rows.append(
            {
                "capital": capital,
                "final": result["final_usdt"],
                "multiple": multiple,
                "annualised": result["annualised"],
                "mdd": result["mdd"],
                "blocked_days": result["blocked_days"],
                "margin_calls": result["margin_calls"],
                "funding_usdt": result["funding_usdt"],
                "price_pnl_usdt": result["price_pnl_usdt"],
                "cost_usdt": result["cost_usdt"],
                "min_leg": result["min_leg"],
                "max_leg": result["max_leg"],
            }
        )
        print(
            f"${capital:>9,.0f} ${result['final_usdt']:>11,.2f} {result['annualised']:>+8.2%} "
            f"{result['mdd']:>7.2%} {result['blocked_days']:>8d}일 {result['margin_calls']:>6d}건 "
            f"${result['funding_usdt']:>+9,.2f} ${result['price_pnl_usdt']:>+10,.2f}",
            flush=True,
        )
        partial_path.write_text(
            json.dumps({"wave": "wave37_capital_probe", "ladder": rows}, indent=2), encoding="utf-8"
        )

    if len(rows) < len(CAPITAL_LADDER):
        remaining = [c for c in CAPITAL_LADDER if c not in {r["capital"] for r in rows}]
        print(f"\n남은 구간 {['${:,.0f}'.format(c) for c in remaining]} — "
              f"'--only <자본>' 으로 이어서 실행하면 됩니다. ({time.time()-started:.0f}s)")
        return 0

    print("\n=== 해석 ===")
    blocked_gone = [r for r in rows if r["blocked_days"] == 0]
    profitable = [r for r in rows if r["annualised"] > 0]
    if blocked_gone:
        smallest = min(r["capital"] for r in blocked_gone)
        print(f"  최소주문 제약이 완전히 사라지는 자본: ${smallest:,.0f} 이상")
    else:
        print(f"  검정한 모든 자본에서 최소주문 제약이 일부 구간을 막았다")
    if profitable:
        labels = ", ".join("${:,.0f}".format(r["capital"]) for r in profitable)
        print(f"  수익이 나는 자본 구간: {labels}")
        print("  => 결론: 신호에는 edge가 있고 $100이 병목이다. 자본 확대가 해결책이다.")
    else:
        print("  검정한 모든 자본에서 연환산이 음수다 (최소주문 제약이 없는 구간에서도).")
        print("  => 결론: 최소주문 제약은 증상이었을 뿐이고 신호 자체에 edge가 없다.")
        print("     횡단면 펀딩 계열은 어떤 자본 규모에서도 닫혔다. $100은 병목이 아니다.")

    mdds = [r["mdd"] for r in rows]
    print(f"\n  MDD는 자본과 무관하게 {min(mdds):.1%}~{max(mdds):.1%} — 낙폭은 자본 문제가 아니다")

    payload = {
        "wave": "wave37_capital_probe",
        "question": "is the $100 mandate the binding constraint, or does the signal lack edge?",
        "min_leg_usdt": MIN_LEG_USDT,
        "ladder": rows,
        "any_profitable": bool(profitable),
        "conclusion": (
            "capital is the binding constraint; scaling up fixes it"
            if profitable
            else "signal has no edge at any capital; the minimum-order block was a symptom"
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "capital_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{time.time()-started:.0f}s · results/capital_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
