#!/usr/bin/env python3
# Wave-54: can the verified carry edge be levered? Measured borrow cost says no.
#
# wave53 established two things that together look like an invitation. The carry signal is real -- 49.3
# sigma above an exhaustive-search null in the recent regime, where random selection LOSES money because it
# pays turnover for nothing. And the recent-regime optimum carries a maximum drawdown of just 0.77% against
# a 25% budget, roughly 32x of apparent headroom. A verified edge with that much risk budget unused is
# exactly where leverage belongs.
#
# wave39 had capped deployment at 1.00 -- all cash in spot, perp margin supplied by portfolio collateral --
# and excluded anything beyond as needing spot margin borrow whose interest was UNMODELED. That exclusion
# was honest but incomplete: an unmodeled cost is not a demonstrated obstacle. So this wave measures the
# rate and models it.
#
# OKX publishes it: USDT borrows at 0.00006864/hour, which is 60.13% annualised, and the median across 169
# currencies is 290.55%. Carry earns funding of roughly 15-35% APR on notional. Every borrowed dollar of
# notional therefore earns ~20% and costs ~60%.
#
# That is an arithmetic verdict, but arithmetic on a spreadsheet is how I5's 10.27% and wave50's +225% got
# believed, so it is run through the evaluator with the borrow charged day by day. The prediction is a
# monotonic decline in return as deployment rises above 1.00, and predicting the shape before looking is
# what makes the result a test rather than a description.

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

from research.wave38_breadth.dataio38 import build_panel, with_threshold
from research.wave38_breadth.engine38 import MIN_ORDER_USDT
from research.wave53_massive.fast53 import ACTIVE_CAPITAL, build_daily_series

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

# Measured 2026-07-31 from OKX api/v5/public/interest-rate-loan-quota.
USDT_BORROW_APR: Final = 0.6013
USDT_BORROW_DAILY: Final = USDT_BORROW_APR / 365.0
# Collateral haircut bounds the borrow: solvency at zero price move needs C > N(1-h), i.e. N < C/(1-h).
HAIRCUT_WORST: Final = 0.70  # wave39's worst measured, the honest stress value for an alt book
HAIRCUT_MAJORS: Final = 0.98  # BTC/ETH, for the most favourable case
THRESHOLD: Final = 0.15  # wave53's optimum in both periods
RECENT_YEAR: Final = 2023


def max_deployment_for(haircut: float) -> float:
    """N < C/(1-h): beyond this the book is insolvent before the price even moves."""
    return 1.0 / (1.0 - haircut)


def evaluate_levered(series, deployment: float, start: int, end: int, borrow_daily: float) -> dict:
    """Compound one deployment level, charging borrow on the portion above 1.0.

    Deployment D means spot notional D x capital. The first 1.0 is funded by cash; the excess (D - 1) is
    borrowed and accrues interest every day the book is held, whether or not it earns anything that day.
    Interest on a position is not optional the way a trade is.
    """
    k = max(series.top_k, 1)
    leg_eff = deployment / k
    capital = ACTIVE_CAPITAL
    peak = capital
    max_dd = 0.0
    borrowed_fraction = max(0.0, deployment - 1.0)
    min_leg = np.inf
    days_held = 0
    interest_paid = 0.0

    for index in range(start, end):
        if series.gap_prev[index] != 0.0:
            capital *= 1.0 + leg_eff * series.gap_prev[index]
        if series.turnover[index] != 0.0:
            capital *= 1.0 - leg_eff * series.turnover[index]
        if series.n_selected[index]:
            leg_notional = capital * leg_eff
            if leg_notional < MIN_ORDER_USDT:
                continue
            min_leg = min(min_leg, leg_notional)
            capital *= 1.0 + leg_eff * series.intraday_sum[index]
            capital *= 1.0 + leg_eff * series.funding_sum[index]
            if borrowed_fraction > 0.0:
                interest = capital * borrowed_fraction * borrow_daily
                capital -= interest
                interest_paid += interest
            days_held += 1
        peak = max(peak, capital)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - capital / peak)
        if capital <= 0.0:
            break

    years = (end - start) / 365.0
    return {
        "deployment": deployment,
        "final": float(capital),
        "annualised": float((max(capital, 1e-9) / ACTIVE_CAPITAL) ** (1.0 / years) - 1.0) if capital > 0 else -1.0,
        "mdd": float(max_dd),
        "days_held": days_held,
        "interest_paid": float(interest_paid),
        "min_leg": float(min_leg) if np.isfinite(min_leg) else float("nan"),
    }


def main() -> int:
    panel = with_threshold(build_panel(), THRESHOLD)
    n_days = len(panel.days)
    recent_start = next(i for i, day in enumerate(panel.days) if day.year >= RECENT_YEAR)

    print("=== wave54: 검증된 캐리 edge에 레버리지를 걸 수 있는가 ===")
    print(f"실측 차입: USDT 연 {USDT_BORROW_APR:.2%} (OKX, 2026-07-31)")
    print(f"캐리 펀딩 수취: 대략 15~35% APR -> 차입 1달러당 약 20% 벌고 60% 낸다")
    print(f"\n담보 할인율이 정하는 지급여력 상한 (x=0 에서도 파산하지 않는 최대 투입):")
    for haircut in (HAIRCUT_WORST, 0.90, HAIRCUT_MAJORS):
        print(f"  할인율 {haircut:.2f} -> 최대 투입 {max_deployment_for(haircut):.2f}x")
    print(f"\n예측: 투입 1.00 초과 구간에서 수익이 단조 감소한다 (차입 60% > 펀딩 20%)")
    print("      이 예측을 먼저 적기 때문에 아래는 서술이 아니라 검정이다.\n")

    rows = []
    for top_k in (2, 12):  # wave53's full-period and recent-regime optima
        series = build_daily_series(panel, top_k)
        for label, start in (("전기간", 1), ("2023년 이후", recent_start)):
            print(f"=== k{top_k} · {label} ===")
            print(f"{'투입':>6} {'연환산':>9} {'MDD':>8} {'최종$':>10} {'차입이자합':>11} {'지급여력':>9}")
            for deployment in (0.50, 1.00, 1.50, 2.00, 3.00, 3.33):
                result = evaluate_levered(series, deployment, start, n_days, USDT_BORROW_DAILY)
                solvent = "가능" if deployment <= max_deployment_for(HAIRCUT_WORST) else "불가"
                rows.append({**result, "top_k": top_k, "period": label, "solvent_worst_haircut": solvent})
                print(f"{deployment:5.2f}x {result['annualised']:+8.2%} {result['mdd']:7.2%} "
                      f"${result['final']:9.2f} ${result['interest_paid']:10.2f} {solvent:>9}")
            print()

    print("=== 판정 ===")
    for top_k in (2, 12):
        for label in ("전기간", "2023년 이후"):
            subset = [r for r in rows if r["top_k"] == top_k and r["period"] == label]
            best = max(subset, key=lambda r: r["annualised"])
            unlevered = next(r for r in subset if r["deployment"] == 1.00)
            print(f"  k{top_k} {label}: 최적 투입 {best['deployment']:.2f}x ({best['annualised']:+.2%}) "
                  f"vs 투입 1.00x ({unlevered['annualised']:+.2%})")
    levered_better = any(
        max((r for r in rows if r["top_k"] == k and r["period"] == p), key=lambda r: r["annualised"])["deployment"] > 1.0
        for k in (2, 12) for p in ("전기간", "2023년 이후")
    )
    if levered_better:
        print("\n  일부 구간에서 레버리지가 이득이다 -> 추가 검정 필요")
    else:
        print("\n  모든 구간에서 최적 투입이 1.00x 이하다.")
        print("  => 실측 차입비용(연 60.13%)이 캐리 수익(15~35%)을 초과하므로")
        print("     레버리지는 검증된 edge를 확대하지 못하고 잠식한다. 레버리지 축은 실측으로 닫힌다.")
        print("     wave39가 '미모델링'으로 제외했던 것이 이제 '측정하여 기각'으로 바뀐다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "borrow.json").write_text(
        json.dumps({
            "wave": "wave54_levered",
            "usdt_borrow_apr": USDT_BORROW_APR,
            "haircut_limits": {str(h): max_deployment_for(h) for h in (0.70, 0.90, 0.98)},
            "rows": rows,
            "levered_better": levered_better,
        }, indent=2),
        encoding="utf-8",
    )
    print("\nresults/borrow.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
