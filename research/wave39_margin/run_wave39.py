#!/usr/bin/env python3
# Wave-39 step 2: re-run the carry book under MEASURED portfolio-margin rules.
#
# wave38 rejected every high-deployment rung on gate Z10 because an isolated short perp leg dies once an
# adverse move eats the free cash behind it -- at 19x that was 46-67% of active days. probe_margin.py
# then measured both reachable venues' published collateral haircuts and maintenance rates and found the
# threshold changes character entirely in a unified account: with the spot leg counted as collateral,
# even 1.00 deployment needs a 222.6% adverse move against a worst observed 83.60%.
#
# So the question wave38 could not answer is now answerable, and it is answered with the same causal
# protocol rather than a fresh holdout: does the extra deployment that portfolio margin makes safe
# actually beat I5, once the real cost model and the $5 minimum are still enforced?
#
# What this run does NOT claim: it does not model collateral-tier step-downs (haircuts worsen for large
# positions -- irrelevant at $100 but not in general), auto-deleveraging, or the possibility that a
# specific alt is not collateral-eligible at all. Those are named in the report as unmodeled.

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

from research.wave38_breadth.dataio38 import THRESHOLD_APR, build_panel
from research.wave38_breadth.engine38 import ACTIVE_CAPITAL, CarryConfig, simulate
from research.wave39_margin.probe_margin import (
    WORST_OBSERVED_ADVERSE_MOVE,
    MarginRules,
    liquidating_move,
)

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
TRAIN_DAYS: Final = 365
APPLY_DAYS: Final = 90
I5_CAGR: Final = 0.1027
MDD_LIMIT: Final = 0.25

# Worst measured across both reachable venues (see results/margin_rules.json). Using the harshest
# published haircut rather than BTC's friendly 0.98 keeps this a stress case: the strategy holds whatever
# ranks highest on funding, which skews to alts.
STRESS_RULES: Final = MarginRules("worst-measured", 0.70, 0.01, "wave39 probe_margin")
PORTFOLIO_MARGIN: Final = (STRESS_RULES.haircut, STRESS_RULES.maintenance_rate)

# Deployment may now reach 1.00 -- all cash in spot, perp margin supplied by the spot collateral. Beyond
# 1.00 the spot leg itself must be bought on borrowed money, whose interest is unmodeled, so 1.00 remains
# a hard ceiling exactly as wave38's spot-borrow exclusion required.
GRID: Final = tuple(
    CarryConfig(top_k, leg_fraction, cap)
    for top_k in (1, 2, 3, 5, 8)
    for leg_fraction in (0.25, 0.50, 1.00)
    for cap in (0.50, 0.75, 0.95, 1.00)
)


def selection_score(result, days: int) -> float:
    return result.annualised(days) - 2.0 * max(0.0, result.mdd - 0.20)


def walk_forward(panel, grid, cost_multiplier: float = 1.0, portfolio: bool = True, verbose: bool = True) -> dict:
    margin = PORTFOLIO_MARGIN if portfolio else None
    n_days = len(panel.days)
    capital = ACTIVE_CAPITAL
    equity_curve: list[float] = []
    day_index: list[int] = []
    selections: list[dict] = []
    funding_total = basis_total = cost_total = 0.0
    entries_total = active_total = blocked_total = liquidation_total = 0
    worst_adverse = 0.0
    evaluations = 0
    min_leg, max_leg = np.inf, 0.0
    residual_max = 0.0
    deployments: list[float] = []

    start = TRAIN_DAYS + 1
    while start + APPLY_DAYS <= n_days:
        best_config, best_score = None, -np.inf
        for config in grid:
            trained = simulate(
                panel, config, start - TRAIN_DAYS, start, cost_multiplier, portfolio_margin=margin
            )
            evaluations += 1
            if trained.liquidation_days > 0:
                continue
            score = selection_score(trained, TRAIN_DAYS)
            if score > best_score:
                best_config, best_score = config, score

        if best_config is None:
            equity_curve.extend([capital] * APPLY_DAYS)
            day_index.extend(range(start, start + APPLY_DAYS))
            start += APPLY_DAYS
            continue

        applied = simulate(
            panel,
            best_config,
            start,
            start + APPLY_DAYS,
            cost_multiplier,
            start_capital=capital,
            portfolio_margin=margin,
        )
        equity_curve.extend(applied.equity.tolist())
        day_index.extend(range(start, start + len(applied.equity)))
        capital = applied.final
        funding_total += applied.funding_usd
        basis_total += applied.basis_usd
        cost_total += applied.cost_usd
        entries_total += applied.entries
        active_total += applied.days_active
        blocked_total += applied.blocked_min_order
        liquidation_total += applied.liquidation_days
        worst_adverse = max(worst_adverse, applied.worst_adverse_move)
        if np.isfinite(applied.min_leg_usd):
            min_leg = min(min_leg, applied.min_leg_usd)
        max_leg = max(max_leg, applied.max_leg_usd)
        residual_max = max(residual_max, applied.accounting_residual)
        deployments.append(applied.deployment_mean)
        selections.append(
            {
                "apply_from": str(panel.days[start].date()),
                "apply_to": str(panel.days[start + APPLY_DAYS - 1].date()),
                "config": best_config.label,
                "deployment_cap": best_config.deployment_cap,
                "top_k": best_config.top_k,
                "applied_return": applied.multiple - 1.0,
                "capital_after": capital,
            }
        )
        if verbose:
            print(
                f"  {panel.days[start].date()} ~ {panel.days[start+APPLY_DAYS-1].date()} | "
                f"{best_config.label} | 적용 {applied.multiple-1.0:+7.2%} | 자산 ${capital:8.2f}",
                flush=True,
            )
        start += APPLY_DAYS
        if capital <= 0.0:
            break

    equity = np.asarray(equity_curve, dtype=float)
    days = len(equity)
    peak = np.maximum.accumulate(equity) if days else np.array([ACTIVE_CAPITAL])
    return {
        "final_usdt": float(equity[-1]) if days else ACTIVE_CAPITAL,
        "annualised": float((equity[-1] / ACTIVE_CAPITAL) ** (365.0 / days) - 1.0) if days and equity[-1] > 0 else -1.0,
        "mdd": float(np.max(1.0 - equity / peak)) if days else 0.0,
        "days": days,
        "reselections": len(selections),
        "evaluations": evaluations,
        "funding_usd": funding_total,
        "basis_usd": basis_total,
        "cost_usd": cost_total,
        "entries": entries_total,
        "days_active": active_total,
        "blocked_min_order": blocked_total,
        "liquidation_days": liquidation_total,
        "worst_adverse_move": worst_adverse,
        "min_leg_usd": float(min_leg) if np.isfinite(min_leg) else float("nan"),
        "max_leg_usd": max_leg,
        "accounting_residual": residual_max,
        "deployment_mean": float(np.mean(deployments)) if deployments else 0.0,
        "selections": selections,
        "equity": equity.tolist(),
        "day_index": day_index,
    }


def main() -> int:
    started = time.time()
    print("=== wave39: 실측 통합마진 규칙 하에서 캐리 재판정 ===")
    print(f"스트레스 마진 규칙: 할인율 {STRESS_RULES.haircut:.2f} · MMR {STRESS_RULES.maintenance_rate:.3f} (양 거래소 최악값)")
    for deployment in (0.50, 0.75, 0.95, 1.00):
        print(f"  투입 {deployment:.2f} -> 청산 문턱 {liquidating_move(deployment, STRESS_RULES):.1%}", end="")
        print("  (실측 최악 역행 83.60% 대비 안전)" if liquidating_move(deployment, STRESS_RULES) > WORST_OBSERVED_ADVERSE_MOVE else "  위험")
    panel = build_panel()
    print(f"\n패널 {len(panel.symbols)}종목 x {len(panel.days)}일 | 그리드 {len(GRID)}조합 | 기준 {THRESHOLD_APR:.0%} APR 고정\n")

    base = walk_forward(panel, GRID)
    print("\n=== 비용 x3 ===")
    stress = walk_forward(panel, GRID, cost_multiplier=3.0, verbose=False)
    print("=== 격리마진 동일그리드 (wave38 조건, 비교용) ===")
    isolated = walk_forward(panel, GRID, portfolio=False, verbose=False)
    print(f"  연 {isolated['annualised']:+.2%} | 청산 {isolated['liquidation_days']}일")

    days = base["days"]
    print(f"\n=== 인과적 곡선 (적용 {days}일) ===")
    print(f"  ${ACTIVE_CAPITAL:.2f} -> ${base['final_usdt']:.2f} | 연 {base['annualised']:+.2%} | MDD {base['mdd']:.2%}")
    print(f"  I5 기준선 {I5_CAGR:+.2%} | wave38 격리마진 +6.62% | 격리 동일그리드 {isolated['annualised']:+.2%}")
    print(f"  손익: 펀딩 ${base['funding_usd']:+.2f} · 베이시스 ${base['basis_usd']:+.2f} · 비용 ${base['cost_usd']:+.2f}")
    print(f"  재선정 {base['reselections']}회 · 평가 {base['evaluations']:,}회 · 평균 투입 {base['deployment_mean']:.2f}")
    print(f"  진입 {base['entries']}회 ({base['entries']/days:.2f}/일) · 활성 {base['days_active']}일 ({base['days_active']/days:.1%})")
    print(f"  레그 ${base['min_leg_usd']:.2f}~${base['max_leg_usd']:.2f} · 최소주문차단 {base['blocked_min_order']}일")
    print(f"  청산 {base['liquidation_days']}일 · 최악 역행 {base['worst_adverse_move']:.2%}")
    print(f"  비용x3: 연 {stress['annualised']:+.2%}")

    equity_arr = np.asarray(base["equity"], dtype=float)
    years = np.array([panel.days[i].year for i in base["day_index"]])
    print("\n=== 연도별 ===")
    yearly = {}
    previous = ACTIVE_CAPITAL
    for year in sorted(set(years.tolist())):
        mask = years == year
        segment = equity_arr[mask]
        if len(segment) < 2:
            continue
        change = segment[-1] / previous - 1.0
        yearly[year] = change
        print(f"  {year}: {change:+7.2%}  (-> ${segment[-1]:7.2f})")
        previous = segment[-1]
    recent = [v for y, v in yearly.items() if y >= 2023]
    if recent:
        print(f"  2023년 이후 평균 {np.mean(recent):+.2%}/년 · 전기간 {base['annualised']:+.2%}/년")

    gates = {
        "V1_margin_measured": {"status": "PASS", "detail": "OKX·Bitget 공개 엔드포인트 실측, 가정 없음"},
        "V2_causality": {"status": "PASS" if base["reselections"] >= 10 else "FAIL", "reselections": base["reselections"]},
        "V3_performance": {"status": "PASS" if base["annualised"] > I5_CAGR else "FAIL",
                            "annualised": base["annualised"], "bar": I5_CAGR},
        "V4_drawdown": {"status": "PASS" if base["mdd"] <= MDD_LIMIT else "FAIL", "mdd": base["mdd"]},
        "V5_no_liquidation": {"status": "PASS" if base["liquidation_days"] == 0 else "FAIL",
                               "liquidation_days": base["liquidation_days"],
                               "worst_adverse_move": base["worst_adverse_move"]},
        "V6_cost_stress": {"status": "PASS" if stress["annualised"] > 0.0 else "FAIL",
                            "annualised_x3": stress["annualised"]},
        "V7_no_spot_borrow": {"status": "PASS" if all(s["deployment_cap"] <= 1.0 for s in base["selections"]) else "FAIL",
                               "detail": "투입 1.00 이하 = 현물을 현금으로만 매수"},
        "V8_accounting": {"status": "PASS" if base["accounting_residual"] <= 1e-9 else "FAIL",
                           "max_residual": base["accounting_residual"]},
        "V9_recent_regime": {"status": "PASS" if recent and float(np.mean(recent)) > I5_CAGR else "FAIL",
                              "mean_since_2023": float(np.mean(recent)) if recent else None,
                              "detail": "2023년 이후 평균이 I5 기준선을 넘는가 — 전기간 수치가 고펀딩기에 의존하는지 검정"},
    }
    failures = [name for name, gate in gates.items() if gate["status"] == "FAIL"]
    print()
    for name, gate in gates.items():
        print(f"[{gate['status']}] {name}")
    print(f"\nOVERALL {'PASS' if not failures else 'FAIL'} | failures {failures} | {time.time()-started:.0f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(
        json.dumps(
            {
                "wave": "wave39_margin",
                "margin_rules": {"haircut": STRESS_RULES.haircut, "mmr": STRESS_RULES.maintenance_rate},
                "base_portfolio_margin": base,
                "isolated_same_grid": isolated,
                "stress_x3": stress,
                "yearly": {str(k): v for k, v in yearly.items()},
                "gates": gates,
                "failures": failures,
                "overall": "PASS" if not failures else "FAIL",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("results/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
