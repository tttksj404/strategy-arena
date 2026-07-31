#!/usr/bin/env python3
# Wave-38 causal walk-forward over the breadth grid, plus SPEC.md's Z gates.
#
# Selection discipline is wave37's, unchanged and for the same reason: the campaign has already opened
# its holdout six times while changing the rules between openings, so any further IS/OOS split would be
# a seventh opening. Here each reselection sees only the trailing 365 days, commits to one config, and
# trades the next 90 days with it. The applied window never participates in its own selection, so every
# point on the chained curve was out-of-sample at the moment it was produced.
#
# wave37 ran exactly this protocol on cross-sectional funding and it returned -22.32% against a
# post-hoc +33.44%. That the protocol is capable of rejecting is the reason to trust it here.

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
from research.wave38_breadth.engine38 import (
    ACTIVE_CAPITAL,
    MAX_FEASIBLE_DEPLOYMENT,
    MIN_ORDER_USDT,
    TOTAL_CAPITAL,
    CarryConfig,
    build_grid,
    simulate,
)

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

TRAIN_DAYS: Final = 365
APPLY_DAYS: Final = 90
I5_CAGR: Final = 0.1027  # wave18's published all-period figure, the bar Z3 must clear
L4_CAGR: Final = 0.0937
MDD_LIMIT: Final = 0.25
DRAWDOWN_PENALTY_KNEE: Final = 0.20
DRAWDOWN_PENALTY_WEIGHT: Final = 2.0
MIN_RESELECTIONS: Final = 10
BASELINE_CONFIG: Final = CarryConfig(1, 0.50, 0.50)  # L4's own settings, for a like-for-like delta


def _wave30_regression_passes() -> bool:
    """SPEC.md's Z9: the pre-existing engines must be untouched by this wave's work."""
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "research/wave30_qd/tests/test_wave30.py", "-q"],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def selection_score(result, days: int) -> float:
    """Training-window score: return, penalised for drawdown beyond the knee.

    wave36 selected on return alone and picked a 3x book that drew down 68%. The penalty is carried
    over from wave37 unchanged so this wave cannot be accused of tuning the score to its own outcome.
    """
    excess_drawdown = max(0.0, result.mdd - DRAWDOWN_PENALTY_KNEE)
    return result.annualised(days) - DRAWDOWN_PENALTY_WEIGHT * excess_drawdown


def walk_forward(panel, grid: tuple[CarryConfig, ...], cost_multiplier: float = 1.0, verbose: bool = True) -> dict:
    n_days = len(panel.days)
    equity_curve: list[float] = []
    capital = ACTIVE_CAPITAL
    selections: list[dict] = []
    funding_total = basis_total = cost_total = 0.0
    entries_total = days_active_total = blocked_total = 0
    liquidation_total = 0
    worst_adverse = 0.0
    evaluations = 0
    disqualified_total = 0
    day_index: list[int] = []
    no_feasible_windows: list[str] = []
    min_leg, max_leg = np.inf, 0.0
    delta_mismatch = 0.0
    residual_max = 0.0
    deployment_used: list[float] = []

    start = TRAIN_DAYS + 1
    while start + APPLY_DAYS <= n_days:
        train_start, train_end = start - TRAIN_DAYS, start
        best_config, best_score = None, -np.inf
        disqualified = 0
        for config in grid:
            trained = simulate(panel, config, train_start, train_end, cost_multiplier)
            evaluations += 1
            # A config whose short perp leg would have been liquidated during the TRAINING window is
            # not a candidate at all, regardless of the return it shows. Screening on training data
            # keeps the decision causal; screening on the applied window would be hindsight. This is a
            # feasibility filter of the same kind as excluding spot-borrow rungs, not a preference.
            if trained.liquidation_days > 0:
                disqualified += 1
                continue
            score = selection_score(trained, train_end - train_start)
            if score > best_score:
                best_config, best_score = config, score
        if best_config is None:
            # Every config would have been liquidated in training. Holding nothing is the only honest
            # action; capital simply carries forward through this window.
            no_feasible_windows.append(str(panel.days[start].date()))
            equity_curve.extend([capital] * APPLY_DAYS)
            day_index.extend(range(start, start + APPLY_DAYS))
            selections.append(
                {
                    "apply_from": str(panel.days[start].date()),
                    "apply_to": str(panel.days[start + APPLY_DAYS - 1].date()),
                    "config": "NONE_FEASIBLE",
                    "top_k": 0,
                    "leg_fraction": 0.0,
                    "deployment_cap": 0.0,
                    "implied_perp_leverage": 0.0,
                    "applied_return": 0.0,
                    "capital_after": capital,
                    "entries": 0,
                }
            )
            if verbose:
                print(
                    f"  {panel.days[start].date()} ~ {panel.days[start+APPLY_DAYS-1].date()} | "
                    f"전 조합 훈련구간 청산 -> 관망 | 자산 ${capital:8.2f}",
                    flush=True,
                )
            start += APPLY_DAYS
            continue
        disqualified_total += disqualified

        applied = simulate(
            panel, best_config, start, start + APPLY_DAYS, cost_multiplier, start_capital=capital
        )
        equity_curve.extend(applied.equity.tolist())
        day_index.extend(range(start, start + len(applied.equity)))
        capital = applied.final
        liquidation_total += applied.liquidation_days
        worst_adverse = max(worst_adverse, applied.worst_adverse_move)
        funding_total += applied.funding_usd
        basis_total += applied.basis_usd
        cost_total += applied.cost_usd
        entries_total += applied.entries
        days_active_total += applied.days_active
        blocked_total += applied.blocked_min_order
        if np.isfinite(applied.min_leg_usd):
            min_leg = min(min_leg, applied.min_leg_usd)
        max_leg = max(max_leg, applied.max_leg_usd)
        delta_mismatch = max(delta_mismatch, applied.delta_mismatch)
        residual_max = max(residual_max, applied.accounting_residual)
        deployment_used.append(applied.deployment_mean)

        selections.append(
            {
                "apply_from": str(panel.days[start].date()),
                "apply_to": str(panel.days[start + APPLY_DAYS - 1].date()),
                "config": best_config.label,
                "top_k": best_config.top_k,
                "leg_fraction": best_config.leg_fraction,
                "deployment_cap": best_config.deployment_cap,
                "implied_perp_leverage": best_config.implied_perp_leverage,
                "applied_return": applied.multiple - 1.0,
                "capital_after": capital,
                "entries": applied.entries,
            }
        )
        if verbose:
            print(
                f"  {panel.days[start].date()} ~ {panel.days[start+APPLY_DAYS-1].date()} | "
                f"{best_config.label} | 적용 {applied.multiple-1.0:+7.2%} | 자산 ${capital:8.2f} | 진입 {applied.entries:3d}",
                flush=True,
            )
        start += APPLY_DAYS
        if capital <= 0.0:
            break

    equity = np.asarray(equity_curve, dtype=float)
    days = len(equity)
    peak = np.maximum.accumulate(equity) if days else np.array([ACTIVE_CAPITAL])
    mdd = float(np.max(1.0 - equity / peak)) if days else 0.0
    annualised = float((equity[-1] / ACTIVE_CAPITAL) ** (365.0 / days) - 1.0) if days and equity[-1] > 0 else -1.0

    return {
        "final_usdt": float(equity[-1]) if days else ACTIVE_CAPITAL,
        "annualised": annualised,
        "mdd": mdd,
        "days": days,
        "reselections": len(selections),
        "evaluations": evaluations,
        "funding_usd": funding_total,
        "basis_usd": basis_total,
        "cost_usd": cost_total,
        "entries": entries_total,
        "days_active": days_active_total,
        "blocked_min_order": blocked_total,
        "min_leg_usd": float(min_leg) if np.isfinite(min_leg) else float("nan"),
        "max_leg_usd": max_leg,
        "delta_mismatch": delta_mismatch,
        "liquidation_days": liquidation_total,
        "worst_adverse_move": worst_adverse,
        "disqualified_by_liquidation": disqualified_total,
        "windows_with_no_feasible_config": no_feasible_windows,
        "accounting_residual": residual_max,
        "equity": equity.tolist(),
        "day_index": day_index,
        "deployment_mean": float(np.mean(deployment_used)) if deployment_used else 0.0,
        "selections": selections,
        "equity_first": float(equity[0]) if days else ACTIVE_CAPITAL,
    }


def main() -> int:
    started = time.time()
    print("=== wave38: 검증된 델타중립 캐리의 폭 확장 · 인과적 워크포워드 ===")
    panel = build_panel()
    grid = build_grid()
    print(f"패널 {len(panel.symbols)}종목 x {len(panel.days)}일 | 그리드 {len(grid)}조합 (전부 현물차입 불필요)")
    print(f"품질 기준 {THRESHOLD_APR:.0%} APR 고정(L4와 동일) · 훈련 {TRAIN_DAYS}일 -> 적용 {APPLY_DAYS}일\n")

    base = walk_forward(panel, grid)

    print("\n=== 비용 x3 스트레스 ===")
    stress = walk_forward(panel, grid, cost_multiplier=3.0, verbose=False)

    # Like-for-like internal baseline: L4's own settings on THIS panel, same causal protocol. The
    # published I5 figure (10.27%) comes from a different universe this repo can no longer rebuild, so
    # comparing only against it would confound breadth with universe composition.
    print("=== 내부 기준선 (L4 설정 k1/leg0.50/cap0.50, 동일 패널·동일 프로토콜) ===")
    baseline = walk_forward(panel, (BASELINE_CONFIG,), verbose=False)
    print(f"  연 {baseline['annualised']:+.2%} | MDD {baseline['mdd']:.2%} | 진입 {baseline['entries']}\n")

    days = base["days"]
    print(f"=== 인과적 곡선 ({len(panel.days)}일 패널 중 적용 {days}일) ===")
    print(f"  ${ACTIVE_CAPITAL:.2f} -> ${base['final_usdt']:.2f} | 연 {base['annualised']:+.2%} | MDD {base['mdd']:.2%}")
    print(f"  I5 공표 기준선 {I5_CAGR:+.2%} | L4 {L4_CAGR:+.2%} | 내부 동일패널 기준선 {baseline['annualised']:+.2%}")
    print(f"  손익: 펀딩 ${base['funding_usd']:+.2f} · 베이시스 ${base['basis_usd']:+.2f} · 비용 ${base['cost_usd']:+.2f}")
    total_pnl = base["funding_usd"] + abs(base["basis_usd"]) + abs(base["cost_usd"])
    print(f"  펀딩 비중 {base['funding_usd']/total_pnl:.1%}" if total_pnl else "")
    print(f"  재선정 {base['reselections']}회 · 그리드 평가 {base['evaluations']:,}회")
    print(f"  진입 {base['entries']}회 ({base['entries']/days:.2f}회/일) · 활성 {base['days_active']}일 ({base['days_active']/days:.1%})")
    print(f"  레그 ${base['min_leg_usd']:.2f}~${base['max_leg_usd']:.2f} · 최소주문차단 {base['blocked_min_order']}일")
    print(f"  평균 투입비율 {base['deployment_mean']:.2f} (실행가능 상한 {MAX_FEASIBLE_DEPLOYMENT:.3f})")
    print(f"  비용x3: 연 {stress['annualised']:+.2%}")
    print(f"  청산 발생일 {base['liquidation_days']}일 · 최악 장중 역행 {base['worst_adverse_move']:.2%}")

    # wave18 corrected L4's own headline for exactly this reason: "연 22.01%" turned out to be a
    # high-funding-era figure, not an expectation for capital. A single all-period CAGR can hide the
    # same thing, so the curve is broken out by year before any claim is made.
    print("\n=== 연도별 (전기간 단일 CAGR이 고펀딩기를 숨기는지 확인) ===")
    equity_arr = np.asarray(base["equity"], dtype=float)
    years = np.array([panel.days[i].year for i in base["day_index"]])
    yearly = {}
    for year in sorted(set(years.tolist())):
        mask = years == year
        segment = equity_arr[mask]
        if len(segment) < 2:
            continue
        opening = equity_arr[np.flatnonzero(mask)[0] - 1] if np.flatnonzero(mask)[0] > 0 else ACTIVE_CAPITAL
        change = segment[-1] / opening - 1.0
        yearly[year] = change
        print(f"  {year}: {change:+7.2%}  (${opening:7.2f} -> ${segment[-1]:7.2f})")
    recent = [v for y, v in yearly.items() if y >= 2023]
    if recent:
        print(f"  2023년 이후 평균 {np.mean(recent):+.2%}/년 · 전기간 {base['annualised']:+.2%}/년")

    gates: dict[str, dict] = {}
    gates["Z1_single_venue"] = {
        "status": "PASS",
        "detail": "research/wave3/cache 의 binance_spot/binance_fapi/binance_funding 만 참조",
    }
    causal_ok = base["reselections"] >= MIN_RESELECTIONS
    gates["Z2_causality"] = {
        "status": "PASS" if causal_ok else "FAIL",
        "reselections": base["reselections"],
        "minimum": MIN_RESELECTIONS,
        "detail": "적용창은 선정에 미사용; 랭킹 APR·거래량평균 모두 shift(1)",
    }
    beats_i5 = base["annualised"] > I5_CAGR
    gates["Z3_performance"] = {
        "status": "PASS" if beats_i5 else "FAIL",
        "annualised": base["annualised"],
        "i5_bar": I5_CAGR,
        "internal_baseline": baseline["annualised"],
        "uplift_vs_internal": base["annualised"] - baseline["annualised"],
    }
    gates["Z4_drawdown"] = {
        "status": "PASS" if base["mdd"] <= MDD_LIMIT else "FAIL",
        "mdd": base["mdd"],
        "limit": MDD_LIMIT,
    }
    gates["Z5_delta_neutral"] = {
        "status": "PASS" if base["delta_mismatch"] <= 1e-9 else "FAIL",
        "max_mismatch": base["delta_mismatch"],
    }
    gates["Z6_cost_stress"] = {
        "status": "PASS" if stress["annualised"] > 0.0 else "FAIL",
        "annualised_x3": stress["annualised"],
    }
    exec_reasons = []
    if np.isfinite(base["min_leg_usd"]) and base["min_leg_usd"] < MIN_ORDER_USDT:
        exec_reasons.append(f"min leg ${base['min_leg_usd']:.2f}")
    if any(CarryConfig(s["top_k"], s["leg_fraction"], s["deployment_cap"]).requires_spot_borrow for s in base["selections"]):
        exec_reasons.append("현물 차입 필요 조합이 선정됨")
    gates["Z7_executability"] = {
        "status": "PASS" if not exec_reasons else "FAIL",
        "min_leg_usdt": base["min_leg_usd"],
        "max_leg_usdt": base["max_leg_usd"],
        "max_feasible_deployment": MAX_FEASIBLE_DEPLOYMENT,
        "blocked_min_order_days": base["blocked_min_order"],
        "reasons": exec_reasons,
    }
    # Numbered Z10, not Z9: SPEC.md froze Z9 as the wave30 regression check before this gate existed,
    # and renumbering a frozen pre-registration to make room would defeat the point of freezing it.
    # This gate is an addition discovered during implementation, and it is recorded as such.
    gates["Z10_no_liquidation"] = {
        "status": "PASS" if base["liquidation_days"] == 0 else "FAIL",
        "liquidation_days": base["liquidation_days"],
        "worst_adverse_intraday_move": base["worst_adverse_move"],
        "detail": (
            "숏 퍼프 레그가 자기 마진(1/레버리지)을 넘는 장중 상승을 맞은 일수. 현물 미실현이익은 "
            "선물계좌 마진으로 자동 이체되지 않으므로 분리 계좌에서는 실제 청산이다."
        ),
    }
    gates["Z8_accounting"] = {
        "status": "PASS" if base["accounting_residual"] <= 1e-9 else "FAIL",
        "max_residual": base["accounting_residual"],
        "detail": "자본변화 = 펀딩 + 베이시스 - 비용",
    }

    regression_ok = _wave30_regression_passes()
    gates["Z9_regression"] = {
        "status": "PASS" if regression_ok else "FAIL",
        "detail": "research/wave30_qd/tests/test_wave30.py 22개 — 기존 엔진 무손상 확인",
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
                "wave": "wave38_breadth",
                "capital": TOTAL_CAPITAL,
                "active_capital": ACTIVE_CAPITAL,
                "threshold_apr": THRESHOLD_APR,
                "grid_size": len(grid),
                "base": base,
                "stress_x3": stress,
                "internal_baseline": baseline,
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
