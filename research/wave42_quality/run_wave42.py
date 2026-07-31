#!/usr/bin/env python3
# Wave-42: causal walk-forward over quality threshold x portfolio-margin deployment.
#
# The two levers this wave combines were each measured alone and each looked like a dead end:
#   - wave38: raising breadth lowers return while lowering drawdown. A risk lever, not a return lever.
#   - wave39: portfolio margin makes deployment 1.00 liquidation-safe and lifts the full period to
#     +13.35%, but the recent regime falls to +2.63%/yr -- WORSE than plain L4's +3.62% -- because extra
#     deployment amplifies a poor funding regime.
#
# wave38's corrected cost model makes a third lever computable for the first time. Round-trip cost is
# 0.384% of notional and L4's mean holding period is 4.05 days, so breakeven sits near 35% APR while the
# entry bar is 15%. Trades near the bar cannot repay their own turnover. Nobody in 40 waves raised that
# bar: I3 lowered it to 8% and lost, which is evidence the gradient points up, not down.
#
# Raising the bar turns out to be another risk lever (drawdown 4.49% -> 1.11%, return flat). The point of
# this wave is that a risk lever and a return lever compose: cutting the trades that amplify a bad regime
# is exactly what deployment expansion needed. Neither single-lever experiment could show that.
#
# Selection is causal, not chosen by eye. A pre-run sweep showed threshold 0.35 / deployment 1.00 doing
# well on both the full period and post-2023, but reading a grid after the fact is the error this campaign
# has documented repeatedly (wave36 -> wave37 flipped sign that way). So the grid is handed to the same
# trailing-365-day protocol used since wave37 and whatever it picks is the answer.

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from research.wave38_breadth.dataio38 import build_panel, with_threshold
from research.wave38_breadth.engine38 import ACTIVE_CAPITAL, CarryConfig, simulate

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
TRAIN_DAYS: Final = 365
APPLY_DAYS: Final = 90
I5_CAGR: Final = 0.1027  # W3 bar
L4_RECENT: Final = 0.0362  # W4 bar: plain L4's own 2023+ mean, measured in wave38
MDD_LIMIT: Final = 0.25
MIN_RESELECTIONS: Final = 20
PORTFOLIO_MARGIN: Final = (0.70, 0.01)  # wave39's worst measured haircut / mmr

THRESHOLDS: Final = (0.15, 0.25, 0.35)
GRID: Final = tuple(
    CarryConfig(top_k, leg_fraction, cap)
    for top_k in (1, 2, 3)
    for leg_fraction in (0.50, 1.00)
    for cap in (0.50, 0.75, 1.00)
)
BASELINE: Final = (0.15, CarryConfig(1, 0.50, 0.50))  # current L4, for W10 convergence test


@dataclass(frozen=True, slots=True)
class Candidate:
    threshold: float
    config: CarryConfig

    @property
    def label(self) -> str:
        return f"thr{self.threshold:.0%} {self.config.label}"


def candidates() -> tuple[Candidate, ...]:
    return tuple(Candidate(threshold, config) for threshold in THRESHOLDS for config in GRID)


def selection_score(result, days: int) -> float:
    return result.annualised(days) - 2.0 * max(0.0, result.mdd - 0.20)


def walk_forward(panels: dict[float, object], cost_multiplier: float = 1.0, verbose: bool = True) -> dict:
    all_candidates = candidates()
    any_panel = panels[THRESHOLDS[0]]
    n_days = len(any_panel.days)

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
        best, best_score = None, -np.inf
        for candidate in all_candidates:
            trained = simulate(
                panels[candidate.threshold],
                candidate.config,
                start - TRAIN_DAYS,
                start,
                cost_multiplier,
                portfolio_margin=PORTFOLIO_MARGIN,
            )
            evaluations += 1
            if trained.liquidation_days > 0:
                continue
            score = selection_score(trained, TRAIN_DAYS)
            if score > best_score:
                best, best_score = candidate, score

        if best is None:
            equity_curve.extend([capital] * APPLY_DAYS)
            day_index.extend(range(start, start + APPLY_DAYS))
            start += APPLY_DAYS
            continue

        applied = simulate(
            panels[best.threshold],
            best.config,
            start,
            start + APPLY_DAYS,
            cost_multiplier,
            start_capital=capital,
            portfolio_margin=PORTFOLIO_MARGIN,
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
                "apply_from": str(any_panel.days[start].date()),
                "apply_to": str(any_panel.days[start + APPLY_DAYS - 1].date()),
                "label": best.label,
                "threshold": best.threshold,
                "top_k": best.config.top_k,
                "leg_fraction": best.config.leg_fraction,
                "deployment_cap": best.config.deployment_cap,
                "applied_return": applied.multiple - 1.0,
                "capital_after": capital,
            }
        )
        if verbose:
            print(
                f"  {any_panel.days[start].date()} ~ {any_panel.days[start+APPLY_DAYS-1].date()} | "
                f"{best.label:26s} | 적용 {applied.multiple-1.0:+7.2%} | 자산 ${capital:8.2f}",
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


def yearly_breakdown(run: dict, days_index) -> dict[str, float]:
    equity = np.asarray(run["equity"], dtype=float)
    years = np.array([days_index[i].year for i in run["day_index"]])
    out: dict[str, float] = {}
    previous = ACTIVE_CAPITAL
    for year in sorted(set(years.tolist())):
        mask = years == year
        segment = equity[mask]
        if len(segment) < 2:
            continue
        out[str(year)] = float(segment[-1] / previous - 1.0)
        previous = segment[-1]
    return out


def _load() -> dict:
    path = RESULTS_DIR / "final.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save(payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wave42 staged runner")
    parser.add_argument("--stage", choices=("base", "stress", "judge"), required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.time()
    payload = _load()
    base_panel = build_panel()
    panels = {threshold: with_threshold(base_panel, threshold) for threshold in THRESHOLDS}
    print(f"패널 {len(base_panel.symbols)}종목 x {len(base_panel.days)}일 | 후보 {len(candidates())}조합")

    if args.stage == "base":
        print("=== 인과 워크포워드 (기본 비용) ===")
        payload["base"] = walk_forward(panels)
        payload["yearly"] = yearly_breakdown(payload["base"], base_panel.days)
        _save(payload)
        print(f"\n연 {payload['base']['annualised']:+.2%} | MDD {payload['base']['mdd']:.2%} | "
              f"청산 {payload['base']['liquidation_days']}일 | {time.time()-started:.0f}s")
        print("다음: --stage stress")
        return 0

    if args.stage == "stress":
        if "base" not in payload:
            print("먼저 --stage base 를 실행해야 한다.")
            return 1
        print("=== 비용 x3 스트레스 ===")
        payload["stress_x3"] = walk_forward(panels, cost_multiplier=3.0, verbose=False)
        _save(payload)
        print(f"연 {payload['stress_x3']['annualised']:+.2%} | {time.time()-started:.0f}s")
        print("다음: --stage judge")
        return 0

    # judge
    if "base" not in payload or "stress_x3" not in payload:
        print("base 와 stress 단계를 먼저 실행해야 한다.")
        return 1
    base = payload["base"]
    stress = payload["stress_x3"]
    yearly = payload["yearly"]
    recent = [v for k, v in yearly.items() if int(k) >= 2023]
    recent_mean = float(np.mean(recent)) if recent else float("nan")

    days = base["days"]
    print(f"\n=== 인과적 곡선 (적용 {days}일) ===")
    print(f"  ${ACTIVE_CAPITAL:.2f} -> ${base['final_usdt']:.2f} | 연 {base['annualised']:+.2%} | MDD {base['mdd']:.2%}")
    print(f"  손익: 펀딩 ${base['funding_usd']:+.2f} · 베이시스 ${base['basis_usd']:+.2f} · 비용 ${base['cost_usd']:+.2f}")
    print(f"  재선정 {base['reselections']}회 · 평가 {base['evaluations']:,}회 · 평균 투입 {base['deployment_mean']:.2f}")
    print(f"  진입 {base['entries']}회 ({base['entries']/days:.3f}/일) · 활성 {base['days_active']}일")
    print(f"  레그 ${base['min_leg_usd']:.2f}~${base['max_leg_usd']:.2f} · 청산 {base['liquidation_days']}일")
    print(f"  비용x3: 연 {stress['annualised']:+.2%}")

    print("\n=== 연도별 ===")
    for year, change in sorted(yearly.items()):
        print(f"  {year}: {change:+7.2%}")
    print(f"  2023년 이후 평균 {recent_mean:+.2%} (현행 L4 {L4_RECENT:+.2%})")

    print("\n=== 선정 분포 ===")
    from collections import Counter
    counts = Counter(s["label"] for s in base["selections"])
    for label, count in counts.most_common(8):
        print(f"  {count:2d}회  {label}")
    thresholds_used = Counter(f"{s['threshold']:.0%}" for s in base["selections"])
    caps_used = Counter(f"{s['deployment_cap']:.2f}" for s in base["selections"])
    print(f"  임계값 분포 {dict(thresholds_used)} · 투입 분포 {dict(caps_used)}")

    converged_to_l4 = all(
        s["threshold"] == BASELINE[0] and s["deployment_cap"] == BASELINE[1].deployment_cap
        for s in base["selections"]
    )
    gates = {
        "W1_single_venue": {"status": "PASS", "detail": "research/wave3/cache Binance 3종만"},
        "W2_causality": {"status": "PASS" if base["reselections"] >= MIN_RESELECTIONS else "FAIL",
                          "reselections": base["reselections"], "minimum": MIN_RESELECTIONS},
        "W3_full_period": {"status": "PASS" if base["annualised"] > I5_CAGR else "FAIL",
                            "annualised": base["annualised"], "bar": I5_CAGR},
        "W4_recent_regime": {"status": "PASS" if recent_mean > L4_RECENT else "FAIL",
                              "mean_since_2023": recent_mean, "bar_l4": L4_RECENT},
        "W5_drawdown": {"status": "PASS" if base["mdd"] <= MDD_LIMIT else "FAIL", "mdd": base["mdd"]},
        "W6_no_liquidation": {"status": "PASS" if base["liquidation_days"] == 0 else "FAIL",
                               "liquidation_days": base["liquidation_days"]},
        "W7_cost_stress": {"status": "PASS" if stress["annualised"] > 0.0 else "FAIL",
                            "annualised_x3": stress["annualised"]},
        "W8_executability": {"status": "PASS" if (base["min_leg_usd"] >= 5.0 and all(s["deployment_cap"] <= 1.0 for s in base["selections"])) else "FAIL",
                              "min_leg_usd": base["min_leg_usd"]},
        "W9_accounting": {"status": "PASS" if base["accounting_residual"] <= 1e-9 else "FAIL",
                           "max_residual": base["accounting_residual"]},
        "W10_combination_matters": {"status": "FAIL" if converged_to_l4 else "PASS",
                                     "detail": "선정이 현행 L4(임계15%·투입0.50)로 수렴하지 않았는가"},
    }
    failures = [name for name, gate in gates.items() if gate["status"] == "FAIL"]
    print()
    for name, gate in gates.items():
        print(f"[{gate['status']}] {name}")
    print(f"\nOVERALL {'PASS' if not failures else 'FAIL'} | failures {failures}")

    payload["recent_mean"] = recent_mean
    payload["gates"] = gates
    payload["failures"] = failures
    payload["overall"] = "PASS" if not failures else "FAIL"
    _save(payload)
    print("results/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
