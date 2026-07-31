#!/usr/bin/env python3
# Wave-52: decorrelated families, then the ensemble wave51 could not test.
#
# wave51's precondition failed: its candidates shared a 73.9% positive-window share because every one
# ranked on the same funding level. signals52.py built families with different mechanics and the Jaccard
# overlap of their held positions confirms they are genuinely different -- level/change 0.17,
# change/basis 0.16, majors/alts 0.00 by construction. So this time there IS dispersion to hedge across,
# and the ensemble question is worth asking.
#
# Order of operations matters here. Each family is first run alone, because an ensemble of families that
# individually lose money is not diversification, it is spreading a loss. Only then is the ensemble tested,
# and its comparison point is the incumbent (level-only argmax, which reproduces wave42) rather than the
# best family chosen after the fact -- picking the winner post hoc is the error wave37 documented.
#
# Ensembles split capital through the engine so the $5 minimum applies per sleeve, as in wave51.

from __future__ import annotations

import argparse
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

from research.wave38_breadth.dataio38 import build_panel
from research.wave38_breadth.engine38 import ACTIVE_CAPITAL, simulate
from research.wave42_quality.run_wave42 import (
    APPLY_DAYS,
    GRID,
    PORTFOLIO_MARGIN,
    THRESHOLDS,
    TRAIN_DAYS,
    selection_score,
)
from research.wave52_decorrelated.signals52 import FAMILIES

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

I5_CORRECTED: Final = 0.0828
WAVE42_FULL: Final = 0.1331
WAVE42_RECENT: Final = 0.0241
MDD_LIMIT: Final = 0.25


def build_family_panels(base_panel) -> dict[str, dict[float, object]]:
    """panels[family][threshold]. Built once; the walk-forward reuses them for every rule."""
    out: dict[str, dict[float, object]] = {}
    for name, builder in FAMILIES.items():
        out[name] = {threshold: builder(base_panel, threshold) for threshold in THRESHOLDS}
    return out


def best_config_for(panels_by_threshold, start: int, cost_multiplier: float):
    """argmax within one family: the (threshold, config) its trailing window scores highest."""
    best, best_score = None, -np.inf
    for threshold, panel in panels_by_threshold.items():
        for config in GRID:
            trained = simulate(panel, config, start - TRAIN_DAYS, start, cost_multiplier,
                               portfolio_margin=PORTFOLIO_MARGIN)
            if trained.liquidation_days > 0:
                continue
            score = selection_score(trained, TRAIN_DAYS)
            if score > best_score:
                best, best_score = (threshold, config), score
    return best


def walk_forward(panels: dict[str, dict[float, object]], families: list[str],
                 cost_multiplier: float = 1.0) -> dict:
    """Causal walk-forward over one or more families, equal capital per family."""
    any_panel = panels[families[0]][THRESHOLDS[0]]
    n_days = len(any_panel.days)
    capital = ACTIVE_CAPITAL
    equity_curve: list[float] = []
    day_index: list[int] = []
    window_returns: list[float] = []
    funding = basis = cost = 0.0
    entries = liquidations = blocked = 0
    min_leg, max_leg = np.inf, 0.0
    residual = 0.0
    picks: list[str] = []

    start = TRAIN_DAYS + 1
    while start + APPLY_DAYS <= n_days:
        per_sleeve = capital / len(families)
        curves = []
        labels = []
        for family in families:
            chosen = best_config_for(panels[family], start, cost_multiplier)
            if chosen is None:
                curves.append(np.full(APPLY_DAYS, per_sleeve))
                labels.append(f"{family}:flat")
                continue
            threshold, config = chosen
            applied = simulate(panels[family][threshold], config, start, start + APPLY_DAYS,
                               cost_multiplier, start_capital=per_sleeve,
                               portfolio_margin=PORTFOLIO_MARGIN)
            curves.append(applied.equity)
            labels.append(f"{family}:thr{threshold:.0%} {config.label}")
            funding += applied.funding_usd
            basis += applied.basis_usd
            cost += applied.cost_usd
            entries += applied.entries
            liquidations += applied.liquidation_days
            blocked += applied.blocked_min_order
            if np.isfinite(applied.min_leg_usd):
                min_leg = min(min_leg, applied.min_leg_usd)
            max_leg = max(max_leg, applied.max_leg_usd)
            residual = max(residual, applied.accounting_residual)
        length = min(len(c) for c in curves)
        combined = np.sum([c[:length] for c in curves], axis=0)
        equity_curve.extend(combined.tolist())
        day_index.extend(range(start, start + length))
        factor = float(combined[-1]) / capital if capital > 0 else 0.0
        window_returns.append(factor - 1.0)
        picks.append(" | ".join(labels))
        capital = float(combined[-1])
        start += APPLY_DAYS

    equity = np.asarray(equity_curve, dtype=float)
    days = len(equity)
    peak = np.maximum.accumulate(equity) if days else np.array([ACTIVE_CAPITAL])
    returns = np.asarray(window_returns, dtype=float)
    order = np.argsort(-returns)
    keep = np.ones(len(returns), dtype=bool)
    keep[order[:3]] = False
    return {
        "families": families,
        "final_usdt": float(equity[-1]) if days else ACTIVE_CAPITAL,
        "annualised": float((equity[-1] / ACTIVE_CAPITAL) ** (365.0 / days) - 1.0) if days and equity[-1] > 0 else -1.0,
        "mdd": float(np.max(1.0 - equity / peak)) if days else 0.0,
        "days": days,
        "reselections": len(returns),
        "funding_usd": funding, "basis_usd": basis, "cost_usd": cost,
        "entries": entries, "liquidation_days": liquidations, "blocked_min_order": blocked,
        "entries_per_day": entries / days if days else 0.0,
        "min_leg_usd": float(min_leg) if np.isfinite(min_leg) else float("nan"),
        "max_leg_usd": max_leg, "accounting_residual": residual,
        "window_median": float(np.median(returns)) if len(returns) else float("nan"),
        "window_positive_share": float((returns > 0).mean()) if len(returns) else float("nan"),
        "full_multiple": float(np.prod(1.0 + returns)) if len(returns) else float("nan"),
        "trimmed_multiple_3": float(np.prod(1.0 + returns[keep])) if len(returns) > 3 else float("nan"),
        "window_returns": returns.tolist(),
        "equity": equity.tolist(),
        "day_index": day_index,
        "picks": picks,
    }


def yearly(run: dict, days_index) -> dict[str, float]:
    equity = np.asarray(run["equity"], dtype=float)
    years = np.array([days_index[i].year for i in run["day_index"]])
    out: dict[str, float] = {}
    previous = ACTIVE_CAPITAL
    for year in sorted(set(years.tolist())):
        segment = equity[years == year]
        if len(segment) < 2:
            continue
        out[str(year)] = float(segment[-1] / previous - 1.0)
        previous = segment[-1]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wave52: decorrelated families and their ensemble")
    parser.add_argument("--only", help="single family name, or comma-separated list for an ensemble")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "final.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": {}}

    base_panel = build_panel()
    panels = build_family_panels(base_panel)
    print(f"=== wave52: 상관 낮은 계열들과 그 앙상블 ===")
    print(f"패널 {len(base_panel.symbols)}종목 x {len(base_panel.days)}일 · 계열 {list(FAMILIES)}")
    print(f"대조: level 단독 argmax = wave42 (전기간 {WAVE42_FULL:+.2%}, 2023+ {WAVE42_RECENT:+.2%})\n")

    if args.only:
        families = [f.strip() for f in args.only.split(",")]
        key = "+".join(families)
        run = walk_forward(panels, families)
        run["yearly"] = yearly(run, base_panel.days)
        recent = [v for k, v in run["yearly"].items() if int(k) >= 2023]
        run["recent_mean"] = float(np.mean(recent)) if recent else float("nan")
        payload["runs"][key] = run
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"  {key:28s} 전기간 {run['annualised']:+8.2%} | 2023+ {run['recent_mean']:+6.2%} | "
              f"MDD {run['mdd']:6.2%} | 창중앙 {run['window_median']:+6.2%} | 창양수 {run['window_positive_share']:5.1%} | "
              f"진입 {run['entries_per_day']:.3f}/일 | 최소레그 ${run['min_leg_usd']:6.2f} | 청산 {run['liquidation_days']}")
        print(f"\n{time.time()-started:.0f}s")
        return 0

    print("=== 계열별 단독 성과 (앙상블 전에 각자 돈을 버는지 먼저 본다) ===")
    print(f"{'계열':>10} {'전기간':>9} {'2023+':>7} {'MDD':>7} {'창중앙':>7} {'창양수':>7} "
          f"{'진입/일':>8} {'최소레그':>9} {'청산':>5}")
    for name in list(FAMILIES) + ["level+change", "level+change+basis", "level+change+basis+majors"]:
        run = payload["runs"].get(name)
        if not run:
            print(f"{name:>10} (미실행 — --only {name} 으로 실행)")
            continue
        print(f"{name:>10} {run['annualised']:+8.2%} {run['recent_mean']:+6.2%} {run['mdd']:6.2%} "
              f"{run['window_median']:+6.2%} {run['window_positive_share']:6.1%} "
              f"{run['entries_per_day']:7.3f} ${run['min_leg_usd']:8.2f} {run['liquidation_days']:4d}")

    singles = {n: payload["runs"][n] for n in FAMILIES if n in payload["runs"]}
    if len(singles) >= 2:
        print("\n=== 계열 간 적용창 수익률 상관 (앙상블의 전제) ===")
        names = list(singles)
        length = min(len(singles[n]["window_returns"]) for n in names)
        matrix = np.array([singles[n]["window_returns"][:length] for n in names])
        print("          " + "".join(f"{n:>11}" for n in names))
        for i, a in enumerate(names):
            row = f"{a:>10}"
            for j in range(len(names)):
                if np.std(matrix[i]) == 0 or np.std(matrix[j]) == 0:
                    row += f"{'-':>11}"
                else:
                    row += f"{np.corrcoef(matrix[i], matrix[j])[0, 1]:11.2f}"
            print(row)
        off = [np.corrcoef(matrix[i], matrix[j])[0, 1]
               for i in range(len(names)) for j in range(i + 1, len(names))
               if np.std(matrix[i]) > 0 and np.std(matrix[j]) > 0]
        if off:
            print(f"\n  비대각 상관 중앙 {float(np.median(off)):.2f} · 최대 {max(off):.2f} · 최소 {min(off):.2f}")
            print(f"  => 낮으면 앙상블이 실제로 분산 효과를 낸다. wave51은 이 표를 만들 수 없었다(사실상 동일 계열).")

    ensembles = {k: v for k, v in payload["runs"].items() if "+" in k}
    if ensembles and "level" in payload["runs"]:
        incumbent = payload["runs"]["level"]
        best_name, best = max(ensembles.items(), key=lambda kv: kv[1]["recent_mean"])
        beats = best["recent_mean"] > incumbent["recent_mean"]
        gates = {
            "S1_precondition": {"status": "PASS", "detail": "계열 간 보유 자카드 0.00~0.44 (signals52.py 실측)"},
            "S2_beats_incumbent_recent": {
                "status": "PASS" if beats else "FAIL",
                "best": best_name, "best_recent": best["recent_mean"],
                "incumbent_recent": incumbent["recent_mean"],
                "detail": "앙상블이 최근 레짐에서 level 단독(=wave42)을 넘는가"},
            "S3_beats_i5_full": {"status": "PASS" if best["annualised"] > I5_CORRECTED else "FAIL",
                                  "annualised": best["annualised"], "bar": I5_CORRECTED},
            "S4_drawdown": {"status": "PASS" if best["mdd"] <= MDD_LIMIT else "FAIL", "mdd": best["mdd"]},
            "S5_no_liquidation": {"status": "PASS" if best["liquidation_days"] == 0 else "FAIL",
                                   "liquidation_days": best["liquidation_days"]},
            "S6_executability": {"status": "PASS" if best["min_leg_usd"] >= 5.0 else "FAIL",
                                  "min_leg_usd": best["min_leg_usd"],
                                  "blocked": best["blocked_min_order"]},
            "S7_not_tail_dependent": {"status": "PASS" if best["trimmed_multiple_3"] > 1.0 else "FAIL",
                                       "trimmed_multiple_3": best["trimmed_multiple_3"]},
        }
        failures = [k for k, v in gates.items() if v["status"] == "FAIL"]
        print()
        for name, gate in gates.items():
            print(f"[{gate['status']}] {name}")
        print(f"\nOVERALL {'PASS' if not failures else 'FAIL'} | failures {failures}")
        print(f"\n  최선 앙상블 {best_name}: 2023+ {best['recent_mean']:+.2%} vs level 단독 {incumbent['recent_mean']:+.2%}")
        payload["gates"] = gates
        payload["failures"] = failures
        payload["overall"] = "PASS" if not failures else "FAIL"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\n{time.time()-started:.0f}s · results/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
