#!/usr/bin/env python3
# Wave-51: attack the selection failure with prescriptions from other fields.
#
# Fifty waves changed the SEARCH METHOD -- GA, GP, CMA-ES, RL, SA, PSO, Bayesian, MCTS, MAP-Elites -- and
# every one failed at the same step. wave34 found method choice does not rescue OOS, wave37 found post-hoc
# and causal selection disagree in sign, wave38 found grid selection lost to a fixed config, wave43 found
# the protocol choosing a rule 23 times out of 23 and still underperforming, and wave50 found a 5x search
# budget WIDENING the seed spread. The bottleneck is not what gets searched. It is the act of taking the
# training window's argmax.
#
# And argmax is the one thing no wave questioned. So this wave holds everything else fixed and varies only
# the selection rule, borrowing two prescriptions that were derived independently elsewhere for exactly
# this situation:
#
#   Ecology -- diversified bet-hedging. Under an unpredictable environment organisms do NOT optimise for
#   the current one; they spread across environments. The specialist dies when conditions shift. Statistics
#   reaches the same place via James-Stein shrinkage (shrink noisy estimates toward the grand mean and
#   total error falls) and finance via DeMiguel-Garlappi-Uppal 2009 (1/N beats optimised portfolios out of
#   sample). Three fields converging on one answer is the reason to try it. Rules B and C.
#
#   Robust control -- H-infinity optimises WORST-CASE rather than average behaviour, because under model
#   uncertainty the average-optimal controller can diverge while the minimax one stays bounded. Translated:
#   split the training window and select on its worst sub-period. This bites directly on wave49's finding
#   that winners won because of a few explosive windows -- removing three of twenty-three took 105.39x down
#   to 1.05x. Minimax structurally refuses such candidates. Rule D.
#
# Ensembles split CAPITAL through the engine rather than averaging returns, so the $5 minimum order is
# enforced for real: a fifth of $90 is $18, and at k=8 that is a $2.25 leg the engine must refuse.
# Averaging returns would silently erase that constraint, which is the defect wave37 hit when it filled
# $0.40 legs.

from __future__ import annotations

import argparse
from collections import Counter
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
from research.wave42_quality.run_wave42 import (
    APPLY_DAYS,
    GRID,
    PORTFOLIO_MARGIN,
    THRESHOLDS,
    TRAIN_DAYS,
    Candidate,
    candidates,
    selection_score,
)

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

I5_CORRECTED: Final = 0.0828  # wave45/46
ARGMAX_RECENT: Final = 0.0241  # wave42's own 2023+ mean, the bar to beat
MDD_LIMIT: Final = 0.25
MINIMAX_SUBWINDOWS: Final = 4  # quarters of the training window


def score_all(panels: dict[float, object], start: int, cost_multiplier: float) -> list[tuple[Candidate, float, float]]:
    """(candidate, mean-score, minimax-score) for every grid point on the trailing training window.

    Both scores come from the SAME simulations, so rules A-D see identical information and differ only in
    how they rank it. Candidates liquidating anywhere in training are dropped for every rule alike.
    """
    scored: list[tuple[Candidate, float, float]] = []
    step = TRAIN_DAYS // MINIMAX_SUBWINDOWS
    for candidate in candidates():
        panel = panels[candidate.threshold]
        trained = simulate(panel, candidate.config, start - TRAIN_DAYS, start,
                           cost_multiplier, portfolio_margin=PORTFOLIO_MARGIN)
        if trained.liquidation_days > 0:
            continue
        mean_score = selection_score(trained, TRAIN_DAYS)

        # Worst sub-period return. Each sub-window is simulated on its own so its return is not an
        # artefact of where the equity curve happened to be sitting.
        worst = np.inf
        for index in range(MINIMAX_SUBWINDOWS):
            sub_start = start - TRAIN_DAYS + index * step
            sub_end = sub_start + step
            sub = simulate(panel, candidate.config, sub_start, sub_end,
                           cost_multiplier, portfolio_margin=PORTFOLIO_MARGIN)
            worst = min(worst, sub.multiple - 1.0)
        scored.append((candidate, mean_score, float(worst)))
    return scored


def apply_sleeves(panels, chosen: list[Candidate], start: int, capital: float, cost_multiplier: float):
    """Apply `chosen` as equal-capital sleeves and return the combined outcome.

    Each sleeve is run through the engine with capital/len(chosen), so the minimum-order rule applies to
    the sleeve's OWN size rather than to the undivided account. That is the point of splitting capital
    instead of averaging returns.
    """
    if not chosen:
        return None
    per_sleeve = capital / len(chosen)
    curves = []
    funding = basis = cost = 0.0
    entries = liquidations = blocked = 0
    min_leg, max_leg = np.inf, 0.0
    residual = 0.0
    for candidate in chosen:
        result = simulate(panels[candidate.threshold], candidate.config, start, start + APPLY_DAYS,
                          cost_multiplier, start_capital=per_sleeve, portfolio_margin=PORTFOLIO_MARGIN)
        curves.append(result.equity)
        funding += result.funding_usd
        basis += result.basis_usd
        cost += result.cost_usd
        entries += result.entries
        liquidations += result.liquidation_days
        blocked += result.blocked_min_order
        if np.isfinite(result.min_leg_usd):
            min_leg = min(min_leg, result.min_leg_usd)
        max_leg = max(max_leg, result.max_leg_usd)
        residual = max(residual, result.accounting_residual)
    length = min(len(c) for c in curves)
    combined = np.sum([c[:length] for c in curves], axis=0)
    return {
        "equity": combined,
        "final": float(combined[-1]),
        "funding": funding, "basis": basis, "cost": cost,
        "entries": entries, "liquidations": liquidations, "blocked": blocked,
        "min_leg": float(min_leg) if np.isfinite(min_leg) else float("nan"),
        "max_leg": max_leg, "residual": residual,
    }


def walk_forward(panels, rule: str, cost_multiplier: float = 1.0, verbose: bool = False) -> dict:
    any_panel = panels[THRESHOLDS[0]]
    n_days = len(any_panel.days)
    capital = ACTIVE_CAPITAL
    equity_curve: list[float] = []
    day_index: list[int] = []
    window_returns: list[float] = []
    labels: list[str] = []
    funding = basis = cost = 0.0
    entries = liquidations = blocked = 0
    min_leg, max_leg = np.inf, 0.0
    residual = 0.0

    start = TRAIN_DAYS + 1
    while start + APPLY_DAYS <= n_days:
        scored = score_all(panels, start, cost_multiplier)
        if not scored:
            equity_curve.extend([capital] * APPLY_DAYS)
            day_index.extend(range(start, start + APPLY_DAYS))
            start += APPLY_DAYS
            continue

        if rule == "argmax":
            chosen = [max(scored, key=lambda row: row[1])[0]]
        elif rule == "ensemble3":
            chosen = [row[0] for row in sorted(scored, key=lambda r: -r[1])[:3]]
        elif rule == "ensemble5":
            chosen = [row[0] for row in sorted(scored, key=lambda r: -r[1])[:5]]
        elif rule == "minimax":
            chosen = [max(scored, key=lambda row: row[2])[0]]
        else:
            raise ValueError(f"unknown rule {rule}")

        outcome = apply_sleeves(panels, chosen, start, capital, cost_multiplier)
        if outcome is None:
            start += APPLY_DAYS
            continue
        equity_curve.extend(outcome["equity"].tolist())
        day_index.extend(range(start, start + len(outcome["equity"])))
        factor = outcome["final"] / capital if capital > 0 else 0.0
        window_returns.append(factor - 1.0)
        labels.append(" + ".join(c.label for c in chosen))
        capital = outcome["final"]
        funding += outcome["funding"]; basis += outcome["basis"]; cost += outcome["cost"]
        entries += outcome["entries"]; liquidations += outcome["liquidations"]; blocked += outcome["blocked"]
        if np.isfinite(outcome["min_leg"]):
            min_leg = min(min_leg, outcome["min_leg"])
        max_leg = max(max_leg, outcome["max_leg"])
        residual = max(residual, outcome["residual"])
        if verbose:
            print(f"    {any_panel.days[start].date()} {factor-1.0:+7.2%} -> ${capital:8.2f} | {labels[-1][:60]}",
                  flush=True)
        start += APPLY_DAYS

    equity = np.asarray(equity_curve, dtype=float)
    days = len(equity)
    peak = np.maximum.accumulate(equity) if days else np.array([ACTIVE_CAPITAL])
    returns = np.asarray(window_returns, dtype=float)
    order = np.argsort(-returns)
    keep = np.ones(len(returns), dtype=bool)
    keep[order[:3]] = False
    return {
        "rule": rule,
        "final_usdt": float(equity[-1]) if days else ACTIVE_CAPITAL,
        "annualised": float((equity[-1] / ACTIVE_CAPITAL) ** (365.0 / days) - 1.0) if days and equity[-1] > 0 else -1.0,
        "mdd": float(np.max(1.0 - equity / peak)) if days else 0.0,
        "days": days,
        "reselections": len(returns),
        "funding_usd": funding, "basis_usd": basis, "cost_usd": cost,
        "entries": entries, "liquidation_days": liquidations, "blocked_min_order": blocked,
        "min_leg_usd": float(min_leg) if np.isfinite(min_leg) else float("nan"),
        "max_leg_usd": max_leg, "accounting_residual": residual,
        "window_median": float(np.median(returns)) if len(returns) else float("nan"),
        "window_positive_share": float((returns > 0).mean()) if len(returns) else float("nan"),
        "full_multiple": float(np.prod(1.0 + returns)) if len(returns) else float("nan"),
        "trimmed_multiple_3": float(np.prod(1.0 + returns[keep])) if len(returns) > 3 else float("nan"),
        "labels": labels,
        "equity": equity.tolist(),
        "day_index": day_index,
        "window_returns": returns.tolist(),
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
    parser = argparse.ArgumentParser(description="wave51: selection rules from other disciplines")
    parser.add_argument("--only", choices=("argmax", "ensemble3", "ensemble5", "minimax"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "final.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rules": {}}

    base_panel = build_panel()
    panels = {t: with_threshold(base_panel, t) for t in THRESHOLDS}
    print(f"=== wave51: 선택 규칙만 바꾼다 (나머지 wave42와 동일) ===")
    print(f"패널 {len(base_panel.symbols)}종목 x {len(base_panel.days)}일 · 그리드 {len(GRID)}구성 x 임계 {len(THRESHOLDS)}")
    print(f"대조: argmax 전기간 +13.31% · 2023+ {ARGMAX_RECENT:+.2%}\n")

    rules = [args.only] if args.only else ["argmax", "ensemble3", "ensemble5", "minimax"]
    for rule in rules:
        if rule in payload["rules"] and not args.only:
            print(f"  {rule} (캐시)")
            continue
        run = walk_forward(panels, rule)
        run["yearly"] = yearly(run, base_panel.days)
        recent = [v for k, v in run["yearly"].items() if int(k) >= 2023]
        run["recent_mean"] = float(np.mean(recent)) if recent else float("nan")
        payload["rules"][rule] = run
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"  {rule:10s} 전기간 {run['annualised']:+7.2%} | 2023+ {run['recent_mean']:+6.2%} | "
              f"MDD {run['mdd']:5.2%} | 창중앙 {run['window_median']:+6.2%} | "
              f"상위3제거 {run['trimmed_multiple_3']:5.2f}x | 최소레그 ${run['min_leg_usd']:6.2f} | "
              f"청산 {run['liquidation_days']}", flush=True)

    if args.only:
        print(f"\n{time.time()-started:.0f}s · 나머지는 --only 없이")
        return 0

    print("\n=== 규칙별 상세 ===")
    print(f"{'규칙':>10} {'전기간':>8} {'2023+':>7} {'MDD':>6} {'창중앙':>7} {'창양수':>7} "
          f"{'전체배수':>8} {'상위3제거':>9} {'최소레그':>8} {'차단일':>6}")
    for rule in ("argmax", "ensemble3", "ensemble5", "minimax"):
        r = payload["rules"].get(rule)
        if not r:
            continue
        print(f"{rule:>10} {r['annualised']:+7.2%} {r['recent_mean']:+6.2%} {r['mdd']:5.2%} "
              f"{r['window_median']:+6.2%} {r['window_positive_share']:6.1%} {r['full_multiple']:7.2f}x "
              f"{r['trimmed_multiple_3']:8.2f}x ${r['min_leg_usd']:7.2f} {r['blocked_min_order']:5d}")

    argmax = payload["rules"]["argmax"]
    alternatives = {k: v for k, v in payload["rules"].items() if k != "argmax"}
    best_name, best = max(alternatives.items(), key=lambda kv: kv[1]["recent_mean"])
    beats = best["recent_mean"] > argmax["recent_mean"]

    print("\n=== 연도별 (최근 레짐 비교) ===")
    years = sorted(set().union(*[set(r["yearly"]) for r in payload["rules"].values()]))
    header = "  연도  " + "".join(f"{r:>11}" for r in ("argmax", "ensemble3", "ensemble5", "minimax"))
    print(header)
    for year in years:
        row = f"  {year}  "
        for rule in ("argmax", "ensemble3", "ensemble5", "minimax"):
            value = payload["rules"].get(rule, {}).get("yearly", {}).get(year)
            row += f"{value:+10.2%} " if value is not None else f"{'-':>10} "
        print(row)

    gates = {
        "T1_causality": {"status": "PASS" if argmax["reselections"] >= 20 else "FAIL",
                          "reselections": argmax["reselections"]},
        "T2_beats_argmax_recent": {"status": "PASS" if beats else "FAIL",
                                    "best_rule": best_name, "best_recent": best["recent_mean"],
                                    "argmax_recent": argmax["recent_mean"],
                                    "detail": "대안 선택 규칙이 최근 레짐에서 argmax를 넘는가"},
        "T3_beats_i5_full": {"status": "PASS" if best["annualised"] > I5_CORRECTED else "FAIL",
                              "annualised": best["annualised"], "bar": I5_CORRECTED},
        "T4_drawdown": {"status": "PASS" if best["mdd"] <= MDD_LIMIT else "FAIL", "mdd": best["mdd"]},
        "T5_no_liquidation": {"status": "PASS" if best["liquidation_days"] == 0 else "FAIL",
                               "liquidation_days": best["liquidation_days"]},
        "T6_executability": {"status": "PASS" if best["min_leg_usd"] >= 5.0 else "FAIL",
                              "min_leg_usd": best["min_leg_usd"],
                              "blocked_min_order": best["blocked_min_order"],
                              "detail": "앙상블 자본분할 후에도 레그가 최소주문을 넘는가"},
        "T7_not_tail_dependent": {"status": "PASS" if best["trimmed_multiple_3"] > 1.0 else "FAIL",
                                   "trimmed_multiple_3": best["trimmed_multiple_3"]},
    }
    failures = [k for k, v in gates.items() if v["status"] == "FAIL"]
    print()
    for name, gate in gates.items():
        print(f"[{gate['status']}] {name}")
    print(f"\nOVERALL {'PASS' if not failures else 'FAIL'} | failures {failures}")

    print("\n=== 판정 ===")
    if beats:
        print(f"  {best_name} 가 argmax를 최근 레짐에서 넘는다: "
              f"{best['recent_mean']:+.2%} vs {argmax['recent_mean']:+.2%}")
        print("  => 선택 규칙 축이 실재한다. 다른 학문의 처방이 작동했다.")
    else:
        print(f"  어떤 대안도 argmax를 넘지 못한다 (최선 {best_name} {best['recent_mean']:+.2%} "
              f"vs argmax {argmax['recent_mean']:+.2%})")
        print("  => argmax가 나쁜 게 아니라 훈련창에 정보가 없다. 선택 규칙 축이 닫힌다.")

    payload["gates"] = gates
    payload["failures"] = failures
    payload["overall"] = "PASS" if not failures else "FAIL"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n{time.time()-started:.0f}s · results/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
