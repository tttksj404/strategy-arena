#!/usr/bin/env python3
# Wave-53: a vectorised carry evaluator, so the search space can be millions instead of dozens.
#
# Every wave so far evaluated configurations one at a time -- engine38 takes about 36ms per config, which
# caps a wave at a few thousand evaluations and forced grids of 54 to 9,720 points. That is not a large
# search by the standards of fields that do this seriously, and it leaves a real objection unanswered: is
# the incumbent good, or merely the best of a small sample?
#
# The key structural fact that makes vectorisation possible: for a fixed threshold the carry book's daily
# return decomposes as
#
#     r_t = leg_eff * (sum of the top-k selected symbols' basis+funding) - turnover_cost_t
#
# where leg_eff = min(leg_fraction, cap / k). The expensive parts -- the hysteresis state machine and the
# per-day ranking -- depend ONLY on the threshold and k, never on leg_fraction or cap. So they are computed
# once per (threshold, k) and then millions of (leg, cap) pairs are pure arithmetic on precomputed daily
# series. A 200 x 20 x 32 x 32 grid is 4.1 million configurations reachable in seconds.
#
# This buys two things a small grid cannot. First, a genuinely exhaustive sweep of the parameter space.
# Second, and more important, an EMPIRICAL NULL: shuffle the funding signal across symbols within each day,
# destroying the cross-sectional information while preserving every marginal distribution, then run the
# same exhaustive search on the shuffled data. The best result found by chance, repeated many times, is the
# distribution the real result must beat. That is White's Reality Check, and it is the only way to answer
# "is this edge or is this search" with a number.
#
# Correctness is not assumed: verify_against_engine38 reproduces engine38's own equity curve for shared
# configurations before any of the above is trusted.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave38_breadth.dataio38 import CarryPanel, build_panel, with_threshold
from research.wave38_breadth.engine38 import MIN_ORDER_USDT

ACTIVE_CAPITAL: Final = 90.0


@dataclass(frozen=True, slots=True)
class DailySeries:
    """Per-day aggregates for one (threshold, k) pair, from which any (leg, cap) follows arithmetically."""

    threshold: float
    top_k: int
    # engine38 applies four steps per day IN ORDER, each compounding on the running capital: the overnight
    # gap on YESTERDAY's holdings, then turnover, then today's intraday basis, then today's funding. Summing
    # them and applying once gave a 2.6e-02 relative error, so each is kept as its own series and the order
    # is preserved in evaluate_grid.
    gap_prev: np.ndarray  # sum of overnight basis gap over YESTERDAY's holdings
    intraday_sum: np.ndarray  # sum of intraday basis over today's holdings
    funding_sum: np.ndarray  # sum of funding over today's holdings
    turnover: np.ndarray  # sum of cost_rate over symbols entering or leaving
    n_selected: np.ndarray  # how many symbols were held
    days: int


def build_daily_series(panel: CarryPanel, top_k: int) -> DailySeries:
    """Aggregate the panel into the two daily series a carry book's return depends on.

    Selection repeats engine38's rule exactly: among symbols whose hysteresis is on, which are tradable and
    which have a finite ranking score, take the `top_k` highest by ranking score. Returns are the same
    delta-neutral quantity engine38 accumulates -- intraday basis plus the day's funding -- and the
    overnight gap term is folded in the same way.
    """
    n_days, n_symbols = panel.spot_close.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        intraday = (panel.spot_close / panel.spot_open - 1.0) - (panel.perp_close / panel.perp_open - 1.0)
        gap = np.full_like(intraday, 0.0)
        gap[1:] = (panel.spot_open[1:] / panel.spot_close[:-1] - 1.0) - (
            panel.perp_open[1:] / panel.perp_close[:-1] - 1.0
        )
    intraday = np.nan_to_num(intraday, nan=0.0, posinf=0.0, neginf=0.0)
    gap = np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)
    funding = np.nan_to_num(panel.funding_daily, nan=0.0)
    cost = np.nan_to_num(panel.cost_rate, nan=0.0)
    per_symbol = intraday + gap + funding

    eligible = (panel.active > 0.0) & panel.tradable & np.isfinite(panel.ranking_apr)
    score = np.where(eligible, panel.ranking_apr, -np.inf)

    gap_prev = np.zeros(n_days)
    intraday_sum = np.zeros(n_days)
    funding_sum = np.zeros(n_days)
    turnover = np.zeros(n_days)
    counts = np.zeros(n_days, dtype=np.int64)
    previous = np.zeros(n_symbols, dtype=bool)
    # argpartition would be faster but ties break differently from argsort, and engine38 uses a stable
    # argsort -- matching it matters more here than a constant factor, since the whole point is that this
    # evaluator agrees with the validated one.
    for day in range(n_days):
        if previous.any():
            gap_prev[day] = float(gap[day][previous].sum())
        row_eligible = eligible[day]
        held = np.zeros(n_symbols, dtype=bool)
        if row_eligible.any():
            candidates = np.flatnonzero(row_eligible)
            order = np.argsort(-score[day][candidates], kind="stable")
            held[candidates[order[:top_k]]] = True
        counts[day] = int(held.sum())
        if counts[day]:
            intraday_sum[day] = float(intraday[day][held].sum())
            funding_sum[day] = float(funding[day][held].sum())
        changed = held ^ previous
        if changed.any():
            turnover[day] = float(cost[day][changed].sum())
        previous = held
    return DailySeries(0.0, top_k, gap_prev, intraday_sum, funding_sum, turnover, counts, n_days)


def evaluate_grid(series: DailySeries, leg_values: np.ndarray, cap_values: np.ndarray,
                  start: int, end: int) -> dict[str, np.ndarray]:
    """Compound every (leg, cap) pair over [start, end) at once.

    Returns matrices indexed [leg, cap]. The compounding loop runs over DAYS, not over configurations, so
    its cost is independent of how many configurations are being evaluated -- that is where the speedup
    comes from.
    """
    leg_grid, cap_grid = np.meshgrid(leg_values, cap_values, indexing="ij")
    k = max(series.top_k, 1)
    leg_eff = np.minimum(leg_grid, cap_grid / k)

    gap_prev = series.gap_prev[start:end]
    intraday = series.intraday_sum[start:end]
    funding = series.funding_sum[start:end]
    turnover = series.turnover[start:end]
    counts = series.n_selected[start:end]

    capital = np.full(leg_eff.shape, ACTIVE_CAPITAL)
    peak = capital.copy()
    max_dd = np.zeros_like(capital)
    min_leg = np.full(leg_eff.shape, np.inf)
    for index in range(len(intraday)):
        # Order matches engine38 exactly: gap on yesterday's book, then turnover, then intraday, then
        # funding, each compounding on the running capital.
        if gap_prev[index] != 0.0:
            capital = capital * (1.0 + leg_eff * gap_prev[index])
        if turnover[index] != 0.0:
            capital = capital * (1.0 - leg_eff * turnover[index])
        if counts[index]:
            leg_notional = capital * leg_eff
            min_leg = np.minimum(min_leg, leg_notional)
            capital = capital * (1.0 + leg_eff * intraday[index])
            capital = capital * (1.0 + leg_eff * funding[index])
        peak = np.maximum(peak, capital)
        max_dd = np.maximum(max_dd, 1.0 - capital / peak)
    return {"final": capital, "mdd": max_dd, "min_leg": min_leg, "leg": leg_grid, "cap": cap_grid}


def verify_against_engine38(panel: CarryPanel) -> dict:
    """This evaluator must reproduce engine38 for the same configuration.

    Without this the speedup is worthless: a fast wrong answer is worse than a slow right one, and wave50
    showed how convincingly a bug can imitate a discovery.
    """
    from research.wave38_breadth.engine38 import CarryConfig, simulate

    n_days = len(panel.days)
    checks = []
    for top_k, leg, cap in ((1, 0.50, 0.50), (2, 0.50, 1.00), (3, 0.25, 0.75)):
        reference = simulate(panel, CarryConfig(top_k, leg, cap), 1, n_days)
        series = build_daily_series(panel, top_k)
        grid = evaluate_grid(series, np.array([leg]), np.array([cap]), 1, n_days)
        mine = float(grid["final"][0, 0])
        checks.append({
            "config": f"k{top_k} leg{leg} cap{cap}",
            "engine38": reference.final,
            "fast53": mine,
            "relative_gap": abs(mine - reference.final) / reference.final if reference.final else float("inf"),
        })
    return {"checks": checks}


def main() -> int:
    panel = with_threshold(build_panel(), 0.15)
    print("=== fast53 검증: 벡터화 평가기가 engine38과 일치하는가 ===")
    report = verify_against_engine38(panel)
    for check in report["checks"]:
        print(f"  {check['config']:22s} engine38 ${check['engine38']:9.2f} · fast53 ${check['fast53']:9.2f} "
              f"· 상대오차 {check['relative_gap']:.2e}")
    worst = max(c["relative_gap"] for c in report["checks"])
    print(f"\n  최대 상대오차 {worst:.2e} — {'일치' if worst < 1e-6 else '불일치, 사용 불가'}")

    import time
    print("\n=== 속도 비교 ===")
    series = build_daily_series(panel, 3)
    legs = np.linspace(0.05, 1.00, 32)
    caps = np.linspace(0.10, 1.00, 32)
    started = time.time()
    evaluate_grid(series, legs, caps, 1, len(panel.days))
    elapsed = time.time() - started
    print(f"  fast53: {len(legs)*len(caps):,}조합 {elapsed:.3f}s -> {len(legs)*len(caps)/max(elapsed,1e-9):,.0f}조합/초")
    print(f"  engine38 기준 36ms/조합 -> 같은 조합 수에 {len(legs)*len(caps)*0.036:,.0f}s 필요")
    print(f"  가속 약 {len(legs)*len(caps)*0.036/max(elapsed,1e-9):,.0f}배")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
