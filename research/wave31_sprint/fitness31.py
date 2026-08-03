# Wave-31 sprint objective: "how much can $100 make in a SHORT time, and what does that cost".
#
# ---------------------------------------------------------------------------------------
# Why the fitness is the MEDIAN 30-day return and not the top quartile
# ---------------------------------------------------------------------------------------
# wave23 asked this same question with `top-quartile 60d window mean - 3*P(window loss>20%)`
# and evolution answered with all-in high-volatility momentum (MDD 91.5%, single worst loss
# 82.3% of principal). That was not a mis-tuned penalty: maximising an upper quantile IS
# maximising the tail, and a tail is cheapest to buy with maximum position size. Raising the
# penalty coefficient only moves the equilibrium, it does not change the direction.
#
# The median cannot be bought with a lucky window. A genome that bets everything and wins
# spectacularly one month in twelve has a median 30-day return near its typical (losing) month,
# so the same search pressure that produced wave23's blow-up now works against it.
#
# The number the user actually asked for -- "maximised return" -- is reported as p95 of the
# same rolling distribution, i.e. "what happened when it went well", NEXT TO the probability of
# being halved in that same window. It is an output, never a target.
#
# ---------------------------------------------------------------------------------------
# Rolling windows are overlapping, and that is deliberate
# ---------------------------------------------------------------------------------------
# Every calendar start date is used, so the windows overlap heavily and the resulting sample is
# autocorrelated. That is fine for the question being asked ("if I start on an arbitrary day,
# what happens in the next W days") and it is NOT used as an inference sample -- no p-value is
# computed from it. The statistical gates (Q3 wipe probability, Q4 ruin, Q5 DSR) all run on
# independent machinery: trade-return bootstrap and daily-return bootstrap.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import MarketCache, OOS_SPLIT, i5_baseline_total_curve
from research.wave30_qd.engine30 import (
    SLEEVE_DEAD_THRESHOLD,
    Wave30Result,
    annualized_return,
    max_drawdown,
    run_genome,
)
from research.wave30_qd.fitness30 import (
    LEVERAGE_EDGES,
    FREQUENCY_EDGES,
    Evaluation,
    bootstrap_wipe_probability,
)
from research.wave30_qd.genome30 import Genome

WINDOWS: Final = (7, 14, 30, 90, 180, 365)  # days, frozen in SPEC.md
FITNESS_WINDOW: Final = 30  # pre-designated in SPEC.md so no post-hoc window shopping
MIN_TRADES_FOR_FITNESS: Final = 20

# MAP-Elites axis 2 for this wave: P(30-day loss > 50%)
HALVING_EDGES: Final = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 1.0001)
GRID_SHAPE: Final = (len(LEVERAGE_EDGES) - 1, len(HALVING_EDGES) - 1, len(FREQUENCY_EDGES) - 1)


def rolling_window_returns(curve: np.ndarray, window_days: int) -> np.ndarray:
    """Return over every W-day window, one per possible start date."""
    curve = np.asarray(curve, dtype=float)
    if len(curve) <= window_days:
        return np.zeros(0)
    start = curve[:-window_days]
    end = curve[window_days:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(start > 0, end / np.maximum(start, 1e-12) - 1.0, -1.0)
    return np.nan_to_num(out, nan=-1.0, posinf=0.0, neginf=-1.0)


def window_statistics(curve: np.ndarray, window_days: int) -> dict:
    returns = rolling_window_returns(curve, window_days)
    if len(returns) == 0:
        return {
            "window_days": window_days,
            "n_windows": 0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "best": 0.0,
            "worst": 0.0,
            "positive_share": 0.0,
            "prob_loss_over_50": 1.0,
            "prob_loss_over_90": 1.0,
        }
    return {
        "window_days": window_days,
        "n_windows": int(len(returns)),
        "p05": float(np.percentile(returns, 5)),
        "p50": float(np.percentile(returns, 50)),
        "p95": float(np.percentile(returns, 95)),
        "best": float(returns.max()),
        "worst": float(returns.min()),
        "positive_share": float((returns > 0).mean()),
        "prob_loss_over_50": float((returns <= -0.50).mean()),
        "prob_loss_over_90": float((returns <= -0.90).mean()),
    }


def days_to_multiple(curve: np.ndarray, multiple: float = 2.0) -> int | None:
    """Calendar days until the total capital FIRST reaches `multiple` of its start value.
    Measured from day 0 only -- this is the "how fast" half of the question."""
    curve = np.asarray(curve, dtype=float)
    if len(curve) == 0 or curve[0] <= 0:
        return None
    reached = np.flatnonzero(curve >= curve[0] * multiple)
    return int(reached[0]) if len(reached) else None


def sprint_profile(curve: np.ndarray) -> dict:
    return {
        "windows": {str(w): window_statistics(curve, w) for w in WINDOWS},
        "days_to_2x": days_to_multiple(curve, 2.0),
        "days_to_5x": days_to_multiple(curve, 5.0),
        "days_to_10x": days_to_multiple(curve, 10.0),
    }


def descriptor_of(mean_leverage: float, prob_halving: float, trades_per_year: float) -> tuple[int, int, int]:
    lev_bin = int(np.clip(np.searchsorted(LEVERAGE_EDGES, max(mean_leverage, 1.0), side="right") - 1, 0, GRID_SHAPE[0] - 1))
    halving_bin = int(np.clip(np.searchsorted(HALVING_EDGES, prob_halving, side="right") - 1, 0, GRID_SHAPE[1] - 1))
    freq_bin = int(np.clip(np.searchsorted(FREQUENCY_EDGES, trades_per_year, side="right") - 1, 0, GRID_SHAPE[2] - 1))
    return lev_bin, halving_bin, freq_bin


def evaluate_sprint(cache: MarketCache, genome: Genome, rng: np.random.Generator) -> Evaluation:
    """IS-only sprint evaluation. Signature matches fitness30.evaluate so it can be injected
    straight into search30.Evaluator; run_genome(mode='is') keeps the OOS seal intact."""
    result = run_genome(cache, genome, mode="is")
    return summarise_sprint(cache, genome, result, rng)


def summarise_sprint(
    cache: MarketCache, genome: Genome, result: Wave30Result, rng: np.random.Generator
) -> Evaluation:
    valid = result.daily_valid
    total = result.total_equity_daily[valid]
    sleeve = result.sleeve_equity_daily[valid]
    days = float(len(total) - 1)
    years = max(days / 365.0, 1e-9)

    profile = sprint_profile(total)
    focus = profile["windows"][str(FITNESS_WINDOW)]
    fitness = focus["p50"]
    if len(result.trades) < MIN_TRADES_FOR_FITNESS:
        # Too few trades for the rolling statistic to describe a strategy rather than an
        # accident. Penalised, but still allowed to occupy a cell so the map stays honest
        # about sparse regions.
        fitness -= 1.0

    trades_per_year = len(result.trades) / years
    wipe = bootstrap_wipe_probability(result.trade_returns, rng)
    return Evaluation(
        genome=genome,
        fitness=float(fitness),
        fold_cagrs=(),
        is_total_cagr=annualized_return(total, days),
        is_total_final=float(total[-1]),
        sleeve_mdd=float(abs(max_drawdown(sleeve))),
        total_mdd=float(abs(max_drawdown(total))),
        trades_per_year=float(trades_per_year),
        n_trades=len(result.trades),
        n_liquidations=result.n_liquidations,
        wipe_probability=wipe,
        descriptor=descriptor_of(result.mean_realized_leverage, focus["prob_loss_over_50"], trades_per_year),
        mean_leverage=float(result.mean_realized_leverage),
        min_notional_usdt=float(result.min_notional_usdt),
        sleeve_survived=bool(sleeve[-1] > SLEEVE_DEAD_THRESHOLD),
        extras={
            "sprint": profile,
            "prob_halving_30d": focus["prob_loss_over_50"],
            # SPEC.md NSGA-II objectives: maximise median 30d return, minimise P(30d halving),
            # minimise sleeve drawdown. Written here so search30's sorter needs no wave31 import.
            "objective_vector": (
                -float(fitness),
                float(focus["prob_loss_over_50"]),
                float(abs(max_drawdown(sleeve))),
            ),
        },
    )


def baseline_sprint_profile(cache: MarketCache) -> dict:
    """The $100-basis I5 baseline measured with the SAME rolling statistics. Without this the
    candidate's numbers have no reference: "median 30-day return +4%" means nothing until you
    know I5's is +0.8%."""
    baseline = i5_baseline_total_curve(cache)
    is_days = int(cache.day_of_bar[int(cache.is_mask.sum()) - 1]) + 1
    oos_start = int(cache.daily_index.searchsorted(OOS_SPLIT, side="right"))
    return {
        "is": sprint_profile(baseline[:is_days]),
        "oos": sprint_profile(baseline[oos_start:]),
        "full": sprint_profile(baseline),
    }
