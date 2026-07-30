# Wave-30 evaluation layer: walk-forward fitness, MAP-Elites behaviour descriptors, and the
# three NSGA-II objectives. Everything here reads IS bars only; `final_evaluation` is the
# single function permitted to look past OOS_SPLIT and it is called exactly once, from
# run_wave30.py, on the already-chosen candidate.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import MarketCache, OOSLeakageError, OOS_SPLIT, i5_baseline_total_curve
from research.wave30_qd.engine30 import (
    SLEEVE_DEAD_THRESHOLD,
    TOTAL_CAPITAL,
    Wave30Result,
    annualized_return,
    max_drawdown,
    run_genome,
)
from research.wave30_qd.genome30 import Genome

WF_FOLDS: Final = 4
BOOTSTRAP_PATHS: Final = 10_000  # full value, used by gate P3 where the verdict is decided
SEARCH_BOOTSTRAP_PATHS: Final = 2_000  # search-time value; SPEC.md "부트스트랩 경로수" explains the split
MIN_TRADES_FOR_FITNESS: Final = 20  # below this the walk-forward folds are not estimable
WIPE_FRACTION: Final = 1e-3  # sleeve down 99.9% of its starting value == wiped

# MAP-Elites behaviour grid (SPEC.md, frozen)
LEVERAGE_EDGES: Final = (1.0, 2.0, 4.0, 7.0, 11.0, 15.0, 20.0001)
MDD_EDGES: Final = (0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0001)
FREQUENCY_EDGES: Final = (0.0, 12.0, 50.0, 150.0, 400.0, np.inf)
GRID_SHAPE: Final = (len(LEVERAGE_EDGES) - 1, len(MDD_EDGES) - 1, len(FREQUENCY_EDGES) - 1)


@dataclass
class Evaluation:
    genome: Genome
    fitness: float  # walk-forward consistency CAGR of the TOTAL system (median - stdev)
    fold_cagrs: tuple[float, ...]
    is_total_cagr: float
    is_total_final: float
    sleeve_mdd: float
    total_mdd: float
    trades_per_year: float
    n_trades: int
    n_liquidations: int
    wipe_probability: float
    descriptor: tuple[int, int, int]
    mean_leverage: float
    min_notional_usdt: float
    sleeve_survived: bool
    extras: dict = field(default_factory=dict)

    @property
    def objectives(self) -> tuple[float, float, float]:
        """NSGA-II objective vector, all three expressed as MINIMISATION."""
        return (-self.fitness, self.sleeve_mdd, self.wipe_probability)


def _fold_boundaries(n_days: int, folds: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_days - 1, folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(folds)]


def bootstrap_wipe_probability(trade_returns: np.ndarray, rng: np.random.Generator, paths: int = SEARCH_BOOTSTRAP_PATHS) -> float:
    """P(the sleeve is wiped) under i.i.d. resampling of the realised trade returns.

    "Wiped" = the sleeve lost at least (1 - WIPE_FRACTION) of its value at ANY point on the
    path, not merely at the end: once a leveraged sleeve is down 99.9% it cannot fund a
    position and the engine stops trading it, so a path that died mid-way and drifted back up
    on paper still counts as dead. Hence the running minimum rather than the final value.
    The threshold is expressed as a FRACTION of the starting sleeve so it means the same thing
    for a $10 sleeve and a $100 one.

    Computed in LOG space with float32: the direct cumulative product of several hundred
    sub-1.0 factors underflows float32 to a spurious zero, which would silently inflate the
    reported wipe probability. log1p(-1.0) = -inf propagates correctly through the cumsum,
    which is exactly the desired "a liquidation kills the path permanently" behaviour.
    """
    n = len(trade_returns)
    if n == 0:
        return 0.0
    with np.errstate(divide="ignore"):
        logs = np.log1p(np.asarray(trade_returns, dtype=np.float64)).astype(np.float32)
    draws = rng.integers(0, n, size=(paths, n), dtype=np.int32)
    sampled = logs[draws]
    np.cumsum(sampled, axis=1, out=sampled)
    worst = sampled.min(axis=1)
    return float((worst <= np.float32(np.log(WIPE_FRACTION))).mean())


def descriptor_of(mean_leverage: float, sleeve_mdd: float, trades_per_year: float) -> tuple[int, int, int]:
    """Bin a run into its MAP-Elites cell. `sleeve_mdd` is passed as a POSITIVE magnitude."""
    lev_bin = int(np.clip(np.searchsorted(LEVERAGE_EDGES, max(mean_leverage, 1.0), side="right") - 1, 0, GRID_SHAPE[0] - 1))
    mdd_bin = int(np.clip(np.searchsorted(MDD_EDGES, sleeve_mdd, side="right") - 1, 0, GRID_SHAPE[1] - 1))
    freq_bin = int(np.clip(np.searchsorted(FREQUENCY_EDGES, trades_per_year, side="right") - 1, 0, GRID_SHAPE[2] - 1))
    return lev_bin, mdd_bin, freq_bin


def evaluate(cache: MarketCache, genome: Genome, rng: np.random.Generator) -> Evaluation:
    """IS-only evaluation. Never reads a bar past OOS_SPLIT (run_genome mode='is')."""
    result = run_genome(cache, genome, mode="is")
    return _summarise(cache, genome, result, rng, span="is")


def _summarise(
    cache: MarketCache, genome: Genome, result: Wave30Result, rng: np.random.Generator, span: str
) -> Evaluation:
    valid = result.daily_valid
    total = result.total_equity_daily[valid]
    sleeve = result.sleeve_equity_daily[valid]
    days = float(len(total) - 1)

    total_cagr = annualized_return(total, days)
    sleeve_mdd = abs(max_drawdown(sleeve)) if result.sleeve_start_usdt > 0 else 0.0
    total_mdd_value = abs(max_drawdown(total))
    years = max(days / 365.0, 1e-9)
    trades_per_year = len(result.trades) / years

    fold_cagrs: list[float] = []
    for start, end in _fold_boundaries(len(total), WF_FOLDS):
        if end <= start:
            continue
        segment = total[start : end + 1]
        fold_cagrs.append(annualized_return(segment, float(end - start)))
    if len(result.trades) < MIN_TRADES_FOR_FITNESS or not fold_cagrs:
        # Too few trades for the walk-forward statistic to mean anything. Report the raw CAGR
        # with a hard penalty so such genomes can still occupy an archive cell (the archive is
        # meant to be honest about sparse regions) but never win one on noise.
        fitness = total_cagr - 1.0
    else:
        fitness = float(np.median(fold_cagrs) - np.std(fold_cagrs, ddof=0))

    wipe = bootstrap_wipe_probability(result.trade_returns, rng)
    return Evaluation(
        genome=genome,
        fitness=float(fitness),
        fold_cagrs=tuple(float(x) for x in fold_cagrs),
        is_total_cagr=float(total_cagr),
        is_total_final=float(total[-1]),
        sleeve_mdd=float(sleeve_mdd),
        total_mdd=float(total_mdd_value),
        trades_per_year=float(trades_per_year),
        n_trades=len(result.trades),
        n_liquidations=result.n_liquidations,
        wipe_probability=wipe,
        descriptor=descriptor_of(result.mean_realized_leverage, sleeve_mdd, trades_per_year),
        mean_leverage=float(result.mean_realized_leverage),
        min_notional_usdt=float(result.min_notional_usdt),
        sleeve_survived=bool(sleeve[-1] > SLEEVE_DEAD_THRESHOLD),
        extras={"span": span, "days": days},
    )


def final_evaluation(cache: MarketCache, genome: Genome, rng: np.random.Generator) -> dict:
    """THE single OOS unsealing (SPEC.md contamination block #1). Runs the full span and
    reports IS, OOS and full-period figures side by side against the $100-basis I5 baseline.
    Must be called exactly once per wave, from run_wave30.py, after the candidate is chosen."""
    result = run_genome(cache, genome, mode="full")
    summary = _summarise(cache, genome, result, rng, span="full")

    daily = cache.daily_index
    oos_start = int(daily.searchsorted(OOS_SPLIT, side="right"))
    total = result.total_equity_daily
    baseline = i5_baseline_total_curve(cache)

    def window(curve: np.ndarray, start: int, end: int) -> dict:
        segment = curve[start : end + 1]
        span_days = float(end - start)
        return {
            "start_usdt": float(segment[0]),
            "end_usdt": float(segment[-1]),
            "days": span_days,
            "annualized": annualized_return(segment, span_days),
            "mdd": float(abs(max_drawdown(segment))),
        }

    last = len(total) - 1
    is_end = max(0, oos_start - 1)
    return {
        "genome": genome.to_dict(),
        "is": window(total, 0, is_end),
        "oos": window(total, oos_start, last),
        "full": window(total, 0, last),
        "baseline_is": window(baseline, 0, is_end),
        "baseline_oos": window(baseline, oos_start, last),
        "baseline_full": window(baseline, 0, last),
        "sleeve_mdd_full": summary.sleeve_mdd,
        "n_trades_full": summary.n_trades,
        "n_liquidations_full": summary.n_liquidations,
        "mean_leverage": summary.mean_leverage,
        "min_notional_usdt": summary.min_notional_usdt,
        "sleeve_survived": summary.sleeve_survived,
        "trades_per_year": summary.trades_per_year,
        "oos_start_day": str(daily[oos_start]) if oos_start < len(daily) else None,
        "_result": result,
        "_summary": summary,
    }


def baseline_reference(cache: MarketCache) -> dict:
    """The $100-basis I5 baseline scored with the SAME walk-forward statistic the archive uses.

    Without this number the archive's fitness values are uninterpretable: a genome whose sleeve
    dies instantly still scores whatever the surviving stable leg earns, so "positive fitness"
    proves nothing. This is the line a candidate has to clear to have added anything at all.
    """
    baseline = i5_baseline_total_curve(cache)
    is_days = int(cache.day_of_bar[int(cache.is_mask.sum()) - 1]) + 1
    is_curve = baseline[:is_days]
    folds = []
    for start, end in _fold_boundaries(len(is_curve), WF_FOLDS):
        if end > start:
            folds.append(annualized_return(is_curve[start : end + 1], float(end - start)))
    return {
        "is_fitness": float(np.median(folds) - np.std(folds, ddof=0)) if folds else 0.0,
        "is_fold_cagrs": [float(x) for x in folds],
        "is_cagr": annualized_return(is_curve, float(len(is_curve) - 1)),
        "is_final_usdt": float(is_curve[-1]),
        "full_cagr": annualized_return(baseline, float(len(baseline) - 1)),
        "full_final_usdt": float(baseline[-1]),
        "is_mdd": float(abs(max_drawdown(is_curve))),
    }


def assert_no_oos_access(mode: str) -> None:
    if mode != "is":
        raise OOSLeakageError(f"search loop attempted mode={mode!r}; only 'is' is permitted")
