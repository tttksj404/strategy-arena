# Wave-23 fitness function (SPEC.md "목표함수 (핵심 변경)"):
#   fitness = 상위 25% (60일 롤링창) 수익률의 평균  -  3 x P(60일 창 수익률 < -20%)
# Deliberately NOT CAGR/Sharpe (SPEC.md: "CAGR·샤프를 목표로 쓰지 않는다") -- this rewards an
# equity curve that produces frequent, large 60-day gains (the "단기 대박" the task asks the GA
# to search for) while strongly penalizing any tendency to produce a 60-day window that loses
# more than 20% (the "단기 파멸" case a short-term-profit objective could otherwise happily
# ignore, since a maximize-upside-only objective has no organic reason to avoid occasional
# blowups). The x3 penalty weight and -20%/top-25% thresholds are SPEC.md's own frozen numbers.

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
import pandas as pd  # noqa: PANDAS_OK

from research.validation.deep_stats import DeepValidationError, TimedValue, deflated_sharpe
from research.wave10_carry100.engine import OOS_SPLIT
from research.wave23_ga_short import engine23
from research.wave23_ga_short.engine23 import MarketCache
from research.wave23_ga_short.genome23 import Genome, genome_key

ROLLING_WINDOW_DAYS: Final = 60  # SPEC.md "60일 롤링창"
TOP_QUANTILE_FRACTION: Final = 0.25  # SPEC.md "상위 25%"
RUIN_WINDOW_RETURN_THRESHOLD: Final = -0.20  # SPEC.md "창수익 < -20%"
RUIN_PENALTY_WEIGHT: Final = 3.0  # SPEC.md "3 x P(...)"


@dataclass(frozen=True, slots=True)
class FitnessResult:
    fitness: float
    n_windows: int
    top_quantile_mean_return: float
    p_window_ruin: float
    median_window_return: float
    full_period_cagr: float
    mdd: float


def rolling_window_returns(equity: pd.Series, window_days: int = ROLLING_WINDOW_DAYS) -> np.ndarray:
    """Every window_days-apart pair of observations in `equity` treated as one non-overlapping
    -in-STARTPOINT-but-overlapping-in-general rolling window's return -- i.e. values[i+window]/
    values[i] - 1 for every valid i, the standard rolling-window-return construction (matches
    e.g. research.validation.deep_stats' own daily_returns' "every consecutive pair" idea, just
    at lag=window_days instead of lag=1). Non-finite entries (a non-positive equity value,
    which should not occur but is guarded the same way engine23/gates23 guard it elsewhere) are
    dropped rather than propagated."""
    values = equity.to_numpy(dtype=float)
    n = len(values)
    if n <= window_days:
        return np.asarray([], dtype=float)
    start = values[: n - window_days]
    end = values[window_days:]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = end / start - 1.0
    return returns[np.isfinite(returns)]


def compute_fitness(equity: pd.Series) -> FitnessResult:
    window_returns = rolling_window_returns(equity)
    full_cagr = engine23.cagr(equity)
    mdd = engine23.max_drawdown(equity)
    if window_returns.size == 0:
        # Too short to form even one 60-day window (can happen on a short OOS slice) -- fail
        # closed to the worst possible fitness rather than silently returning 0.0, so a
        # too-short series can never look "safe" (no windows observed is not the same as no
        # ruin observed).
        return FitnessResult(fitness=-1.0, n_windows=0, top_quantile_mean_return=0.0, p_window_ruin=1.0, median_window_return=0.0, full_period_cagr=full_cagr, mdd=mdd)
    cutoff = np.quantile(window_returns, 1.0 - TOP_QUANTILE_FRACTION)
    top_mask = window_returns >= cutoff
    top_mean = float(np.mean(window_returns[top_mask])) if np.any(top_mask) else 0.0
    p_ruin = float(np.mean(window_returns < RUIN_WINDOW_RETURN_THRESHOLD))
    fitness = top_mean - RUIN_PENALTY_WEIGHT * p_ruin
    return FitnessResult(
        fitness=fitness,
        n_windows=int(window_returns.size),
        top_quantile_mean_return=top_mean,
        p_window_ruin=p_ruin,
        median_window_return=float(np.median(window_returns)),
        full_period_cagr=full_cagr,
        mdd=mdd,
    )


def evaluate_genome(genome: Genome, cache: MarketCache) -> FitnessResult:
    """The ONLY fitness entry point ga23.py/random_search23.py ever call. mode is hardcoded to
    MODE_IS -- there is no parameter here through which a caller could request OOS data (same
    structural OOS seal as research.wave21_ga.fitness.evaluate_genome)."""
    equity = engine23.run_backtest(genome, cache, engine23.MODE_IS)
    return compute_fitness(equity)


def evaluate_genome_cached(genome: Genome, cache: MarketCache, fitness_cache: dict[tuple, FitnessResult]) -> tuple[FitnessResult, bool]:
    """evaluate_genome with a caller-owned cache dict (task instruction: '평가 캐싱 필수').
    Returns (result, was_cache_hit)."""
    key = genome_key(genome)
    if key in fitness_cache:
        return fitness_cache[key], True
    result = evaluate_genome(genome, cache)
    fitness_cache[key] = result
    return result, False


# ---------------------------------------------------------------------------
# One-time final evaluation (OOS seal opened here, and ONLY here).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinalEvaluation:
    genome: Genome
    full_equity: pd.Series
    is_equity: pd.Series
    oos_equity: pd.Series
    stress_equity: pd.Series
    is_fitness: FitnessResult
    oos_fitness: FitnessResult | None  # None if the OOS window is too short for a single 60-day window
    full_period_cagr: float
    mdd_full: float


def final_evaluation(genome: Genome, cache: MarketCache) -> FinalEvaluation:
    """Called AT MOST ONCE per wave run, on the single selected final candidate -- the only
    function in this package that runs the backtest over the OOS range at all."""
    full_equity = engine23.run_backtest(genome, cache, engine23.MODE_OOS_FINAL, stress=False)
    stress_equity = engine23.run_backtest(genome, cache, engine23.MODE_OOS_FINAL, stress=True)
    is_equity = full_equity[full_equity.index <= OOS_SPLIT]  # reading the <= side is never gated; only the > OOS_SPLIT side goes through oos_slice
    oos_equity = engine23.oos_slice(full_equity, engine23.MODE_OOS_FINAL)
    is_fitness = compute_fitness(is_equity)
    oos_fitness = compute_fitness(oos_equity) if len(oos_equity) > ROLLING_WINDOW_DAYS else None
    return FinalEvaluation(
        genome=genome,
        full_equity=full_equity,
        is_equity=is_equity,
        oos_equity=oos_equity,
        stress_equity=stress_equity,
        is_fitness=is_fitness,
        oos_fitness=oos_fitness,
        full_period_cagr=engine23.cagr(full_equity),
        mdd_full=engine23.max_drawdown(full_equity),
    )


def deflated_sharpe_for_trials(equity: pd.Series, trials: int) -> dict | None:
    clean = equity.dropna()
    if len(clean) < 4:
        return None
    timed = tuple(TimedValue(pd.Timestamp(ts).to_pydatetime(), float(value)) for ts, value in clean.items())
    try:
        dsr = deflated_sharpe(timed, trials=trials)
    except DeepValidationError:
        return None
    return {"score": dsr.score, "probability": dsr.probability, "trials": dsr.trials, "observed_sharpe": dsr.observed_sharpe}


__all__ = [
    "RUIN_PENALTY_WEIGHT",
    "RUIN_WINDOW_RETURN_THRESHOLD",
    "ROLLING_WINDOW_DAYS",
    "TOP_QUANTILE_FRACTION",
    "FinalEvaluation",
    "FitnessResult",
    "compute_fitness",
    "deflated_sharpe_for_trials",
    "evaluate_genome",
    "evaluate_genome_cached",
    "final_evaluation",
    "rolling_window_returns",
]
