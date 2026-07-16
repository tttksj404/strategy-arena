"""Deterministic statistics used by the deep validation runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import e, sqrt
from statistics import NormalDist
from typing import Final, Mapping, Sequence

import numpy as np


INITIAL_CAPITAL: Final = 300.0
MC_PATHS: Final = 10_000
REGIME_PATHS: Final = 1_000
TRADES_PER_WEEK: Final = 21
ANNUAL_DAYS: Final = 365.0


@dataclass(frozen=True, slots=True)
class DeepValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class TimedValue:
    """A timestamped scalar from a result or a local cache."""

    timestamp: datetime
    value: float


@dataclass(frozen=True, slots=True)
class CapitalDistribution:
    """Summary of a simulated final-capital distribution."""

    p05: float
    ruin_probability: float
    mean: float
    median: float


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Trade bootstrap and Kelly sizing outputs."""

    unit: CapitalDistribution
    quarter_kelly: CapitalDistribution
    kelly_fraction: float
    quarter_kelly_fraction: float
    paths: int
    trade_count: int


@dataclass(frozen=True, slots=True)
class LeaveOneOutRow:
    """One calendar-year pseudo-OOS comparison."""

    year: int
    heldout_return: float
    heldout_sharpe: float
    remaining_return: float
    remaining_sharpe: float
    heldout_days: int
    remaining_days: int


@dataclass(frozen=True, slots=True)
class DsrResult:
    """Deflated Sharpe score and its probability interpretation."""

    observed_sharpe: float
    benchmark_sharpe: float
    score: float
    probability: float
    n_days: int
    skew: float
    kurtosis: float
    trials: int


@dataclass(frozen=True, slots=True)
class BlockBootstrapResult:
    """MDD and final-capital distribution from 90-day block permutations."""

    block_days: int
    block_count: int
    paths: int
    mdd_p05: float
    mdd_median: float
    mdd_p95: float
    final_p05: float
    final_median: float
    final_p95: float
    final_capital_invariant: bool


@dataclass(frozen=True, slots=True)
class FundingComparison:
    """Aggregate Binance/Bitget funding-score comparison."""

    symbols: tuple[str, ...]
    observations: int
    score_correlation: float
    entry_agreement: float
    sign_agreement: float
    coverage_days: int


def _capital_summary(finals: np.ndarray) -> CapitalDistribution:
    return CapitalDistribution(
        p05=float(np.quantile(finals, 0.05)),
        ruin_probability=float(np.mean(finals < INITIAL_CAPITAL / 2.0)),
        mean=float(np.mean(finals)),
        median=float(np.median(finals)),
    )


def _simulate_capital(trades: np.ndarray, scale: float, rng: np.random.Generator) -> CapitalDistribution:
    scaled = np.clip(scale * trades, -0.999999, None)
    finals = np.empty(MC_PATHS, dtype=float)
    for start in range(0, MC_PATHS, 500):
        stop = min(start + 500, MC_PATHS)
        samples = rng.choice(scaled, size=(stop - start, len(scaled)), replace=True)
        finals[start:stop] = INITIAL_CAPITAL * np.prod(1.0 + samples, axis=1)
    return _capital_summary(finals)


def kelly_fraction(trades: Sequence[float]) -> float:
    """Return the existing gate-compatible mean/variance Kelly estimate."""
    values = np.asarray(tuple(trades), dtype=float)
    variance = float(values.var(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean() / variance) if variance > 0.0 else 0.0


def trade_bootstrap(trades: Sequence[float], seed: int) -> BootstrapResult:
    """Run 10,000 full-period trade resamples at unit and quarter-Kelly scale."""
    values = np.asarray(tuple(trades), dtype=float)
    if values.size == 0:
        raise DeepValidationError("trade bootstrap requires at least one trade")
    fraction = kelly_fraction(values)
    rng = np.random.default_rng(seed)
    return BootstrapResult(
        unit=_simulate_capital(values, 1.0, rng),
        quarter_kelly=_simulate_capital(values, 0.25 * fraction, rng),
        kelly_fraction=fraction,
        quarter_kelly_fraction=0.25 * fraction,
        paths=MC_PATHS,
        trade_count=int(values.size),
    )


def _sharpe(values: Sequence[float]) -> float:
    returns = np.asarray(tuple(values), dtype=float)
    volatility = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    return float(returns.mean() / volatility * sqrt(ANNUAL_DAYS)) if volatility > 0.0 else 0.0


def _compound(values: Sequence[float]) -> float:
    returns = np.asarray(tuple(values), dtype=float)
    return float(np.prod(1.0 + np.clip(returns, -0.999999, None)) - 1.0)


def daily_returns(equity: Sequence[TimedValue]) -> tuple[TimedValue, ...]:
    """Convert ordered equity observations into timestamped simple returns."""
    ordered = tuple(sorted(equity, key=lambda item: item.timestamp))
    result: list[TimedValue] = []
    for previous, current in zip(ordered, ordered[1:]):
        if previous.value <= 0.0:
            raise DeepValidationError("equity must remain positive")
        result.append(TimedValue(current.timestamp, current.value / previous.value - 1.0))
    return tuple(result)


def leave_one_year_out(equity: Sequence[TimedValue]) -> tuple[LeaveOneOutRow, ...]:
    """Compare each calendar year with the daily history left after removing it."""
    daily = daily_returns(equity)
    years = sorted({item.timestamp.year for item in daily})
    rows: list[LeaveOneOutRow] = []
    for year in years:
        heldout = tuple(item.value for item in daily if item.timestamp.year == year)
        remaining = tuple(item.value for item in daily if item.timestamp.year != year)
        rows.append(LeaveOneOutRow(
            year=year,
            heldout_return=_compound(heldout),
            heldout_sharpe=_sharpe(heldout),
            remaining_return=_compound(remaining),
            remaining_sharpe=_sharpe(remaining),
            heldout_days=len(heldout),
            remaining_days=len(remaining),
        ))
    return tuple(rows)


def _normal_ppf(probability: float) -> float:
    return NormalDist().inv_cdf(probability)


def deflated_sharpe(equity: Sequence[TimedValue], trials: int = 28) -> DsrResult:
    """Calculate Bailey-López de Prado's deflated Sharpe z-score."""
    returns = np.asarray(tuple(item.value for item in daily_returns(equity)), dtype=float)
    if returns.size < 3:
        raise DeepValidationError("DSR requires at least three daily returns")
    mean = float(returns.mean())
    centered = returns - mean
    variance = float(np.mean(centered**2))
    standard_deviation = sqrt(variance)
    skew = float(np.mean(centered**3) / standard_deviation**3) if standard_deviation > 0.0 else 0.0
    kurtosis = float(np.mean(centered**4) / standard_deviation**4) if standard_deviation > 0.0 else 3.0
    observed = _sharpe(returns)
    sr_variance = (1.0 - skew * observed + (kurtosis - 1.0) * observed**2 / 4.0) / (returns.size - 1)
    sr_standard_error = sqrt(max(sr_variance, 1e-18))
    gamma = 0.5772156649015329
    expected_max_z = (1.0 - gamma) * _normal_ppf(1.0 - 1.0 / trials)
    expected_max_z += gamma * _normal_ppf(1.0 - 1.0 / (trials * e))
    benchmark = sr_standard_error * expected_max_z
    score = (observed - benchmark) / sr_standard_error
    return DsrResult(observed, benchmark, score, float(NormalDist().cdf(score)), int(returns.size), skew, kurtosis, trials)


def _max_drawdown(capital: np.ndarray) -> float:
    curve = INITIAL_CAPITAL * np.cumprod(1.0 + np.clip(capital, -0.999999, None))
    peaks = np.maximum.accumulate(np.concatenate(([INITIAL_CAPITAL], curve)))
    return float(np.max(1.0 - curve / peaks[1:]))


def block_bootstrap(trades: Sequence[TimedValue], seed: int, block_days: int = 90) -> BlockBootstrapResult:
    """Permute complete 90-day trade blocks 1,000 times, preserving block order internally."""
    ordered = tuple(sorted(trades, key=lambda item: item.timestamp))
    if not ordered: raise DeepValidationError("block bootstrap requires at least one trade")
    anchor = ordered[0].timestamp
    grouped: dict[int, list[float]] = {}
    for item in ordered:
        block = (item.timestamp - anchor).days // block_days
        grouped.setdefault(block, []).append(item.value)
    blocks = tuple(np.asarray(grouped[key], dtype=float) for key in sorted(grouped))
    rng = np.random.default_rng(seed)
    mdds = np.empty(REGIME_PATHS, dtype=float)
    finals = np.empty(REGIME_PATHS, dtype=float)
    for index in range(REGIME_PATHS):
        path = np.concatenate(tuple(blocks[position] for position in rng.permutation(len(blocks))))
        mdds[index] = _max_drawdown(path)
        finals[index] = INITIAL_CAPITAL * float(np.prod(1.0 + np.clip(path, -0.999999, None)))
    return BlockBootstrapResult(
        block_days=block_days,
        block_count=len(blocks),
        paths=REGIME_PATHS,
        mdd_p05=float(np.quantile(mdds, 0.05)),
        mdd_median=float(np.median(mdds)),
        mdd_p95=float(np.quantile(mdds, 0.95)),
        final_p05=float(np.quantile(finals, 0.05)),
        final_median=float(np.median(finals)),
        final_p95=float(np.quantile(finals, 0.95)),
        final_capital_invariant=bool(np.allclose(finals, finals[0], rtol=1e-12, atol=1e-12)),
    )


def _bucket(timestamp: datetime) -> datetime:
    hour = timestamp.hour - timestamp.hour % 8
    return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)


def _bucketed_values(values: Sequence[TimedValue]) -> dict[datetime, float]:
    bucketed: dict[datetime, float] = {}
    for item in sorted(values, key=lambda value: value.timestamp): bucketed.setdefault(_bucket(item.timestamp), item.value)
    return bucketed


def _rolling_score(timestamps: Sequence[datetime], values: Sequence[float]) -> tuple[float, ...]:
    scores: list[float] = []
    for index in range(len(values)):
        window_timestamps = timestamps[max(0, index - TRADES_PER_WEEK + 1):index + 1]
        window = values[max(0, index - TRADES_PER_WEEK + 1):index + 1]
        contiguous = len(window) == TRADES_PER_WEEK and all(
            right - left == timedelta(hours=8)
            for left, right in zip(window_timestamps, window_timestamps[1:])
        )
        scores.append(float(np.mean(window) * 3.0 * ANNUAL_DAYS) if contiguous else float("nan"))
    return tuple(scores)


def compare_funding(
    bitget: Mapping[str, Sequence[TimedValue]],
    binance: Mapping[str, Sequence[TimedValue]],
    threshold_apr: float,
) -> FundingComparison:
    score_pairs: list[tuple[float, float]] = []
    symbols: list[str] = []
    global_timestamps: list[datetime] = []
    for symbol in sorted(set(bitget) & set(binance)):
        left, right = _bucketed_values(bitget[symbol]), _bucketed_values(binance[symbol])
        timestamps = sorted(set(left) & set(right))
        if len(timestamps) < TRADES_PER_WEEK: continue
        left_scores = _rolling_score(timestamps, tuple(left[timestamp] for timestamp in timestamps))
        right_scores = _rolling_score(timestamps, tuple(right[timestamp] for timestamp in timestamps))
        pairs = [(left_scores[i], right_scores[i]) for i in range(len(timestamps)) if np.isfinite(left_scores[i]) and np.isfinite(right_scores[i])]
        if not pairs: continue
        symbols.append(symbol)
        global_timestamps.extend(timestamps[i] for i in range(len(timestamps)) if np.isfinite(left_scores[i]) and np.isfinite(right_scores[i]))
        score_pairs.extend(pairs)
    if not score_pairs: return FundingComparison((), 0, 0.0, 0.0, 0.0, 0)
    values = np.asarray(score_pairs, dtype=float)
    correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]) if np.std(values[:, 0]) > 0 and np.std(values[:, 1]) > 0 else 0.0
    entry = float(np.mean((values[:, 0] > threshold_apr) == (values[:, 1] > threshold_apr)))
    sign = float(np.mean(np.sign(values[:, 0]) == np.sign(values[:, 1])))
    coverage = (max(global_timestamps) - min(global_timestamps)).days + 1
    return FundingComparison(tuple(symbols), len(score_pairs), correlation, entry, sign, coverage)
