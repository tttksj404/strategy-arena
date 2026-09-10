#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///

"""CSCV probability-of-backtest-overfitting and deflated-Sharpe calculations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import e, log, sqrt
from statistics import NormalDist

import numpy as np
import numpy.typing as npt


FloatMatrix = npt.NDArray[np.float64]
FloatVector = npt.NDArray[np.float64]


class SelectionBiasError(ValueError):
    """The supplied returns cannot support a selection-bias statistic."""


@dataclass(frozen=True, slots=True)
class PboResult:
    pbo: float
    splits: int
    candidates: int
    median_logit: float


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    benchmark_sharpe: float
    probability: float
    trials: int
    observations: int


def _validate_matrix(returns: FloatMatrix, bins: int) -> None:
    if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < bins * 2:
        raise SelectionBiasError("CSCV requires two candidates and at least two observations per bin")
    if bins < 2 or bins % 2 or not np.isfinite(returns).all():
        raise SelectionBiasError("CSCV requires an even bin count and finite returns")


def _sharpe_from_moments(sums: FloatVector, squares: FloatVector, count: int) -> FloatVector:
    if count < 2:
        raise SelectionBiasError("Sharpe requires at least two observations")
    mean = sums / count
    variance = np.maximum((squares - sums * sums / count) / (count - 1), 0.0)
    standard_deviation = np.sqrt(variance)
    result = np.zeros_like(mean)
    positive = (standard_deviation == 0.0) & (mean > 0.0)
    negative = (standard_deviation == 0.0) & (mean < 0.0)
    finite = standard_deviation > 0.0
    result[positive] = np.inf
    result[negative] = -np.inf
    result[finite] = mean[finite] / standard_deviation[finite]
    return result


def bin_moments(returns: FloatMatrix, *, bins: int = 8) -> tuple[FloatMatrix, FloatMatrix, tuple[int, ...]]:
    """Compress candidate return paths into sufficient CSCV bin moments."""
    values = np.asarray(returns, dtype=float)
    _validate_matrix(values, bins)
    segments = tuple(np.asarray(segment, dtype=float) for segment in np.array_split(values, bins, axis=1))
    sums = np.stack([segment.sum(axis=1) for segment in segments], axis=1)
    squares = np.stack([(segment * segment).sum(axis=1) for segment in segments], axis=1)
    counts = tuple(segment.shape[1] for segment in segments)
    return sums, squares, counts


def cscv_pbo_from_moments(sums: FloatMatrix, squares: FloatMatrix, counts: tuple[int, ...]) -> PboResult:
    """Estimate PBO from sufficient return moments without retaining every path."""
    if sums.ndim != 2 or squares.shape != sums.shape or sums.shape[0] < 2 or sums.shape[1] < 2 or sums.shape[1] % 2:
        raise SelectionBiasError("CSCV moments must have two candidates and an even bin count")
    if len(counts) != sums.shape[1] or any(count < 1 for count in counts) or not np.isfinite(sums).all() or not np.isfinite(squares).all():
        raise SelectionBiasError("CSCV moments are invalid")
    bins = sums.shape[1]
    logits: list[float] = []
    all_bins = tuple(range(bins))
    for in_sample in combinations(all_bins, bins // 2):
        out_sample = tuple(index for index in all_bins if index not in in_sample)
        in_count = sum(counts[index] for index in in_sample)
        out_count = sum(counts[index] for index in out_sample)
        in_sharpe = _sharpe_from_moments(sums[:, in_sample].sum(axis=1), squares[:, in_sample].sum(axis=1), in_count)
        out_sharpe = _sharpe_from_moments(sums[:, out_sample].sum(axis=1), squares[:, out_sample].sum(axis=1), out_count)
        selected = int(np.argmax(in_sharpe))
        rank = int(np.count_nonzero(out_sharpe <= out_sharpe[selected]))
        relative_rank = rank / (sums.shape[0] + 1.0)
        logits.append(log(relative_rank / (1.0 - relative_rank)))
    return PboResult(
        pbo=float(np.mean(np.asarray(logits) <= 0.0)),
        splits=len(logits),
        candidates=sums.shape[0],
        median_logit=float(np.median(np.asarray(logits))),
    )


def cscv_pbo(returns: FloatMatrix, *, bins: int = 8) -> PboResult:
    """Estimate PBO by selecting the in-sample Sharpe winner across CSCV splits."""
    return cscv_pbo_from_moments(*bin_moments(returns, bins=bins))


def deflated_sharpe(returns: FloatVector, *, trials: int) -> DeflatedSharpeResult:
    """Calculate a trial-count-adjusted probability for a periodic Sharpe ratio."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 3 or trials < 1 or not np.isfinite(values).all():
        raise SelectionBiasError("DSR requires finite one-dimensional returns and a positive trial count")
    mean = float(values.mean())
    centered = values - mean
    standard_deviation = float(np.sqrt(np.mean(centered * centered)))
    if standard_deviation == 0.0:
        raise SelectionBiasError("DSR requires non-constant returns")
    observed = mean / standard_deviation
    skew = float(np.mean(centered**3) / standard_deviation**3)
    kurtosis = float(np.mean(centered**4) / standard_deviation**4)
    variance = (1.0 - skew * observed + (kurtosis - 1.0) * observed * observed / 4.0) / (values.size - 1)
    standard_error = sqrt(max(variance, 1e-18))
    normal = NormalDist()
    if trials == 1:
        expected_max_z = 0.0
    else:
        expected_max_z = (1.0 - 0.5772156649015329) * normal.inv_cdf(1.0 - 1.0 / trials)
        expected_max_z += 0.5772156649015329 * normal.inv_cdf(1.0 - 1.0 / (trials * e))
    benchmark = standard_error * expected_max_z
    probability = float(normal.cdf((observed - benchmark) / standard_error))
    return DeflatedSharpeResult(observed, benchmark, probability, trials, values.size)
