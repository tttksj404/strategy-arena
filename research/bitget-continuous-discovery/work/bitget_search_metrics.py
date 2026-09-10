"""Shared interval-aware metrics for public-only Bitget paper searches."""

from __future__ import annotations

from math import sqrt

import numpy as np

from alt_search_models import PathResult, SegmentMetrics
from selection_bias_stats import SelectionBiasError


def annualized_metrics(returns: np.ndarray, *, bars_per_year: float) -> SegmentMetrics:
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        return SegmentMetrics(0.0, 0.0, 0.0, 0, 0)
    equity = np.cumprod(1.0 + values)
    drawdown = 1.0 - equity / np.maximum.accumulate(equity)
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
    sharpe = float(values.mean() / standard_deviation * sqrt(bars_per_year)) if standard_deviation > 0.0 else 0.0
    blocks = np.array_split(values, 4)
    positive_blocks = sum(bool(np.prod(1.0 + block) > 1.0) for block in blocks if block.size)
    return SegmentMetrics(float(equity[-1] - 1.0), float(drawdown.max()), sharpe, positive_blocks, values.size)


def fixed_notional_metrics(returns: np.ndarray, *, bars_per_year: float) -> SegmentMetrics:
    """Measure a no-compounding, fixed-notional counterfactual PnL stream."""

    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        return SegmentMetrics(0.0, 0.0, 0.0, 0, 0)
    pnl = np.cumsum(values)
    peak = np.maximum.accumulate(np.concatenate((np.asarray([0.0]), pnl)))[:-1]
    drawdown = np.maximum(peak - pnl, 0.0)
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
    sharpe = float(values.mean() / standard_deviation * sqrt(bars_per_year)) if standard_deviation > 0.0 else 0.0
    blocks = np.array_split(values, 4)
    positive_blocks = sum(bool(block.sum() > 0.0) for block in blocks if block.size)
    return SegmentMetrics(float(values.sum()), float(drawdown.max()), sharpe, positive_blocks, values.size)


def segment_metrics(path: PathResult, start: int, end: int, *, bars_per_year: float) -> tuple[SegmentMetrics, SegmentMetrics, bool]:
    liquidated = start <= path.first_liquidation_index < end
    return (
        annualized_metrics(path.returns[start:end], bars_per_year=bars_per_year),
        annualized_metrics(path.stress_returns[start:end], bars_per_year=bars_per_year),
        liquidated,
    )


def development_moments(returns: np.ndarray, *, development_end: int, bins: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    values = np.asarray(returns[:development_end], dtype=float)
    if development_end <= 0 or development_end >= len(returns) or bins < 2 or bins % 2 or values.size < bins * 2:
        raise SelectionBiasError("development period cannot support CSCV moments")
    if not np.isfinite(values).all():
        raise SelectionBiasError("development returns must be finite for CSCV moments")
    segments = tuple(np.asarray(segment, dtype=float) for segment in np.array_split(values, bins))
    return (
        np.asarray([segment.sum() for segment in segments], dtype=float),
        np.asarray([(segment * segment).sum() for segment in segments], dtype=float),
        tuple(segment.size for segment in segments),
    )


def metric_payload(metrics: SegmentMetrics) -> dict[str, float | int]:
    return {
        "net_return": metrics.net_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe_annualized": metrics.sharpe,
        "positive_blocks": metrics.positive_blocks,
        "observations": metrics.observations,
    }
