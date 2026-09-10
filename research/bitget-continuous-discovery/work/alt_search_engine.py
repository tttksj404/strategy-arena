"""Cost, volatility, and conservative liquidation calculations for paper paths."""
# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from alt_search_models import (
    BASE_FEE,
    BASE_SLIPPAGE,
    LIQUIDATION_RESERVE,
    STRESS_SLIPPAGE,
    Candidate,
    FloatMatrix,
    FloatVector,
    MarketData,
    PathResult,
    RiskOverlay,
    SegmentMetrics,
)


def _lagged_rolling_std(values: FloatVector, window: int) -> FloatVector:
    """Return rolling standard deviations that exclude each evaluated return."""

    result = np.full_like(values, np.nan)
    if window <= 1:
        return result
    prefix = np.concatenate((np.array([0.0]), np.cumsum(values)))
    square_prefix = np.concatenate((np.array([0.0]), np.cumsum(values * values)))
    totals = prefix[window:-1] - prefix[: -window - 1]
    square_totals = square_prefix[window:-1] - square_prefix[: -window - 1]
    variance = np.maximum(square_totals / window - (totals / window) ** 2, 0.0)
    result[window:] = np.sqrt(variance)
    return result


def _metrics(returns: FloatVector) -> SegmentMetrics:
    """Measure an already cost-adjusted chronological return slice."""

    if len(returns) == 0:
        return SegmentMetrics(0.0, 0.0, 0.0, 0, 0)
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / peak
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(365.0)) if std > 0.0 else 0.0
    blocks = np.array_split(returns, 4)
    positive_blocks = sum(bool(np.prod(1.0 + block) > 1.0) for block in blocks if len(block))
    return SegmentMetrics(float(equity[-1] - 1.0), float(drawdown.max()), sharpe, positive_blocks, len(returns))


def segment_metrics(path: PathResult, start: int, end: int) -> tuple[SegmentMetrics, SegmentMetrics, bool]:
    """Return base/stress metrics and whether a liquidation occurs in the slice."""

    hits = path.liquidation_indices or ((path.first_liquidation_index,) if path.first_liquidation_index >= 0 else ())
    liquidated = any(start <= index < end for index in hits)
    return _metrics(path.returns[start:end]), _metrics(path.stress_returns[start:end]), liquidated


def _evaluate_path(data: MarketData, positions: FloatMatrix, risk: RiskOverlay, *, stop_at_first_liquidation: bool) -> PathResult:
    """Simulate next-bar PnL and record every intrabar liquidation proxy hit."""

    asset_returns = np.zeros_like(data.close_price)
    asset_returns[1:] = data.close_price[1:] / data.close_price[:-1] - 1.0
    raw_unscaled = (positions * asset_returns).sum(axis=1)
    volatility = _lagged_rolling_std(raw_unscaled, risk.volatility_window)
    active = np.abs(positions).sum(axis=1) > 0.0
    fallback = risk.target_daily_vol / risk.max_leverage
    safe_volatility = np.where(np.isfinite(volatility), np.maximum(volatility, fallback), fallback)
    leverage = np.zeros(len(positions))
    leverage[active] = np.minimum(risk.max_leverage, risk.target_daily_vol / safe_volatility[active])
    scaled = positions * leverage[:, None]
    turnover = np.abs(scaled).sum(axis=1)
    turnover[1:] = np.abs(scaled[1:] - scaled[:-1]).sum(axis=1)
    net = raw_unscaled * leverage - turnover * (BASE_FEE + BASE_SLIPPAGE)
    stress = raw_unscaled * leverage - turnover * (BASE_FEE + STRESS_SLIPPAGE)

    long_adverse = np.zeros_like(data.close_price)
    short_adverse = np.zeros_like(data.close_price)
    long_adverse[1:] = (data.close_price[:-1] - data.low_price[1:]) / data.close_price[:-1]
    short_adverse[1:] = (data.high_price[1:] - data.close_price[:-1]) / data.close_price[:-1]
    threshold = np.full_like(leverage, np.inf)
    leveraged = leverage > 0.0
    threshold[leveraged] = 1.0 / leverage[leveraged] - LIQUIDATION_RESERVE
    touched = ((scaled > 0.0) & (long_adverse >= threshold[:, None])) | ((scaled < 0.0) & (short_adverse >= threshold[:, None]))
    hits = np.flatnonzero(touched.any(axis=1))
    if len(hits) and stop_at_first_liquidation:
        first = int(hits[0])
        net[first], stress[first] = -1.0, -1.0
        net[first + 1 :], stress[first + 1 :] = 0.0, 0.0
        return PathResult(net, stress, leverage, len(hits), first, tuple(int(index) for index in hits))
    return PathResult(net, stress, leverage, len(hits), int(hits[0]) if len(hits) else -1, tuple(int(index) for index in hits))


def evaluate_path(data: MarketData, positions: FloatMatrix, risk: RiskOverlay) -> PathResult:
    """Simulate executable next-bar PnL, forcing a full loss on the first liquidation proxy hit."""

    return _evaluate_path(data, positions, risk, stop_at_first_liquidation=True)


def evaluate_counterfactual_path(data: MarketData, positions: FloatMatrix, risk: RiskOverlay) -> PathResult:
    """Simulate a research-only no-stop path while retaining every liquidation proxy hit."""

    return _evaluate_path(data, positions, risk, stop_at_first_liquidation=False)


def validation_score(metrics: SegmentMetrics, stress: SegmentMetrics, liquidated: bool) -> float:
    """Score validation return only after hard capital-preservation gates pass."""

    if liquidated or metrics.net_return <= 0.0 or stress.net_return <= 0.0:
        return float("-inf")
    if metrics.max_drawdown > 0.25 or metrics.positive_blocks < 3:
        return float("-inf")
    return metrics.net_return / max(metrics.max_drawdown, 0.01)


def test_passes(metrics: SegmentMetrics, stress: SegmentMetrics, liquidated: bool) -> bool:
    """Apply the predeclared locked-test safety gate."""

    return (
        not liquidated
        and metrics.observations >= 90
        and metrics.net_return > 0.0
        and stress.net_return > 0.0
        and metrics.max_drawdown <= 0.25
        and metrics.positive_blocks >= 3
    )


def select_validation_winners(
    candidates: tuple[Candidate, ...],
    validation_scores: Mapping[str, float],
    validation_passes: Mapping[str, bool],
) -> tuple[Candidate, ...]:
    """Select one validation-only winner per family without any locked-test input."""

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.family.value].append(candidate)
    winners: list[Candidate] = []
    for family in sorted(grouped):
        eligible = [candidate for candidate in grouped[family] if validation_passes[candidate.identifier]]
        if eligible:
            winners.append(max(eligible, key=lambda candidate: validation_scores[candidate.identifier]))
    return tuple(winners)
