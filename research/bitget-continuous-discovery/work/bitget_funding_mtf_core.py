"""Pure funding-aware signal and paper-path calculations for the public Bitget study."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

import numpy as np

from alt_search_engine import evaluate_path
from alt_search_models import FloatMatrix, FloatVector, MarketData, RiskOverlay


FUNDING_WINDOWS_DAYS: Final = (1, 3, 7)
FUNDING_APR_FLOORS: Final = (0.0, 0.05, 0.15)
PAIR_COUNTS: Final = (1, 2)
REBALANCE_BARS: Final = (1, 3, 6)
LEVERAGE_CAPS: Final = (1.0, 2.0, 3.0, 5.0)


class FundingFamily(StrEnum):
    """Closed set of predeclared funding-aware portfolio constructions."""

    CONTRARIAN = "funding_contrarian"
    MOMENTUM = "funding_momentum"
    TREND_FILTERED = "trend_filtered_contrarian"


@dataclass(frozen=True, slots=True)
class FundingCandidate:
    """One signal configuration fixed before public data collection."""

    family: FundingFamily
    funding_window_days: int
    minimum_apr: float
    pair_count: int
    rebalance_bars: int
    trend_window_days: int

    @property
    def identifier(self) -> str:
        """Return a deterministic report identifier."""

        return (
            f"{self.family.value}|fw={self.funding_window_days}|apr={self.minimum_apr:g}"
            f"|pairs={self.pair_count}|reb={self.rebalance_bars}|trend={self.trend_window_days}"
        )


@dataclass(frozen=True, slots=True)
class SignalInputs:
    """Aligned prices and already-settled funding rates for a closed-bar grid."""

    close_price: FloatMatrix
    funding_rate: FloatMatrix
    interval_hours: float


@dataclass(frozen=True, slots=True)
class FundingSettlement:
    """Signed funding cash flows applied to an evaluated price path."""

    positions: FloatMatrix
    funding_rate: FloatMatrix
    leverage: FloatVector
    first_liquidation_index: int


@dataclass(frozen=True, slots=True)
class FundingPathInputs:
    """The typed inputs required for one funding-aware paper path."""

    market: MarketData
    funding_rate: FloatMatrix
    positions: FloatMatrix
    risk: RiskOverlay


@dataclass(frozen=True, slots=True)
class FundingPath:
    """Price, funding, cost, and surrogate-liquidation path evidence."""

    returns: FloatVector
    stress_returns: FloatVector
    first_liquidation_index: int
    max_realized_leverage: float


@dataclass(frozen=True, slots=True)
class PaperMetrics:
    """Chronological return statistics for one paper slice."""

    net_return: float
    max_drawdown: float
    sharpe: float
    positive_blocks: int
    observations: int


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Train/validation/test boundaries without test-time selection input."""

    train_end: int
    validation_end: int


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """A validation-only candidate score eligible for selection."""

    candidate: FundingCandidate
    max_leverage: float
    score: float
    passes: bool


def candidate_grid() -> tuple[FundingCandidate, ...]:
    """Enumerate the fixed funding signal grid before inspecting validation."""

    return tuple(
        FundingCandidate(family, window, floor, pair_count, rebalance, 1)
        for family in FundingFamily
        for window in FUNDING_WINDOWS_DAYS
        for floor in FUNDING_APR_FLOORS
        for pair_count in PAIR_COUNTS
        for rebalance in REBALANCE_BARS
    )


def _bars_per_day(interval_hours: float) -> int:
    """Convert a supported closed-bar duration to an integer daily bar count."""

    return max(1, int(round(24.0 / interval_hours)))


def _rolling_sum(values: FloatMatrix, window: int) -> FloatMatrix:
    """Return full-window rolling sums with unavailable leading rows as NaN."""

    result = np.full_like(values, np.nan)
    if len(values) < window:
        return result
    prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    result[window - 1 :] = prefix[window:] - prefix[: -window]
    return result


def _trend(close_price: FloatMatrix, window: int) -> FloatMatrix:
    """Return closed-bar price trend that can be observed at each row end."""

    result = np.full_like(close_price, np.nan)
    if len(close_price) > window:
        result[window:] = close_price[window:] / close_price[:-window] - 1.0
    return result


def _funding_score(inputs: SignalInputs, candidate: FundingCandidate) -> FloatMatrix:
    """Annualize settled funding inside the declared trailing observation window."""

    window = candidate.funding_window_days * _bars_per_day(inputs.interval_hours)
    return _rolling_sum(inputs.funding_rate, window) * 365.0 / candidate.funding_window_days


def _select_legs(score: FloatVector, trend: FloatVector, candidate: FundingCandidate) -> tuple[FloatVector, FloatVector]:
    """Choose equal-sized long and short legs from a single prior-bar score row."""

    positive = np.flatnonzero(np.isfinite(score) & (score > candidate.minimum_apr))
    negative = np.flatnonzero(np.isfinite(score) & (score < -candidate.minimum_apr))
    match candidate.family:
        case FundingFamily.CONTRARIAN:
            longs, shorts = negative, positive
        case FundingFamily.MOMENTUM:
            longs, shorts = positive, negative
        case FundingFamily.TREND_FILTERED:
            longs = negative[np.isfinite(trend[negative]) & (trend[negative] > 0.0)]
            shorts = positive[np.isfinite(trend[positive]) & (trend[positive] < 0.0)]
        case unreachable:
            assert_never(unreachable)
    return longs, shorts


def build_positions(inputs: SignalInputs, candidate: FundingCandidate) -> FloatMatrix:
    """Build dollar-neutral positions from strictly prior settled funding observations."""

    scores = _funding_score(inputs, candidate)
    trend = _trend(inputs.close_price, candidate.trend_window_days * _bars_per_day(inputs.interval_hours))
    positions = np.zeros_like(inputs.close_price)
    for row in range(1, len(positions)):
        if row % candidate.rebalance_bars != 0:
            positions[row] = positions[row - 1]
            continue
        longs, shorts = _select_legs(scores[row - 1], trend[row - 1], candidate)
        if len(longs) < candidate.pair_count or len(shorts) < candidate.pair_count:
            continue
        selected_longs = longs[np.argsort(scores[row - 1, longs])[: candidate.pair_count]]
        selected_shorts = shorts[np.argsort(scores[row - 1, shorts])[-candidate.pair_count :]]
        weight = 1.0 / (2.0 * candidate.pair_count)
        positions[row, selected_longs] = weight
        positions[row, selected_shorts] = -weight
    return positions


def funding_component(settlement: FundingSettlement) -> FloatVector:
    """Credit short-positive and long-negative funding only while the path survives."""

    component = -np.sum(settlement.positions * settlement.funding_rate * settlement.leverage[:, None], axis=1)
    if settlement.first_liquidation_index >= 0:
        component[settlement.first_liquidation_index :] = 0.0
    return component


def evaluate_funding_path(inputs: FundingPathInputs) -> FundingPath:
    """Add settled funding to the shared cost and liquidation-aware price path."""

    price_path = evaluate_path(inputs.market, inputs.positions, inputs.risk)
    settlement = FundingSettlement(
        inputs.positions,
        inputs.funding_rate,
        price_path.leverage,
        price_path.first_liquidation_index,
    )
    component = funding_component(settlement)
    return FundingPath(
        price_path.returns + component,
        price_path.stress_returns + component,
        price_path.first_liquidation_index,
        float(price_path.leverage.max(initial=0.0)),
    )


def metrics(returns: FloatVector, bars_per_day: int) -> PaperMetrics:
    """Measure a return path without looking beyond the requested chronological slice."""

    if len(returns) == 0:
        return PaperMetrics(0.0, 0.0, 0.0, 0, 0)
    equity = np.cumprod(1.0 + returns)
    drawdown = 1.0 - equity / np.maximum.accumulate(equity)
    deviation = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / deviation * np.sqrt(365.0 * bars_per_day)) if deviation > 0.0 else 0.0
    blocks = np.array_split(returns, 4)
    positive = sum(bool(np.prod(1.0 + block) > 1.0) for block in blocks if len(block))
    return PaperMetrics(float(equity[-1] - 1.0), float(drawdown.max()), sharpe, positive, len(returns))


def chronological_split(observations: int) -> ChronologicalSplit:
    """Split an aligned series into fixed 50/25/25 chronological segments."""

    train_end = observations // 2
    return ChronologicalSplit(train_end, train_end + (observations - train_end) // 2)


def validation_score(result: PaperMetrics, stress: PaperMetrics, liquidated: bool) -> tuple[float, bool]:
    """Score only validation results that first satisfy preservation gates."""

    passes = (
        not liquidated
        and result.net_return > 0.0
        and stress.net_return > 0.0
        and result.max_drawdown <= 0.25
        and result.positive_blocks >= 3
    )
    return (result.net_return / max(result.max_drawdown, 0.01), True) if passes else (float("-inf"), False)


def select_validation_winners(records: tuple[ValidationRecord, ...]) -> tuple[ValidationRecord, ...]:
    """Select one candidate per family and leverage using validation evidence alone."""

    groups: dict[tuple[FundingFamily, float], list[ValidationRecord]] = {}
    for record in records:
        groups.setdefault((record.candidate.family, record.max_leverage), []).append(record)
    return tuple(
        max(eligible, key=lambda record: record.score)
        for _, group in sorted(groups.items(), key=lambda item: (item[0][0].value, item[0][1]))
        if (eligible := [record for record in group if record.passes])
    )


def short_horizon_pass(result: PaperMetrics, stress: PaperMetrics, liquidated: bool) -> bool:
    """Apply the declared short-horizon paper gate without authorizing promotion."""

    return (
        not liquidated
        and result.net_return > 0.0
        and stress.net_return > 0.0
        and result.max_drawdown <= 0.25
        and result.positive_blocks >= 3
    )
