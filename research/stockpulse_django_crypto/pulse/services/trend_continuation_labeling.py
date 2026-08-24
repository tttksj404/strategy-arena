from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from math import isfinite
from typing import assert_never

from .long_forecast import Candle
from .trend_continuation_contract import Direction
from .trend_continuation_evaluation_types import (
    HISTORICAL_EXPLORATORY_COST_FLOOR_PCT,
    EvaluationRow,
    FeePolicySnapshot,
    ForwardQuoteSnapshot,
    FundingProof,
    LabelRequest,
    ObservationSource,
)
from .trend_continuation_features import TrendContinuationFeatures


def _directional_funding_cost_pct(
    direction: Direction,
    evidence_time: datetime,
    label_end_time: datetime,
    proof: FundingProof,
) -> float | None:
    if not proof.covers(evidence_time, label_end_time):
        return None
    selected = tuple(event for event in proof.events if evidence_time < event.funding_time <= label_end_time)
    times = tuple(event.funding_time for event in selected)
    if len(times) != len(frozenset(times)):
        return None
    if any(event.funding_time.tzinfo is None or not isfinite(event.funding_rate) for event in selected):
        return None
    match direction:
        case Direction.UP:
            return sum(max(event.funding_rate, 0.0) * 100 for event in selected)
        case Direction.DOWN:
            return sum(max(-event.funding_rate, 0.0) * 100 for event in selected)
        case Direction.NEUTRAL:
            return None
        case unreachable:
            assert_never(unreachable)


def compute_cost_pct(
    *,
    source: ObservationSource,
    direction: Direction,
    evidence_time: datetime,
    label_end_time: datetime,
    fee_policy: FeePolicySnapshot | None,
    funding: FundingProof,
    quote: ForwardQuoteSnapshot | None = None,
) -> float | None:
    if (
        evidence_time.tzinfo is None
        or label_end_time.tzinfo is None
        or evidence_time >= label_end_time
        or fee_policy is None
        or not fee_policy.valid_at(evidence_time)
    ):
        return None
    funding_cost = _directional_funding_cost_pct(direction, evidence_time, label_end_time, funding)
    if funding_cost is None:
        return None
    match source:
        case ObservationSource.HISTORICAL_EXPLORATORY:
            if quote is not None:
                return None
            return HISTORICAL_EXPLORATORY_COST_FLOOR_PCT + funding_cost
        case ObservationSource.FORWARD:
            if quote is None or not quote.valid_for(evidence_time):
                return None
            return 2 * fee_policy.taker_fee_rate * 100 + max(0.05, quote.spread_pct) + funding_cost
        case unreachable:
            assert_never(unreachable)


def _unresolved(request: LabelRequest, rejection_code: str) -> EvaluationRow:
    return EvaluationRow(
        evidence_time=request.evidence_time,
        label_end_time=request.label_end_time,
        reference_price=request.reference_price,
        invalidation_price=request.invalidation_price,
        outcome_success=None,
        net_return_pct=None,
        source=request.source,
        rejection_code=rejection_code,
    )


def evaluate_label_window(
    request: LabelRequest,
    future_candles: Sequence[Candle],
    *,
    settled_as_of: datetime,
    features: TrendContinuationFeatures | None = None,
) -> EvaluationRow:
    if (
        request.horizon_bars <= 0
        or request.evidence_time.tzinfo is None
        or settled_as_of.tzinfo is None
        or settled_as_of < request.label_end_time
        or request.reference_price <= 0
        or request.invalidation_price <= 0
        or not isfinite(request.reference_price)
        or not isfinite(request.invalidation_price)
        or request.direction is Direction.NEUTRAL
        or (request.direction is Direction.UP and request.invalidation_price >= request.reference_price)
        or (request.direction is Direction.DOWN and request.invalidation_price <= request.reference_price)
    ):
        return _unresolved(request, "invalid_label_request")
    interval = request.trigger_interval.interval
    if any(candle.opened_at.tzinfo is None for candle in future_candles):
        return _unresolved(request, "naive_label_window")
    selected = tuple(
        candle
        for candle in future_candles
        if request.evidence_time < candle.opened_at + interval <= request.label_end_time
    )
    expected_closes = tuple(request.evidence_time + interval * index for index in range(1, request.horizon_bars + 1))
    actual_closes = tuple(candle.opened_at + interval for candle in selected)
    if actual_closes != expected_closes or len(frozenset(actual_closes)) != len(actual_closes):
        return _unresolved(request, "incomplete_label_window")
    if any(
        not all(isfinite(value) for value in (candle.high, candle.low, candle.close))
        or not candle.low <= candle.close <= candle.high
        for candle in selected
    ):
        return _unresolved(request, "nonfinite_label_window")
    cost_pct = compute_cost_pct(
        source=request.source,
        direction=request.direction,
        evidence_time=request.evidence_time,
        label_end_time=request.label_end_time,
        fee_policy=request.fee_policy,
        funding=request.funding,
        quote=request.quote,
    )
    if cost_pct is None:
        return _unresolved(request, "incomplete_cost_provenance")
    terminal_close = selected[-1].close
    match request.direction:
        case Direction.UP:
            invalidated = any(candle.low <= request.invalidation_price for candle in selected)
            gross_return_pct = (terminal_close / request.reference_price - 1) * 100
        case Direction.DOWN:
            invalidated = any(candle.high >= request.invalidation_price for candle in selected)
            gross_return_pct = -((terminal_close / request.reference_price) - 1) * 100
        case Direction.NEUTRAL:
            return _unresolved(request, "neutral_direction")
        case unreachable:
            assert_never(unreachable)
    net_return_pct = round(gross_return_pct - cost_pct, 12)
    if not isfinite(net_return_pct):
        return _unresolved(request, "nonfinite_return")
    return EvaluationRow(
        evidence_time=request.evidence_time,
        label_end_time=request.label_end_time,
        reference_price=request.reference_price,
        invalidation_price=request.invalidation_price,
        outcome_success=not invalidated and net_return_pct > 0,
        net_return_pct=net_return_pct,
        terminal_close=terminal_close,
        cost_pct=cost_pct,
        source=request.source,
        features=features,
    )
