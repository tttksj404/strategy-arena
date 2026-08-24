from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from .bitget_market import BitgetContract, TextFetcher
from .long_forecast import Candle
from .market_timeframes import CandleGranularity
from .trend_continuation_contract import Direction
from .trend_continuation_evaluation_types import (
    EvaluationRow,
    FeePolicySnapshot,
    FundingProof,
    LabelRequest,
    ObservationSource,
)
from .trend_continuation_features import (
    FeatureInputError,
    _invalidation,
    build_features,
    fetch_window_history,
)
from .trend_continuation_labeling import evaluate_label_window


def evaluate_candle_series(
    candles: Sequence[Candle],
    *,
    trigger_interval: CandleGranularity,
    horizon_bars: int,
    direction: Direction,
    fee_pct: float | None = None,
    fee_policy: FeePolicySnapshot | None = None,
    funding_proof: FundingProof | None = None,
    min_lookback_bars: int = 60,
) -> list[EvaluationRow]:
    """Evaluate causally; legacy fee_pct is accepted but never trusted as proof."""
    del fee_pct
    ordered = tuple(candles)
    if len(ordered) < min_lookback_bars:
        return []
    proof = funding_proof or FundingProof(complete=False, events=())
    settled_as_of = max(candle.opened_at + trigger_interval.interval for candle in ordered)
    rows: list[EvaluationRow] = []
    for index in range(min_lookback_bars - 1, len(ordered)):
        prefix = ordered[: index + 1]
        evidence_time = prefix[-1].opened_at + trigger_interval.interval
        try:
            features = build_features(prefix, evidence_time, trigger_interval)
        except FeatureInputError:
            continue
        if features.trend_direction() is not direction:
            continue
        invalidation_price = _invalidation(features.candles, features.atr14, direction)
        if invalidation_price is None:
            continue
        request = LabelRequest(
            evidence_time=evidence_time,
            trigger_interval=trigger_interval,
            horizon_bars=horizon_bars,
            direction=direction,
            reference_price=prefix[-1].close,
            invalidation_price=invalidation_price,
            source=ObservationSource.HISTORICAL_EXPLORATORY,
            fee_policy=fee_policy,
            funding=proof,
        )
        rows.append(
            evaluate_label_window(
                request,
                ordered[index + 1 :],
                settled_as_of=settled_as_of,
                features=features,
            )
        )
    return rows


def build_evaluation_row(
    *,
    contract: BitgetContract,
    trigger_interval: CandleGranularity,
    evidence_time: datetime,
    horizon_bars: int,
    direction: Direction,
    fetch_text: TextFetcher,
    fee_pct: float | None = None,
    fee_policy: FeePolicySnapshot | None = None,
    funding_proof: FundingProof | None = None,
    lookback_bars: int,
) -> EvaluationRow | None:
    """Compatibility adapter; an unsourced fee_pct never satisfies provenance."""
    del fee_pct
    trigger_window = fetch_window_history(
        contract,
        trigger_interval,
        evidence_time - trigger_interval.interval * lookback_bars,
        evidence_time,
        fetch_text,
    )
    if len(trigger_window) < lookback_bars:
        return None
    try:
        features = build_features(trigger_window, evidence_time, trigger_interval)
    except FeatureInputError:
        return None
    if features.trend_direction() is not direction:
        return None
    invalidation_price = _invalidation(features.candles, features.atr14, direction)
    if invalidation_price is None:
        return None
    label_end = evidence_time + trigger_interval.interval * horizon_bars
    future = fetch_window_history(contract, trigger_interval, evidence_time, label_end, fetch_text)
    request = LabelRequest(
        evidence_time=evidence_time,
        trigger_interval=trigger_interval,
        horizon_bars=horizon_bars,
        direction=direction,
        reference_price=trigger_window[-1].close,
        invalidation_price=invalidation_price,
        source=ObservationSource.HISTORICAL_EXPLORATORY,
        fee_policy=fee_policy,
        funding=funding_proof or FundingProof(complete=False, events=()),
    )
    return evaluate_label_window(request, future, settled_as_of=label_end, features=features)
