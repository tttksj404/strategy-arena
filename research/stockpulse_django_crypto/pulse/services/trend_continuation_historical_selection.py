from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final

from django.db import IntegrityError

from ..trend_continuation_models import TrendContinuationSelectionManifest
from .market_timeframes import CandleGranularity
from .trend_continuation_contract import CandidateIdentity, Direction
from .trend_continuation_evaluation import (
    CandidateDevelopment,
    CandidateDevelopmentInputError,
    EvaluationRow,
    HistoricalSelection,
    build_candidate_development,
    evaluate_candle_series,
)
from .trend_continuation_features import (
    BASELINE_RULE_NAME,
    DEFAULT_HORIZONS,
    FEATURE_SCHEMA_VERSION,
    TECHNICAL_SCORE_RULE_NAME,
)
from .trend_continuation_fixture import HistoricalDataset, HistoricalResearchRequest
from .trend_continuation_manifest import (
    TrendContinuationSelectionSnapshot,
    freeze_selection,
)


STRATEGY_VERSION: Final = "trend-continuation-strategy.v1"


def freeze_selection_idempotently(
    snapshot: TrendContinuationSelectionSnapshot,
    *,
    frozen_at: datetime,
) -> TrendContinuationSelectionManifest:
    existing = TrendContinuationSelectionManifest.objects.filter(digest=snapshot.digest).first()
    if existing is not None:
        return existing
    try:
        return freeze_selection(snapshot, frozen_at=frozen_at)
    except IntegrityError:
        return TrendContinuationSelectionManifest.objects.get(digest=snapshot.digest)


def build_historical_developments(
    request: HistoricalResearchRequest,
    dataset: HistoricalDataset,
    *,
    eligible: bool,
) -> tuple[tuple[CandidateDevelopment, ...], int]:
    if not eligible:
        return (), 0
    trigger = (
        CandleGranularity.FIVE_MINUTES
        if request.profile == "standard_15m_5m"
        else CandleGranularity.ONE_MINUTE
    )
    trigger_series = next(item for item in dataset.series if item.granularity is trigger)
    developments: list[CandidateDevelopment] = []
    unresolved = 0
    for direction in (Direction.UP, Direction.DOWN):
        for horizon in DEFAULT_HORIZONS[request.profile]:
            rows = evaluate_candle_series(
                trigger_series.candles,
                trigger_interval=trigger,
                horizon_bars=horizon,
                direction=direction,
                fee_policy=dataset.fee_policy,
                funding_proof=dataset.funding,
            )
            accepted = _non_overlapping(rows)
            unresolved += sum(not row.resolved for row in accepted)
            for rule_name in (BASELINE_RULE_NAME, TECHNICAL_SCORE_RULE_NAME):
                identity = CandidateIdentity(
                    request.symbol,
                    request.profile,
                    direction,
                    rule_name,
                    horizon,
                    STRATEGY_VERSION,
                    FEATURE_SCHEMA_VERSION,
                )
                try:
                    developments.append(build_candidate_development(identity, accepted))
                except CandidateDevelopmentInputError:
                    continue
    return tuple(developments), unresolved


def fee_covers_history(
    dataset: HistoricalDataset,
    request: HistoricalResearchRequest,
    frozen_at: datetime,
) -> bool:
    fee_policy = dataset.fee_policy
    return (
        fee_policy is not None
        and fee_policy.valid_at(request.starts_at)
        and fee_policy.valid_at(request.ends_at - timedelta(microseconds=1))
        and fee_policy.valid_at(frozen_at)
    )


def freeze_historical_selection(
    selection: HistoricalSelection | None,
    dataset: HistoricalDataset,
    frozen_at: datetime,
) -> TrendContinuationSelectionManifest | None:
    fee_policy = dataset.fee_policy
    if selection is None or fee_policy is None or not fee_policy.valid_at(frozen_at):
        return None
    identity = selection.identity
    snapshot = TrendContinuationSelectionSnapshot(
        payload={
            "schema": "trend-continuation-selection.v1",
            "exploratory_only": True,
            "candidate": {
                "contract": identity.contract,
                "profile": identity.profile,
                "trend_direction": identity.trend_direction.value,
                "rule_name": identity.rule_name,
                "candidate_horizon_bars": identity.candidate_horizon_bars,
                "strategy_version": identity.strategy_version,
                "feature_schema_version": identity.feature_schema_version,
            },
            "fee_policy": fee_policy.to_payload(),
            "historical_metrics": {
                "resolved_count": selection.metrics.resolved_count,
                "wilson_lower_95": selection.metrics.wilson_lower_95,
                "mean_return_lower_95": selection.metrics.mean_return_lower_95,
            },
        }
    )
    return freeze_selection_idempotently(snapshot, frozen_at=frozen_at)


def _non_overlapping(rows: Sequence[EvaluationRow]) -> tuple[EvaluationRow, ...]:
    accepted: list[EvaluationRow] = []
    for row in sorted(rows, key=lambda item: item.evidence_time):
        if not accepted or accepted[-1].label_end_time < row.evidence_time:
            accepted.append(row)
    return tuple(accepted)
