from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ..trend_continuation_models import (
    SOURCE_FORWARD,
    TrendContinuationObservation,
    TrendContinuationSelectionManifest,
)
from .trend_continuation_contract import stable_manifest_digest
from .trend_continuation_evaluation_types import EvaluationResult, EvaluationRow, ObservationSource
from .trend_continuation_statistics import evaluate_rows


@dataclass(frozen=True, slots=True)
class PreparedForwardBatch:
    records: tuple[TrendContinuationObservation, ...]
    rejection_code: str | None

    @property
    def admissible(self) -> bool:
        return self.rejection_code is None


@dataclass(frozen=True, slots=True)
class ForwardResultFields:
    metrics: EvaluationResult
    interval_start: datetime | None
    interval_end: datetime | None
    sample_digest: str


def _candidate_matches(manifest: TrendContinuationSelectionManifest, row: TrendContinuationObservation) -> bool:
    return (
        row.contract_symbol == manifest.contract_symbol
        and row.profile == manifest.profile
        and row.trend_direction == manifest.trend_direction
        and row.rule_name == manifest.rule_name
        and row.candidate_horizon_bars == manifest.candidate_horizon_bars
        and row.strategy_version == manifest.strategy_version
        and row.feature_schema_version == manifest.feature_schema_version
    )


def prepare_forward_batch(
    manifest: TrendContinuationSelectionManifest,
    observations: Sequence[TrendContinuationObservation],
) -> PreparedForwardBatch:
    ids = tuple(observation.pk for observation in observations if observation.pk is not None)
    if len(ids) != len(observations) or len(ids) != len(frozenset(ids)):
        return PreparedForwardBatch((), "invalid_or_duplicate_observation")
    records_by_id = {
        record.pk: record
        for record in TrendContinuationObservation.objects.filter(pk__in=ids).select_related("settlement")
    }
    if len(records_by_id) != len(ids):
        return PreparedForwardBatch((), "missing_observation")
    ordered = tuple(sorted((records_by_id[row_id] for row_id in ids), key=lambda row: row.evidence_time))
    if any(record.source_kind != SOURCE_FORWARD for record in ordered):
        return PreparedForwardBatch(ordered, "forward_result_requires_forward_source")
    if any(not _candidate_matches(manifest, record) for record in ordered):
        return PreparedForwardBatch(ordered, "candidate_identity_mismatch")
    if any(record.evidence_time <= manifest.frozen_at for record in ordered):
        return PreparedForwardBatch(ordered, "observation_not_newer_than_manifest")
    for record in ordered:
        try:
            record.settlement
        except TrendContinuationObservation.settlement.RelatedObjectDoesNotExist:
            return PreparedForwardBatch(ordered, "unresolved_observation")
    if any(
        previous.settlement.label_end_time >= current.evidence_time
        for previous, current in zip(ordered, ordered[1:])
    ):
        return PreparedForwardBatch(ordered, "stream_interval_overlap")
    return PreparedForwardBatch(ordered, None)


def result_fields(records: Sequence[TrendContinuationObservation]) -> ForwardResultFields:
    rows = tuple(
        EvaluationRow(
            evidence_time=record.evidence_time,
            label_end_time=record.settlement.label_end_time,
            reference_price=float(record.reference_price),
            invalidation_price=float(record.invalidation_price),
            outcome_success=record.settlement.outcome == "success",
            net_return_pct=record.settlement.net_return_pct,
            terminal_close=float(record.settlement.terminal_price),
            cost_pct=record.settlement.cost_pct,
            source=ObservationSource.FORWARD,
        )
        for record in records
    )
    payload = [
        {
            "observation_id": record.pk,
            "evidence_time": record.evidence_time.isoformat(),
            "label_end_time": record.settlement.label_end_time.isoformat(),
            "net_return_pct": record.settlement.net_return_pct,
            "outcome": record.settlement.outcome,
        }
        for record in records
    ]
    return ForwardResultFields(
        metrics=evaluate_rows(rows),
        interval_start=min((record.evidence_time for record in records), default=None),
        interval_end=max((record.settlement.label_end_time for record in records), default=None),
        sample_digest=stable_manifest_digest(payload),
    )


def rejected_result_fields() -> ForwardResultFields:
    return ForwardResultFields(
        metrics=EvaluationResult(
            resolved_count=0,
            success_count=0,
            wilson_lower_95=0.0,
            mean_return_lower_95=0.0,
            mean_return_pct=0.0,
            statistics_valid=False,
        ),
        interval_start=None,
        interval_end=None,
        sample_digest=stable_manifest_digest([]),
    )
