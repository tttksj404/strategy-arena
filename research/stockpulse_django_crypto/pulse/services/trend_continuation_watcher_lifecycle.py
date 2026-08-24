from __future__ import annotations

from enum import StrEnum
from typing import assert_never

from ..trend_continuation_models import (
    SOURCE_FORWARD,
    TrendContinuationObservation,
    TrendContinuationSelectionManifest,
)
from .trend_continuation_contract import CandidateIdentity, Direction
from .trend_continuation_manifest import append_holdout_result, append_validation_result


class WatcherLifecyclePhase(StrEnum):
    COLLECT_VALIDATION = "collect_validation"
    SETTLE_VALIDATION = "settle_validation"
    FINALIZE_VALIDATION = "finalize_validation"
    COLLECT_HOLDOUT = "collect_holdout"
    SETTLE_HOLDOUT = "settle_holdout"
    FINALIZE_HOLDOUT = "finalize_holdout"
    COMPLETE = "complete"
    TERMINAL_REJECTED = "terminal_rejected"


def watcher_lifecycle_phase(manifest: TrendContinuationSelectionManifest) -> WatcherLifecyclePhase:
    current = TrendContinuationSelectionManifest.objects.get(pk=manifest.pk)
    if hasattr(current, "terminal_rejection"):
        return WatcherLifecyclePhase.TERMINAL_REJECTED
    if hasattr(current, "holdout_result"):
        return WatcherLifecyclePhase.COMPLETE if current.holdout_result.passed else WatcherLifecyclePhase.TERMINAL_REJECTED
    validation = current.validation_result if hasattr(current, "validation_result") else None
    if validation is not None and not validation.passed:
        return WatcherLifecyclePhase.TERMINAL_REJECTED
    records = manifest_records(current)
    if validation is None:
        return _phase_for_counts(records, 80, holdout=False)
    later = tuple(
        record
        for record in records
        if validation.interval_end is not None and record.evidence_time > validation.interval_end
    )
    return _phase_for_counts(later, 20, holdout=True)


def advance_watcher_lifecycle(manifest: TrendContinuationSelectionManifest) -> WatcherLifecyclePhase:
    phase = watcher_lifecycle_phase(manifest)
    current = TrendContinuationSelectionManifest.objects.get(pk=manifest.pk)
    records = manifest_records(current)
    match phase:
        case WatcherLifecyclePhase.FINALIZE_VALIDATION:
            append_validation_result(current, records)
        case WatcherLifecyclePhase.FINALIZE_HOLDOUT:
            validation = current.validation_result
            later = tuple(
                record
                for record in records
                if validation.interval_end is not None and record.evidence_time > validation.interval_end
            )
            append_holdout_result(current, later)
        case (
            WatcherLifecyclePhase.COLLECT_VALIDATION
            | WatcherLifecyclePhase.SETTLE_VALIDATION
            | WatcherLifecyclePhase.COLLECT_HOLDOUT
            | WatcherLifecyclePhase.SETTLE_HOLDOUT
            | WatcherLifecyclePhase.COMPLETE
            | WatcherLifecyclePhase.TERMINAL_REJECTED
        ):
            pass
        case unreachable:
            assert_never(unreachable)
    return watcher_lifecycle_phase(current)


def manifest_records(
    manifest: TrendContinuationSelectionManifest,
) -> tuple[TrendContinuationObservation, ...]:
    identity = CandidateIdentity(
        manifest.contract_symbol,
        manifest.profile,
        Direction(manifest.trend_direction),
        manifest.rule_name,
        manifest.candidate_horizon_bars,
        manifest.strategy_version,
        manifest.feature_schema_version,
    )
    return tuple(identity_records(identity).filter(evidence_time__gt=manifest.frozen_at).order_by("evidence_time"))


def identity_records(identity: CandidateIdentity):
    return TrendContinuationObservation.objects.filter(
        contract_symbol=identity.contract,
        profile=identity.profile,
        trend_direction=identity.trend_direction.value,
        rule_name=identity.rule_name,
        candidate_horizon_bars=identity.candidate_horizon_bars,
        strategy_version=identity.strategy_version,
        feature_schema_version=identity.feature_schema_version,
        source_kind=SOURCE_FORWARD,
    )


def _phase_for_counts(
    records: tuple[TrendContinuationObservation, ...],
    required: int,
    *,
    holdout: bool,
) -> WatcherLifecyclePhase:
    finalize = WatcherLifecyclePhase.FINALIZE_HOLDOUT if holdout else WatcherLifecyclePhase.FINALIZE_VALIDATION
    settle = WatcherLifecyclePhase.SETTLE_HOLDOUT if holdout else WatcherLifecyclePhase.SETTLE_VALIDATION
    collect = WatcherLifecyclePhase.COLLECT_HOLDOUT if holdout else WatcherLifecyclePhase.COLLECT_VALIDATION
    if len(records) < required:
        return collect
    resolved = sum(hasattr(record, "settlement") for record in records)
    return finalize if resolved == len(records) else settle
