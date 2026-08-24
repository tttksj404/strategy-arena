from __future__ import annotations

from collections.abc import Sequence

from django.db import transaction

from ..trend_continuation_models import (
    TrendContinuationHoldoutResult,
    TrendContinuationManifestRejection,
    TrendContinuationObservation,
    TrendContinuationSelectionManifest,
    TrendContinuationValidationResult,
)
from .trend_continuation_consumption import consume_forward_batch
from .trend_continuation_evaluation_types import FINAL_HOLDOUT_OBSERVATIONS, MIN_SELECTION_OBSERVATIONS
from .trend_continuation_forward_batch import rejected_result_fields, prepare_forward_batch, result_fields
from .trend_continuation_manifest_types import ManifestLifecycleError
from .trend_continuation_statistics import forward_gate


def append_validation_result(
    manifest: TrendContinuationSelectionManifest,
    observations: Sequence[TrendContinuationObservation],
) -> TrendContinuationValidationResult:
    duplicate_reason: str | None = None
    with transaction.atomic():
        locked = TrendContinuationSelectionManifest.objects.select_for_update().get(pk=manifest.pk)
        if hasattr(locked, "terminal_rejection"):
            duplicate_reason = locked.terminal_rejection.rejection_code
        elif hasattr(locked, "validation_result"):
            duplicate_reason = "validation_already_accessed"
            TrendContinuationManifestRejection.objects.create(
                manifest=locked,
                rejection_code=duplicate_reason,
                provenance={"terminal": True, "duplicate_result_access": True},
            )
        else:
            batch = prepare_forward_batch(locked, observations)
            rejection = batch.rejection_code
            if len(observations) != MIN_SELECTION_OBSERVATIONS:
                rejection = "validation_requires_exactly_80"
            fields = result_fields(batch.records) if rejection is None else rejected_result_fields()
            passed = rejection is None and forward_gate(fields.metrics, exact_count=MIN_SELECTION_OBSERVATIONS)
            if rejection is None and not passed:
                rejection = "validation_fixed_metrics_failed"
            if passed:
                rejection = consume_forward_batch(locked, batch.records, phase="validation")
                passed = rejection is None
            return TrendContinuationValidationResult.objects.create(
                manifest=locked,
                resolved_count=fields.metrics.resolved_count,
                success_count=fields.metrics.success_count,
                wilson_lower_95=fields.metrics.wilson_lower_95,
                mean_return_lower_95=fields.metrics.mean_return_lower_95,
                mean_return_pct=fields.metrics.mean_return_pct,
                passed=passed,
                rejection_code="" if passed else (rejection or "validation_failed"),
                interval_start=fields.interval_start,
                interval_end=fields.interval_end,
                sample_digest=fields.sample_digest,
                provenance={"immutable_forward": True, "exact_required_count": MIN_SELECTION_OBSERVATIONS},
            )
    raise ManifestLifecycleError(manifest.digest, duplicate_reason or "validation_already_accessed")


def append_holdout_result(
    manifest: TrendContinuationSelectionManifest,
    observations: Sequence[TrendContinuationObservation],
) -> TrendContinuationHoldoutResult:
    duplicate_reason: str | None = None
    with transaction.atomic():
        locked = TrendContinuationSelectionManifest.objects.select_for_update().get(pk=manifest.pk)
        if hasattr(locked, "terminal_rejection"):
            duplicate_reason = locked.terminal_rejection.rejection_code
        elif hasattr(locked, "holdout_result"):
            duplicate_reason = "holdout_already_accessed"
            TrendContinuationManifestRejection.objects.create(
                manifest=locked,
                rejection_code=duplicate_reason,
                provenance={"terminal": True, "duplicate_result_access": True},
            )
        else:
            try:
                validation = locked.validation_result
            except TrendContinuationSelectionManifest.validation_result.RelatedObjectDoesNotExist as error:
                raise ManifestLifecycleError(locked.digest, "validation_not_passed") from error
            if not validation.passed or validation.interval_end is None:
                raise ManifestLifecycleError(locked.digest, "validation_not_passed")
            batch = prepare_forward_batch(locked, observations)
            rejection = batch.rejection_code
            if len(observations) != FINAL_HOLDOUT_OBSERVATIONS:
                rejection = "holdout_requires_exactly_20"
            if batch.records and batch.records[0].evidence_time <= validation.interval_end:
                rejection = rejection or "holdout_not_strictly_later"
            fields = result_fields(batch.records) if rejection is None else rejected_result_fields()
            passed = rejection is None and forward_gate(fields.metrics, exact_count=FINAL_HOLDOUT_OBSERVATIONS)
            if rejection is None and not passed:
                rejection = "holdout_fixed_metrics_failed"
            if passed:
                rejection = consume_forward_batch(locked, batch.records, phase="holdout")
                passed = rejection is None
            return TrendContinuationHoldoutResult.objects.create(
                manifest=locked,
                resolved_count=fields.metrics.resolved_count,
                success_count=fields.metrics.success_count,
                wilson_lower_95=fields.metrics.wilson_lower_95,
                mean_return_lower_95=fields.metrics.mean_return_lower_95,
                mean_return_pct=fields.metrics.mean_return_pct,
                passed=passed,
                rejection_code="" if passed else (rejection or "holdout_failed"),
                interval_start=fields.interval_start,
                interval_end=fields.interval_end,
                sample_digest=fields.sample_digest,
                provenance={"immutable_forward": True, "exact_required_count": FINAL_HOLDOUT_OBSERVATIONS},
            )
    raise ManifestLifecycleError(manifest.digest, duplicate_reason or "holdout_already_accessed")
