from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from django.db.models import Max, Min

from .trend_continuation_contract import (
    CandidateIdentity,
    HorizonStatus,
    ResearchDecision,
    TrendContinuationDiagnostic,
)
from .trend_continuation_evaluation_types import (
    FINAL_HOLDOUT_OBSERVATIONS,
    MIN_SELECTION_OBSERVATIONS,
    EvaluationResult,
)

if TYPE_CHECKING:
    from ..trend_continuation_models import TrendContinuationSelectionManifest


def choose_diagnostic(
    *,
    identity: CandidateIdentity,
    as_of: datetime,
    evidence_time: datetime,
    trigger_interval_seconds: int,
    reference_price: float,
    invalidation_price: float,
    validation_result: EvaluationResult | None = None,
    manifest_digest: str | None = None,
) -> TrendContinuationDiagnostic:
    """Observe only after immutable passed 80-validation and later 20-holdout rows."""
    del validation_result
    manifest = _validated_manifest(identity, manifest_digest)
    if manifest is not None:
        holdout = manifest.holdout_result
        return TrendContinuationDiagnostic(
            identity=identity,
            as_of=as_of,
            evidence_time=evidence_time,
            trigger_interval_seconds=trigger_interval_seconds,
            reference_price=reference_price,
            invalidation_price=invalidation_price,
            decision=ResearchDecision.CONTINUATION_OBSERVE,
            horizon_status=HorizonStatus.VALIDATED,
            rejection_code=None,
            all_observe_gates_passed=True,
            holdout_empirical_continuation_rate_pct=holdout.success_count / holdout.resolved_count * 100,
            wilson_lower_95=holdout.wilson_lower_95,
            mean_return_lower_95=holdout.mean_return_lower_95,
            holdout_sample_count=holdout.resolved_count,
            forward_provenance={"manifest_digest": manifest.digest, "validation_count": MIN_SELECTION_OBSERVATIONS},
        )
    return TrendContinuationDiagnostic(
        identity=identity,
        as_of=as_of,
        evidence_time=evidence_time,
        trigger_interval_seconds=trigger_interval_seconds,
        reference_price=reference_price,
        invalidation_price=invalidation_price,
        decision=ResearchDecision.NO_TRADE,
        horizon_status=HorizonStatus.PENDING_VALIDATION,
        rejection_code="pending_forward_validation",
    )


def _validated_manifest(
    identity: CandidateIdentity,
    manifest_digest: str | None,
) -> TrendContinuationSelectionManifest | None:
    if manifest_digest is None:
        return None
    from ..trend_continuation_models import TrendContinuationSelectionManifest

    manifest = (
        TrendContinuationSelectionManifest.objects.filter(
            digest=manifest_digest,
            contract_symbol=identity.contract,
            profile=identity.profile,
            trend_direction=identity.trend_direction.value,
            rule_name=identity.rule_name,
            candidate_horizon_bars=identity.candidate_horizon_bars,
            strategy_version=identity.strategy_version,
            feature_schema_version=identity.feature_schema_version,
            terminal_rejection__isnull=True,
            validation_result__passed=True,
            validation_result__resolved_count=MIN_SELECTION_OBSERVATIONS,
            holdout_result__passed=True,
            holdout_result__resolved_count=FINAL_HOLDOUT_OBSERVATIONS,
        )
        .select_related("validation_result", "holdout_result")
        .first()
    )
    if manifest is None:
        return None
    validation = manifest.validation_result
    holdout = manifest.holdout_result
    validation_rows = manifest.forward_consumptions.filter(phase="validation")
    holdout_rows = manifest.forward_consumptions.filter(phase="holdout")
    validation_bounds = validation_rows.aggregate(start=Min("interval_start"), end=Max("interval_end"))
    holdout_bounds = holdout_rows.aggregate(start=Min("interval_start"), end=Max("interval_end"))
    consumption_proof = (
        validation_rows.count() == MIN_SELECTION_OBSERVATIONS
        and validation_rows.filter(observation__isnull=False).count() == MIN_SELECTION_OBSERVATIONS
        and holdout_rows.count() == FINAL_HOLDOUT_OBSERVATIONS
        and holdout_rows.filter(observation__isnull=False).count() == FINAL_HOLDOUT_OBSERVATIONS
        and validation_bounds == {"start": validation.interval_start, "end": validation.interval_end}
        and holdout_bounds == {"start": holdout.interval_start, "end": holdout.interval_end}
        and validation.interval_end is not None
        and holdout.interval_start is not None
        and validation.interval_end < holdout.interval_start
    )
    return manifest if consumption_proof else None
