from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from threading import Lock, RLock

from django.core.exceptions import ValidationError
from django.db import transaction

from ..trend_continuation_models import (
    TrendContinuationContractLedger,
    TrendContinuationForwardConsumption,
    TrendContinuationObservation,
    TrendContinuationSelectionManifest,
)
from .trend_continuation_manifest_types import ForwardIntervalOverlapError, JsonObject


_CONTRACT_LOCKS_GUARD = Lock()
_CONTRACT_LOCKS: dict[str, RLock] = {}


def _contract_mutex(contract_symbol: str) -> RLock:
    with _CONTRACT_LOCKS_GUARD:
        return _CONTRACT_LOCKS.setdefault(contract_symbol, RLock())


def _intervals_overlap(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    return start_a <= end_b and start_b <= end_a


def register_forward_consumption(
    *,
    contract_symbol: str,
    interval_start: datetime,
    interval_end: datetime,
    manifest_digest: str,
    profile: str,
    trend_direction: str,
    rule_name: str,
    candidate_horizon_bars: int,
    phase: str = "validation",
    strategy_version: str | None = None,
    feature_schema_version: str | None = None,
    observation: TrendContinuationObservation | None = None,
    provenance: JsonObject | None = None,
) -> TrendContinuationForwardConsumption:
    if interval_start.tzinfo is None or interval_end.tzinfo is None or interval_start > interval_end:
        raise ValidationError({"interval_start": "closed interval must be aware and ordered"})
    with _contract_mutex(contract_symbol), transaction.atomic():
        manifest = TrendContinuationSelectionManifest.objects.get(digest=manifest_digest)
        ledger, _ = TrendContinuationContractLedger.objects.get_or_create(contract_symbol=contract_symbol)
        locked_ledger = TrendContinuationContractLedger.objects.select_for_update().get(pk=ledger.pk)
        overlap = locked_ledger.consumptions.filter(
            interval_start__lte=interval_end,
            interval_end__gte=interval_start,
        ).first()
        if overlap is not None and _intervals_overlap(
            interval_start,
            interval_end,
            overlap.interval_start,
            overlap.interval_end,
        ):
            raise ForwardIntervalOverlapError(contract_symbol, interval_start, interval_end)
        return TrendContinuationForwardConsumption.objects.create(
            ledger=locked_ledger,
            manifest=manifest,
            observation=observation,
            contract_symbol=contract_symbol,
            interval_start=interval_start,
            interval_end=interval_end,
            manifest_digest=manifest_digest,
            profile=profile,
            trend_direction=trend_direction,
            rule_name=rule_name,
            candidate_horizon_bars=candidate_horizon_bars,
            strategy_version=strategy_version or manifest.strategy_version,
            feature_schema_version=feature_schema_version or manifest.feature_schema_version,
            phase=phase,
            provenance=provenance or {"closed_interval": True},
        )


def consume_forward_batch(
    manifest: TrendContinuationSelectionManifest,
    records: Sequence[TrendContinuationObservation],
    *,
    phase: str,
) -> str | None:
    try:
        with transaction.atomic():
            for record in records:
                register_forward_consumption(
                    contract_symbol=manifest.contract_symbol,
                    interval_start=record.evidence_time,
                    interval_end=record.settlement.label_end_time,
                    manifest_digest=manifest.digest,
                    profile=manifest.profile,
                    trend_direction=manifest.trend_direction,
                    rule_name=manifest.rule_name,
                    candidate_horizon_bars=manifest.candidate_horizon_bars,
                    strategy_version=manifest.strategy_version,
                    feature_schema_version=manifest.feature_schema_version,
                    observation=record,
                    phase=phase,
                    provenance={"closed_interval": True},
                )
    except ForwardIntervalOverlapError:
        return "global_interval_overlap"
    except ValidationError:
        return "invalid_consumption_metadata"
    return None
