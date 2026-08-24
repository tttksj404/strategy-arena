from __future__ import annotations

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q
from django.utils import timezone

from .trend_continuation_manifest_models import TrendContinuationSelectionManifest
from .trend_continuation_model_base import PHASE_CHOICES, SOURCE_FORWARD, ImmutableModelMixin
from .trend_continuation_observation_models import TrendContinuationObservation


class TrendContinuationContractLedger(ImmutableModelMixin):
    contract_symbol = models.CharField(max_length=32, unique=True)


class TrendContinuationForwardConsumption(ImmutableModelMixin):
    ledger = models.ForeignKey(
        TrendContinuationContractLedger,
        on_delete=models.PROTECT,
        related_name="consumptions",
    )
    manifest = models.ForeignKey(
        TrendContinuationSelectionManifest,
        on_delete=models.PROTECT,
        related_name="forward_consumptions",
    )
    observation = models.ForeignKey(
        TrendContinuationObservation,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="forward_consumptions",
    )
    contract_symbol = models.CharField(max_length=32)
    interval_start = models.DateTimeField()
    interval_end = models.DateTimeField()
    manifest_digest = models.CharField(max_length=64)
    profile = models.CharField(max_length=48)
    trend_direction = models.CharField(max_length=8)
    rule_name = models.CharField(max_length=64)
    candidate_horizon_bars = models.PositiveIntegerField()
    strategy_version = models.CharField(max_length=64)
    feature_schema_version = models.CharField(max_length=64)
    phase = models.CharField(max_length=16, choices=PHASE_CHOICES)
    provenance = models.JSONField(default=dict)

    class Meta:
        ordering = ("contract_symbol", "interval_start")
        constraints = [
            models.CheckConstraint(condition=Q(interval_start__lte=F("interval_end")), name="trend_consumption_valid_interval"),
            models.UniqueConstraint(
                fields=("ledger", "interval_start", "interval_end"),
                name="unique_trend_consumed_closed_interval",
            ),
        ]

    def clean(self) -> None:
        errors: dict[str, str] = {}
        if self.interval_start > self.interval_end:
            errors["interval_start"] = "consumption interval must be ordered"
        if not timezone.is_aware(self.interval_start) or not timezone.is_aware(self.interval_end):
            errors["interval_start"] = "consumption interval must be timezone-aware"
        if self.ledger_id and self.ledger.contract_symbol != self.contract_symbol:
            errors["ledger"] = "contract ledger must match consumption contract"
        if self.manifest_id:
            manifest_identity = (
                self.manifest.digest,
                self.manifest.contract_symbol,
                self.manifest.profile,
                self.manifest.trend_direction,
                self.manifest.rule_name,
                self.manifest.candidate_horizon_bars,
                self.manifest.strategy_version,
                self.manifest.feature_schema_version,
            )
            consumption_identity = (
                self.manifest_digest,
                self.contract_symbol,
                self.profile,
                self.trend_direction,
                self.rule_name,
                self.candidate_horizon_bars,
                self.strategy_version,
                self.feature_schema_version,
            )
            if manifest_identity != consumption_identity:
                errors["manifest"] = "manifest identity must exactly match consumption metadata"
        if self.observation_id:
            observation_identity = (
                self.observation.contract_symbol,
                self.observation.profile,
                self.observation.trend_direction,
                self.observation.rule_name,
                self.observation.candidate_horizon_bars,
                self.observation.strategy_version,
                self.observation.feature_schema_version,
            )
            consumption_candidate = (
                self.contract_symbol,
                self.profile,
                self.trend_direction,
                self.rule_name,
                self.candidate_horizon_bars,
                self.strategy_version,
                self.feature_schema_version,
            )
            if self.observation.source_kind != SOURCE_FORWARD or observation_identity != consumption_candidate:
                errors["observation"] = "forward observation identity must match consumption metadata"
            if self.interval_start != self.observation.evidence_time:
                errors["interval_start"] = "consumption must start at observation evidence time"
            try:
                settlement = self.observation.settlement
            except TrendContinuationObservation.settlement.RelatedObjectDoesNotExist:
                errors["observation"] = "consumption observation must be resolved"
            else:
                if self.interval_end != settlement.label_end_time:
                    errors["interval_end"] = "consumption must end at linked settlement label end"
        if errors:
            raise ValidationError(errors)
