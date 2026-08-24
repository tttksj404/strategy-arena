from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Protocol

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q
from django.utils import timezone

from .services.trend_continuation_contract import stable_manifest_digest
from .trend_continuation_model_base import DIRECTION_CHOICES, ImmutableModelMixin


class TrendContinuationSelectionManifest(ImmutableModelMixin):
    digest = models.CharField(max_length=64, unique=True)
    schema_version = models.CharField(max_length=48, default="trend-continuation-selection.v1")
    payload = models.JSONField(default=dict)
    frozen_at = models.DateTimeField()
    contract_symbol = models.CharField(max_length=32)
    profile = models.CharField(max_length=48)
    trend_direction = models.CharField(max_length=8, choices=DIRECTION_CHOICES)
    rule_name = models.CharField(max_length=64)
    candidate_horizon_bars = models.PositiveIntegerField()
    strategy_version = models.CharField(max_length=64)
    feature_schema_version = models.CharField(max_length=64)
    exploratory_only = models.BooleanField(default=True)

    class Meta:
        constraints = [
            models.CheckConstraint(condition=Q(exploratory_only=True), name="trend_manifest_exploratory_only"),
        ]

    def clean(self) -> None:
        if self.digest != stable_manifest_digest(self.payload):
            raise ValidationError({"digest": "manifest digest does not match its immutable payload"})
        if not self.exploratory_only:
            raise ValidationError({"exploratory_only": "historical selection never promotes directly"})
        candidate = self.payload.get("candidate") if isinstance(self.payload, dict) else None
        expected = {
            "contract": self.contract_symbol,
            "profile": self.profile,
            "trend_direction": self.trend_direction,
            "rule_name": self.rule_name,
            "candidate_horizon_bars": self.candidate_horizon_bars,
            "strategy_version": self.strategy_version,
            "feature_schema_version": self.feature_schema_version,
        }
        if not isinstance(candidate, dict) or any(candidate.get(key) != value for key, value in expected.items()):
            raise ValidationError({"payload": "manifest columns must match the digest-frozen candidate"})
        if not timezone.is_aware(self.frozen_at):
            raise ValidationError({"frozen_at": "manifest freeze time must be timezone-aware"})


class TrendContinuationManifestRejection(ImmutableModelMixin):
    manifest = models.OneToOneField(
        TrendContinuationSelectionManifest,
        on_delete=models.PROTECT,
        related_name="terminal_rejection",
    )
    rejection_code = models.CharField(max_length=80)
    provenance = models.JSONField(default=dict)

    def clean(self) -> None:
        if not self.rejection_code:
            raise ValidationError({"rejection_code": "terminal rejection requires a code"})


class _ResultProtocol(Protocol):
    resolved_count: int
    success_count: int
    wilson_lower_95: float
    mean_return_lower_95: float
    mean_return_pct: float
    passed: bool
    rejection_code: str
    interval_start: datetime | None
    interval_end: datetime | None
    sample_digest: str


def _clean_result(result: _ResultProtocol, *, exact_count: int) -> None:
    errors: dict[str, str] = {}
    values = (result.wilson_lower_95, result.mean_return_lower_95, result.mean_return_pct)
    if result.success_count > result.resolved_count:
        errors["success_count"] = "success count cannot exceed resolved count"
    if not all(isfinite(value) for value in values):
        errors["wilson_lower_95"] = "all metrics must be finite"
    if result.passed and (
        result.resolved_count != exact_count
        or result.wilson_lower_95 < 50
        or result.mean_return_lower_95 <= 0
        or result.rejection_code
    ):
        errors["passed"] = "passed result violates the fixed count or statistical gates"
    if not result.passed and not result.rejection_code:
        errors["rejection_code"] = "failed result requires a terminal rejection code"
    if (result.interval_start is None) != (result.interval_end is None):
        errors["interval_start"] = "result interval endpoints must be both present or both absent"
    if result.resolved_count and (result.interval_start is None or result.interval_start > result.interval_end):
        errors["interval_start"] = "resolved samples require a valid result interval"
    if result.interval_start is not None and (
        result.interval_end is None
        or not timezone.is_aware(result.interval_start)
        or not timezone.is_aware(result.interval_end)
    ):
        errors["interval_start"] = "result interval must be timezone-aware"
    if len(result.sample_digest) != 64:
        errors["sample_digest"] = "sample digest must be SHA-256"
    elif any(character not in "0123456789abcdefABCDEF" for character in result.sample_digest):
        errors["sample_digest"] = "sample digest must be hexadecimal"
    if errors:
        raise ValidationError(errors)


class TrendContinuationValidationResult(ImmutableModelMixin):
    manifest = models.OneToOneField(
        TrendContinuationSelectionManifest,
        on_delete=models.PROTECT,
        related_name="validation_result",
    )
    resolved_count = models.PositiveIntegerField()
    success_count = models.PositiveIntegerField()
    wilson_lower_95 = models.FloatField()
    mean_return_lower_95 = models.FloatField()
    mean_return_pct = models.FloatField()
    passed = models.BooleanField(default=False)
    rejection_code = models.CharField(max_length=80, blank=True)
    interval_start = models.DateTimeField(null=True, blank=True)
    interval_end = models.DateTimeField(null=True, blank=True)
    sample_digest = models.CharField(max_length=64)
    provenance = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.CheckConstraint(condition=Q(success_count__lte=F("resolved_count")), name="trend_validation_success_lte_n"),
            models.CheckConstraint(condition=Q(passed=False) | Q(resolved_count=80), name="trend_validation_pass_exact_80"),
        ]

    def clean(self) -> None:
        _clean_result(self, exact_count=80)


class TrendContinuationHoldoutResult(ImmutableModelMixin):
    manifest = models.OneToOneField(
        TrendContinuationSelectionManifest,
        on_delete=models.PROTECT,
        related_name="holdout_result",
    )
    resolved_count = models.PositiveIntegerField()
    success_count = models.PositiveIntegerField()
    wilson_lower_95 = models.FloatField()
    mean_return_lower_95 = models.FloatField()
    mean_return_pct = models.FloatField()
    passed = models.BooleanField(default=False)
    rejection_code = models.CharField(max_length=80, blank=True)
    interval_start = models.DateTimeField(null=True, blank=True)
    interval_end = models.DateTimeField(null=True, blank=True)
    sample_digest = models.CharField(max_length=64)
    provenance = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.CheckConstraint(condition=Q(success_count__lte=F("resolved_count")), name="trend_holdout_success_lte_n"),
            models.CheckConstraint(condition=Q(passed=False) | Q(resolved_count=20), name="trend_holdout_pass_exact_20"),
        ]

    def clean(self) -> None:
        _clean_result(self, exact_count=20)
        if self.manifest_id:
            try:
                validation = self.manifest.validation_result
            except TrendContinuationSelectionManifest.validation_result.RelatedObjectDoesNotExist as error:
                raise ValidationError({"manifest": "holdout requires a passed validation result"}) from error
            if not validation.passed or validation.interval_end is None:
                raise ValidationError({"manifest": "holdout requires a passed validation result"})
            if self.interval_start is not None and self.interval_start <= validation.interval_end:
                raise ValidationError({"interval_start": "holdout must be strictly later than validation"})
