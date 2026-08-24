from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django.utils import timezone

from .trend_continuation_model_base import (
    DIRECTION_CHOICES,
    OUTCOME_CHOICES,
    SOURCE_CHOICES,
    SOURCE_FORWARD,
    SOURCE_HISTORICAL,
    ImmutableModelMixin,
)


class TrendContinuationObservation(ImmutableModelMixin):
    contract_symbol = models.CharField(max_length=32)
    underlying_symbol = models.CharField(max_length=24)
    profile = models.CharField(max_length=48)
    trend_direction = models.CharField(max_length=8, choices=DIRECTION_CHOICES)
    rule_name = models.CharField(max_length=64)
    candidate_horizon_bars = models.PositiveIntegerField()
    strategy_version = models.CharField(max_length=64)
    feature_schema_version = models.CharField(max_length=64)
    evidence_time = models.DateTimeField()
    reference_price = models.DecimalField(max_digits=24, decimal_places=8)
    invalidation_price = models.DecimalField(max_digits=24, decimal_places=8)
    trigger_interval_seconds = models.PositiveIntegerField()
    source_kind = models.CharField(max_length=32, choices=SOURCE_CHOICES)
    quote_observed_at = models.DateTimeField(null=True, blank=True)
    bid_price = models.DecimalField(max_digits=24, decimal_places=8, null=True, blank=True)
    ask_price = models.DecimalField(max_digits=24, decimal_places=8, null=True, blank=True)
    bid_ask_spread_pct = models.FloatField(null=True, blank=True)
    fee_rate = models.FloatField(null=True, blank=True)
    raw_response_sha256 = models.CharField(max_length=64)
    provenance = models.JSONField(default=dict)
    feature_snapshot = models.JSONField(default=dict)
    quote_snapshot = models.JSONField(default=dict, blank=True)
    funding_snapshot = models.JSONField(default=dict)
    fee_policy_snapshot = models.JSONField(default=dict)

    class Meta:
        ordering = ("evidence_time", "contract_symbol")
        constraints = [
            models.UniqueConstraint(
                fields=(
                    "contract_symbol",
                    "profile",
                    "trend_direction",
                    "rule_name",
                    "candidate_horizon_bars",
                    "strategy_version",
                    "feature_schema_version",
                    "evidence_time",
                ),
                name="unique_trend_observation_identity_time",
            ),
            models.CheckConstraint(
                condition=Q(trigger_interval_seconds__gt=0, candidate_horizon_bars__gt=0),
                name="trend_observation_positive_durations",
            ),
            models.CheckConstraint(
                condition=(
                    Q(source_kind=SOURCE_FORWARD)
                    | Q(
                        source_kind=SOURCE_HISTORICAL,
                        quote_observed_at__isnull=True,
                        bid_price__isnull=True,
                        ask_price__isnull=True,
                        bid_ask_spread_pct__isnull=True,
                    )
                ),
                name="historical_trend_observation_no_quote",
            ),
        ]

    @property
    def expected_label_end_time(self) -> datetime:
        return self.evidence_time + timedelta(
            seconds=self.trigger_interval_seconds * self.candidate_horizon_bars
        )

    def clean(self) -> None:
        errors: dict[str, str] = {}
        if self.reference_price <= 0 or self.invalidation_price <= 0:
            errors["reference_price"] = "prices must be positive"
        if self.trend_direction == "UP" and self.invalidation_price >= self.reference_price:
            errors["invalidation_price"] = "UP invalidation must be below reference"
        if self.trend_direction == "DOWN" and self.invalidation_price <= self.reference_price:
            errors["invalidation_price"] = "DOWN invalidation must be above reference"
        if len(self.raw_response_sha256) != 64:
            errors["raw_response_sha256"] = "raw response hash must be SHA-256"
        elif any(character not in "0123456789abcdefABCDEF" for character in self.raw_response_sha256):
            errors["raw_response_sha256"] = "raw response hash must be hexadecimal"
        if not timezone.is_aware(self.evidence_time):
            errors["evidence_time"] = "evidence time must be timezone-aware"
        quote_fields = (self.quote_observed_at, self.bid_price, self.ask_price, self.bid_ask_spread_pct)
        if self.source_kind == SOURCE_HISTORICAL and (any(value is not None for value in quote_fields) or self.quote_snapshot):
            errors["source_kind"] = "historical observations cannot contain quote or spread evidence"
        funding_complete = self.funding_snapshot.get("complete") if isinstance(self.funding_snapshot, dict) else None
        fee_hash = self.fee_policy_snapshot.get("raw_sha256") if isinstance(self.fee_policy_snapshot, dict) else None
        if funding_complete is not True:
            errors["funding_snapshot"] = "funding coverage must be complete"
        if not isinstance(fee_hash, str) or len(fee_hash) != 64:
            errors["fee_policy_snapshot"] = "fee policy requires its raw SHA-256"
        if self.source_kind == SOURCE_FORWARD:
            if any(value is None for value in quote_fields) or not self.quote_snapshot:
                errors["quote_snapshot"] = "forward observations require a complete quote snapshot"
            elif not self.evidence_time <= self.quote_observed_at <= self.evidence_time + timedelta(seconds=15):
                errors["quote_observed_at"] = "forward quote must be observed within 15 seconds"
            if self.bid_price is not None and self.ask_price is not None and self.ask_price < self.bid_price:
                errors["ask_price"] = "forward ask must not be below bid"
            if self.bid_ask_spread_pct is not None and (
                not isfinite(self.bid_ask_spread_pct) or self.bid_ask_spread_pct < 0
            ):
                errors["bid_ask_spread_pct"] = "forward spread must be finite and nonnegative"
            if self.fee_rate is None or not isfinite(self.fee_rate) or self.fee_rate < 0:
                errors["fee_rate"] = "forward fee rate must be finite and nonnegative"
            quote_hash = self.quote_snapshot.get("raw_sha256") if isinstance(self.quote_snapshot, dict) else None
            if not isinstance(quote_hash, str) or len(quote_hash) != 64:
                errors["quote_snapshot"] = "forward quote snapshot requires its raw SHA-256"
            if not self.funding_snapshot or not self.fee_policy_snapshot:
                errors["funding_snapshot"] = "forward funding and fee policy snapshots are required"
        if errors:
            raise ValidationError(errors)


class TrendContinuationSettlement(ImmutableModelMixin):
    observation = models.OneToOneField(
        TrendContinuationObservation,
        on_delete=models.PROTECT,
        related_name="settlement",
    )
    outcome = models.CharField(max_length=16, choices=OUTCOME_CHOICES)
    label_end_time = models.DateTimeField()
    terminal_price = models.DecimalField(max_digits=24, decimal_places=8)
    cost_pct = models.FloatField()
    net_return_pct = models.FloatField()
    settled_at = models.DateTimeField()
    provenance = models.JSONField(default=dict)

    def clean(self) -> None:
        errors: dict[str, str] = {}
        if self.observation_id and self.label_end_time != self.observation.expected_label_end_time:
            errors["label_end_time"] = "label end must equal evidence plus fixed horizon"
        if not timezone.is_aware(self.label_end_time) or not timezone.is_aware(self.settled_at):
            errors["label_end_time"] = "settlement timestamps must be timezone-aware"
        if self.terminal_price <= 0:
            errors["terminal_price"] = "terminal price must be positive"
        if not all(isfinite(value) for value in (self.cost_pct, self.net_return_pct)) or self.cost_pct < 0:
            errors["cost_pct"] = "cost and return must be finite and cost nonnegative"
        if self.outcome == "success" and self.net_return_pct <= 0:
            errors["outcome"] = "success requires a strictly positive net return"
        if self.outcome == "failure" and self.net_return_pct > 0 and not self.provenance.get("invalidated"):
            errors["outcome"] = "positive-return failure must record invalidation"
        if errors:
            raise ValidationError(errors)
