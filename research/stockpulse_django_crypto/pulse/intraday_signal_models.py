from __future__ import annotations

from enum import StrEnum

from django.db import models

from .models import TimeStampedModel


class SignalOutcome(StrEnum):
    """Persisted lifecycle states for a forward-tested intraday signal."""

    PENDING = "pending"
    TARGET = "target"
    STOP = "stop"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


OUTCOME_CHOICES = tuple((outcome.value, outcome.name.title()) for outcome in SignalOutcome)


class IntradayLongSignal(TimeStampedModel):
    """Immutable-at-entry long signal retained for forward-only outcome measurement."""

    contract_symbol = models.CharField(max_length=32)
    underlying_symbol = models.CharField(max_length=24)
    profile = models.CharField(max_length=48)
    strategy_version = models.CharField(max_length=64, default="mtf_breakout_v1_unquoted")
    signal_opened_at = models.DateTimeField()
    starts_at = models.DateTimeField()
    expires_at = models.DateTimeField()
    entry_price = models.DecimalField(max_digits=24, decimal_places=8)
    target_price = models.DecimalField(max_digits=24, decimal_places=8)
    stop_price = models.DecimalField(max_digits=24, decimal_places=8)
    taker_fee_rate = models.FloatField()
    entry_quote_observed_at = models.DateTimeField(null=True, blank=True)
    entry_quote_spread_pct = models.FloatField(null=True, blank=True)
    estimated_margin_loss_at_stop_pct = models.FloatField()
    status = models.CharField(max_length=16, choices=OUTCOME_CHOICES, default=SignalOutcome.PENDING)
    terminal_price = models.DecimalField(max_digits=24, decimal_places=8, null=True, blank=True)
    post_cost_margin_return_pct = models.FloatField(null=True, blank=True)
    settled_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("contract_symbol", "profile", "strategy_version", "signal_opened_at"),
                name="unique_intraday_long_signal_per_closed_trigger_and_version",
            ),
        ]
        ordering = ("-signal_opened_at", "contract_symbol")
