from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from math import sqrt
from typing import Final, assert_never

from pulse.intraday_signal_models import IntradayLongSignal, SignalOutcome

from .intraday_long_forecast import (
    FixedLeverageFeasibility,
    InstrumentProfile,
    IntradayLongPlan,
    round_trip_margin_cost_pct,
)


MIN_SETTLED_SIGNALS: Final = 100
MIN_VALIDATION_SIGNALS: Final = 80
MIN_HELD_OUT_SIGNALS: Final = 20
MIN_TARGET_FIRST_RATE_PCT: Final = 55.0
MIN_WILSON_LOWER_BOUND_PCT: Final = 50.0
MAX_WORST_MARGIN_LOSS_PCT: Final = 8.0
MIN_CONFIDENCE_RETURN_SAMPLES: Final = 20
CONSERVATIVE_MEAN_RETURN_T_CRITICAL: Final = 2.093


@dataclass(frozen=True, slots=True)
class SignalObservation:
    """One currently valid public-market plan that a watcher can persist before its outcome exists."""

    contract_symbol: str
    underlying_symbol: str
    profile: InstrumentProfile
    strategy_version: str
    signal_opened_at: datetime
    plan: IntradayLongPlan
    taker_fee_rate: float
    feasibility: FixedLeverageFeasibility
    entry_quote_observed_at: datetime
    entry_quote_spread_pct: float


@dataclass(frozen=True, slots=True)
class ForwardPerformance:
    """Settled outcomes from immutable entries, grouped by symbol and frozen profile."""

    settled_signal_count: int
    target_first_count: int
    stop_first_count: int
    unresolved_count: int
    ambiguous_count: int
    post_cost_expected_margin_return_pct: float
    post_cost_margin_return_lower_bound_pct: float
    worst_margin_loss_at_stop_pct: float

    @property
    def target_first_rate_pct(self) -> float:
        if self.settled_signal_count == 0:
            return 0.0
        return self.target_first_count / self.settled_signal_count * 100

    @property
    def wilson_lower_bound_pct(self) -> float:
        if self.settled_signal_count == 0:
            return 0.0
        successes = self.target_first_count
        total = self.settled_signal_count
        z_squared = 1.96**2
        proportion = successes / total
        centre = proportion + z_squared / (2 * total)
        variance = proportion * (1 - proportion) / total + z_squared / (4 * total**2)
        lower = (centre - 1.96 * variance**0.5) / (1 + z_squared / total)
        return max(lower * 100, 0.0)


@dataclass(frozen=True, slots=True)
class ForwardPromotion:
    """Chronological validation and untouched final-holdout evidence for one contract profile."""

    total: ForwardPerformance
    validation: ForwardPerformance
    held_out: ForwardPerformance


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def record_observations(observations: tuple[SignalObservation, ...]) -> int:
    """Persist only new closed-trigger observations; duplicate watcher passes are idempotent."""

    recorded = 0
    for observation in observations:
        _, created = IntradayLongSignal.objects.get_or_create(
            contract_symbol=observation.contract_symbol,
            profile=observation.profile.value,
            strategy_version=observation.strategy_version,
            signal_opened_at=observation.signal_opened_at,
            defaults={
                "underlying_symbol": observation.underlying_symbol,
                "starts_at": observation.plan.starts_at,
                "expires_at": observation.plan.expires_at,
                "entry_price": _decimal(observation.plan.entry_price),
                "target_price": _decimal(observation.plan.target_price),
                "stop_price": _decimal(observation.plan.stop_price),
                "taker_fee_rate": observation.taker_fee_rate,
                "entry_quote_observed_at": observation.entry_quote_observed_at,
                "entry_quote_spread_pct": observation.entry_quote_spread_pct,
                "estimated_margin_loss_at_stop_pct": observation.feasibility.estimated_margin_loss_at_stop_pct_before_fees,
            },
        )
        if created:
            recorded += 1
    return recorded


def _conservative_mean_return_lower_bound(returns: tuple[float, ...]) -> float:
    """Return a conservative 95% lower mean bound, or zero before the minimum sample exists."""

    if len(returns) < MIN_CONFIDENCE_RETURN_SAMPLES:
        return 0.0
    mean_return = sum(returns) / len(returns)
    sample_variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
    return mean_return - CONSERVATIVE_MEAN_RETURN_T_CRITICAL * sqrt(sample_variance / len(returns))


def _performance(records: tuple[IntradayLongSignal, ...]) -> ForwardPerformance:
    target_first = 0
    stop_first = 0
    unresolved = 0
    ambiguous = 0
    returns: list[float] = []
    worst_loss = 0.0
    for record in records:
        match SignalOutcome(record.status):
            case SignalOutcome.TARGET:
                target_first += 1
            case SignalOutcome.STOP:
                stop_first += 1
            case SignalOutcome.UNRESOLVED:
                unresolved += 1
            case SignalOutcome.AMBIGUOUS:
                ambiguous += 1
            case SignalOutcome.PENDING:
                raise AssertionError("pending signal cannot enter settled performance")
            case unreachable:
                assert_never(unreachable)
        if record.post_cost_margin_return_pct is not None:
            returns.append(record.post_cost_margin_return_pct)
        all_in_stop_loss = record.estimated_margin_loss_at_stop_pct + round_trip_margin_cost_pct(
            record.taker_fee_rate,
            record.entry_quote_spread_pct,
        )
        worst_loss = max(worst_loss, all_in_stop_loss)
    expected_return = sum(returns) / len(returns) if returns else 0.0
    return ForwardPerformance(
        settled_signal_count=len(records),
        target_first_count=target_first,
        stop_first_count=stop_first,
        unresolved_count=unresolved,
        ambiguous_count=ambiguous,
        post_cost_expected_margin_return_pct=expected_return,
        post_cost_margin_return_lower_bound_pct=_conservative_mean_return_lower_bound(tuple(returns)),
        worst_margin_loss_at_stop_pct=worst_loss,
    )


def promotion_for(profile: InstrumentProfile, contract_symbol: str, strategy_version: str) -> ForwardPromotion:
    """Evaluate only quoted, settled records for one immutable strategy, symbol, and profile."""

    records = tuple(
        IntradayLongSignal.objects.exclude(status=SignalOutcome.PENDING)
        .filter(
            profile=profile.value,
            contract_symbol=contract_symbol,
            strategy_version=strategy_version,
            entry_quote_observed_at__isnull=False,
        )
        .order_by("signal_opened_at")
    )
    boundary = len(records) * 4 // 5
    return ForwardPromotion(
        total=_performance(records),
        validation=_performance(records[:boundary]),
        held_out=_performance(records[boundary:]),
    )


def _metrics_clear(performance: ForwardPerformance) -> bool:
    return (
        performance.target_first_rate_pct >= MIN_TARGET_FIRST_RATE_PCT
        and performance.wilson_lower_bound_pct >= MIN_WILSON_LOWER_BOUND_PCT
        and performance.post_cost_expected_margin_return_pct > 0
        and performance.post_cost_margin_return_lower_bound_pct > 0
        and performance.worst_margin_loss_at_stop_pct <= MAX_WORST_MARGIN_LOSS_PCT
    )


def promotion_is_ready(performance: ForwardPerformance) -> bool:
    """Gate an aggregate result only when its sample is large enough for fixed-5x eligibility."""

    return performance.settled_signal_count >= MIN_SETTLED_SIGNALS and _metrics_clear(performance)


def promotion_evaluation_is_ready(evaluation: ForwardPromotion) -> bool:
    """Require the frozen profile to clear validation and the untouched chronological holdout."""

    return (
        promotion_is_ready(evaluation.total)
        and evaluation.validation.settled_signal_count >= MIN_VALIDATION_SIGNALS
        and evaluation.held_out.settled_signal_count >= MIN_HELD_OUT_SIGNALS
        and _metrics_clear(evaluation.validation)
        and _metrics_clear(evaluation.held_out)
    )
