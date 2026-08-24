from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, assert_never

from .bitget_ticker import BitgetTicker
from .intraday_long_forecast import (
    FixedLeverageFeasibility,
    MAX_MARGIN_LOSS_AT_STOP_PCT,
    MIN_NET_MARGIN_GAIN_AT_TARGET_PCT,
    REQUESTED_LEVERAGE,
    IntradayLongPlan,
    evaluate_fixed_leverage_feasibility,
    round_trip_margin_cost_pct,
)


MAX_ENTRY_DRIFT_FROM_TRIGGER_PCT: Final = 0.10
PARTIAL_EXIT_R_MULTIPLIER: Final = 1.5


class LiveEntryState(StrEnum):
    """Every possible manual response to a live quote for a previously confirmed plan."""

    ENTER_NOW = "enter_now"
    WAIT_FOR_ENTRY_ZONE = "wait_for_entry_zone"
    SKIP_CHASED = "skip_chased"
    SKIP_TRIGGER_STALE = "skip_trigger_stale"
    SKIP_EXPIRED = "skip_expired"
    SKIP_INVALIDATED = "skip_invalidated"
    SKIP_LIQUIDITY_GATE = "skip_liquidity_gate"
    SKIP_RISK_GATE = "skip_risk_gate"


@dataclass(frozen=True, slots=True)
class LiveLongInstruction:
    """Concrete manual entry and exit levels for one fresh quote; it never guarantees profit."""

    state: LiveEntryState
    quote: BitgetTicker
    entry_price: float
    entry_floor_price: float
    entry_ceiling_price: float
    stop_price: float
    partial_take_profit_price: float
    final_take_profit_price: float
    expires_at: datetime
    feasibility: FixedLeverageFeasibility
    reasons: tuple[str, ...]

    @property
    def is_enter_now(self) -> bool:
        return self.state is LiveEntryState.ENTER_NOW

    @property
    def action(self) -> str:
        match self.state:
            case LiveEntryState.ENTER_NOW:
                return "ENTER_LONG"
            case LiveEntryState.WAIT_FOR_ENTRY_ZONE:
                return "DO_NOT_ENTER_YET"
            case (
                LiveEntryState.SKIP_CHASED
                | LiveEntryState.SKIP_TRIGGER_STALE
                | LiveEntryState.SKIP_EXPIRED
                | LiveEntryState.SKIP_INVALIDATED
                | LiveEntryState.SKIP_LIQUIDITY_GATE
                | LiveEntryState.SKIP_RISK_GATE
            ):
                return "DO_NOT_ENTER"
            case unreachable:
                assert_never(unreachable)


def _entry_floor(plan: IntradayLongPlan) -> float:
    return max(plan.stop_price, plan.entry_price * (1 - MAX_ENTRY_DRIFT_FROM_TRIGGER_PCT / 100))


def _entry_ceiling(plan: IntradayLongPlan, taker_fee_rate: float, quote_spread_pct: float) -> float:
    """Keep the planned trigger, 5x stop cap, and post-cost target gate true at any permitted entry."""

    signal_ceiling = plan.entry_price * (1 + MAX_ENTRY_DRIFT_FROM_TRIGGER_PCT / 100)
    stop_ratio = MAX_MARGIN_LOSS_AT_STOP_PCT / (100 * REQUESTED_LEVERAGE)
    stop_ceiling = plan.stop_price / (1 - stop_ratio)
    target_ratio = (
        MIN_NET_MARGIN_GAIN_AT_TARGET_PCT + round_trip_margin_cost_pct(taker_fee_rate, quote_spread_pct)
    ) / (100 * REQUESTED_LEVERAGE)
    target_ceiling = plan.target_price / (1 + target_ratio)
    return min(signal_ceiling, stop_ceiling, target_ceiling)


def _state_for(
    plan: IntradayLongPlan,
    quote: BitgetTicker,
    feasibility: FixedLeverageFeasibility,
    now: datetime,
    floor_price: float,
    ceiling_price: float,
) -> tuple[LiveEntryState, tuple[str, ...]]:
    if now < plan.starts_at:
        return LiveEntryState.WAIT_FOR_ENTRY_ZONE, ("trigger_window_not_open",)
    if now >= plan.expires_at:
        return LiveEntryState.SKIP_EXPIRED, ("signal_window_expired",)
    if now >= plan.starts_at + plan.trigger_granularity.interval:
        return LiveEntryState.SKIP_TRIGGER_STALE, ("closed_trigger_stale",)
    if quote.ask_price <= plan.stop_price:
        return LiveEntryState.SKIP_INVALIDATED, ("quote_at_or_below_stop",)
    if floor_price > ceiling_price:
        return LiveEntryState.SKIP_RISK_GATE, ("entry_range_collapsed_by_actual_execution_cost",)
    if quote.ask_price < floor_price:
        if not feasibility.passes:
            return LiveEntryState.SKIP_RISK_GATE, feasibility.reasons
        return LiveEntryState.WAIT_FOR_ENTRY_ZONE, ("quote_below_entry_range",)
    if quote.ask_price > ceiling_price:
        return LiveEntryState.SKIP_CHASED, ("quote_above_no_chase_ceiling",)
    if not feasibility.passes:
        return LiveEntryState.SKIP_RISK_GATE, feasibility.reasons
    return LiveEntryState.ENTER_NOW, ()


def build_live_long_instruction(
    plan: IntradayLongPlan,
    quote: BitgetTicker,
    taker_fee_rate: float,
    now: datetime | None = None,
) -> LiveLongInstruction:
    """Turn a fresh public ask into an exact manual entry/exit instruction within the fixed 5x risk envelope."""

    reference_time = now or datetime.now(UTC)
    repriced_plan = replace(plan, entry_price=quote.ask_price)
    feasibility = evaluate_fixed_leverage_feasibility(
        repriced_plan,
        taker_fee_rate,
        quote_spread_pct=quote.bid_ask_spread_pct,
    )
    floor_price = _entry_floor(plan)
    ceiling_price = _entry_ceiling(plan, taker_fee_rate, quote.bid_ask_spread_pct)
    state, reasons = _state_for(plan, quote, feasibility, reference_time, floor_price, ceiling_price)
    partial_take_profit_price = min(
        plan.target_price,
        quote.ask_price + (quote.ask_price - plan.stop_price) * PARTIAL_EXIT_R_MULTIPLIER,
    )
    return LiveLongInstruction(
        state=state,
        quote=quote,
        entry_price=quote.ask_price,
        entry_floor_price=floor_price,
        entry_ceiling_price=ceiling_price,
        stop_price=plan.stop_price,
        partial_take_profit_price=partial_take_profit_price,
        final_take_profit_price=plan.target_price,
        expires_at=plan.expires_at,
        feasibility=feasibility,
        reasons=reasons,
    )
