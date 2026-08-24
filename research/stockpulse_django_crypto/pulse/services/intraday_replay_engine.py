from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from typing import Final, assert_never

from pulse.intraday_signal_models import SignalOutcome

from .intraday_long_forecast import (
    REQUESTED_LEVERAGE,
    SLIPPAGE_BUFFER_PRICE_PCT,
    InstrumentProfile,
    IntradayCandleSet,
    IntradayLongPlan,
    evaluate_fixed_leverage_feasibility,
    plan_intraday_long,
)
from .intraday_replay import ReplayEntryFilter, ReplayRule, ReplayTrade
from .long_forecast import Candle
from .market_timeframes import CandleGranularity
from .trend_continuation_contract import Direction, ResearchDecision
from .trend_continuation_features import (
    STANDARD_PROFILE,
    TECHNICAL_SCORE_RULE_NAME,
    TRIPLE_PROFILE,
    analyze_trend_continuation,
)


BOLLINGER_WINDOW: Final = 20
BOLLINGER_REFERENCE_WINDOWS: Final = 120
BOLLINGER_SQUEEZE_FRACTION: Final = 0.5


@dataclass(frozen=True, slots=True)
class ReplayMarket:
    """Completed Bitget candles needed to replay one contract without reading future candles as features."""

    contract_symbol: str
    profile: InstrumentProfile
    taker_fee_rate: float
    context_15m: tuple[Candle, ...]
    confirmation_5m: tuple[Candle, ...]
    trigger_1m: tuple[Candle, ...]
    settlement_1m: tuple[Candle, ...]


def _trigger_candles(market: ReplayMarket) -> tuple[Candle, ...]:
    match market.profile:
        case InstrumentProfile.STANDARD:
            return market.confirmation_5m
        case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
            return market.trigger_1m
        case unreachable:
            assert_never(unreachable)


def _trend_research_profile(profile: InstrumentProfile) -> str:
    """Map the fixed intraday profile to its existing research-only feature profile."""

    match profile:
        case InstrumentProfile.STANDARD:
            return STANDARD_PROFILE
        case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
            return TRIPLE_PROFILE
        case unreachable:
            assert_never(unreachable)


def _bollinger_width_pct(candles: tuple[Candle, ...]) -> float:
    """Return the causal 20-bar Bollinger width derived only from closed prices."""

    closes = tuple(candle.close for candle in candles[-BOLLINGER_WINDOW:])
    mean = sum(closes) / len(closes)
    if mean <= 0:
        return 0.0
    variance = sum((close - mean) ** 2 for close in closes) / len(closes)
    return 4 * sqrt(variance) / mean * 100


def _passes_bollinger_squeeze(profile: InstrumentProfile, replay_window: IntradayCandleSet) -> bool:
    """Require the breakout to emerge from a completed low-width regime."""

    match profile:
        case InstrumentProfile.STANDARD:
            candles = replay_window.confirmation_5m
        case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
            candles = replay_window.trigger_1m
        case unreachable:
            assert_never(unreachable)
    if len(candles) < BOLLINGER_WINDOW + BOLLINGER_REFERENCE_WINDOWS:
        return False
    widths = tuple(
        _bollinger_width_pct(candles[index - BOLLINGER_WINDOW : index])
        for index in range(BOLLINGER_WINDOW, len(candles) + 1)
    )
    reference_widths = tuple(sorted(widths[-(BOLLINGER_REFERENCE_WINDOWS + 1) : -1]))
    reference_median = reference_widths[len(reference_widths) // 2]
    return widths[-1] <= reference_median * BOLLINGER_SQUEEZE_FRACTION


def _passes_entry_filter(
    rule: ReplayRule,
    profile: InstrumentProfile,
    replay_window: IntradayCandleSet,
) -> bool:
    """Keep prior technical research as an optional causal filter, never a live instruction."""

    match rule.entry_filter:
        case ReplayEntryFilter.BREAKOUT:
            return True
        case ReplayEntryFilter.TECHNICAL_SCORE:
            match profile:
                case InstrumentProfile.STANDARD:
                    evidence_time = replay_window.confirmation_5m[-1].opened_at + profile.trigger_granularity.interval
                case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
                    evidence_time = replay_window.trigger_1m[-1].opened_at + profile.trigger_granularity.interval
                case unreachable:
                    assert_never(unreachable)
            decision = analyze_trend_continuation(
                profile=_trend_research_profile(profile),
                context_15m=replay_window.context_15m,
                confirmation_5m=replay_window.confirmation_5m,
                trigger_1m=replay_window.trigger_1m,
                evidence_time=evidence_time,
                rule_name=TECHNICAL_SCORE_RULE_NAME,
            )
            return (
                decision.decision is ResearchDecision.CONTINUATION_OBSERVE
                and decision.trend_direction is Direction.UP
            )
        case ReplayEntryFilter.BOLLINGER_SQUEEZE:
            return _passes_bollinger_squeeze(profile, replay_window)
        case unreachable:
            assert_never(unreachable)


def _completed_window(
    candles: tuple[Candle, ...],
    close_times: tuple[datetime, ...],
    cutoff: datetime,
) -> tuple[Candle, ...]:
    end = bisect_right(close_times, cutoff)
    return candles[max(0, end - 200) : end]


def _complete_future_window(
    settlement_candles: tuple[Candle, ...],
    settlement_starts: tuple[datetime, ...],
    plan: IntradayLongPlan,
    profile: InstrumentProfile,
) -> tuple[Candle, ...]:
    starts_at = bisect_left(settlement_starts, plan.starts_at)
    ends_at = bisect_left(settlement_starts, plan.expires_at)
    future = settlement_candles[starts_at:ends_at]
    expected_count = int(profile.horizon / CandleGranularity.ONE_MINUTE.interval)
    if len(future) != expected_count:
        return ()
    if any(current.opened_at - previous.opened_at != timedelta(minutes=1) for previous, current in zip(future, future[1:])):
        return ()
    return future


def _settle_plan(plan: IntradayLongPlan, future: tuple[Candle, ...], taker_fee_rate: float) -> tuple[SignalOutcome, float]:
    terminal_price = future[-1].close
    outcome = SignalOutcome.UNRESOLVED
    for candle in future:
        target_hit = candle.high >= plan.target_price
        stop_hit = candle.low <= plan.stop_price
        if target_hit and stop_hit:
            return SignalOutcome.AMBIGUOUS, _post_cost_return(plan.stop_price, plan, taker_fee_rate)
        if target_hit:
            return SignalOutcome.TARGET, _post_cost_return(plan.target_price, plan, taker_fee_rate)
        if stop_hit:
            return SignalOutcome.STOP, _post_cost_return(plan.stop_price, plan, taker_fee_rate)
    return outcome, _post_cost_return(terminal_price, plan, taker_fee_rate)


def _post_cost_return(terminal_price: float, plan: IntradayLongPlan, taker_fee_rate: float) -> float:
    raw_return_pct = (terminal_price / plan.entry_price - 1) * 100
    round_trip_cost_pct = (taker_fee_rate * 2 * 100 + SLIPPAGE_BUFFER_PRICE_PCT) * REQUESTED_LEVERAGE
    return raw_return_pct * REQUESTED_LEVERAGE - round_trip_cost_pct


def replay_market(market: ReplayMarket, rule: ReplayRule) -> tuple[ReplayTrade, ...]:
    """Replay a precommitted rule from completed multi-timeframe candles and settled one-minute outcomes."""

    context_close_times = tuple(candle.opened_at + CandleGranularity.FIFTEEN_MINUTES.interval for candle in market.context_15m)
    confirmation_close_times = tuple(
        candle.opened_at + CandleGranularity.FIVE_MINUTES.interval for candle in market.confirmation_5m
    )
    settlement_starts = tuple(candle.opened_at for candle in market.settlement_1m)
    trigger = _trigger_candles(market)
    trades: list[ReplayTrade] = []
    for trigger_index, trigger_candle in enumerate(trigger):
        decision_time = trigger_candle.opened_at + market.profile.trigger_granularity.interval
        replay_window = IntradayCandleSet(
            context_15m=_completed_window(market.context_15m, context_close_times, decision_time),
            confirmation_5m=_completed_window(market.confirmation_5m, confirmation_close_times, decision_time),
            trigger_1m=trigger[: trigger_index + 1],
        )
        plan = plan_intraday_long(
            market.profile,
            replay_window,
            minimum_volume_ratio=rule.minimum_volume_ratio,
        )
        if plan is None or not _passes_entry_filter(rule, market.profile, replay_window):
            continue
        feasibility = evaluate_fixed_leverage_feasibility(plan, market.taker_fee_rate)
        if not feasibility.passes:
            continue
        future = _complete_future_window(market.settlement_1m, settlement_starts, plan, market.profile)
        if not future:
            continue
        outcome, post_cost_return = _settle_plan(plan, future, market.taker_fee_rate)
        trades.append(
            ReplayTrade(
                rule=rule,
                contract_symbol=market.contract_symbol,
                signal_opened_at=trigger_candle.opened_at,
                outcome=outcome,
                post_cost_margin_return_pct=post_cost_return,
                margin_loss_at_stop_pct=feasibility.estimated_all_in_margin_loss_at_stop_pct,
            )
        )
    return tuple(trades)
