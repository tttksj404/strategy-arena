from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final, assert_never

from .long_forecast import Candle
from .market_timeframes import CandleGranularity


ATR_PERIOD: Final = 14
FAST_WINDOW: Final = 20
SLOW_WINDOW: Final = 50
BREAKOUT_WINDOW: Final = 20
MIN_HISTORY_BARS: Final = 60
STANDARD_HORIZON: Final = timedelta(minutes=60)
TRIPLE_LEVERAGED_HORIZON: Final = timedelta(minutes=15)
TARGET_ATR_MULTIPLIER: Final = 3.0
MINIMUM_VOLUME_RATIO: Final = 1.0
REQUESTED_LEVERAGE: Final = 5
MAX_MARGIN_LOSS_AT_STOP_PCT: Final = 8.0
SLIPPAGE_BUFFER_PRICE_PCT: Final = 0.05
# Require a nontrivial post-cost target after modeling the fixed 5x execution friction.
MIN_NET_MARGIN_GAIN_AT_TARGET_PCT: Final = 0.75
STRATEGY_VERSION: Final = "bitget_mtf_breakout_spread_5x_v3"


class InstrumentProfile(StrEnum):
    """Frozen intraday profiles with distinct trigger-timeframe requirements."""

    STANDARD = "standard_15m_5m"
    TRIPLE_LEVERAGED_ETF = "triple_leveraged_15m_5m_1m"

    @property
    def trigger_granularity(self) -> CandleGranularity:
        match self:
            case InstrumentProfile.STANDARD:
                return CandleGranularity.FIVE_MINUTES
            case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
                return CandleGranularity.ONE_MINUTE
            case unreachable:
                assert_never(unreachable)

    @property
    def horizon(self) -> timedelta:
        match self:
            case InstrumentProfile.STANDARD:
                return STANDARD_HORIZON
            case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
                return TRIPLE_LEVERAGED_HORIZON
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class IntradayCandleSet:
    """Only completed candles available when a multi-timeframe decision is made."""

    context_15m: tuple[Candle, ...]
    confirmation_5m: tuple[Candle, ...]
    trigger_1m: tuple[Candle, ...]


@dataclass(frozen=True, slots=True)
class IntradayLongPlan:
    """A causal intraday target and stop plan, never a promise of future return."""

    profile: InstrumentProfile
    trigger_granularity: CandleGranularity
    entry_price: float
    target_price: float
    stop_price: float
    score: int
    starts_at: datetime
    expires_at: datetime

    @property
    def target_return_pct(self) -> float:
        return (self.target_price / self.entry_price - 1) * 100

    @property
    def stop_return_pct(self) -> float:
        return (self.stop_price / self.entry_price - 1) * 100


@dataclass(frozen=True, slots=True)
class FixedLeverageFeasibility:
    """Public-data estimate of a plan's risk/cost envelope at the fixed 5x request."""

    passes: bool
    estimated_margin_loss_at_stop_pct_before_fees: float
    estimated_round_trip_margin_cost_pct: float
    estimated_net_margin_gain_at_target_pct: float
    reasons: tuple[str, ...]

    @property
    def estimated_all_in_margin_loss_at_stop_pct(self) -> float:
        """Return the modeled 5x stop loss including fixed round-trip execution costs."""

        return self.estimated_margin_loss_at_stop_pct_before_fees + self.estimated_round_trip_margin_cost_pct


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _atr(candles: tuple[Candle, ...]) -> float:
    recent = candles[-(ATR_PERIOD + 1) :]
    ranges = tuple(
        max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close))
        for previous, current in zip(recent, recent[1:])
    )
    return _mean(ranges)


def _is_context_uptrend(candles: tuple[Candle, ...]) -> bool:
    if len(candles) < MIN_HISTORY_BARS:
        return False
    closes = tuple(candle.close for candle in candles)
    return closes[-1] > _mean(closes[-FAST_WINDOW:]) > _mean(closes[-SLOW_WINDOW:]) and closes[-1] > closes[-6]


def _is_confirmation_long(candles: tuple[Candle, ...], minimum_volume_ratio: float) -> bool:
    if len(candles) < MIN_HISTORY_BARS:
        return False
    closes = tuple(candle.close for candle in candles)
    volumes = tuple(candle.quote_volume for candle in candles)
    return (
        closes[-1] > _mean(closes[-FAST_WINDOW:])
        and closes[-1] > closes[-13]
        and volumes[-1] >= _mean(volumes[-FAST_WINDOW:]) * minimum_volume_ratio
    )


def _is_trigger_long(candles: tuple[Candle, ...], minimum_volume_ratio: float) -> bool:
    if len(candles) < MIN_HISTORY_BARS:
        return False
    current = candles[-1]
    prior_high = max(candle.high for candle in candles[-(BREAKOUT_WINDOW + 1) : -1])
    average_volume = _mean(tuple(candle.quote_volume for candle in candles[-(BREAKOUT_WINDOW + 1) : -1]))
    return current.close >= prior_high and current.quote_volume >= average_volume * minimum_volume_ratio


def _trigger_candles(profile: InstrumentProfile, market: IntradayCandleSet) -> tuple[Candle, ...]:
    match profile:
        case InstrumentProfile.STANDARD:
            return market.confirmation_5m
        case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
            return market.trigger_1m
        case unreachable:
            assert_never(unreachable)


def plan_intraday_long(
    profile: InstrumentProfile,
    market: IntradayCandleSet,
    minimum_volume_ratio: float = MINIMUM_VOLUME_RATIO,
) -> IntradayLongPlan | None:
    """Build a plan only when every required completed timeframe agrees on long continuation."""

    if minimum_volume_ratio < MINIMUM_VOLUME_RATIO:
        return None
    trigger = _trigger_candles(profile, market)
    if not (
        _is_context_uptrend(market.context_15m)
        and _is_confirmation_long(market.confirmation_5m, minimum_volume_ratio)
    ):
        return None
    if not _is_trigger_long(trigger, minimum_volume_ratio):
        return None
    entry_price = trigger[-1].close
    average_true_range = _atr(trigger)
    if average_true_range <= 0:
        return None
    starts_at = trigger[-1].opened_at + profile.trigger_granularity.interval
    score = 80
    if entry_price > _mean(tuple(candle.close for candle in trigger[-FAST_WINDOW:])):
        score += 10
    if trigger[-1].quote_volume > _mean(tuple(candle.quote_volume for candle in trigger[-FAST_WINDOW:])):
        score += 10
    return IntradayLongPlan(
        profile=profile,
        trigger_granularity=profile.trigger_granularity,
        entry_price=entry_price,
        target_price=entry_price + average_true_range * TARGET_ATR_MULTIPLIER,
        stop_price=entry_price - average_true_range,
        score=score,
        starts_at=starts_at,
        expires_at=starts_at + profile.horizon,
    )


def round_trip_margin_cost_pct(taker_fee_rate: float, quote_spread_pct: float | None = None) -> float:
    """Model 5x taker fees plus the larger of the fixed impact floor and an observed quote spread."""

    observed_spread_pct = max(quote_spread_pct or 0.0, 0.0)
    execution_friction_pct = max(SLIPPAGE_BUFFER_PRICE_PCT, observed_spread_pct)
    return (taker_fee_rate * 2 * 100 + execution_friction_pct) * REQUESTED_LEVERAGE


def evaluate_fixed_leverage_feasibility(
    plan: IntradayLongPlan,
    taker_fee_rate: float,
    quote_spread_pct: float | None = None,
) -> FixedLeverageFeasibility:
    """Reject plans whose 5x loss envelope or post-cost target fails after actual public-quote friction."""

    stop_loss = abs(plan.stop_return_pct) * REQUESTED_LEVERAGE
    round_trip_cost = round_trip_margin_cost_pct(taker_fee_rate, quote_spread_pct)
    net_target_gain = plan.target_return_pct * REQUESTED_LEVERAGE - round_trip_cost
    reasons: list[str] = []
    if stop_loss + round_trip_cost > MAX_MARGIN_LOSS_AT_STOP_PCT:
        reasons.append("total_stop_loss_cap_exceeded")
    if net_target_gain < MIN_NET_MARGIN_GAIN_AT_TARGET_PCT:
        reasons.append("post_cost_target_too_small")
    return FixedLeverageFeasibility(
        passes=not reasons,
        estimated_margin_loss_at_stop_pct_before_fees=stop_loss,
        estimated_round_trip_margin_cost_pct=round_trip_cost,
        estimated_net_margin_gain_at_target_pct=net_target_gain,
        reasons=tuple(reasons),
    )
