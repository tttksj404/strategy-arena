from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Callable, Final, TypeAlias


BAR_INTERVAL: Final = timedelta(hours=4)
ATR_PERIOD: Final = 14
FAST_WINDOW: Final = 20
SLOW_WINDOW: Final = 50
RSI_PERIOD: Final = 14
BREAKOUT_WINDOW: Final = 20
MIN_HISTORY_BARS: Final = 60
HORIZON_BARS: Final = 12
TARGET_ATR_MULTIPLIER: Final = 2.0
STOP_ATR_MULTIPLIER: Final = 1.0
MIN_TARGET_HIT_RATE_PCT: Final = 55.0
MIN_BASELINE_UPLIFT_PCT: Final = 5.0
MIN_PROMOTION_SIGNALS: Final = 10


@dataclass(frozen=True, slots=True)
class Candle:
    """One completed 4-hour market candle parsed at the external boundary."""
    opened_at: datetime
    open: float
    high: float
    low: float
    close: float
    quote_volume: float


@dataclass(frozen=True, slots=True)
class Forecast:
    """A causal target/stop band, not a claim of an exact future price."""
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
class EvaluationMetrics:
    """Outcomes for one fixed strategy on one chronological segment."""
    signal_count: int
    target_hits: int
    stop_hits: int
    unresolved_count: int
    ambiguous_count: int
    mean_terminal_return_pct: float

    @property
    def target_hit_rate_pct(self) -> float:
        if self.signal_count == 0:
            return 0.0
        return self.target_hits / self.signal_count * 100


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    trend: EvaluationMetrics
    baseline: EvaluationMetrics


@dataclass(frozen=True, slots=True)
class PromotionEvaluation:
    validation: SplitEvaluation
    held_out: SplitEvaluation


class Outcome(StrEnum):
    TARGET = "target"
    STOP = "stop"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


ForecastBuilder: TypeAlias = Callable[[tuple[Candle, ...]], Forecast | None]


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _atr(candles: tuple[Candle, ...]) -> float:
    recent = candles[-(ATR_PERIOD + 1) :]
    ranges = tuple(
        max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close))
        for previous, current in zip(recent, recent[1:])
    )
    return _mean(ranges)


def _rsi(closes: tuple[float, ...]) -> float:
    changes = tuple(current - previous for previous, current in zip(closes[-(RSI_PERIOD + 1) :], closes[-RSI_PERIOD:]))
    average_gain = _mean(tuple(max(change, 0.0) for change in changes))
    average_loss = _mean(tuple(max(-change, 0.0) for change in changes))
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def _plan(candles: tuple[Candle, ...], score: int) -> Forecast:
    entry_price = candles[-1].close
    average_true_range = _atr(candles)
    starts_at = candles[-1].opened_at + BAR_INTERVAL
    return Forecast(
        entry_price=entry_price,
        target_price=entry_price + average_true_range * TARGET_ATR_MULTIPLIER,
        stop_price=entry_price - average_true_range * STOP_ATR_MULTIPLIER,
        score=score,
        starts_at=starts_at,
        expires_at=starts_at + BAR_INTERVAL * HORIZON_BARS,
    )


def unconditional_long_forecast(candles: tuple[Candle, ...]) -> Forecast | None:
    """Return the fixed no-selection baseline using the same target/stop geometry."""
    if len(candles) < MIN_HISTORY_BARS:
        return None
    return _plan(candles, score=0)


def trend_atr_forecast(candles: tuple[Candle, ...]) -> Forecast | None:
    """Emit a fixed long signal only when causal trend and risk conditions agree."""
    if len(candles) < MIN_HISTORY_BARS:
        return None
    closes = tuple(candle.close for candle in candles)
    quote_volumes = tuple(candle.quote_volume for candle in candles)
    current_price = closes[-1]
    fast_average = _mean(closes[-FAST_WINDOW:])
    slow_average = _mean(closes[-SLOW_WINDOW:])
    momentum = current_price / closes[-13] - 1
    rsi = _rsi(closes)
    prior_high = max(candle.high for candle in candles[-(BREAKOUT_WINDOW + 1) : -1])
    prior_volume = _mean(quote_volumes[-(BREAKOUT_WINDOW + 1) : -1])
    volatility_pct = _atr(candles) / current_price * 100
    score = 0
    if current_price > fast_average > slow_average:
        score += 35
    if momentum > 0:
        score += 25
    if 45 <= rsi <= 68:
        score += 15
    if current_price >= prior_high:
        score += 15
    if prior_volume > 0 and quote_volumes[-1] >= prior_volume:
        score += 10
    if score < 70 or not 0.30 <= volatility_pct <= 4.00:
        return None
    return _plan(candles, score=score)


def _outcome(forecast: Forecast, future: tuple[Candle, ...]) -> Outcome:
    for candle in future:
        target_hit = candle.high >= forecast.target_price
        stop_hit = candle.low <= forecast.stop_price
        if target_hit and stop_hit:
            return Outcome.AMBIGUOUS
        if target_hit:
            return Outcome.TARGET
        if stop_hit:
            return Outcome.STOP
    return Outcome.UNRESOLVED


def evaluate_strategy(
    candles: tuple[Candle, ...],
    builder: ForecastBuilder,
    start_index: int,
    end_index: int,
) -> EvaluationMetrics:
    """Score only windows whose future candles are strictly after the signal bar."""
    target_hits = 0
    stop_hits = 0
    unresolved_count = 0
    ambiguous_count = 0
    terminal_returns: list[float] = []
    for index in range(start_index, end_index):
        forecast = builder(candles[: index + 1])
        if forecast is None:
            continue
        future = candles[index + 1 : index + 1 + HORIZON_BARS]
        terminal_returns.append((future[-1].close / forecast.entry_price - 1) * 100)
        match _outcome(forecast, future):
            case Outcome.TARGET:
                target_hits += 1
            case Outcome.STOP:
                stop_hits += 1
            case Outcome.UNRESOLVED:
                unresolved_count += 1
            case Outcome.AMBIGUOUS:
                ambiguous_count += 1
            case unreachable:
                raise AssertionError(f"unexpected outcome: {unreachable}")
    signal_count = target_hits + stop_hits + unresolved_count + ambiguous_count
    mean_return = _mean(tuple(terminal_returns)) if terminal_returns else 0.0
    return EvaluationMetrics(
        signal_count=signal_count,
        target_hits=target_hits,
        stop_hits=stop_hits,
        unresolved_count=unresolved_count,
        ambiguous_count=ambiguous_count,
        mean_terminal_return_pct=mean_return,
    )


def evaluate_promotion(candles: tuple[Candle, ...]) -> PromotionEvaluation:
    """Reserve the final chronological fifth of eligible observations as untouched holdout."""
    first_signal_index = MIN_HISTORY_BARS - 1
    last_signal_index = len(candles) - HORIZON_BARS
    eligible_count = max(last_signal_index - first_signal_index, 0)
    validation_start = first_signal_index + eligible_count * 3 // 5
    held_out_start = first_signal_index + eligible_count * 4 // 5
    validation = SplitEvaluation(
        trend=evaluate_strategy(candles, trend_atr_forecast, validation_start, held_out_start),
        baseline=evaluate_strategy(candles, unconditional_long_forecast, validation_start, held_out_start),
    )
    held_out = SplitEvaluation(
        trend=evaluate_strategy(candles, trend_atr_forecast, held_out_start, last_signal_index),
        baseline=evaluate_strategy(candles, unconditional_long_forecast, held_out_start, last_signal_index),
    )
    return PromotionEvaluation(validation=validation, held_out=held_out)


def _split_passes(evaluation: SplitEvaluation) -> bool:
    uplift = evaluation.trend.target_hit_rate_pct - evaluation.baseline.target_hit_rate_pct
    return (
        evaluation.trend.signal_count >= MIN_PROMOTION_SIGNALS
        and evaluation.trend.target_hit_rate_pct >= MIN_TARGET_HIT_RATE_PCT
        and uplift >= MIN_BASELINE_UPLIFT_PCT
    )


def promotion_is_reliable(validation: SplitEvaluation, held_out: SplitEvaluation) -> bool:
    """Require the fixed rule to clear both validation and untouched-holdout gates."""
    return _split_passes(validation) and _split_passes(held_out)
