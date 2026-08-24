from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite, sqrt
from typing import Final

from .bitget_market import (
    BitgetContract,
    TextFetcher,
    fetch_completed_candles,
    fetch_historical_completed_candles,
    HistoricalCandleQuery,
)
from .long_forecast import Candle
from .market_timeframes import CandleGranularity
from .trend_continuation_contract import Direction, ResearchDecision


# allow: SIZE_OK - Task3 ownership requires v1 feature formulas, profile gates,
# and compatibility helpers to stay in this single module.
FEATURE_SCHEMA_VERSION: Final = "trend-continuation-features.v1"
BASELINE_RULE_NAME: Final = "baseline_mtf_v1"
TECHNICAL_SCORE_RULE_NAME: Final = "technical_score_v1"
STANDARD_PROFILE: Final = "standard_15m_5m"
TRIPLE_PROFILE: Final = "triple_leveraged_15m_5m_1m"
STANDARD_HORIZONS: Final = (6, 12, 18)
TRIPLE_HORIZONS: Final = (5, 10, 15)
DEFAULT_HORIZONS: Final = {
    STANDARD_PROFILE: STANDARD_HORIZONS,
    TRIPLE_PROFILE: TRIPLE_HORIZONS,
}
MIN_SCORE: Final = 70


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    name: str
    formula: str
    window: int
    timeframe: str
    value: float
    direction: Direction
    evidence_time: datetime
    complete: bool


@dataclass(frozen=True, slots=True)
class TimeframeFeatures:
    timeframe: CandleGranularity
    evidence_time: datetime
    closes: tuple[float, ...]
    candles: tuple[Candle, ...]
    sma20: float
    sma50: float
    sma20_slope: float
    ema12: float
    ema26: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    rsi14: float
    roc1: float
    atr14: float
    bollinger_width20: float
    relative_volume20: float
    adx14: float
    feature_records: tuple[FeatureRecord, ...]

    def trend_direction(self) -> Direction:
        close = self.closes[-1]
        if close > self.sma20 > self.sma50 and self.sma20_slope > 0:
            return Direction.UP
        if close < self.sma20 < self.sma50 and self.sma20_slope < 0:
            return Direction.DOWN
        return Direction.NEUTRAL


TrendContinuationFeatures = TimeframeFeatures


@dataclass(frozen=True, slots=True)
class TrendFeatureDecision:
    profile: str
    rule_name: str
    decision: ResearchDecision
    trend_direction: Direction
    candidate_horizon_bars: int
    candidate_horizons: tuple[int, ...]
    score: int
    invalidation_price: float | None
    reference_price: float | None
    evidence_time: datetime
    rejection_code: str | None
    features: tuple[FeatureRecord, ...]


def analyze_trend_continuation(
    *,
    profile: str,
    context_15m: tuple[Candle, ...],
    confirmation_5m: tuple[Candle, ...],
    trigger_1m: tuple[Candle, ...] = (),
    evidence_time: datetime,
    rule_name: str = TECHNICAL_SCORE_RULE_NAME,
) -> TrendFeatureDecision:
    profile_meta = _profile_meta(profile)
    match profile_meta:
        case None:
            return _no_trade(profile, rule_name, evidence_time, "unknown_profile")
        case (trigger_interval, candidate_horizons, default_horizon):
            pass

    prepared_15m = _prepare_candles(context_15m, CandleGranularity.FIFTEEN_MINUTES, evidence_time)
    prepared_5m = _prepare_candles(confirmation_5m, CandleGranularity.FIVE_MINUTES, evidence_time)
    prepared_1m = _prepare_candles(trigger_1m, CandleGranularity.ONE_MINUTE, evidence_time) if profile == TRIPLE_PROFILE else ()
    if isinstance(prepared_15m, str):
        return _no_trade(profile, rule_name, evidence_time, f"15m_{prepared_15m}", candidate_horizons, default_horizon)
    if isinstance(prepared_5m, str):
        return _no_trade(profile, rule_name, evidence_time, f"5m_{prepared_5m}", candidate_horizons, default_horizon)
    if isinstance(prepared_1m, str):
        return _no_trade(profile, rule_name, evidence_time, f"1m_{prepared_1m}", candidate_horizons, default_horizon)

    try:
        features_15m = build_features(prepared_15m, evidence_time, CandleGranularity.FIFTEEN_MINUTES)
        features_5m = build_features(prepared_5m, evidence_time, CandleGranularity.FIVE_MINUTES)
        features_1m = (
            build_features(prepared_1m, evidence_time, CandleGranularity.ONE_MINUTE)
            if profile == TRIPLE_PROFILE
            else None
        )
    except FeatureInputError as error:
        return _no_trade(profile, rule_name, evidence_time, str(error), candidate_horizons, default_horizon)

    direction = _aligned_direction(features_15m, features_5m, features_1m)
    if direction is Direction.NEUTRAL:
        return _decision(
            profile=profile,
            rule_name=rule_name,
            decision=ResearchDecision.NO_TRADE,
            direction=Direction.NEUTRAL,
            candidate_horizons=candidate_horizons,
            default_horizon=default_horizon,
            score=0,
            invalidation_price=None,
            reference_price=features_5m.closes[-1] if trigger_interval == CandleGranularity.FIVE_MINUTES else features_1m.closes[-1],
            evidence_time=evidence_time,
            rejection_code="direction_not_aligned",
            feature_groups=(features_15m, features_5m, features_1m),
        )

    trigger_features = features_5m if trigger_interval == CandleGranularity.FIVE_MINUTES else features_1m
    invalidation = _invalidation(trigger_features.candles, trigger_features.atr14, direction)
    if invalidation is None:
        return _decision(
            profile=profile,
            rule_name=rule_name,
            decision=ResearchDecision.NO_TRADE,
            direction=direction,
            candidate_horizons=candidate_horizons,
            default_horizon=default_horizon,
            score=0,
            invalidation_price=None,
            reference_price=trigger_features.closes[-1],
            evidence_time=evidence_time,
            rejection_code="missing_directional_invalidation",
            feature_groups=(features_15m, features_5m, features_1m),
        )
    if direction is Direction.UP and invalidation >= trigger_features.closes[-1]:
        return _no_trade(profile, rule_name, evidence_time, "invalid_up_invalidation", candidate_horizons, default_horizon)
    if direction is Direction.DOWN and invalidation <= trigger_features.closes[-1]:
        return _no_trade(profile, rule_name, evidence_time, "invalid_down_invalidation", candidate_horizons, default_horizon)

    score = _technical_score(features_15m, features_5m, features_1m, direction)
    baseline_pass = _baseline_mtf_pass(features_15m, features_5m, features_1m, direction)
    rule_pass = baseline_pass if rule_name == BASELINE_RULE_NAME else score >= MIN_SCORE
    return _decision(
        profile=profile,
        rule_name=rule_name,
        decision=ResearchDecision.CONTINUATION_OBSERVE if rule_pass else ResearchDecision.NO_TRADE,
        direction=direction,
        candidate_horizons=candidate_horizons,
        default_horizon=default_horizon,
        score=score,
        invalidation_price=invalidation,
        reference_price=trigger_features.closes[-1],
        evidence_time=evidence_time,
        rejection_code=None if rule_pass else "technical_gate_failed",
        feature_groups=(features_15m, features_5m, features_1m),
    )


def build_features(
    candles: tuple[Candle, ...],
    evidence_time: datetime,
    trigger_interval: CandleGranularity,
) -> TimeframeFeatures:
    prepared = _prepare_candles(candles, trigger_interval, evidence_time)
    if isinstance(prepared, str):
        raise FeatureInputError(prepared)
    closes = tuple(candle.close for candle in prepared)
    volumes = tuple(candle.quote_volume for candle in prepared)
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    sma20_slope = sma20 - _sma(closes[:-1], 20)
    ema12 = _ema_series(closes, 12)[-1]
    ema26 = _ema_series(closes, 26)[-1]
    macd_line, macd_signal, macd_hist = _macd(closes)
    rsi14 = _rsi(closes, 14)
    roc1 = _roc(closes, 1)
    atr14 = _atr(prepared, 14)
    bollinger_width20 = _bollinger_width(closes, 20)
    relative_volume20 = _relative_volume(volumes, 20)
    adx14 = _adx(prepared, 14)
    values = (
        ("SMA20", "mean(close[-20:])", 20, sma20),
        ("SMA50", "mean(close[-50:])", 50, sma50),
        ("SMA20_SLOPE", "SMA20(t)-SMA20(t-1)", 20, sma20_slope),
        ("EMA12", "seeded_SMA12_then_alpha_2_over_13", 12, ema12),
        ("EMA26", "seeded_SMA26_then_alpha_2_over_27", 26, ema26),
        ("MACD_LINE", "EMA12-EMA26", 26, macd_line),
        ("MACD_SIGNAL", "EMA9(MACD_LINE)", 9, macd_signal),
        ("MACD_HIST", "MACD_LINE-MACD_SIGNAL", 9, macd_hist),
        ("RSI14", "Wilder_RSI14", 14, rsi14),
        ("ROC1", "(close_t/close_t-1-1)*100", 1, roc1),
        ("ATR14", "Wilder_ATR14", 14, atr14),
        ("BOLLINGER_WIDTH20", "(upper-lower)/SMA20*100", 20, bollinger_width20),
        ("RVOL20", "volume_t/mean(volume_t-20_to_t-1)", 20, relative_volume20),
        ("ADX14", "Wilder_ADX14", 14, adx14),
    )
    timeframe = trigger_interval.value
    direction = _basic_direction(closes[-1], sma20, sma50, sma20_slope)
    return TimeframeFeatures(
        timeframe=trigger_interval,
        evidence_time=evidence_time,
        closes=closes,
        candles=prepared,
        sma20=sma20,
        sma50=sma50,
        sma20_slope=sma20_slope,
        ema12=ema12,
        ema26=ema26,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        rsi14=rsi14,
        roc1=roc1,
        atr14=atr14,
        bollinger_width20=bollinger_width20,
        relative_volume20=relative_volume20,
        adx14=adx14,
        feature_records=tuple(
            FeatureRecord(
                name=name,
                formula=formula,
                window=window,
                timeframe=timeframe,
                value=value,
                direction=direction,
                evidence_time=evidence_time,
                complete=True,
            )
            for name, formula, window, value in values
        ),
    )


def is_complete_candle(candle: Candle, now: datetime, trigger_interval: CandleGranularity) -> bool:
    return candle.opened_at + trigger_interval.interval <= now


def fetch_recent_candles(
    contract: BitgetContract,
    trigger_interval: CandleGranularity,
    fetch_text: TextFetcher,
    now: datetime,
    lookback_bars: int,
) -> tuple[Candle, ...]:
    raw = fetch_completed_candles(contract, trigger_interval, fetch_text, now)
    return raw[-lookback_bars:]


def fetch_window_history(
    contract: BitgetContract,
    trigger_interval: CandleGranularity,
    starts_at: datetime,
    ends_at: datetime,
    fetch_text: TextFetcher,
) -> tuple[Candle, ...]:
    return fetch_historical_completed_candles(
        HistoricalCandleQuery(
            contract=contract,
            granularity=trigger_interval,
            starts_at=starts_at,
            ends_at=ends_at,
        ),
        fetch_text,
    )


class FeatureInputError(ValueError):
    """Raised when point-in-time feature inputs cannot satisfy the v1 contract."""


def _prepare_candles(
    candles: tuple[Candle, ...],
    interval: CandleGranularity,
    evidence_time: datetime,
) -> tuple[Candle, ...] | str:
    if evidence_time.tzinfo is None:
        return "naive_evidence_time"
    filtered = tuple(candle for candle in candles if candle.opened_at + interval.interval <= evidence_time)
    if len(filtered) != len(candles):
        return "open_or_future_candle"
    if len(filtered) < 60:
        return "insufficient_lookback"
    previous_open: datetime | None = None
    seen: set[datetime] = set()
    for candle in filtered:
        if candle.opened_at.tzinfo is None:
            return "naive_candle_time"
        if candle.opened_at in seen:
            return "duplicate_candle"
        if previous_open is not None and candle.opened_at <= previous_open:
            return "nonascending_candle"
        if previous_open is not None and candle.opened_at - previous_open != interval.interval:
            return "noncontinuous_candle"
        if min(candle.open, candle.high, candle.low, candle.close, candle.quote_volume) <= 0:
            return "zero_or_negative_value"
        seen.add(candle.opened_at)
        previous_open = candle.opened_at
    return filtered


def _profile_meta(profile: str) -> tuple[CandleGranularity, tuple[int, ...], int] | None:
    if profile == STANDARD_PROFILE:
        return CandleGranularity.FIVE_MINUTES, STANDARD_HORIZONS, 12
    if profile == TRIPLE_PROFILE:
        return CandleGranularity.ONE_MINUTE, TRIPLE_HORIZONS, 10
    return None


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _sma(values: tuple[float, ...], period: int) -> float:
    if len(values) < period:
        raise FeatureInputError("insufficient_lookback")
    denominator = period
    if denominator == 0:
        raise FeatureInputError("zero_denominator")
    return _mean(values[-period:])


def _ema_series(values: tuple[float, ...], period: int) -> tuple[float, ...]:
    if len(values) < period:
        raise FeatureInputError("insufficient_lookback")
    alpha = 2 / (period + 1)
    ema = _mean(values[:period])
    series = [ema]
    for value in values[period:]:
        ema = (value - ema) * alpha + ema
        series.append(ema)
    return tuple(series)


def _macd(closes: tuple[float, ...]) -> tuple[float, float, float]:
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    offset = len(ema12) - len(ema26)
    macd_series = tuple(fast - slow for fast, slow in zip(ema12[offset:], ema26))
    if len(macd_series) < 9:
        raise FeatureInputError("insufficient_macd_signal")
    signal = _ema_series(macd_series, 9)[-1]
    line = macd_series[-1]
    return line, signal, line - signal


def _rsi(closes: tuple[float, ...], period: int) -> float:
    if len(closes) <= period:
        raise FeatureInputError("insufficient_lookback")
    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(closes, closes[1:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    average_gain = _mean(tuple(gains[:period]))
    average_loss = _mean(tuple(losses[:period]))
    for gain, loss in zip(gains[period:], losses[period:]):
        average_gain = ((period - 1) * average_gain + gain) / period
        average_loss = ((period - 1) * average_loss + loss) / period
    if average_loss == 0:
        return 100.0
    return 100 - 100 / (1 + average_gain / average_loss)


def _atr(candles: tuple[Candle, ...], period: int) -> float:
    if len(candles) <= period:
        raise FeatureInputError("insufficient_lookback")
    ranges = tuple(
        max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close))
        for previous, current in zip(candles, candles[1:])
    )
    return _wilder(ranges, period)


def _adx(candles: tuple[Candle, ...], period: int) -> float:
    if len(candles) < (period * 2) + 1:
        raise FeatureInputError("insufficient_adx_lookback")
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr: list[float] = []
    for previous, current in zip(candles, candles[1:]):
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr.append(max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close)))
    smoothed_tr = _mean(tuple(tr[:period]))
    smoothed_plus = _mean(tuple(plus_dm[:period]))
    smoothed_minus = _mean(tuple(minus_dm[:period]))
    dx_values: list[float] = []
    for index in range(period, len(tr)):
        smoothed_tr = ((period - 1) * smoothed_tr + tr[index]) / period
        smoothed_plus = ((period - 1) * smoothed_plus + plus_dm[index]) / period
        smoothed_minus = ((period - 1) * smoothed_minus + minus_dm[index]) / period
        if smoothed_tr == 0:
            raise FeatureInputError("zero_denominator")
        plus_di = 100 * smoothed_plus / smoothed_tr
        minus_di = 100 * smoothed_minus / smoothed_tr
        denominator = plus_di + minus_di
        if denominator == 0:
            raise FeatureInputError("zero_denominator")
        dx_values.append(100 * abs(plus_di - minus_di) / denominator)
    if len(dx_values) < period:
        raise FeatureInputError("insufficient_adx_lookback")
    adx = _mean(tuple(dx_values[:period]))
    for dx in dx_values[period:]:
        adx = ((period - 1) * adx + dx) / period
    return adx


def _wilder(values: tuple[float, ...], period: int) -> float:
    if len(values) < period:
        raise FeatureInputError("insufficient_lookback")
    current = _mean(values[:period])
    for value in values[period:]:
        current = ((period - 1) * current + value) / period
    return current


def _roc(closes: tuple[float, ...], period: int) -> float:
    base = closes[-period - 1]
    if base == 0:
        raise FeatureInputError("zero_denominator")
    return (closes[-1] / base - 1) * 100


def _bollinger_width(closes: tuple[float, ...], period: int) -> float:
    window = closes[-period:]
    mean = _mean(window)
    if mean == 0:
        raise FeatureInputError("zero_denominator")
    sd = sqrt(sum((value - mean) ** 2 for value in window) / period)
    return (4 * sd) / mean * 100


def _relative_volume(volumes: tuple[float, ...], period: int) -> float:
    previous = volumes[-(period + 1) : -1]
    mean = _mean(previous)
    if mean <= 0:
        raise FeatureInputError("zero_denominator")
    return volumes[-1] / mean


def _basic_direction(close: float, sma20: float, sma50: float, sma20_slope: float) -> Direction:
    if close > sma20 > sma50 and sma20_slope > 0:
        return Direction.UP
    if close < sma20 < sma50 and sma20_slope < 0:
        return Direction.DOWN
    return Direction.NEUTRAL


def _aligned_direction(
    features_15m: TimeframeFeatures,
    features_5m: TimeframeFeatures,
    features_1m: TimeframeFeatures | None,
) -> Direction:
    for direction in (Direction.UP, Direction.DOWN):
        if _timeframe_gate(features_15m, direction) and _timeframe_gate(features_5m, direction):
            if features_1m is None or _triple_trigger_gate(features_1m, direction):
                return direction
    return Direction.NEUTRAL


def _timeframe_gate(features: TimeframeFeatures, direction: Direction) -> bool:
    if direction is Direction.UP:
        return features.closes[-1] > features.sma20 > features.sma50 and features.sma20_slope > 0
    if direction is Direction.DOWN:
        return features.closes[-1] < features.sma20 < features.sma50 and features.sma20_slope < 0
    return False


def _baseline_mtf_pass(
    features_15m: TimeframeFeatures,
    features_5m: TimeframeFeatures,
    features_1m: TimeframeFeatures | None,
    direction: Direction,
) -> bool:
    if not (_timeframe_gate(features_15m, direction) and _confirmation_gate(features_5m, direction)):
        return False
    return features_1m is None or _triple_trigger_gate(features_1m, direction)


def _confirmation_gate(features: TimeframeFeatures, direction: Direction) -> bool:
    if direction is Direction.UP:
        return _timeframe_gate(features, direction) and 50 <= features.rsi14 <= 70 and features.roc1 > 0
    if direction is Direction.DOWN:
        return _timeframe_gate(features, direction) and 30 <= features.rsi14 <= 50 and features.roc1 < 0
    return False


def _triple_trigger_gate(features: TimeframeFeatures, direction: Direction) -> bool:
    if direction is Direction.UP:
        return features.closes[-1] > features.sma20 and features.relative_volume20 >= 1.0
    if direction is Direction.DOWN:
        return features.closes[-1] < features.sma20 and features.relative_volume20 >= 1.0
    return False


def _technical_score(
    features_15m: TimeframeFeatures,
    features_5m: TimeframeFeatures,
    features_1m: TimeframeFeatures | None,
    direction: Direction,
) -> int:
    score = 0
    if _timeframe_gate(features_15m, direction) and _timeframe_gate(features_5m, direction):
        score += 30
    if (direction is Direction.UP and 50 <= features_5m.rsi14 <= 70) or (
        direction is Direction.DOWN and 30 <= features_5m.rsi14 <= 50
    ):
        score += 10
    if _signed(features_5m.macd_line - features_5m.macd_signal, direction) > 0:
        score += 10
    if _signed(features_5m.roc1, direction) > 0:
        score += 15
    if features_5m.relative_volume20 >= 1.0:
        score += 15
    atr_pct = features_5m.atr14 / features_5m.closes[-1] * 100
    if 0.10 <= atr_pct <= 8.00:
        score += 10
    if features_5m.adx14 >= 20:
        score += 10
    if features_1m is not None and not _triple_trigger_gate(features_1m, direction):
        return 0
    return score


def _signed(value: float, direction: Direction) -> float:
    if direction is Direction.UP:
        return value
    if direction is Direction.DOWN:
        return -value
    return 0.0


def _invalidation(candles: tuple[Candle, ...], atr14: float, direction: Direction) -> float | None:
    reference = candles[-1].close
    if direction is Direction.UP:
        lows = tuple(candle.low for candle in candles)
        for index in range(len(candles) - 3, 1, -1):
            low = lows[index]
            if low < min(lows[index - 2], lows[index - 1], lows[index + 1], lows[index + 2]):
                invalidation = low - 0.25 * atr14
                return invalidation if invalidation < reference else None
    if direction is Direction.DOWN:
        highs = tuple(candle.high for candle in candles)
        for index in range(len(candles) - 3, 1, -1):
            high = highs[index]
            if high > max(highs[index - 2], highs[index - 1], highs[index + 1], highs[index + 2]):
                invalidation = high + 0.25 * atr14
                return invalidation if invalidation > reference else None
    return None


def _decision(
    *,
    profile: str,
    rule_name: str,
    decision: ResearchDecision,
    direction: Direction,
    candidate_horizons: tuple[int, ...],
    default_horizon: int,
    score: int,
    invalidation_price: float | None,
    reference_price: float | None,
    evidence_time: datetime,
    rejection_code: str | None,
    feature_groups: tuple[TimeframeFeatures | None, ...],
) -> TrendFeatureDecision:
    return TrendFeatureDecision(
        profile=profile,
        rule_name=rule_name,
        decision=decision,
        trend_direction=direction,
        candidate_horizon_bars=default_horizon,
        candidate_horizons=candidate_horizons,
        score=score,
        invalidation_price=_finite_or_none(invalidation_price),
        reference_price=_finite_or_none(reference_price),
        evidence_time=evidence_time,
        rejection_code=rejection_code,
        features=tuple(record for group in feature_groups if group is not None for record in group.feature_records),
    )


def _no_trade(
    profile: str,
    rule_name: str,
    evidence_time: datetime,
    rejection_code: str,
    candidate_horizons: tuple[int, ...] = STANDARD_HORIZONS,
    default_horizon: int = 12,
) -> TrendFeatureDecision:
    return TrendFeatureDecision(
        profile=profile,
        rule_name=rule_name,
        decision=ResearchDecision.NO_TRADE,
        trend_direction=Direction.NEUTRAL,
        candidate_horizon_bars=default_horizon,
        candidate_horizons=candidate_horizons,
        score=0,
        invalidation_price=None,
        reference_price=None,
        evidence_time=evidence_time,
        rejection_code=rejection_code,
        features=(),
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if isfinite(value) else None
