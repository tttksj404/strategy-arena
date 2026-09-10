"""Pure calculations for the read-only Bitget multi-timeframe paper experiment."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Final, Mapping, Sequence

import numpy as np


LOOKBACK_BARS: Final = 40
REGIME_BARS: Final = 20
TOP_K: Final = 5
TARGET_VOLATILITY: Final = 0.08
MAX_LEVERAGE: Final = 5.0
MIN_QUOTE_VOLUME_USDT: Final = 5_000_000.0
MAX_SPREAD_FRACTION: Final = 0.001
MAX_STRESS_DISTANCE: Final = 0.08
LIQUIDATION_RESERVE: Final = 0.02
MIN_OOS_HOURS: Final = 168.0
MIN_OOS_RETURNS: Final = 30


@dataclass(frozen=True, slots=True)
class Candle:
    """A closed market candle parsed from a Bitget public response."""

    timestamp_ms: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    quote_volume_usdt: float


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Current public liquidity fields for one contract."""

    quote_volume_usdt: float
    spread_fraction: float


@dataclass(frozen=True, slots=True)
class Leg:
    """One long or short component of a delta-neutral paper book."""

    symbol: str
    weight: float


@dataclass(frozen=True, slots=True)
class ReturnMetrics:
    """Cost-adjusted return and drawdown over a contiguous walk-forward slice."""

    net_return: float
    max_drawdown: float


@dataclass(frozen=True, slots=True)
class IntervalResult:
    """One timeframe's read-only paper-trading verdict."""

    interval: str
    observation_hours: float
    oos_hours: float
    regime_on: bool
    leverage: float
    trend_fraction: float
    breadth_fraction: float
    stress_distance: float
    train: ReturnMetrics
    test: ReturnMetrics
    simulated_liquidations: int
    legs: tuple[Leg, ...]
    gate_reasons: tuple[str, ...]


def build_weights(signals: Mapping[str, float], top_k: int = TOP_K) -> Mapping[str, float]:
    """Return equal-weighted top/bottom momentum legs with zero net exposure."""

    if len(signals) < 2 * top_k:
        return {}
    ascending = sorted(signals, key=lambda symbol: (signals[symbol], symbol))
    short_symbols = ascending[:top_k]
    long_symbols = ascending[-top_k:]
    weight = 1.0 / top_k
    book = {symbol: weight for symbol in long_symbols}
    book.update({symbol: -weight for symbol in short_symbols})
    return book


def gate_reasons(
    *,
    regime_on: bool,
    stress_distance: float,
    candidate_symbols: Sequence[str],
    markets: Mapping[str, MarketSnapshot],
) -> tuple[str, ...]:
    """List entry blockers whose values come from predefined, not fitted, gates."""

    reasons: list[str] = []
    if not regime_on:
        reasons.append("regime_off")
    if stress_distance > MAX_STRESS_DISTANCE:
        reasons.append("stress_buffer")
    for symbol in candidate_symbols:
        market = markets.get(symbol)
        if market is None or market.quote_volume_usdt < MIN_QUOTE_VOLUME_USDT:
            reasons.append("liquidity")
            break
        if market.spread_fraction > MAX_SPREAD_FRACTION:
            reasons.append("spread")
            break
    return tuple(reasons)


def _align(
    candles_by_symbol: Mapping[str, Sequence[Candle]],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_symbol = {symbol: {candle.timestamp_ms: candle for candle in candles} for symbol, candles in candles_by_symbol.items()}
    symbols = tuple(sorted(by_symbol))
    common = set(by_symbol[symbols[0]])
    for symbol in symbols[1:]:
        common.intersection_update(by_symbol[symbol])
    timestamps = np.asarray(sorted(common), dtype=np.int64)
    close = np.asarray([[by_symbol[symbol][stamp].close_price for symbol in symbols] for stamp in timestamps], dtype=float)
    high = np.asarray([[by_symbol[symbol][stamp].high_price for symbol in symbols] for stamp in timestamps], dtype=float)
    low = np.asarray([[by_symbol[symbol][stamp].low_price for symbol in symbols] for stamp in timestamps], dtype=float)
    return symbols, timestamps, close, high, low


def _regime(close: np.ndarray, index: int) -> tuple[bool, float, float]:
    if index < REGIME_BARS:
        return False, 0.0, 0.0
    returns = close[index - REGIME_BARS + 1 : index + 1] / close[index - REGIME_BARS : index] - 1.0
    trend = prod(1.0 + returns.mean(axis=1)) - 1.0
    breadth = float((returns.mean(axis=0) > 0.0).mean())
    return bool(trend > 0.0 and breadth >= 0.6), float(trend), breadth


def _metrics(returns: np.ndarray) -> ReturnMetrics:
    if len(returns) == 0:
        return ReturnMetrics(net_return=0.0, max_drawdown=0.0)
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / peak
    return ReturnMetrics(net_return=float(equity[-1] - 1.0), max_drawdown=float(drawdown.max()))


def _walk_forward(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    markets: Sequence[MarketSnapshot],
) -> tuple[np.ndarray, np.ndarray, int]:
    returns: list[float] = []
    timestamps: list[int] = []
    raw_history: list[float] = []
    previous = np.zeros(close.shape[1])
    liquidations = 0
    for index in range(max(LOOKBACK_BARS, REGIME_BARS), len(close) - 1):
        regime_on, _, _ = _regime(close, index)
        signals = close[index] / close[index - LOOKBACK_BARS] - 1.0
        weights = np.zeros(close.shape[1])
        if regime_on:
            ranked = build_weights({str(item): float(signal) for item, signal in enumerate(signals)})
            for key, value in ranked.items():
                weights[int(key)] = value
        raw_return = float(np.dot(weights, close[index + 1] / close[index] - 1.0))
        raw_history.append(raw_return)
        realized = float(np.std(raw_history[-20:])) if len(raw_history) >= 20 else TARGET_VOLATILITY
        leverage = float(np.clip(TARGET_VOLATILITY / max(realized, 1e-9), 0.3, MAX_LEVERAGE)) if regime_on else 0.0
        liquidation_distance = 1.0 / leverage - LIQUIDATION_RESERVE if leverage else float("inf")
        long_adverse = (close[index] - low[index + 1]) / close[index]
        short_adverse = (high[index + 1] - close[index]) / close[index]
        touched = bool(((weights > 0.0) & (long_adverse >= liquidation_distance)).any()) or bool(
            ((weights < 0.0) & (short_adverse >= liquidation_distance)).any()
        )
        if touched:
            liquidations += 1
            returns.append(-1.0)
            timestamps.append(index + 1)
            break
        trade_cost = np.asarray([0.0006 + market.spread_fraction / 2.0 + 0.0002 for market in markets])
        turnover_cost = float(np.abs(weights - previous).dot(trade_cost)) * leverage
        returns.append(raw_return * leverage - turnover_cost)
        timestamps.append(index + 1)
        previous = weights
    return np.asarray(returns), np.asarray(timestamps, dtype=int), liquidations


def analyze_interval(
    *,
    interval: str,
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    markets: Mapping[str, MarketSnapshot],
) -> IntervalResult:
    """Apply a fixed, no-lookahead trend-and-volatility paper test to one timeframe."""

    symbols, timestamps, close, high, low = _align(candles_by_symbol)
    empty = ReturnMetrics(net_return=0.0, max_drawdown=0.0)
    if len(timestamps) < LOOKBACK_BARS + REGIME_BARS + 2:
        return IntervalResult(interval, 0.0, 0.0, False, 0.0, 0.0, 0.0, 0.0, empty, empty, 0, (), ("data_insufficient",))
    market_rows = tuple(markets.get(symbol, MarketSnapshot(0.0, float("inf"))) for symbol in symbols)
    walk_returns, return_indices, liquidations = _walk_forward(close, high, low, market_rows)
    cut = int(len(walk_returns) * 0.6)
    train, test = _metrics(walk_returns[:cut]), _metrics(walk_returns[cut:])
    current_regime, trend, breadth = _regime(close, len(close) - 1)
    current_signals = {symbol: float(close[-1, offset] / close[-1 - LOOKBACK_BARS, offset] - 1.0) for offset, symbol in enumerate(symbols)}
    weights = build_weights(current_signals)
    latest_raw = walk_returns[-20:] if len(walk_returns) >= 20 else walk_returns
    realized = float(np.std(latest_raw)) if len(latest_raw) else TARGET_VOLATILITY
    leverage = float(np.clip(TARGET_VOLATILITY / max(realized, 1e-9), 0.3, MAX_LEVERAGE)) if current_regime else 0.0
    selected_indices = [symbols.index(symbol) for symbol in weights]
    ranges = (high[-20:, selected_indices] - low[-20:, selected_indices]) / close[-20:, selected_indices] if selected_indices else np.zeros((1, 1))
    stress_distance = float(np.median(ranges, axis=0).max() * 3.0) if selected_indices else 0.0
    observation_hours = float((timestamps[-1] - timestamps[0]) / 3_600_000)
    oos_hours = float((timestamps[return_indices[-1]] - timestamps[return_indices[cut]]) / 3_600_000) if len(return_indices) > cut + 1 else 0.0
    reasons = list(gate_reasons(regime_on=current_regime, stress_distance=stress_distance, candidate_symbols=tuple(weights), markets=markets))
    if oos_hours < MIN_OOS_HOURS:
        reasons.append("oos_window_short")
    if len(walk_returns) - cut < MIN_OOS_RETURNS:
        reasons.append("oos_observations_short")
    if test.net_return <= 0.0:
        reasons.append("oos_not_positive")
    if test.max_drawdown > 0.25:
        reasons.append("oos_drawdown")
    if liquidations:
        reasons.append("simulated_liquidation")
    legs = tuple(Leg(symbol=symbol, weight=weight) for symbol, weight in sorted(weights.items()))
    return IntervalResult(
        interval=interval,
        observation_hours=observation_hours,
        oos_hours=oos_hours,
        regime_on=current_regime,
        leverage=leverage,
        trend_fraction=trend,
        breadth_fraction=breadth,
        stress_distance=stress_distance,
        train=train,
        test=test,
        simulated_liquidations=liquidations,
        legs=legs,
        gate_reasons=tuple(dict.fromkeys(reasons)),
    )
