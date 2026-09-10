"""Lagged long-only altcoin trend signals for a BTC-positive risk regime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import sqrt

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


class LongOnlyStyle(StrEnum):
    MOMENTUM = "long_only_momentum"
    MULTI_HORIZON = "long_only_multi_horizon_trend"
    BREAKOUT = "long_only_breakout"


@dataclass(frozen=True, slots=True)
class LongOnlyConfig:
    style: LongOnlyStyle
    short_lookback: int
    long_lookback: int
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"style={self.style.value}|lookbacks={self.short_lookback}/{self.long_lookback}|top_k={self.top_k}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _validate(data: MarketData, config: LongOnlyConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.short_lookback < 1 or config.long_lookback < config.short_lookback:
        raise SelectionBiasError("long-only signal requires ordered positive lookbacks")
    if config.top_k < 1 or config.rebalance_bars < 1 or config.minimum_quote_volume < 0.0:
        raise SelectionBiasError("long-only signal requires usable risk and liquidity parameters")
    if "BTC" not in data.symbols or len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("long-only signal requires BTC plus distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every long-only alt must exist in the panel") from error
    if config.top_k > len(indices):
        raise SelectionBiasError("long-only top-k exceeds the available alt cross section")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("long-only panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("long-only panel requires finite positive close prices")
    return indices


def btc_positive_regime_mask(
    close_price: np.ndarray,
    *,
    trend_bars: int,
    volatility_bars: int,
    minimum_trend_z: float,
    volatility_cap: float,
) -> np.ndarray:
    """Allow only a positive, completed BTC trend with bounded trailing volatility at bar *t*."""
    prices = np.asarray(close_price, dtype=float)
    if trend_bars < 1 or volatility_bars < 2 or minimum_trend_z < 0.0 or volatility_cap <= 0.0:
        raise SelectionBiasError("BTC positive-regime parameters are invalid")
    if prices.ndim != 1 or prices.size < max(trend_bars, volatility_bars) + 2 or not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise SelectionBiasError("BTC positive-regime input is not a finite positive price path")
    log_returns = np.zeros_like(prices)
    log_returns[1:] = np.log(prices[1:] / prices[:-1])
    mask = np.zeros(prices.size, dtype=bool)
    first = max(trend_bars, volatility_bars) + 1
    for index in range(first, prices.size):
        previous = index - 1
        sample = log_returns[previous - volatility_bars + 1 : previous + 1]
        volatility = float(sample.std(ddof=1))
        if volatility <= 0.0 or volatility > volatility_cap:
            continue
        trend = float(np.log(prices[previous] / prices[previous - trend_bars]))
        if trend / (volatility * sqrt(trend_bars)) >= minimum_trend_z:
            mask[index] = True
    return mask


def _long_positions(scores: np.ndarray, *, eligible: np.ndarray, top_k: int, indices: np.ndarray, width: int) -> np.ndarray:
    positions = np.zeros(width)
    valid = np.flatnonzero(eligible & np.isfinite(scores) & (scores > 0.0))
    if valid.size < top_k:
        return positions
    selected = valid[np.argsort(scores[valid], kind="stable")[-top_k:]]
    positions[indices[selected]] = 1.0 / top_k
    return positions


def long_only_positions(data: MarketData, config: LongOnlyConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Form long-only positions from values known at or before the prior completed bar."""
    alt_indices = _validate(data, config, alt_symbols)
    positions = np.zeros_like(data.close_price)
    first = config.long_lookback + 1
    for index in range(first, len(positions)):
        previous = index - 1
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        if index % config.rebalance_bars:
            positions[index] = positions[index - 1]
            positions[index, alt_indices[~liquid]] = 0.0
            continue
        short = np.log(data.close_price[previous, alt_indices] / data.close_price[previous - config.short_lookback, alt_indices])
        long = np.log(data.close_price[previous, alt_indices] / data.close_price[previous - config.long_lookback, alt_indices])
        if config.style is LongOnlyStyle.MOMENTUM:
            score = short
        elif config.style is LongOnlyStyle.MULTI_HORIZON:
            score = short + long
            liquid &= (short > 0.0) & (long > 0.0)
        else:
            prior_high = data.high_price[previous - config.long_lookback : previous, alt_indices].max(axis=0)
            score = np.log(data.close_price[previous, alt_indices] / prior_high)
        positions[index] = _long_positions(score, eligible=liquid, top_k=config.top_k, indices=alt_indices, width=len(data.symbols))
    return positions
