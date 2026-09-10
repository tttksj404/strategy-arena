"""Lagged composite trend signals with raw and BTC-residual variants for paper research."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


class CompositeStyle(StrEnum):
    MULTI_HORIZON = "multi_horizon_trend"
    MULTI_SIGNAL = "multi_signal_trend"
    BTC_RESIDUAL_MULTI_SIGNAL = "btc_residual_multi_signal_trend"


@dataclass(frozen=True, slots=True)
class CTrendConfig:
    style: CompositeStyle
    short_lookback: int
    medium_lookback: int
    long_lookback: int
    volatility_window: int
    beta_window: int
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"style={self.style.value}|lookbacks={self.short_lookback}/{self.medium_lookback}/{self.long_lookback}"
            f"|volatility_window={self.volatility_window}|beta_window={self.beta_window}|top_k={self.top_k}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _validate(data: MarketData, config: CTrendConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if not (1 <= config.short_lookback <= config.medium_lookback <= config.long_lookback):
        raise SelectionBiasError("composite trend requires ordered positive lookbacks")
    if config.volatility_window < 2 or config.beta_window < 2 or config.top_k < 1 or config.rebalance_bars < 1:
        raise SelectionBiasError("composite trend requires usable risk and rebalance parameters")
    if config.minimum_quote_volume < 0.0 or "BTC" not in data.symbols:
        raise SelectionBiasError("composite trend requires BTC and a non-negative liquidity gate")
    if len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("composite trend requires distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every composite-trend alt must exist in the panel") from error
    if config.top_k * 2 > len(indices):
        raise SelectionBiasError("composite-trend top-k needs distinct long and short assets")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("composite-trend panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("composite-trend panel requires finite positive close prices")
    return indices


def _rank_component(values: np.ndarray, *, eligible: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(eligible & np.isfinite(values))
    if valid.size == 0:
        return result
    ordered = valid[np.argsort(values[valid], kind="stable")]
    if ordered.size == 1:
        result[ordered] = 0.0
        return result
    result[ordered] = np.linspace(-0.5, 0.5, ordered.size)
    return result


def _beta(log_returns: np.ndarray, *, previous: int, beta_window: int, btc_index: int, alt_indices: np.ndarray) -> np.ndarray:
    sample = log_returns[previous - beta_window + 1 : previous + 1]
    market = sample[:, btc_index]
    centered_market = market - market.mean()
    denominator = float(centered_market @ centered_market)
    if denominator <= 0.0:
        return np.zeros(len(alt_indices))
    assets = sample[:, alt_indices]
    return (assets - assets.mean(axis=0)).T @ centered_market / denominator


def _positions_from_score(score: np.ndarray, *, eligible: np.ndarray, top_k: int, indices: np.ndarray, width: int) -> np.ndarray:
    positions = np.zeros(width)
    valid = np.flatnonzero(eligible & np.isfinite(score))
    if valid.size < 2 * top_k:
        return positions
    ordered = valid[np.argsort(score[valid], kind="stable")]
    positions[indices[ordered[:top_k]]] = -0.5 / top_k
    positions[indices[ordered[-top_k:]]] = 0.5 / top_k
    return positions


def composite_trend_positions(data: MarketData, config: CTrendConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Score bar *t* only from bars closed at or before *t - 1*, then form a neutral alt basket."""
    alt_indices = _validate(data, config, alt_symbols)
    btc_index = data.symbols.index("BTC")
    log_returns = np.zeros_like(data.close_price)
    log_returns[1:] = np.log(data.close_price[1:] / data.close_price[:-1])
    first = max(config.long_lookback, config.volatility_window, config.beta_window) + 1
    positions = np.zeros_like(data.close_price)
    for index in range(first, len(positions)):
        previous = index - 1
        eligible = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        if index % config.rebalance_bars:
            positions[index] = positions[index - 1]
            positions[index, alt_indices[~eligible]] = 0.0
            continue
        beta = _beta(log_returns, previous=previous, beta_window=config.beta_window, btc_index=btc_index, alt_indices=alt_indices)
        components: list[np.ndarray] = []
        for lookback in (config.short_lookback, config.medium_lookback, config.long_lookback):
            momentum = np.log(data.close_price[previous, alt_indices] / data.close_price[previous - lookback, alt_indices])
            if config.style is CompositeStyle.BTC_RESIDUAL_MULTI_SIGNAL:
                btc_momentum = float(np.log(data.close_price[previous, btc_index] / data.close_price[previous - lookback, btc_index]))
                momentum = momentum - beta * btc_momentum
            components.append(_rank_component(momentum, eligible=eligible))
        if config.style is not CompositeStyle.MULTI_HORIZON:
            range_high = data.high_price[previous - config.long_lookback + 1 : previous + 1, alt_indices].max(axis=0)
            breakout = np.log(data.close_price[previous, alt_indices] / range_high)
            volume_average = data.quote_volume[previous - config.medium_lookback : previous, alt_indices].mean(axis=0)
            volume_confirmed = np.full(len(alt_indices), np.nan)
            positive_volume = volume_average > 0.0
            volume_confirmed[positive_volume] = (
                np.sign(np.log(data.close_price[previous, alt_indices[positive_volume]] / data.close_price[previous - config.medium_lookback, alt_indices[positive_volume]]))
                * np.log(data.quote_volume[previous, alt_indices[positive_volume]] / volume_average[positive_volume])
            )
            volatility = log_returns[previous - config.volatility_window + 1 : previous + 1, alt_indices].std(axis=0, ddof=1)
            components.extend(
                (
                    _rank_component(breakout, eligible=eligible),
                    _rank_component(volume_confirmed, eligible=eligible),
                    _rank_component(-volatility, eligible=eligible),
                )
            )
        score = np.nansum(np.asarray(components), axis=0)
        score[np.any(~np.isfinite(np.asarray(components)), axis=0)] = np.nan
        positions[index] = _positions_from_score(score, eligible=eligible, top_k=config.top_k, indices=alt_indices, width=len(data.symbols))
    return positions
