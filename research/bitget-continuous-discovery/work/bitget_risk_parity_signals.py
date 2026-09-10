"""Lagged long-only risk-adjusted momentum and inverse-volatility allocation signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


@dataclass(frozen=True, slots=True)
class RiskParityConfig:
    momentum_lookback: int
    volatility_window: int
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"momentum_lookback={self.momentum_lookback}|volatility_window={self.volatility_window}"
            f"|top_k={self.top_k}|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _validate(data: MarketData, config: RiskParityConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.momentum_lookback < 1 or config.volatility_window < 2 or config.top_k < 1 or config.rebalance_bars < 1:
        raise SelectionBiasError("risk-parity momentum requires usable positive horizons")
    if config.minimum_quote_volume < 0.0 or "BTC" not in data.symbols:
        raise SelectionBiasError("risk-parity momentum requires BTC and a non-negative liquidity gate")
    if len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("risk-parity momentum requires distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every risk-parity alt must exist in the panel") from error
    if config.top_k > len(indices):
        raise SelectionBiasError("risk-parity top-k exceeds the available alt cross section")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("risk-parity panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("risk-parity panel requires finite positive close prices")
    return indices


def risk_parity_momentum_positions(data: MarketData, config: RiskParityConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Select positive risk-adjusted momentum from bar *t - 1*, then allocate inverse trailing volatility."""
    alt_indices = _validate(data, config, alt_symbols)
    log_returns = np.zeros_like(data.close_price)
    log_returns[1:] = np.log(data.close_price[1:] / data.close_price[:-1])
    positions = np.zeros_like(data.close_price)
    first = max(config.momentum_lookback, config.volatility_window) + 1
    for index in range(first, len(positions)):
        previous = index - 1
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        if index % config.rebalance_bars:
            positions[index] = positions[index - 1]
            positions[index, alt_indices[~liquid]] = 0.0
            continue
        momentum = np.log(data.close_price[previous, alt_indices] / data.close_price[previous - config.momentum_lookback, alt_indices])
        volatility = log_returns[previous - config.volatility_window + 1 : previous + 1, alt_indices].std(axis=0, ddof=1)
        score = np.full(len(alt_indices), np.nan)
        valid_volatility = volatility > 0.0
        score[valid_volatility] = momentum[valid_volatility] / (volatility[valid_volatility] * np.sqrt(config.momentum_lookback))
        valid = np.flatnonzero(liquid & np.isfinite(score) & (score > 0.0))
        if valid.size < config.top_k:
            continue
        selected = valid[np.argsort(score[valid], kind="stable")[-config.top_k:]]
        inverse_volatility = 1.0 / volatility[selected]
        positions[index, alt_indices[selected]] = inverse_volatility / inverse_volatility.sum()
    return positions
