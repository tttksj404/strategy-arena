"""A strictly lagged, market-neutral adaptation of online moving-average reversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class OlmarConfig:
    moving_average_lookback: int
    target_edge: float
    maximum_step: float
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"ma={self.moving_average_lookback}|target_edge={self.target_edge:g}|max_step={self.maximum_step:g}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _validate(data: MarketData, config: OlmarConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.moving_average_lookback < 2 or config.rebalance_bars < 1 or config.maximum_step <= 0.0:
        raise SelectionBiasError("OLMAR windows and maximum step must be positive")
    if config.target_edge <= 0.0:
        raise SelectionBiasError("OLMAR target edge must be positive")
    if config.minimum_quote_volume < 0.0:
        raise SelectionBiasError("OLMAR liquidity threshold must be non-negative")
    if "BTC" not in data.symbols or len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("OLMAR portfolio requires BTC and distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every OLMAR alt must exist in the panel") from error
    if len(indices) < 2 or data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("OLMAR requires an aligned multi-asset price and volume panel")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("OLMAR panel requires finite positive close prices")
    return indices


def _market_neutral_projection(values: np.ndarray, liquid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    valid = np.flatnonzero(liquid & np.isfinite(values))
    if valid.size < 2:
        return result
    centered = values[valid] - values[valid].mean()
    gross = float(np.abs(centered).sum())
    if gross <= _EPSILON:
        return result
    result[valid] = centered / gross
    return result


def _moving_average_prediction(close: np.ndarray, *, previous: int, lookback: int) -> np.ndarray:
    average = close[previous - lookback + 1 : previous + 1].mean(axis=0)
    return average / close[previous]


def walkforward_olmar_positions(data: MarketData, config: OlmarConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Apply OLMAR's passive-aggressive update using only bars observed through t-1.

    Original OLMAR uses a long-only simplex.  This experiment instead projects every
    update onto a zero-net, unit-gross altcoin portfolio so it can be evaluated under
    the repository's market-neutral 5x paper-risk gates.
    """
    alt_indices = _validate(data, config, alt_symbols)
    positions = np.zeros_like(data.close_price)
    first = config.moving_average_lookback
    for index in range(first, len(data.close_price)):
        previous = index - 1
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        prior = positions[index - 1, alt_indices]
        if (index - first) % config.rebalance_bars:
            positions[index, alt_indices] = np.zeros_like(prior) if np.any((np.abs(prior) > 0.0) & ~liquid) else prior
            continue
        predicted_relative = _moving_average_prediction(data.close_price[:, alt_indices], previous=previous, lookback=config.moving_average_lookback)
        signal = np.zeros_like(predicted_relative)
        valid = np.flatnonzero(liquid & np.isfinite(predicted_relative))
        if valid.size < 2:
            continue
        signal[valid] = predicted_relative[valid] - predicted_relative[valid].mean()
        squared_norm = float(signal @ signal)
        edge = float(prior @ signal)
        step = min(config.maximum_step, max(0.0, config.target_edge - edge) / max(squared_norm, _EPSILON))
        candidate = prior + step * signal
        candidate[~liquid] = 0.0
        positions[index, alt_indices] = _market_neutral_projection(candidate, liquid)
    return positions
