"""Strictly lagged directional-change event signals for paper-only altcoin research."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class DirectionalChangeConfig:
    threshold: float
    follow_trend: bool
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"threshold={self.threshold:g}|follow_trend={str(self.follow_trend).lower()}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def directional_change_scores(close: FloatMatrix, *, threshold: float) -> FloatMatrix:
    """Return event-time trend scores at t using only each close through t-1."""
    if close.ndim != 2 or len(close) < 2 or not np.isfinite(close).all() or np.any(close <= 0.0):
        raise SelectionBiasError("directional-change scores require finite positive price panels")
    if threshold <= 0.0:
        raise SelectionBiasError("directional-change threshold must be positive")
    scores = np.zeros_like(close)
    for asset in range(close.shape[1]):
        high = low = anchor = float(close[0, asset])
        direction = 0
        for index in range(1, len(close)):
            observed = float(close[index - 1, asset])
            if direction >= 0:
                high = max(high, observed)
                if observed <= high * (1.0 - threshold):
                    direction, low, anchor = -1, observed, high
            if direction <= 0:
                low = min(low, observed)
                if observed >= low * (1.0 + threshold):
                    direction, high, anchor = 1, observed, low
            scores[index, asset] = direction * abs(log((high if direction > 0 else low) / anchor)) if direction else 0.0
    return scores


def _validate(data: MarketData, config: DirectionalChangeConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.threshold <= 0.0 or config.rebalance_bars < 1 or config.minimum_quote_volume < 0.0:
        raise SelectionBiasError("directional-change threshold, rebalance, and liquidity parameters are invalid")
    if "BTC" not in data.symbols or "BTC" in alt_symbols or len(set(alt_symbols)) != len(alt_symbols):
        raise SelectionBiasError("directional-change portfolio requires BTC and distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every directional-change alt must exist in the panel") from error
    if len(indices) < 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("directional-change requires aligned price and volume panels")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("directional-change requires finite positive close prices")
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


def walkforward_directional_change_positions(data: MarketData, config: DirectionalChangeConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Trade the prior-close directional-change state in a market-neutral form."""
    alt_indices = _validate(data, config, alt_symbols)
    scores = directional_change_scores(data.close_price[:, alt_indices], threshold=config.threshold)
    if not config.follow_trend:
        scores = -scores
    positions = np.zeros_like(data.close_price)
    for index in range(1, len(positions)):
        prior = positions[index - 1, alt_indices]
        liquid = data.quote_volume[index - 1, alt_indices] >= config.minimum_quote_volume
        if (index - 1) % config.rebalance_bars:
            positions[index, alt_indices] = np.zeros_like(prior) if np.any((np.abs(prior) > 0.0) & ~liquid) else prior
            continue
        positions[index, alt_indices] = _market_neutral_projection(scores[index], liquid)
    return positions
