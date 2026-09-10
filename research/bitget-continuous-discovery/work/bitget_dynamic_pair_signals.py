"""Rolling-hedge, half-life-filtered pair mean-reversion signals."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


@dataclass(frozen=True, slots=True)
class DynamicPairConfig:
    pair: tuple[str, str]
    beta_window: int
    z_window: int
    entry_z: float
    max_half_life: float
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"pair={self.pair[0]}-{self.pair[1]}|beta_window={self.beta_window}|z_window={self.z_window}"
            f"|entry_z={self.entry_z:g}|max_half_life={self.max_half_life:g}"
            f"|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _half_life(residual: np.ndarray) -> float:
    if residual.size < 3:
        return float("inf")
    previous = residual[:-1] - residual[:-1].mean()
    current = residual[1:] - residual[1:].mean()
    denominator = float(np.dot(previous, previous))
    if denominator <= 0.0:
        return float("inf")
    coefficient = float(np.dot(previous, current) / denominator)
    if coefficient <= 0.0 or coefficient >= 1.0:
        return float("inf")
    return -log(2.0) / log(coefficient)


def _validate(data: MarketData, config: DynamicPairConfig) -> tuple[int, int]:
    if config.pair[0] == config.pair[1] or config.beta_window < 3 or config.z_window < 3:
        raise SelectionBiasError("dynamic pair needs two symbols and usable rolling windows")
    if config.z_window > config.beta_window or config.entry_z <= 0.0 or config.max_half_life <= 0.0:
        raise SelectionBiasError("dynamic pair configuration has an invalid z-score or half-life gate")
    if config.minimum_quote_volume < 0.0:
        raise SelectionBiasError("dynamic pair requires a non-negative liquidity gate")
    try:
        indices = data.symbols.index(config.pair[0]), data.symbols.index(config.pair[1])
    except ValueError as error:
        raise SelectionBiasError("every dynamic-pair symbol must exist in the panel") from error
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("dynamic pair requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("dynamic pair requires finite positive close prices")
    return indices


def _rolling_residual(log_base: np.ndarray, log_hedge: np.ndarray) -> tuple[np.ndarray, float]:
    centered_hedge = log_hedge - log_hedge.mean()
    denominator = float(np.dot(centered_hedge, centered_hedge))
    if denominator <= 0.0:
        return np.zeros_like(log_base), 0.0
    beta = float(np.dot(log_base - log_base.mean(), centered_hedge) / denominator)
    intercept = float(log_base.mean() - beta * log_hedge.mean())
    return log_base - (intercept + beta * log_hedge), beta


def dynamic_pair_positions(data: MarketData, config: DynamicPairConfig) -> FloatMatrix:
    """Use only observations closed at *t - 1* to set a beta-hedged position for bar *t*."""
    base, hedge = _validate(data, config)
    log_prices = np.log(data.close_price)
    positions = np.zeros_like(data.close_price)
    first = config.beta_window + 1
    for index in range(first, len(positions)):
        previous = index - 1
        if np.any(data.quote_volume[previous, (base, hedge)] < config.minimum_quote_volume):
            continue
        start = previous - config.beta_window + 1
        residual, beta = _rolling_residual(log_prices[start : previous + 1, base], log_prices[start : previous + 1, hedge])
        if not 0.25 <= abs(beta) <= 4.0:
            continue
        sample = residual[-config.z_window :]
        standard_deviation = float(sample.std(ddof=1))
        if standard_deviation <= 0.0 or _half_life(sample) > config.max_half_life:
            continue
        zscore = float((sample[-1] - sample.mean()) / standard_deviation)
        base_weight = 1.0 / (1.0 + abs(beta))
        hedge_weight = beta / (1.0 + abs(beta))
        if zscore >= config.entry_z:
            positions[index, base], positions[index, hedge] = -base_weight, hedge_weight
        elif zscore <= -config.entry_z:
            positions[index, base], positions[index, hedge] = base_weight, -hedge_weight
    return positions
