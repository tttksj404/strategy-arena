"""Lagged Kalman-filter pair mean-reversion signals for paper-only research."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


@dataclass(frozen=True, slots=True)
class KalmanPairConfig:
    pair: tuple[str, str]
    z_window: int
    entry_z: float
    process_noise: float
    observation_noise: float
    max_half_life: float
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"pair={self.pair[0]}-{self.pair[1]}|z_window={self.z_window}|entry_z={self.entry_z:g}"
            f"|process_noise={self.process_noise:g}|observation_noise={self.observation_noise:g}"
            f"|max_half_life={self.max_half_life:g}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _rolling_statistics(innovations: np.ndarray, *, z_window: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = np.concatenate((np.asarray((0.0,)), np.cumsum(innovations)))
    squares = np.concatenate((np.asarray((0.0,)), np.cumsum(innovations * innovations)))
    products = np.concatenate((np.asarray((0.0,)), np.cumsum(innovations[:-1] * innovations[1:])))
    zscores = np.full(len(innovations), np.nan)
    half_lives = np.full(len(innovations), np.inf)
    paired_count = z_window - 1
    for previous in range(z_window - 1, len(innovations)):
        start = previous - z_window + 1
        total = prefix[previous + 1] - prefix[start]
        square_total = squares[previous + 1] - squares[start]
        variance = (square_total - total * total / z_window) / paired_count
        if variance <= 0.0:
            continue
        zscores[previous] = (innovations[previous] - total / z_window) / np.sqrt(variance)
        previous_total = prefix[previous] - prefix[start]
        current_total = prefix[previous + 1] - prefix[start + 1]
        previous_squares = squares[previous] - squares[start]
        product_total = products[previous] - products[start]
        denominator = previous_squares - previous_total * previous_total / paired_count
        coefficient = (product_total - previous_total * current_total / paired_count) / denominator if denominator > 0.0 else 0.0
        if 0.0 < coefficient < 1.0:
            half_lives[previous] = -log(2.0) / log(coefficient)
    return zscores, half_lives


def _validate(data: MarketData, config: KalmanPairConfig) -> tuple[int, int]:
    if config.pair[0] == config.pair[1] or config.z_window < 3:
        raise SelectionBiasError("Kalman pair needs two symbols and a usable z-score window")
    if config.entry_z <= 0.0 or config.process_noise <= 0.0 or config.observation_noise <= 0.0 or config.max_half_life <= 0.0:
        raise SelectionBiasError("Kalman pair needs positive threshold, noise, and half-life values")
    if config.minimum_quote_volume < 0.0:
        raise SelectionBiasError("Kalman pair requires a non-negative liquidity gate")
    try:
        indices = data.symbols.index(config.pair[0]), data.symbols.index(config.pair[1])
    except ValueError as error:
        raise SelectionBiasError("every Kalman-pair symbol must exist in the panel") from error
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("Kalman pair requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("Kalman pair requires finite positive close prices")
    return indices


def _innovations(log_base: np.ndarray, log_hedge: np.ndarray, *, process_noise: float, observation_noise: float) -> tuple[np.ndarray, np.ndarray]:
    intercept, beta = 0.0, 1.0
    covariance_00, covariance_01, covariance_11 = 100.0, 0.0, 100.0
    innovations = np.full(len(log_base), np.nan)
    betas = np.full(len(log_base), np.nan)
    for index, (base, hedge) in enumerate(zip(log_base, log_hedge, strict=True)):
        covariance_00 += process_noise
        covariance_11 += process_noise
        covariance_intercept = covariance_00 + hedge * covariance_01
        covariance_beta = covariance_01 + hedge * covariance_11
        innovation = base - intercept - beta * hedge
        innovation_variance = covariance_intercept + hedge * covariance_beta + observation_noise
        gain_intercept = covariance_intercept / innovation_variance
        gain_beta = covariance_beta / innovation_variance
        innovations[index] = innovation
        betas[index] = beta
        intercept += gain_intercept * innovation
        beta += gain_beta * innovation
        covariance_00 -= gain_intercept * covariance_intercept
        covariance_01 -= gain_intercept * covariance_beta
        covariance_11 -= gain_beta * covariance_beta
    return innovations, betas


def kalman_pair_positions(data: MarketData, config: KalmanPairConfig) -> FloatMatrix:
    """Use only innovation and hedge estimates available at *t - 1* for bar *t*."""
    base, hedge = _validate(data, config)
    log_prices = np.log(data.close_price)
    innovations, betas = _innovations(
        log_prices[:, base],
        log_prices[:, hedge],
        process_noise=config.process_noise,
        observation_noise=config.observation_noise,
    )
    zscores, half_lives = _rolling_statistics(innovations, z_window=config.z_window)
    positions = np.zeros_like(data.close_price)
    for index in range(config.z_window + 1, len(positions)):
        previous = index - 1
        if np.any(data.quote_volume[previous, (base, hedge)] < config.minimum_quote_volume):
            continue
        beta = float(betas[previous])
        zscore = float(zscores[previous])
        if not 0.25 <= abs(beta) <= 4.0 or not np.isfinite(zscore) or half_lives[previous] > config.max_half_life:
            continue
        base_weight = 1.0 / (1.0 + abs(beta))
        hedge_weight = beta / (1.0 + abs(beta))
        if zscore >= config.entry_z:
            positions[index, base], positions[index, hedge] = -base_weight, hedge_weight
        elif zscore <= -config.entry_z:
            positions[index, base], positions[index, hedge] = base_weight, -hedge_weight
    return positions
