"""Strictly lagged random-Fourier-feature ridge forecasts for nonlinear paper-only research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


_FREQUENCIES = np.asarray(
    (
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 1.0),
    ),
    dtype=float,
)


@dataclass(frozen=True, slots=True)
class RffConfig:
    short_lookback: int
    long_lookback: int
    volatility_window: int
    training_bars: int
    kernel_gamma: float
    penalty: float
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"short={self.short_lookback}|long={self.long_lookback}|vol={self.volatility_window}"
            f"|train={self.training_bars}|gamma={self.kernel_gamma:g}|ridge={self.penalty:g}|top_k={self.top_k}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _validate(data: MarketData, config: RffConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.short_lookback < 1 or config.long_lookback < config.short_lookback or config.volatility_window < 2:
        raise SelectionBiasError("RFF configuration requires ordered positive feature windows")
    if config.training_bars < 8 or config.kernel_gamma <= 0.0 or config.penalty < 0.0 or config.top_k < 1 or config.rebalance_bars < 1:
        raise SelectionBiasError("RFF configuration requires usable training and risk parameters")
    if config.minimum_quote_volume < 0.0 or "BTC" not in data.symbols:
        raise SelectionBiasError("RFF panel requires BTC and a non-negative liquidity gate")
    if len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("RFF cross section requires distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every RFF alt symbol must exist in the panel") from error
    if config.top_k * 2 > len(indices):
        raise SelectionBiasError("RFF cross-sectional top-k needs distinct long and short assets")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("RFF panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("RFF panel requires finite positive close prices")
    return indices


def _feature_cube(data: MarketData, config: RffConfig, *, alt_indices: np.ndarray, btc_index: int) -> tuple[np.ndarray, np.ndarray, int]:
    close = data.close_price
    log_returns = np.zeros_like(close)
    log_returns[1:] = np.log(close[1:] / close[:-1])
    feature_start = max(config.long_lookback, config.volatility_window) + 1
    features = np.full((len(close), len(alt_indices), 5), np.nan)
    for index in range(feature_start, len(close)):
        previous = index - 1
        short = np.log(close[previous, alt_indices] / close[previous - config.short_lookback, alt_indices])
        long = np.log(close[previous, alt_indices] / close[previous - config.long_lookback, alt_indices])
        volatility = log_returns[previous - config.volatility_window + 1 : previous + 1, alt_indices].std(axis=0, ddof=1)
        prior_volume = data.quote_volume[previous - config.long_lookback : previous, alt_indices].mean(axis=0)
        volume_ratio = np.full(len(alt_indices), np.nan)
        valid_volume = prior_volume > 0.0
        volume_ratio[valid_volume] = np.log(data.quote_volume[previous, alt_indices[valid_volume]] / prior_volume[valid_volume])
        btc_return = np.log(close[previous, btc_index] / close[previous - config.short_lookback, btc_index])
        features[index] = np.column_stack((short, long, -volatility, volume_ratio, short - btc_return))
    return features, log_returns, feature_start


def _features_to_rff(values: np.ndarray, *, gamma: float) -> np.ndarray:
    projection = np.sqrt(2.0 * gamma) * values @ _FREQUENCIES.T
    return np.concatenate((np.cos(projection), np.sin(projection)), axis=1) / np.sqrt(_FREQUENCIES.shape[0])


def _rff_scores(features: np.ndarray, returns: np.ndarray, *, start: int, end: int, prediction: np.ndarray, gamma: float, penalty: float) -> np.ndarray:
    samples = features[start:end]
    targets = returns[start:end]
    design = samples.reshape(-1, samples.shape[-1])
    response = targets.reshape(-1)
    valid = np.isfinite(design).all(axis=1) & np.isfinite(response)
    if valid.sum() <= _FREQUENCIES.shape[0] * 2 + 1 or not np.isfinite(prediction).all():
        return np.full(prediction.shape[0], np.nan)
    selected = design[valid]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    if np.any(scale <= 0.0):
        return np.full(prediction.shape[0], np.nan)
    transformed = _features_to_rff((selected - mean) / scale, gamma=gamma)
    matrix = np.column_stack((np.ones(len(transformed)), transformed))
    penalty_matrix = np.eye(matrix.shape[1]) * penalty
    penalty_matrix[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(matrix.T @ matrix + penalty_matrix, matrix.T @ response[valid])
    except np.linalg.LinAlgError:
        return np.full(prediction.shape[0], np.nan)
    return coefficients[0] + _features_to_rff((prediction - mean) / scale, gamma=gamma) @ coefficients[1:]


def _positions_from_scores(scores: np.ndarray, *, eligible: np.ndarray, top_k: int, indices: np.ndarray, width: int) -> np.ndarray:
    positions = np.zeros(width)
    valid = np.flatnonzero(eligible & np.isfinite(scores))
    if valid.size < top_k * 2:
        return positions
    ordered = valid[np.argsort(scores[valid], kind="stable")]
    positions[indices[ordered[:top_k]]] = -0.5 / top_k
    positions[indices[ordered[-top_k:]]] = 0.5 / top_k
    return positions


def walkforward_rff_positions(data: MarketData, config: RffConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Fit the nonlinear RFF ridge model only through *t - 1* before positioning for bar *t*."""
    alt_indices = _validate(data, config, alt_symbols)
    btc_index = data.symbols.index("BTC")
    features, log_returns, feature_start = _feature_cube(data, config, alt_indices=alt_indices, btc_index=btc_index)
    positions = np.zeros_like(data.close_price)
    first = feature_start + config.training_bars
    for index in range(first, len(positions)):
        previous = index - 1
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        if index % config.rebalance_bars:
            positions[index] = positions[index - 1]
            positions[index, alt_indices[~liquid]] = 0.0
            continue
        start = max(feature_start, previous - config.training_bars + 1)
        scores = _rff_scores(
            features,
            log_returns[:, alt_indices],
            start=start,
            end=previous + 1,
            prediction=features[index],
            gamma=config.kernel_gamma,
            penalty=config.penalty,
        )
        positions[index] = _positions_from_scores(scores, eligible=liquid, top_k=config.top_k, indices=alt_indices, width=len(data.symbols))
    return positions
