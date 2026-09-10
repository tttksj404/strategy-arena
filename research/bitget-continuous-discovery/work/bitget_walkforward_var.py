"""Strictly lagged multivariate ridge vector-autoregression paper signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


@dataclass(frozen=True, slots=True)
class VarConfig:
    return_lags: int
    volatility_window: int
    training_bars: int
    penalty: float
    model_refit_bars: int
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"lags={self.return_lags}|vol={self.volatility_window}|train={self.training_bars}"
            f"|ridge={self.penalty:g}|refit={self.model_refit_bars}|top_k={self.top_k}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _validate(data: MarketData, config: VarConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.return_lags < 1 or config.volatility_window < 2:
        raise SelectionBiasError("VAR requires positive return lags and a usable volatility window")
    if config.training_bars < 8 or config.penalty < 0.0 or config.model_refit_bars < 1 or config.top_k < 1 or config.rebalance_bars < 1:
        raise SelectionBiasError("VAR requires usable training and risk parameters")
    if config.minimum_quote_volume < 0.0 or "BTC" not in data.symbols:
        raise SelectionBiasError("VAR panel requires BTC and a non-negative liquidity gate")
    if len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("VAR cross section requires distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every VAR alt must exist in the panel") from error
    if config.top_k * 2 > len(indices):
        raise SelectionBiasError("VAR cross-sectional top-k needs distinct long and short assets")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("VAR panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("VAR panel requires finite positive close prices")
    return indices


def _feature_matrix(data: MarketData, config: VarConfig) -> tuple[np.ndarray, np.ndarray, int]:
    close = data.close_price
    returns = np.zeros_like(close)
    returns[1:] = np.log(close[1:] / close[:-1])
    feature_start = max(config.return_lags + 1, config.volatility_window) + 1
    features = np.full((len(close), data.close_price.shape[1] * (config.return_lags + 2)), np.nan)
    for index in range(feature_start, len(features)):
        previous = index - 1
        lagged = np.concatenate(tuple(returns[previous - lag] for lag in range(config.return_lags)))
        volatility = returns[previous - config.volatility_window + 1 : previous + 1].std(axis=0, ddof=1)
        prior_volume = data.quote_volume[previous - config.volatility_window : previous].mean(axis=0)
        volume_surprise = np.full(close.shape[1], np.nan)
        valid_volume = prior_volume > 0.0
        volume_surprise[valid_volume] = np.log(data.quote_volume[previous, valid_volume] / prior_volume[valid_volume])
        features[index] = np.concatenate((lagged, -volatility, volume_surprise))
    return features, returns, feature_start


def _var_scores(features: np.ndarray, returns: np.ndarray, *, start: int, end: int, prediction: np.ndarray, penalty: float) -> np.ndarray:
    design = features[start:end]
    response = returns[start:end]
    valid = np.isfinite(design).all(axis=1) & np.isfinite(response).all(axis=1)
    if valid.sum() <= design.shape[1] + 1 or not np.isfinite(prediction).all():
        return np.full(response.shape[1], np.nan)
    selected = design[valid]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    if np.any(scale <= 0.0):
        return np.full(response.shape[1], np.nan)
    normalized = (selected - mean) / scale
    matrix = np.column_stack((np.ones(len(normalized)), normalized))
    penalty_matrix = np.eye(matrix.shape[1]) * penalty
    penalty_matrix[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(matrix.T @ matrix + penalty_matrix, matrix.T @ response[valid])
    except np.linalg.LinAlgError:
        return np.full(response.shape[1], np.nan)
    return coefficients[0] + ((prediction - mean) / scale) @ coefficients[1:]


def _positions_from_scores(scores: np.ndarray, *, eligible: np.ndarray, top_k: int, indices: np.ndarray, width: int) -> np.ndarray:
    positions = np.zeros(width)
    valid = np.flatnonzero(eligible & np.isfinite(scores))
    if valid.size < top_k * 2:
        return positions
    ordered = valid[np.argsort(scores[valid], kind="stable")]
    positions[indices[ordered[:top_k]]] = -0.5 / top_k
    positions[indices[ordered[-top_k:]]] = 0.5 / top_k
    return positions


def walkforward_var_positions(data: MarketData, config: VarConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Fit only completed multivariate returns through *t - 1* before bar *t*."""
    alt_indices = _validate(data, config, alt_symbols)
    features, returns, feature_start = _feature_matrix(data, config)
    positions = np.zeros_like(data.close_price)
    scores = np.full(len(alt_indices), np.nan)
    first = feature_start + config.training_bars
    for index in range(first, len(positions)):
        previous = index - 1
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        if (index - first) % config.model_refit_bars == 0:
            start = max(feature_start, previous - config.training_bars + 1)
            scores = _var_scores(features, returns[:, alt_indices], start=start, end=previous + 1, prediction=features[index], penalty=config.penalty)
        if (index - first) % config.rebalance_bars:
            carried = positions[index - 1]
            positions[index] = np.zeros_like(carried) if np.any((np.abs(carried[alt_indices]) > 0.0) & ~liquid) else carried
        else:
            positions[index] = _positions_from_scores(scores, eligible=liquid, top_k=config.top_k, indices=alt_indices, width=len(data.symbols))
    return positions
