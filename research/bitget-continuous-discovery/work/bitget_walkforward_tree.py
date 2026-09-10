"""Strictly lagged ExtraTrees forecasts for public-data, paper-only research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


@dataclass(frozen=True, slots=True)
class TreeConfig:
    short_lookback: int
    long_lookback: int
    volatility_window: int
    training_bars: int
    model_refit_bars: int
    trees: int
    max_depth: int
    min_samples_leaf: int
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"short={self.short_lookback}|long={self.long_lookback}|vol={self.volatility_window}"
            f"|train={self.training_bars}|refit={self.model_refit_bars}|trees={self.trees}"
            f"|depth={self.max_depth}|leaf={self.min_samples_leaf}|top_k={self.top_k}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


def _validate(data: MarketData, config: TreeConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.short_lookback < 1 or config.long_lookback < config.short_lookback or config.volatility_window < 2:
        raise SelectionBiasError("tree configuration requires ordered positive feature windows")
    if (
        config.training_bars < 8
        or config.model_refit_bars < 1
        or config.trees < 1
        or config.max_depth < 1
        or config.min_samples_leaf < 1
        or config.top_k < 1
        or config.rebalance_bars < 1
    ):
        raise SelectionBiasError("tree configuration requires usable training and risk parameters")
    if config.minimum_quote_volume < 0.0 or "BTC" not in data.symbols:
        raise SelectionBiasError("tree panel requires BTC and a non-negative liquidity gate")
    if len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("tree cross section requires distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every tree alt must exist in the panel") from error
    if config.top_k * 2 > len(indices):
        raise SelectionBiasError("tree cross-sectional top-k needs distinct long and short assets")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("tree panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("tree panel requires finite positive close prices")
    return indices


def _feature_cube(data: MarketData, config: TreeConfig, *, alt_indices: np.ndarray, btc_index: int) -> tuple[np.ndarray, np.ndarray, int]:
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


def _tree_scores(
    features: np.ndarray,
    returns: np.ndarray,
    *,
    start: int,
    end: int,
    prediction: np.ndarray,
    config: TreeConfig,
) -> np.ndarray:
    samples = features[start:end]
    targets = returns[start:end]
    design = samples.reshape(-1, samples.shape[-1])
    response = targets.reshape(-1)
    valid = np.isfinite(design).all(axis=1) & np.isfinite(response)
    minimum_samples = max(config.min_samples_leaf * 2, 8)
    if valid.sum() < minimum_samples or not np.isfinite(prediction).all():
        return np.full(prediction.shape[0], np.nan)
    model = ExtraTreesRegressor(
        n_estimators=config.trees,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=1.0,
        random_state=0,
        n_jobs=1,
    )
    try:
        model.fit(design[valid], response[valid])
        return model.predict(prediction)
    except ValueError:
        return np.full(prediction.shape[0], np.nan)


def _positions_from_scores(scores: np.ndarray, *, eligible: np.ndarray, top_k: int, indices: np.ndarray, width: int) -> np.ndarray:
    positions = np.zeros(width)
    valid = np.flatnonzero(eligible & np.isfinite(scores))
    if valid.size < top_k * 2:
        return positions
    ordered = valid[np.argsort(scores[valid], kind="stable")]
    positions[indices[ordered[:top_k]]] = -0.5 / top_k
    positions[indices[ordered[-top_k:]]] = 0.5 / top_k
    return positions


def walkforward_tree_positions(data: MarketData, config: TreeConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Fit each tree only through *t - 1* before positioning for bar *t*."""
    alt_indices = _validate(data, config, alt_symbols)
    btc_index = data.symbols.index("BTC")
    features, log_returns, feature_start = _feature_cube(data, config, alt_indices=alt_indices, btc_index=btc_index)
    positions = np.zeros_like(data.close_price)
    scores = np.full(len(alt_indices), np.nan)
    first = feature_start + config.training_bars
    for index in range(first, len(positions)):
        previous = index - 1
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        if (index - first) % config.model_refit_bars == 0:
            start = max(feature_start, previous - config.training_bars + 1)
            scores = _tree_scores(
                features,
                log_returns[:, alt_indices],
                start=start,
                end=previous + 1,
                prediction=features[index],
                config=config,
            )
        if (index - first) % config.rebalance_bars:
            carried = positions[index - 1]
            positions[index] = np.zeros_like(carried) if np.any((np.abs(carried[alt_indices]) > 0.0) & ~liquid) else carried
        else:
            positions[index] = _positions_from_scores(scores, eligible=liquid, top_k=config.top_k, indices=alt_indices, width=len(data.symbols))
    return positions
