"""Strictly lagged rolling-PCA residual statistical-arbitrage signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError


@dataclass(frozen=True, slots=True)
class PcaStatArbConfig:
    training_bars: int
    factor_count: int
    signal_bars: int
    residual_z_window: int
    entry_z: float
    model_refit_bars: int
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"train={self.training_bars}|factors={self.factor_count}|signal={self.signal_bars}"
            f"|residual_z={self.residual_z_window}|entry_z={self.entry_z:g}|refit={self.model_refit_bars}"
            f"|top_k={self.top_k}|rebalance={self.rebalance_bars}"
            f"|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


@dataclass(frozen=True, slots=True)
class _PcaModel:
    mean: np.ndarray
    scale: np.ndarray
    loadings: np.ndarray


def _validate(data: MarketData, config: PcaStatArbConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if min(config.training_bars, config.signal_bars, config.residual_z_window, config.model_refit_bars, config.top_k, config.rebalance_bars) < 1:
        raise SelectionBiasError("PCA statistical arbitrage requires positive windows and cadence")
    if config.entry_z < 0.0 or config.minimum_quote_volume < 0.0:
        raise SelectionBiasError("PCA statistical arbitrage requires non-negative thresholds")
    if "BTC" not in data.symbols or len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("PCA statistical arbitrage requires distinct non-BTC alts and BTC")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every PCA alt must exist in the panel") from error
    if config.factor_count >= len(indices):
        raise SelectionBiasError("PCA factor count must be below the number of alts")
    if config.top_k * 2 > len(indices) or config.training_bars <= config.factor_count + 2:
        raise SelectionBiasError("PCA portfolio dimension is not usable")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("PCA panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("PCA panel requires finite positive close prices")
    return indices


def _fit_model(train_returns: np.ndarray, factor_count: int) -> _PcaModel | None:
    mean = train_returns.mean(axis=0)
    scale = train_returns.std(axis=0, ddof=1)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        return None
    standardized = (train_returns - mean) / scale
    covariance = standardized.T @ standardized / (len(standardized) - 1)
    try:
        _, vectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None
    return _PcaModel(mean, scale, vectors[:, -factor_count:])


def _residual_zscores(
    returns: np.ndarray,
    model: _PcaModel,
    *,
    stop: int,
    signal_bars: int,
    z_window: int,
) -> np.ndarray:
    required = signal_bars + z_window + 1
    start = max(1, stop - required)
    sample = returns[start:stop]
    if len(sample) < required:
        return np.full(returns.shape[1], np.nan)
    standardized = (sample - model.mean) / model.scale
    residual = standardized - (standardized @ model.loadings) @ model.loadings.T
    cumulative = np.vstack((np.zeros(residual.shape[1]), np.cumsum(residual, axis=0)))
    sums = cumulative[signal_bars:] - cumulative[:-signal_bars]
    baseline = sums[-z_window - 1 : -1]
    scale = baseline.std(axis=0, ddof=1)
    zscores = np.full(returns.shape[1], np.nan)
    valid = np.isfinite(scale) & (scale > 0.0)
    zscores[valid] = (sums[-1, valid] - baseline[:, valid].mean(axis=0)) / scale[valid]
    return zscores


def _positions_from_scores(scores: np.ndarray, *, eligible: np.ndarray, top_k: int, indices: np.ndarray, width: int) -> np.ndarray:
    positions = np.zeros(width)
    valid = np.flatnonzero(eligible & np.isfinite(scores))
    if valid.size < top_k * 2:
        return positions
    ordered = valid[np.argsort(scores[valid], kind="stable")]
    positions[indices[ordered[:top_k]]] = 0.5 / top_k
    positions[indices[ordered[-top_k:]]] = -0.5 / top_k
    return positions


def walkforward_pca_stat_arb_positions(data: MarketData, config: PcaStatArbConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Fit PCA through bar *t - 1* and trade mean-reverting residuals at bar *t*."""
    alt_indices = _validate(data, config, alt_symbols)
    close = data.close_price[:, alt_indices]
    returns = np.zeros_like(close)
    returns[1:] = np.log(close[1:] / close[:-1])
    positions = np.zeros_like(data.close_price)
    first = max(config.training_bars + 1, config.signal_bars + config.residual_z_window + 2)
    model: _PcaModel | None = None
    for index in range(first, len(positions)):
        previous = index - 1
        if (index - first) % config.model_refit_bars == 0:
            start = previous - config.training_bars + 1
            model = _fit_model(returns[start : previous + 1], config.factor_count)
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        if (index - first) % config.rebalance_bars:
            carried = positions[index - 1]
            positions[index] = np.zeros_like(carried) if np.any((np.abs(carried[alt_indices]) > 0.0) & ~liquid) else carried
            continue
        if model is None:
            continue
        scores = _residual_zscores(
            returns,
            model,
            stop=previous + 1,
            signal_bars=config.signal_bars,
            z_window=config.residual_z_window,
        )
        eligible = liquid & (np.abs(scores) >= config.entry_z)
        positions[index] = _positions_from_scores(scores, eligible=eligible, top_k=config.top_k, indices=alt_indices, width=len(data.symbols))
    return positions
