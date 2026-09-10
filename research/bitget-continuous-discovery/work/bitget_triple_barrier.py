"""Strictly lagged triple-barrier meta-label positions for paper-only research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError

_FEATURE_COUNT: Final = 3
_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class TripleBarrierConfig:
    feature_lookback: int
    barrier_horizon: int
    profit_take: float
    stop_loss: float
    ridge: float
    rebalance_bars: int
    minimum_training_samples: int
    minimum_quote_volume: float
    prediction_threshold: float

    @property
    def identifier(self) -> str:
        return (
            f"feature_lb={self.feature_lookback}|barrier_h={self.barrier_horizon}|pt={self.profit_take:g}"
            f"|sl={self.stop_loss:g}|ridge={self.ridge:g}|rebalance={self.rebalance_bars}"
            f"|minimum_samples={self.minimum_training_samples}|minimum_quote_volume={self.minimum_quote_volume:g}"
            f"|prediction_threshold={self.prediction_threshold:g}"
        )


def _validate(data: MarketData, config: TripleBarrierConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.feature_lookback < 2 or config.barrier_horizon < 1 or config.rebalance_bars < 1:
        raise SelectionBiasError("triple-barrier windows must be positive")
    if config.profit_take <= 0.0 or config.stop_loss <= 0.0:
        raise SelectionBiasError("triple-barrier barriers must be positive")
    if config.ridge <= 0.0 or config.minimum_training_samples < 1 or config.minimum_quote_volume < 0.0 or config.prediction_threshold < 0.0:
        raise SelectionBiasError("triple-barrier model parameters are invalid")
    if "BTC" not in data.symbols or "BTC" in alt_symbols or len(set(alt_symbols)) != len(alt_symbols):
        raise SelectionBiasError("triple-barrier portfolio requires BTC and distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every triple-barrier alt must exist in the panel") from error
    if len(indices) < 2 or data.close_price.shape != data.high_price.shape or data.close_price.shape != data.low_price.shape:
        raise SelectionBiasError("triple-barrier requires aligned OHLC panels")
    if data.quote_volume.shape != data.close_price.shape or not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("triple-barrier panel requires finite positive closes and aligned volume")
    return indices


def triple_barrier_labels(data: MarketData, config: TripleBarrierConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Label an entry only after its vertical barrier has fully elapsed.

    A same-bar profit/stop collision is deliberately neutral because intrabar
    ordering is unknowable from public candles.
    """
    indices = _validate(data, config, alt_symbols)
    close, high, low = (matrix[:, indices] for matrix in (data.close_price, data.high_price, data.low_price))
    labels = np.full(close.shape, np.nan)
    for origin in range(1, len(close) - config.barrier_horizon + 1):
        entry = close[origin - 1]
        unresolved = np.ones(len(indices), dtype=bool)
        for bar in range(origin, origin + config.barrier_horizon):
            take = high[bar] >= entry * (1.0 + config.profit_take)
            stop = low[bar] <= entry * (1.0 - config.stop_loss)
            tie = unresolved & take & stop
            labels[origin, tie] = 0.0
            labels[origin, unresolved & take & ~stop] = 1.0
            labels[origin, unresolved & stop & ~take] = -1.0
            unresolved &= ~(take | stop)
            if not unresolved.any():
                break
        if unresolved.any():
            terminal = close[origin + config.barrier_horizon - 1] / entry - 1.0
            labels[origin, unresolved] = np.sign(terminal[unresolved])
    return labels


def _lagged_features(close: FloatMatrix, *, lookback: int) -> np.ndarray:
    features = np.full((len(close), close.shape[1], _FEATURE_COUNT), np.nan)
    log_close = np.log(close)
    log_returns = np.zeros_like(log_close)
    log_returns[1:] = log_close[1:] - log_close[:-1]
    sums = np.concatenate((np.zeros((1, close.shape[1])), np.cumsum(log_returns, axis=0)))
    square_sums = np.concatenate((np.zeros((1, close.shape[1])), np.cumsum(log_returns * log_returns, axis=0)))
    indices = np.arange(lookback + 1, len(close))
    previous = indices - 1
    totals = sums[previous + 1] - sums[previous - lookback + 1]
    squares = square_sums[previous + 1] - square_sums[previous - lookback + 1]
    variance = np.maximum((squares - totals * totals / lookback) / (lookback - 1), 0.0)
    zscore = (log_close[previous] - log_close[previous - lookback]) / np.maximum(np.sqrt(variance * lookback), _EPSILON)
    features[indices, :, 0] = 1.0
    features[indices, :, 1] = zscore
    features[indices, :, 2] = zscore * np.abs(zscore)
    return features


def _market_neutral_projection(values: np.ndarray, liquid: np.ndarray, *, threshold: float) -> np.ndarray:
    result = np.zeros_like(values)
    valid = np.flatnonzero(liquid & np.isfinite(values) & (np.abs(values) >= threshold))
    if valid.size < 2:
        return result
    centered = values[valid] - values[valid].mean()
    gross = float(np.abs(centered).sum())
    if gross <= _EPSILON:
        return result
    result[valid] = centered / gross
    return result


def walkforward_triple_barrier_positions(data: MarketData, config: TripleBarrierConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Fit only settled triple-barrier labels, then trade the next candle.

    At index ``t`` the model receives the origin ``t - barrier_horizon`` label,
    whose final candle is ``t - 1``.  Its position for ``t`` therefore never
    observes the candle whose return it will be evaluated against.
    """
    alt_indices = _validate(data, config, alt_symbols)
    close = data.close_price[:, alt_indices]
    labels = triple_barrier_labels(data, config, alt_symbols=alt_symbols)
    features = _lagged_features(close, lookback=config.feature_lookback)
    positions = np.zeros_like(data.close_price)
    gram = np.eye(_FEATURE_COUNT) * config.ridge
    target = np.zeros(_FEATURE_COUNT)
    samples = 0
    first = max(config.feature_lookback + 1, config.barrier_horizon + 1)
    for index in range(first, len(data.close_price)):
        origin = index - config.barrier_horizon
        mature_features, mature_labels = features[origin], labels[origin]
        mature_liquid = data.quote_volume[origin - 1, alt_indices] >= config.minimum_quote_volume
        usable = mature_liquid & np.isfinite(mature_labels) & np.isfinite(mature_features).all(axis=1)
        if usable.any():
            design = mature_features[usable]
            responses = mature_labels[usable]
            gram += design.T @ design
            target += design.T @ responses
            samples += len(responses)
        prior = positions[index - 1, alt_indices]
        liquid = data.quote_volume[index - 1, alt_indices] >= config.minimum_quote_volume
        should_rebalance = (index - first) % config.rebalance_bars == 0
        if samples < config.minimum_training_samples:
            continue
        if not should_rebalance:
            positions[index, alt_indices] = np.zeros_like(prior) if np.any((np.abs(prior) > 0.0) & ~liquid) else prior
            continue
        scores = features[index] @ np.linalg.solve(gram, target)
        positions[index, alt_indices] = _market_neutral_projection(scores, liquid, threshold=config.prediction_threshold)
    return positions
