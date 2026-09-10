"""Strictly lagged Gaussian-HMM Bitcoin-regime altcoin portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError

MIN_VARIANCE = 1e-12
EM_ITERATIONS = 12


@dataclass(frozen=True, slots=True)
class HmmRegimeConfig:
    training_bars: int
    state_count: int
    transition_persistence: float
    model_refit_bars: int
    trend_lookback: int
    reversal_lookback: int
    high_volatility_cash: bool
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"train={self.training_bars}|states={self.state_count}|persistence={self.transition_persistence:g}"
            f"|refit={self.model_refit_bars}|trend={self.trend_lookback}|reversal={self.reversal_lookback}"
            f"|high_vol_cash={str(self.high_volatility_cash).lower()}|top_k={self.top_k}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


@dataclass(frozen=True, slots=True)
class _HmmModel:
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray


def _validate(data: MarketData, config: HmmRegimeConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    windows = (config.training_bars, config.model_refit_bars, config.trend_lookback, config.reversal_lookback, config.top_k, config.rebalance_bars)
    if min(windows) < 1 or config.state_count < 2:
        raise SelectionBiasError("HMM state count and windows must be positive")
    if config.training_bars < config.state_count * 8:
        raise SelectionBiasError("HMM training window is too short for the state count")
    if not 0.5 <= config.transition_persistence < 1.0 or config.minimum_quote_volume < 0.0:
        raise SelectionBiasError("HMM persistence and liquidity threshold are invalid")
    if "BTC" not in data.symbols or len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("HMM regime portfolio requires BTC and distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every HMM alt must exist in the panel") from error
    if config.top_k * 2 > len(indices):
        raise SelectionBiasError("HMM cross-sectional top-k needs distinct long and short assets")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("HMM panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("HMM panel requires finite positive close prices")
    return indices


def _emissions(observations: np.ndarray, model: _HmmModel) -> np.ndarray:
    variance = np.maximum(model.variances, MIN_VARIANCE)
    log_density = -0.5 * ((observations[:, None] - model.means) ** 2 / variance + np.log(2.0 * pi * variance))
    return np.maximum(np.exp(np.clip(log_density, -700.0, 700.0)), 1e-300)


def _forward_backward(observations: np.ndarray, model: _HmmModel) -> tuple[np.ndarray, np.ndarray] | None:
    emission = _emissions(observations, model)
    count, states = emission.shape
    alpha = np.zeros((count, states))
    scale = np.zeros(count)
    initial = model.initial * emission[0]
    scale[0] = initial.sum()
    if scale[0] <= 0.0:
        return None
    alpha[0] = initial / scale[0]
    for index in range(1, count):
        value = (alpha[index - 1] @ model.transition) * emission[index]
        scale[index] = value.sum()
        if scale[index] <= 0.0:
            return None
        alpha[index] = value / scale[index]
    beta = np.ones((count, states))
    for index in range(count - 2, -1, -1):
        beta[index] = (model.transition @ (emission[index + 1] * beta[index + 1])) / scale[index + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    transition_counts = np.zeros((states, states))
    for index in range(count - 1):
        weights = alpha[index, :, None] * model.transition * (emission[index + 1] * beta[index + 1])[None, :]
        total = weights.sum()
        if total <= 0.0:
            return None
        transition_counts += weights / total
    return gamma, transition_counts


def _transition_prior(state_count: int, persistence: float) -> np.ndarray:
    off_diagonal = (1.0 - persistence) / (state_count - 1)
    return np.full((state_count, state_count), off_diagonal) + np.eye(state_count) * (persistence - off_diagonal)


def _fit_hmm(observations: np.ndarray, *, state_count: int, persistence: float) -> tuple[_HmmModel, np.ndarray] | None:
    if observations.ndim != 1 or len(observations) < state_count * 8 or not np.isfinite(observations).all():
        return None
    variance = max(float(np.var(observations, ddof=1)), MIN_VARIANCE)
    quantiles = np.linspace(0.0, 1.0, state_count + 2)[1:-1]
    transition_prior = _transition_prior(state_count, persistence)
    model = _HmmModel(
        initial=np.full(state_count, 1.0 / state_count),
        transition=transition_prior,
        means=np.quantile(observations, quantiles),
        variances=np.full(state_count, variance),
    )
    for _ in range(EM_ITERATIONS):
        fitted = _forward_backward(observations, model)
        if fitted is None:
            return None
        gamma, transition_counts = fitted
        weights = gamma.sum(axis=0)
        if np.any(weights < 2.0):
            return None
        means = gamma.T @ observations / weights
        variances = (gamma * (observations[:, None] - means) ** 2).sum(axis=0) / weights
        empirical_transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        transition = 0.9 * empirical_transition + 0.1 * transition_prior
        initial = 0.95 * gamma[0] + 0.05 / state_count
        model = _HmmModel(initial / initial.sum(), transition / transition.sum(axis=1, keepdims=True), means, np.maximum(variances, MIN_VARIANCE))
    fitted = _forward_backward(observations, model)
    return None if fitted is None else (model, fitted[0][-1])


def _filter_step(posterior: np.ndarray, model: _HmmModel, observation: float) -> np.ndarray:
    likelihood = _emissions(np.asarray([observation]), model)[0]
    value = (posterior @ model.transition) * likelihood
    total = value.sum()
    return np.full(len(value), 1.0 / len(value)) if total <= 0.0 else value / total


def _state_action(model: _HmmModel, posterior: np.ndarray, *, high_volatility_cash: bool) -> int:
    state = int(np.argmax(posterior))
    volatility = np.sqrt(model.variances)
    if high_volatility_cash and volatility[state] > 1.5 * np.median(volatility):
        return 0
    ranking = np.argsort(model.means, kind="stable")
    if state == ranking[-1]:
        return 1
    if state == ranking[0]:
        return -1
    return 0


def _positions_from_scores(scores: np.ndarray, *, liquid: np.ndarray, top_k: int, indices: np.ndarray, width: int) -> np.ndarray:
    positions = np.zeros(width)
    valid = np.flatnonzero(liquid & np.isfinite(scores))
    if valid.size < top_k * 2:
        return positions
    ordered = valid[np.argsort(scores[valid], kind="stable")]
    positions[indices[ordered[-top_k:]]] = 0.5 / top_k
    positions[indices[ordered[:top_k]]] = -0.5 / top_k
    return positions


def walkforward_hmm_regime_position_variants(
    data: MarketData,
    config: HmmRegimeConfig,
    *,
    alt_symbols: tuple[str, ...],
) -> dict[bool, FloatMatrix]:
    """Build both high-volatility-cash variants from one causal HMM state path."""
    alt_indices = _validate(data, config, alt_symbols)
    btc_index = data.symbols.index("BTC")
    btc_returns = np.zeros(len(data.dates))
    btc_returns[1:] = np.log(data.close_price[1:, btc_index] / data.close_price[:-1, btc_index])
    positions_by_cash = {False: np.zeros_like(data.close_price), True: np.zeros_like(data.close_price)}
    first = max(config.training_bars + 1, config.trend_lookback + 1, config.reversal_lookback + 1)
    model: _HmmModel | None = None
    posterior: np.ndarray | None = None
    for index in range(first, len(data.close_price)):
        previous = index - 1
        if (index - first) % config.model_refit_bars == 0:
            start = previous - config.training_bars + 1
            fitted = _fit_hmm(btc_returns[start : previous + 1], state_count=config.state_count, persistence=config.transition_persistence)
            model, posterior = (None, None) if fitted is None else fitted
        elif model is not None and posterior is not None:
            posterior = _filter_step(posterior, model, btc_returns[previous])
        if model is None or posterior is None:
            continue
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        for high_volatility_cash, positions in positions_by_cash.items():
            action = _state_action(model, posterior, high_volatility_cash=high_volatility_cash)
            if action == 0:
                continue
            if (index - first) % config.rebalance_bars:
                carried = positions[index - 1]
                positions[index] = np.zeros_like(carried) if np.any((np.abs(carried[alt_indices]) > 0.0) & ~liquid) else carried
                continue
            lookback = config.trend_lookback if action > 0 else config.reversal_lookback
            scores = np.log(data.close_price[previous, alt_indices] / data.close_price[previous - lookback, alt_indices])
            positions[index] = _positions_from_scores(action * scores, liquid=liquid, top_k=config.top_k, indices=alt_indices, width=len(data.symbols))
    return positions_by_cash


def walkforward_hmm_regime_positions(data: MarketData, config: HmmRegimeConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Infer a BTC regime only through *t - 1*, then choose trend, reversal, or cash at *t*."""
    return walkforward_hmm_regime_position_variants(data, config, alt_symbols=alt_symbols)[config.high_volatility_cash]
