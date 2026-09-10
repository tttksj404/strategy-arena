"""Strictly lagged Bayesian-online-changepoint BTC-regime altcoin portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from alt_search_models import FloatMatrix, MarketData
from selection_bias_stats import SelectionBiasError

_EPSILON = 1e-12
_PRIOR_PRECISION = 1.0
_MEAN_Z_THRESHOLD = 0.05
_HIGH_VOLATILITY_Z = 1.5


@dataclass(frozen=True, slots=True)
class BocpdRegimeConfig:
    hazard_bars: int
    maximum_run_length: int
    change_probability_threshold: float
    volatility_lookback: int
    trend_lookback: int
    reversal_lookback: int
    high_volatility_cash: bool
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"hazard={self.hazard_bars}|max_run={self.maximum_run_length}|cp_threshold={self.change_probability_threshold:g}"
            f"|volatility={self.volatility_lookback}|trend={self.trend_lookback}|reversal={self.reversal_lookback}"
            f"|high_vol_cash={str(self.high_volatility_cash).lower()}|top_k={self.top_k}|rebalance={self.rebalance_bars}"
            f"|minimum_quote_volume={self.minimum_quote_volume:g}"
        )


@dataclass(frozen=True, slots=True)
class _BocpdState:
    probabilities: np.ndarray
    precisions: np.ndarray
    means: np.ndarray


def _validate(data: MarketData, config: BocpdRegimeConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    windows = (
        config.hazard_bars,
        config.maximum_run_length,
        config.volatility_lookback,
        config.trend_lookback,
        config.reversal_lookback,
        config.top_k,
        config.rebalance_bars,
    )
    if min(windows) < 1 or not 0.0 < config.change_probability_threshold < 1.0:
        raise SelectionBiasError("BOCPD windows and probability threshold must be positive and bounded")
    if config.minimum_quote_volume < 0.0:
        raise SelectionBiasError("BOCPD liquidity threshold must be non-negative")
    if "BTC" not in data.symbols or len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("BOCPD regime portfolio requires BTC and distinct non-BTC alts")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every BOCPD alt must exist in the panel") from error
    if config.top_k * 2 > len(indices):
        raise SelectionBiasError("BOCPD cross-sectional top-k needs distinct long and short assets")
    if data.close_price.ndim != 2 or data.close_price.shape != data.quote_volume.shape:
        raise SelectionBiasError("BOCPD panel requires aligned price and volume matrices")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("BOCPD panel requires finite positive close prices")
    return indices


def _causal_standardized_returns(close: np.ndarray, *, volatility_lookback: int) -> np.ndarray:
    returns = np.zeros(len(close))
    returns[1:] = np.log(close[1:] / close[:-1])
    standardized = np.zeros_like(returns)
    for index in range(1, len(returns)):
        start = max(1, index - volatility_lookback + 1)
        scale = float(np.sqrt(np.mean(np.square(returns[start : index + 1]))))
        standardized[index] = returns[index] / max(scale, _EPSILON)
    return standardized


def _initial_state(maximum_run_length: int) -> _BocpdState:
    probabilities = np.zeros(maximum_run_length + 1)
    probabilities[0] = 1.0
    return _BocpdState(probabilities, np.full_like(probabilities, _PRIOR_PRECISION), np.zeros_like(probabilities))


def _normal_likelihood(observation: float, means: np.ndarray, precisions: np.ndarray) -> np.ndarray:
    variance = 1.0 + 1.0 / np.maximum(precisions, _EPSILON)
    log_density = -0.5 * ((observation - means) ** 2 / variance + np.log(2.0 * pi * variance))
    return np.exp(np.clip(log_density, -700.0, 700.0))


def _updated_posterior(precisions: np.ndarray, means: np.ndarray, observation: float) -> tuple[np.ndarray, np.ndarray]:
    updated_precision = precisions + 1.0
    updated_mean = (precisions * means + observation) / updated_precision
    return updated_precision, updated_mean


def _update_bocpd(state: _BocpdState, observation: float, *, hazard_bars: int) -> _BocpdState:
    hazard = 1.0 / hazard_bars
    growth_likelihood = _normal_likelihood(observation, state.means, state.precisions)
    prior_likelihood = float(_normal_likelihood(observation, np.array([0.0]), np.array([_PRIOR_PRECISION]))[0])
    probabilities = np.zeros_like(state.probabilities)
    probabilities[0] = hazard * prior_likelihood * state.probabilities.sum()
    probabilities[1:] = (1.0 - hazard) * state.probabilities[:-1] * growth_likelihood[:-1]
    probabilities[-1] += (1.0 - hazard) * state.probabilities[-1] * growth_likelihood[-1]
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        return _initial_state(len(probabilities) - 1)
    updated_precision, updated_mean = _updated_posterior(state.precisions, state.means, observation)
    precisions = np.empty_like(state.precisions)
    means = np.empty_like(state.means)
    precisions[0] = _PRIOR_PRECISION + 1.0
    means[0] = observation / precisions[0]
    precisions[1:], means[1:] = updated_precision[:-1], updated_mean[:-1]
    precisions[-1], means[-1] = updated_precision[-1], updated_mean[-1]
    return _BocpdState(probabilities / total, precisions, means)


def _state_action(state: _BocpdState, config: BocpdRegimeConfig, *, high_volatility_cash: bool) -> int:
    posterior_mean = float(state.probabilities @ state.means)
    predictive_variance = float(state.probabilities @ (1.0 + 1.0 / state.precisions + (state.means - posterior_mean) ** 2))
    predictive_std = np.sqrt(max(predictive_variance, _EPSILON))
    if state.probabilities[0] >= config.change_probability_threshold:
        return 0
    if high_volatility_cash and predictive_std > _HIGH_VOLATILITY_Z:
        return 0
    if abs(posterior_mean / predictive_std) < _MEAN_Z_THRESHOLD:
        return 0
    return 1 if posterior_mean > 0.0 else -1


def _positions_from_scores(scores: np.ndarray, *, liquid: np.ndarray, top_k: int, indices: np.ndarray, width: int) -> np.ndarray:
    positions = np.zeros(width)
    valid = np.flatnonzero(liquid & np.isfinite(scores))
    if valid.size < top_k * 2:
        return positions
    ordered = valid[np.argsort(scores[valid], kind="stable")]
    positions[indices[ordered[-top_k:]]] = 0.5 / top_k
    positions[indices[ordered[:top_k]]] = -0.5 / top_k
    return positions


def walkforward_bocpd_regime_position_variants(
    data: MarketData,
    config: BocpdRegimeConfig,
    *,
    alt_symbols: tuple[str, ...],
) -> dict[bool, FloatMatrix]:
    """Use BTC observations through t-1 to select trend, reversal, or cash at t."""
    alt_indices = _validate(data, config, alt_symbols)
    btc_index = data.symbols.index("BTC")
    observations = _causal_standardized_returns(data.close_price[:, btc_index], volatility_lookback=config.volatility_lookback)
    positions_by_cash = {False: np.zeros_like(data.close_price), True: np.zeros_like(data.close_price)}
    first = max(config.volatility_lookback, config.trend_lookback, config.reversal_lookback) + 1
    state = _initial_state(config.maximum_run_length)
    for index in range(1, len(data.close_price)):
        previous = index - 1
        state = _update_bocpd(state, float(observations[previous]), hazard_bars=config.hazard_bars)
        if index < first:
            continue
        liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
        for high_volatility_cash, positions in positions_by_cash.items():
            action = _state_action(state, config, high_volatility_cash=high_volatility_cash)
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


def walkforward_bocpd_regime_positions(data: MarketData, config: BocpdRegimeConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Build one causal BOCPD BTC-regime portfolio selected by the configuration."""
    return walkforward_bocpd_regime_position_variants(data, config, alt_symbols=alt_symbols)[config.high_volatility_cash]
