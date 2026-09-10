"""Lagged, dollar-neutral signal families for the Bitget paper-only factor search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import numpy as np

from alt_search_models import FloatMatrix, FloatVector, MarketData
from selection_bias_stats import SelectionBiasError


class FactorFamily(StrEnum):
    TREND = "cross_sectional_trend"
    RESIDUAL_TREND = "btc_residual_trend"
    RESIDUAL_REVERSAL = "btc_residual_reversal"
    BREAKOUT = "cross_sectional_breakout"
    LOW_VOLATILITY = "cross_sectional_low_volatility"
    VOLUME_TREND = "volume_confirmed_trend"
    ANCHORED_REVERSAL = "anchored_cross_sectional_reversal"


class FactorExposure(StrEnum):
    DOLLAR_NEUTRAL = "dollar_neutral"
    LONG_ONLY = "long_only"


@dataclass(frozen=True, slots=True)
class FactorConfig:
    family: FactorFamily
    lookback: int
    top_k: int
    rebalance_bars: int
    minimum_quote_volume: float
    beta_window: int
    volume_multiplier: float

    @property
    def identifier(self) -> str:
        return (
            f"family={self.family.value}|lookback={self.lookback}|top_k={self.top_k}"
            f"|rebalance={self.rebalance_bars}|minimum_quote_volume={self.minimum_quote_volume:g}"
            f"|beta_window={self.beta_window}|volume_multiplier={self.volume_multiplier:g}"
        )


@dataclass(frozen=True, slots=True)
class FactorPositionVariants:
    dollar_neutral: FloatMatrix
    long_only: FloatMatrix

    def positions_for(self, exposure: FactorExposure) -> FloatMatrix:
        match exposure:
            case FactorExposure.DOLLAR_NEUTRAL:
                return self.dollar_neutral
            case FactorExposure.LONG_ONLY:
                return self.long_only
            case unreachable:
                assert_never(unreachable)


def _validate_config(data: MarketData, config: FactorConfig, alt_symbols: tuple[str, ...]) -> np.ndarray:
    if config.lookback < 2 or config.top_k < 1 or config.rebalance_bars < 1:
        raise SelectionBiasError("factor configuration requires positive lagged horizons")
    if config.minimum_quote_volume < 0.0 or config.volume_multiplier < 0.0:
        raise SelectionBiasError("factor configuration requires non-negative volume gates")
    if config.family in {FactorFamily.RESIDUAL_TREND, FactorFamily.RESIDUAL_REVERSAL} and config.beta_window < 2:
        raise SelectionBiasError("residual factor requires a beta window of at least two returns")
    if len(set(alt_symbols)) != len(alt_symbols) or "BTC" in alt_symbols:
        raise SelectionBiasError("alt symbols must be distinct and exclude BTC")
    if "BTC" not in data.symbols:
        raise SelectionBiasError("BTC is required as the residual-factor market input")
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in alt_symbols], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("every requested alt symbol must exist in the panel") from error
    if config.top_k * 2 > len(indices):
        raise SelectionBiasError("cross-sectional top-k needs distinct long and short assets")
    if data.close_price.shape != data.quote_volume.shape or data.close_price.ndim != 2:
        raise SelectionBiasError("factor panel must have aligned two-dimensional prices and volumes")
    if not np.isfinite(data.close_price).all() or np.any(data.close_price <= 0.0):
        raise SelectionBiasError("factor panel requires finite positive close prices")
    return indices


def _residual_scores(
    log_returns: FloatMatrix,
    *,
    previous: int,
    lookback: int,
    beta_window: int,
    btc_index: int,
    alt_indices: np.ndarray,
) -> np.ndarray:
    beta_start = previous - beta_window + 1
    returns = log_returns[beta_start : previous + 1]
    market = returns[:, btc_index]
    centered_market = market - market.mean()
    denominator = float(np.dot(centered_market, centered_market))
    assets = returns[:, alt_indices]
    if denominator <= 0.0:
        beta = np.zeros(len(alt_indices))
    else:
        beta = (assets - assets.mean(axis=0)).T @ centered_market / denominator
    cumulative = log_returns[previous - lookback + 1 : previous + 1]
    return cumulative[:, alt_indices].sum(axis=0) - beta * cumulative[:, btc_index].sum()


def _scores(
    data: MarketData,
    config: FactorConfig,
    *,
    previous: int,
    btc_index: int,
    alt_indices: np.ndarray,
    log_returns: FloatMatrix,
) -> np.ndarray:
    match config.family:
        case FactorFamily.RESIDUAL_TREND:
            return _residual_scores(
                log_returns,
                previous=previous,
                lookback=config.lookback,
                beta_window=config.beta_window,
                btc_index=btc_index,
                alt_indices=alt_indices,
            )
        case FactorFamily.RESIDUAL_REVERSAL:
            return -_residual_scores(
                log_returns,
                previous=previous,
                lookback=config.lookback,
                beta_window=config.beta_window,
                btc_index=btc_index,
                alt_indices=alt_indices,
            )
        case FactorFamily.LOW_VOLATILITY:
            trailing = log_returns[previous - config.lookback + 1 : previous + 1, alt_indices]
            return -trailing.std(axis=0, ddof=1)
        case FactorFamily.BREAKOUT:
            history = data.close_price[previous - config.lookback : previous, alt_indices]
            return np.log(data.close_price[previous, alt_indices] / np.median(history, axis=0))
        case FactorFamily.ANCHORED_REVERSAL:
            trailing = data.close_price[previous - config.lookback + 1 : previous + 1, alt_indices]
            trailing_low = trailing.min(axis=0)
            return -np.log(data.close_price[previous, alt_indices] / trailing_low)
        case FactorFamily.TREND | FactorFamily.VOLUME_TREND:
            return np.log(data.close_price[previous, alt_indices] / data.close_price[previous - config.lookback, alt_indices])
        case unreachable:
            assert_never(unreachable)


def _eligible(
    data: MarketData,
    config: FactorConfig,
    *,
    previous: int,
    alt_indices: np.ndarray,
) -> np.ndarray:
    liquid = data.quote_volume[previous, alt_indices] >= config.minimum_quote_volume
    if config.family is not FactorFamily.VOLUME_TREND:
        return liquid
    history = data.quote_volume[previous - config.lookback : previous, alt_indices]
    average = history.mean(axis=0)
    return liquid & (data.quote_volume[previous, alt_indices] >= average * config.volume_multiplier)


def _selected_positions(
    scores: np.ndarray,
    *,
    eligible: np.ndarray,
    top_k: int,
    assets: np.ndarray,
    width: int,
) -> tuple[FloatVector, FloatVector]:
    dollar_neutral = np.zeros(width)
    long_only = np.zeros(width)
    valid = np.flatnonzero(eligible & np.isfinite(scores))
    ordered = valid[np.argsort(scores[valid], kind="stable")]
    if valid.size >= top_k * 2:
        dollar_neutral[assets[ordered[:top_k]]] = -0.5 / top_k
        dollar_neutral[assets[ordered[-top_k:]]] = 0.5 / top_k
    if valid.size >= top_k:
        long_only[assets[ordered[-top_k:]]] = 1.0 / top_k
    return dollar_neutral, long_only


def _build_position_outputs(
    data: MarketData,
    config: FactorConfig,
    *,
    alt_symbols: tuple[str, ...],
    exposures: tuple[FactorExposure, ...],
) -> dict[FactorExposure, FloatMatrix]:
    alt_indices = _validate_config(data, config, alt_symbols)
    btc_index = data.symbols.index("BTC")
    log_returns = np.zeros_like(data.close_price)
    log_returns[1:] = np.log(data.close_price[1:] / data.close_price[:-1])
    first = config.lookback + 1
    if config.family in {FactorFamily.RESIDUAL_TREND, FactorFamily.RESIDUAL_REVERSAL}:
        first = max(first, config.beta_window + 1)
    outputs = {exposure: np.zeros_like(data.close_price) for exposure in exposures}
    for index in range(first, len(data.close_price)):
        previous = index - 1
        eligible = _eligible(data, config, previous=previous, alt_indices=alt_indices)
        if index % config.rebalance_bars:
            for positions in outputs.values():
                positions[index] = positions[index - 1]
                positions[index, alt_indices[~eligible]] = 0.0
            continue
        scores = _scores(
            data,
            config,
            previous=previous,
            btc_index=btc_index,
            alt_indices=alt_indices,
            log_returns=log_returns,
        )
        dollar_neutral, long_only = _selected_positions(
            scores,
            eligible=eligible,
            top_k=config.top_k,
            assets=alt_indices,
            width=len(data.symbols),
        )
        if FactorExposure.DOLLAR_NEUTRAL in outputs:
            outputs[FactorExposure.DOLLAR_NEUTRAL][index] = dollar_neutral
        if FactorExposure.LONG_ONLY in outputs:
            outputs[FactorExposure.LONG_ONLY][index] = long_only
    return outputs


def build_factor_position_variants(
    data: MarketData,
    config: FactorConfig,
    *,
    alt_symbols: tuple[str, ...],
) -> FactorPositionVariants:
    """Generate both factor exposures from data closed no later than bar *t - 1*."""

    outputs = _build_position_outputs(
        data,
        config,
        alt_symbols=alt_symbols,
        exposures=(FactorExposure.DOLLAR_NEUTRAL, FactorExposure.LONG_ONLY),
    )
    return FactorPositionVariants(
        dollar_neutral=outputs[FactorExposure.DOLLAR_NEUTRAL],
        long_only=outputs[FactorExposure.LONG_ONLY],
    )


def build_factor_positions(
    data: MarketData,
    config: FactorConfig,
    *,
    alt_symbols: tuple[str, ...],
    exposure: FactorExposure = FactorExposure.DOLLAR_NEUTRAL,
) -> FloatMatrix:
    """Generate one selected factor exposure from data closed no later than bar *t - 1*."""

    return _build_position_outputs(data, config, alt_symbols=alt_symbols, exposures=(exposure,))[exposure]
