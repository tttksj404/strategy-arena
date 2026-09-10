"""Shared adaptive configurations for no-regime-gate counterfactual research."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

from alt_search_models import FloatMatrix, MarketData, RiskOverlay
from bitget_directional_change import (
    DirectionalChangeConfig,
    walkforward_directional_change_positions,
)
from bitget_olmar import OlmarConfig, walkforward_olmar_positions
from run_bitget_4h_pair_regime_search import SearchSpec
from run_bitget_directional_change_search import (
    _search_configs as directional_change_search_configs,
)
from run_bitget_olmar_search import search_configs as olmar_search_configs

MAX_LEVERAGE: Final = 5.0


class AdaptiveFamily(StrEnum):
    OLMAR = "olmar"
    DIRECTIONAL_CHANGE = "directional_change"


@dataclass(frozen=True, slots=True)
class OlmarCounterfactual:
    model: OlmarConfig
    target_bar_volatility: float
    risk_window: int


@dataclass(frozen=True, slots=True)
class DirectionalChangeCounterfactual:
    model: DirectionalChangeConfig
    target_bar_volatility: float
    risk_window: int


type AdaptiveConfig = OlmarCounterfactual | DirectionalChangeCounterfactual


def adaptive_configs(spec: SearchSpec) -> tuple[AdaptiveConfig, ...]:
    """Return unique OLMAR and directional-change models without BTC-gate duplicates."""

    olmar_models = {entry.model.identifier: entry.model for entry in olmar_search_configs(spec)}
    directional_models = {
        entry.model.identifier: entry.model for entry in directional_change_search_configs(spec)
    }
    risk_window = spec.risk_windows[0]
    olmar = tuple(
        OlmarCounterfactual(model, target_volatility, risk_window)
        for model in olmar_models.values()
        for target_volatility in spec.target_bar_volatilities
    )
    directional_change = tuple(
        DirectionalChangeCounterfactual(model, target_volatility, risk_window)
        for model in directional_models.values()
        for target_volatility in spec.target_bar_volatilities
    )
    return olmar + directional_change


def family(config: AdaptiveConfig) -> AdaptiveFamily:
    """Return the exhaustive algorithm family for a configuration."""

    match config:
        case OlmarCounterfactual():
            return AdaptiveFamily.OLMAR
        case DirectionalChangeCounterfactual():
            return AdaptiveFamily.DIRECTIONAL_CHANGE
        case unreachable:
            assert_never(unreachable)


def identifier(config: AdaptiveConfig) -> str:
    """Return a stable report identifier."""

    return (
        f"family={family(config).value}|{config.model.identifier}"
        f"|target_bar_vol={config.target_bar_volatility:g}|risk_window={config.risk_window}"
        f"|counterfactual_max_leverage={MAX_LEVERAGE:g}|regime_gate=disabled"
    )


def overlay(config: AdaptiveConfig) -> RiskOverlay:
    """Return the common 5x cap overlay for a configuration."""

    return RiskOverlay(MAX_LEVERAGE, config.target_bar_volatility, config.risk_window)


def positions_for(data: MarketData, config: AdaptiveConfig, *, alt_symbols: tuple[str, ...]) -> FloatMatrix:
    """Generate strictly lagged positions for either adaptive family."""

    match config:
        case OlmarCounterfactual(model=model):
            return walkforward_olmar_positions(data, model, alt_symbols=alt_symbols)
        case DirectionalChangeCounterfactual(model=model):
            return walkforward_directional_change_positions(data, model, alt_symbols=alt_symbols)
        case unreachable:
            assert_never(unreachable)
