from __future__ import annotations

import math

from quant_binance.settings import Settings
from quant_binance.strategy.coin_profiles import get_profile, is_profiled


def _edge_to_cost_multiple(net_expected_edge_bps: float, estimated_round_trip_cost_bps: float) -> float:
    if estimated_round_trip_cost_bps <= 0:
        return float("inf")
    return net_expected_edge_bps / estimated_round_trip_cost_bps


def select_futures_leverage(
    *,
    symbol: str = "",
    predictability_score: float,
    trend_strength: float,
    volume_confirmation: float,
    liquidity_score: float,
    volatility_penalty: float,
    overheat_penalty: float,
    net_expected_edge_bps: float,
    estimated_round_trip_cost_bps: float,
    settings: Settings,
    adx_1h: float = 0.0,
    ema_cross_signal: int = 0,
    trend_direction: int = 0,
) -> int:
    thresholds = settings.mode_thresholds
    exposure = settings.futures_exposure
    risk = settings.risk
    max_leverage = max(1, int(math.ceil(risk.max_futures_leverage)))
    target_leverage = max(1, min(int(math.ceil(risk.target_futures_leverage)), max_leverage))
    # Optional per-coin leverage profile. Keep disabled for general profiles
    # to preserve deterministic profile-level leverage behavior.
    cp = get_profile(symbol)
    enable_profiled_leverage = settings.strategy_profile in {"autoresearch-leverage-strong"}
    if enable_profiled_leverage and is_profiled(symbol):
        target_leverage = max(target_leverage, min(cp.optimal_leverage, max_leverage))
    elif settings.strategy_profile == "live-ultra-aggressive":
        if symbol == "ETHUSDT":
            target_leverage = min(target_leverage + 2, max_leverage)
        elif symbol == "BTCUSDT":
            target_leverage = min(target_leverage + 1, max_leverage)
    soft_leverage = max(1, target_leverage - 1)
    if settings.strategy_profile == "live-ultra-aggressive":
        # In one-shot live mode we avoid automatic 1x leverage downshifts on
        # "soft" setups; sizing controls should handle downside, not leverage decay.
        soft_leverage = target_leverage
    edge_to_cost_multiple = _edge_to_cost_multiple(
        net_expected_edge_bps=net_expected_edge_bps,
        estimated_round_trip_cost_bps=estimated_round_trip_cost_bps,
    )
    strong_setup = (
        predictability_score >= max(thresholds.futures_score_min + exposure.strong_score_buffer + 12.0, 72.0)
        and trend_strength >= max(exposure.strong_trend_strength_min, 0.8)
        and volume_confirmation >= max(exposure.strong_volume_confirmation_min, 0.72)
        and liquidity_score >= max(exposure.strong_liquidity_min, 0.82)
        and volatility_penalty <= min(exposure.strong_volatility_penalty_max + 0.02, 0.35)
        and overheat_penalty <= min(exposure.strong_overheat_penalty_max + 0.02, 0.3)
        and net_expected_edge_bps >= max(exposure.min_entry_net_edge_bps + 8.0, 12.0)
        and edge_to_cost_multiple >= max(exposure.strong_edge_to_cost_multiple_min, 1.8)
    )
    soft_setup = (
        predictability_score < thresholds.futures_score_min + 2.0
        or trend_strength < thresholds.futures_trend_strength_min + 0.04
        or volume_confirmation < 0.58
        or liquidity_score < thresholds.futures_liquidity_min + 0.02
        or volatility_penalty > thresholds.futures_volatility_penalty_max
        or overheat_penalty > thresholds.futures_overheat_penalty_max
        or net_expected_edge_bps < max(exposure.reduced_entry_net_edge_bps, 4.0)
        or edge_to_cost_multiple < max(1.25, settings.cost_gate.edge_to_cost_multiple_min - 0.1)
    )
    # ADX-based dynamic leverage: 374d validated per-coin profiles
    adx_cross_aligned = (
        is_profiled(symbol)
        and adx_1h >= cp.adx_floor
        and ema_cross_signal != 0
        and ema_cross_signal == trend_direction
        and (trend_direction > 0 or cp.side_filter != "long")
    )
    if adx_cross_aligned:
        adx_norm = min((adx_1h - cp.adx_floor) / 17.0, 1.0)
        adx_leverage = target_leverage + int(round((max_leverage - target_leverage) * adx_norm))
        adx_leverage = max(target_leverage, min(adx_leverage, max_leverage))
    else:
        adx_leverage = 0

    if strong_setup:
        return max_leverage
    if adx_leverage > 0:
        return max(target_leverage, adx_leverage)
    if soft_setup:
        return soft_leverage
    return target_leverage


def position_notional_and_stop_bps(
    *,
    last_trade_price: float,
    atr_14_1h_bps: float,
    equity_usd: float,
    remaining_portfolio_capacity_usd: float,
    settings: Settings,
    size_multiplier: float = 1.0,
    leverage_multiplier: float = 1.0,
    symbol: str = "",
    side: str = "long",
) -> tuple[float, float]:
    # Per-coin SL ATR multiplier — use short-specific if available
    cp = get_profile(symbol)
    if side == "short" and cp.short_sl_mult > 0:
        sl_mult = cp.short_sl_mult
    elif is_profiled(symbol):
        sl_mult = cp.sl_atr_mult
    else:
        sl_mult = settings.sizing.atr_multiple_for_stop
    stop_distance_bps = max(
        sl_mult * atr_14_1h_bps,
        settings.sizing.stop_floor_bps,
        1.0,  # absolute minimum: 1 bps to prevent division by zero
    )
    risk_dollars = equity_usd * settings.risk.per_trade_equity_risk
    adjusted_size_multiplier = max(size_multiplier, 0.0)
    adjusted_leverage_multiplier = max(leverage_multiplier, 1.0)
    raw_notional_usd = (
        risk_dollars / (stop_distance_bps / 10000.0) * adjusted_size_multiplier
        * adjusted_leverage_multiplier
    )
    symbol_cap_multiplier = max(adjusted_size_multiplier, 1.0)
    capped_notional = min(
        raw_notional_usd,
        equity_usd
        * settings.risk.max_symbol_notional_fraction
        * symbol_cap_multiplier
        * adjusted_leverage_multiplier,
        remaining_portfolio_capacity_usd,
    )
    # Floor: minimum viable notional = equity × 50% × leverage (never go below)
    if is_profiled(symbol) and adjusted_leverage_multiplier > 1:
        notional_floor = equity_usd * 0.5 * adjusted_leverage_multiplier
        capped_notional = max(capped_notional, min(notional_floor, remaining_portfolio_capacity_usd))
    return round(capped_notional, 6), round(stop_distance_bps, 6)


def quantity_from_notional(notional_usd: float, reference_price: float) -> float:
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    return round(notional_usd / reference_price, 8)
