from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from quant_binance.models import FeatureVector
from quant_binance.settings import Settings
from quant_binance.strategy.normalize import clamp, midpoint_percentile_rank, zscore_to_unit


@dataclass(frozen=True)
class FeatureHistoryContext:
    returns_1h: tuple[float, ...]
    returns_4h: tuple[float, ...]
    quote_volume_5m: tuple[float, ...]
    quote_volume_1h: tuple[float, ...]
    realized_vol_1h: tuple[float, ...]
    realized_vol_4h: tuple[float, ...]
    funding_abs: tuple[float, ...]
    basis_abs: tuple[float, ...]
    oi_surge: tuple[float, ...]


@dataclass(frozen=True)
class PrimitiveInputs:
    ret_1h: float
    ret_4h: float
    trend_direction: int
    ema_stack_score: float
    breakout_reference_price: float
    last_trade_price: float
    atr_14_1h_price: float
    quote_volume_5m: float
    quote_volume_1h: float
    buy_taker_volume: float
    sell_taker_volume: float
    spread_bps: float
    probe_slippage_bps: float
    depth_usd_within_10bps: float
    order_book_imbalance_std: float
    realized_vol_1h: float
    realized_vol_4h: float
    median_realized_vol_1h_30d: float
    funding_rate: float
    open_interest: float
    open_interest_ema: float
    basis_bps: float
    gross_expected_edge_bps: float
    intraday_trend_direction: int = 0
    intraday_trend_strength: float = 0.0
    adx_1h: float = 0.0
    ema_cross_signal: int = 0


def _safe_mean(values: tuple[float, ...]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: tuple[float, ...]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def build_feature_vector_from_primitives(
    *,
    inputs: PrimitiveInputs,
    history: FeatureHistoryContext,
    settings: Settings,
) -> FeatureVector:
    thresholds = settings.feature_thresholds
    ret_rank_1h = midpoint_percentile_rank(inputs.ret_1h, history.returns_1h)
    ret_rank_4h = midpoint_percentile_rank(inputs.ret_4h, history.returns_4h)
    breakout_norm = clamp(
        abs(inputs.last_trade_price - inputs.breakout_reference_price)
        / max(0.75 * inputs.atr_14_1h_price, 0.0025 * inputs.last_trade_price)
    )
    vol_z_5m_norm = zscore_to_unit(
        inputs.quote_volume_5m,
        _safe_mean(history.quote_volume_5m),
        _safe_std(history.quote_volume_5m),
    )
    vol_z_1h_norm = zscore_to_unit(
        inputs.quote_volume_1h,
        _safe_mean(history.quote_volume_1h),
        _safe_std(history.quote_volume_1h),
    )
    taker_imbalance_norm = clamp(
        (
            (inputs.buy_taker_volume - inputs.sell_taker_volume)
            / max(inputs.buy_taker_volume + inputs.sell_taker_volume, 1e-9)
        )
        * 0.5
        + 0.5
    )
    spread_bps_norm = clamp(inputs.spread_bps / max(thresholds.spread_bps_ceiling, 1e-6))
    probe_slippage_bps_norm = clamp(inputs.probe_slippage_bps / max(thresholds.slippage_bps_ceiling, 1e-6))
    depth_10bps_norm = clamp(inputs.depth_usd_within_10bps / max(thresholds.depth_usd_target, 1e-6))
    book_stability_norm = 1.0 - clamp(
        inputs.order_book_imbalance_std / max(thresholds.order_book_imbalance_std_ceiling, 1e-6)
    )
    realized_vol_1h_norm = midpoint_percentile_rank(inputs.realized_vol_1h, history.realized_vol_1h)
    realized_vol_4h_norm = midpoint_percentile_rank(inputs.realized_vol_4h, history.realized_vol_4h)
    vol_shock_norm = clamp(
        max(inputs.realized_vol_1h / max(inputs.median_realized_vol_1h_30d, 1e-9) - 1.0, 0.0)
        / max(thresholds.vol_shock_ceiling, 1e-6)
    )
    funding_abs_percentile = midpoint_percentile_rank(abs(inputs.funding_rate), history.funding_abs)
    oi_surge_value = max(inputs.open_interest / max(inputs.open_interest_ema, 1e-9) - 1.0, 0.0)
    oi_surge_percentile = midpoint_percentile_rank(oi_surge_value, history.oi_surge)
    basis_stretch_percentile = midpoint_percentile_rank(abs(inputs.basis_bps), history.basis_abs)
    intraday_alignment = (
        1.0
        if inputs.trend_direction != 0 and inputs.intraday_trend_direction == inputs.trend_direction
        else 0.0
        if inputs.intraday_trend_direction != 0 and inputs.intraday_trend_direction != inputs.trend_direction
        else 0.5
    )
    regime_alignment = clamp(
        (
            1.0 if inputs.trend_direction != 0 and inputs.ema_stack_score == 1.0 else 0.5 if inputs.trend_direction != 0 else 0.0
        ) * 0.75
        + intraday_alignment * max(inputs.intraday_trend_strength, 0.0) * 0.25
    )

    trend_strength = round(
        0.30 * ret_rank_1h
        + 0.25 * ret_rank_4h
        + 0.15 * breakout_norm
        + 0.15 * inputs.ema_stack_score
        + 0.15 * inputs.intraday_trend_strength,
        6,
    )
    volume_confirmation = round(
        0.36 * vol_z_5m_norm
        + 0.31 * vol_z_1h_norm
        + 0.23 * taker_imbalance_norm
        + 0.10 * max(inputs.intraday_trend_strength, 0.0),
        6,
    )
    liquidity_score = round(
        0.35 * (1.0 - spread_bps_norm)
        + 0.35 * depth_10bps_norm
        + 0.20 * (1.0 - probe_slippage_bps_norm)
        + 0.10 * book_stability_norm,
        6,
    )
    volatility_penalty = round(
        0.45 * realized_vol_1h_norm
        + 0.35 * realized_vol_4h_norm
        + 0.20 * vol_shock_norm,
        6,
    )
    overheat_penalty = round(
        0.40 * funding_abs_percentile
        + 0.35 * oi_surge_percentile
        + 0.25 * basis_stretch_percentile,
        6,
    )

    return FeatureVector(
        ret_rank_1h=ret_rank_1h,
        ret_rank_4h=ret_rank_4h,
        breakout_norm=breakout_norm,
        ema_stack_score=inputs.ema_stack_score,
        vol_z_5m_norm=vol_z_5m_norm,
        vol_z_1h_norm=vol_z_1h_norm,
        taker_imbalance_norm=taker_imbalance_norm,
        spread_bps_norm=spread_bps_norm,
        probe_slippage_bps_norm=probe_slippage_bps_norm,
        depth_10bps_norm=depth_10bps_norm,
        book_stability_norm=book_stability_norm,
        realized_vol_1h_norm=realized_vol_1h_norm,
        realized_vol_4h_norm=realized_vol_4h_norm,
        vol_shock_norm=vol_shock_norm,
        funding_abs_percentile=funding_abs_percentile,
        oi_surge_percentile=oi_surge_percentile,
        basis_stretch_percentile=basis_stretch_percentile,
        regime_alignment=regime_alignment,
        trend_direction=inputs.trend_direction,
        trend_strength=trend_strength,
        volume_confirmation=volume_confirmation,
        liquidity_score=liquidity_score,
        volatility_penalty=volatility_penalty,
        overheat_penalty=overheat_penalty,
        intraday_trend_direction=inputs.intraday_trend_direction,
        intraday_trend_strength=inputs.intraday_trend_strength,
        adx_1h=inputs.adx_1h,
        ema_cross_signal=inputs.ema_cross_signal,
        atr_14_1h_bps=round(
            (inputs.atr_14_1h_price / inputs.last_trade_price * 10000.0)
            if inputs.last_trade_price > 0 and inputs.atr_14_1h_price > 0
            else 0.0,
            6,
        ),
        gross_expected_edge_bps=inputs.gross_expected_edge_bps,
    )
