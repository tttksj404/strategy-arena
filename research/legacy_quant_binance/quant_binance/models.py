from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class FeatureVector:
    ret_rank_1h: float
    ret_rank_4h: float
    breakout_norm: float
    ema_stack_score: float
    vol_z_5m_norm: float
    vol_z_1h_norm: float
    taker_imbalance_norm: float
    spread_bps_norm: float
    probe_slippage_bps_norm: float
    depth_10bps_norm: float
    book_stability_norm: float
    realized_vol_1h_norm: float
    realized_vol_4h_norm: float
    vol_shock_norm: float
    funding_abs_percentile: float
    oi_surge_percentile: float
    basis_stretch_percentile: float
    regime_alignment: float
    trend_direction: int
    trend_strength: float
    volume_confirmation: float
    liquidity_score: float
    volatility_penalty: float
    overheat_penalty: float
    support_alignment: float = 0.0
    resistance_penalty: float = 0.0
    macro_regime: str = "neutral"
    macro_risk_penalty: float = 0.0
    macro_liquidity_support_score: float = 0.5
    macro_event_risk_score: float = 0.0
    macro_trade_restraint: str = "none"
    macro_size_multiplier: float = 1.0
    macro_leverage_cap: int = 0
    macro_symbol_bias: str = "neutral"
    macro_directional_bearish_score: float = 0.0
    macro_execution_risk_score: float = 0.0
    sentiment_regime: str = "neutral"
    sentiment_support_score: float = 0.5
    alt_market_regime: str = "neutral"
    alt_breadth_score: float = 0.5
    alt_liquidity_support_score: float = 0.5
    alt_fundamental_score: float = 0.5
    alt_smart_money_score: float = 0.5
    alt_rotation_penalty: float = 0.0
    intraday_trend_direction: int = 0
    intraday_trend_strength: float = 0.0
    adx_1h: float = 0.0
    ema_cross_signal: int = 0  # +1 = fast crossed above slow, -1 = below, 0 = none
    pullback_signal: int = 0   # +1 = long pullback (oversold recovery in uptrend), -1 = short, 0 = none
    atr_14_1h_bps: float = 0.0  # actual 1h ATR-14 in basis points
    oi_divergence_score: float = 0.0  # -1 to +1: OI-price divergence (negative = fake breakout)
    vwap_price: float = 0.0  # current session VWAP price
    vwap_deviation_z: float = 0.0  # z-score of price deviation from VWAP (-4 to +4)
    smc_fvg_score: float = 0.0  # 0-1: Fair Value Gap quality
    smc_ob_score: float = 0.0  # 0-1: Order Block quality
    smc_structure_score: float = 0.0  # 0-1: BOS/CHoCH confirmation
    smc_composite_score: float = 0.0  # 0-1: weighted SMC composite
    predictability_score: float = 0.0
    gross_expected_edge_bps: float = 0.0
    net_expected_edge_bps: float = 0.0
    estimated_round_trip_cost_bps: float = 0.0
    empirical_fee_bps: float = 0.0
    empirical_entry_slippage_bps: float = 0.0
    empirical_exit_slippage_bps: float = 0.0
    b3_msb_signal: int = 0  # -1=short breakout, 0=no signal, 1=long breakout
    b3_msb_strength: float = 0.0  # 0~1, breakout confidence

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketSnapshot:
    snapshot_id: str
    config_version: str
    snapshot_schema_version: str
    symbol: str
    decision_time: datetime
    last_trade_price: float
    best_bid: float
    best_ask: float
    funding_rate: float
    open_interest: float
    basis_bps: float
    data_freshness_ms: int
    feature_values: FeatureVector

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision_time"] = self.decision_time.isoformat()
        return data


@dataclass(frozen=True)
class ModePrediction:
    mode: str
    side: str
    predictability_score: float
    gross_expected_edge_bps: float
    net_expected_edge_bps: float
    estimated_round_trip_cost_bps: float
    trend_direction: int
    trend_strength: float
    volume_confirmation: float
    liquidity_score: float
    volatility_penalty: float
    overheat_penalty: float
    macro_regime: str = "neutral"
    macro_trade_restraint: str = "none"
    macro_size_multiplier: float = 1.0
    macro_leverage_cap: int = 0
    macro_symbol_bias: str = "neutral"
    eligibility_reasons: tuple[str, ...] = field(default_factory=tuple)
    constraint_reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyPrediction:
    prediction_id: str
    snapshot_id: str
    config_version: str
    timestamp: datetime
    symbol: str
    candidate_mode: str
    spot: ModePrediction
    futures: ModePrediction
    selected_mode_hint: str = ""
    prediction_schema_version: str = "1.0.0"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(frozen=True)
class PortfolioIntent:
    intent_id: str
    prediction_id: str
    snapshot_id: str
    config_version: str
    timestamp: datetime
    symbol: str
    candidate_mode: str
    selected_mode: str
    side: str
    target_notional_usd: float
    stop_distance_bps: float
    target_leverage: float = 0.0
    strategy_size_multiplier: float = 1.0
    entry_relaxation_reasons: tuple[str, ...] = field(default_factory=tuple)
    size_boost_reasons: tuple[str, ...] = field(default_factory=tuple)
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    divergence_code: str = ""
    predictability_score: float = 0.0
    gross_expected_edge_bps: float = 0.0
    net_expected_edge_bps: float = 0.0
    estimated_round_trip_cost_bps: float = 0.0
    trend_direction: int = 0
    trend_strength: float = 0.0
    volume_confirmation: float = 0.0
    liquidity_score: float = 0.0
    volatility_penalty: float = 0.0
    overheat_penalty: float = 0.0
    macro_regime: str = "neutral"
    macro_trade_restraint: str = "none"
    macro_size_multiplier: float = 1.0
    macro_leverage_cap: int = 0
    macro_symbol_bias: str = "neutral"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(frozen=True)
class DecisionIntent:
    decision_id: str
    decision_hash: str
    snapshot_id: str
    config_version: str
    timestamp: datetime
    symbol: str
    candidate_mode: str
    final_mode: str
    side: str
    trend_direction: int
    trend_strength: float
    volume_confirmation: float
    liquidity_score: float
    volatility_penalty: float
    overheat_penalty: float
    predictability_score: float
    gross_expected_edge_bps: float
    net_expected_edge_bps: float
    estimated_round_trip_cost_bps: float
    order_intent_notional_usd: float
    stop_distance_bps: float
    strategy_size_multiplier: float = 1.0
    entry_relaxation_reasons: tuple[str, ...] = field(default_factory=tuple)
    size_boost_reasons: tuple[str, ...] = field(default_factory=tuple)
    macro_regime: str = "neutral"
    macro_trade_restraint: str = "none"
    macro_size_multiplier: float = 1.0
    macro_leverage_cap: int = 0
    macro_symbol_bias: str = "neutral"
    execution_quality_sample_size: int = 0
    execution_quality_size_multiplier: float = 1.0
    execution_quality_leverage_multiplier: float = 1.0
    execution_quality_edge_penalty_bps: float = 0.0
    execution_quality_entry_threshold_bps: float = 0.0
    execution_quality_expected_profit_floor_bps: float = 0.0
    execution_quality_trade_restraint: str = "none"
    execution_quality_avg_slippage_bps: float = 0.0
    execution_quality_avg_fill_ratio: float = 1.0
    execution_quality_avg_realized_edge_bps: float = 0.0
    execution_quality_avg_edge_retention_ratio: float = 1.0
    execution_quality_reject_rate: float = 0.0
    execution_quality_timeout_rate: float = 0.0
    execution_quality_partial_fill_rate: float = 0.0
    execution_quality_protection_degraded_rate: float = 0.0
    linked_order_ids: tuple[str, ...] = field(default_factory=tuple)
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    exit_reason_code: str = ""
    divergence_code: str = ""
    execution_symbol: str = ""
    planned_leverage: int = 0
    spot_base_asset: str = ""
    spot_quote_asset: str = ""
    spot_funding_asset: str = ""
    spot_quote_asset_usd_price: float = 0.0
    # Cross-coin OI quadrant gate metadata (Phase 1+2, opt-in via QUANT_CROSS_COIN_GATE).
    # All fields default-empty so legacy code paths and disabled-gate runs stay byte-compatible.
    cross_coin_own_quadrant: str = ""
    cross_coin_own_dir: str = ""
    cross_coin_leader_symbol: str = ""
    cross_coin_leader_quadrant: str = ""
    cross_coin_leader_dir: str = ""
    cross_coin_ev_bps: float = 0.0
    cross_coin_winrate: float = 0.0
    cross_coin_n: int = 0
    cross_coin_rank: int = 0
    cross_coin_blocker_reason: str = ""
    cross_coin_is_top_priority: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


ExecutionIntent = DecisionIntent
