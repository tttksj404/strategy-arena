from __future__ import annotations

from quant_binance.models import FeatureVector, MarketSnapshot


REQUIRED_FEATURE_KEYS = (
    "ret_rank_1h",
    "ret_rank_4h",
    "breakout_norm",
    "ema_stack_score",
    "vol_z_5m_norm",
    "vol_z_1h_norm",
    "taker_imbalance_norm",
    "spread_bps_norm",
    "probe_slippage_bps_norm",
    "depth_10bps_norm",
    "book_stability_norm",
    "realized_vol_1h_norm",
    "realized_vol_4h_norm",
    "vol_shock_norm",
    "funding_abs_percentile",
    "oi_surge_percentile",
    "basis_stretch_percentile",
    "regime_alignment",
    "trend_strength",
    "volume_confirmation",
    "liquidity_score",
    "volatility_penalty",
    "overheat_penalty",
    "predictability_score",
    "gross_expected_edge_bps",
    "net_expected_edge_bps",
    "estimated_round_trip_cost_bps",
)


def validate_feature_schema(features: FeatureVector) -> None:
    feature_data = features.as_dict()
    missing = [key for key in REQUIRED_FEATURE_KEYS if key not in feature_data]
    if missing:
        raise ValueError(f"missing required feature keys: {missing}")


def validate_snapshot(snapshot: MarketSnapshot) -> None:
    if not snapshot.snapshot_id:
        raise ValueError("snapshot_id is required")
    if not snapshot.config_version:
        raise ValueError("config_version is required")
    if not snapshot.snapshot_schema_version:
        raise ValueError("snapshot_schema_version is required")
    if snapshot.data_freshness_ms < 0:
        raise ValueError("data_freshness_ms must be non-negative")
    validate_feature_schema(snapshot.feature_values)
