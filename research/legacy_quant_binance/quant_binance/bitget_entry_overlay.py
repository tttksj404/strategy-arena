from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


DEFAULT_BITGET_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "PEPEUSDT",
)

BITGET_TAKER_FEE_BPS = 6.0


@dataclass(frozen=True)
class BitgetMarketMetrics:
    symbol: str
    last_price: float
    quote_volume_24h: float
    change_24h_pct: float
    spread_bps: float
    funding_pct: float
    mark_basis_bps: float
    ema20_60_bps: float
    rv15_bps: float
    taker_buy_ratio_6h: float | None = None
    taker_buy_minus_sell_6h: float | None = None
    taker_error: str | None = None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def slippage_buffer_bps(metrics: BitgetMarketMetrics) -> float:
    if metrics.quote_volume_24h >= 1_000_000_000:
        base = 0.8
    elif metrics.quote_volume_24h >= 250_000_000:
        base = 1.6
    elif metrics.quote_volume_24h >= 75_000_000:
        base = 2.4
    else:
        base = 3.8
    volatility_add = min(max(metrics.rv15_bps - 25.0, 0.0) * 0.06, 1.8)
    spread_add = min(metrics.spread_bps * 0.75, 1.5)
    return round(base + volatility_add + spread_add, 6)


def effective_round_trip_cost_bps(metrics: BitgetMarketMetrics) -> float:
    # Base assumption is taker entry + taker exit on Bitget USDT futures.
    return round(2.0 * BITGET_TAKER_FEE_BPS + metrics.spread_bps + slippage_buffer_bps(metrics), 6)


def _symbol_volume_floor(symbol: str) -> float:
    if symbol in {"BTCUSDT", "ETHUSDT"}:
        return 750_000_000.0
    if symbol in {"SOLUSDT", "XRPUSDT"}:
        return 120_000_000.0
    return 30_000_000.0


def _symbol_spread_cap(symbol: str) -> float:
    return {
        "BTCUSDT": 0.08,
        "ETHUSDT": 0.10,
        "SOLUSDT": 0.25,
        "XRPUSDT": 0.85,
        "DOGEUSDT": 1.20,
        "PEPEUSDT": 0.45,
    }.get(symbol, 0.50)


def _counterfactual_miss_rate(counterfactual_summary: dict[str, Any] | None) -> float:
    summary = counterfactual_summary or {}
    decisions = max(int(safe_float(summary.get("decision_count"), 0.0)), 0)
    labels = dict(summary.get("label_counts") or {})
    possible = int(safe_float(labels.get("possible_missed_entry"), 0.0))
    return 0.0 if decisions <= 0 else possible / decisions


def _copy_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(profile or {}))


def _nudge(profile: dict[str, Any], field: str, delta: float, *, floor: float | None = None, ceiling: float | None = None) -> None:
    current = safe_float(profile.get(field), 0.0)
    value = current + delta
    if floor is not None:
        value = max(value, floor)
    if ceiling is not None:
        value = min(value, ceiling)
    profile[field] = round(value, 6)


def tune_symbol_profile(
    *,
    metrics: BitgetMarketMetrics,
    baseline_profile: dict[str, Any],
    counterfactual_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    symbol = metrics.symbol.upper()
    tuned = _copy_profile(baseline_profile)
    reasons: list[str] = []
    warnings: list[str] = []
    cost_bps = effective_round_trip_cost_bps(metrics)

    volume_ok = metrics.quote_volume_24h >= _symbol_volume_floor(symbol)
    spread_ok = metrics.spread_bps <= _symbol_spread_cap(symbol)
    trend_ok = metrics.ema20_60_bps >= (8.0 if symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT"} else 20.0)
    basis_ok = metrics.mark_basis_bps <= -2.0
    funding_long_support = metrics.funding_pct <= 0.0
    taker_ratio = metrics.taker_buy_ratio_6h
    taker_ok = taker_ratio is None or taker_ratio >= 0.52
    taker_sell_warning = taker_ratio is not None and taker_ratio < 0.49
    miss_rate = _counterfactual_miss_rate(counterfactual_summary)
    repeated_misses = miss_rate >= 0.08

    if not volume_ok:
        warnings.append("low_bitget_quote_volume")
    if not spread_ok:
        warnings.append("spread_near_or_above_symbol_cap")
    if taker_sell_warning:
        warnings.append("recent_taker_flow_sell_dominant")
    if metrics.taker_error:
        warnings.append("taker_flow_unavailable")

    long_overlay_score = 0
    for ok, reason in (
        (volume_ok, "volume_ok"),
        (spread_ok, "spread_ok"),
        (trend_ok, "trend_supports_long"),
        (basis_ok, "mark_below_index"),
        (funding_long_support, "funding_not_crowded_long"),
        (taker_ok, "taker_flow_not_blocking"),
    ):
        if ok:
            long_overlay_score += 1
            reasons.append(reason)

    action = "hold_shadow"
    side_bias = "long" if trend_ok else "neutral"
    size_multiplier = 1.0

    if symbol == "BTCUSDT":
        action = "confirm_only"
        size_multiplier = 1.0
    elif symbol == "ETHUSDT":
        action = "hold_shadow" if metrics.funding_pct > 0.0 or metrics.change_24h_pct < -1.0 else "confirm_only"
        size_multiplier = 0.85
    elif symbol == "SOLUSDT":
        action = "relax_one_step" if long_overlay_score >= 5 and spread_ok else "confirm_only"
        size_multiplier = 0.85
    elif symbol == "XRPUSDT":
        action = "relax_selectively" if long_overlay_score >= 4 and trend_ok and taker_ok else "confirm_only"
        size_multiplier = 0.75
    elif symbol == "DOGEUSDT":
        action = "confirm_only" if taker_sell_warning else "relax_selectively"
        size_multiplier = 0.65
    elif symbol == "PEPEUSDT":
        action = "relax_one_step" if long_overlay_score >= 4 and trend_ok and funding_long_support else "confirm_only"
        size_multiplier = 0.60

    if repeated_misses and action in {"confirm_only", "hold_shadow"} and long_overlay_score >= 5:
        action = "relax_selectively"
        reasons.append("counterfactual_repeated_missed_entries")

    if action == "relax_one_step":
        _nudge(tuned, "min_predictability_score", -1.0, floor=65.0)
        _nudge(tuned, "min_volume_confirmation", -0.01, floor=0.44)
        _nudge(tuned, "min_edge_to_cost", -0.05, floor=1.8)
        _nudge(tuned, "min_net_edge_bps", -1.0, floor=max(20.0, cost_bps * 1.45))
    elif action == "relax_selectively":
        _nudge(tuned, "min_predictability_score", -0.5, floor=65.0)
        _nudge(tuned, "min_volume_confirmation", -0.005, floor=0.44)
        _nudge(tuned, "min_net_edge_bps", 0.0, floor=max(20.0, cost_bps * 1.55))
    elif action == "hold_shadow":
        _nudge(tuned, "min_predictability_score", 0.5, ceiling=84.0)
    elif action == "confirm_only":
        _nudge(tuned, "min_net_edge_bps", 0.5, floor=max(20.0, cost_bps * 1.55))

    if taker_sell_warning:
        _nudge(tuned, "min_net_edge_bps", 1.0)
        _nudge(tuned, "min_edge_to_cost", 0.05)

    tuned["size_multiplier"] = round(min(safe_float(tuned.get("size_multiplier"), 1.0), size_multiplier), 6)

    return {
        "symbol": symbol,
        "action": action,
        "side_bias": side_bias,
        "effective_round_trip_cost_bps": cost_bps,
        "slippage_buffer_bps": slippage_buffer_bps(metrics),
        "long_overlay_score": long_overlay_score,
        "counterfactual_miss_rate": round(miss_rate, 6),
        "reasons": reasons,
        "warnings": warnings,
        "baseline_profile": _copy_profile(baseline_profile),
        "tuned_profile": tuned,
    }


def apply_tuned_profiles(base_config: dict[str, Any], tuning_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tuned_config = copy.deepcopy(base_config)
    profiles = dict(tuned_config.get("symbol_filter_profiles") or {})
    for row in tuning_rows:
        symbol = str(row.get("symbol") or "").upper()
        profile = row.get("tuned_profile")
        if symbol and isinstance(profile, dict):
            profiles[symbol] = copy.deepcopy(profile)
    tuned_config["symbol_filter_profiles"] = profiles
    tuned_config["bitget_entry_overlay"] = {
        "enabled": True,
        "paper_only": True,
        "description": "Generated by quant_bitget_overlay_tuner.py; use as read-only paper override until forward outcomes validate it.",
        "symbols": [str(row.get("symbol") or "").upper() for row in tuning_rows],
    }
    return tuned_config
