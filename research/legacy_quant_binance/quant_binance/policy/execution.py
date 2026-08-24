from __future__ import annotations

from dataclasses import replace

from quant_binance.models import DecisionIntent, ExecutionIntent


def build_execution_intent(
    *,
    decision: DecisionIntent,
    final_mode: str | None = None,
    side: str | None = None,
    order_intent_notional_usd: float | None = None,
    stop_distance_bps: float | None = None,
    rejection_reasons: tuple[str, ...] | None = None,
    divergence_code: str | None = None,
    execution_symbol: str | None = None,
    spot_base_asset: str | None = None,
    spot_quote_asset: str | None = None,
    spot_funding_asset: str | None = None,
    spot_quote_asset_usd_price: float | None = None,
    linked_order_ids: tuple[str, ...] | None = None,
) -> ExecutionIntent:
    return replace(
        decision,
        final_mode=decision.final_mode if final_mode is None else final_mode,
        side=decision.side if side is None else side,
        order_intent_notional_usd=decision.order_intent_notional_usd if order_intent_notional_usd is None else order_intent_notional_usd,
        stop_distance_bps=decision.stop_distance_bps if stop_distance_bps is None else stop_distance_bps,
        rejection_reasons=decision.rejection_reasons if rejection_reasons is None else rejection_reasons,
        divergence_code=decision.divergence_code if divergence_code is None else divergence_code,
        execution_symbol=decision.execution_symbol if execution_symbol is None else execution_symbol,
        spot_base_asset=decision.spot_base_asset if spot_base_asset is None else spot_base_asset,
        spot_quote_asset=decision.spot_quote_asset if spot_quote_asset is None else spot_quote_asset,
        spot_funding_asset=decision.spot_funding_asset if spot_funding_asset is None else spot_funding_asset,
        spot_quote_asset_usd_price=decision.spot_quote_asset_usd_price if spot_quote_asset_usd_price is None else spot_quote_asset_usd_price,
        linked_order_ids=decision.linked_order_ids if linked_order_ids is None else linked_order_ids,
    )


def decision_from_execution_intent(*, intent: ExecutionIntent) -> DecisionIntent:
    return intent


def is_major_futures_symbol(symbol: str, *, major_symbols: tuple[str, ...]) -> bool:
    return symbol in set(major_symbols)


def is_major_strong_futures_decision(*, decision: DecisionIntent, major_symbols: tuple[str, ...], exposure: object, thresholds: object) -> bool:
    if decision.final_mode != "futures" or not is_major_futures_symbol(decision.symbol, major_symbols=major_symbols):
        return False
    edge_to_cost_multiple = float("inf") if decision.estimated_round_trip_cost_bps <= 0.0 else decision.gross_expected_edge_bps / decision.estimated_round_trip_cost_bps
    min_strong_net_edge_bps = max(exposure.min_entry_net_edge_bps, exposure.pyramid_min_net_edge_bps)
    return (
        decision.predictability_score >= thresholds.futures_score_min + exposure.strong_score_buffer
        and decision.trend_strength >= exposure.strong_trend_strength_min
        and decision.volume_confirmation >= exposure.strong_volume_confirmation_min
        and decision.liquidity_score >= exposure.strong_liquidity_min
        and decision.volatility_penalty <= exposure.strong_volatility_penalty_max
        and decision.overheat_penalty <= exposure.strong_overheat_penalty_max
        and decision.net_expected_edge_bps >= min_strong_net_edge_bps
        and edge_to_cost_multiple >= exposure.strong_edge_to_cost_multiple_min
    )


def is_major_medium_futures_decision(*, decision: DecisionIntent, major_symbols: tuple[str, ...], exposure: object, thresholds: object) -> bool:
    if decision.final_mode != "futures" or not is_major_futures_symbol(decision.symbol, major_symbols=major_symbols):
        return False
    if is_major_strong_futures_decision(decision=decision, major_symbols=major_symbols, exposure=exposure, thresholds=thresholds):
        return False
    edge_to_cost_multiple = float("inf") if decision.estimated_round_trip_cost_bps <= 0.0 else decision.gross_expected_edge_bps / decision.estimated_round_trip_cost_bps
    return (
        decision.predictability_score >= exposure.pyramid_min_predictability_score
        and decision.trend_strength >= exposure.pyramid_min_trend_strength
        and decision.volume_confirmation >= exposure.pyramid_min_volume_confirmation
        and decision.liquidity_score >= exposure.soft_liquidity_floor
        and decision.net_expected_edge_bps >= max(exposure.min_entry_net_edge_bps, exposure.pyramid_min_net_edge_bps)
        and edge_to_cost_multiple >= max(exposure.priority_edge_to_cost_multiple_min, 1.0)
    )
