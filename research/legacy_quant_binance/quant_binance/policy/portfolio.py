from __future__ import annotations

from uuid import uuid4

from quant_binance.models import DecisionIntent, PortfolioIntent, StrategyPrediction


def build_portfolio_intent(
    *,
    prediction: StrategyPrediction,
    selected_mode: str,
    side: str,
    target_notional_usd: float,
    stop_distance_bps: float,
    target_leverage: float = 0.0,
    strategy_size_multiplier: float = 1.0,
    entry_relaxation_reasons: tuple[str, ...] = (),
    size_boost_reasons: tuple[str, ...] = (),
    rejection_reasons: tuple[str, ...] = (),
    divergence_code: str = "",
) -> PortfolioIntent:
    mode_prediction = prediction.futures if selected_mode == "futures" else prediction.spot
    return PortfolioIntent(
        intent_id=str(uuid4()),
        prediction_id=prediction.prediction_id,
        snapshot_id=prediction.snapshot_id,
        config_version=prediction.config_version,
        timestamp=prediction.timestamp,
        symbol=prediction.symbol,
        candidate_mode=prediction.candidate_mode,
        selected_mode=selected_mode,
        side=side,
        target_notional_usd=target_notional_usd,
        stop_distance_bps=stop_distance_bps,
        target_leverage=target_leverage,
        strategy_size_multiplier=strategy_size_multiplier,
        entry_relaxation_reasons=entry_relaxation_reasons,
        size_boost_reasons=size_boost_reasons,
        rejection_reasons=rejection_reasons,
        divergence_code=divergence_code,
        predictability_score=mode_prediction.predictability_score,
        gross_expected_edge_bps=mode_prediction.gross_expected_edge_bps,
        net_expected_edge_bps=mode_prediction.net_expected_edge_bps,
        estimated_round_trip_cost_bps=mode_prediction.estimated_round_trip_cost_bps,
        trend_direction=mode_prediction.trend_direction,
        trend_strength=mode_prediction.trend_strength,
        volume_confirmation=mode_prediction.volume_confirmation,
        liquidity_score=mode_prediction.liquidity_score,
        volatility_penalty=mode_prediction.volatility_penalty,
        overheat_penalty=mode_prediction.overheat_penalty,
        macro_regime=mode_prediction.macro_regime,
        macro_trade_restraint=mode_prediction.macro_trade_restraint,
        macro_size_multiplier=mode_prediction.macro_size_multiplier,
        macro_leverage_cap=mode_prediction.macro_leverage_cap,
        macro_symbol_bias=mode_prediction.macro_symbol_bias,
    )


def decision_from_portfolio_intent(*, intent: PortfolioIntent) -> DecisionIntent:
    payload = {
        "snapshot_id": intent.snapshot_id,
        "config_version": intent.config_version,
        "final_mode": intent.selected_mode,
        "side": intent.side,
        "predictability_score": intent.predictability_score,
        "reasons": intent.rejection_reasons,
    }
    from quant_binance.observability.decision_log import hash_decision_payload

    return DecisionIntent(
        decision_id=str(uuid4()),
        decision_hash=hash_decision_payload(payload),
        snapshot_id=intent.snapshot_id,
        config_version=intent.config_version,
        timestamp=intent.timestamp,
        symbol=intent.symbol,
        candidate_mode=intent.candidate_mode,
        final_mode=intent.selected_mode,
        side=intent.side,
        trend_direction=intent.trend_direction,
        trend_strength=intent.trend_strength,
        volume_confirmation=intent.volume_confirmation,
        liquidity_score=intent.liquidity_score,
        volatility_penalty=intent.volatility_penalty,
        overheat_penalty=intent.overheat_penalty,
        predictability_score=intent.predictability_score,
        gross_expected_edge_bps=intent.gross_expected_edge_bps,
        net_expected_edge_bps=intent.net_expected_edge_bps,
        estimated_round_trip_cost_bps=intent.estimated_round_trip_cost_bps,
        order_intent_notional_usd=intent.target_notional_usd,
        stop_distance_bps=intent.stop_distance_bps,
        strategy_size_multiplier=intent.strategy_size_multiplier,
        entry_relaxation_reasons=intent.entry_relaxation_reasons,
        size_boost_reasons=intent.size_boost_reasons,
        macro_regime=intent.macro_regime,
        macro_trade_restraint=intent.macro_trade_restraint,
        macro_size_multiplier=intent.macro_size_multiplier,
        macro_leverage_cap=intent.macro_leverage_cap,
        macro_symbol_bias=intent.macro_symbol_bias,
        rejection_reasons=intent.rejection_reasons,
        divergence_code=intent.divergence_code,
        planned_leverage=int(intent.target_leverage),
    )
