from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import uuid4

from quant_binance.models import DecisionIntent, FeatureVector, MarketSnapshot, ModePrediction, StrategyPrediction
from quant_binance.policy.portfolio import build_portfolio_intent, decision_from_portfolio_intent
from quant_binance.risk.sizing import position_notional_and_stop_bps, select_futures_leverage
from quant_binance.overlays import is_alt_symbol
from quant_binance.settings import Settings
from quant_binance.strategy.coin_profiles import get_profile, is_profiled
from quant_binance.strategy.scorer import apply_score_and_costs, passes_cost_gate


FUTURES_SOFT_RISK_REASONS = {"SCORE_TOO_LOW", "LIQUIDITY_TOO_WEAK", "VOL_TOO_HIGH", "FUTURES_OVERHEAT", "EDGE_BELOW_COST", "SENTIMENT_CAUTION"}
BTC_ETH_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})
BTC_ETH_STRONG_SIZE_BOOST_REASON = "BTC_ETH_STRONG_EDGE_SIZE_BOOST"
BTC_ETH_SOFT_SPOT_MISS_TOLERANCE = 0.12
BTC_ETH_SECONDARY_SUPPORT_RELAX_TOLERANCE = 0.18
BTC_ETH_SECONDARY_SENTIMENT_RELAX_TOLERANCE = 0.16
BTC_ETH_CONFIRMATION_INTRADAY_STRENGTH_MIN = 0.0
BTC_ETH_CONFIRMATION_SPOT_INTRADAY_STRENGTH_MIN = 0.0
BTC_SIZE_BOOST_MULTIPLIER_CAP = 1.18
BTC_SIZE_BOOST_ABS_CAP = 0.18
ETH_SIZE_BOOST_MULTIPLIER_CAP = 1.25
ETH_SIZE_BOOST_ABS_CAP = 0.25


def _candidate_mode(features: FeatureVector, settings: Settings) -> str:
    if features.predictability_score >= settings.mode_thresholds.futures_score_min:
        return "futures"
    if features.predictability_score >= settings.mode_thresholds.spot_score_min:
        return "spot"
    return "cash"


def _mode_prediction_from_features(*, mode: str, features: FeatureVector) -> ModePrediction:
    if mode == "spot":
        side = "long" if features.trend_direction > 0 else "flat"
    else:
        if features.trend_direction > 0:
            side = "long"
        elif features.trend_direction < 0:
            side = "short"
        else:
            side = "flat"
    return ModePrediction(
        mode=mode,
        side=side,
        predictability_score=features.predictability_score,
        gross_expected_edge_bps=features.gross_expected_edge_bps,
        net_expected_edge_bps=features.net_expected_edge_bps,
        estimated_round_trip_cost_bps=features.estimated_round_trip_cost_bps,
        trend_direction=features.trend_direction,
        trend_strength=features.trend_strength,
        volume_confirmation=features.volume_confirmation,
        liquidity_score=features.liquidity_score,
        volatility_penalty=features.volatility_penalty,
        overheat_penalty=features.overheat_penalty,
        macro_regime=features.macro_regime,
        macro_trade_restraint=features.macro_trade_restraint,
        macro_size_multiplier=features.macro_size_multiplier,
        macro_leverage_cap=features.macro_leverage_cap,
        macro_symbol_bias=features.macro_symbol_bias,
    )




def _major_prediction_specialist_adjustment(*, symbol: str, prediction: ModePrediction) -> ModePrediction:
    if symbol not in BTC_ETH_SYMBOLS or prediction.mode != "futures" or prediction.side == "flat":
        return prediction

    score_boost = 0.0
    edge_boost = 0.0
    if prediction.trend_strength >= 0.75 and prediction.volume_confirmation >= 0.68:
        score_boost += 1.5
        edge_boost += 1.0
    if prediction.liquidity_score >= 0.8 and prediction.volatility_penalty <= 0.32:
        score_boost += 1.0
        edge_boost += 0.75
    if prediction.estimated_round_trip_cost_bps > 0 and prediction.gross_expected_edge_bps / prediction.estimated_round_trip_cost_bps >= 1.8:
        score_boost += 1.0
        edge_boost += 0.75
    if prediction.macro_symbol_bias == "majors_only" and prediction.macro_regime == "supportive":
        score_boost += 1.0
        edge_boost += 0.5

    if score_boost == 0.0 and edge_boost == 0.0:
        return prediction
    return replace(
        prediction,
        predictability_score=round(min(100.0, prediction.predictability_score + score_boost), 6),
        gross_expected_edge_bps=round(prediction.gross_expected_edge_bps + edge_boost, 6),
        net_expected_edge_bps=round(prediction.net_expected_edge_bps + edge_boost, 6),
    )


def _apply_major_prediction_bias(*, symbol: str, prediction: ModePrediction) -> ModePrediction:
    if symbol not in BTC_ETH_SYMBOLS:
        return prediction

    score_boost = 0.0
    edge_boost = 0.0
    if prediction.macro_symbol_bias == "majors_only":
        score_boost += 3.0
        edge_boost += 2.0
    if prediction.macro_trade_restraint == "none":
        score_boost += 1.0
    if prediction.macro_regime == "supportive":
        score_boost += 1.5
        edge_boost += 1.0
    if prediction.mode == "futures" and prediction.side != "flat":
        score_boost += 1.0
        edge_boost += 0.5

    if score_boost == 0.0 and edge_boost == 0.0:
        return prediction

    return replace(
        prediction,
        predictability_score=round(min(100.0, prediction.predictability_score + score_boost), 6),
        gross_expected_edge_bps=round(prediction.gross_expected_edge_bps + edge_boost, 6),
        net_expected_edge_bps=round(prediction.net_expected_edge_bps + edge_boost, 6),
    )



def _select_major_candidate_mode(*, symbol: str, candidate_mode: str, spot_prediction: ModePrediction, futures_prediction: ModePrediction, settings: Settings) -> str:
    if symbol not in BTC_ETH_SYMBOLS:
        return candidate_mode
    if futures_prediction.side == "flat":
        return candidate_mode
    if futures_prediction.macro_trade_restraint not in {"none", "pre_event_reduce"}:
        return candidate_mode
    if futures_prediction.predictability_score >= settings.mode_thresholds.futures_score_min - 1.0 and futures_prediction.net_expected_edge_bps >= max(spot_prediction.net_expected_edge_bps + 1.0, 8.0):
        return "futures"
    return candidate_mode


def configured_funding_drag_bps(snapshot: MarketSnapshot, settings: Settings) -> float:
    exposure = settings.futures_exposure
    if not exposure.funding_bias_enabled:
        return 0.0
    direction = int(snapshot.feature_values.trend_direction)
    if direction not in {-1, 1}:
        return 0.0
    raw_drag_bps = float(snapshot.funding_rate or 0.0) * 10000.0 * float(direction)
    if abs(raw_drag_bps) < max(float(exposure.funding_bias_min_abs_bps), 0.0):
        return 0.0
    cap = max(float(exposure.funding_bias_max_abs_bps), 0.0)
    if cap > 0.0:
        raw_drag_bps = max(-cap, min(raw_drag_bps, cap))
    return round(raw_drag_bps, 6)


def build_strategy_prediction(
    snapshot: MarketSnapshot,
    settings: Settings,
    *,
    expected_funding_drag_bps: float | None = None,
) -> StrategyPrediction:
    funding_drag_bps = (
        configured_funding_drag_bps(snapshot, settings)
        if expected_funding_drag_bps is None
        else float(expected_funding_drag_bps)
    )
    futures_features = apply_score_and_costs(
        snapshot.feature_values,
        settings=settings,
        mode="futures",
        expected_funding_drag_bps=funding_drag_bps,
    )
    spot_features = apply_score_and_costs(
        snapshot.feature_values,
        settings=settings,
        mode="spot",
    )
    candidate_mode = _candidate_mode(futures_features, settings)
    spot_prediction = _apply_major_prediction_bias(symbol=snapshot.symbol, prediction=_mode_prediction_from_features(mode="spot", features=spot_features))
    futures_prediction = _major_prediction_specialist_adjustment(
        symbol=snapshot.symbol,
        prediction=_apply_major_prediction_bias(symbol=snapshot.symbol, prediction=_mode_prediction_from_features(mode="futures", features=futures_features)),
    )
    candidate_mode = _select_major_candidate_mode(
        symbol=snapshot.symbol,
        candidate_mode=candidate_mode,
        spot_prediction=spot_prediction,
        futures_prediction=futures_prediction,
        settings=settings,
    )
    selected_mode_hint = candidate_mode
    if candidate_mode == "cash" and spot_features.predictability_score >= settings.mode_thresholds.spot_score_min:
        selected_mode_hint = "spot"
    return StrategyPrediction(
        prediction_id=str(uuid4()),
        snapshot_id=snapshot.snapshot_id,
        config_version=settings.config_version,
        timestamp=snapshot.decision_time,
        symbol=snapshot.symbol,
        candidate_mode=candidate_mode,
        spot=spot_prediction,
        futures=futures_prediction,
        selected_mode_hint=selected_mode_hint,
    )


def _is_btc_eth_symbol(symbol: str) -> bool:
    return symbol in BTC_ETH_SYMBOLS


def _observe_only_reasons(features: FeatureVector, settings: Settings, symbol: str) -> tuple[bool, list[str]]:
    if not is_alt_symbol(symbol):
        return False, []
    rules = settings.symbol_eligibility
    if (
        features.estimated_round_trip_cost_bps >= rules.observe_only_cost_bps_min
        and (
            features.liquidity_score <= rules.observe_only_liquidity_max
            or features.alt_liquidity_support_score <= rules.observe_only_alt_liquidity_max
        )
    ):
        return True, ["OBSERVE_ONLY_SYMBOL", "LIQUIDITY_TOO_WEAK"]
    return False, []


def observe_only_reasons(features: FeatureVector, settings: Settings, symbol: str) -> tuple[str, ...]:
    observe_only, reasons = _observe_only_reasons(features, settings, symbol)
    if not observe_only:
        return ()
    return tuple(reasons)


def _edge_to_cost_multiple(features: FeatureVector) -> float:
    if features.estimated_round_trip_cost_bps <= 0:
        return float("inf")
    return features.gross_expected_edge_bps / features.estimated_round_trip_cost_bps


def _macro_risk_off(features: FeatureVector) -> bool:
    return features.macro_event_risk_score >= 0.6 or features.macro_risk_penalty >= 0.55 or features.macro_regime == "high_risk"


def _macro_event_halt(features: FeatureVector) -> bool:
    return features.macro_trade_restraint == "halt_high_impact_window" or features.macro_event_risk_score >= 0.85


def _macro_symbol_bias_reasons(features: FeatureVector, *, symbol: str) -> list[str]:
    if not is_alt_symbol(symbol):
        return []
    if _macro_event_halt(features):
        return ["MACRO_EVENT_WINDOW", "MACRO_MAJORS_ONLY"]
    if _macro_risk_off(features):
        return ["MACRO_MAJORS_ONLY"]
    return []


def _macro_futures_risk_controls(features: FeatureVector, *, symbol: str) -> tuple[bool, list[str], float, int]:
    reasons = _macro_symbol_bias_reasons(features, symbol=symbol)
    size_multiplier = max(features.macro_size_multiplier, 0.0)
    leverage_cap = max(int(features.macro_leverage_cap), 0)
    if _macro_event_halt(features):
        effective_cap = leverage_cap if leverage_cap > 0 else 1
        if is_alt_symbol(symbol):
            return False, reasons, 0.0, effective_cap
        return True, [], min(size_multiplier if size_multiplier > 0.0 else 0.35, 0.35), effective_cap
    if _macro_risk_off(features):
        if is_alt_symbol(symbol):
            return False, reasons, 0.0, max(leverage_cap, 2)
        return True, [], min(size_multiplier if size_multiplier > 0.0 else 0.7, 0.7), max(leverage_cap, 2)
    return True, [], size_multiplier if size_multiplier > 0.0 else 1.0, leverage_cap


def _macro_spot_risk_reasons(features: FeatureVector, *, symbol: str) -> list[str]:
    reasons = _macro_symbol_bias_reasons(features, symbol=symbol)
    if _macro_event_halt(features) and symbol not in {"BTCUSDT", "ETHUSDT"}:
        return reasons or ["MACRO_EVENT_WINDOW"]
    return reasons


def _is_objectively_strong_futures_setup(
    features: FeatureVector,
    settings: Settings,
    *,
    symbol: str = "",
) -> bool:
    thresholds = settings.mode_thresholds
    exposure = settings.futures_exposure
    strong_liquidity_min = max(exposure.strong_liquidity_min, 0.82)
    if symbol == "PEPEUSDT":
        # PEPE has its own validated high-beta profile; do not force it to clear the same
        # liquidity floor as majors if the rest of the setup is already strong.
        strong_liquidity_min = max(exposure.strong_liquidity_min, 0.50)
    return (
        features.predictability_score >= thresholds.futures_score_min + exposure.strong_score_buffer
        and features.trend_strength >= exposure.strong_trend_strength_min
        and features.volume_confirmation >= exposure.strong_volume_confirmation_min
        and features.liquidity_score >= strong_liquidity_min
        and features.volatility_penalty <= exposure.strong_volatility_penalty_max
        and features.overheat_penalty <= exposure.strong_overheat_penalty_max
        and features.macro_risk_penalty < settings.macro_gates.futures_block_penalty
        and features.sentiment_regime != "caution"
        and _edge_to_cost_multiple(features) >= exposure.strong_edge_to_cost_multiple_min
    )


def _is_objectively_medium_major_futures_setup(
    features: FeatureVector,
    settings: Settings,
    *,
    symbol: str,
) -> bool:
    exposure = settings.futures_exposure
    if symbol not in set(exposure.major_symbols):
        return False
    if _is_objectively_strong_futures_setup(features, settings, symbol=symbol):
        return False
    return (
        features.predictability_score >= exposure.pyramid_min_predictability_score
        and features.trend_strength >= exposure.pyramid_min_trend_strength
        and features.volume_confirmation >= exposure.pyramid_min_volume_confirmation
        and features.liquidity_score >= exposure.soft_liquidity_floor
        and features.net_expected_edge_bps >= max(exposure.min_entry_net_edge_bps, exposure.pyramid_min_net_edge_bps - 2.0)
        and _edge_to_cost_multiple(features) >= max(exposure.priority_edge_to_cost_multiple_min, 1.0)
    )


def _strong_futures_size_multiplier(
    features: FeatureVector,
    settings: Settings,
    *,
    symbol: str = "",
) -> float:
    thresholds = settings.mode_thresholds
    exposure = settings.futures_exposure

    def _normalized(value: float, floor: float, span: float) -> float:
        if span <= 0:
            return 1.0
        return max(0.0, min((value - floor) / span, 1.0))

    score_strength = _normalized(
        features.predictability_score,
        thresholds.futures_score_min + exposure.strong_score_buffer,
        18.0,
    )
    trend_strength = _normalized(
        features.trend_strength,
        exposure.strong_trend_strength_min,
        0.25,
    )
    volume_strength = _normalized(
        features.volume_confirmation,
        exposure.strong_volume_confirmation_min,
        0.3,
    )
    strong_liquidity_min = max(exposure.strong_liquidity_min, 0.82)
    if symbol == "PEPEUSDT":
        strong_liquidity_min = max(exposure.strong_liquidity_min, 0.50)
    liquidity_strength = _normalized(
        features.liquidity_score,
        strong_liquidity_min,
        0.35,
    )
    edge_strength = _normalized(
        _edge_to_cost_multiple(features),
        exposure.strong_edge_to_cost_multiple_min,
        1.5,
    )
    composite = (
        score_strength
        + trend_strength
        + volume_strength
        + liquidity_strength
        + edge_strength
    ) / 5.0
    strong_ceiling = max(float(exposure.strong_size_multiplier), 1.0)
    return round(1.0 + ((strong_ceiling - 1.0) * composite), 6)


def _futures_soft_entry_allowed(
    features: FeatureVector,
    settings: Settings,
    *,
    futures_score_min: float,
    priority_symbol: bool,
    supportive_macro: bool,
) -> bool:
    thresholds = settings.mode_thresholds
    exposure = settings.futures_exposure
    bearish_relief = 3.0 if features.trend_direction < 0 else 0.0
    soft_score_floor = max(futures_score_min - 2.0 - bearish_relief, thresholds.spot_score_min - bearish_relief)
    soft_trend_floor = max(
        thresholds.futures_trend_strength_min - (0.08 if features.trend_direction < 0 else 0.04),
        thresholds.spot_trend_strength_min,
    )
    edge_to_cost_multiple_min = exposure.macro_edge_to_cost_multiple_min
    if priority_symbol and supportive_macro:
        edge_to_cost_multiple_min = min(
            edge_to_cost_multiple_min,
            exposure.priority_edge_to_cost_multiple_min,
        )
    return (
        features.predictability_score >= soft_score_floor
        and features.trend_direction in {1, -1}
        and features.trend_strength >= soft_trend_floor
        and features.volume_confirmation >= (0.28 if features.trend_direction < 0 else 0.35)
        and features.liquidity_score >= (exposure.soft_liquidity_floor - (0.05 if features.trend_direction < 0 else 0.0))
        and features.volatility_penalty <= (exposure.soft_volatility_penalty_max + 0.05)
        and features.overheat_penalty <= (exposure.soft_overheat_penalty_max + 0.05)
        and features.net_expected_edge_bps >= (exposure.reduced_entry_net_edge_bps - (1.0 if features.trend_direction < 0 else 0.0))
        and _edge_to_cost_multiple(features) >= edge_to_cost_multiple_min
    )


def _btc_eth_confirmation_relaxation_allowed(features: FeatureVector, settings: Settings, *, symbol: str) -> bool:
    return (
        _is_btc_eth_symbol(symbol)
        and features.macro_trade_restraint == "none"
        and features.macro_event_risk_score < 0.55
        and passes_cost_gate(features, settings)
        and features.net_expected_edge_bps >= max(settings.futures_exposure.min_entry_net_edge_bps + 1.0, 8.0)
        and _edge_to_cost_multiple(features) >= max(settings.cost_gate.edge_to_cost_multiple_min, 1.2)
    )


def _reduced_size_futures_confirmation_required(*, symbol: str, futures_size_multiplier: float) -> bool:
    if futures_size_multiplier >= 1.0:
        return False
    return not _is_btc_eth_symbol(symbol)


def _btc_eth_spot_confirmation_relaxation_allowed(features: FeatureVector, settings: Settings, *, symbol: str) -> bool:
    return (
        _is_btc_eth_symbol(symbol)
        and features.macro_trade_restraint == "none"
        and features.macro_event_risk_score < 0.55
        and passes_cost_gate(features, settings)
        and features.net_expected_edge_bps >= 6.0
        and _edge_to_cost_multiple(features) >= max(settings.cost_gate.edge_to_cost_multiple_min, 1.15)
    )


def _btc_eth_futures_sentiment_relaxation_allowed(
    features: FeatureVector,
    settings: Settings,
    *,
    symbol: str,
    futures_score_min: float,
    futures_liquidity_min: float,
) -> bool:
    exposure = settings.futures_exposure
    thresholds = settings.mode_thresholds
    macro_gates = settings.macro_gates
    return (
        _is_btc_eth_symbol(symbol)
        and features.macro_trade_restraint == "none"
        and features.macro_event_risk_score < 0.45
        and features.macro_risk_penalty < macro_gates.futures_block_penalty
        and features.predictability_score >= max(futures_score_min, thresholds.futures_score_min)
        and features.trend_direction in {1, -1}
        and features.trend_strength >= thresholds.futures_trend_strength_min + 0.03
        and features.volume_confirmation >= max(exposure.strong_volume_confirmation_min - 0.08, 0.64)
        and features.liquidity_score >= max(futures_liquidity_min, exposure.soft_liquidity_floor)
        and features.volatility_penalty <= exposure.soft_volatility_penalty_max
        and features.overheat_penalty <= exposure.soft_overheat_penalty_max
        and features.net_expected_edge_bps >= max(
            exposure.min_entry_net_edge_bps + 2.0,
            exposure.reduced_entry_net_edge_bps,
            8.0,
        )
        and passes_cost_gate(features, settings)
        and _edge_to_cost_multiple(features) >= max(
            settings.cost_gate.edge_to_cost_multiple_min + 0.1,
            min(exposure.strong_edge_to_cost_multiple_min, 1.6),
            1.35,
        )
    )


def _short_entry_strict_reasons(
    features: FeatureVector,
    settings: Settings,
    *,
    futures_score_min: float,
    futures_liquidity_min: float,
    min_entry_net_edge_bps: float,
) -> list[str]:
    if features.trend_direction >= 0:
        # Long / flat trend: short-side gating is not applicable.
        return []
    if settings.futures_exposure.short_disabled:
        return ["SHORT_DISABLED"]
    reasons: list[str] = []
    extra_score = settings.futures_exposure.short_extra_score_floor
    extra_edge = settings.futures_exposure.short_extra_edge_bps
    extra_liq = settings.futures_exposure.short_extra_liquidity_floor
    extra_cost = settings.futures_exposure.short_extra_cost_multiple_floor
    short_score_floor = max(futures_score_min + extra_score, settings.mode_thresholds.futures_score_min + 1.0)
    short_liquidity_floor = max(futures_liquidity_min + extra_liq, settings.futures_exposure.soft_liquidity_floor + extra_liq)
    short_edge_floor = max(min_entry_net_edge_bps + extra_edge, settings.futures_exposure.reduced_entry_net_edge_bps + 1.0)
    short_cost_multiple_floor = max(settings.cost_gate.edge_to_cost_multiple_min + extra_cost, 0.95)
    short_trend_strength_floor = max(settings.mode_thresholds.futures_trend_strength_min + 0.02, 0.12)
    short_volume_floor = 0.48

    if features.predictability_score < short_score_floor:
        reasons.append("SCORE_TOO_LOW")
    if features.trend_strength < short_trend_strength_floor:
        reasons.append("SCORE_TOO_LOW")
    if features.volume_confirmation < short_volume_floor:
        reasons.append("SCORE_TOO_LOW")
    if features.liquidity_score < short_liquidity_floor:
        reasons.append("LIQUIDITY_TOO_WEAK")
    if features.net_expected_edge_bps < short_edge_floor:
        reasons.append("EDGE_TOO_THIN")
    if _edge_to_cost_multiple(features) < short_cost_multiple_floor:
        reasons.append("EDGE_BELOW_COST")
    return reasons


def _btc_eth_spot_relaxation_reasons(
    features: FeatureVector,
    settings: Settings,
    *,
    symbol: str,
    support_alignment_min: float,
    sentiment_support_min: float,
    resistance_penalty_max: float,
) -> tuple[str, ...]:
    thresholds = settings.mode_thresholds
    macro_gates = settings.macro_gates
    if not _is_btc_eth_symbol(symbol):
        return ()
    strong_context = (
        features.macro_trade_restraint != "halt_high_impact_window"
        and features.macro_risk_penalty < macro_gates.spot_block_penalty
        and features.trend_direction == 1
        and features.predictability_score >= thresholds.spot_score_min + 3.0
        and features.trend_strength >= thresholds.spot_trend_strength_min + 0.07
        and features.volume_confirmation >= 0.64
        and features.liquidity_score >= thresholds.spot_liquidity_min
        and features.resistance_penalty <= resistance_penalty_max
        and passes_cost_gate(features, settings)
        and features.net_expected_edge_bps >= max(4.0, features.estimated_round_trip_cost_bps * 0.25)
    )
    if not strong_context:
        return ()
    relaxed_reasons: list[str] = []
    if (
        features.support_alignment < support_alignment_min
        and features.support_alignment >= max(support_alignment_min - BTC_ETH_SECONDARY_SUPPORT_RELAX_TOLERANCE, 0.0)
        and features.breakout_norm >= 0.6
    ):
        relaxed_reasons.append("SUPPORT_NOT_CONFIRMED")
    if (
        features.sentiment_support_score < sentiment_support_min
        and features.sentiment_support_score >= max(sentiment_support_min - BTC_ETH_SECONDARY_SENTIMENT_RELAX_TOLERANCE, 0.0)
    ):
        relaxed_reasons.append("SENTIMENT_TOO_WEAK")
    return tuple(relaxed_reasons)


def _btc_eth_strong_size_boost_multiplier(
    features: FeatureVector,
    settings: Settings,
    *,
    symbol: str,
    base_size_multiplier: float,
    strong_setup: bool,
) -> float:
    exposure = settings.futures_exposure
    macro_gates = settings.macro_gates
    if not (_is_btc_eth_symbol(symbol) and strong_setup):
        return base_size_multiplier
    execution_quality_ok = (
        features.spread_bps_norm <= 0.3
        and features.probe_slippage_bps_norm <= 0.35
        and features.book_stability_norm >= 0.65
        and features.depth_10bps_norm >= 0.6
    )
    if not (
        execution_quality_ok
        and features.macro_trade_restraint == "none"
        and features.macro_event_risk_score < 0.45
        and features.macro_risk_penalty < macro_gates.futures_block_penalty
        and features.estimated_round_trip_cost_bps <= 14.0
        and features.net_expected_edge_bps >= max(
            exposure.min_entry_net_edge_bps + 4.0,
            exposure.reduced_entry_net_edge_bps + 2.0,
            10.0,
        )
        and _edge_to_cost_multiple(features) >= max(exposure.strong_edge_to_cost_multiple_min, 1.6)
    ):
        return base_size_multiplier
    multiplier_cap = BTC_SIZE_BOOST_MULTIPLIER_CAP if symbol == "BTCUSDT" else ETH_SIZE_BOOST_MULTIPLIER_CAP
    abs_cap = BTC_SIZE_BOOST_ABS_CAP if symbol == "BTCUSDT" else ETH_SIZE_BOOST_ABS_CAP
    boosted_multiplier = min(
        base_size_multiplier * min(max(exposure.major_size_boost_multiplier, 1.0), multiplier_cap),
        base_size_multiplier + abs_cap,
    )
    return round(max(base_size_multiplier, boosted_multiplier), 6)


def _adx_cross_signal_strength(features: FeatureVector, *, symbol: str = "") -> float:
    """Return a signal quality score [0, 1] based on ADX + EMA cross alignment.

    374-day validated per-coin profiles. Supports separate short params.
    """
    adx = features.adx_1h
    cross = features.ema_cross_signal
    td = features.trend_direction
    cp = get_profile(symbol)

    # Side filter
    if cp.side_filter == "long" and td < 0:
        return 0.0
    if cp.side_filter == "short" and td > 0:
        return 0.0

    # Use short-specific ADX floor if available and direction is short
    if td < 0 and cp.short_adx_floor > 0:
        adx_floor = cp.short_adx_floor
    else:
        adx_floor = cp.adx_floor

    # Mirror signal (横보장 역추세)
    mirror_signal = False
    if cp.mirror_adx_max > 0 and adx <= cp.mirror_adx_max:
        # Mirror works in LOW ADX (no trend) — RSI extreme reversal
        # This is checked separately, doesn't need ADX floor
        if features.pullback_signal != 0 and features.pullback_signal == td:
            mirror_signal = True
            return 0.8  # Strong mirror signal

    if adx < adx_floor:
        return 0.0
    adx_score = min((adx - adx_floor) / 20.0, 1.0)
    cross_aligned = (cross != 0 and cross == td)
    pullback_aligned = (features.pullback_signal != 0 and features.pullback_signal == td)
    signal_bonus = 0.3 if (cross_aligned or pullback_aligned) else 0.0
    if pullback_aligned:
        signal_bonus = 0.4
    return min(adx_score + signal_bonus, 1.0)


def _futures_entry_plan(
    features: FeatureVector,
    settings: Settings,
    symbol: str,
) -> tuple[bool, list[str], float, tuple[str, ...], tuple[str, ...], int]:
    thresholds = settings.mode_thresholds
    macro_gates = settings.macro_gates
    exposure = settings.futures_exposure
    reasons: list[str] = []
    relaxed_reasons: list[str] = []
    size_boost_reasons: list[str] = []
    size_multiplier = 1.0
    reduced_size = False
    effective_direction = int(features.trend_direction)
    priority_symbol = symbol in set(exposure.priority_symbols)
    alt_symbol = is_alt_symbol(symbol)

    # ── ADX + EMA Cross signal quality ──────────────
    # 374-day validated per-coin profiles (9 coins, $100 micro-capital optimized)
    if is_profiled(symbol):
        adx_signal = _adx_cross_signal_strength(features, symbol=symbol)
    else:
        adx_signal = 0.0
    adx_dampener = 0.5 if is_alt_symbol(symbol) else 1.0
    adx_score_relax = 5.0 * adx_signal * adx_dampener
    adx_trend_relax = 0.08 * adx_signal * adx_dampener

    directional_bearish_macro = (
        _is_btc_eth_symbol(symbol)
        and features.trend_direction < 0
        and features.macro_directional_bearish_score >= 0.6
        and features.macro_execution_risk_score < 0.75
        and features.macro_trade_restraint != "halt_high_impact_window"
    )
    supportive_macro = (
        features.macro_liquidity_support_score >= exposure.macro_support_min
        and features.macro_event_risk_score <= 0.45
        and features.macro_risk_penalty < macro_gates.futures_block_penalty
    ) or directional_bearish_macro
    macro_allowed, macro_reasons, macro_size_multiplier, _macro_leverage_cap = _macro_futures_risk_controls(features, symbol=symbol)
    if not macro_allowed:
        return False, macro_reasons, 0.0, (), (), effective_direction
    if macro_reasons:
        reasons.extend(macro_reasons)
        return False, reasons, 0.0, (), (), effective_direction
    size_multiplier = min(size_multiplier, macro_size_multiplier) if macro_size_multiplier > 0.0 else size_multiplier
    bearish_trend = effective_direction < 0
    futures_score_min = thresholds.futures_score_min - (
        exposure.macro_score_relaxation if supportive_macro else 0.0
    ) - adx_score_relax
    if alt_symbol and not supportive_macro:
        futures_score_min += exposure.alt_score_penalty_without_macro
    if bearish_trend:
        futures_score_min -= 3.0
    if priority_symbol and supportive_macro:
        futures_score_min -= exposure.priority_score_relaxation
    futures_liquidity_min = thresholds.futures_liquidity_min - (
        exposure.macro_liquidity_relaxation if supportive_macro else 0.0
    )
    if alt_symbol and not supportive_macro:
        futures_liquidity_min += exposure.alt_liquidity_penalty_without_macro
    if bearish_trend:
        futures_liquidity_min -= 0.06
    futures_overheat_penalty_max = thresholds.futures_overheat_penalty_max + (
        exposure.macro_overheat_relaxation if supportive_macro else 0.0
    )
    if bearish_trend:
        futures_overheat_penalty_max += 0.05
    futures_volatility_penalty_max = thresholds.futures_volatility_penalty_max + (
        exposure.macro_volatility_relaxation if supportive_macro else 0.0
    )
    if bearish_trend:
        futures_volatility_penalty_max += 0.06
    if priority_symbol and supportive_macro:
        futures_volatility_penalty_max += exposure.priority_volatility_relaxation
    soft_entry_allowed = _futures_soft_entry_allowed(
        features,
        settings,
        futures_score_min=futures_score_min,
        priority_symbol=priority_symbol,
        supportive_macro=supportive_macro,
    )
    if features.predictability_score < futures_score_min:
        reasons.append("SCORE_TOO_LOW")
    if abs(effective_direction) != 1:
        if int(features.ema_cross_signal) in {-1, 1}:
            effective_direction = int(features.ema_cross_signal)
            relaxed_reasons.append("DIRECTION_FROM_EMA_CROSS")
        elif int(features.pullback_signal) in {-1, 1}:
            effective_direction = int(features.pullback_signal)
            relaxed_reasons.append("DIRECTION_FROM_PULLBACK")
        else:
            reasons.append("DIRECTION_CONFLICT")
    if features.trend_strength < thresholds.futures_trend_strength_min - (0.06 if bearish_trend else 0.0) - adx_trend_relax:
        reasons.append("SCORE_TOO_LOW")
    if features.liquidity_score < futures_liquidity_min:
        if features.liquidity_score < exposure.soft_liquidity_floor:
            reasons.append("LIQUIDITY_TOO_WEAK")
        else:
            reduced_size = True
            size_multiplier = min(size_multiplier, exposure.reduced_size_multiplier)
    if features.volatility_penalty > futures_volatility_penalty_max:
        if features.volatility_penalty > exposure.soft_volatility_penalty_max:
            reasons.append("VOL_TOO_HIGH")
        else:
            reduced_size = True
            size_multiplier = min(size_multiplier, exposure.reduced_size_multiplier)
    if features.overheat_penalty > futures_overheat_penalty_max:
        if features.overheat_penalty > exposure.soft_overheat_penalty_max:
            reasons.append("FUTURES_OVERHEAT")
        else:
            reduced_size = True
            size_multiplier = min(size_multiplier, exposure.reduced_size_multiplier)
    if features.macro_risk_penalty >= macro_gates.futures_block_penalty:
        reasons.append("MACRO_RISK_HIGH")
    bearish_caution_override = (
        effective_direction < 0
        and features.predictability_score >= futures_score_min + 2.0
        and features.trend_strength >= thresholds.futures_trend_strength_min + 0.04
        and features.volume_confirmation >= 0.58
        and features.liquidity_score >= max(futures_liquidity_min, exposure.soft_liquidity_floor)
        and features.net_expected_edge_bps >= max(exposure.reduced_entry_net_edge_bps, exposure.min_entry_net_edge_bps)
        and _edge_to_cost_multiple(features) >= max(settings.cost_gate.edge_to_cost_multiple_min, 1.15)
        and features.volatility_penalty <= exposure.soft_volatility_penalty_max
        and features.overheat_penalty <= exposure.soft_overheat_penalty_max
    )
    priority_caution_override = (
        priority_symbol
        and exposure.priority_allow_caution
        and features.predictability_score >= max(futures_score_min, thresholds.futures_score_min)
        and features.net_expected_edge_bps > 0.0
        and features.volume_confirmation >= 0.7
        and features.liquidity_score >= exposure.soft_liquidity_floor
    )
    ultra_aggressive_caution_override = (
        settings.strategy_profile == "live-ultra-aggressive"
        and features.predictability_score >= (futures_score_min - 4.0)
        and features.trend_strength >= (thresholds.futures_trend_strength_min - 0.08)
        and features.volume_confirmation >= 0.3
        and features.liquidity_score >= max(exposure.soft_liquidity_floor - 0.04, 0.0)
        and features.net_expected_edge_bps >= (exposure.reduced_entry_net_edge_bps - 2.0)
    )
    if features.sentiment_regime == "caution" and ultra_aggressive_caution_override:
        reduced_size = True
        size_multiplier = min(size_multiplier, exposure.reduced_size_multiplier)
        relaxed_reasons.append("SENTIMENT_CAUTION_OVERRIDE")
    if features.sentiment_regime == "caution" and not (
        (supportive_macro and exposure.macro_allow_caution and features.volume_confirmation >= 0.62)
        or priority_caution_override
        or bearish_caution_override
        or ultra_aggressive_caution_override
    ):
        if _btc_eth_futures_sentiment_relaxation_allowed(
            features,
            settings,
            symbol=symbol,
            futures_score_min=futures_score_min,
            futures_liquidity_min=futures_liquidity_min,
        ):
            relaxed_reasons.append("SENTIMENT_CAUTION")
        elif soft_entry_allowed:
            reduced_size = True
            size_multiplier = min(size_multiplier, exposure.reduced_size_multiplier)
        else:
            reasons.append("SENTIMENT_CAUTION")
    if is_alt_symbol(symbol):
        if features.alt_market_regime == "defensive":
            reasons.append("ALT_REGIME_DEFENSIVE")
        if features.alt_liquidity_support_score < settings.altcoin_overlays.liquidity_floor:
            reasons.append("ALT_LIQUIDITY_WEAK")
        if features.alt_smart_money_score < settings.altcoin_overlays.smart_money_floor:
            reasons.append("ALT_FLOW_WEAK")
        if features.alt_rotation_penalty >= settings.altcoin_overlays.rotation_block_penalty:
            reasons.append("ALT_ROTATION_HEADWIND")
    # ── OI-VWAP-SMC integrated gate ──────────────
    from quant_binance.strategy.oi_vwap_smc_strategy import evaluate_entry as _oi_vwap_smc_eval
    oi_vwap_smc = _oi_vwap_smc_eval(features, symbol)
    if not oi_vwap_smc.entry_ok:
        return False, list(oi_vwap_smc.reasons), 0.0, (), ()
    if oi_vwap_smc.reasons:
        reduced_size = True
    size_multiplier = min(size_multiplier, oi_vwap_smc.size_multiplier)

    cost_gate_pass = passes_cost_gate(features, settings)
    macro_cost_multiple_min = exposure.macro_edge_to_cost_multiple_min
    if priority_symbol and supportive_macro:
        macro_cost_multiple_min = min(macro_cost_multiple_min, exposure.priority_edge_to_cost_multiple_min)
    if not cost_gate_pass and not (
        supportive_macro and _edge_to_cost_multiple(features) >= macro_cost_multiple_min
    ):
        if soft_entry_allowed:
            reduced_size = True
            size_multiplier = min(size_multiplier, exposure.reduced_size_multiplier)
        else:
            reasons.append("EDGE_BELOW_COST")
    min_entry_net_edge_bps = (
        exposure.macro_min_entry_net_edge_bps if supportive_macro else exposure.min_entry_net_edge_bps
    )
    if alt_symbol and not supportive_macro:
        min_entry_net_edge_bps += exposure.alt_min_entry_net_edge_bps_without_macro
    if priority_symbol and supportive_macro:
        min_entry_net_edge_bps = min(min_entry_net_edge_bps, exposure.priority_min_entry_net_edge_bps)
    if (
        alt_symbol
        and not supportive_macro
        and features.liquidity_score < futures_liquidity_min
        and features.net_expected_edge_bps <= (min_entry_net_edge_bps + 1.0)
    ):
        reasons.append("ALT_FUTURES_NOT_READY")
    if features.net_expected_edge_bps < min_entry_net_edge_bps:
        if soft_entry_allowed:
            reduced_size = True
            size_multiplier = min(size_multiplier, exposure.reduced_size_multiplier)
        else:
            reasons.append("EDGE_TOO_THIN")
    reasons.extend(
        _short_entry_strict_reasons(
            features,
            settings,
            futures_score_min=futures_score_min,
            futures_liquidity_min=futures_liquidity_min,
            min_entry_net_edge_bps=min_entry_net_edge_bps,
        )
    )
    if reasons:
        return False, reasons, 0.0, (), (), effective_direction
    if reduced_size and features.net_expected_edge_bps < exposure.reduced_entry_net_edge_bps:
        return False, ["EDGE_TOO_THIN"], 0.0, (), (), effective_direction
    if reduced_size and alt_symbol and exposure.alt_reduced_size_multiplier > 0.0:
        size_multiplier = min(size_multiplier, exposure.alt_reduced_size_multiplier)
    strong_setup = _is_objectively_strong_futures_setup(features, settings, symbol=symbol)
    # ADX+cross confirms strong trend: size boost for profiled coins
    adx_strong_trend = (
        is_profiled(symbol)
        and adx_signal > 0.5
        and features.ema_cross_signal != 0
        and features.ema_cross_signal == effective_direction
    )
    if adx_strong_trend and not reduced_size:
        strong_setup = True  # ADX-confirmed trend qualifies as strong
    if not reduced_size and strong_setup:
        size_multiplier = max(size_multiplier, _strong_futures_size_multiplier(features, settings, symbol=symbol))
    boosted_size_multiplier = _btc_eth_strong_size_boost_multiplier(
        features,
        settings,
        symbol=symbol,
        base_size_multiplier=size_multiplier,
        strong_setup=strong_setup and not reduced_size,
    )
    if boosted_size_multiplier > size_multiplier:
        size_multiplier = boosted_size_multiplier
        size_boost_reasons.append(BTC_ETH_STRONG_SIZE_BOOST_REASON)
    if symbol in set(exposure.demoted_symbols) and exposure.demoted_symbol_size_cap > 0.0:
        size_multiplier = min(size_multiplier, exposure.demoted_symbol_size_cap)
    # OI-VWAP-SMC size boost: all signals aligned = confidence boost
    if oi_vwap_smc.signal_quality == "strong" and not reduced_size:
        size_multiplier = max(size_multiplier, oi_vwap_smc.size_multiplier)
        size_boost_reasons.extend(oi_vwap_smc.boost_reasons)
    # Volatility scaling: higher vol → moderately larger size (trend breakouts)
    # 374d validated for ETH only (BTC excluded from ADX strategy).
    if symbol == "ETHUSDT" and features.volatility_penalty > 0.5 and not reduced_size:
        vol_boost = min((features.volatility_penalty - 0.5) * 2.0, 0.5)  # up to +0.5x
        size_multiplier = round(size_multiplier * (1.0 + vol_boost), 6)
    return True, reasons, size_multiplier, tuple(relaxed_reasons), tuple(size_boost_reasons), effective_direction


def _spot_passes(features: FeatureVector, settings: Settings, symbol: str) -> tuple[bool, list[str], tuple[str, ...]]:
    thresholds = settings.mode_thresholds
    macro_gates = settings.macro_gates
    support_cfg = settings.spot_support
    mode_behavior = settings.mode_behavior
    reasons: list[str] = []
    priority_symbol = symbol in set(support_cfg.priority_symbols)
    support_alignment_min = support_cfg.priority_support_alignment_min if priority_symbol else support_cfg.support_alignment_min
    resistance_penalty_max = support_cfg.priority_resistance_penalty_max if priority_symbol else support_cfg.resistance_penalty_max
    sentiment_support_min = support_cfg.priority_sentiment_support_min if priority_symbol else support_cfg.sentiment_support_min
    liquidity_relaxation = support_cfg.priority_liquidity_relaxation if priority_symbol else support_cfg.liquidity_relaxation

    relaxed_reasons = _btc_eth_spot_relaxation_reasons(
        features,
        settings,
        symbol=symbol,
        support_alignment_min=support_alignment_min,
        sentiment_support_min=sentiment_support_min,
        resistance_penalty_max=resistance_penalty_max,
    )

    macro_reasons = _macro_spot_risk_reasons(features, symbol=symbol)
    if macro_reasons:
        reasons.extend(macro_reasons)

    can_bottom_fish = (
        mode_behavior.spot_allow_bottoming_reversal
        and features.sentiment_regime == "bottoming"
        and features.support_alignment >= support_cfg.bottoming_support_override_min
        and features.macro_risk_penalty < macro_gates.spot_block_penalty
        and features.alt_rotation_penalty < settings.altcoin_overlays.rotation_block_penalty
    )

    if features.predictability_score < thresholds.spot_score_min:
        reasons.append("SCORE_TOO_LOW")
    if mode_behavior.spot_require_positive_trend:
        if features.trend_direction != 1:
            reasons.append("DIRECTION_CONFLICT")
    elif features.trend_direction == -1:
        if not can_bottom_fish:
            reasons.append("DIRECTION_CONFLICT")
    if features.trend_strength < thresholds.spot_trend_strength_min:
        reasons.append("SCORE_TOO_LOW")
    liquidity_override = (
        features.liquidity_score >= thresholds.spot_liquidity_min - liquidity_relaxation
        and (
            features.net_expected_edge_bps >= -support_cfg.priority_edge_relaxation_bps
            if priority_symbol
            else features.net_expected_edge_bps > 0.0
        )
        and (
            features.alt_liquidity_support_score >= 0.5
            or features.sentiment_support_score >= 0.55
            or symbol in {"BTCUSDT", "ETHUSDT"}
        )
    )
    if features.liquidity_score < thresholds.spot_liquidity_min and not liquidity_override:
        reasons.append("LIQUIDITY_TOO_WEAK")
    if features.volatility_penalty > thresholds.spot_volatility_penalty_max:
        reasons.append("VOL_TOO_HIGH")
    if features.support_alignment < support_alignment_min and "SUPPORT_NOT_CONFIRMED" not in relaxed_reasons:
        reasons.append("SUPPORT_NOT_CONFIRMED")
    resistance_override = (
        features.breakout_norm >= support_cfg.breakout_resistance_override_min
        and features.volume_confirmation >= 0.6
        and (
            features.net_expected_edge_bps >= -support_cfg.priority_edge_relaxation_bps
            if priority_symbol
            else features.net_expected_edge_bps > 0.0
        )
        and features.trend_direction >= 0
    )
    if features.resistance_penalty > resistance_penalty_max and not resistance_override:
        reasons.append("BUYING_INTO_RESISTANCE")
    if features.macro_risk_penalty >= macro_gates.spot_block_penalty:
        reasons.append("MACRO_RISK_HIGH")
    if features.sentiment_support_score < sentiment_support_min and "SENTIMENT_TOO_WEAK" not in relaxed_reasons:
        reasons.append("SENTIMENT_TOO_WEAK")
    if is_alt_symbol(symbol):
        if features.alt_market_regime == "defensive":
            reasons.append("ALT_REGIME_DEFENSIVE")
        if features.alt_breadth_score < settings.altcoin_overlays.breadth_floor:
            reasons.append("ALT_BREADTH_WEAK")
        if features.alt_liquidity_support_score < settings.altcoin_overlays.liquidity_floor:
            reasons.append("ALT_LIQUIDITY_WEAK")
        if features.alt_rotation_penalty >= settings.altcoin_overlays.rotation_block_penalty:
            reasons.append("ALT_ROTATION_HEADWIND")
    edge_gate_failed = not passes_cost_gate(features, settings)
    spot_accumulation_override = (
        features.support_alignment >= max(support_alignment_min, 0.12)
        and features.resistance_penalty <= min(resistance_penalty_max, 0.5 if priority_symbol else 0.35)
        and features.sentiment_support_score >= max(sentiment_support_min, 0.2 if priority_symbol else 0.35)
        and features.macro_risk_penalty < macro_gates.spot_block_penalty
        and features.net_expected_edge_bps > (-8.0 if priority_symbol else -5.0)
    )
    if edge_gate_failed and not spot_accumulation_override:
        reasons.append("EDGE_BELOW_COST")
    return not reasons, reasons, relaxed_reasons


def evaluate_snapshot(
    snapshot: MarketSnapshot,
    settings: Settings,
    equity_usd: float,
    remaining_portfolio_capacity_usd: float,
    expected_funding_drag_bps: float | None = None,
    cash_reserve_fraction: float = 0.0,
) -> DecisionIntent:
    thresholds = settings.mode_thresholds
    funding_drag_bps = (
        configured_funding_drag_bps(snapshot, settings)
        if expected_funding_drag_bps is None
        else float(expected_funding_drag_bps)
    )
    prediction = build_strategy_prediction(
        snapshot,
        settings,
        expected_funding_drag_bps=funding_drag_bps,
    )
    futures_prediction = prediction.futures
    spot_prediction = prediction.spot
    futures_features = apply_score_and_costs(
        snapshot.feature_values,
        settings=settings,
        mode="futures",
        expected_funding_drag_bps=funding_drag_bps,
    )
    spot_features = apply_score_and_costs(
        snapshot.feature_values,
        settings=settings,
        mode="spot",
    )
    candidate_mode = prediction.candidate_mode
    now = snapshot.decision_time

    observe_only, observe_reasons = _observe_only_reasons(spot_features, settings, snapshot.symbol)
    if observe_only:
        cash_reasons = tuple(sorted(set(observe_reasons)))
        intent = build_portfolio_intent(
            prediction=prediction,
            selected_mode="cash",
            side="flat",
            target_notional_usd=0.0,
            stop_distance_bps=0.0,
            rejection_reasons=cash_reasons,
        )
        return decision_from_portfolio_intent(intent=intent)

    futures_ok, futures_reasons, futures_size_multiplier, futures_relaxed_reasons, futures_size_boost_reasons, futures_direction = _futures_entry_plan(
        futures_features,
        settings,
        snapshot.symbol,
    )
    futures_relaxed_reasons = list(futures_relaxed_reasons)

    # ENSEMBLE TOP 3: evaluate FIRST, bypass base filters if signal fires
    # Backtest: 200 tr/375d, WR 36%, PF 6.75, Ruin 0.0%, WF 4/4
    import statistics as _stat
    _ens_sig_fired = False
    _ens_sig_name = ""
    _bars_1h = snapshot.state.klines.get("1h", []) if hasattr(snapshot, 'state') else []

    if is_profiled(snapshot.symbol) and len(_bars_1h) >= 168:
        def _mom1h(bars, p):
            if len(bars) < p + 1: return 0
            _p0 = bars[-1 - p].close_price
            return (bars[-1].close_price / _p0 - 1) if _p0 > 0 else 0

        def _rsi_1h(bars, p=14):
            if len(bars) < p + 1: return 50
            _gains, _losses = [], []
            for _j in range(p):
                _d = bars[-1 - _j].close_price - bars[-2 - _j].close_price
                _gains.append(max(_d, 0)); _losses.append(max(-_d, 0))
            _ag = sum(_gains) / p; _al = sum(_losses) / p
            return 100 - 100 / (1 + _ag / _al) if _al > 0 else 100

        def _bb_width(bars, p=48):
            if len(bars) < p: return 1.0
            _closes = [b.close_price for b in bars[-p:]]
            _m = sum(_closes) / p
            _s = _stat.stdev(_closes) if len(_closes) > 1 else 0
            return (2 * _s) / _m if _m > 0 else 1.0

        _m1d = _mom1h(_bars_1h, 24)
        _m3d = _mom1h(_bars_1h, 72)
        _rsi14 = _rsi_1h(_bars_1h, 14)
        _bw48 = _bb_width(_bars_1h, 48)

        _sig_a = _bw48 < 0.04 and _m1d > 0.01
        _sig_b = _m3d > 0.03
        _sig_c = _m3d < -0.05 and _rsi14 < 25

        if _sig_a or _sig_b or _sig_c:
            _ens_sig_fired = True
            _ens_sig_name = "A" if _sig_a else ("B" if _sig_b else "C")

    # If ensemble fired, FORCE futures_ok = True (bypass base filters)
    if _ens_sig_fired:
        if not futures_ok:
            futures_reasons = list(futures_reasons) + [f"ENSEMBLE_OVERRIDE_{_ens_sig_name}"]
        futures_ok = True
        from dataclasses import replace as _dc_replace
        prediction = _dc_replace(prediction,
            futures=_dc_replace(prediction.futures, side="long")
        )
    else:
        # No ensemble signal: block only when explicitly required
        ensemble_signal_required = bool(getattr(settings, "ensemble_signal_required", True))
        if futures_ok and ensemble_signal_required:
            futures_reasons = list(futures_reasons) + ["NO_ENSEMBLE_SIGNAL"]
        if ensemble_signal_required:
            futures_ok = False
        elif futures_ok:
            futures_relaxed_reasons.append("ENSEMBLE_OPTIONAL")

    if futures_ok:
        if futures_direction not in {-1, 1}:
            futures_ok = False
            futures_reasons = list(futures_reasons) + ["DIRECTION_CONFLICT"]
    if futures_ok:
        futures_side = "long" if futures_direction > 0 else "short"
        planned_leverage = select_futures_leverage(
            symbol=snapshot.symbol,
            predictability_score=futures_features.predictability_score,
            trend_strength=futures_features.trend_strength,
            volume_confirmation=futures_features.volume_confirmation,
            liquidity_score=futures_features.liquidity_score,
            volatility_penalty=futures_features.volatility_penalty,
            overheat_penalty=futures_features.overheat_penalty,
            net_expected_edge_bps=futures_features.net_expected_edge_bps,
            estimated_round_trip_cost_bps=futures_features.estimated_round_trip_cost_bps,
            settings=settings,
            adx_1h=futures_features.adx_1h,
            ema_cross_signal=futures_features.ema_cross_signal,
            trend_direction=futures_features.trend_direction,
        )
        if futures_features.macro_leverage_cap > 0:
            planned_leverage = min(planned_leverage, futures_features.macro_leverage_cap)
        notional, stop_distance_bps = position_notional_and_stop_bps(
            last_trade_price=snapshot.last_trade_price,
            atr_14_1h_bps=snapshot.feature_values.atr_14_1h_bps if snapshot.feature_values.atr_14_1h_bps > 0 else snapshot.feature_values.realized_vol_1h_norm * 100.0,
            equity_usd=equity_usd,
            remaining_portfolio_capacity_usd=remaining_portfolio_capacity_usd,
            settings=settings,
            size_multiplier=futures_size_multiplier,
            leverage_multiplier=planned_leverage,
            symbol=snapshot.symbol,
            side=futures_side,
        )
        required_margin_usd = notional / max(float(planned_leverage), 1.0)
        if equity_usd > 0 and (equity_usd - required_margin_usd) / equity_usd < cash_reserve_fraction:
            futures_ok = False
            futures_reasons.append("CASH_RESERVE_BLOCK")
        confirmation_intraday_strength_min = 0.5
        allow_btc_eth_confirmation_relaxation = _btc_eth_confirmation_relaxation_allowed(
            futures_features,
            settings,
            symbol=snapshot.symbol,
        )
        if allow_btc_eth_confirmation_relaxation:
            confirmation_intraday_strength_min = BTC_ETH_CONFIRMATION_INTRADAY_STRENGTH_MIN
        size_confirmation_required = _reduced_size_futures_confirmation_required(
            symbol=snapshot.symbol,
            futures_size_multiplier=futures_size_multiplier,
        )
        confirmation_required = size_confirmation_required or ((
            futures_features.intraday_trend_strength < confirmation_intraday_strength_min
            or (
                futures_features.intraday_trend_direction != 0
                and futures_features.intraday_trend_direction != futures_features.trend_direction
            )
        ) and not allow_btc_eth_confirmation_relaxation)
        intent = build_portfolio_intent(
            prediction=prediction,
            selected_mode="futures",
            side=futures_side,
            target_notional_usd=notional,
            stop_distance_bps=stop_distance_bps,
            target_leverage=planned_leverage,
            strategy_size_multiplier=futures_size_multiplier,
            entry_relaxation_reasons=futures_relaxed_reasons,
            size_boost_reasons=futures_size_boost_reasons,
            divergence_code="ENTRY_CONFIRMATION_REQUIRED" if confirmation_required else "",
        )
        return decision_from_portfolio_intent(intent=intent)

    spot_ok, spot_reasons, spot_relaxed_reasons = _spot_passes(spot_features, settings, snapshot.symbol)
    if spot_ok:
        notional, stop_distance_bps = position_notional_and_stop_bps(
            last_trade_price=snapshot.last_trade_price,
            atr_14_1h_bps=snapshot.feature_values.atr_14_1h_bps if snapshot.feature_values.atr_14_1h_bps > 0 else snapshot.feature_values.realized_vol_1h_norm * 100.0,
            equity_usd=equity_usd,
            remaining_portfolio_capacity_usd=remaining_portfolio_capacity_usd,
            settings=settings,
            symbol=snapshot.symbol,
            side="long",
        )
        if equity_usd > 0 and (equity_usd - notional) / equity_usd < cash_reserve_fraction:
            spot_ok = False
            spot_reasons.append("CASH_RESERVE_BLOCK")
    if spot_ok:
        confirmation_support_min = 0.25
        confirmation_resistance_max = 0.3
        confirmation_intraday_strength_min = 0.45
        if _btc_eth_spot_confirmation_relaxation_allowed(spot_features, settings, symbol=snapshot.symbol):
            confirmation_support_min = 0.18
            confirmation_resistance_max = 0.38
            confirmation_intraday_strength_min = BTC_ETH_CONFIRMATION_SPOT_INTRADAY_STRENGTH_MIN
        confirmation_required = (
            spot_features.support_alignment < confirmation_support_min
            or spot_features.resistance_penalty > confirmation_resistance_max
            or spot_features.liquidity_score < thresholds.spot_liquidity_min
            or spot_features.intraday_trend_strength < confirmation_intraday_strength_min
        )
        intent = build_portfolio_intent(
            prediction=prediction,
            selected_mode="spot",
            side="long",
            target_notional_usd=notional,
            stop_distance_bps=stop_distance_bps,
            entry_relaxation_reasons=spot_relaxed_reasons,
            rejection_reasons=tuple(sorted(set(futures_reasons))),
            divergence_code="ENTRY_CONFIRMATION_REQUIRED" if confirmation_required else "",
        )
        return decision_from_portfolio_intent(intent=intent)

    # ── Alpha sub-strategies: find trades the regime engine skips ──
    from quant_binance.strategy.alpha_strategies import best_alpha_signal
    alpha = best_alpha_signal(futures_features)
    if alpha is not None and alpha.confidence >= 0.55:
        alpha_leverage = min(alpha.leverage_cap, int(settings.risk.max_futures_leverage))
        alpha_notional = (
            equity_usd * settings.risk.per_trade_equity_risk
            / max(alpha.stop_bps / 10000.0, 1e-6)
            * alpha.size_fraction
        )
        alpha_notional = min(alpha_notional, remaining_portfolio_capacity_usd)
        intent = build_portfolio_intent(
            prediction=prediction,
            selected_mode="futures",
            side=alpha.side,
            target_notional_usd=round(alpha_notional, 6),
            stop_distance_bps=alpha.stop_bps,
            target_leverage=float(alpha_leverage),
            strategy_size_multiplier=alpha.size_fraction,
            size_boost_reasons=(alpha.reason,),
            entry_relaxation_reasons=("ALPHA_SUB_STRATEGY",),
        )
        return decision_from_portfolio_intent(intent=intent)

    cash_reasons = tuple(sorted(set(futures_reasons + spot_reasons))) or ("SCORE_TOO_LOW",)
    futures_reason_view = tuple(sorted(set(futures_reasons))) or ("NONE",)
    spot_reason_view = tuple(sorted(set(spot_reasons))) or ("NONE",)
    divergence_summary = (
        f"FUTURES[{','.join(futures_reason_view[:6])}]|"
        f"SPOT[{','.join(spot_reason_view[:6])}]"
    )
    intent = build_portfolio_intent(
        prediction=prediction,
        selected_mode="cash",
        side="flat",
        target_notional_usd=0.0,
        stop_distance_bps=0.0,
        rejection_reasons=cash_reasons,
        divergence_code=divergence_summary,
    )
    return decision_from_portfolio_intent(intent=intent)
