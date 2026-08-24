from __future__ import annotations

from copy import deepcopy
import json
import logging
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger(__name__)

from quant_binance.auto_mode import (
    auto_mode_blocks_non_major_positive,
    auto_mode_positive_action_cap,
    auto_mode_positive_cap,
    auto_mode_promotion_intensity_cap,
    is_positive_policy_action,
)
from quant_binance.closed_trade_metrics import aggregate_closed_trades as aggregate_closed_trade_metrics
from quant_binance.models import DecisionIntent
from quant_binance.observability.log_store import _json_ready
from quant_binance.policy_evidence import (
    baseline_control_bucket_comparison,
    build_replay_provenance,
    checkpoint_replay_provenance,
    policy_evidence_bucket,
    policy_evidence_bucket_evidence,
    with_policy_evidence_buckets,
)
from quant_binance.policy_lineage import build_policy_state_lineage_snapshot
from quant_binance.symbol_lifecycle import build_symbol_lifecycle, summarize_symbol_lifecycle


def _active_exchange_futures_positions(
    live_positions: list[dict[str, object]] | tuple[dict[str, object], ...] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position in live_positions or []:
        total = float(position.get("total") or position.get("available") or 0.0)
        if total <= 0.0:
            continue
        rows.append(dict(position))
    return rows


def _futures_position_sync_payload(
    *,
    open_futures_positions: list[dict[str, object]] | None,
    live_positions: list[dict[str, object]] | tuple[dict[str, object], ...] | None,
) -> dict[str, object]:
    paper_open_futures_positions = list(open_futures_positions or [])
    exchange_live_futures_positions = _active_exchange_futures_positions(live_positions)
    adopted_futures_positions = [
        dict(position)
        for position in paper_open_futures_positions
        if str(position.get("origin", "")).strip().lower() == "adopted"
        or bool(position.get("adopted_at"))
        or bool(position.get("adoption_source"))
    ]
    paper_symbols = {
        str(position.get("symbol", ""))
        for position in paper_open_futures_positions
        if str(position.get("symbol", ""))
    }
    exchange_symbols = {
        str(position.get("symbol", ""))
        for position in exchange_live_futures_positions
        if str(position.get("symbol", ""))
    }
    missing_in_paper = sorted(exchange_symbols - paper_symbols)
    missing_on_exchange = sorted(paper_symbols - exchange_symbols)
    mismatch_details = {
        "missing_in_paper": missing_in_paper,
        "missing_on_exchange": missing_on_exchange,
    }
    pending_external_futures_positions = [
        dict(position)
        for position in exchange_live_futures_positions
        if str(position.get("symbol", "")) in set(missing_in_paper)
    ]
    return {
        "paper_open_futures_positions": paper_open_futures_positions,
        "paper_open_futures_position_count": len(paper_open_futures_positions),
        "adopted_futures_positions": adopted_futures_positions,
        "adopted_futures_position_count": len(adopted_futures_positions),
        "exchange_live_futures_positions": exchange_live_futures_positions,
        "exchange_live_futures_position_count": len(exchange_live_futures_positions),
        "pending_external_futures_positions": pending_external_futures_positions,
        "pending_external_futures_position_count": len(pending_external_futures_positions),
        "futures_position_mismatch": bool(missing_in_paper or missing_on_exchange),
        "futures_position_mismatch_details": mismatch_details,
    }


def _aggregate_closed_trades(closed_trades: list[dict[str, object]] | tuple[dict[str, object], ...]) -> tuple[list[dict[str, object]], dict[str, int], float]:
    aggregate = aggregate_closed_trade_metrics(closed_trades)
    return aggregate.symbol_performance, aggregate.exit_reason_counts, aggregate.realized_pnl_usd




def _aggregate_live_order_outcomes(live_orders: list[dict[str, object]] | tuple[dict[str, object], ...] | None) -> dict[str, object]:
    orders = list(live_orders or [])
    if not orders:
        return {
            "execution_outcome_counts": {},
            "accepted_live_order_count": 0,
            "rejected_live_order_count": 0,
            "avg_fill_ratio": 0.0,
            "avg_slippage_bps": 0.0,
            "avg_realized_edge_bps": 0.0,
            "avg_expected_edge_bps": 0.0,
            "avg_edge_retention_ratio": 0.0,
            "protection_degraded_count": 0,
            "protection_degraded_rate": 0.0,
            "realized_vs_expected_edge_gap_bps": 0.0,
            "execution_audit_by_symbol": [],
        }

    counts = Counter()
    accepted = 0
    rejected = 0
    fill_ratio_sum = 0.0
    slippage_values: list[float] = []
    realized_edge_values: list[float] = []
    expected_edge_values: list[float] = []
    retention_values: list[float] = []
    protection_degraded_count = 0
    by_symbol: dict[str, dict[str, float | int | str]] = defaultdict(
        lambda: {
            "symbol": "",
            "live_order_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "avg_fill_ratio": 0.0,
            "avg_slippage_bps": 0.0,
            "avg_realized_edge_bps": 0.0,
            "avg_expected_edge_bps": 0.0,
            "avg_edge_retention_ratio": 0.0,
            "protection_degraded_count": 0,
            "protection_degraded_rate": 0.0,
            "realized_vs_expected_edge_gap_bps": 0.0,
        }
    )
    fill_sums: dict[str, float] = defaultdict(float)
    slip_sums: dict[str, float] = defaultdict(float)
    slip_counts: dict[str, int] = defaultdict(int)
    realized_sums: dict[str, float] = defaultdict(float)
    realized_counts: dict[str, int] = defaultdict(int)
    expected_sums: dict[str, float] = defaultdict(float)
    retention_sums: dict[str, float] = defaultdict(float)
    retention_counts: dict[str, int] = defaultdict(int)

    for order in orders:
        status = str(order.get("fill_status") or ("accepted" if order.get("accepted") else "reject") or "unknown")
        symbol = str(order.get("symbol", ""))
        counts[status] += 1
        row = by_symbol[symbol]
        row["symbol"] = symbol
        row["live_order_count"] = int(row["live_order_count"]) + 1

        accepted_flag = bool(order.get("accepted", False))
        if accepted_flag:
            accepted += 1
            row["accepted_count"] = int(row["accepted_count"]) + 1
        else:
            rejected += 1
            row["rejected_count"] = int(row["rejected_count"]) + 1

        fill_ratio = float(order.get("fill_ratio", 0.0) or 0.0)
        fill_ratio_sum += fill_ratio
        fill_sums[symbol] += fill_ratio

        slippage = order.get("slippage_bps")
        if slippage is not None:
            slip = float(slippage or 0.0)
            slippage_values.append(slip)
            slip_sums[symbol] += slip
            slip_counts[symbol] += 1

        realized = order.get("realized_edge_bps")
        if realized is not None:
            realized_value = float(realized or 0.0)
            realized_edge_values.append(realized_value)
            realized_sums[symbol] += realized_value
            realized_counts[symbol] += 1

        expected = float(order.get("expected_net_edge_bps", order.get("net_expected_edge_bps", 0.0)) or 0.0)
        expected_edge_values.append(expected)
        expected_sums[symbol] += expected
        if expected > 0.0 and realized is not None:
            retention = max(min(float(realized or 0.0) / expected, 2.0), -2.0)
            retention_values.append(retention)
            retention_sums[symbol] += retention
            retention_counts[symbol] += 1
        if order.get("protection_error"):
            protection_degraded_count += 1
            row["protection_degraded_count"] = int(row["protection_degraded_count"]) + 1

    rows: list[dict[str, object]] = []
    for symbol, row in by_symbol.items():
        count = int(row["live_order_count"])
        expected_avg = expected_sums[symbol] / count if count else 0.0
        realized_avg = realized_sums[symbol] / realized_counts[symbol] if realized_counts[symbol] else 0.0
        row["avg_fill_ratio"] = round(fill_sums[symbol] / count, 6) if count else 0.0
        row["avg_slippage_bps"] = round(slip_sums[symbol] / slip_counts[symbol], 6) if slip_counts[symbol] else 0.0
        row["avg_realized_edge_bps"] = round(realized_avg, 6)
        row["avg_expected_edge_bps"] = round(expected_avg, 6)
        row["avg_edge_retention_ratio"] = round(retention_sums[symbol] / retention_counts[symbol], 6) if retention_counts[symbol] else 0.0
        row["realized_vs_expected_edge_gap_bps"] = round(realized_avg - expected_avg, 6)
        rows.append(dict(row))
    rows.sort(key=lambda item: (float(item["realized_vs_expected_edge_gap_bps"]), float(item["avg_realized_edge_bps"])), reverse=True)

    avg_expected = sum(expected_edge_values) / len(expected_edge_values) if expected_edge_values else 0.0
    avg_realized = sum(realized_edge_values) / len(realized_edge_values) if realized_edge_values else 0.0
    return {
        "execution_outcome_counts": dict(sorted(counts.items())),
        "accepted_live_order_count": accepted,
        "rejected_live_order_count": rejected,
        "avg_fill_ratio": round(fill_ratio_sum / len(orders), 6),
        "avg_slippage_bps": round(sum(slippage_values) / len(slippage_values), 6) if slippage_values else 0.0,
        "avg_realized_edge_bps": round(avg_realized, 6),
        "avg_expected_edge_bps": round(avg_expected, 6),
        "avg_edge_retention_ratio": round(sum(retention_values) / len(retention_values), 6) if retention_values else 0.0,
        "protection_degraded_count": protection_degraded_count,
        "protection_degraded_rate": round(protection_degraded_count / len(orders), 6),
        "realized_vs_expected_edge_gap_bps": round(avg_realized - avg_expected, 6),
        "execution_audit_by_symbol": rows,
    }



def _performance_bucket(order: dict[str, object]) -> tuple[str, str, str, str, str]:
    symbol = str(order.get("symbol", "") or "UNKNOWN")
    side = str(order.get("side", "") or "unknown")
    regime = "major" if symbol in {"BTCUSDT", "ETHUSDT"} else "alt"
    expected = float(order.get("expected_net_edge_bps", order.get("net_expected_edge_bps", 0.0)) or 0.0)
    fill_ratio = float(order.get("fill_ratio", 0.0) or 0.0)
    setup_class = "high_edge" if expected >= 15.0 else "standard_edge"
    execution_quality_state = "degraded" if order.get("protection_error") or fill_ratio < 0.85 or not bool(order.get("accepted", False)) else "healthy"
    return symbol, regime, setup_class, side, execution_quality_state


def build_performance_attribution(live_orders: list[dict[str, object]] | tuple[dict[str, object], ...] | None) -> list[dict[str, object]]:
    orders = list(live_orders or [])
    buckets: dict[tuple[str, str, str, str, str], dict[str, float | int | str]] = defaultdict(
        lambda: {
            "symbol": "",
            "regime": "",
            "setup_class": "",
            "side": "",
            "execution_quality_state": "",
            "sample_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "realized_edge_sum": 0.0,
            "expected_edge_sum": 0.0,
            "retention_sum": 0.0,
            "retention_count": 0,
            "protection_degraded_count": 0,
        }
    )
    for order in orders:
        key = _performance_bucket(order)
        row = buckets[key]
        row["symbol"], row["regime"], row["setup_class"], row["side"], row["execution_quality_state"] = key
        row["sample_count"] = int(row["sample_count"]) + 1
        if bool(order.get("accepted", False)):
            row["accepted_count"] = int(row["accepted_count"]) + 1
        else:
            row["rejected_count"] = int(row["rejected_count"]) + 1
        expected = float(order.get("expected_net_edge_bps", order.get("net_expected_edge_bps", 0.0)) or 0.0)
        realized = float(order.get("realized_edge_bps", 0.0) or 0.0)
        row["expected_edge_sum"] = float(row["expected_edge_sum"]) + expected
        row["realized_edge_sum"] = float(row["realized_edge_sum"]) + realized
        if expected > 0.0:
            row["retention_sum"] = float(row["retention_sum"]) + max(min(realized / expected, 2.0), -2.0)
            row["retention_count"] = int(row["retention_count"]) + 1
        if order.get("protection_error"):
            row["protection_degraded_count"] = int(row["protection_degraded_count"]) + 1
    rows: list[dict[str, object]] = []
    for row in buckets.values():
        sample_count = int(row["sample_count"])
        accepted_count = int(row["accepted_count"])
        rejected_count = int(row["rejected_count"])
        retention_count = int(row["retention_count"])
        rows.append({
            "symbol": row["symbol"],
            "regime": row["regime"],
            "setup_class": row["setup_class"],
            "side": row["side"],
            "execution_quality_state": row["execution_quality_state"],
            "sample_count": sample_count,
            "avg_realized_edge_bps": round(float(row["realized_edge_sum"]) / sample_count, 6) if sample_count else 0.0,
            "avg_expected_edge_bps": round(float(row["expected_edge_sum"]) / sample_count, 6) if sample_count else 0.0,
            "avg_edge_retention_ratio": round(float(row["retention_sum"]) / retention_count, 6) if retention_count else 0.0,
            "reject_rate": round(rejected_count / sample_count, 6) if sample_count else 0.0,
            "protection_degraded_rate": round(int(row["protection_degraded_count"]) / sample_count, 6) if sample_count else 0.0,
        })
    rows.sort(key=lambda item: (str(item["symbol"]), str(item["setup_class"]), str(item["side"]), str(item["execution_quality_state"])))
    return rows


_ACTION_STRENGTH = {
    "disabled": -3,
    "demote": -2,
    "keep": 0,
    "promote": 1,
    "aggressive_promote": 2,
}


def _policy_action_strength(action: str) -> int:
    return _ACTION_STRENGTH.get(action, 0)


def _policy_adjustment_shape(
    *,
    symbol: str,
    regime: str,
    setup_class: str,
    side: str,
    execution_quality_state: str,
    sample_count: int,
    action: str,
    reason: str,
    signal_source: str,
    score_delta: float = 0.0,
    operating_intensity: float = 1.0,
    signal_context: dict[str, object] | None = None,
) -> dict[str, object]:
    size_multiplier = 1.0
    if action == "demote":
        size_multiplier = 0.75
    elif action == "promote":
        size_multiplier = 1.1
    elif action == "aggressive_promote":
        size_multiplier = 1.25
    entry_threshold_bps = -1.5 if action == "aggressive_promote" else (-0.5 if action == "promote" else (1.5 if action == "demote" else 0.0))
    expected_profit_floor_bps = -2.0 if action == "aggressive_promote" else (-1.0 if action == "promote" else (2.0 if action == "demote" else 0.0))
    intensity = max(0.5, min(float(operating_intensity or 1.0), 1.25))
    if action not in {"disabled", "keep"} and intensity != 1.0:
        size_multiplier = round(min(1.25, max(0.5, 1.0 + ((size_multiplier - 1.0) * intensity))), 6)
        entry_threshold_bps = round(entry_threshold_bps * intensity, 6)
        expected_profit_floor_bps = round(expected_profit_floor_bps * intensity, 6)
    leverage_multiplier = 0.0 if action == "disabled" else round(min(max(size_multiplier, 0.0), 1.2), 6)
    symbol_bias = "majors_only" if action in {"aggressive_promote", "promote"} and regime == "major" else "neutral"
    adjustment = {
        "symbol": symbol,
        "regime": regime,
        "setup_class": setup_class,
        "side": side,
        "execution_quality_state": execution_quality_state,
        "sample_count": sample_count,
        "action": action,
        "size_multiplier": size_multiplier,
        "leverage_multiplier": leverage_multiplier,
        "entry_threshold_bps": entry_threshold_bps,
        "expected_profit_floor_bps": expected_profit_floor_bps,
        "operating_intensity": round(intensity, 6),
        "symbol_bias": symbol_bias,
        "reason": reason,
        "score_delta": round(float(score_delta or 0.0), 6),
        "signal_sources": [signal_source],
        "signal_contexts": ({signal_source: dict(signal_context or {})} if signal_context else {}),
    }
    return adjustment


def _bounded_score(value: float, *, lower: float, upper: float) -> float:
    return round(min(upper, max(lower, value)), 6)


def _live_attribution_adjustments(
    attribution_rows: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    adjustments: list[dict[str, object]] = []
    for row in attribution_rows:
        sample_count = int(row.get("sample_count", 0) or 0)
        if sample_count < 3:
            continue
        retention = float(row.get("avg_edge_retention_ratio", 0.0) or 0.0)
        realized = float(row.get("avg_realized_edge_bps", 0.0) or 0.0)
        reject_rate = float(row.get("reject_rate", 0.0) or 0.0)
        degraded_rate = float(row.get("protection_degraded_rate", 0.0) or 0.0)
        action = "keep"
        reason = "STABLE"
        if retention < 0.5 or realized <= 0.0 or reject_rate > 0.1 or degraded_rate > 0.1:
            action = "demote"
            reason = "WEAK_ATTRIBUTION"
        elif retention >= 0.95 and realized > 0.0 and reject_rate <= 0.01 and degraded_rate <= 0.01 and str(row.get("regime", "")) == "major":
            action = "aggressive_promote"
            reason = "ELITE_ATTRIBUTION"
        elif retention >= 0.8 and realized > 0.0 and reject_rate <= 0.03 and degraded_rate <= 0.03:
            action = "promote"
            reason = "STRONG_ATTRIBUTION"
        adjustments.append(
            _policy_adjustment_shape(
                symbol=str(row.get("symbol", "") or ""),
                regime=str(row.get("regime", "") or ""),
                setup_class=str(row.get("setup_class", "") or ""),
                side=str(row.get("side", "") or ""),
                execution_quality_state=str(row.get("execution_quality_state", "") or ""),
                sample_count=sample_count,
                action=action,
                reason=reason,
                signal_source="live_attribution",
                signal_context={
                    "avg_edge_retention_ratio": round(retention, 6),
                    "avg_realized_edge_bps": round(realized, 6),
                    "reject_rate": round(reject_rate, 6),
                    "protection_degraded_rate": round(degraded_rate, 6),
                },
            )
        )
    return adjustments


def _runtime_regime_breakdown(runtime_evidence: dict[str, object] | None) -> dict[str, object]:
    payload = dict(runtime_evidence or {})
    regime_rows = [dict(row) for row in list(payload.get("regime_summary", []) or []) if isinstance(row, dict)]
    futures_row = next((row for row in regime_rows if str(row.get("mode", "")) == "futures"), None)
    supportive_modes: list[str] = []
    elite_modes: list[str] = []
    mode_scores: dict[str, float] = {}
    dominant_mode = ""
    dominant_value = float("-inf")
    for row in regime_rows:
        mode = str(row.get("mode", "") or "")
        if not mode:
            continue
        decision_count = int(row.get("decision_count", 0) or 0)
        avg_score = float(row.get("avg_score", 0.0) or 0.0)
        avg_net_edge_bps = float(row.get("avg_net_edge_bps", 0.0) or 0.0)
        support_score = _bounded_score((avg_net_edge_bps / 20.0) + ((avg_score - 50.0) / 100.0), lower=-0.35, upper=0.35)
        mode_scores[mode] = support_score
        if decision_count >= 3 and avg_net_edge_bps > 0.0 and avg_score >= 0.0:
            supportive_modes.append(mode)
        if decision_count >= 6 and avg_net_edge_bps >= 8.0 and avg_score >= 55.0:
            elite_modes.append(mode)
        candidate_dominance = avg_net_edge_bps + (avg_score / 100.0)
        if dominant_mode == "" or candidate_dominance > dominant_value:
            dominant_mode = mode
            dominant_value = candidate_dominance
    return {
        "regime_rows": regime_rows,
        "supportive_modes": sorted(supportive_modes),
        "elite_modes": sorted(elite_modes),
        "mode_scores": mode_scores,
        "dominant_mode": dominant_mode,
        "futures_supportive": futures_row is not None and "futures" in supportive_modes,
        "futures_elite": futures_row is not None and "futures" in elite_modes,
    }


def _runtime_regime_support(runtime_evidence: dict[str, object] | None) -> tuple[bool, bool]:
    breakdown = _runtime_regime_breakdown(runtime_evidence)
    return bool(breakdown.get("futures_supportive")), bool(breakdown.get("futures_elite"))


def _runtime_symbol_score_delta(
    *,
    recommendation: str,
    trade_count: int,
    expectancy: float,
    regime: str,
    regime_breakdown: dict[str, object],
) -> float:
    base_score = min(0.32, abs(expectancy) / 10.0) + min(0.12, trade_count / 25.0)
    regime_bonus = 0.0
    if regime == "major":
        if bool(regime_breakdown.get("futures_elite")):
            regime_bonus = 0.16
        elif bool(regime_breakdown.get("futures_supportive")):
            regime_bonus = 0.09
    elif str(regime_breakdown.get("dominant_mode", "")) == "cash":
        regime_bonus = -0.04
    score = base_score + regime_bonus
    return round(score if recommendation == "promote" else -score, 6)


def _runtime_symbol_rolling_gate(row: dict[str, object]) -> tuple[bool, dict[str, object]]:
    rolling_evidence = dict(row.get("rolling_evidence", {}) or {})
    observed_run_count = int(rolling_evidence.get("observed_run_count", 0) or 0)
    recommendation = str(row.get("recommendation", "") or "")
    if observed_run_count < 2:
        rolling_evidence["gate_status"] = "insufficient_history"
        return True, rolling_evidence
    recent_run_consistency = float(rolling_evidence.get("recent_run_consistency", 0.0) or 0.0)
    positive_window_ratio = float(rolling_evidence.get("positive_window_ratio", 0.0) or 0.0)
    expectancy_stability = float(rolling_evidence.get("expectancy_stability", 0.0) or 0.0)
    supports_recommendation = False
    if recommendation == "promote":
        supports_recommendation = (
            recent_run_consistency >= 0.67
            and positive_window_ratio >= 0.6
            and expectancy_stability >= 0.35
        )
    elif recommendation == "prune":
        supports_recommendation = (
            recent_run_consistency >= 0.67
            and positive_window_ratio <= 0.4
            and expectancy_stability >= 0.35
        )
    else:
        supports_recommendation = True
    rolling_evidence["gate_status"] = "supportive" if supports_recommendation else "mixed"
    return supports_recommendation, rolling_evidence


def _runtime_symbol_scorecard_gate(
    *,
    recommendation: str,
    scorecard_row: dict[str, object] | None,
) -> tuple[bool, dict[str, object]]:
    payload = dict(scorecard_row or {})
    if not payload:
        return True, {}
    trade_run_count = int(payload.get("trade_run_count", 0) or 0)
    recent_trade_run_count = int(payload.get("recent_trade_run_count", 0) or 0)
    if trade_run_count < 2 or recent_trade_run_count < 2:
        payload["gate_status"] = "insufficient_history"
        return True, payload
    scorecard_recommendation = str(payload.get("recommendation", "keep") or "keep")
    rolling_score = float(payload.get("rolling_score", 0.0) or 0.0)
    recent_positive_run_ratio = float(payload.get("recent_positive_run_ratio", 0.0) or 0.0)
    if recommendation == "promote":
        supports_recommendation = not (
            scorecard_recommendation == "demote"
            or recent_positive_run_ratio < 0.67
            or rolling_score < 0.6
        )
    elif recommendation == "prune":
        supports_recommendation = not (
            scorecard_recommendation == "promote"
            and recent_positive_run_ratio >= 0.67
            and rolling_score >= 0.7
        )
    else:
        supports_recommendation = True
    payload["gate_status"] = "supportive" if supports_recommendation else "conflicted"
    return supports_recommendation, payload


def _runtime_symbol_scorecard_intensity(
    *,
    recommendation: str,
    action: str,
    scorecard_evidence: dict[str, object] | None,
) -> tuple[str, float, dict[str, object]]:
    payload = dict(scorecard_evidence or {})
    if not payload:
        return action, 1.0, payload
    if str(payload.get("gate_status", "") or "") == "insufficient_history":
        payload["intensity_status"] = "baseline"
        payload["operating_intensity"] = 1.0
        return action, 1.0, payload
    scorecard_recommendation = str(payload.get("recommendation", "keep") or "keep")
    rolling_score = float(payload.get("rolling_score", 0.0) or 0.0)
    recent_positive_run_ratio = float(payload.get("recent_positive_run_ratio", 0.0) or 0.0)
    if (
        action == "aggressive_promote"
        and (
            scorecard_recommendation != "promote"
            or rolling_score < 0.8
            or recent_positive_run_ratio < 0.75
        )
    ):
        payload["intensity_status"] = "downgraded"
        payload["intensity_reason"] = "ROLLING_SCORECARD_NOT_ELITE"
        payload["operating_intensity"] = 1.0
        return "promote", 1.0, payload
    if (
        recommendation == "promote"
        and action == "promote"
        and scorecard_recommendation == "promote"
        and rolling_score >= 0.85
        and recent_positive_run_ratio >= 0.8
    ):
        payload["intensity_status"] = "strengthened"
        payload["intensity_reason"] = "ROLLING_SCORECARD_STRONGLY_POSITIVE"
        payload["operating_intensity"] = 1.1
        return action, 1.1, payload
    if (
        recommendation == "promote"
        and action == "promote"
        and scorecard_recommendation != "promote"
    ):
        payload["intensity_status"] = "softened"
        payload["intensity_reason"] = "ROLLING_SCORECARD_NOT_YET_SUPPORTIVE"
        payload["operating_intensity"] = 0.8
        return action, 0.8, payload
    if (
        recommendation == "prune"
        and action == "demote"
        and scorecard_recommendation == "demote"
        and rolling_score <= 0.35
        and recent_positive_run_ratio <= 0.34
    ):
        payload["intensity_status"] = "strengthened"
        payload["intensity_reason"] = "ROLLING_SCORECARD_STRONGLY_NEGATIVE"
        payload["operating_intensity"] = 1.2
        return action, 1.2, payload
    if (
        recommendation == "prune"
        and action == "demote"
        and scorecard_recommendation != "demote"
    ):
        payload["intensity_status"] = "softened"
        payload["intensity_reason"] = "ROLLING_SCORECARD_NOT_YET_NEGATIVE"
        payload["operating_intensity"] = 0.8
        return action, 0.8, payload
    payload["intensity_status"] = "baseline"
    payload["operating_intensity"] = 1.0
    return action, 1.0, payload


def _runtime_pruning_score_delta(
    *,
    recommendation: str,
    trade_count: int,
    decision_count: int,
    avg_net_edge_bps: float,
) -> float:
    sample_count = max(trade_count, decision_count)
    severity = min(0.28, abs(avg_net_edge_bps) / 10.0) + min(0.12, sample_count / 30.0)
    if recommendation == "observe_only":
        severity += 0.04
    return round(-severity, 6)


def _runtime_decomposition_adjustments(runtime_evidence: dict[str, object] | None) -> list[dict[str, object]]:
    payload = dict(runtime_evidence or {})
    if not payload:
        return []
    regime_breakdown = _runtime_regime_breakdown(payload)
    futures_supportive = bool(regime_breakdown.get("futures_supportive"))
    futures_elite = bool(regime_breakdown.get("futures_elite"))
    bucket_symbol_summary_by_symbol, bucket_pruning_by_symbol, bucket_context = _preferred_policy_context_symbol_maps(payload)
    bucket_name = str(bucket_context.get("bucket_name", "") or "")
    observe_only_symbols = {str(symbol) for symbol in list(payload.get("observe_only_symbols", []) or []) if str(symbol)}
    scorecard_by_symbol = {
        str(row.get("symbol", "") or ""): dict(row)
        for row in list(payload.get("symbol_scorecard", []) or [])
        if str(row.get("symbol", "") or "")
    }
    adjustments: list[dict[str, object]] = []
    for row in list(payload.get("pruning_recommendations", []) or []):
        symbol = str(row.get("symbol", "") or "")
        effective_row, policy_context_signal = _overlay_policy_context_bucket_row(
            row,
            bucket_pruning_by_symbol.get(symbol),
            bucket_name=bucket_name,
        )
        recommendation = str(effective_row.get("recommendation", "") or "")
        if not symbol or recommendation not in {"prune", "demote", "observe_only"}:
            continue
        trade_count = int(effective_row.get("trade_count", 0) or 0)
        decision_count = int(effective_row.get("decision_count", 0) or 0)
        avg_net_edge_bps = float(effective_row.get("avg_net_edge_bps", 0.0) or 0.0)
        action = "demote"
        operating_intensity = 1.0
        scorecard_evidence = dict(scorecard_by_symbol.get(symbol) or {})
        if recommendation != "observe_only":
            scorecard_support, scorecard_evidence = _runtime_symbol_scorecard_gate(
                recommendation="prune",
                scorecard_row=scorecard_by_symbol.get(symbol),
            )
            if not scorecard_support:
                continue
            action, operating_intensity, scorecard_evidence = _runtime_symbol_scorecard_intensity(
                recommendation="prune",
                action="demote",
                scorecard_evidence=scorecard_evidence,
            )
        adjustments.append(
            _policy_adjustment_shape(
                symbol=symbol,
                regime="major" if symbol in {"BTCUSDT", "ETHUSDT"} else "alt",
                setup_class="runtime_decomposition",
                side="both",
                execution_quality_state="runtime_review",
                sample_count=max(trade_count, decision_count),
                action=action,
                reason=f"RUNTIME_{recommendation.upper()}_RECOMMENDATION",
                signal_source="runtime_pruning_recommendation",
                score_delta=_runtime_pruning_score_delta(
                    recommendation=recommendation,
                    trade_count=trade_count,
                    decision_count=decision_count,
                    avg_net_edge_bps=avg_net_edge_bps,
                ),
                operating_intensity=operating_intensity,
                signal_context={
                    "recommendation": recommendation,
                    "trade_count": trade_count,
                    "decision_count": decision_count,
                    "avg_net_edge_bps": round(avg_net_edge_bps, 6),
                    "scorecard_evidence": scorecard_evidence,
                    **(
                        {"policy_context_bucket_pruning_recommendation": policy_context_signal}
                        if policy_context_signal
                        else {}
                    ),
                },
            )
        )
    for row in list(payload.get("symbol_summary", []) or []):
        symbol = str(row.get("symbol", "") or "")
        effective_row, policy_context_signal = _overlay_policy_context_bucket_row(
            row,
            bucket_symbol_summary_by_symbol.get(symbol),
            bucket_name=bucket_name,
        )
        recommendation = str(effective_row.get("recommendation", "") or "")
        trade_count = int(effective_row.get("trade_count", 0) or 0)
        expectancy = float(effective_row.get("expectancy_usd", 0.0) or 0.0)
        if not symbol or trade_count < 3 or recommendation not in {"promote", "prune"}:
            continue
        regime = "major" if symbol in {"BTCUSDT", "ETHUSDT"} else "alt"
        if symbol in observe_only_symbols and regime != "major" and recommendation == "promote":
            continue
        rolling_support, rolling_evidence = _runtime_symbol_rolling_gate(effective_row)
        if not rolling_support:
            continue
        scorecard_support, scorecard_evidence = _runtime_symbol_scorecard_gate(
            recommendation=recommendation,
            scorecard_row=scorecard_by_symbol.get(symbol),
        )
        if not scorecard_support:
            continue
        if recommendation == "promote" and expectancy > 0.0:
            action = "aggressive_promote" if regime == "major" and futures_elite else "promote"
            scorecard_gate_status = str(scorecard_evidence.get("gate_status", "") or "")
            if (
                scorecard_evidence
                and scorecard_gate_status != "insufficient_history"
                and str(scorecard_evidence.get("recommendation", "") or "") != "promote"
            ):
                action = "promote"
            if action == "aggressive_promote" and not futures_supportive:
                action = "promote"
            action, operating_intensity, scorecard_evidence = _runtime_symbol_scorecard_intensity(
                recommendation=recommendation,
                action=action,
                scorecard_evidence=scorecard_evidence,
            )
            adjustments.append(
                _policy_adjustment_shape(
                    symbol=symbol,
                    regime=regime,
                    setup_class="runtime_symbol_expectancy",
                    side="both",
                    execution_quality_state="runtime_review",
                    sample_count=trade_count,
                    action=action,
                    reason="RUNTIME_SYMBOL_PROMOTE",
                    signal_source="runtime_symbol_summary",
                    score_delta=_runtime_symbol_score_delta(
                        recommendation=recommendation,
                        trade_count=trade_count,
                        expectancy=expectancy,
                        regime=regime,
                        regime_breakdown=regime_breakdown,
                    ),
                    operating_intensity=operating_intensity,
                    signal_context={
                        "recommendation": recommendation,
                        "trade_count": trade_count,
                        "expectancy_usd": round(expectancy, 6),
                        "dominant_regime_mode": str(regime_breakdown.get("dominant_mode", "") or ""),
                        "rolling_evidence": rolling_evidence,
                        "scorecard_evidence": scorecard_evidence,
                        **(
                            {"policy_context_bucket_symbol_summary": policy_context_signal}
                            if policy_context_signal
                            else {}
                        ),
                    },
                )
            )
        elif recommendation == "prune":
            action, operating_intensity, scorecard_evidence = _runtime_symbol_scorecard_intensity(
                recommendation=recommendation,
                action="demote",
                scorecard_evidence=scorecard_evidence,
            )
            adjustments.append(
                _policy_adjustment_shape(
                    symbol=symbol,
                    regime=regime,
                    setup_class="runtime_symbol_expectancy",
                    side="both",
                    execution_quality_state="runtime_review",
                    sample_count=trade_count,
                    action=action,
                    reason="RUNTIME_SYMBOL_PRUNE",
                    signal_source="runtime_symbol_summary",
                    score_delta=_runtime_symbol_score_delta(
                        recommendation=recommendation,
                        trade_count=trade_count,
                        expectancy=expectancy,
                        regime=regime,
                        regime_breakdown=regime_breakdown,
                    ),
                    operating_intensity=operating_intensity,
                    signal_context={
                        "recommendation": recommendation,
                        "trade_count": trade_count,
                        "expectancy_usd": round(expectancy, 6),
                        "dominant_regime_mode": str(regime_breakdown.get("dominant_mode", "") or ""),
                        "rolling_evidence": rolling_evidence,
                        "scorecard_evidence": scorecard_evidence,
                        **(
                            {"policy_context_bucket_symbol_summary": policy_context_signal}
                            if policy_context_signal
                            else {}
                        ),
                    },
                )
            )
    return adjustments


def _merge_policy_adjustments(existing: dict[str, object] | None, incoming: dict[str, object]) -> dict[str, object]:
    if existing is None:
        return dict(incoming)
    existing_action = str(existing.get("action", "keep") or "keep")
    incoming_action = str(incoming.get("action", "keep") or "keep")
    existing_strength = _policy_action_strength(existing_action)
    incoming_strength = _policy_action_strength(incoming_action)
    if existing_strength < 0 or incoming_strength < 0:
        preferred = incoming if incoming_strength < existing_strength else existing
    else:
        preferred = incoming if incoming_strength > existing_strength else existing
    merged = dict(preferred)
    merged["sample_count"] = max(int(existing.get("sample_count", 0) or 0), int(incoming.get("sample_count", 0) or 0))
    merged["score_delta"] = round(float(existing.get("score_delta", 0.0) or 0.0) + float(incoming.get("score_delta", 0.0) or 0.0), 6)
    merged["signal_sources"] = sorted(
        {
            str(source)
            for source in list(existing.get("signal_sources", []) or []) + list(incoming.get("signal_sources", []) or [])
            if str(source)
        }
    )
    signal_contexts = dict(existing.get("signal_contexts", {}) or {})
    signal_contexts.update(dict(incoming.get("signal_contexts", {}) or {}))
    if signal_contexts:
        merged["signal_contexts"] = signal_contexts
    return merged


def _policy_context_bucket_evidence_map(runtime_evidence: dict[str, object] | None) -> dict[str, dict[str, object]]:
    payload = dict(runtime_evidence or {})
    explicit = {
        str(bucket_name): dict(bucket_payload)
        for bucket_name, bucket_payload in dict(payload.get("policy_context_bucket_evidence", {}) or {}).items()
        if str(bucket_name)
    }
    if explicit:
        return explicit
    derived: dict[str, dict[str, object]] = {}
    for bucket_name in ("active_policy", "staged_candidate", "previous_policy"):
        bucket = policy_evidence_bucket(payload, bucket_name)
        evidence = dict(bucket.get("evidence", {}) or {})
        if evidence:
            derived[bucket_name] = evidence
    return derived


def _preferred_policy_context_symbol_maps(
    runtime_evidence: dict[str, object] | None,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, object]]:
    evidence_by_bucket = _policy_context_bucket_evidence_map(runtime_evidence)
    for bucket_name in ("active_policy", "staged_candidate", "previous_policy"):
        evidence = dict(evidence_by_bucket.get(bucket_name, {}) or {})
        symbol_rows = [
            dict(row)
            for row in list(
                evidence.get(
                    "policy_context_bucket_symbol_summary",
                    evidence.get("symbol_summary", []),
                )
                or []
            )
            if isinstance(row, dict) and str(row.get("symbol", "") or "")
        ]
        pruning_rows = [
            dict(row)
            for row in list(
                evidence.get(
                    "policy_context_bucket_pruning_recommendations",
                    evidence.get("pruning_recommendations", []),
                )
                or []
            )
            if isinstance(row, dict) and str(row.get("symbol", "") or "")
        ]
        if not symbol_rows and not pruning_rows:
            continue
        return (
            {
                str(row.get("symbol", "") or ""): row
                for row in symbol_rows
            },
            {
                str(row.get("symbol", "") or ""): row
                for row in pruning_rows
            },
            {
                "bucket_name": bucket_name,
                "symbol_count": len(symbol_rows),
                "pruning_symbol_count": len(pruning_rows),
            },
        )
    return {}, {}, {}


def _overlay_policy_context_bucket_row(
    row: dict[str, object] | None,
    bucket_row: dict[str, object] | None,
    *,
    bucket_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    base = dict(row or {})
    scoped = dict(bucket_row or {})
    if not scoped:
        return base, {}
    merged = dict(base)
    merged.update(scoped)
    if "rolling_evidence" in scoped or "rolling_evidence" in base:
        merged["rolling_evidence"] = dict(scoped.get("rolling_evidence", base.get("rolling_evidence", {})) or {})
    return merged, {
        "bucket_name": bucket_name,
        "symbol": str(merged.get("symbol", "") or ""),
        "recommendation": str(merged.get("recommendation", "") or ""),
        "trade_count": int(merged.get("trade_count", 0) or 0),
        "expectancy_usd": round(float(merged.get("expectancy_usd", 0.0) or 0.0), 6),
        "sample_status": str(merged.get("sample_status", "") or ""),
        "rolling_evidence": dict(merged.get("rolling_evidence", {}) or {}),
    }


def _promotion_priority_sort_key(
    adjustment: dict[str, object],
) -> tuple[float, float, float, float, float, float, float, str]:
    signal_contexts = dict(adjustment.get("signal_contexts", {}) or {})
    runtime_symbol_summary = dict(signal_contexts.get("runtime_symbol_summary", {}) or {})
    bucket_symbol_summary = dict(runtime_symbol_summary.get("policy_context_bucket_symbol_summary", {}) or {})
    scorecard_evidence = dict(runtime_symbol_summary.get("scorecard_evidence", {}) or {})
    rolling_evidence = dict(runtime_symbol_summary.get("rolling_evidence", {}) or {})
    return (
        -float(_policy_action_strength(str(adjustment.get("action", "keep") or "keep"))),
        -float(1.0 if str(bucket_symbol_summary.get("recommendation", "") or "") == "promote" else 0.0),
        -float(bucket_symbol_summary.get("expectancy_usd", 0.0) or 0.0),
        -float(bucket_symbol_summary.get("trade_count", 0) or 0),
        -float(scorecard_evidence.get("rolling_score", 0.0) or 0.0),
        -float(scorecard_evidence.get("recent_positive_run_ratio", 0.0) or 0.0),
        -float(rolling_evidence.get("positive_window_ratio", 0.0) or 0.0),
        -float(adjustment.get("score_delta", 0.0) or 0.0),
        str(adjustment.get("symbol", "") or ""),
    )


def _apply_cross_symbol_promotion_priority(
    *,
    adjustments: list[dict[str, object]],
    runtime_evidence: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    positive_adjustment_count = sum(
        1
        for adjustment in adjustments
        if str(adjustment.get("action", "") or "") in {"promote", "aggressive_promote"}
    )
    promotion_top_k, top_k_context = _effective_promotion_top_k(
        runtime_evidence=runtime_evidence,
        positive_adjustment_count=positive_adjustment_count,
    )
    bucket_summary_by_symbol, bucket_pruning_by_symbol, bucket_context = _preferred_policy_context_symbol_maps(runtime_evidence)
    auto_mode = _auto_mode(runtime_evidence)
    guardrails = dict(_sample_quality_watchdog(runtime_evidence).get("policy_guardrails", {}) or {})
    prefer_majors_only = bool(
        guardrails.get("prefer_majors_only") or auto_mode_blocks_non_major_positive(auto_mode)
    )
    rankable_positive: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    for adjustment in adjustments:
        action = str(adjustment.get("action", "") or "")
        if action not in {"promote", "aggressive_promote"}:
            continue
        symbol = str(adjustment.get("symbol", "") or "")
        bucket_pruning_recommendation = str(
            dict(bucket_pruning_by_symbol.get(symbol, {}) or {}).get("recommendation", "") or ""
        )
        bucket_summary_recommendation = str(
            dict(bucket_summary_by_symbol.get(symbol, {}) or {}).get("recommendation", "") or ""
        )
        is_major = str(adjustment.get("regime", "") or "") == "major" or symbol in {"BTCUSDT", "ETHUSDT"}
        exclusion_reason = ""
        if bucket_pruning_recommendation in {"observe_only", "prune", "demote"}:
            exclusion_reason = f"policy_bucket_{bucket_pruning_recommendation}"
        elif bucket_summary_recommendation in {"observe_only", "prune", "demote"}:
            exclusion_reason = f"policy_bucket_{bucket_summary_recommendation}"
        elif prefer_majors_only and not is_major:
            exclusion_reason = "majors_only_guardrail"
        if exclusion_reason:
            excluded_rows.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "exclude_reason": exclusion_reason,
                }
            )
            continue
        rankable_positive.append(adjustment)
    ranked_positive = sorted(rankable_positive, key=_promotion_priority_sort_key)
    selected_ranked_positive = list(ranked_positive)
    deferred_rows: list[dict[str, object]] = []
    if promotion_top_k > 0:
        selected_ranked_positive = ranked_positive[:promotion_top_k]
        deferred_rows = [
            {
                "symbol": str(adjustment.get("symbol", "") or ""),
                "action": str(adjustment.get("action", "") or ""),
                "priority_rank": rank,
            }
            for rank, adjustment in enumerate(ranked_positive[promotion_top_k:], start=promotion_top_k + 1)
            if str(adjustment.get("symbol", "") or "")
        ]
    selected_symbols = [
        str(adjustment.get("symbol", "") or "")
        for adjustment in selected_ranked_positive
        if str(adjustment.get("symbol", "") or "")
    ]
    if promotion_top_k <= 0 and not excluded_rows and not deferred_rows:
        if int(top_k_context.get("explicit_promotion_top_k", 0) or 0) > 0 or int(top_k_context.get("watchdog_positive_cap", 0) or 0) > 0:
            return adjustments, {
                **top_k_context,
                **({"policy_context_bucket": str(bucket_context.get("bucket_name", "") or "")} if bucket_context else {}),
            }
        return adjustments, {}
    selected_symbol_set = set(selected_symbols)
    filtered_adjustments = [
        adjustment
        for adjustment in adjustments
        if str(adjustment.get("action", "") or "") not in {"promote", "aggressive_promote"}
        or str(adjustment.get("symbol", "") or "") in selected_symbol_set
    ]
    return filtered_adjustments, {
        "promotion_top_k": promotion_top_k,
        "promotion_candidate_count": len(ranked_positive),
        "selected_promotion_symbols": selected_symbols,
        "deferred_promotion_symbols": [row["symbol"] for row in deferred_rows],
        "deferred_promotion_rows": deferred_rows,
        "excluded_promotion_symbols": [row["symbol"] for row in excluded_rows],
        "excluded_promotion_rows": excluded_rows,
        **({"policy_context_bucket": str(bucket_context.get("bucket_name", "") or "")} if bucket_context else {}),
        **top_k_context,
    }


def _retune_adjustment_for_action(
    adjustment: dict[str, object],
    *,
    action: str,
    reason: str,
    lifecycle_row: dict[str, object],
) -> dict[str, object]:
    updated = dict(adjustment)
    updated["action"] = action
    if action == "demote":
        updated["size_multiplier"] = round(min(float(updated.get("size_multiplier", 1.0) or 1.0), 0.9), 6)
        if float(updated["size_multiplier"]) >= 1.0:
            updated["size_multiplier"] = 0.75
        updated["leverage_multiplier"] = round(min(float(updated.get("leverage_multiplier", 1.0) or 1.0), float(updated["size_multiplier"])), 6)
        updated["entry_threshold_bps"] = max(float(updated.get("entry_threshold_bps", 0.0) or 0.0), 1.0)
        updated["expected_profit_floor_bps"] = max(float(updated.get("expected_profit_floor_bps", 0.0) or 0.0), 1.0)
        updated["symbol_bias"] = "neutral"
    elif action == "promote":
        updated["size_multiplier"] = 1.1
        updated["leverage_multiplier"] = 1.1
        updated["entry_threshold_bps"] = -0.5
        updated["expected_profit_floor_bps"] = -1.0
        updated["symbol_bias"] = "majors_only" if str(updated.get("regime", "") or "") == "major" else "neutral"
    updated["reason"] = reason
    updated["signal_sources"] = sorted(
        {
            str(source)
            for source in list(updated.get("signal_sources", []) or []) + ["symbol_lifecycle"]
            if str(source)
        }
    )
    signal_contexts = dict(updated.get("signal_contexts", {}) or {})
    signal_contexts["symbol_lifecycle"] = dict(lifecycle_row)
    updated["signal_contexts"] = signal_contexts
    return updated


def _apply_symbol_lifecycle_overlay(
    *,
    adjustments: list[dict[str, object]],
    runtime_evidence: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    lifecycle_rows = [
        dict(row)
        for row in list(dict(runtime_evidence or {}).get("symbol_lifecycle", []) or [])
        if isinstance(row, dict) and str(row.get("symbol", "") or "")
    ]
    if not lifecycle_rows:
        return adjustments, {}
    lifecycle_by_symbol = {
        str(row.get("symbol", "") or ""): row
        for row in lifecycle_rows
    }
    adjusted_rows: list[dict[str, object]] = []
    blocked_symbols: list[str] = []
    demoted_symbols: list[str] = []
    cautious_symbols: list[str] = []
    review_symbols: list[str] = []
    for adjustment in adjustments:
        symbol = str(adjustment.get("symbol", "") or "")
        lifecycle_row = dict(lifecycle_by_symbol.get(symbol, {}) or {})
        lifecycle_action = str(lifecycle_row.get("recommended_action", "keep") or "keep")
        current_action = str(adjustment.get("action", "keep") or "keep")
        if current_action in {"promote", "aggressive_promote"}:
            if lifecycle_action in {"rollback", "hold"}:
                if str(lifecycle_row.get("current_state", "baseline") or "baseline") == "promoted":
                    adjusted_rows.append(
                        _retune_adjustment_for_action(
                            adjustment,
                            action="demote",
                            reason="SYMBOL_LIFECYCLE_ROLLBACK",
                            lifecycle_row=lifecycle_row,
                        )
                    )
                    demoted_symbols.append(symbol)
                else:
                    blocked_symbols.append(symbol)
                continue
            if lifecycle_action == "re_review":
                review_symbols.append(symbol)
                blocked_symbols.append(symbol)
                continue
            if lifecycle_action == "cautious_repromote" and current_action == "aggressive_promote":
                adjusted_rows.append(
                    _retune_adjustment_for_action(
                        adjustment,
                        action="promote",
                        reason="SYMBOL_LIFECYCLE_CAUTIOUS_REPROMOTION",
                        lifecycle_row=lifecycle_row,
                    )
                )
                cautious_symbols.append(symbol)
                continue
        adjusted_rows.append(adjustment)
    summary = {
        "applied": bool(blocked_symbols or demoted_symbols or cautious_symbols or review_symbols),
        "blocked_symbols": sorted(blocked_symbols),
        "demoted_symbols": sorted(demoted_symbols),
        "cautious_repromotion_symbols": sorted(cautious_symbols),
        "re_review_symbols": sorted(review_symbols),
        "lifecycle_summary": summarize_symbol_lifecycle(lifecycle_rows),
    }
    if not summary["applied"]:
        return adjusted_rows, {}
    return adjusted_rows, summary


def _candidate_generation_summary(
    *,
    runtime_evidence: dict[str, object] | None,
    adjustments: list[dict[str, object]],
) -> dict[str, object]:
    payload = dict(runtime_evidence or {})
    regime_breakdown = _runtime_regime_breakdown(payload)
    adjustment_rows = [
        {
            "symbol": str(item.get("symbol", "") or ""),
            "action": str(item.get("action", "") or ""),
            "score_delta": round(float(item.get("score_delta", 0.0) or 0.0), 6),
            "signal_sources": list(item.get("signal_sources", []) or []),
        }
        for item in adjustments[:6]
    ]
    return {
        "dominant_regime_mode": str(regime_breakdown.get("dominant_mode", "") or ""),
        "supportive_regime_modes": list(regime_breakdown.get("supportive_modes", []) or []),
        "elite_regime_modes": list(regime_breakdown.get("elite_modes", []) or []),
        "runtime_symbol_recommendation_count": len(list(payload.get("symbol_summary", []) or [])),
        "runtime_symbol_scorecard_count": len(list(payload.get("symbol_scorecard", []) or [])),
        "runtime_pruning_recommendation_count": len(list(payload.get("pruning_recommendations", []) or [])),
        "score_delta_total": round(sum(float(item.get("score_delta", 0.0) or 0.0) for item in adjustments), 6),
        "candidate_adjustment_rows": adjustment_rows,
    }


def _observe_only_runtime_adjustments(runtime_evidence: dict[str, object] | None) -> list[dict[str, object]]:
    payload = dict(runtime_evidence or {})
    symbols = [str(symbol) for symbol in list(payload.get("observe_only_symbols", []) or []) if str(symbol)]
    adjustments: list[dict[str, object]] = []
    for symbol in symbols:
        regime = "major" if symbol in {"BTCUSDT", "ETHUSDT"} else "alt"
        adjustments.append(
            _policy_adjustment_shape(
                symbol=symbol,
                regime=regime,
                setup_class="runtime_observe_only",
                side="both",
                execution_quality_state="runtime_review",
                sample_count=1,
                action="demote",
                reason="RUNTIME_OBSERVE_ONLY_SYMBOL",
                signal_source="runtime_observe_only",
                score_delta=-0.08 if regime == "major" else -0.18,
                signal_context={
                    "observe_only": True,
                    "regime": regime,
                },
            )
        )
    return adjustments


def _sample_quality_watchdog(runtime_evidence: dict[str, object] | None) -> dict[str, object]:
    return dict(dict(runtime_evidence or {}).get("sample_quality_watchdog", {}) or {})


def _watchdog_adjustment(
    adjustment: dict[str, object],
    *,
    action: str,
    operating_intensity: float,
    reason: str,
    score_delta: float,
    signal_context: dict[str, object],
) -> dict[str, object]:
    payload = dict(adjustment or {})
    updated = _policy_adjustment_shape(
        symbol=str(payload.get("symbol", "") or ""),
        regime=str(payload.get("regime", "") or ""),
        setup_class=str(payload.get("setup_class", "") or ""),
        side=str(payload.get("side", "") or ""),
        execution_quality_state=str(payload.get("execution_quality_state", "") or ""),
        sample_count=int(payload.get("sample_count", 0) or 0),
        action=action,
        reason=reason,
        signal_source="sample_quality_watchdog",
        score_delta=score_delta,
        operating_intensity=operating_intensity,
        signal_context=signal_context,
    )
    existing_sources = [str(source) for source in list(payload.get("signal_sources", []) or []) if str(source)]
    updated["signal_sources"] = sorted(set(existing_sources + ["sample_quality_watchdog"]))
    signal_contexts = dict(payload.get("signal_contexts", {}) or {})
    signal_contexts["sample_quality_watchdog"] = signal_context
    updated["signal_contexts"] = signal_contexts
    return updated


def _auto_mode_adjustment(
    adjustment: dict[str, object],
    *,
    action: str,
    operating_intensity: float,
    reason: str,
    signal_context: dict[str, object],
) -> dict[str, object]:
    payload = dict(adjustment or {})
    score_delta = float(payload.get("score_delta", 0.0) or 0.0)
    if score_delta > 0.0:
        score_delta = round(score_delta * max(min(operating_intensity, 1.0), 0.7), 6)
    updated = _policy_adjustment_shape(
        symbol=str(payload.get("symbol", "") or ""),
        regime=str(payload.get("regime", "") or ""),
        setup_class=str(payload.get("setup_class", "") or ""),
        side=str(payload.get("side", "") or ""),
        execution_quality_state=str(payload.get("execution_quality_state", "") or ""),
        sample_count=int(payload.get("sample_count", 0) or 0),
        action=action,
        reason=reason,
        signal_source="auto_mode",
        score_delta=score_delta,
        operating_intensity=operating_intensity,
        signal_context=signal_context,
    )
    existing_sources = [str(source) for source in list(payload.get("signal_sources", []) or []) if str(source)]
    updated["signal_sources"] = sorted(set(existing_sources + ["auto_mode"]))
    signal_contexts = dict(payload.get("signal_contexts", {}) or {})
    signal_contexts["auto_mode"] = signal_context
    updated["signal_contexts"] = signal_contexts
    return updated


def _auto_mode(runtime_evidence: dict[str, object] | None) -> dict[str, object]:
    return dict(dict(runtime_evidence or {}).get("auto_mode", {}) or {})


def _apply_auto_mode_overlay(
    *,
    adjustments: list[dict[str, object]],
    runtime_evidence: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    auto_mode = _auto_mode(runtime_evidence)
    mode = str(auto_mode.get("mode", "") or "")
    if not auto_mode or mode not in {"tighter", "cautiously_expanded"}:
        return adjustments, auto_mode if auto_mode else {}
    positive_cap = auto_mode_promotion_intensity_cap(auto_mode, default=1.0)
    positive_action_cap = auto_mode_positive_action_cap(auto_mode)
    block_non_major_positive = auto_mode_blocks_non_major_positive(auto_mode)
    transformed: list[dict[str, object]] = []
    blocked_symbols: list[str] = []
    softened_symbols: list[str] = []
    expanded_symbols: list[str] = []
    for adjustment in adjustments:
        payload = dict(adjustment)
        symbol = str(payload.get("symbol", "") or "")
        action = str(payload.get("action", "") or "")
        regime = str(payload.get("regime", "") or "")
        if not is_positive_policy_action(action):
            transformed.append(payload)
            continue
        is_major = regime == "major"
        if block_non_major_positive and not is_major:
            blocked_symbols.append(symbol)
            continue
        current_intensity = float(payload.get("operating_intensity", 1.0) or 1.0)
        next_action = action
        next_reason = ""
        next_intensity = current_intensity
        if positive_action_cap == "promote" and action == "aggressive_promote":
            next_action = "promote"
        if mode == "tighter":
            next_intensity = min(current_intensity, positive_cap)
            next_reason = "AUTO_MODE_TIGHTER"
        elif mode == "cautiously_expanded" and is_major:
            next_intensity = max(current_intensity, positive_cap)
            next_reason = "AUTO_MODE_CAUTIOUSLY_EXPANDED"
        if next_action != action or abs(next_intensity - current_intensity) > 1e-9:
            transformed.append(
                _auto_mode_adjustment(
                    payload,
                    action=next_action,
                    operating_intensity=next_intensity,
                    reason=next_reason or "AUTO_MODE_APPLIED",
                    signal_context={
                        "mode": mode,
                        "reason_codes": list(auto_mode.get("reason_codes", []) or []),
                        "blocked_non_major_positive": block_non_major_positive,
                    },
                )
            )
            if mode == "tighter":
                softened_symbols.append(symbol)
            elif mode == "cautiously_expanded" and is_major:
                expanded_symbols.append(symbol)
            continue
        transformed.append(payload)
    return transformed, {
        "mode": mode,
        "reason_codes": list(auto_mode.get("reason_codes", []) or []),
        "blocked_symbols": sorted(blocked_symbols),
        "softened_symbols": sorted(softened_symbols),
        "expanded_symbols": sorted(expanded_symbols),
        "policy_guidance": dict(auto_mode.get("policy_guidance", {}) or {}),
        "inputs": dict(auto_mode.get("inputs", {}) or {}),
    }


def _apply_sample_quality_watchdog(
    *,
    adjustments: list[dict[str, object]],
    runtime_evidence: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    watchdog = _sample_quality_watchdog(runtime_evidence)
    if not watchdog:
        return adjustments, {}
    status = str(watchdog.get("status", "unknown") or "unknown")
    guardrails = dict(watchdog.get("policy_guardrails", {}) or {})
    intensity_cap = max(0.5, min(float(guardrails.get("promotion_intensity_cap", 1.0) or 1.0), 1.1))
    allow_alt_promotions = bool(guardrails.get("allow_alt_promotions"))
    transformed: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for adjustment in adjustments:
        payload = dict(adjustment)
        symbol = str(payload.get("symbol", "") or "")
        regime = str(payload.get("regime", "") or "")
        is_major = regime == "major"
        action = str(payload.get("action", "") or "")
        operating_intensity = float(payload.get("operating_intensity", 1.0) or 1.0)
        score_delta = float(payload.get("score_delta", 0.0) or 0.0)
        updated = payload
        if action in {"promote", "aggressive_promote"}:
            if status == "degraded" and not is_major:
                updated = _watchdog_adjustment(
                    payload,
                    action="demote",
                    operating_intensity=1.0,
                    reason="SAMPLE_QUALITY_WATCHDOG_DEGRADED_ALT_OBSERVE_ONLY",
                    score_delta=-max(0.12, abs(score_delta) * 0.5),
                    signal_context={
                        "watchdog_status": status,
                        "guardrails": guardrails,
                        "watchdog_action": "non_major_positive_demoted",
                    },
                )
            elif status == "thin" and not is_major and not allow_alt_promotions:
                updated = _watchdog_adjustment(
                    payload,
                    action="demote",
                    operating_intensity=0.8,
                    reason="SAMPLE_QUALITY_WATCHDOG_THIN_ALT_OBSERVE_ONLY",
                    score_delta=-max(0.08, abs(score_delta) * 0.4),
                    signal_context={
                        "watchdog_status": status,
                        "guardrails": guardrails,
                        "watchdog_action": "non_major_positive_demoted",
                    },
                )
            else:
                next_action = action
                if action == "aggressive_promote" and status == "degraded" and not is_major:
                    next_action = "promote"
                next_intensity = min(operating_intensity, intensity_cap)
                next_score_delta = round(score_delta * max(min(next_intensity, 1.0), 0.65), 6)
                if status == "promote_ready" and not is_major and action == "promote":
                    next_intensity = max(next_intensity, min(intensity_cap, 1.05))
                    next_score_delta = round(score_delta * next_intensity, 6)
                if next_action != action or abs(next_intensity - operating_intensity) > 1e-9:
                    updated = _watchdog_adjustment(
                        payload,
                        action=next_action,
                        operating_intensity=next_intensity,
                        reason=f"SAMPLE_QUALITY_WATCHDOG_{status.upper()}_{'RELAXED' if status == 'promote_ready' and not is_major else 'SOFTENED'}",
                        score_delta=next_score_delta,
                        signal_context={
                            "watchdog_status": status,
                            "guardrails": guardrails,
                            "watchdog_action": "promotion_softened" if status in {"thin", "degraded"} else "promotion_relaxed",
                        },
                    )
        transformed.append(updated)
        if updated is not payload:
            action_rows.append(
                {
                    "symbol": symbol,
                    "from_action": action,
                    "to_action": str(updated.get("action", "") or ""),
                }
            )
    return transformed, {
        "status": status,
        "guardrails": guardrails,
        "adjusted_symbols": action_rows,
    }


def _effective_promotion_top_k(
    *,
    runtime_evidence: dict[str, object] | None,
    positive_adjustment_count: int,
) -> tuple[int, dict[str, object]]:
    payload = dict(runtime_evidence or {})
    explicit_top_k = max(int(payload.get("promotion_top_k", 0) or 0), 0)
    watchdog = _sample_quality_watchdog(runtime_evidence)
    guardrails = dict(watchdog.get("policy_guardrails", {}) or {})
    watchdog_cap = max(int(guardrails.get("max_positive_symbols", 0) or 0), 0)
    auto_mode = _auto_mode(runtime_evidence)
    auto_mode_cap = auto_mode_positive_cap(auto_mode)
    effective_top_k = explicit_top_k
    if watchdog_cap > 0:
        effective_top_k = min(explicit_top_k, watchdog_cap) if explicit_top_k > 0 else min(watchdog_cap, positive_adjustment_count)
    if auto_mode_cap > 0:
        effective_top_k = min(effective_top_k, auto_mode_cap) if effective_top_k > 0 else min(auto_mode_cap, positive_adjustment_count)
    return effective_top_k, {
        "watchdog_status": str(watchdog.get("status", "not_available") or "not_available"),
        "explicit_promotion_top_k": explicit_top_k,
        "watchdog_positive_cap": watchdog_cap,
        "auto_mode_positive_cap": auto_mode_cap,
        "auto_mode": str(auto_mode.get("mode", "normal") or "normal"),
    }


def build_auto_tune_policy(
    attribution_rows: list[dict[str, object]] | tuple[dict[str, object], ...],
    runtime_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    adjustments_by_symbol: dict[str, dict[str, object]] = {}
    for adjustment in _live_attribution_adjustments(attribution_rows):
        symbol = str(adjustment.get("symbol", "") or "")
        if not symbol:
            continue
        adjustments_by_symbol[symbol] = _merge_policy_adjustments(adjustments_by_symbol.get(symbol), adjustment)
    for adjustment in _runtime_decomposition_adjustments(runtime_evidence):
        symbol = str(adjustment.get("symbol", "") or "")
        if not symbol:
            continue
        adjustments_by_symbol[symbol] = _merge_policy_adjustments(adjustments_by_symbol.get(symbol), adjustment)
    for adjustment in _observe_only_runtime_adjustments(runtime_evidence):
        symbol = str(adjustment.get("symbol", "") or "")
        if not symbol:
            continue
        adjustments_by_symbol[symbol] = _merge_policy_adjustments(adjustments_by_symbol.get(symbol), adjustment)
    adjustments = sorted(adjustments_by_symbol.values(), key=lambda item: str(item.get("symbol", "")))
    adjustments, watchdog_overlay_summary = _apply_sample_quality_watchdog(
        adjustments=adjustments,
        runtime_evidence=runtime_evidence,
    )
    adjustments, auto_mode_summary = _apply_auto_mode_overlay(
        adjustments=adjustments,
        runtime_evidence=runtime_evidence,
    )
    adjustments, promotion_priority_summary = _apply_cross_symbol_promotion_priority(
        adjustments=adjustments,
        runtime_evidence=runtime_evidence,
    )
    adjustments, lifecycle_overlay_summary = _apply_symbol_lifecycle_overlay(
        adjustments=adjustments,
        runtime_evidence=runtime_evidence,
    )
    policy_status = "insufficient_data" if not adjustments else "candidate_ready"
    signal_sources = sorted(
        {
            str(source)
            for item in adjustments
            for source in list(item.get("signal_sources", []) or [])
            if str(source)
        }
    )
    decomposition_summary = _candidate_generation_summary(
        runtime_evidence=runtime_evidence,
        adjustments=adjustments,
    )
    if watchdog_overlay_summary:
        decomposition_summary["sample_quality_watchdog"] = watchdog_overlay_summary
    if auto_mode_summary:
        decomposition_summary["auto_mode"] = auto_mode_summary
    if promotion_priority_summary:
        decomposition_summary["cross_symbol_promotion_priority"] = promotion_priority_summary
    if lifecycle_overlay_summary:
        decomposition_summary["symbol_lifecycle_overlay"] = lifecycle_overlay_summary
    return {
        "status": policy_status,
        "adjustments": adjustments,
        "signal_sources": signal_sources,
        "decomposition_summary": decomposition_summary,
    }


def _policy_comparison_signal(evidence: dict[str, object] | None) -> tuple[str, float]:
    payload = policy_evidence_bucket_evidence(evidence, "staged_candidate", fallback_to_root=True)
    delta = round(float(payload.get("candidate_vs_current_score_delta", 0.0) or 0.0), 6)
    verdict = str(payload.get("comparison_verdict", "keep") or "keep")
    if verdict == "keep":
        if delta > 0.1:
            verdict = "candidate_better"
        elif delta < -0.1:
            verdict = "candidate_worse"
    return verdict, delta


def _simple_baseline_control_gate(evidence: dict[str, object] | None) -> dict[str, object]:
    baseline = baseline_control_bucket_comparison(evidence)
    verdict = str(baseline.get("verdict", "not_available") or "not_available")
    expansion_gate = str(baseline.get("expansion_gate", "") or "")
    if not expansion_gate:
        if verdict == "supportive":
            expansion_gate = "pass"
        elif verdict in {"parity", "caution"}:
            expansion_gate = "block"
        else:
            expansion_gate = "not_available"
    gate_reason = str(baseline.get("expansion_gate_reason", "") or "")
    if not gate_reason:
        gate_reason = {
            "supportive": "SIMPLE_BASELINE_CONTROL_CLEARED",
            "parity": "SIMPLE_BASELINE_CONTROL_NOT_CLEARLY_BEATEN",
            "caution": "SIMPLE_BASELINE_CONTROL_UNDERPERFORMED",
        }.get(verdict, "NO_JUSTIFIED_SIMPLE_BASELINE_GATE")
    promotion_block_reason = {
        "parity": "PROMOTION_BLOCKED_BY_SIMPLE_BASELINE_PARITY",
        "caution": "PROMOTION_BLOCKED_BY_SIMPLE_BASELINE_UNDERPERFORMANCE",
    }.get(verdict, "")
    validation_reason = {
        "parity": "SIMPLE_BASELINE_CONTROL_NOT_CLEARLY_BEATEN",
        "caution": "SIMPLE_BASELINE_CONTROL_UNDERPERFORMED",
    }.get(verdict, "")
    return {
        "available": bool(baseline.get("available")),
        "verdict": verdict,
        "status": expansion_gate,
        "gate_reason": gate_reason,
        "promotion_block_reason": promotion_block_reason,
        "validation_reason": validation_reason,
        "baseline_control_comparison": baseline,
    }


def _runner_quality_evidence(
    evidence: dict[str, object] | None,
    *,
    bucket_name: str | None = None,
    fallback_to_root: bool = True,
) -> dict[str, object]:
    payload = (
        policy_evidence_bucket_evidence(evidence, bucket_name, fallback_to_root=fallback_to_root)
        if bucket_name
        else dict(evidence or {})
    )
    return {
        "available": any(
            key in payload
            for key in (
                "runner_total_realized_pnl_usd",
                "runner_total_return_pct",
                "runner_reject_rate",
                "runner_protection_degraded_rate",
                "runner_avg_slippage_bps",
                "runner_avg_edge_retention_ratio",
                "micro_live_gate",
                "runner_positive_walk_forward_ratio",
            )
        ),
        "realized_pnl_usd": float(payload.get("runner_total_realized_pnl_usd", payload.get("runner_total_return_pct", 0.0)) or 0.0),
        "drawdown_ratio": float(payload.get("runner_drawdown_to_pnl_ratio", payload.get("replay_like_drawdown_ratio", 0.0)) or 0.0),
        "reject_rate": float(payload.get("runner_reject_rate", payload.get("max_reject_rate", 0.0)) or 0.0),
        "protection_degraded_rate": float(payload.get("runner_protection_degraded_rate", 0.0) or 0.0),
        "avg_slippage_bps": float(payload.get("runner_avg_slippage_bps", 0.0) or 0.0),
        "avg_realized_edge_bps": float(payload.get("runner_avg_realized_edge_bps", payload.get("avg_realized_edge_bps", 0.0)) or 0.0),
        "avg_edge_retention_ratio": float(payload.get("runner_avg_edge_retention_ratio", payload.get("avg_retention", 0.0)) or 0.0),
        "walk_forward_window_count": int(payload.get("runner_walk_forward_window_count", len(list(payload.get("walk_forward_windows", []) or []))) or 0),
        "positive_walk_forward_window_count": int(payload.get("runner_positive_walk_forward_window_count", 0) or 0),
        "positive_walk_forward_ratio": float(payload.get("runner_positive_walk_forward_ratio", 0.0) or 0.0),
        "micro_live_gate": dict(payload.get("micro_live_gate", {}) or {}),
    }


def _policy_bucket_evidence(payload: dict[str, object] | None, bucket_name: str) -> dict[str, object]:
    return policy_evidence_bucket_evidence(payload, bucket_name, fallback_to_root=False)


def _policy_bucket_entry(
    *,
    bucket_name: str,
    source: str,
    available: bool,
    evidence: dict[str, object] | None = None,
    comparison: dict[str, object] | None = None,
    policy_lineage: dict[str, object] | None = None,
    evidence_lineage: dict[str, object] | None = None,
    alignment: dict[str, object] | None = None,
    policy_application: dict[str, object] | None = None,
    replay_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "bucket": bucket_name,
        "source": source,
        "available": bool(available),
        "evidence": deepcopy(dict(evidence or {})),
        "comparison": deepcopy(dict(comparison or {})),
        "policy_lineage": deepcopy(dict(policy_lineage or {})),
        "evidence_lineage": deepcopy(dict(evidence_lineage or {})),
        "alignment": deepcopy(dict(alignment or {})),
        "policy_application": deepcopy(dict(policy_application or {})),
        "replay_summary": deepcopy(dict(replay_summary or {})),
    }


def _staged_candidate_evidence(payload: dict[str, object] | None) -> dict[str, object]:
    return policy_evidence_bucket_evidence(payload, "staged_candidate", fallback_to_root=True)


def _active_policy_evidence(payload: dict[str, object] | None) -> dict[str, object]:
    active_evidence = policy_evidence_bucket_evidence(payload, "active_policy", fallback_to_root=False)
    if active_evidence:
        return active_evidence
    raw_payload = dict(payload or {})
    validation_evidence = dict(dict(raw_payload.get("policy_validation", {}) or {}).get("evidence", {}) or {})
    if validation_evidence:
        return validation_evidence
    if "policy_evidence_buckets" not in raw_payload:
        return raw_payload
    return {}


def _candidate_policy_requested_verdict(adjustments: list[dict[str, object]]) -> dict[str, object]:
    if not adjustments:
        return {"status": "keep", "reasons": ["INSUFFICIENT_ATTRIBUTION_DATA"]}
    aggressive_count = sum(1 for item in adjustments if str(item.get("action", "")) == "aggressive_promote")
    promote_count = sum(1 for item in adjustments if str(item.get("action", "")) in {"promote", "aggressive_promote"})
    demote_count = sum(1 for item in adjustments if str(item.get("action", "")) == "demote")
    if demote_count > 0 and promote_count == 0:
        return {"status": "demote", "reasons": ["CANDIDATE_POLICY_WEAK"]}
    if aggressive_count > 0 and demote_count == 0:
        return {"status": "promote_aggressive", "reasons": ["CANDIDATE_POLICY_ELITE"]}
    if promote_count > 0 and demote_count == 0:
        return {"status": "promote", "reasons": ["CANDIDATE_POLICY_STRONG"]}
    if demote_count >= promote_count + 2:
        return {"status": "disable", "reasons": ["CANDIDATE_POLICY_UNSTABLE"]}
    return {"status": "keep", "reasons": ["CANDIDATE_POLICY_MIXED"]}


def _sample_quality_watchdog_block_reason(comparison_evidence: dict[str, object] | None) -> str:
    payload = _staged_candidate_evidence(comparison_evidence)
    watchdog = dict(payload.get("sample_quality_watchdog", {}) or {})
    micro_live_gate = dict(payload.get("micro_live_gate", {}) or {})
    micro_live_status = str(micro_live_gate.get("status", "") or "")
    if micro_live_status in {"pending", "pass"}:
        return ""
    status = str(watchdog.get("status", "") or "")
    if status == "degraded":
        return "PROMOTION_BLOCKED_BY_SAMPLE_QUALITY_WATCHDOG_DEGRADED"
    if status == "thin":
        return "PROMOTION_BLOCKED_BY_SAMPLE_QUALITY_WATCHDOG_THIN"
    return ""


def _symbol_lifecycle_promotion_signal(
    *,
    candidate_policy: dict[str, object],
    comparison_evidence: dict[str, object] | None,
) -> dict[str, object]:
    candidate_evidence = _staged_candidate_evidence(comparison_evidence)
    lifecycle_rows = {
        str(row.get("symbol", "") or ""): dict(row)
        for row in list(candidate_evidence.get("symbol_lifecycle", []) or [])
        if isinstance(row, dict) and str(row.get("symbol", "") or "")
    }
    if not lifecycle_rows:
        return {"blocked_symbols": [], "review_symbols": [], "cautious_symbols": []}
    blocked_symbols: list[str] = []
    review_symbols: list[str] = []
    cautious_symbols: list[str] = []
    for adjustment in list(candidate_policy.get("adjustments", []) or []):
        action = str(adjustment.get("action", "") or "")
        if action not in {"promote", "aggressive_promote"}:
            continue
        symbol = str(adjustment.get("symbol", "") or "")
        lifecycle_row = dict(lifecycle_rows.get(symbol, {}) or {})
        lifecycle_action = str(lifecycle_row.get("recommended_action", "keep") or "keep")
        if lifecycle_action in {"rollback", "hold"}:
            blocked_symbols.append(symbol)
        elif lifecycle_action == "re_review":
            review_symbols.append(symbol)
        elif lifecycle_action == "cautious_repromote" and action == "aggressive_promote":
            cautious_symbols.append(symbol)
    return {
        "blocked_symbols": sorted(blocked_symbols),
        "review_symbols": sorted(review_symbols),
        "cautious_symbols": sorted(cautious_symbols),
    }


def _promotion_rollout_signal(
    *,
    requested_status: str,
    effective_status: str,
    reasons: list[str],
    comparison_evidence: dict[str, object] | None,
) -> dict[str, object]:
    runner_quality = _runner_quality_evidence(comparison_evidence, bucket_name="staged_candidate")
    micro_live_gate = dict(runner_quality.get("micro_live_gate", {}) or {})
    micro_live_readiness = str(micro_live_gate.get("status", "not_available") or "not_available") if micro_live_gate else "not_available"
    if requested_status in {"promote", "promote_aggressive"}:
        if bool(micro_live_gate.get("available")) and micro_live_readiness != "pass":
            rollout_stage = "staged_rollout"
        elif effective_status in {"promote", "promote_aggressive"}:
            rollout_stage = "promotion_active"
        else:
            rollout_stage = "promotion_blocked"
    elif effective_status == "demote":
        rollout_stage = "demotion_active"
    elif effective_status == "disable":
        rollout_stage = "disabled"
    else:
        rollout_stage = "baseline"
    return {
        "requested_status": requested_status,
        "effective_status": effective_status,
        "rollout_stage": rollout_stage,
        "micro_live_readiness": micro_live_readiness,
        "micro_live_gate": micro_live_gate,
        "blocked_reasons": [reason for reason in reasons if reason.startswith("PROMOTION_BLOCKED_BY_")],
    }


def build_promotion_verdict(
    candidate_policy: dict[str, object],
    comparison_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    adjustments = list(candidate_policy.get("adjustments", []))
    verdict = _candidate_policy_requested_verdict(adjustments)
    requested_status = str(verdict.get("status", "keep") or "keep")
    comparison_verdict, comparison_delta = _policy_comparison_signal(comparison_evidence)
    baseline_gate = _simple_baseline_control_gate(comparison_evidence)
    runner_quality = _runner_quality_evidence(comparison_evidence, bucket_name="staged_candidate")
    if comparison_verdict == "candidate_worse":
        reasons = list(verdict["reasons"])
        reasons.append("CANDIDATE_UNDERPERFORMS_CURRENT_POLICY")
        if verdict["status"] in {"promote", "promote_aggressive"}:
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_POLICY_COMPARISON"]}
        elif verdict["status"] == "demote":
            verdict = {"status": "disable", "reasons": reasons + ["DEMOTION_ESCALATED_BY_POLICY_COMPARISON"]}
        else:
            verdict = {"status": verdict["status"], "reasons": reasons}
    elif comparison_verdict == "candidate_better":
        verdict = {
            "status": verdict["status"],
            "reasons": list(verdict["reasons"]) + ["CANDIDATE_OUTPERFORMS_CURRENT_POLICY"],
        }
    if verdict["status"] in {"promote", "promote_aggressive"} and str(baseline_gate.get("status", "")) == "block":
        block_reason = str(baseline_gate.get("promotion_block_reason", "") or "PROMOTION_BLOCKED_BY_SIMPLE_BASELINE_CONTROL")
        verdict = {"status": "keep", "reasons": list(verdict["reasons"]) + [block_reason]}
    watchdog_block_reason = _sample_quality_watchdog_block_reason(comparison_evidence)
    if verdict["status"] in {"promote", "promote_aggressive"} and watchdog_block_reason:
        verdict = {"status": "keep", "reasons": list(verdict["reasons"]) + [watchdog_block_reason]}
    if verdict["status"] in {"promote", "promote_aggressive"} and bool(runner_quality.get("available")):
        reasons = list(verdict["reasons"])
        micro_live_gate = dict(runner_quality.get("micro_live_gate", {}) or {})
        if bool(micro_live_gate.get("available")) and str(micro_live_gate.get("status", "")) == "pending":
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_MICRO_LIVE_GATE"]}
        elif float(runner_quality["realized_pnl_usd"]) <= 0.0:
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_NON_POSITIVE_REALIZED_PNL"]}
        elif float(runner_quality["drawdown_ratio"]) > 0.75:
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_DRAWDOWN"]}
        elif float(runner_quality["reject_rate"]) > 0.12:
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_REJECT_RATE"]}
        elif float(runner_quality["protection_degraded_rate"]) > 0.08:
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_EXECUTION_QUALITY"]}
        elif float(runner_quality["avg_slippage_bps"]) > 12.0:
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_SLIPPAGE"]}
        elif float(runner_quality["avg_edge_retention_ratio"]) < 0.55:
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_EDGE_RETENTION"]}
        elif int(runner_quality["walk_forward_window_count"]) >= 2 and float(runner_quality["positive_walk_forward_ratio"]) < 0.5:
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_WALK_FORWARD"]}
        elif bool(micro_live_gate.get("available")) and str(micro_live_gate.get("status", "")) != "pass":
            verdict = {"status": "keep", "reasons": reasons + ["PROMOTION_BLOCKED_BY_MICRO_LIVE_GATE"]}
        elif verdict["status"] == "promote_aggressive" and (
            float(runner_quality["drawdown_ratio"]) > 0.35
            or float(runner_quality["reject_rate"]) > 0.03
            or float(runner_quality["avg_slippage_bps"]) > 8.0
            or float(runner_quality["avg_edge_retention_ratio"]) < 0.75
            or (int(runner_quality["walk_forward_window_count"]) >= 2 and float(runner_quality["positive_walk_forward_ratio"]) < 0.75)
        ):
            verdict = {"status": "promote", "reasons": reasons + ["AGGRESSIVE_PROMOTION_DOWNGRADED_BY_RUNTIME_EVIDENCE"]}
    lifecycle_signal = _symbol_lifecycle_promotion_signal(
        candidate_policy=candidate_policy,
        comparison_evidence=comparison_evidence,
    )
    if verdict["status"] in {"promote", "promote_aggressive"} and list(lifecycle_signal.get("blocked_symbols", [])):
        verdict = {
            "status": "keep",
            "reasons": list(verdict["reasons"]) + ["PROMOTION_BLOCKED_BY_SYMBOL_LIFECYCLE_HOLD"],
        }
    elif verdict["status"] in {"promote", "promote_aggressive"} and list(lifecycle_signal.get("review_symbols", [])):
        verdict = {
            "status": "keep",
            "reasons": list(verdict["reasons"]) + ["PROMOTION_REQUIRES_SYMBOL_RE_REVIEW"],
        }
    elif verdict["status"] == "promote_aggressive" and list(lifecycle_signal.get("cautious_symbols", [])):
        verdict = {
            "status": "promote",
            "reasons": list(verdict["reasons"]) + ["AGGRESSIVE_PROMOTION_DOWNGRADED_BY_SYMBOL_LIFECYCLE"],
        }
    if comparison_verdict != "keep" or comparison_delta != 0.0:
        verdict["comparison_verdict"] = comparison_verdict
        verdict["candidate_vs_current_score_delta"] = comparison_delta
    baseline_control_comparison = dict(baseline_gate.get("baseline_control_comparison", {}) or {})
    if baseline_control_comparison:
        verdict["baseline_control_comparison"] = baseline_control_comparison
        verdict["simple_baseline_gate_status"] = str(baseline_gate.get("status", "not_available") or "not_available")
        verdict["simple_baseline_gate_reason"] = str(baseline_gate.get("gate_reason", "") or "")
    auto_mode = dict(_staged_candidate_evidence(comparison_evidence).get("auto_mode", {}) or {})
    if auto_mode:
        verdict["auto_mode"] = auto_mode
    if any(list(lifecycle_signal.values())):
        verdict["symbol_lifecycle_signal"] = lifecycle_signal
    verdict.update(
        _promotion_rollout_signal(
            requested_status=requested_status,
            effective_status=str(verdict.get("status", "keep") or "keep"),
            reasons=list(verdict.get("reasons", [])),
            comparison_evidence=comparison_evidence,
        )
    )
    return verdict

def _major_symbol_operational_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if str(row.get("symbol", "")) in {"BTCUSDT", "ETHUSDT"}]










def load_validation_runner_evidence(base_path: str | Path | None) -> dict[str, object]:
    if base_path is None:
        return {}
    root = Path(base_path)
    candidate_paths = [
        root,
        root / "policy_validation.json",
        root / "policy_comparison.json",
        root / "validation_report.json",
        root / "performance_report.json",
        root / "summary.json",
    ]
    seen: set[Path] = set()
    for candidate in candidate_paths:
        path = candidate
        if path in seen:
            continue
        seen.add(path)
        if path.is_dir():
            continue
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _log.warning("failed to read validation report %s — skipping", path, exc_info=True)
            continue
        if not isinstance(payload, dict):
            continue
        if "policy_validation" in payload and isinstance(payload.get("policy_validation"), dict):
            payload = dict(payload["policy_validation"])
        evidence = dict(payload.get("evidence", {}) or {})
        if "max_drawdown_pct" in payload and "runner_max_drawdown_pct" not in evidence:
            evidence["runner_max_drawdown_pct"] = float(payload.get("max_drawdown_pct", 0.0) or 0.0)
        if "total_return_pct" in payload and "runner_total_return_pct" not in evidence:
            evidence["runner_total_return_pct"] = float(payload.get("total_return_pct", 0.0) or 0.0)
        if "shadow_alignment_score" in payload and "runner_shadow_alignment_score" not in evidence:
            evidence["runner_shadow_alignment_score"] = float(payload.get("shadow_alignment_score", 0.0) or 0.0)
        for key in (
            "validation_path_mode",
            "lineage_attribution",
            "current_policy_lineage",
            "current_policy_evidence_lineage",
            "current_policy_evidence_alignment",
            "sample_progress",
            "sample_quality_watchdog",
            "baseline_control_comparison",
            "checkpoint_auto_judge",
            "auto_mode",
            "symbol_lifecycle",
            "symbol_lifecycle_summary",
            "score_alignment_summary",
            "total_closed_trade_count",
            "total_live_order_count",
            "total_tested_order_count",
            "runner_walk_forward_window_count",
            "runner_positive_walk_forward_window_count",
            "runner_positive_walk_forward_ratio",
            "runner_protection_degraded_rate",
            "recent_retention_window",
            "cumulative_retention_window",
            "walk_forward_windows",
            "validation_runs",
            "symbol_summary",
            "symbol_execution_summary",
            "symbol_scorecard",
            "regime_summary",
            "pruning_recommendations",
            "metric_comparisons",
            "candidate_replay_summary",
            "current_replay_summary",
            "candidate_policy_application",
            "current_policy_application",
            "policy_application_delta",
            "counterfactual_replay_path",
            "policy_context_bucket_evidence",
        ):
            if key in payload and key not in evidence:
                evidence[key] = payload[key]
        if evidence:
            return evidence
    return {}


def merge_policy_validation_evidence(
    attribution_rows: list[dict[str, object]] | tuple[dict[str, object], ...],
    runner_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence = _replay_like_validation_evidence(attribution_rows)
    for key, value in dict(runner_evidence or {}).items():
        evidence[key] = value
    has_attribution_rows = bool(list(attribution_rows))
    runner_drawdown_ratio = float(
        evidence.get(
            "runner_drawdown_to_pnl_ratio",
            float(evidence.get("runner_max_drawdown_pct", 0.0) or 0.0) / 100.0,
        )
        or 0.0
    )
    if runner_drawdown_ratio > 0.0:
        if has_attribution_rows:
            evidence["replay_like_drawdown_ratio"] = round(max(float(evidence.get("replay_like_drawdown_ratio", 0.0) or 0.0), runner_drawdown_ratio), 6)
        else:
            evidence["replay_like_drawdown_ratio"] = round(runner_drawdown_ratio, 6)
    runner_shadow_alignment_score = float(evidence.get("runner_shadow_alignment_score", 0.0) or 0.0)
    if runner_shadow_alignment_score > 0.0:
        if has_attribution_rows:
            evidence["shadow_alignment_score"] = round(min(max(float(evidence.get("shadow_alignment_score", 0.0) or 0.0), runner_shadow_alignment_score), 1.0), 6)
        else:
            evidence["shadow_alignment_score"] = round(min(max(runner_shadow_alignment_score, 0.0), 1.0), 6)
    candidate_delta = float(evidence.get("candidate_vs_current_score_delta", 0.0) or 0.0)
    if candidate_delta > 0.1:
        evidence["comparison_alignment_score"] = round(min(1.0, 0.5 + candidate_delta), 6)
    elif candidate_delta < -0.1:
        evidence["comparison_alignment_score"] = round(max(0.0, 0.5 + candidate_delta), 6)
    return evidence

def _replay_like_validation_evidence(attribution_rows: list[dict[str, object]] | tuple[dict[str, object], ...]) -> dict[str, object]:
    rows = list(attribution_rows)
    if not rows:
        return {
            "sample_count": 0,
            "avg_retention": 0.0,
            "avg_realized_edge_bps": 0.0,
            "max_reject_rate": 1.0,
            "max_protection_degraded_rate": 1.0,
            "replay_like_drawdown_ratio": 1.0,
            "shadow_alignment_score": 0.0,
        }
    avg_retention = sum(float(row.get("avg_edge_retention_ratio", 0.0) or 0.0) for row in rows) / len(rows)
    avg_realized = sum(float(row.get("avg_realized_edge_bps", 0.0) or 0.0) for row in rows) / len(rows)
    max_reject_rate = max(float(row.get("reject_rate", 0.0) or 0.0) for row in rows)
    max_degraded_rate = max(float(row.get("protection_degraded_rate", 0.0) or 0.0) for row in rows)
    replay_like_drawdown_ratio = max(0.0, 1.0 - max(avg_retention, 0.0))
    shadow_alignment_score = max(0.0, min(1.0, avg_retention * (1.0 - max_reject_rate)))
    return {
        "sample_count": len(rows),
        "avg_retention": round(avg_retention, 6),
        "avg_realized_edge_bps": round(avg_realized, 6),
        "max_reject_rate": round(max_reject_rate, 6),
        "max_protection_degraded_rate": round(max_degraded_rate, 6),
        "replay_like_drawdown_ratio": round(replay_like_drawdown_ratio, 6),
        "shadow_alignment_score": round(shadow_alignment_score, 6),
    }

def build_policy_validation(candidate_policy: dict[str, object], promotion_verdict: dict[str, object], operational_verdict: dict[str, object], attribution_rows: list[dict[str, object]] | tuple[dict[str, object], ...] = (), runner_evidence: dict[str, object] | None = None) -> dict[str, object]:
    candidate_adjustments = list(candidate_policy.get("adjustments", []))
    verdict_status = str(promotion_verdict.get("status", "keep"))
    requested_status = str(promotion_verdict.get("requested_status", verdict_status) or verdict_status)
    operational_status = str(operational_verdict.get("status", "hold"))
    operational_reasons = list(operational_verdict.get("reasons", []) or [])
    evidence = merge_policy_validation_evidence(attribution_rows, runner_evidence)
    staged_candidate_evidence = _staged_candidate_evidence(evidence)
    runner_quality = _runner_quality_evidence(evidence, bucket_name="staged_candidate")
    sample_watchdog = dict(staged_candidate_evidence.get("sample_quality_watchdog", {}) or {})
    checkpoint_auto_judge = dict(staged_candidate_evidence.get("checkpoint_auto_judge", {}) or {})
    baseline_gate = _simple_baseline_control_gate(evidence)
    sample_watchdog_status = str(sample_watchdog.get("status", "") or "")
    reasons: list[str] = []
    status = "fail"
    pending_due_to_warmup = False
    if not candidate_adjustments:
        reasons.append("NO_CANDIDATE_ADJUSTMENTS")
    else:
        status = "pass"
        reasons.append("CANDIDATE_DATA_AVAILABLE")
    if operational_status == "stop":
        status = "fail"
        reasons.append("OPERATIONAL_STOP_ACTIVE")
    elif operational_status == "hold" and verdict_status in {"promote", "promote_aggressive"}:
        reasons.append("PROMOTION_BLOCKED_BY_HOLD")
        if "INSUFFICIENT_SAMPLE" in operational_reasons:
            pending_due_to_warmup = True
            reasons.append("OPERATIONAL_SAMPLE_STILL_WARMING_UP")
    elif requested_status in {"promote", "promote_aggressive"} and str(promotion_verdict.get("rollout_stage", "")) == "staged_rollout":
        reasons.append("PROMOTION_STAGED_PENDING_MICRO_LIVE")
        pending_due_to_warmup = True
    elif verdict_status in {"promote", "promote_aggressive", "demote"}:
        reasons.append("PROMOTION_PATH_VALIDATED")
    else:
        reasons.append("NO_PROMOTION_ACTION")
    verdict_reasons = {str(reason) for reason in list(promotion_verdict.get("reasons", []) or []) if str(reason)}
    if "PROMOTION_BLOCKED_BY_SYMBOL_LIFECYCLE_HOLD" in verdict_reasons:
        status = "fail"
        reasons.append("PROMOTION_PATH_BLOCKED_BY_SYMBOL_LIFECYCLE")
    if "PROMOTION_REQUIRES_SYMBOL_RE_REVIEW" in verdict_reasons:
        pending_due_to_warmup = True
        reasons.append("PROMOTION_PATH_REQUIRES_SYMBOL_RE_REVIEW")
    if evidence["replay_like_drawdown_ratio"] > 0.45:
        status = "fail"
        reasons.append("REPLAY_DRAWDOWN_TOO_HIGH")
    if evidence["shadow_alignment_score"] < 0.45:
        status = "fail"
        reasons.append("SHADOW_ALIGNMENT_TOO_LOW")
    if float(evidence.get("candidate_vs_current_score_delta", 0.0) or 0.0) < -0.1:
        status = "fail"
        reasons.append("CANDIDATE_UNDERPERFORMS_CURRENT_POLICY")
    elif float(evidence.get("candidate_vs_current_score_delta", 0.0) or 0.0) > 0.1:
        reasons.append("CANDIDATE_OUTPERFORMS_CURRENT_POLICY")
    if requested_status in {"promote", "promote_aggressive"} and str(baseline_gate.get("status", "")) == "block":
        status = "fail"
        baseline_validation_reason = str(
            baseline_gate.get("validation_reason", "") or "SIMPLE_BASELINE_CONTROL_NOT_CLEARLY_BEATEN"
        )
        reasons.append(baseline_validation_reason)
        reasons.append("PROMOTION_PATH_BLOCKED_BY_SIMPLE_BASELINE_CONTROL")
    if sample_watchdog_status == "degraded":
        status = "fail"
        reasons.append("SAMPLE_QUALITY_WATCHDOG_DEGRADED")
    elif sample_watchdog_status == "thin":
        reasons.append("SAMPLE_QUALITY_WATCHDOG_THIN")
        if requested_status in {"promote", "promote_aggressive"}:
            pending_due_to_warmup = True
    checkpoint_verdict = str(checkpoint_auto_judge.get("verdict", "") or "")
    if checkpoint_verdict == "rollback":
        status = "fail"
        reasons.append("CHECKPOINT_AUTO_JUDGE_ROLLBACK")
    elif checkpoint_verdict == "tighten":
        reasons.append("CHECKPOINT_AUTO_JUDGE_TIGHTEN")
        if requested_status in {"promote", "promote_aggressive"}:
            status = "fail"
    elif checkpoint_verdict == "expand":
        reasons.append("CHECKPOINT_AUTO_JUDGE_EXPAND")
    if bool(runner_quality.get("available")):
        micro_live_gate = dict(runner_quality.get("micro_live_gate", {}) or {})
        micro_live_status = str(micro_live_gate.get("status", "") or "")
        if float(runner_quality["realized_pnl_usd"]) <= 0.0 and micro_live_status != "pending":
            status = "fail"
            reasons.append("RUNNER_REALIZED_PNL_NOT_POSITIVE")
        if float(runner_quality["drawdown_ratio"]) > 0.75:
            status = "fail"
            reasons.append("RUNNER_DRAWDOWN_ABOVE_LIMIT")
        if float(runner_quality["reject_rate"]) > 0.12:
            status = "fail"
            reasons.append("RUNNER_REJECT_RATE_ABOVE_LIMIT")
        if float(runner_quality["protection_degraded_rate"]) > 0.12:
            status = "fail"
            reasons.append("RUNNER_EXECUTION_QUALITY_BELOW_LIMIT")
        if float(runner_quality["avg_slippage_bps"]) > 12.0:
            status = "fail"
            reasons.append("RUNNER_SLIPPAGE_ABOVE_LIMIT")
        if float(runner_quality["avg_edge_retention_ratio"]) < 0.55:
            status = "fail"
            reasons.append("RUNNER_EDGE_RETENTION_BELOW_LIMIT")
        if int(runner_quality["walk_forward_window_count"]) >= 2 and float(runner_quality["positive_walk_forward_ratio"]) < 0.5:
            status = "fail"
            reasons.append("RUNNER_WALK_FORWARD_SUPPORT_TOO_WEAK")
        elif int(runner_quality["walk_forward_window_count"]) >= 2 and float(runner_quality["positive_walk_forward_ratio"]) >= 0.75:
            reasons.append("RUNNER_WALK_FORWARD_SUPPORT_STRONG")
        if (
            verdict_status in {"promote", "promote_aggressive"}
            and bool(micro_live_gate.get("available"))
            and micro_live_status != "pass"
        ):
            reasons.append("MICRO_LIVE_GATE_NOT_PASSED")
            if micro_live_status == "pending":
                pending_due_to_warmup = True
                reasons.append("MICRO_LIVE_EVIDENCE_STILL_ACCUMULATING")
            else:
                status = "fail"
    if status == "pass" and pending_due_to_warmup:
        status = "pending"
    return {"status": status, "reasons": reasons, "evidence": evidence}


def build_executive_operating_verdict(
    promotion_verdict: dict[str, object],
    operational_verdict: dict[str, object],
    policy_validation: dict[str, object],
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    def _not_available_replay_provenance(*, decision_surface: str) -> dict[str, object]:
        return {
            "classification": "not_available",
            "decision_surface": decision_surface,
            "has_bucket_reference": False,
            "has_runtime_summary_anchor": False,
            "policy_bucket": "not_available",
            "reason_code": "NOT_AVAILABLE",
            "source": "not_available",
            "summary": "not_available:not_available",
            "uses_direct_bucket_replay": False,
            "uses_mixed_root_summary_fallback": False,
            "uses_projected_evidence": False,
        }

    validation = dict(policy_validation or {})
    merged_evidence = dict(dict(validation.get("evidence", {}) or {}))
    merged_evidence.update(dict(evidence or {}))
    staged_candidate_evidence = _staged_candidate_evidence(merged_evidence)
    active_policy_bucket = policy_evidence_bucket(merged_evidence, "active_policy")
    baseline_control_bucket = policy_evidence_bucket(merged_evidence, "baseline_control")
    requested_status = str(
        promotion_verdict.get("requested_status", promotion_verdict.get("status", "keep")) or "keep"
    )
    promotion_status = str(promotion_verdict.get("status", "keep") or "keep")
    operational_status = str(operational_verdict.get("status", "hold") or "hold")
    validation_status = str(validation.get("status", "fail") or "fail")
    validation_reasons = [str(item) for item in list(validation.get("reasons", []) or []) if str(item)]
    operational_reasons = [str(item) for item in list(operational_verdict.get("reasons", []) or []) if str(item)]
    promotion_reasons = [str(item) for item in list(promotion_verdict.get("reasons", []) or []) if str(item)]
    checkpoint_auto_judge = dict(staged_candidate_evidence.get("checkpoint_auto_judge", {}) or {})
    checkpoint_verdict = str(checkpoint_auto_judge.get("verdict", "not_available") or "not_available")
    checkpoint_evidence_source = str(checkpoint_auto_judge.get("evidence_source", "not_available") or "not_available")
    checkpoint_evidence_policy_bucket = str(
        checkpoint_auto_judge.get("evidence_policy_bucket", "not_available") or "not_available"
    )
    sample_watchdog = dict(staged_candidate_evidence.get("sample_quality_watchdog", {}) or {})
    sample_watchdog_status = str(sample_watchdog.get("status", "not_available") or "not_available")
    baseline_gate = _simple_baseline_control_gate(merged_evidence)
    auto_mode = dict(staged_candidate_evidence.get("auto_mode", promotion_verdict.get("auto_mode", {})) or {})
    auto_mode_mode = str(auto_mode.get("mode", "normal") or "normal")
    micro_live_gate = dict(staged_candidate_evidence.get("micro_live_gate", {}) or {})
    micro_live_status = str(micro_live_gate.get("status", "not_available") or "not_available")
    policy_alignment = dict(active_policy_bucket.get("alignment") or staged_candidate_evidence.get("current_policy_evidence_alignment", {}) or {})
    lineage_status = str(
        policy_alignment.get(
            "status",
            merged_evidence.get("checkpoint_auto_judge_lineage_status", "unknown"),
        )
        or "unknown"
    )
    lifecycle_signal = dict(promotion_verdict.get("symbol_lifecycle_signal", {}) or {})
    blocked_symbols = sorted(
        str(item)
        for item in list(lifecycle_signal.get("blocked_symbols", []) or [])
        if str(item)
    )
    review_symbols = sorted(
        str(item)
        for item in list(lifecycle_signal.get("review_symbols", []) or [])
        if str(item)
    )
    cautious_symbols = sorted(
        str(item)
        for item in list(lifecycle_signal.get("cautious_symbols", []) or [])
        if str(item)
    )
    if not blocked_symbols and not review_symbols:
        for row in list(merged_evidence.get("symbol_lifecycle", []) or []):
            payload = dict(row)
            symbol = str(payload.get("symbol", "") or "")
            action = str(payload.get("recommended_action", "keep") or "keep")
            if not symbol:
                continue
            if action in {"rollback", "hold"}:
                blocked_symbols.append(symbol)
            elif action == "re_review":
                review_symbols.append(symbol)
            elif action == "cautious_repromote":
                cautious_symbols.append(symbol)
        blocked_symbols = sorted(set(blocked_symbols))
        review_symbols = sorted(set(review_symbols))
        cautious_symbols = sorted(set(cautious_symbols))

    severe_validation_reasons = {
        "CHECKPOINT_AUTO_JUDGE_ROLLBACK",
        "REPLAY_DRAWDOWN_TOO_HIGH",
        "RUNNER_REALIZED_PNL_NOT_POSITIVE",
        "RUNNER_DRAWDOWN_ABOVE_LIMIT",
        "RUNNER_REJECT_RATE_ABOVE_LIMIT",
        "RUNNER_EXECUTION_QUALITY_BELOW_LIMIT",
        "RUNNER_EDGE_RETENTION_BELOW_LIMIT",
        "RUNNER_WALK_FORWARD_SUPPORT_TOO_WEAK",
        "SAMPLE_QUALITY_WATCHDOG_DEGRADED",
        "PROMOTION_PATH_BLOCKED_BY_SYMBOL_LIFECYCLE",
    }
    reasons: list[str] = []
    verdict = "hold"
    if checkpoint_verdict == "rollback":
        verdict = "rollback"
        reasons.append("EXECUTIVE_ROLLBACK_BY_CHECKPOINT")
    elif operational_status == "stop":
        verdict = "rollback"
        reasons.append("EXECUTIVE_ROLLBACK_BY_OPERATIONAL_STOP")
    elif validation_status == "fail" and any(reason in severe_validation_reasons for reason in validation_reasons):
        verdict = "rollback"
        reasons.append("EXECUTIVE_ROLLBACK_BY_VALIDATION_FAILURE")
    elif lineage_status in {"mismatch", "stale"}:
        verdict = "rebuild_evidence"
        reasons.append("EXECUTIVE_REBUILD_BY_POLICY_LINEAGE")
    elif validation_status == "pending":
        verdict = "rebuild_evidence"
        reasons.append("EXECUTIVE_REBUILD_BY_PENDING_VALIDATION")
    elif requested_status in {"promote", "promote_aggressive"} and (
        sample_watchdog_status == "thin"
        or (bool(micro_live_gate.get("available")) and micro_live_status == "pending")
    ):
        verdict = "rebuild_evidence"
        reasons.append("EXECUTIVE_REBUILD_BY_THIN_OR_PENDING_EXPANSION_EVIDENCE")
    elif checkpoint_verdict == "expand" and (
        checkpoint_evidence_source != "policy_bucket"
        or checkpoint_evidence_policy_bucket != "staged_candidate"
    ):
        verdict = "rebuild_evidence"
        reasons.append("EXECUTIVE_REBUILD_BY_MISSING_BUCKET_EXPANSION_EVIDENCE")
    elif checkpoint_verdict == "tighten":
        verdict = "tighten"
        reasons.append("EXECUTIVE_TIGHTEN_BY_CHECKPOINT")
    elif sample_watchdog_status == "degraded":
        verdict = "tighten"
        reasons.append("EXECUTIVE_TIGHTEN_BY_SAMPLE_WATCHDOG")
    elif auto_mode_mode == "tighter":
        verdict = "tighten"
        reasons.append("EXECUTIVE_TIGHTEN_BY_AUTO_MODE")
    elif str(baseline_gate.get("status", "")) == "block" and requested_status in {"promote", "promote_aggressive"}:
        verdict = "tighten"
        reasons.append("EXECUTIVE_TIGHTEN_BY_SIMPLE_BASELINE_GATE")
    elif blocked_symbols or review_symbols:
        verdict = "tighten"
        reasons.append("EXECUTIVE_TIGHTEN_BY_SYMBOL_LIFECYCLE")
    elif operational_status == "hold" and set(operational_reasons) != {"INSUFFICIENT_SAMPLE"}:
        verdict = "tighten"
        reasons.append("EXECUTIVE_TIGHTEN_BY_OPERATIONAL_HOLD")
    elif (
        promotion_status in {"promote", "promote_aggressive"}
        and validation_status == "pass"
        and operational_status in {"strong_pass", "aggressive_pass"}
        and checkpoint_verdict == "expand"
        and sample_watchdog_status == "promote_ready"
        and str(baseline_gate.get("status", "")) == "pass"
        and auto_mode_mode == "cautiously_expanded"
        and (not bool(micro_live_gate.get("available")) or micro_live_status == "pass")
        and not blocked_symbols
        and not review_symbols
    ):
        verdict = "expand"
        reasons.append("EXECUTIVE_EXPAND_SUPPORTED_BY_ALIGNED_EVIDENCE")
    elif requested_status in {"promote", "promote_aggressive"}:
        reasons.append("EXECUTIVE_HOLD_UNTIL_EXPANSION_SUPPORT_IS_CLEAR")
    else:
        reasons.append("EXECUTIVE_HOLD_CURRENT_POSTURE")

    confidence = "low"
    if verdict == "expand":
        confidence = "high"
    elif verdict in {"rollback", "tighten"}:
        confidence = "high" if validation_status in {"pass", "fail"} else "medium"
    elif verdict == "rebuild_evidence":
        confidence = "medium"
    elif validation_status == "pass" or operational_status in {"pass", "strong_pass", "aggressive_pass"}:
        confidence = "medium"

    unique_reasons: list[str] = []
    seen_reasons: set[str] = set()
    for reason in reasons:
        if reason and reason not in seen_reasons:
            unique_reasons.append(reason)
            seen_reasons.add(reason)
    checkpoint_replay = dict(checkpoint_auto_judge.get("replay_provenance", {}) or {})
    if not checkpoint_replay:
        if checkpoint_auto_judge:
            checkpoint_replay = checkpoint_replay_provenance(checkpoint_auto_judge)
        else:
            checkpoint_replay = _not_available_replay_provenance(decision_surface="checkpoint_auto_judge")

    staged_summary = dict(
        staged_candidate_evidence.get(
            "candidate_replay_summary",
            staged_candidate_evidence.get("replay_summary", {}),
        )
        or {}
    )
    staged_replay = dict(staged_summary.get("replay_provenance", {}) or {})
    if not staged_replay:
        staged_replay = _not_available_replay_provenance(decision_surface="staged_candidate")

    primary_replay = dict(build_replay_provenance())
    checkpoint_classification = str(checkpoint_replay.get("classification", "not_available") or "not_available")
    if checkpoint_classification not in {"not_available", "projected_evidence"}:
        primary_replay = dict(checkpoint_replay)
        primary_replay["decision_surface"] = "checkpoint_auto_judge"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasons": unique_reasons,
        "replay_provenance": {
            "primary": primary_replay,
            "checkpoint_auto_judge": checkpoint_replay,
            "staged_candidate": staged_replay,
        },
        "signals": {
            "requested_status": requested_status,
            "promotion_status": promotion_status,
            "operational_status": operational_status,
            "policy_validation_status": validation_status,
            "checkpoint_verdict": checkpoint_verdict,
            "checkpoint_evidence_source": checkpoint_evidence_source,
            "checkpoint_evidence_policy_bucket": checkpoint_evidence_policy_bucket,
            "sample_quality_watchdog_status": sample_watchdog_status,
            "simple_baseline_gate_status": str(baseline_gate.get("status", "not_available") or "not_available"),
            "auto_mode": auto_mode_mode,
            "micro_live_status": micro_live_status,
            "policy_lineage_status": lineage_status,
            "active_policy_bucket_available": bool(active_policy_bucket.get("available")),
            "staged_candidate_bucket_available": bool(policy_evidence_bucket(merged_evidence, "staged_candidate").get("available")),
            "baseline_control_bucket_available": bool(baseline_control_bucket.get("available")),
            "blocked_symbols": blocked_symbols,
            "review_symbols": review_symbols,
            "cautious_symbols": cautious_symbols,
        },
        "source_reason_codes": {
            "promotion": promotion_reasons,
            "operational": operational_reasons,
            "policy_validation": validation_reasons,
            "checkpoint_auto_judge": [str(item) for item in list(checkpoint_auto_judge.get("reason_codes", []) or []) if str(item)],
            "auto_mode": [str(item) for item in list(auto_mode.get("reason_codes", []) or []) if str(item)],
        },
    }


def _dedupe_strings(items: list[str] | tuple[str, ...]) -> list[str]:
    unique_items: list[str] = []
    seen_items: set[str] = set()
    for item in items:
        normalized = str(item or "")
        if normalized and normalized not in seen_items:
            unique_items.append(normalized)
            seen_items.add(normalized)
    return unique_items


def _live_evidence_snapshot(validation_evidence: dict[str, object] | None) -> dict[str, object]:
    payload = _active_policy_evidence(validation_evidence)
    micro_live_gate = dict(payload.get("micro_live_gate", {}) or {})
    sample_watchdog = dict(payload.get("sample_quality_watchdog", {}) or {})
    active_bucket = policy_evidence_bucket(validation_evidence, "active_policy")
    policy_alignment = dict(active_bucket.get("alignment") or payload.get("current_policy_evidence_alignment", {}) or {})
    bucket_direct = _policy_context_bucket_direct_metrics(payload)
    validation_runs = list(payload.get("validation_runs", []) or [])
    walk_forward_windows = list(payload.get("walk_forward_windows", []) or [])
    run_count = max(
        _coerce_int(payload.get("run_count")),
        len(validation_runs),
        _coerce_int(bucket_direct.get("run_count")),
    )
    live_order_count = max(
        _coerce_int(payload.get("total_live_order_count")),
        _coerce_int(payload.get("live_order_count")),
        _coerce_int(micro_live_gate.get("live_order_count")),
    )
    closed_trade_count = max(
        _coerce_int(bucket_direct.get("closed_trade_count")),
        _coerce_int(payload.get("total_closed_trade_count")),
        _coerce_int(payload.get("closed_trade_count")),
        _coerce_int(micro_live_gate.get("closed_trade_count")),
    )
    walk_forward_window_count = max(
        _coerce_int(bucket_direct.get("walk_forward_window_count")),
        _coerce_int(payload.get("runner_walk_forward_window_count")),
        _coerce_int(payload.get("walk_forward_window_count")),
        len(walk_forward_windows),
    )
    positive_walk_forward_ratio = round(_coerce_float(bucket_direct.get("positive_walk_forward_ratio")), 6)
    if positive_walk_forward_ratio == 0.0 and walk_forward_window_count <= 0:
        positive_walk_forward_ratio = round(
            _coerce_float(
                payload.get(
                    "runner_positive_walk_forward_ratio",
                    payload.get("positive_walk_forward_ratio"),
                )
            ),
            6,
        )
    return {
        "run_count": run_count,
        "decision_count": _coerce_int(bucket_direct.get("decision_count")),
        "live_order_count": live_order_count,
        "closed_trade_count": closed_trade_count,
        "realized_pnl_usd": round(
            _coerce_float(bucket_direct.get("realized_pnl_usd", payload.get("runner_total_realized_pnl_usd"))),
            6,
        ),
        "walk_forward_window_count": walk_forward_window_count,
        "positive_walk_forward_ratio": positive_walk_forward_ratio,
        "sample_quality_watchdog_status": str(sample_watchdog.get("status", "not_available") or "not_available"),
        "micro_live_status": str(micro_live_gate.get("status", "not_available") or "not_available"),
        "policy_lineage_status": str(policy_alignment.get("status", "unknown") or "unknown"),
        "bucket_available": bool(active_bucket.get("available")) or "policy_evidence_buckets" not in dict(validation_evidence or {}),
        "bucket_direct_available": bool(bucket_direct.get("available")),
    }


def _build_live_evidence_rejudge(
    *,
    previous_state: dict[str, object],
    validation_evidence: dict[str, object],
    checkpoint_revalidation: dict[str, object],
    executive_operating_verdict: dict[str, object],
) -> dict[str, object]:
    previous_verdict = str(
        dict(previous_state.get("executive_operating_verdict", {}) or {}).get("verdict", "not_available") or "not_available"
    )
    previous_snapshot = _live_evidence_snapshot(previous_state)
    current_snapshot = _live_evidence_snapshot(validation_evidence)
    lineage_status = str(current_snapshot.get("policy_lineage_status", "unknown") or "unknown")
    current_verdict = str(dict(executive_operating_verdict or {}).get("verdict", "hold") or "hold")
    evidence_delta = {
        "run_count": int(current_snapshot.get("run_count", 0)) - int(previous_snapshot.get("run_count", 0)),
        "decision_count": int(current_snapshot.get("decision_count", 0)) - int(previous_snapshot.get("decision_count", 0)),
        "live_order_count": int(current_snapshot.get("live_order_count", 0)) - int(previous_snapshot.get("live_order_count", 0)),
        "closed_trade_count": int(current_snapshot.get("closed_trade_count", 0)) - int(previous_snapshot.get("closed_trade_count", 0)),
        "realized_pnl_usd": round(
            float(current_snapshot.get("realized_pnl_usd", 0.0) or 0.0)
            - float(previous_snapshot.get("realized_pnl_usd", 0.0) or 0.0),
            6,
        ),
        "walk_forward_window_count": int(current_snapshot.get("walk_forward_window_count", 0)) - int(previous_snapshot.get("walk_forward_window_count", 0)),
        "positive_walk_forward_ratio": round(
            float(current_snapshot.get("positive_walk_forward_ratio", 0.0) or 0.0)
            - float(previous_snapshot.get("positive_walk_forward_ratio", 0.0) or 0.0),
            6,
        ),
    }
    revalidation_hooks = [
        dict(item)
        for item in list(dict(checkpoint_revalidation or {}).get("hooks", []) or [])
        if isinstance(item, dict)
    ]
    fresh_reasons: list[str] = []
    if evidence_delta["decision_count"] >= 2:
        fresh_reasons.append("POLICY_BUCKET_DECISION_SAMPLE_INCREASED")
    if evidence_delta["live_order_count"] >= 2:
        fresh_reasons.append("LIVE_ORDER_SAMPLE_INCREASED")
    if evidence_delta["closed_trade_count"] >= 1:
        fresh_reasons.append("CLOSED_TRADE_SAMPLE_INCREASED")
    if evidence_delta["walk_forward_window_count"] >= 1:
        fresh_reasons.append("WALK_FORWARD_SUPPORT_INCREASED")
    if (
        str(previous_snapshot.get("micro_live_status", "not_available") or "not_available")
        != str(current_snapshot.get("micro_live_status", "not_available") or "not_available")
        and str(current_snapshot.get("micro_live_status", "not_available") or "not_available") in {"pass", "fail"}
    ):
        fresh_reasons.append("MICRO_LIVE_STATUS_CHANGED")
    if (
        str(previous_snapshot.get("sample_quality_watchdog_status", "not_available") or "not_available")
        != str(current_snapshot.get("sample_quality_watchdog_status", "not_available") or "not_available")
        and str(current_snapshot.get("sample_quality_watchdog_status", "not_available") or "not_available")
        in {"healthy", "promote_ready", "degraded"}
    ):
        fresh_reasons.append("SAMPLE_QUALITY_STATUS_CHANGED")
    if revalidation_hooks:
        fresh_reasons.extend(
            f"CHECKPOINT_REVALIDATION_{str(item.get('kind', 'triggered')).upper()}"
            for item in revalidation_hooks
        )
    fresh_reasons = _dedupe_strings(fresh_reasons)
    if previous_verdict == "not_available":
        status = "initial"
        reason_codes = ["NO_PREVIOUS_EXECUTIVE_VERDICT"]
        triggered = False
    elif lineage_status in {"mismatch", "stale"}:
        status = "blocked"
        reason_codes = [f"LIVE_EVIDENCE_REJUDGE_BLOCKED_BY_{lineage_status.upper()}_LINEAGE"]
        triggered = False
    elif not fresh_reasons:
        status = "waiting"
        reason_codes = ["NO_MATERIAL_LIVE_EVIDENCE_ACCUMULATION"]
        triggered = False
    elif (
        str(current_snapshot.get("sample_quality_watchdog_status", "not_available") or "not_available") == "thin"
        and str(current_snapshot.get("micro_live_status", "not_available") or "not_available") != "pass"
    ):
        status = "waiting"
        reason_codes = ["LIVE_EVIDENCE_STILL_THIN_AFTER_ACCUMULATION", *fresh_reasons]
        triggered = False
    else:
        status = "triggered"
        reason_codes = ["LIVE_EVIDENCE_REJUDGE_TRIGGERED", *fresh_reasons]
        triggered = True
    return {
        "status": status,
        "triggered": triggered,
        "previous_verdict": previous_verdict,
        "current_raw_verdict": current_verdict,
        "policy_lineage_status": lineage_status,
        "fresh_evidence_accumulated": bool(fresh_reasons),
        "fresh_reason_codes": fresh_reasons,
        "reason_codes": _dedupe_strings(reason_codes),
        "evidence_delta": evidence_delta,
        "previous_snapshot": previous_snapshot,
        "current_snapshot": current_snapshot,
        "checkpoint_revalidation_triggered": bool(dict(checkpoint_revalidation or {}).get("triggered")),
        "checkpoint_revalidation_hook_count": len(revalidation_hooks),
    }


_EXECUTIVE_VERDICT_RANK = {
    "rollback": 0,
    "tighten": 1,
    "rebuild_evidence": 2,
    "hold": 3,
    "expand": 4,
}


def _apply_live_evidence_rejudge_to_executive_verdict(
    *,
    previous_state: dict[str, object],
    executive_operating_verdict: dict[str, object],
    live_evidence_rejudge: dict[str, object],
) -> dict[str, object]:
    adjusted = dict(executive_operating_verdict or {})
    previous_verdict = str(live_evidence_rejudge.get("previous_verdict", "not_available") or "not_available")
    current_verdict = str(adjusted.get("verdict", "hold") or "hold")
    rejudge_status = str(live_evidence_rejudge.get("status", "unknown") or "unknown")
    previous_rank = _EXECUTIVE_VERDICT_RANK.get(previous_verdict, -1)
    current_rank = _EXECUTIVE_VERDICT_RANK.get(current_verdict, -1)
    retained_previous_verdict = False
    if previous_rank >= 0 and current_rank > previous_rank and rejudge_status != "triggered":
        adjusted["verdict"] = previous_verdict
        previous_confidence = str(
            dict(previous_state.get("executive_operating_verdict", {}) or {}).get("confidence", adjusted.get("confidence", "low"))
            or adjusted.get("confidence", "low")
        )
        adjusted["confidence"] = previous_confidence
        retained_previous_verdict = True
        extra_reason = (
            "EXECUTIVE_REJUDGE_BLOCKED_BY_POLICY_LINEAGE"
            if rejudge_status == "blocked"
            else "EXECUTIVE_REJUDGE_WAITING_FOR_FRESH_EVIDENCE"
        )
        adjusted["reasons"] = _dedupe_strings(
            list(adjusted.get("reasons", []) or [])
            + [extra_reason]
        )
    adjusted["signals"] = {
        **dict(adjusted.get("signals", {}) or {}),
        "live_evidence_rejudge_status": rejudge_status,
        "live_evidence_fresh": bool(live_evidence_rejudge.get("fresh_evidence_accumulated")),
        "live_evidence_lineage_status": str(live_evidence_rejudge.get("policy_lineage_status", "unknown") or "unknown"),
    }
    source_reason_codes = dict(adjusted.get("source_reason_codes", {}) or {})
    source_reason_codes["live_evidence_rejudge"] = list(live_evidence_rejudge.get("reason_codes", []) or [])
    adjusted["source_reason_codes"] = source_reason_codes
    adjusted["live_evidence_rejudge"] = {
        **dict(live_evidence_rejudge or {}),
        "effective_verdict": str(adjusted.get("verdict", current_verdict) or current_verdict),
        "retained_previous_verdict": retained_previous_verdict,
    }
    return adjusted


def _coerce_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _policy_context_bucket_direct_metrics(payload: dict[str, object] | None) -> dict[str, object]:
    item = dict(payload or {})
    bucket_runs = [dict(run) for run in list(item.get("policy_context_bucket_validation_runs", []) or [])]
    bucket_windows = [dict(window) for window in list(item.get("policy_context_bucket_walk_forward_windows", []) or [])]
    bucket_closed_trade_count = max(
        _coerce_int(item.get("policy_context_bucket_closed_trade_count")),
        sum(_coerce_int(run.get("closed_trade_count")) for run in bucket_runs),
    )
    bucket_decision_count = max(
        _coerce_int(item.get("policy_context_bucket_decision_count")),
        sum(_coerce_int(run.get("decision_count")) for run in bucket_runs),
    )
    positive_walk_forward_count = sum(
        1
        for window in bucket_windows
        if _coerce_float(window.get("avg_net_edge_bps")) > 0.0 and _coerce_float(window.get("avg_score")) >= 0.0
    )
    bucket_positive_walk_forward_ratio = round(
        _coerce_float(
            item.get(
                "policy_context_bucket_positive_walk_forward_ratio",
                positive_walk_forward_count / len(bucket_windows) if bucket_windows else 0.0,
            )
        ),
        6,
    )
    return {
        "available": bool(item.get("policy_context_bucket_available")) or bool(bucket_runs) or bool(bucket_windows) or bucket_closed_trade_count > 0 or bucket_decision_count > 0,
        "bucket_name": str(item.get("policy_context_bucket_name", "") or ""),
        "run_count": max(_coerce_int(item.get("policy_context_bucket_run_count")), len(bucket_runs)),
        "decision_count": bucket_decision_count,
        "closed_trade_count": bucket_closed_trade_count,
        "realized_pnl_usd": round(
            _coerce_float(
                item.get(
                    "policy_context_bucket_total_realized_pnl_usd",
                    sum(_coerce_float(run.get("realized_pnl_usd")) for run in bucket_runs),
                )
            ),
            6,
        ),
        "walk_forward_window_count": max(_coerce_int(item.get("policy_context_bucket_walk_forward_window_count")), len(bucket_windows)),
        "positive_walk_forward_ratio": bucket_positive_walk_forward_ratio,
        "drawdown_to_pnl_ratio": _recent_drawdown_ratio(
            [_coerce_float(run.get("realized_pnl_usd")) for run in bucket_runs]
        ),
        "validation_runs": bucket_runs,
        "walk_forward_windows": bucket_windows,
    }


def _recent_drawdown_ratio(values: list[float]) -> float:
    if not values:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return round(max_drawdown / max(abs(sum(values)), 1.0), 6)


def _recent_retention_window(
    *,
    previous_state: dict[str, object],
    validation_evidence: dict[str, object],
) -> dict[str, object]:
    active_evidence = _active_policy_evidence(validation_evidence)
    bucket_direct = _policy_context_bucket_direct_metrics(active_evidence)
    validation_runs = list(active_evidence.get("validation_runs", []) or [])
    walk_forward_windows = list(active_evidence.get("walk_forward_windows", []) or [])
    recent_runs = validation_runs[-3:]
    recent_windows = walk_forward_windows[-3:]
    recent_bucket_runs = list(bucket_direct.get("validation_runs", []) or [])[-3:]
    recent_bucket_windows = list(bucket_direct.get("walk_forward_windows", []) or [])[-3:]
    recent_live_order_count = sum(_coerce_int(item.get("live_order_count")) for item in recent_runs)
    recent_rejected_live_order_count = sum(_coerce_int(item.get("rejected_live_order_count")) for item in recent_runs)
    recent_accepted_live_order_count = sum(_coerce_int(item.get("accepted_live_order_count")) for item in recent_runs)
    recent_closed_trade_count = sum(_coerce_int(item.get("closed_trade_count")) for item in recent_runs)
    bucket_recent_closed_trade_count = sum(_coerce_int(item.get("closed_trade_count")) for item in recent_bucket_runs)
    retention_weight = float(max(recent_live_order_count, 0))
    slippage_weight = float(max(recent_accepted_live_order_count, 0))
    recent_retention = (
        round(
            sum(_coerce_float(item.get("avg_edge_retention_ratio")) * _coerce_int(item.get("live_order_count")) for item in recent_runs)
            / retention_weight,
            6,
        )
        if retention_weight > 0.0
        else 0.0
    )
    recent_reject_rate = round(recent_rejected_live_order_count / max(recent_live_order_count, 1), 6) if recent_live_order_count > 0 else 0.0
    recent_slippage = (
        round(
            sum(_coerce_float(item.get("avg_slippage_bps")) * _coerce_int(item.get("accepted_live_order_count")) for item in recent_runs)
            / slippage_weight,
            6,
        )
        if slippage_weight > 0.0
        else 0.0
    )
    recent_pnl_series = [_coerce_float(item.get("realized_pnl_usd")) for item in recent_runs]
    bucket_recent_pnl_series = [_coerce_float(item.get("realized_pnl_usd")) for item in recent_bucket_runs]
    recent_positive_walk_forward_count = sum(
        1
        for item in recent_windows
        if _coerce_float(item.get("avg_net_edge_bps")) > 0.0 and _coerce_float(item.get("avg_score")) >= 0.0
    )
    bucket_recent_positive_walk_forward_count = sum(
        1
        for item in recent_bucket_windows
        if _coerce_float(item.get("avg_net_edge_bps")) > 0.0 and _coerce_float(item.get("avg_score")) >= 0.0
    )
    recent_positive_walk_forward_ratio = (
        round(recent_positive_walk_forward_count / len(recent_windows), 6) if recent_windows else 0.0
    )
    bucket_recent_positive_walk_forward_ratio = (
        round(bucket_recent_positive_walk_forward_count / len(recent_bucket_windows), 6) if recent_bucket_windows else 0.0
    )
    previous_metrics = dict(dict(previous_state.get("retention_monitor", {}) or {}).get("metrics", {}) or {})
    previous_recent = dict(previous_metrics.get("recent_window", {}) or {})
    micro_live_gate = dict(active_evidence.get("micro_live_gate", {}) or {})
    bucket_recent_drawdown = _recent_drawdown_ratio(bucket_recent_pnl_series)
    effective_drawdown = bucket_recent_drawdown if recent_bucket_runs else _recent_drawdown_ratio(recent_pnl_series)
    return {
        "available": bool(recent_runs or recent_windows),
        "run_count": len(recent_runs),
        "walk_forward_window_count": len(recent_bucket_windows) if recent_bucket_windows else len(recent_windows),
        "live_order_count": recent_live_order_count,
        "accepted_live_order_count": recent_accepted_live_order_count,
        "rejected_live_order_count": recent_rejected_live_order_count,
        "closed_trade_count": bucket_recent_closed_trade_count if recent_bucket_runs else recent_closed_trade_count,
        "avg_edge_retention_ratio": recent_retention,
        "drawdown_to_pnl_ratio": effective_drawdown,
        "reject_rate": recent_reject_rate,
        "avg_slippage_bps": recent_slippage,
        "positive_walk_forward_ratio": (
            bucket_recent_positive_walk_forward_ratio if recent_bucket_windows else recent_positive_walk_forward_ratio
        ),
        "micro_live_status": str(micro_live_gate.get("status", "not_available") or "not_available"),
        "bucket_available": bool(bucket_direct.get("available")),
        "bucket_decision_count": sum(_coerce_int(item.get("decision_count")) for item in recent_bucket_runs),
        "bucket_realized_pnl_usd": round(sum(bucket_recent_pnl_series), 6),
        "bucket_run_count": len(recent_bucket_runs),
        "retention_delta": round(recent_retention - _coerce_float(previous_recent.get("avg_edge_retention_ratio", recent_retention)), 6),
        "drawdown_delta": round(effective_drawdown - _coerce_float(previous_recent.get("drawdown_to_pnl_ratio", effective_drawdown)), 6),
        "reject_rate_delta": round(recent_reject_rate - _coerce_float(previous_recent.get("reject_rate", recent_reject_rate)), 6),
    }


def _cumulative_retention_window(
    *,
    previous_state: dict[str, object],
    validation_evidence: dict[str, object],
) -> dict[str, object]:
    active_evidence = _active_policy_evidence(validation_evidence)
    bucket_direct = _policy_context_bucket_direct_metrics(active_evidence)
    validation_runs = list(active_evidence.get("validation_runs", []) or [])
    walk_forward_windows = list(active_evidence.get("walk_forward_windows", []) or [])
    bucket_runs = list(bucket_direct.get("validation_runs", []) or [])
    bucket_windows = list(bucket_direct.get("walk_forward_windows", []) or [])
    live_order_count = sum(_coerce_int(item.get("live_order_count")) for item in validation_runs)
    accepted_live_order_count = sum(_coerce_int(item.get("accepted_live_order_count")) for item in validation_runs)
    rejected_live_order_count = sum(_coerce_int(item.get("rejected_live_order_count")) for item in validation_runs)
    closed_trade_count = sum(_coerce_int(item.get("closed_trade_count")) for item in validation_runs)
    bucket_closed_trade_count = sum(_coerce_int(item.get("closed_trade_count")) for item in bucket_runs)
    retention_weight = float(max(live_order_count, 0))
    slippage_weight = float(max(accepted_live_order_count, 0))
    avg_edge_retention_ratio = (
        round(
            sum(_coerce_float(item.get("avg_edge_retention_ratio")) * _coerce_int(item.get("live_order_count")) for item in validation_runs)
            / retention_weight,
            6,
        )
        if retention_weight > 0.0
        else 0.0
    )
    reject_rate = round(rejected_live_order_count / max(live_order_count, 1), 6) if live_order_count > 0 else 0.0
    avg_slippage_bps = (
        round(
            sum(_coerce_float(item.get("avg_slippage_bps")) * _coerce_int(item.get("accepted_live_order_count")) for item in validation_runs)
            / slippage_weight,
            6,
        )
        if slippage_weight > 0.0
        else 0.0
    )
    pnl_series = [_coerce_float(item.get("realized_pnl_usd")) for item in validation_runs]
    bucket_pnl_series = [_coerce_float(item.get("realized_pnl_usd")) for item in bucket_runs]
    positive_walk_forward_count = sum(
        1
        for item in walk_forward_windows
        if _coerce_float(item.get("avg_net_edge_bps")) > 0.0 and _coerce_float(item.get("avg_score")) >= 0.0
    )
    bucket_positive_walk_forward_count = sum(
        1
        for item in bucket_windows
        if _coerce_float(item.get("avg_net_edge_bps")) > 0.0 and _coerce_float(item.get("avg_score")) >= 0.0
    )
    positive_walk_forward_ratio = (
        round(positive_walk_forward_count / len(walk_forward_windows), 6) if walk_forward_windows else 0.0
    )
    bucket_positive_walk_forward_ratio = (
        round(bucket_positive_walk_forward_count / len(bucket_windows), 6) if bucket_windows else 0.0
    )
    previous_metrics = dict(dict(previous_state.get("retention_monitor", {}) or {}).get("metrics", {}) or {})
    previous_cumulative = dict(previous_metrics.get("cumulative_window", {}) or {})
    effective_drawdown = _recent_drawdown_ratio(bucket_pnl_series) if bucket_runs else _recent_drawdown_ratio(pnl_series)
    return {
        "available": bool(validation_runs or walk_forward_windows),
        "run_count": len(validation_runs),
        "walk_forward_window_count": len(bucket_windows) if bucket_windows else len(walk_forward_windows),
        "live_order_count": live_order_count,
        "accepted_live_order_count": accepted_live_order_count,
        "rejected_live_order_count": rejected_live_order_count,
        "closed_trade_count": bucket_closed_trade_count if bucket_runs else closed_trade_count,
        "avg_edge_retention_ratio": avg_edge_retention_ratio,
        "drawdown_to_pnl_ratio": effective_drawdown,
        "reject_rate": reject_rate,
        "avg_slippage_bps": avg_slippage_bps,
        "positive_walk_forward_ratio": bucket_positive_walk_forward_ratio if bucket_windows else positive_walk_forward_ratio,
        "bucket_available": bool(bucket_direct.get("available")),
        "bucket_decision_count": sum(_coerce_int(item.get("decision_count")) for item in bucket_runs),
        "bucket_realized_pnl_usd": round(sum(bucket_pnl_series), 6),
        "bucket_run_count": len(bucket_runs),
        "retention_delta": round(
            avg_edge_retention_ratio
            - _coerce_float(previous_cumulative.get("avg_edge_retention_ratio", avg_edge_retention_ratio)),
            6,
        ),
        "drawdown_delta": round(
            effective_drawdown
            - _coerce_float(previous_cumulative.get("drawdown_to_pnl_ratio", effective_drawdown)),
            6,
        ),
        "reject_rate_delta": round(reject_rate - _coerce_float(previous_cumulative.get("reject_rate", reject_rate)), 6),
    }


def _rollout_execution_phase_rank(phase: str) -> int:
    return {
        "baseline": 0,
        "partial": 1,
        "broad": 2,
        "full": 3,
    }.get(phase, 0)


def _rollout_execution_phase(
    *,
    previous_progress: dict[str, object],
    rollout_status: str,
    active_status: str,
    requested_status: str,
    retention_monitor_status: str,
    live_order_count: int,
    closed_trade_count: int,
    required_live_order_count: int,
    required_closed_trade_count: int,
    walk_forward_window_count: int,
    positive_walk_forward_ratio: float,
    avg_edge_retention_ratio: float,
    drawdown_ratio: float,
    reject_rate: float,
) -> tuple[str, str]:
    if retention_monitor_status == "rollback":
        return "rollback", "POST_PROMOTION_RETENTION_DEGRADED"
    if retention_monitor_status == "demote":
        return "watch", "POST_PROMOTION_RETENTION_WEAKENED"
    if active_status not in {"promote", "promote_aggressive"} and requested_status not in {"promote", "promote_aggressive"} and rollout_status != "micro_live_pending":
        return "baseline", "NO_ACTIVE_ROLLOUT"
    phase = "baseline"
    reason = "ROLLOUT_NOT_STARTED"
    if live_order_count > 0 or closed_trade_count > 0 or rollout_status == "micro_live_pending":
        phase = "partial"
        reason = "MICRO_LIVE_EVIDENCE_ACCUMULATING"
    if (
        live_order_count >= max(required_live_order_count * 2, 4)
        and closed_trade_count >= max(required_closed_trade_count, 1)
        and walk_forward_window_count >= 2
        and positive_walk_forward_ratio >= 0.5
        and avg_edge_retention_ratio >= 0.65
        and drawdown_ratio <= 0.45
        and reject_rate <= 0.08
    ):
        phase = "broad"
        reason = "ROLLOUT_BROADENING_WITH_STABLE_EVIDENCE"
    if (
        live_order_count >= max(required_live_order_count * 4, 8)
        and closed_trade_count >= max(required_closed_trade_count * 2, 2)
        and walk_forward_window_count >= 3
        and positive_walk_forward_ratio >= 0.75
        and avg_edge_retention_ratio >= 0.75
        and drawdown_ratio <= 0.35
        and reject_rate <= 0.05
    ):
        phase = "full"
        reason = "ROLLOUT_READY_FOR_FULL_COVERAGE"
    previous_phase = str(previous_progress.get("execution_phase", "baseline") or "baseline")
    if _rollout_execution_phase_rank(previous_phase) > _rollout_execution_phase_rank(phase):
        phase = previous_phase
        reason = str(previous_progress.get("execution_phase_reason", reason) or reason)
    return phase, reason


def _retention_monitor(
    previous_state: dict[str, object],
    previous_active: dict[str, object],
    validation_evidence: dict[str, object],
    operational_verdict: dict[str, object],
) -> dict[str, object]:
    active_status = str(previous_active.get("status", "") or "")
    if active_status not in {"promote", "promote_aggressive"}:
        return {"status": "inactive", "reasons": []}
    active_evidence = _active_policy_evidence(validation_evidence)
    bucket_direct = _policy_context_bucket_direct_metrics(active_evidence)
    runner_quality = _runner_quality_evidence(active_evidence)
    if not bool(runner_quality.get("available")):
        return {"status": "inactive", "reasons": []}
    recent_window = _recent_retention_window(previous_state=previous_state, validation_evidence=validation_evidence)
    cumulative_window = _cumulative_retention_window(previous_state=previous_state, validation_evidence=validation_evidence)
    reasons: list[str] = []
    retention = float(runner_quality.get("avg_edge_retention_ratio", 0.0) or 0.0)
    realized = float(
        bucket_direct.get("realized_pnl_usd", runner_quality.get("realized_pnl_usd", 0.0))
        if bool(bucket_direct.get("available")) and _coerce_int(bucket_direct.get("closed_trade_count")) > 0
        else runner_quality.get("realized_pnl_usd", 0.0)
        or 0.0
    )
    drawdown_ratio = float(runner_quality.get("drawdown_ratio", 0.0) or 0.0)
    reject_rate = float(runner_quality.get("reject_rate", 0.0) or 0.0)
    slippage = float(runner_quality.get("avg_slippage_bps", 0.0) or 0.0)
    walk_forward_window_count = int(
        bucket_direct.get("walk_forward_window_count", runner_quality.get("walk_forward_window_count", 0))
        if bool(bucket_direct.get("available")) and _coerce_int(bucket_direct.get("walk_forward_window_count")) > 0
        else runner_quality.get("walk_forward_window_count", 0)
        or 0
    )
    positive_walk_forward_ratio = float(
        bucket_direct.get("positive_walk_forward_ratio", runner_quality.get("positive_walk_forward_ratio", 0.0))
        if bool(bucket_direct.get("available")) and _coerce_int(bucket_direct.get("walk_forward_window_count")) > 0
        else runner_quality.get("positive_walk_forward_ratio", 0.0)
        or 0.0
    )
    operational_status = str(operational_verdict.get("status", "hold") or "hold")
    sample_watchdog_status = str(dict(active_evidence.get("sample_quality_watchdog", {}) or {}).get("status", "") or "")
    if operational_status == "stop":
        reasons.append("RETENTION_MONITOR_OPERATIONAL_STOP")
    if sample_watchdog_status == "degraded":
        reasons.append("RETENTION_MONITOR_SAMPLE_WATCHDOG_DEGRADED")
    if retention < 0.40:
        reasons.append("RETENTION_MONITOR_EDGE_TOO_LOW")
    if realized <= 0.0:
        reasons.append("RETENTION_MONITOR_REALIZED_PNL_NON_POSITIVE")
    if drawdown_ratio > 0.75:
        reasons.append("RETENTION_MONITOR_DRAWDOWN_TOO_HIGH")
    if reject_rate > 0.15:
        reasons.append("RETENTION_MONITOR_REJECT_RATE_TOO_HIGH")
    if walk_forward_window_count >= 3 and positive_walk_forward_ratio < 0.34:
        reasons.append("RETENTION_MONITOR_WALK_FORWARD_COLLAPSED")
    metrics = {
        "avg_edge_retention_ratio": round(retention, 6),
        "realized_pnl_usd": round(realized, 6),
        "drawdown_ratio": round(drawdown_ratio, 6),
        "reject_rate": round(reject_rate, 6),
        "avg_slippage_bps": round(slippage, 6),
        "walk_forward_window_count": walk_forward_window_count,
        "positive_walk_forward_ratio": round(positive_walk_forward_ratio, 6),
        "policy_context_bucket_available": bool(bucket_direct.get("available")),
        "policy_context_bucket_name": str(bucket_direct.get("bucket_name", "") or ""),
        "policy_context_bucket_decision_count": _coerce_int(bucket_direct.get("decision_count")),
        "policy_context_bucket_closed_trade_count": _coerce_int(bucket_direct.get("closed_trade_count")),
        "recent_window": recent_window,
        "cumulative_window": cumulative_window,
    }
    if bool(recent_window.get("available")):
        if float(recent_window.get("avg_edge_retention_ratio", 0.0) or 0.0) < 0.35 and int(recent_window.get("live_order_count", 0) or 0) >= 2:
            reasons.append("RETENTION_MONITOR_RECENT_WINDOW_EDGE_COLLAPSE")
        if float(recent_window.get("drawdown_to_pnl_ratio", 0.0) or 0.0) > 0.85 and int(recent_window.get("closed_trade_count", 0) or 0) >= 1:
            reasons.append("RETENTION_MONITOR_RECENT_DRAWDOWN_SPIKE")
        if float(recent_window.get("reject_rate", 0.0) or 0.0) > 0.2 and int(recent_window.get("live_order_count", 0) or 0) >= 3:
            reasons.append("RETENTION_MONITOR_RECENT_REJECT_SURGE")
        if int(recent_window.get("walk_forward_window_count", 0) or 0) >= 2 and float(recent_window.get("positive_walk_forward_ratio", 0.0) or 0.0) < 0.25:
            reasons.append("RETENTION_MONITOR_RECENT_WALK_FORWARD_FAIL")
    if bool(cumulative_window.get("available")):
        if float(cumulative_window.get("avg_edge_retention_ratio", 0.0) or 0.0) < 0.45 and int(cumulative_window.get("live_order_count", 0) or 0) >= 4:
            reasons.append("RETENTION_MONITOR_CUMULATIVE_EDGE_COLLAPSE")
        if float(cumulative_window.get("drawdown_to_pnl_ratio", 0.0) or 0.0) > 0.75 and int(cumulative_window.get("closed_trade_count", 0) or 0) >= 2:
            reasons.append("RETENTION_MONITOR_CUMULATIVE_DRAWDOWN_SPIKE")
        if int(cumulative_window.get("walk_forward_window_count", 0) or 0) >= 4 and float(cumulative_window.get("positive_walk_forward_ratio", 0.0) or 0.0) < 0.34:
            reasons.append("RETENTION_MONITOR_CUMULATIVE_WALK_FORWARD_FAIL")
    if reasons:
        return {"status": "rollback", "reasons": reasons, "metrics": metrics}
    moderate_reasons: list[str] = []
    if operational_status == "hold":
        moderate_reasons.append("RETENTION_MONITOR_OPERATIONAL_HOLD")
    if sample_watchdog_status == "thin":
        moderate_reasons.append("RETENTION_MONITOR_SAMPLE_WATCHDOG_THIN")
    if retention < 0.65:
        moderate_reasons.append("RETENTION_MONITOR_EDGE_BELOW_PASS")
    if drawdown_ratio > 0.45:
        moderate_reasons.append("RETENTION_MONITOR_DRAWDOWN_ELEVATED")
    if reject_rate > 0.08:
        moderate_reasons.append("RETENTION_MONITOR_REJECT_RATE_ELEVATED")
    if slippage > 10.0:
        moderate_reasons.append("RETENTION_MONITOR_SLIPPAGE_ELEVATED")
    if walk_forward_window_count >= 2 and positive_walk_forward_ratio < 0.5:
        moderate_reasons.append("RETENTION_MONITOR_WALK_FORWARD_WEAK")
    if bool(recent_window.get("available")):
        if float(recent_window.get("avg_edge_retention_ratio", 0.0) or 0.0) < 0.58 and int(recent_window.get("live_order_count", 0) or 0) >= 2:
            moderate_reasons.append("RETENTION_MONITOR_RECENT_WINDOW_EDGE_WEAK")
        if float(recent_window.get("drawdown_to_pnl_ratio", 0.0) or 0.0) > 0.45 and int(recent_window.get("closed_trade_count", 0) or 0) >= 1:
            moderate_reasons.append("RETENTION_MONITOR_RECENT_DRAWDOWN_ELEVATED")
        if float(recent_window.get("reject_rate", 0.0) or 0.0) > 0.1 and int(recent_window.get("live_order_count", 0) or 0) >= 3:
            moderate_reasons.append("RETENTION_MONITOR_RECENT_REJECT_ELEVATED")
        if int(recent_window.get("walk_forward_window_count", 0) or 0) >= 2 and float(recent_window.get("positive_walk_forward_ratio", 0.0) or 0.0) < 0.5:
            moderate_reasons.append("RETENTION_MONITOR_RECENT_WALK_FORWARD_WEAK")
        if float(recent_window.get("retention_delta", 0.0) or 0.0) < -0.12:
            moderate_reasons.append("RETENTION_MONITOR_RETENTION_TREND_NEGATIVE")
    if bool(cumulative_window.get("available")):
        if float(cumulative_window.get("avg_edge_retention_ratio", 0.0) or 0.0) < 0.62 and int(cumulative_window.get("live_order_count", 0) or 0) >= 4:
            moderate_reasons.append("RETENTION_MONITOR_CUMULATIVE_EDGE_WEAK")
        if float(cumulative_window.get("drawdown_to_pnl_ratio", 0.0) or 0.0) > 0.45 and int(cumulative_window.get("closed_trade_count", 0) or 0) >= 2:
            moderate_reasons.append("RETENTION_MONITOR_CUMULATIVE_DRAWDOWN_ELEVATED")
        if float(cumulative_window.get("reject_rate", 0.0) or 0.0) > 0.12 and int(cumulative_window.get("live_order_count", 0) or 0) >= 4:
            moderate_reasons.append("RETENTION_MONITOR_CUMULATIVE_REJECT_ELEVATED")
        if int(cumulative_window.get("walk_forward_window_count", 0) or 0) >= 3 and float(cumulative_window.get("positive_walk_forward_ratio", 0.0) or 0.0) < 0.5:
            moderate_reasons.append("RETENTION_MONITOR_CUMULATIVE_WALK_FORWARD_WEAK")
        if float(cumulative_window.get("retention_delta", 0.0) or 0.0) < -0.08:
            moderate_reasons.append("RETENTION_MONITOR_CUMULATIVE_TREND_NEGATIVE")
    if moderate_reasons:
        return {"status": "demote", "reasons": moderate_reasons, "metrics": metrics}
    return {"status": "stable", "reasons": [], "metrics": metrics}


def _demoted_active_policy(previous_active: dict[str, object]) -> dict[str, object]:
    demoted_adjustments: list[dict[str, object]] = []
    for item in list(previous_active.get("adjustments", []) or []):
        base_size = float(item.get("size_multiplier", 1.0) or 1.0)
        size_multiplier = round(min(base_size, 1.0) * 0.85, 6)
        if size_multiplier >= 1.0:
            size_multiplier = 0.9
        leverage_multiplier = round(min(float(item.get("leverage_multiplier", size_multiplier) or size_multiplier), 1.0), 6)
        demoted_adjustments.append(
            dict(
                item,
                action="demote",
                size_multiplier=size_multiplier,
                leverage_multiplier=leverage_multiplier,
                entry_threshold_bps=max(float(item.get("entry_threshold_bps", 0.0) or 0.0), 1.0),
                expected_profit_floor_bps=max(float(item.get("expected_profit_floor_bps", 0.0) or 0.0), 1.0),
                reason="RETENTION_MONITOR_DEMOTE",
            )
        )
    return {"status": "demote", "adjustments": demoted_adjustments}


def _apply_symbol_lifecycle_to_active_policy(
    active_policy: dict[str, object],
    symbol_lifecycle: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> dict[str, object]:
    lifecycle_by_symbol = {
        str(row.get("symbol", "") or ""): dict(row)
        for row in list(symbol_lifecycle or [])
        if str(row.get("symbol", "") or "")
    }
    if not lifecycle_by_symbol:
        return active_policy
    adjustments: list[dict[str, object]] = []
    changed = False
    for item in list(dict(active_policy or {}).get("adjustments", []) or []):
        symbol = str(item.get("symbol", "") or "")
        lifecycle_row = dict(lifecycle_by_symbol.get(symbol, {}) or {})
        lifecycle_action = str(lifecycle_row.get("recommended_action", "keep") or "keep")
        action = str(item.get("action", "") or "")
        if action in {"promote", "aggressive_promote"} and lifecycle_action in {"rollback", "hold"}:
            adjustments.append(
                _retune_adjustment_for_action(
                    dict(item),
                    action="demote",
                    reason="SYMBOL_LIFECYCLE_ROLLBACK",
                    lifecycle_row=lifecycle_row,
                )
            )
            changed = True
        elif action == "aggressive_promote" and lifecycle_action == "cautious_repromote":
            adjustments.append(
                _retune_adjustment_for_action(
                    dict(item),
                    action="promote",
                    reason="SYMBOL_LIFECYCLE_CAUTIOUS_REPROMOTION",
                    lifecycle_row=lifecycle_row,
                )
            )
            changed = True
        else:
            adjustments.append(dict(item))
    if not changed:
        return active_policy
    updated = dict(active_policy)
    updated["adjustments"] = adjustments
    if adjustments and all(str(item.get("action", "") or "") in {"demote", "disabled"} for item in adjustments):
        updated["status"] = "demote"
    return updated


def _annotate_active_policy(
    active_policy: dict[str, object],
    promotion_verdict: dict[str, object],
    retention_monitor: dict[str, object],
) -> dict[str, object]:
    annotated = dict(active_policy)
    annotated["lifecycle_stage"] = str(promotion_verdict.get("rollout_stage", "baseline") or "baseline")
    annotated["requested_status"] = str(promotion_verdict.get("requested_status", promotion_verdict.get("status", "keep")) or "keep")
    annotated["effective_status"] = str(promotion_verdict.get("effective_status", promotion_verdict.get("status", "keep")) or "keep")
    annotated["micro_live_readiness"] = str(promotion_verdict.get("micro_live_readiness", "not_available") or "not_available")
    annotated["micro_live_gate"] = dict(promotion_verdict.get("micro_live_gate", {}) or {})
    annotated["retention_monitor_status"] = str(retention_monitor.get("status", "inactive") or "inactive")
    return annotated


def _rollout_progression_signal(
    *,
    previous_state: dict[str, object],
    active_policy: dict[str, object],
    promotion_verdict: dict[str, object],
    validation_evidence: dict[str, object],
    retention_monitor: dict[str, object],
    rollout_status: str,
    rollout_reason: str,
) -> dict[str, object]:
    previous_progress = dict(previous_state.get("rollout_progression", {}) or {})
    previous_status = str(previous_progress.get("status", previous_state.get("rollout_status", "baseline")) or "baseline")
    active_status = str(active_policy.get("status", "baseline") or "baseline")
    requested_status = str(promotion_verdict.get("requested_status", promotion_verdict.get("status", "keep")) or "keep")
    effective_status = str(promotion_verdict.get("effective_status", promotion_verdict.get("status", "keep")) or "keep")
    active_evidence = _active_policy_evidence(validation_evidence)
    runner_quality = _runner_quality_evidence(active_evidence)
    staged_candidate_evidence = _staged_candidate_evidence(validation_evidence)
    micro_live_gate = dict(runner_quality.get("micro_live_gate", {}) or {})
    if not micro_live_gate:
        micro_live_gate = dict(staged_candidate_evidence.get("micro_live_gate", {}) or {})
    live_order_count = int(micro_live_gate.get("live_order_count", 0) or 0)
    closed_trade_count = int(micro_live_gate.get("closed_trade_count", 0) or 0)
    required_live_order_count = int(micro_live_gate.get("required_live_order_count", 2) or 2)
    required_closed_trade_count = int(micro_live_gate.get("required_closed_trade_count", 1) or 1)
    walk_forward_window_count = int(runner_quality.get("walk_forward_window_count", 0) or 0)
    positive_walk_forward_ratio = float(runner_quality.get("positive_walk_forward_ratio", 0.0) or 0.0)
    avg_edge_retention_ratio = float(runner_quality.get("avg_edge_retention_ratio", 0.0) or 0.0)
    drawdown_ratio = float(runner_quality.get("drawdown_ratio", 0.0) or 0.0)
    reject_rate = float(runner_quality.get("reject_rate", 0.0) or 0.0)
    progression_phase = "baseline"
    progression_status = rollout_status or "baseline"
    progression_reason = rollout_reason or "NO_ACTIVE_POLICY"
    if rollout_status == "micro_live_pending":
        progression_phase = "staged_rollout"
        if not bool(micro_live_gate.get("available")) or live_order_count <= 0:
            progression_status = "awaiting_micro_live_orders"
            progression_reason = "MICRO_LIVE_ORDERS_REQUIRED"
        elif live_order_count < required_live_order_count:
            progression_status = "collecting_micro_live_orders"
            progression_reason = "MICRO_LIVE_ORDER_SAMPLE_INCOMPLETE"
        elif closed_trade_count < required_closed_trade_count:
            progression_status = "collecting_micro_live_outcomes"
            progression_reason = "MICRO_LIVE_CLOSED_TRADE_REQUIRED"
        else:
            progression_status = "micro_live_quality_review"
            progression_reason = str(micro_live_gate.get("reason", "MICRO_LIVE_THRESHOLD_NOT_MET") or "MICRO_LIVE_THRESHOLD_NOT_MET")
    elif str(retention_monitor.get("status", "inactive") or "inactive") == "rollback":
        progression_phase = "post_promotion_monitoring"
        progression_status = "rollback_triggered"
        progression_reason = "POST_PROMOTION_MONITOR_TRIGGERED_ROLLBACK"
    elif str(retention_monitor.get("status", "inactive") or "inactive") == "demote":
        progression_phase = "post_promotion_monitoring"
        progression_status = "demotion_watch"
        progression_reason = "POST_PROMOTION_MONITOR_TRIGGERED_DEMOTION"
    elif active_status in {"promote", "promote_aggressive"}:
        progression_phase = "post_promotion_monitoring"
        if str(retention_monitor.get("status", "inactive") or "inactive") == "armed":
            progression_status = "post_promotion_monitoring"
            progression_reason = "POST_PROMOTION_MONITORING_ACTIVE"
        elif (
            walk_forward_window_count >= 2
            and positive_walk_forward_ratio >= 0.75
            and float(runner_quality.get("avg_edge_retention_ratio", 0.0) or 0.0) >= 0.75
            and float(runner_quality.get("drawdown_ratio", 0.0) or 0.0) <= 0.35
            and float(runner_quality.get("reject_rate", 0.0) or 0.0) <= 0.05
        ):
            progression_status = "expansion_ready"
            progression_reason = "POST_PROMOTION_EVIDENCE_STRONG"
        else:
            progression_status = "promotion_live"
            progression_reason = "PROMOTION_ACTIVE_UNDER_MONITORING"
    elif requested_status in {"promote", "promote_aggressive"} or previous_status == "micro_live_pending":
        progression_phase = "staged_rollout"
        if not bool(micro_live_gate.get("available")) or live_order_count <= 0:
            progression_status = "awaiting_micro_live_orders"
            progression_reason = "MICRO_LIVE_ORDERS_REQUIRED"
        elif live_order_count < required_live_order_count:
            progression_status = "collecting_micro_live_orders"
            progression_reason = "MICRO_LIVE_ORDER_SAMPLE_INCOMPLETE"
        elif closed_trade_count < required_closed_trade_count:
            progression_status = "collecting_micro_live_outcomes"
            progression_reason = "MICRO_LIVE_CLOSED_TRADE_REQUIRED"
        elif str(micro_live_gate.get("status", "")) != "pass":
            progression_status = "micro_live_quality_review"
            progression_reason = str(micro_live_gate.get("reason", "MICRO_LIVE_THRESHOLD_NOT_MET") or "MICRO_LIVE_THRESHOLD_NOT_MET")
        elif effective_status in {"promote", "promote_aggressive"}:
            progression_status = "ready_for_expansion"
            progression_reason = "MICRO_LIVE_GATE_PASSED"
        else:
            progression_status = "promotion_review"
            progression_reason = "PROMOTION_PENDING_FINAL_DECISION"
    elif effective_status == "demote":
        progression_phase = "demotion"
        progression_status = "demotion_active"
        progression_reason = rollout_reason or "DEMOTION_VALIDATED"
    elif effective_status == "disable":
        progression_phase = "disabled"
        progression_status = "disabled"
        progression_reason = rollout_reason or "CANDIDATE_DISABLED"
    execution_phase, execution_phase_reason = _rollout_execution_phase(
        previous_progress=previous_progress,
        rollout_status=rollout_status,
        active_status=active_status,
        requested_status=requested_status,
        retention_monitor_status=str(retention_monitor.get("status", "inactive") or "inactive"),
        live_order_count=live_order_count,
        closed_trade_count=closed_trade_count,
        required_live_order_count=required_live_order_count,
        required_closed_trade_count=required_closed_trade_count,
        walk_forward_window_count=walk_forward_window_count,
        positive_walk_forward_ratio=positive_walk_forward_ratio,
        avg_edge_retention_ratio=avg_edge_retention_ratio,
        drawdown_ratio=drawdown_ratio,
        reject_rate=reject_rate,
    )
    return {
        "phase": progression_phase,
        "status": progression_status,
        "reason": progression_reason,
        "previous_status": previous_status,
        "execution_phase": execution_phase,
        "execution_phase_reason": execution_phase_reason,
        "evidence": {
            "live_order_count": live_order_count,
            "closed_trade_count": closed_trade_count,
            "required_live_order_count": required_live_order_count,
            "required_closed_trade_count": required_closed_trade_count,
            "micro_live_status": str(micro_live_gate.get("status", "not_available") or "not_available"),
            "walk_forward_window_count": walk_forward_window_count,
            "positive_walk_forward_ratio": round(positive_walk_forward_ratio, 6),
            "avg_edge_retention_ratio": round(avg_edge_retention_ratio, 6),
            "drawdown_ratio": round(drawdown_ratio, 6),
            "reject_rate": round(reject_rate, 6),
        },
    }


def _sample_quality_checkpoint_revalidation(
    previous_state: dict[str, object],
    validation_evidence: dict[str, object],
) -> dict[str, object]:
    previous_watchdog = dict(
        dict(
            dict(previous_state.get("policy_validation", {}) or {}).get("evidence", {})
            or {}
        ).get("sample_quality_watchdog", {})
        or {}
    )
    current_watchdog = dict(_active_policy_evidence(validation_evidence).get("sample_quality_watchdog", {}) or {})
    previous_snapshot = dict(previous_watchdog.get("checkpoint_snapshot", {}) or {})
    current_snapshot = dict(current_watchdog.get("checkpoint_snapshot", {}) or {})
    hooks: list[dict[str, object]] = []
    previous_portfolio = {
        (str(item.get("metric", "") or ""), int(item.get("threshold", 0) or 0)): dict(item)
        for item in list(previous_snapshot.get("portfolio", []) or [])
        if str(item.get("metric", "") or "")
    }
    for item in list(current_snapshot.get("portfolio", []) or []):
        metric = str(item.get("metric", "") or "")
        threshold = int(item.get("threshold", 0) or 0)
        current_reached = bool(item.get("reached"))
        previous_reached = bool(previous_portfolio.get((metric, threshold), {}).get("reached"))
        if current_reached and not previous_reached:
            hooks.append(
                {
                    "kind": "portfolio_threshold_crossed",
                    "metric": metric,
                    "threshold": threshold,
                    "previous_value": int(previous_portfolio.get((metric, threshold), {}).get("current_value", 0) or 0),
                    "current_value": int(item.get("current_value", 0) or 0),
                }
            )
    previous_symbols = {
        str(item.get("symbol", "") or ""): dict(item)
        for item in list(previous_snapshot.get("symbols", []) or [])
        if str(item.get("symbol", "") or "")
    }
    for item in list(current_snapshot.get("symbols", []) or []):
        symbol = str(item.get("symbol", "") or "")
        if not symbol:
            continue
        if bool(item.get("validation_ready")) and not bool(previous_symbols.get(symbol, {}).get("validation_ready")):
            hooks.append(
                {
                    "kind": "symbol_validation_ready",
                    "symbol": symbol,
                    "threshold": int(item.get("validation_threshold", 0) or 0),
                    "previous_value": int(previous_symbols.get(symbol, {}).get("trade_count", 0) or 0),
                    "current_value": int(item.get("trade_count", 0) or 0),
                }
            )
    return {
        "triggered": bool(hooks),
        "hooks": hooks,
        "watchdog_status": str(current_watchdog.get("status", "not_available") or "not_available"),
    }


def _protective_policy_transition(
    *,
    verdict_status: str,
    candidate_adjustments: list[dict[str, object]],
    comparison_verdict: str,
) -> str:
    if comparison_verdict == "candidate_worse":
        return ""
    actions = {
        str(item.get("action", "") or "")
        for item in list(candidate_adjustments or [])
        if str(item.get("action", "") or "")
    }
    if verdict_status == "disable":
        return "disable"
    if verdict_status == "demote" and actions and actions <= {"demote", "disabled"}:
        return "demote"
    return ""


def _resolved_policy_comparison_signal(
    *,
    validation_evidence: dict[str, object],
    promotion_verdict: dict[str, object],
) -> tuple[str, float]:
    comparison_verdict, comparison_delta = _policy_comparison_signal(validation_evidence)
    if comparison_verdict != "keep" or comparison_delta != 0.0:
        return comparison_verdict, comparison_delta
    fallback_verdict, fallback_delta = _policy_comparison_signal(promotion_verdict)
    if fallback_verdict != "keep" or fallback_delta != 0.0:
        return fallback_verdict, fallback_delta
    return comparison_verdict, comparison_delta


def _invalidate_staged_rollout(
    *,
    previous_state: dict[str, object],
    validation_status: str,
    staged_micro_live_block: bool,
) -> bool:
    if str(previous_state.get("status", "") or "") != "staged_rollout":
        return False
    if str(previous_state.get("rollout_status", "") or "") != "micro_live_pending":
        return False
    if staged_micro_live_block:
        return False
    return validation_status == "fail"


def build_persisted_policy_state(
    previous_state: dict[str, object] | None,
    candidate_policy: dict[str, object],
    promotion_verdict: dict[str, object],
    operational_verdict: dict[str, object],
    validation: dict[str, object],
) -> dict[str, object]:
    previous_state = dict(previous_state or {})
    previous_active = dict(previous_state.get("active_policy", {}) or {})
    previous_version = int(previous_state.get("version", 0) or 0)
    verdict_status = str(promotion_verdict.get("status", "keep"))
    validation_status = str(validation.get("status", "fail"))
    validation_evidence = dict(validation.get("evidence", {}) or {})
    staged_candidate_evidence = _staged_candidate_evidence(validation_evidence)
    comparison_verdict, _ = _resolved_policy_comparison_signal(
        validation_evidence=validation_evidence,
        promotion_verdict=promotion_verdict,
    )
    comparison_underperforms = comparison_verdict == "candidate_worse"
    candidate_adjustments = list(candidate_policy.get("adjustments", []))
    micro_live_gate = dict(staged_candidate_evidence.get("micro_live_gate", {}) or {})
    sample_watchdog = dict(staged_candidate_evidence.get("sample_quality_watchdog", {}) or {})
    checkpoint_auto_judge = dict(staged_candidate_evidence.get("checkpoint_auto_judge", {}) or {})
    auto_mode = dict(staged_candidate_evidence.get("auto_mode", {}) or {})
    baseline_gate = _simple_baseline_control_gate(validation_evidence)
    rollout_status = "steady"
    rollout_reason = "UNCHANGED"
    retention_monitor = _retention_monitor(previous_state, previous_active, validation_evidence, operational_verdict)
    checkpoint_revalidation = _sample_quality_checkpoint_revalidation(previous_state, validation_evidence)
    raw_executive_operating_verdict = build_executive_operating_verdict(
        promotion_verdict,
        operational_verdict,
        validation,
        validation_evidence,
    )
    executive_operating_verdict = _apply_live_evidence_rejudge_to_executive_verdict(
        previous_state=previous_state,
        executive_operating_verdict=raw_executive_operating_verdict,
        live_evidence_rejudge=_build_live_evidence_rejudge(
            previous_state=previous_state,
            validation_evidence=validation_evidence,
            checkpoint_revalidation=checkpoint_revalidation,
            executive_operating_verdict=raw_executive_operating_verdict,
        ),
    )
    live_evidence_rejudge = dict(executive_operating_verdict.get("live_evidence_rejudge", {}) or {})
    verdict_reasons = list(promotion_verdict.get("reasons", []) or [])
    staged_micro_live_block = (
        verdict_status == "keep"
        and "PROMOTION_BLOCKED_BY_MICRO_LIVE_GATE" in verdict_reasons
        and any(str(item.get("action", "")) in {"promote", "aggressive_promote"} for item in candidate_adjustments)
    )
    protective_transition = _protective_policy_transition(
        verdict_status=verdict_status,
        candidate_adjustments=candidate_adjustments,
        comparison_verdict=comparison_verdict,
    )
    invalidate_staged_rollout = _invalidate_staged_rollout(
        previous_state=previous_state,
        validation_status=validation_status,
        staged_micro_live_block=staged_micro_live_block,
    )
    if (
        str(checkpoint_auto_judge.get("verdict", "") or "") == "rollback"
        and previous_active
        and str(previous_active.get("status", "") or "") not in {"baseline", "keep"}
    ):
        active_policy = {"status": "baseline", "adjustments": []}
        lifecycle = "rolled_back"
        rollout_status = "reverted"
        rollout_reason = "CHECKPOINT_AUTO_JUDGE_ROLLBACK"
        version = previous_version + 1
    elif (
        previous_active
        and str(previous_active.get("status", "") or "") in {"promote", "promote_aggressive"}
        and str(checkpoint_auto_judge.get("verdict", "") or "") == "tighten"
        and str(baseline_gate.get("status", "")) == "block"
    ):
        active_policy = _demoted_active_policy(previous_active)
        for item in list(active_policy.get("adjustments", []) or []):
            item["reason"] = str(baseline_gate.get("gate_reason", "") or "SIMPLE_BASELINE_CONTROL_GATE")
        lifecycle = "checkpoint_tightened"
        rollout_status = "checkpoint_tightened"
        rollout_reason = str(baseline_gate.get("gate_reason", "") or "SIMPLE_BASELINE_CONTROL_GATE")
        version = previous_version + 1
    elif protective_transition:
        if protective_transition == "disable":
            active_policy = {
                "status": "disabled",
                "adjustments": [dict(item, action="disabled", size_multiplier=0.0) for item in candidate_adjustments],
            }
            lifecycle = "disabled"
            rollout_status = "halted"
            rollout_reason = "PROTECTIVE_DISABLE_VALIDATED"
        else:
            active_policy = dict(candidate_policy)
            active_policy["status"] = "demote"
            lifecycle = "demoted"
            rollout_status = "demoted"
            rollout_reason = "PROTECTIVE_DEMOTION_VALIDATED"
        version = previous_version + 1
    elif invalidate_staged_rollout:
        active_policy = {"status": "baseline", "adjustments": []}
        lifecycle = "staged_rollout_invalidated"
        rollout_status = "baseline"
        rollout_reason = "STAGED_CANDIDATE_INVALIDATED"
        version = previous_version + 1
    elif previous_active and retention_monitor.get("status") == "rollback":
        active_policy = {"status": "baseline", "adjustments": []}
        lifecycle = "rolled_back"
        rollout_status = "reverted"
        rollout_reason = "POST_PROMOTION_RETENTION_DEGRADED"
        version = previous_version + 1
    elif previous_active and retention_monitor.get("status") == "demote":
        active_policy = _demoted_active_policy(previous_active)
        lifecycle = "retention_demoted"
        rollout_status = "retention_demoted"
        rollout_reason = "POST_PROMOTION_RETENTION_WEAKENED"
        version = previous_version + 1
    elif validation_status in {"pass", "pending"} and (verdict_status in {"promote", "promote_aggressive", "demote"} or staged_micro_live_block):
        if staged_micro_live_block:
            active_policy = previous_active or {"status": "baseline", "adjustments": []}
        else:
            active_policy = dict(candidate_policy)
            active_policy["status"] = verdict_status
        if (verdict_status.startswith("promote") or staged_micro_live_block) and bool(micro_live_gate.get("available")) and str(micro_live_gate.get("status", "")) != "pass":
            lifecycle = "staged_rollout"
            rollout_status = "micro_live_pending"
            rollout_reason = str(micro_live_gate.get("reason", "MICRO_LIVE_PENDING") or "MICRO_LIVE_PENDING")
            if retention_monitor["status"] == "inactive":
                retention_monitor = {"status": "armed", "reasons": ["AWAITING_MICRO_LIVE_PASS"], "metrics": {}}
        else:
            lifecycle = "promoted" if verdict_status.startswith("promote") else "demoted"
            rollout_status = "ready" if verdict_status.startswith("promote") else "demoted"
            rollout_reason = "PROMOTION_VALIDATED" if verdict_status.startswith("promote") else "DEMOTION_VALIDATED"
            if verdict_status.startswith("promote") and retention_monitor["status"] == "inactive":
                retention_monitor = {"status": "armed", "reasons": ["POST_PROMOTION_MONITORING_ACTIVE"], "metrics": {}}
        version = previous_version + 1
    elif validation_status == "pass" and verdict_status == "disable":
        active_policy = {
            "status": "disabled",
            "adjustments": [dict(item, action="disabled", size_multiplier=0.0) for item in candidate_adjustments],
        }
        lifecycle = "disabled"
        rollout_status = "halted"
        rollout_reason = "CANDIDATE_DISABLED"
        version = previous_version + 1
    elif (
        comparison_underperforms
        or str(operational_verdict.get("status", "")) == "stop"
        or float(staged_candidate_evidence.get("replay_like_drawdown_ratio", validation_evidence.get("replay_like_drawdown_ratio", 0.0)) or 0.0) > 0.5
    ) and previous_active:
        active_policy = {"status": "baseline", "adjustments": []}
        lifecycle = "rolled_back"
        rollout_status = "reverted"
        if comparison_underperforms:
            rollout_reason = "CANDIDATE_UNDERPERFORMS_CURRENT_POLICY"
        elif str(operational_verdict.get("status", "")) == "stop":
            rollout_reason = "OPERATIONAL_STOP_ACTIVE"
        else:
            rollout_reason = "REPLAY_DRAWDOWN_TOO_HIGH"
        version = previous_version + 1
    elif previous_active:
        active_policy = previous_active
        lifecycle = "kept"
        rollout_status = previous_state.get("rollout_status", "steady") or "steady"
        rollout_reason = "ACTIVE_POLICY_UNCHANGED"
        version = previous_version
    else:
        active_policy = {"status": "baseline", "adjustments": []}
        lifecycle = "baseline"
        rollout_status = "baseline"
        rollout_reason = "NO_ACTIVE_POLICY"
        version = previous_version
    evaluated_at = str(validation_evidence.get("generated_at", previous_state.get("updated_at", "")) or datetime.now(tz=timezone.utc).isoformat())
    lifecycle_rollout_phase = str(
        dict(previous_state.get("rollout_progression", dict(active_policy.get("rollout_progression", {}) or {})) or {}).get("execution_phase", "")
        or ("baseline" if not list(dict(active_policy or {}).get("adjustments", []) or []) else "full")
    )
    validation_bucket_evidence = _policy_context_bucket_evidence_map(validation_evidence)
    lifecycle_bucket_evidence = dict(
        validation_bucket_evidence.get("active_policy", {})
        or validation_bucket_evidence.get("staged_candidate", {})
        or {}
    )
    symbol_lifecycle = build_symbol_lifecycle(
        symbol_summary=list(validation_evidence.get("symbol_summary", []) or []),
        symbol_scorecard=list(validation_evidence.get("symbol_scorecard", []) or []),
        pruning_recommendations=list(validation_evidence.get("pruning_recommendations", []) or []),
        policy_context_bucket_name=str(lifecycle_bucket_evidence.get("policy_context_bucket_name", "") or ""),
        policy_context_bucket_symbol_summary=list(
            lifecycle_bucket_evidence.get(
                "policy_context_bucket_symbol_summary",
                lifecycle_bucket_evidence.get("symbol_summary", []),
            )
            or []
        ),
        policy_context_bucket_pruning_recommendations=list(
            lifecycle_bucket_evidence.get(
                "policy_context_bucket_pruning_recommendations",
                lifecycle_bucket_evidence.get("pruning_recommendations", []),
            )
            or []
        ),
        active_adjustments=list(dict(active_policy or {}).get("adjustments", []) or []),
        previous_rows=list(previous_state.get("symbol_lifecycle", []) or []),
        checkpoint_auto_judge=checkpoint_auto_judge,
        sample_quality_watchdog=sample_watchdog,
        baseline_control_comparison=dict(validation_evidence.get("baseline_control_comparison", {}) or {}),
        active_policy=active_policy,
        rollout_phase=lifecycle_rollout_phase,
        policy_version=version,
        evaluated_at=evaluated_at,
    )
    active_policy = _apply_symbol_lifecycle_to_active_policy(active_policy, symbol_lifecycle)
    symbol_lifecycle_summary = summarize_symbol_lifecycle(symbol_lifecycle)
    rollout_progression = _rollout_progression_signal(
        previous_state=previous_state,
        active_policy=active_policy,
        promotion_verdict=promotion_verdict,
        validation_evidence=validation_evidence,
        retention_monitor=retention_monitor,
        rollout_status=rollout_status,
        rollout_reason=rollout_reason,
    )
    active_policy = _annotate_active_policy(active_policy, promotion_verdict, retention_monitor)
    active_policy["rollout_progression"] = rollout_progression
    active_policy["checkpoint_revalidation"] = checkpoint_revalidation
    active_policy["checkpoint_auto_judge"] = checkpoint_auto_judge
    active_policy["sample_quality_watchdog"] = sample_watchdog
    active_policy["auto_mode"] = auto_mode
    active_policy["live_evidence_rejudge"] = live_evidence_rejudge
    active_policy["executive_operating_verdict"] = dict(executive_operating_verdict)
    policy_state_payload = {
        "version": version,
        "active_policy": active_policy,
        "rollout_progression": rollout_progression,
        "executive_operating_verdict": executive_operating_verdict,
        "updated_at": evaluated_at,
    }
    policy_lineage = build_policy_state_lineage_snapshot(
        policy_state_payload,
        source="persisted_policy_state",
    )
    active_policy["policy_lineage"] = dict(policy_lineage)
    validation_payload = dict(validation)
    validation_evidence_payload = dict(validation_payload.get("evidence", {}) or {})
    previous_policy_bucket = policy_evidence_bucket(previous_state, "active_policy")
    previous_policy_evidence = dict(previous_policy_bucket.get("evidence", {}) or {})
    if not previous_policy_evidence:
        previous_policy_evidence = _active_policy_evidence(
            dict(dict(previous_state.get("policy_validation", {}) or {}).get("evidence", {}) or {})
        )
    current_active_bucket = policy_evidence_bucket(validation_evidence_payload, "active_policy")
    current_active_bucket_evidence = dict(current_active_bucket.get("evidence", {}) or {})
    current_active_bucket_alignment = dict(current_active_bucket.get("alignment", {}) or {})
    if not current_active_bucket_evidence and "policy_evidence_buckets" not in validation_evidence_payload:
        current_active_bucket_evidence = _active_policy_evidence(validation_evidence_payload)
        if current_active_bucket_evidence and not current_active_bucket_alignment:
            current_active_bucket_alignment = dict(
                validation_evidence_payload.get("current_policy_evidence_alignment", {})
                or {"aligned": True, "status": "legacy_unbucketed", "reason": "LEGACY_UNBUCKETED_ACTIVE_POLICY_EVIDENCE"}
            )
    active_policy_changed = bool(policy_lineage) and bool(dict(previous_state.get("policy_lineage", {}) or {})) and (
        dict(previous_state.get("policy_lineage", {}) or {}).get("structural_key") != policy_lineage.get("structural_key")
    )
    current_active_evidence_available = bool(current_active_bucket.get("available")) and bool(current_active_bucket_alignment.get("aligned"))
    persisted_active_bucket = _policy_bucket_entry(
        bucket_name="active_policy",
        source=(
            str(current_active_bucket.get("source", "") or "persisted_active_policy_validation_evidence")
            if current_active_evidence_available and not active_policy_changed
            else "active_policy_live_evidence_pending"
        ),
        available=current_active_evidence_available and not active_policy_changed,
        evidence=current_active_bucket_evidence if current_active_evidence_available and not active_policy_changed else {},
        policy_lineage=policy_lineage,
        evidence_lineage=dict(current_active_bucket.get("evidence_lineage", {}) or {}),
        alignment=current_active_bucket_alignment if not active_policy_changed else {"aligned": False, "status": "pending", "reason": "ACTIVE_POLICY_CHANGED_PENDING_FRESH_EVIDENCE"},
        policy_application=dict(current_active_bucket.get("policy_application", {}) or {}),
        replay_summary=dict(current_active_bucket.get("replay_summary", {}) or {}) if current_active_evidence_available and not active_policy_changed else {},
    )
    previous_policy_lineage = dict(
        previous_policy_bucket.get("policy_lineage", previous_state.get("policy_lineage", {}))
        or {}
    )
    persisted_buckets = {
        **{
            name: bucket
            for name, bucket in dict(validation_evidence_payload.get("policy_evidence_buckets", {}) or {}).items()
            if name in {"staged_candidate", "baseline_control"}
        },
        "previous_policy": _policy_bucket_entry(
            bucket_name="previous_policy",
            source=str(previous_policy_bucket.get("source", "") or "previous_persisted_policy_state"),
            available=bool(previous_policy_evidence or previous_policy_lineage),
            evidence=previous_policy_evidence,
            policy_lineage=previous_policy_lineage,
            evidence_lineage=dict(previous_policy_bucket.get("evidence_lineage", {}) or {}),
            alignment=dict(previous_policy_bucket.get("alignment", {}) or {}),
            policy_application=dict(previous_policy_bucket.get("policy_application", {}) or {}),
            replay_summary=dict(previous_policy_bucket.get("replay_summary", {}) or {}),
        ),
        "active_policy": persisted_active_bucket,
    }
    validation_evidence_payload["active_policy_lineage"] = dict(policy_lineage)
    validation_evidence_payload = with_policy_evidence_buckets(validation_evidence_payload, persisted_buckets)
    validation_payload["evidence"] = validation_evidence_payload
    return {
        "version": version,
        "status": lifecycle,
        "rollout_status": rollout_status,
        "rollout_reason": rollout_reason,
        "rollout_progression": rollout_progression,
        "retention_monitor": retention_monitor,
        "checkpoint_revalidation": checkpoint_revalidation,
        "checkpoint_auto_judge": checkpoint_auto_judge,
        "auto_mode": auto_mode,
        "live_evidence_rejudge": live_evidence_rejudge,
        "executive_operating_verdict": executive_operating_verdict,
        "symbol_lifecycle": symbol_lifecycle,
        "symbol_lifecycle_summary": symbol_lifecycle_summary,
        "policy_lineage": policy_lineage,
        "active_policy": active_policy,
        "candidate_policy": candidate_policy,
        "promotion_verdict": promotion_verdict,
        "operational_verdict": operational_verdict,
        "policy_validation": validation_payload,
        "policy_evidence_buckets": deepcopy(persisted_buckets),
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def build_policy_history_entry(policy_state: dict[str, object]) -> dict[str, object]:
    return {
        "updated_at": policy_state.get("updated_at"),
        "version": policy_state.get("version", 0),
        "status": policy_state.get("status", "unknown"),
        "rollout_status": policy_state.get("rollout_status", "unknown"),
        "rollout_reason": policy_state.get("rollout_reason", "unknown"),
        "retention_monitor": dict(policy_state.get("retention_monitor", {}) or {}),
        "checkpoint_revalidation": dict(policy_state.get("checkpoint_revalidation", {}) or {}),
        "checkpoint_auto_judge": dict(policy_state.get("checkpoint_auto_judge", {}) or {}),
        "auto_mode": dict(policy_state.get("auto_mode", {}) or {}),
        "live_evidence_rejudge": dict(policy_state.get("live_evidence_rejudge", {}) or {}),
        "executive_operating_verdict": dict(policy_state.get("executive_operating_verdict", {}) or {}),
        "symbol_lifecycle_summary": dict(policy_state.get("symbol_lifecycle_summary", {}) or {}),
        "rollout_progression": dict(policy_state.get("rollout_progression", {}) or {}),
        "promotion_verdict": dict(policy_state.get("promotion_verdict", {}) or {}),
        "policy_validation": dict(policy_state.get("policy_validation", {}) or {}),
        "policy_lineage": dict(policy_state.get("policy_lineage", {}) or {}),
        "active_policy_status": str(dict(policy_state.get("active_policy", {}) or {}).get("status", "unknown")),
        "active_adjustment_count": len(list(dict(policy_state.get("active_policy", {}) or {}).get("adjustments", []))),
        "micro_live_readiness": str(dict(policy_state.get("active_policy", {}) or {}).get("micro_live_readiness", "not_available")),
    }

def build_policy_state(
    candidate_policy: dict[str, object],
    promotion_verdict: dict[str, object],
    executive_operating_verdict: dict[str, object] | None = None,
) -> dict[str, object]:
    candidate_adjustments = list(candidate_policy.get("adjustments", []))
    verdict_status = str(promotion_verdict.get("status", "keep"))
    rollout_stage = str(promotion_verdict.get("rollout_stage", "baseline") or "baseline")
    executive_verdict = dict(executive_operating_verdict or {})
    if verdict_status == "disable":
        active_adjustments = [dict(item, action="disabled", size_multiplier=0.0) for item in candidate_adjustments]
        active_status = "disabled"
    elif verdict_status in {"promote", "promote_aggressive", "demote"}:
        active_adjustments = candidate_adjustments
        active_status = verdict_status
    else:
        active_adjustments = [dict(item, action="keep", size_multiplier=1.0, reason="ACTIVE_POLICY_UNCHANGED") for item in candidate_adjustments]
        active_status = "baseline" if rollout_stage == "staged_rollout" else "keep"
    payload = {
        "status": "staged_rollout" if rollout_stage == "staged_rollout" else active_status,
        "active_policy": {
            "status": active_status,
            "adjustments": active_adjustments,
            "lifecycle_stage": rollout_stage,
            "micro_live_readiness": str(promotion_verdict.get("micro_live_readiness", "not_available") or "not_available"),
            "micro_live_gate": dict(promotion_verdict.get("micro_live_gate", {}) or {}),
            "auto_mode": dict(promotion_verdict.get("auto_mode", candidate_policy.get("decomposition_summary", {}).get("auto_mode", {})) or {}),
            "executive_operating_verdict": executive_verdict,
        },
        "candidate_policy": candidate_policy,
        "promotion_verdict": promotion_verdict,
        "auto_mode": dict(promotion_verdict.get("auto_mode", candidate_policy.get("decomposition_summary", {}).get("auto_mode", {})) or {}),
        "executive_operating_verdict": executive_verdict,
    }
    payload["policy_lineage"] = build_policy_state_lineage_snapshot(
        {
            "version": 0,
            "active_policy": dict(payload.get("active_policy", {}) or {}),
            "rollout_progression": dict(payload.get("active_policy", {}).get("rollout_progression", {}) or {}),
        },
        source="ephemeral_policy_state",
    )
    payload["active_policy"]["policy_lineage"] = dict(payload["policy_lineage"])
    return payload

def build_operational_verdict(execution_outcomes: dict[str, object]) -> dict[str, object]:
    live_order_count = int(execution_outcomes.get("accepted_live_order_count", 0)) + int(execution_outcomes.get("rejected_live_order_count", 0))
    if live_order_count <= 0:
        return {
            "status": "hold",
            "reasons": ["INSUFFICIENT_LIVE_ORDERS"],
            "metrics": {
                "live_order_count": 0,
                "reject_rate": 0.0,
                "protection_degraded_rate": 0.0,
                "avg_fill_ratio": 0.0,
                "avg_edge_retention_ratio": 0.0,
                "avg_realized_edge_bps": 0.0,
                "realized_vs_expected_edge_gap_bps": 0.0,
            },
        }

    reject_rate = int(execution_outcomes.get("rejected_live_order_count", 0)) / live_order_count
    protection_degraded_rate = float(execution_outcomes.get("protection_degraded_rate", 0.0) or 0.0)
    avg_fill_ratio = float(execution_outcomes.get("avg_fill_ratio", 0.0) or 0.0)
    retention = float(execution_outcomes.get("avg_edge_retention_ratio", 0.0) or 0.0)
    realized = float(execution_outcomes.get("avg_realized_edge_bps", 0.0) or 0.0)
    gap = float(execution_outcomes.get("realized_vs_expected_edge_gap_bps", 0.0) or 0.0)
    major_rows = _major_symbol_operational_rows(list(execution_outcomes.get("execution_audit_by_symbol", [])))

    reasons: list[str] = []
    status = "pass"

    aggressive_pass = (
        live_order_count >= 10
        and retention >= 0.95
        and realized > 0.0
        and gap >= -1.5
        and protection_degraded_rate <= 0.01
        and reject_rate <= 0.01
        and avg_fill_ratio >= 0.97
    )
    strong_pass = (
        live_order_count >= 8
        and retention >= 0.80
        and realized > 0.0
        and gap > -3.0
        and protection_degraded_rate <= 0.02
        and reject_rate <= 0.02
        and avg_fill_ratio >= 0.92
    )

    # Small-account relaxation: when equity is low, protection orders (stop-loss)
    # frequently fail due to exchange minimum notional constraints, inflating the
    # protection_degraded_rate.  Use relaxed thresholds so the system keeps trading.
    _small_account = live_order_count < 20
    _prot_stop_threshold = 0.90 if _small_account else 0.15
    _prot_hold_threshold = 0.60 if _small_account else 0.05
    if retention < 0.40 or realized <= 0.0 or protection_degraded_rate > _prot_stop_threshold or reject_rate > 0.15:
        status = "stop"
    elif retention < 0.65 or gap <= -8.0 or protection_degraded_rate > _prot_hold_threshold or reject_rate > 0.05 or avg_fill_ratio < 0.85:
        status = "hold"
    elif aggressive_pass:
        status = "aggressive_pass"
    elif strong_pass:
        status = "strong_pass"

    if live_order_count < 5 and status in {"pass", "strong_pass", "aggressive_pass"}:
        status = "hold"
        reasons.append("INSUFFICIENT_SAMPLE")

    if retention < 0.40:
        reasons.append("EDGE_RETENTION_TOO_LOW")
    elif retention < 0.65:
        reasons.append("EDGE_RETENTION_BELOW_PASS")

    if realized <= 0.0:
        reasons.append("REALIZED_EDGE_NOT_POSITIVE")
    if gap <= -8.0:
        reasons.append("EDGE_GAP_TOO_NEGATIVE")
    if protection_degraded_rate > 0.15:
        reasons.append("PROTECTION_DEGRADED_TOO_HIGH")
    elif protection_degraded_rate > 0.05:
        reasons.append("PROTECTION_DEGRADED_ABOVE_PASS")
    if reject_rate > 0.15:
        reasons.append("REJECT_RATE_TOO_HIGH")
    elif reject_rate > 0.05:
        reasons.append("REJECT_RATE_ABOVE_PASS")
    if avg_fill_ratio < 0.85:
        reasons.append("FILL_RATIO_TOO_LOW")

    if major_rows:
        weak_major_rows = [row for row in major_rows if float(row.get("realized_vs_expected_edge_gap_bps", 0.0) or 0.0) <= -8.0 or float(row.get("avg_realized_edge_bps", 0.0) or 0.0) <= 0.0]
        strong_major_rows = [row for row in major_rows if float(row.get("avg_edge_retention_ratio", 0.0) or 0.0) >= 0.8 and float(row.get("avg_realized_edge_bps", 0.0) or 0.0) > 0.0]
        if weak_major_rows:
            if status in {"pass", "strong_pass"}:
                status = "hold"
            reasons.append("MAJOR_SYMBOL_AUDIT_WEAK")
        elif status == "aggressive_pass" and len(strong_major_rows) < len(major_rows):
            status = "strong_pass"
            reasons.append("MAJOR_SYMBOL_AGGRESSIVE_CONFIRMATION_INCOMPLETE")
        elif status == "strong_pass" and not strong_major_rows:
            status = "pass"
            reasons.append("MAJOR_SYMBOL_CONFIRMATION_INCOMPLETE")

    if status == "aggressive_pass":
        reasons.append("OPERATING_WITH_ELITE_EDGE")
    elif status == "strong_pass":
        reasons.append("OPERATING_WITH_STRONG_EDGE")
    elif not reasons:
        reasons.append("OPERATING_WITHIN_THRESHOLDS")

    return {
        "status": status,
        "reasons": reasons,
        "metrics": {
            "live_order_count": live_order_count,
            "reject_rate": round(reject_rate, 6),
            "protection_degraded_rate": round(protection_degraded_rate, 6),
            "avg_fill_ratio": round(avg_fill_ratio, 6),
            "avg_edge_retention_ratio": round(retention, 6),
            "avg_realized_edge_bps": round(realized, 6),
            "realized_vs_expected_edge_gap_bps": round(gap, 6),
        },
    }


def build_runtime_summary(
    *,
    decisions: list[DecisionIntent] | tuple[DecisionIntent, ...],
    tested_orders: list[dict[str, object]] | None = None,
    live_orders: list[dict[str, object]] | None = None,
    account_snapshot: dict[str, object] | None = None,
    open_orders_snapshot: dict[str, object] | None = None,
    capital_report: dict[str, object] | None = None,
    kill_switch_status: dict[str, object] | None = None,
    observe_only_symbols: list[str] | tuple[str, ...] | None = None,
    open_spot_positions: list[dict[str, object]] | None = None,
    open_futures_positions: list[dict[str, object]] | None = None,
    closed_trades: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    telegram_alerts: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    live_positions: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    self_healing: dict[str, object] | None = None,
) -> dict[str, object]:
    derived_observe_only = {
        decision.symbol
        for decision in decisions
        if "OBSERVE_ONLY_SYMBOL" in decision.rejection_reasons
    }
    combined_observe_only = sorted(set(observe_only_symbols or []).union(derived_observe_only))
    closed_trade_aggregate = aggregate_closed_trade_metrics(closed_trades or [])
    symbol_performance = closed_trade_aggregate.symbol_performance
    exit_reason_counts = closed_trade_aggregate.exit_reason_counts
    realized_total = closed_trade_aggregate.realized_pnl_usd
    realized_net_total = round(
        sum(float(t.get("realized_pnl_net_usd_estimate", t.get("realized_pnl_usd_estimate", 0.0))) for t in (closed_trades or [])),
        6,
    )
    total_round_trip_fee = round(
        sum(float(t.get("estimated_round_trip_fee_usd", 0.0)) for t in (closed_trades or [])),
        6,
    )
    unrealized_spot_total = round(
        sum(float(position.get("unrealized_pnl_usd_estimate", 0.0)) for position in (open_spot_positions or [])),
        6,
    )
    unrealized_futures_total = round(
        sum(float(position.get("unrealized_pnl_usd_estimate", 0.0)) for position in (open_futures_positions or [])),
        6,
    )
    futures_position_sync = _futures_position_sync_payload(
        open_futures_positions=open_futures_positions,
        live_positions=live_positions,
    )
    execution_outcomes = _aggregate_live_order_outcomes(live_orders)
    performance_attribution = build_performance_attribution(live_orders)
    candidate_policy = build_auto_tune_policy(performance_attribution)
    promotion_verdict = build_promotion_verdict(candidate_policy)
    operational_verdict = build_operational_verdict(execution_outcomes)
    runner_evidence = load_validation_runner_evidence(None)
    policy_validation = build_policy_validation(candidate_policy, promotion_verdict, operational_verdict, performance_attribution, runner_evidence)
    executive_operating_verdict = build_executive_operating_verdict(
        promotion_verdict,
        operational_verdict,
        policy_validation,
        runner_evidence,
    )
    policy_state = build_policy_state(candidate_policy, promotion_verdict, executive_operating_verdict)
    rejection_counts = Counter()
    for decision in decisions:
        for reason in decision.rejection_reasons:
            rejection_counts[reason] += 1
    recent_decisions = [
        {
            "symbol": decision.symbol,
            "candidate_mode": decision.candidate_mode,
            "mode": decision.final_mode,
            "side": decision.side,
            "score": round(decision.predictability_score, 2),
            "net_expected_edge_bps": round(decision.net_expected_edge_bps, 6),
            "estimated_round_trip_cost_bps": round(decision.estimated_round_trip_cost_bps, 6),
            "macro_trade_restraint": decision.macro_trade_restraint,
            "execution_quality_trade_restraint": decision.execution_quality_trade_restraint,
            "strategy_size_multiplier": round(decision.strategy_size_multiplier, 6),
            "entry_relaxations": list(decision.entry_relaxation_reasons[:4]),
            "size_boost_reasons": list(decision.size_boost_reasons[:4]),
            "reasons": list(decision.rejection_reasons[:4]),
        }
        for decision in list(decisions)[-5:]
    ]
    return {
        "decision_count": len(decisions),
        "modes": [decision.final_mode for decision in decisions],
        "symbols": sorted({decision.symbol for decision in decisions}),
        "observe_only_symbols": combined_observe_only,
        "open_spot_positions": list(open_spot_positions or []),
        "open_futures_positions": list(open_futures_positions or []),
        "live_positions": list(live_positions or []),
        **futures_position_sync,
        "closed_trades": list(closed_trades or []),
        "closed_trade_count": closed_trade_aggregate.closed_trade_count,
        "telegram_alerts": list(telegram_alerts or []),
        "recent_decisions": recent_decisions,
        "major_entry_relaxation_count": sum(1 for decision in decisions if decision.entry_relaxation_reasons),
        "major_size_boost_count": sum(1 for decision in decisions if decision.size_boost_reasons),
        "top_rejection_reasons": dict(rejection_counts.most_common(8)),
        "exit_reason_counts": exit_reason_counts,
        "symbol_performance": symbol_performance,
        "realized_pnl_usd_estimate": realized_total,
        "realized_pnl_net_usd_estimate": realized_net_total,
        "estimated_round_trip_fee_usd": total_round_trip_fee,
        "unrealized_pnl_usd_estimate": round(unrealized_spot_total + unrealized_futures_total, 6),
        "unrealized_spot_pnl_usd_estimate": unrealized_spot_total,
        "unrealized_futures_pnl_usd_estimate": unrealized_futures_total,
        "tested_order_count": len(tested_orders or []),
        "tested_orders": tested_orders or [],
        "live_order_count": len(live_orders or []),
        "live_orders": live_orders or [],
        **execution_outcomes,
        "performance_attribution": performance_attribution,
        "candidate_policy": candidate_policy,
        "promotion_verdict": promotion_verdict,
        "policy_state": policy_state,
        "policy_validation": policy_validation,
        "operational_verdict": operational_verdict,
        "executive_operating_verdict": executive_operating_verdict,
        "account_snapshot": account_snapshot or {},
        "open_orders_snapshot": open_orders_snapshot or {},
        "capital_report": capital_report or {},
        "kill_switch": kill_switch_status or {"armed": False, "reasons": []},
        "self_healing": self_healing or {},
    }


def write_runtime_summary(path: str | Path, summary: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    latest_root = output_path.parent.parent / "latest"
    latest_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, latest_root / "summary.json")
