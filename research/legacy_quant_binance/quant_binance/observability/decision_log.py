from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from quant_binance.models import DecisionIntent, StrategyPrediction


def hash_decision_payload(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def serialize_decision(decision: DecisionIntent) -> dict[str, Any]:
    data = asdict(decision)
    data["timestamp"] = decision.timestamp.isoformat()
    return data


def serialize_prediction(prediction: StrategyPrediction) -> dict[str, Any]:
    data = asdict(prediction)
    data["timestamp"] = prediction.timestamp.isoformat()
    return data


def render_prediction_report(prediction: StrategyPrediction) -> str:
    data = serialize_prediction(prediction)
    futures = data["futures"]
    spot = data["spot"]
    lines = [
        f"# Strategy Prediction: {prediction.prediction_id}",
        "",
        f"- timestamp: {data['timestamp']}",
        f"- symbol: {prediction.symbol}",
        f"- candidate_mode: {prediction.candidate_mode}",
        f"- selected_mode_hint: {prediction.selected_mode_hint}",
        f"- futures_net_expected_edge_bps: {futures['net_expected_edge_bps']}",
        f"- futures_predictability_score: {futures['predictability_score']}",
        f"- spot_net_expected_edge_bps: {spot['net_expected_edge_bps']}",
        f"- spot_predictability_score: {spot['predictability_score']}",
    ]
    return "\n".join(lines)




def serialize_execution_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    payload = dict(outcome)
    timestamp = payload.get("timestamp")
    if hasattr(timestamp, "isoformat"):
        payload["timestamp"] = timestamp.isoformat()
    return payload


def render_outcome_audit_report(outcome: dict[str, Any]) -> str:
    data = serialize_execution_outcome(outcome)
    expected_edge = float(data.get("expected_net_edge_bps", data.get("net_expected_edge_bps", 0.0)) or 0.0)
    realized_edge = float(data.get("realized_edge_bps", 0.0) or 0.0)
    lines = [
        f"# Execution Outcome Audit: {data.get('order_id') or data.get('client_oid') or data.get('symbol', 'unknown')}",
        "",
        f"- timestamp: {data.get('timestamp', '')}",
        f"- symbol: {data.get('symbol', '')}",
        f"- market: {data.get('market', '')}",
        f"- side: {data.get('side', '')}",
        f"- accepted: {data.get('accepted', False)}",
        f"- fill_ratio: {data.get('fill_ratio', 0.0)}",
        f"- fill_status: {data.get('fill_status', '')}",
        f"- avg_fill_price: {data.get('avg_fill_price', 0.0)}",
        f"- slippage_bps: {data.get('slippage_bps', 0.0)}",
        f"- expected_net_edge_bps: {expected_edge}",
        f"- realized_edge_bps: {realized_edge}",
        f"- edge_retention_ratio: {round((realized_edge / expected_edge), 6) if expected_edge > 0 else 0.0}",
        f"- protection_degraded: {bool(data.get('protection_error'))}",
        f"- realized_vs_expected_edge_gap_bps: {round(realized_edge - expected_edge, 6)}",
    ]
    return "\n".join(lines)

def render_audit_report(decision: DecisionIntent) -> str:
    data = serialize_decision(decision)
    lines = [
        f"# Decision Audit: {decision.decision_id}",
        "",
        f"- timestamp: {data['timestamp']}",
        f"- symbol: {decision.symbol}",
        f"- candidate_mode: {decision.candidate_mode}",
        f"- final_mode: {decision.final_mode}",
        f"- side: {decision.side}",
        f"- decision_hash: {decision.decision_hash}",
        f"- predictability_score: {decision.predictability_score}",
        f"- gross_expected_edge_bps: {decision.gross_expected_edge_bps}",
        f"- net_expected_edge_bps: {decision.net_expected_edge_bps}",
        f"- estimated_round_trip_cost_bps: {decision.estimated_round_trip_cost_bps}",
        f"- rejection_reasons: {list(decision.rejection_reasons)}",
        f"- linked_order_ids: {list(decision.linked_order_ids)}",
        f"- exit_reason_code: {decision.exit_reason_code}",
        f"- divergence_code: {decision.divergence_code}",
    ]
    return "\n".join(lines)
