from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_runtime_overview(*, summary: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    capital_report = summary.get("capital_report") or state.get("capital_report") or {}
    self_healing = summary.get("self_healing") or state.get("self_healing") or {}
    exchange_positions = summary.get("exchange_live_futures_positions") or state.get("exchange_live_futures_positions") or []
    active_positions = []
    for position in exchange_positions:
        total = float(position.get("total") or position.get("available") or 0.0)
        if total <= 0.0:
            continue
        margin = float(position.get("marginSize") or 0.0)
        unrealized = float(position.get("unrealizedPL") or 0.0)
        roe = (unrealized / margin * 100.0) if margin > 0 else 0.0
        active_positions.append(
            {
                "symbol": position.get("symbol"),
                "side": position.get("holdSide") or position.get("posSide"),
                "unrealized_pnl_usd": round(unrealized, 6),
                "roe_percent": round(roe, 4),
            }
        )
    return {
        "updated_at": state.get("updated_at"),
        "status": self_healing.get("status") or "unknown",
        "decision_count": state.get("decision_count", summary.get("decision_count")),
        "heartbeat_count": state.get("heartbeat_count"),
        "last_event_timestamp": state.get("last_event_timestamp"),
        "last_decision_timestamp": state.get("last_decision_timestamp"),
        "last_decision_emitted_at": state.get("last_decision_emitted_at"),
        "live_order_count": summary.get("live_order_count", state.get("live_order_count")),
        "tested_order_count": summary.get("tested_order_count", state.get("tested_order_count")),
        "realized_pnl_usd_estimate": summary.get("realized_pnl_usd_estimate", 0.0),
        "unrealized_pnl_usd_estimate": summary.get("unrealized_pnl_usd_estimate", 0.0),
        "kill_switch": summary.get("kill_switch") or state.get("kill_switch") or {"armed": False, "reasons": []},
        "reference_price_guard": summary.get("reference_price_guard") or state.get("reference_price_guard") or {},
        "top_rejection_reasons": summary.get("top_rejection_reasons") or {},
        "recent_decisions": summary.get("recent_decisions") or [],
        "exchange_live_futures_position_count": summary.get(
            "exchange_live_futures_position_count",
            state.get("exchange_live_futures_position_count", len(active_positions)),
        ),
        "exchange_live_futures_positions": active_positions,
        "futures_available_balance_usd": capital_report.get("futures_available_balance_usd"),
        "futures_execution_balance_usd": capital_report.get("futures_execution_balance_usd"),
        "futures_transferable_execution_balance_usd": capital_report.get("futures_transferable_execution_balance_usd"),
        "futures_collateral_candidate_balance_usd": capital_report.get("futures_collateral_candidate_balance_usd"),
        "futures_manual_handling_balance_usd": capital_report.get("futures_manual_handling_balance_usd"),
        "futures_total_reusable_balance_usd": capital_report.get("futures_total_reusable_balance_usd"),
        "spot_available_balance_usd": capital_report.get("spot_available_balance_usd"),
    }


def write_runtime_overview(path: str | Path, overview: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(overview), indent=2, sort_keys=True), encoding="utf-8")
    latest_root = output_path.parent.parent / "latest"
    latest_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, latest_root / "overview.json")
