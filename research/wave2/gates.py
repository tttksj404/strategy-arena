from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import JsonValue, load_json, save_json
from research.wave1.gate_reporting import _series
from research.wave1.gates import GateRow
from research.wave1.gate_reporting import evaluate_result_file


OOS_SPLIT = pd.Timestamp("2025-09-30T23:59:59Z")
OOS_DEPENDENT_GATES = frozenset({2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19})


def _has_oos_position(payload: dict[str, JsonValue]) -> bool:
    positions = _series(payload.get("positions"))
    return bool((positions[positions.index > OOS_SPLIT].abs() > 0.0).any())


def evaluate_result_file_wave2(path: Path, btc_returns: pd.Series) -> tuple[GateRow, ...]:
    rows = evaluate_result_file(path, btc_returns)
    payload = load_json(path)
    if not isinstance(payload, dict):
        return rows
    validation = payload.get("validation")
    metrics = payload.get("metrics")
    is_metrics = metrics.get("is") if isinstance(metrics, dict) else None
    is_return = is_metrics.get("total_ret") if isinstance(is_metrics, dict) else None
    validation_payload = validation if isinstance(validation, dict) else {}
    if isinstance(is_return, (int, float)):
        validation_payload = {
            **validation_payload,
            "gate4_is_after_cost": {
                "status": "PASS" if is_return > 0.0 else "FAIL",
                "value": float(is_return),
            },
        }
    oos_trade_count = validation.get("oos_trade_count") if isinstance(validation, dict) else None
    if oos_trade_count == 0 and not _has_oos_position(payload):
        rows = tuple(
            replace(row, status="UNTESTED_IN_OOS") if row.gate in OOS_DEPENDENT_GATES else row
            for row in rows
        )
        payload["validation"] = {**validation_payload, "oos_label": "UNTESTED_IN_OOS"}
        payload["gates"] = [
            {"gate": row.gate, "name": row.name, "status": row.status, "value": row.value}
            for row in rows
        ]
        save_json(path, payload)
    elif validation_payload != validation:
        payload["validation"] = validation_payload
        save_json(path, payload)
    return rows
