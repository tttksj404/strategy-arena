# Verifies holdout-only trade sampling and fail-closed registry reporting.

from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
from pathlib import Path
import tempfile

from research.wave1.common import load_json, save_json
from research.wave1.gate_reporting import evaluate_result_file


def _records(series: pd.Series) -> list[dict[str, str | float]]:
    return [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), "value": float(value)}
        for timestamp, value in series.items()
    ]


def test_oos_gates_exclude_in_sample_trades() -> None:
    # Given
    equity_index = pd.to_datetime(["2025-09-29", "2025-09-30", "2025-10-01", "2025-10-02"], utc=True)
    equity = pd.Series([300.0, 303.0, 306.0, 309.0], index=equity_index)
    trade_index = pd.date_range("2025-09-20", periods=20, freq="6h", tz="UTC")
    trades = pd.Series([0.01 if offset % 2 == 0 else 0.02 for offset in range(20)], index=trade_index)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        path = Path(directory) / "F1a.json"
        save_json(path, {
            "candidate_id": "F1a",
            "family": "F1",
            "equity": _records(equity),
            "trade_returns": _records(trades),
            "positions": _records(pd.Series(1.0, index=equity_index)),
            "turnover": _records(pd.Series([1.0, 0.0, 0.0, 1.0], index=equity_index)),
            "stress_total_return": 0.01,
            "metadata": {
                "data_valid": True,
                "cost_model_valid": True,
                "intended_factor": "funding_carry",
                "neighbor_is_sharpes": [1.0, 1.1],
                "max_concurrent_positions": 2,
                "max_position_weight": 0.5,
                "min_position_weight": 0.5,
                "min_order_usdt": 5.0,
            },
        })

        # When
        rows = evaluate_result_file(path, pd.Series(dtype=float))
        payload = load_json(path)

        # Then
        gate_five = next(row for row in rows if row.gate == 5)
        assert gate_five.status == "UNDETERMINED"
        assert isinstance(payload, dict)
        assert payload["validation"]["oos_trade_count"] == 0
