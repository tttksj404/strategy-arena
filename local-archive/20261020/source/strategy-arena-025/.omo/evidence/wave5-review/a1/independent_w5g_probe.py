from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.wave5.engine import basis_round_trip_cost, combine_returns, funding_capitulation_position, zscore_hysteresis_position


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "research" / "wave5" / "results"


def as_series(payload: dict[str, object], key: str) -> pd.Series:
    rows = payload[key]
    assert isinstance(rows, list)
    return pd.Series({pd.Timestamp(row["timestamp"]): float(row["value"]) for row in rows}).sort_index()


def main() -> None:
    z = pd.Series([0.0, -2.1, -1.8, -0.4, 2.6, 2.0, 0.2])
    assert zscore_hysteresis_position(z, 2.0, 0.5).tolist() == [0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0]

    assert abs(basis_round_trip_cost(1000.0, 0.0001) - 3.6) < 1e-12

    funding = pd.Series([0.0, -0.0006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert funding_capitulation_position(funding, -0.0005, 6).tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    baseline = pd.Series([0.10, 0.0, -0.02])
    candidate = pd.Series([0.0, 0.10, -0.02])
    assert combine_returns(baseline, candidate, 0.5).tolist() == [0.05, 0.05, -0.02]

    w2c = json.loads((RESULTS / "W2c.json").read_text(encoding="utf-8"))
    w5a = json.loads((RESULTS / "W5a.json").read_text(encoding="utf-8"))
    w5g = json.loads((RESULTS / "W5g.json").read_text(encoding="utf-8"))
    base_equity = as_series(w2c, "equity")
    candidate_equity = as_series(w5a, "equity")
    actual_combined = as_series(w5g, "equity")
    base_returns = base_equity.pct_change().fillna(0.0)
    candidate_returns = candidate_equity.pct_change().reindex(base_equity.index).fillna(0.0)
    expected_combined = 300.0 * (1.0 + 0.5 * base_returns + 0.5 * candidate_returns).cumprod()
    max_abs_delta = float((actual_combined - expected_combined).abs().max())
    assert max_abs_delta < 1e-9
    combination = w5g["combination_gates"]
    assert combination["verdict"] == "FAIL"
    assert w5g["metadata"]["selected_candidate"] == "W5a"
    print(json.dumps({
        "synthetic_cases": 4,
        "w5g_selected_candidate": w5g["metadata"]["selected_candidate"],
        "w5g_verdict": combination["verdict"],
        "w5g_formula_max_abs_delta": max_abs_delta,
        "w5g_oos_pass": combination["oos_pass"],
        "w5g_mdd_pass": combination["mdd_pass"],
        "w5g_cagr_pass": combination["cagr_pass"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
