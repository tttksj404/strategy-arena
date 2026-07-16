from __future__ import annotations

import json
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK
import pytest
from research.wave1.fam_funding import FundingCandidate, FundingMarket, run_portfolio
from research.wave2.funding import run_maker_portfolio
from research.wave2.gates import evaluate_result_file_wave2
from research.wave2.spike import spike_position


def _synthetic_market() -> dict[str, FundingMarket]:
    daily_index = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    funding_index = pd.date_range("2026-01-01", periods=24, freq="8h", tz="UTC")
    flat = pd.DataFrame(
        {"open": 100.0, "close": 100.0},
        index=daily_index,
    )
    funding = pd.Series(0.001, index=funding_index, name="funding_rate")
    return {"BTCUSDT": FundingMarket(flat, flat, funding)}


def test_maker_cost_path_reduces_two_leg_cost_without_slippage() -> None:
    # Given
    market = _synthetic_market()
    candidate = FundingCandidate("SYN", 1, 0.05, 1)

    # When
    result = run_maker_portfolio(market, candidate)
    taker = run_portfolio(market, candidate)

    # Then
    assert result.equity.iloc[-1] == pytest.approx(306.1119477816681)
    assert result.equity.iloc[-1] > taker.equity.iloc[-1]
    assert result.turnover.iloc[[1, -1]].tolist() == [1.0, 1.0]
    assert result.max_concurrent_positions == 1


def test_spike_entry_is_delayed_until_next_daily_bar() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=6, freq="8h", tz="UTC")
    funding = pd.Series([0.0, 0.0, 0.0006, 0.0, 0.0, 0.0], index=index)
    score = pd.Series(0.10, index=index)

    # When
    positions = spike_position(funding, score, entry_rate=0.0005, exit_threshold_apr=0.025)

    # Then
    assert positions.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_oos_no_position_is_labeled_untested() -> None:
    # Given
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    records = lambda values: [
        {"timestamp": timestamp.isoformat(), "value": value}
        for timestamp, value in zip(index, values, strict=True)
    ]
    path = Path(__file__).parent / "_wave2_untested.json"
    path.write_text(json.dumps({
                "candidate_id": "W2a",
                "family": "F1",
                "equity": records([300.0, 301.0, 302.0]),
                "positions": records([0.0, 0.0, 0.0]),
                "turnover": records([0.0, 0.0, 0.0]),
                "trade_returns": [],
                "stress_total_return": 0.0,
                "metadata": {
                    "data_valid": True,
                    "cost_model_valid": True,
                    "intended_factor": "funding_carry",
                    "neighbor_is_sharpes": [1.0, 1.1],
                    "max_concurrent_positions": 1,
                    "max_position_weight": 1.0,
                    "min_position_weight": 1.0,
                },
            }), encoding="utf-8")

    try:
        # When
        rows = evaluate_result_file_wave2(path, pd.Series(dtype=float))

        # Then
        assert next(row for row in rows if row.gate == 4).status == "UNTESTED_IN_OOS"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["validation"]["oos_label"] == "UNTESTED_IN_OOS"
        assert payload["validation"]["gate4_is_after_cost"]["status"] == "PASS"
    finally:
        path.unlink(missing_ok=True)
