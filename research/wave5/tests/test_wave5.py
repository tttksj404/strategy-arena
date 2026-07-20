from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave5.engine import (
    basis_round_trip_cost,
    combine_returns,
    funding_capitulation_position,
    zscore_hysteresis_position,
)
from research.wave5.gates import combination_gates
from research.wave5.strategies import select_top_active_positions


def test_zscore_entry_and_exit_use_hysteresis() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=7, freq="D", tz="UTC")
    zscore = pd.Series([0.0, -2.1, -1.8, -0.4, 2.6, 2.0, 0.2], index=index)

    # When
    position = zscore_hysteresis_position(zscore, entry_z=2.0, exit_z=0.5)

    # Then
    assert position.tolist() == [0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0]


def test_basis_round_trip_cost_charges_four_legs() -> None:
    # Given
    notional = 1_000.0

    # When
    cost = basis_round_trip_cost(notional, slippage_rate=0.0001)

    # Then
    assert cost == pytest.approx(3.6)


def test_funding_capitulation_trigger_holds_for_48_hours() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=9, freq="8h", tz="UTC")
    funding = pd.Series([0.0, -0.0006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)

    # When
    position = funding_capitulation_position(funding, threshold=-0.0005, hold_bars=6)

    # Then
    assert position.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_funding_trigger_does_not_use_future_observations() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=3, freq="8h", tz="UTC")
    funding = pd.Series([0.0, 0.0, -0.0006], index=index)

    # When
    position = funding_capitulation_position(funding, threshold=-0.0005, hold_bars=6)

    # Then
    assert position.iloc[:2].eq(0.0).all()
    assert position.iloc[2] == 1.0


def test_combined_portfolio_is_equal_weighted() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    w2c = pd.Series([0.10, 0.0, -0.02], index=index)
    candidate = pd.Series([0.0, 0.10, -0.02], index=index)

    # When
    combined = combine_returns(w2c, candidate, weight=0.5)

    # Then
    pd.testing.assert_series_equal(combined, pd.Series([0.05, 0.05, -0.02], index=index))


def test_combination_rejects_missing_candidate_oos() -> None:
    # Given
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    baseline = pd.Series([300.0, 303.0, 306.0], index=index)
    candidate = pd.Series([300.0, 301.0, 302.0], index=index)
    combined = pd.Series([300.0, 302.0, 304.0], index=index)

    # When
    gates = combination_gates(baseline, candidate, combined)

    # Then
    assert gates["verdict"] == "UNTESTED_IN_OOS"
    assert gates["oos_pass"] is None


def test_funding_universe_ranking_is_point_in_time() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    positions = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]}, index=index)
    volume_rank = pd.DataFrame({"A": [100.0, 100.0, 1.0], "B": [1.0, 1.0, 1_000.0]}, index=index)

    # When
    selected = select_top_active_positions(positions, volume_rank, max_universe=1, max_positions=1)

    # Then
    assert selected["A"].tolist() == [1.0, 1.0, 0.0]
    assert selected["B"].tolist() == [0.0, 0.0, 1.0]
