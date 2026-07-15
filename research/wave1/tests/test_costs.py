# Verifies the preregistered fee and slippage cost model.

from __future__ import annotations

import pytest

from research.wave1.costs import LegCost, f1_round_trip_cost, transaction_cost


def test_transaction_cost_charges_fee_and_slippage_per_side() -> None:
    # Given
    leg = LegCost(fee_rate=0.0006, slippage_rate=0.0001)

    # When
    cost = transaction_cost(notional=1_000.0, leg=leg)

    # Then
    assert cost == pytest.approx(0.7)


def test_f1_round_trip_charges_four_legs() -> None:
    # Given
    notional = 1_000.0

    # When
    cost = f1_round_trip_cost(notional=notional, slippage_rate=0.0001)

    # Then
    assert cost == pytest.approx(3.6)

