# Verifies that positive funding transfers capital from longs to shorts.

from __future__ import annotations

import pytest

from research.wave1.costs import funding_cashflow


def test_positive_funding_debits_long() -> None:
    # Given / When
    cashflow = funding_cashflow(notional=1_000.0, funding_rate=0.001, position=1.0)

    # Then
    assert cashflow == pytest.approx(-1.0)


def test_positive_funding_credits_short() -> None:
    # Given / When
    cashflow = funding_cashflow(notional=1_000.0, funding_rate=0.001, position=-1.0)

    # Then
    assert cashflow == pytest.approx(1.0)

