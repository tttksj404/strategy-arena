from __future__ import annotations

import pytest

from research.wave4_leverage.sweep import (
    BORROW_APR,
    LIQUIDATION_FEE_RATE,
    asym_capital_efficiency,
    liquidation_loss,
    notional_multiplier,
    spot_borrow_fraction,
)


def test_liquidation_triggers_on_worst_basis_and_adds_fee() -> None:
    notional = 100.0
    initial_margin = 10.0
    loss = liquidation_loss(notional, -0.10, initial_margin)
    assert loss == pytest.approx(10.06)
    assert liquidation_loss(notional, -0.09, initial_margin) is None


def test_sym_borrow_interest_is_zero_at_one_x_and_scales() -> None:
    assert spot_borrow_fraction("SYM", 1.0) == pytest.approx(0.0)
    assert spot_borrow_fraction("SYM", 3.0) == pytest.approx(1.0)
    assert 300.0 * spot_borrow_fraction("SYM", 3.0) * BORROW_APR / 365.0 == pytest.approx(0.0821917808)


def test_asym_capital_efficiency_matches_preregistered_formula() -> None:
    assert asym_capital_efficiency(1.0) == pytest.approx(1.0)
    assert asym_capital_efficiency(10.0) == pytest.approx(20.0 / 11.0)
    assert notional_multiplier("ASYM", 10.0) == pytest.approx(10.0 / 11.0)
    assert LIQUIDATION_FEE_RATE == pytest.approx(0.0006)
