from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from research.wave1.fam_funding import F1_CANDIDATES, load_markets, run_portfolio
from research.wave2.funding import W2_FUNDING_CANDIDATES, run_maker_portfolio
from research.wave4_leverage.sweep import (
    INITIAL_CAPITAL,
    build_trace,
    simulate,
)

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
    assert notional_multiplier("ASYM", 10.0) == pytest.approx(20.0 / 11.0)
    assert notional_multiplier("SYM", 1.0) == pytest.approx(1.0)
    assert LIQUIDATION_FEE_RATE == pytest.approx(0.0006)


@pytest.mark.parametrize(
    ("candidate_id", "structure"),
    (("W2c", "SYM"), ("F1f", "ASYM")),
)
def test_l1_reconciles_with_wave_engine_cagr_and_mdd(
    candidate_id: str,
    structure: Literal["SYM", "ASYM"],
) -> None:
    root = Path(__file__).resolve().parents[3]
    cache_dir = root / "research" / "wave1" / "cache"
    symbols = tuple(str(symbol) for symbol in json.loads((cache_dir / "universe.json").read_text(encoding="utf-8"))["symbols"])
    candidate = next(
        candidate
        for candidate in (*F1_CANDIDATES, *W2_FUNDING_CANDIDATES)
        if candidate.candidate_id == candidate_id
    )
    markets = load_markets(cache_dir, symbols)
    trace, _ = build_trace(cache_dir, symbols, candidate)
    engine = run_maker_portfolio(markets, candidate) if candidate_id == "W2c" else run_portfolio(markets, candidate)
    result = simulate(
        trace,
        candidate_id,
        structure,
        1.0,
        float(engine.equity.iloc[-1]),
        tuple(markets),
        20_260_716,
        engine_equity=engine.equity,
        engine_trade_returns=engine.trade_returns,
    )
    assert result.cagr == pytest.approx(
        float((engine.equity.iloc[-1] / INITIAL_CAPITAL) ** (365.0 / (engine.equity.index[-1] - engine.equity.index[0]).days) - 1.0),
        rel=0.01,
    )
    engine_mdd = float((1.0 - engine.equity / engine.equity.cummax()).max())
    assert result.mdd == pytest.approx(engine_mdd, rel=0.01)
    assert result.reconciliation_pass is True
    if candidate_id == "W2c":
        assert result.mc_p05 == pytest.approx(746.3225248264143, rel=0.01)
