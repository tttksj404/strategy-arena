from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingCandidate, FundingMarket
from research.wave2.funding import run_maker_portfolio
from research.wave10_carry100.configs import CONFIG_IDS, CONFIGS, Wave10Config, get_config
from research.wave10_carry100.engine import (
    ACTIVE_CAPITAL,
    MIN_ORDER_USDT,
    OOS_SPLIT,
    Wave10Result,
    run_fixed_fraction_portfolio,
)
from research.wave10_carry100.gates import gate_a_feasibility, gate_e_oos_dormancy, gross_usdt, leg_usdt


# ---------------------------------------------------------------------------
# Synthetic markets (same style as research/wave2/tests/test_wave2.py::_synthetic_market)
# ---------------------------------------------------------------------------


def _flat_single_symbol_market(funding_rate: float = 0.001, periods: int = 8) -> dict[str, FundingMarket]:
    daily_index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2026-01-01", periods=periods * 3, freq="8h", tz="UTC")
    flat = pd.DataFrame({"open": 100.0, "close": 100.0}, index=daily_index)
    funding = pd.Series(funding_rate, index=funding_index, name="funding_rate")
    return {"BTCUSDT": FundingMarket(flat, flat, funding)}


def _flat_two_symbol_market(periods: int = 8) -> dict[str, FundingMarket]:
    daily_index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2026-01-01", periods=periods * 3, freq="8h", tz="UTC")
    flat = pd.DataFrame({"open": 100.0, "close": 100.0}, index=daily_index)
    funding_a = pd.Series(0.001, index=funding_index, name="funding_rate")
    funding_b = pd.Series(0.0009, index=funding_index, name="funding_rate")
    return {"BTCUSDT": FundingMarket(flat, flat, funding_a), "ETHUSDT": FundingMarket(flat, flat, funding_b)}


def _diverging_basis_market(periods: int = 10) -> dict[str, FundingMarket]:
    """Spot compounds +1%/day, perp compounds +0.8%/day (open[t] == close[t-1] for both, so
    the overnight gap channel is exactly zero and only the intraday spot-vs-perp channel is
    live). Funding is a small constant so the position activates. Used to prove that price
    moving does NOT leak into P&L except through the spot-vs-perp *difference*, which can only
    happen if the engine holds equal-and-opposite notional on both legs (true delta-neutral)."""
    spot_closes = [100.0 * (1.01**i) for i in range(periods)]
    perp_closes = [100.0 * (1.008**i) for i in range(periods)]
    spot_opens = [100.0, *spot_closes[:-1]]
    perp_opens = [100.0, *perp_closes[:-1]]
    daily_index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2026-01-01", periods=periods * 3, freq="8h", tz="UTC")
    spot = pd.DataFrame({"open": spot_opens, "close": spot_closes}, index=daily_index)
    perp = pd.DataFrame({"open": perp_opens, "close": perp_closes}, index=daily_index)
    funding = pd.Series(0.0001, index=funding_index, name="funding_rate")
    return {"BTCUSDT": FundingMarket(spot, perp, funding)}


# ---------------------------------------------------------------------------
# 1) Sizing arithmetic: fixed-fraction weight replaces wave2's 1/len(ranked) equal split.
# ---------------------------------------------------------------------------


def test_fixed_fraction_weight_replaces_wave2_equal_split_weight() -> None:
    market = _flat_single_symbol_market()
    same_signal = FundingCandidate("SYN", 1, 0.05, 1)

    # wave2's own (imported, unmodified) engine: 1 ranked symbol -> weight = 1/1 = 1.0 (100%).
    baseline = run_maker_portfolio(market, same_signal)
    assert baseline.positions[baseline.positions > 0.0].iloc[0] == pytest.approx(1.0)

    # wave10: same signal, but a fixed 50% leg fraction regardless of how many symbols ranked.
    config = Wave10Config(same_signal, 0.50, "test")
    result = run_fixed_fraction_portfolio(market, config)
    assert result.positions[result.positions > 0.0].iloc[0] == pytest.approx(0.50)
    # sizing changed; the underlying signal (which day activates) did not.
    assert (baseline.positions > 0.0).tolist() == (result.positions > 0.0).tolist()


def test_fixed_fraction_weight_is_independent_of_ranked_count() -> None:
    """C3-style: top_k=2, both symbols concurrently eligible -> EACH gets leg_fraction (0.25),
    not 1/2 = 0.5 each (which is what wave2/wave1's equal-split rule would have given)."""
    market = _flat_two_symbol_market()
    config = Wave10Config(FundingCandidate("SYN2", 1, 0.05, 2), 0.25, "test")
    result = run_fixed_fraction_portfolio(market, config)
    active_day = result.positions[result.positions > 0.0].index[0]
    assert result.max_concurrent_positions == 2
    # positions == sum(|weight|) across symbols == 0.25 + 0.25, NOT 0.5 + 0.5
    assert float(result.positions.loc[active_day]) == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# 2) Gross exposure calculation: matches the pre-registered $ figures and stays <= active capital.
# ---------------------------------------------------------------------------


def test_gross_exposure_matches_registered_dollar_figures() -> None:
    expected = {"C1": (45.0, 90.0, 1.0), "C2": (36.0, 72.0, 0.8), "C3": (22.5, 90.0, 1.0), "C4": (40.5, 81.0, 0.9)}
    assert set(CONFIG_IDS) == set(expected)
    for config in CONFIGS:
        candidate_id = config.candidate.candidate_id
        expected_leg, expected_gross, expected_multiplier = expected[candidate_id]
        assert leg_usdt(config) == pytest.approx(expected_leg)
        assert gross_usdt(config) == pytest.approx(expected_gross)
        assert gross_usdt(config) / ACTIVE_CAPITAL == pytest.approx(expected_multiplier)
        # the wave-8 failure mode this wave exists to avoid: gross must never exceed active capital.
        assert gross_usdt(config) <= ACTIVE_CAPITAL + 1e-9


def test_c3_two_pairs_gross_equals_c1_single_pair_gross_by_design() -> None:
    # 2 pairs x 25%/leg and 1 pair x 50%/leg both land on gross == 1.0x active capital;
    # this is the "reduce pair count and notional together" hypothesis made concrete.
    c1, c3 = get_config("C1"), get_config("C3")
    assert gross_usdt(c1) == pytest.approx(gross_usdt(c3))
    assert c1.candidate.top_k == 1 and c3.candidate.top_k == 2
    assert c1.leg_fraction == pytest.approx(0.50) and c3.leg_fraction == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 3) Minimum order gate: registered configs clear $5/leg; an undersized config is caught.
# ---------------------------------------------------------------------------


def test_registered_configs_clear_minimum_order() -> None:
    for config in CONFIGS:
        assert leg_usdt(config) >= MIN_ORDER_USDT


def test_undersized_leg_fraction_fails_the_min_order_gate() -> None:
    # 2% of $90 active capital = $1.80, below the $5 minimum -> gate must catch this, proving
    # gate_a_feasibility actually discriminates rather than being vacuously true.
    tiny = Wave10Config(FundingCandidate("TINY", 7, 0.15, 1), 0.02, "undersized (regression guard)")
    assert leg_usdt(tiny) == pytest.approx(1.8)
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    flat_result = Wave10Result(
        equity=pd.Series([90.0, 90.0, 90.0], index=idx),
        positions=pd.Series([0.0, 1.0, 1.0], index=idx),
        turnover=pd.Series([0.0, 0.02, 0.0], index=idx),
        trade_returns=pd.Series(dtype=float),
        max_concurrent_positions=1,
        symbols_used=("BTCUSDT",),
    )
    gate = gate_a_feasibility(tiny, flat_result)
    assert gate["min_order_feasible"] is False
    assert gate["status"] == "FAIL"


# ---------------------------------------------------------------------------
# 4) Delta-neutral invariant: a single shared weight drives both legs, so only the
#    spot-vs-perp *difference* (plus funding) ever reaches P&L -- outright price direction
#    cancels. This is the mechanism that keeps every position delta-neutral by construction.
# ---------------------------------------------------------------------------


def test_delta_neutral_hedge_captures_only_the_spot_perp_difference() -> None:
    market = _diverging_basis_market()
    config = Wave10Config(FundingCandidate("SYN4", 1, 0.05, 1), 0.5, "test")
    result = run_fixed_fraction_portfolio(market, config)

    # position turns on at index 1 and stays on (no further turnover) through the end.
    assert result.positions.iloc[1:].tolist() == [0.5] * (len(result.positions) - 1)

    # independent re-derivation from the raw synthetic inputs (not the engine's own formula):
    # on any day with unchanged weight (no entry/exit cost), growth must equal
    # 1 + weight * (spot_intraday_return - perp_intraday_return + funding_daily).
    spot = market["BTCUSDT"].spot
    perp = market["BTCUSDT"].perp
    funding_daily = 0.0001 * 3.0
    # index 1 is the entry day (entry cost) and the last index carries the loop's final unwind
    # cost (the position is still open at series end) -- both are excluded from the no-turnover
    # steady-state check below by construction, not by cherry-picking a convenient result.
    for i in range(2, len(result.equity) - 1):
        spot_ret = float(spot["close"].iloc[i] / spot["open"].iloc[i] - 1.0)
        perp_ret = float(perp["close"].iloc[i] / perp["open"].iloc[i] - 1.0)
        expected_factor = 1.0 + 0.5 * (spot_ret - perp_ret + funding_daily)
        actual_factor = float(result.equity.iloc[i] / result.equity.iloc[i - 1])
        assert actual_factor == pytest.approx(expected_factor, rel=1e-9)


def test_delta_neutral_flat_price_leaves_only_funding_pnl() -> None:
    # Degenerate case of the above with spot == perp always (basis permanently zero): ANY
    # price level should contribute nothing, so growth is driven purely by funding and costs.
    market = _flat_single_symbol_market(funding_rate=0.001, periods=8)
    config = Wave10Config(FundingCandidate("SYN", 1, 0.05, 1), 0.5, "test")
    result = run_fixed_fraction_portfolio(market, config)
    funding_daily = 0.001 * 3.0
    for i in range(2, len(result.equity) - 1):  # interior steady-state days (skip entry/exit cost days)
        actual_factor = float(result.equity.iloc[i] / result.equity.iloc[i - 1])
        assert actual_factor == pytest.approx(1.0 + 0.5 * funding_daily, rel=1e-9)


# ---------------------------------------------------------------------------
# 5) Gate E dormancy labeling + config registry integrity (bonus coverage).
# ---------------------------------------------------------------------------


def test_gate_e_labels_zero_oos_position_as_untested_not_fail() -> None:
    idx = pd.date_range("2025-10-15", periods=4, freq="D", tz="UTC")
    assert idx[0] > OOS_SPLIT
    result = Wave10Result(
        equity=pd.Series([90.0, 90.1, 90.2, 90.3], index=idx),
        positions=pd.Series([0.0, 0.0, 0.0, 0.0], index=idx),
        turnover=pd.Series([0.0, 0.0, 0.0, 0.0], index=idx),
        trade_returns=pd.Series(dtype=float),
        max_concurrent_positions=0,
        symbols_used=("BTCUSDT",),
    )
    gate = gate_e_oos_dormancy(result)
    assert gate["has_position"] is False
    assert gate["status"] == "UNTESTED_IN_OOS"
    assert gate["status"] not in {"PASS", "FAIL"}


def test_config_registry_is_frozen_to_four_preregistered_ids() -> None:
    assert CONFIG_IDS == ("C1", "C2", "C3", "C4")
    assert len(CONFIGS) == 4
    # top_k on the candidate IS the pair count by design (no separate/divergent pair-count field).
    assert [c.candidate.top_k for c in CONFIGS] == [1, 1, 2, 1]
