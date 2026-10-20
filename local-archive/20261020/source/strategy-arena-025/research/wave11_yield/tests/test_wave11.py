from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingCandidate, FundingMarket
from research.wave10_carry100.configs import Wave10Config
from research.wave10_carry100.engine import ACTIVE_CAPITAL, cost_rate
from research.wave10_carry100.engine import run_fixed_fraction_portfolio as wave10_run_fixed_fraction_portfolio
from research.wave11_yield.configs import CONFIG_IDS, CONFIGS, Y6_SPIKE_ENTRY_RATE, Y6_SPIKE_EXIT_APR, Y6_SPIKE_HOLD_DAYS, get_config
from research.wave11_yield.engine_y import (
    _run_fixed_fraction_loop,
    missing_market_files,
    run_8h_fixed_fraction,
    run_daily_fixed_fraction,
    y6_spike_active_builder,
)
from research.wave11_yield.gates_y import (
    PROMOTION_BLOCK_MDD_MAX,
    PROMOTION_HIGH_FUNDING_ANNUALIZED_MIN,
    S2_RUIN_PROBABILITY_MAX,
    S3_BLOCK_MDD_P95_MAX,
    gross_usdt,
    leg_usdt,
    promotion_check,
)


# ---------------------------------------------------------------------------
# 1) Config registry integrity: frozen 6 candidates, $ figures match SPEC.md.
# ---------------------------------------------------------------------------


def test_config_registry_is_frozen_to_six_preregistered_ids() -> None:
    assert CONFIG_IDS == ("Y1", "Y2", "Y3", "Y4", "Y5", "Y6")
    assert len(CONFIGS) == 6


def test_registered_leg_and_gross_dollar_figures_match_spec() -> None:
    expected = {
        "Y1": (45.0, 90.0, 1.0),
        "Y2": (11.25, 90.0, 1.0),
        "Y3": (11.25, 90.0, 1.0),
        "Y4": (45.0, 90.0, 1.0),
        "Y5": (45.0, 90.0, 1.0),
        "Y6": (22.5, 90.0, 1.0),
    }
    assert set(CONFIG_IDS) == set(expected)
    for config in CONFIGS:
        candidate_id = config.candidate.candidate_id
        expected_leg, expected_gross, expected_multiplier = expected[candidate_id]
        assert leg_usdt(config) == pytest.approx(expected_leg)
        assert gross_usdt(config) == pytest.approx(expected_gross)
        assert gross_usdt(config) / ACTIVE_CAPITAL == pytest.approx(expected_multiplier)
        assert gross_usdt(config) <= ACTIVE_CAPITAL + 1e-9  # S4/S1: never exceed active capital


def test_y1_entry_exit_thresholds_match_spec_8pct_4pct() -> None:
    # SPEC.md: "Y1 | 임계↓ | ... 진입 8%APR / 청산 4%APR". carry_position()'s built-in
    # hysteresis exit is threshold_apr/2, so registering threshold_apr=0.08 IS the
    # 8%/4% pair -- this test pins that arithmetic so a future edit can't silently
    # drift the exit rule away from what SPEC.md promises.
    y1 = get_config("Y1")
    assert y1.candidate.threshold_apr == pytest.approx(0.08)
    assert y1.candidate.threshold_apr / 2.0 == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# 2) Y6 spike active-builder: entry trigger, 3-day hold, early-exit-on-low-score.
# ---------------------------------------------------------------------------


def _y6_synthetic_inputs(daily_apr: list[float]) -> tuple[FundingCandidate, FundingMarket, pd.Series]:
    periods = len(daily_apr)
    idx_daily = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    funding_idx = pd.date_range("2026-01-01", periods=periods * 3, freq="8h", tz="UTC")
    funding = pd.Series(0.0001, index=funding_idx, name="funding_rate")  # baseline, below entry rate
    funding.iloc[0] = 0.0008  # single spike on day0's first 8h print, > Y6_SPIKE_ENTRY_RATE (0.05%)
    market = FundingMarket(spot=pd.DataFrame(), perp=pd.DataFrame(), funding=funding)
    candidate = FundingCandidate("TESTY6", 7, Y6_SPIKE_EXIT_APR, 2)
    funding_apr = pd.Series(daily_apr, index=idx_daily)
    return candidate, market, funding_apr


def test_y6_spike_active_builder_holds_exactly_three_days_then_exits() -> None:
    # apr stays well above the exit bar throughout, so only the hold-day timeout can end the trade.
    candidate, market, funding_apr = _y6_synthetic_inputs([0.10] * 6)
    active = y6_spike_active_builder(candidate, market, funding_apr)
    assert active.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]  # shift(1)-lagged: 3 active marks, then flat


def test_y6_spike_active_builder_exits_early_when_score_drops_below_exit_apr() -> None:
    # apr drops below Y6_SPIKE_EXIT_APR (0.05) on day1 (hold_days=2), before the 3-day timeout would fire.
    candidate, market, funding_apr = _y6_synthetic_inputs([0.10, 0.02, 0.02, 0.02, 0.02, 0.02])
    active = y6_spike_active_builder(candidate, market, funding_apr)
    assert active.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # only 2 active marks: score-exit beat the hold timeout


def test_y6_spike_active_builder_no_spike_never_activates() -> None:
    idx_daily = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    funding_idx = pd.date_range("2026-01-01", periods=12, freq="8h", tz="UTC")
    funding = pd.Series(0.0001, index=funding_idx, name="funding_rate")  # never exceeds entry rate
    market = FundingMarket(spot=pd.DataFrame(), perp=pd.DataFrame(), funding=funding)
    candidate = FundingCandidate("TESTY6", 7, Y6_SPIKE_EXIT_APR, 2)
    funding_apr = pd.Series([0.10] * 4, index=idx_daily)  # high APR alone must NOT trigger entry -- only a raw spike does
    active = y6_spike_active_builder(candidate, market, funding_apr)
    assert active.tolist() == [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 3) Equivalence: wave11's daily engine (default active_builder) reproduces
#    research.wave10_carry100.engine.run_fixed_fraction_portfolio's output exactly.
#    This is the evidence backing the report's claim that Y1-Y4 run on a numerically
#    identical engine to wave10's, not a reimplementation.
# ---------------------------------------------------------------------------


def _two_symbol_synthetic_market(periods: int = 12) -> dict[str, FundingMarket]:
    daily_index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2026-01-01", periods=periods * 3, freq="8h", tz="UTC")
    a_close = [100.0 * (1.004**i) for i in range(periods)]
    a_open = [100.0, *a_close[:-1]]
    b_close = [50.0 * (1.002**i) for i in range(periods)]
    b_open = [50.0, *b_close[:-1]]
    spot_a = pd.DataFrame({"open": a_open, "close": a_close}, index=daily_index)
    perp_a = pd.DataFrame({"open": [v * 0.999 for v in a_open], "close": [v * 0.999 for v in a_close]}, index=daily_index)
    spot_b = pd.DataFrame({"open": b_open, "close": b_close}, index=daily_index)
    perp_b = pd.DataFrame({"open": [v * 1.001 for v in b_open], "close": [v * 1.001 for v in b_close]}, index=daily_index)
    funding_a = pd.Series(0.0010, index=funding_index, name="funding_rate")  # ~109.5% annualized
    funding_b = pd.Series(0.0006, index=funding_index, name="funding_rate")  # ~65.7% annualized
    return {"BTCUSDT": FundingMarket(spot_a, perp_a, funding_a), "ETHUSDT": FundingMarket(spot_b, perp_b, funding_b)}


def test_default_daily_engine_matches_wave10_engine_exactly() -> None:
    markets = _two_symbol_synthetic_market()
    candidate = FundingCandidate("EQTEST", 1, 0.50, 1)
    leg_fraction = 0.5
    wave10_result = wave10_run_fixed_fraction_portfolio(markets, Wave10Config(candidate, leg_fraction, "equivalence test"))
    wave11_result, total_cost = run_daily_fixed_fraction(markets, candidate, leg_fraction)
    assert wave11_result.equity.tolist() == pytest.approx(wave10_result.equity.tolist(), rel=1e-12)
    assert wave11_result.positions.tolist() == pytest.approx(wave10_result.positions.tolist(), rel=1e-12)
    assert wave11_result.turnover.tolist() == pytest.approx(wave10_result.turnover.tolist(), rel=1e-12)
    assert wave11_result.trade_returns.tolist() == pytest.approx(wave10_result.trade_returns.tolist(), rel=1e-12)
    assert wave11_result.max_concurrent_positions == wave10_result.max_concurrent_positions
    assert total_cost > 0.0  # trades did occur in this synthetic market, so cost must be strictly positive


# ---------------------------------------------------------------------------
# 4) Y5's 8h engine: delta-neutral invariant (same style of proof as
#    research/wave10_carry100/tests/test_wave10_engine.py's diverging-basis test, at 8h bars).
# ---------------------------------------------------------------------------


def _diverging_basis_market_8h(periods: int = 9) -> dict[str, FundingMarket]:
    spot_closes = [100.0 * (1.003**i) for i in range(periods)]
    perp_closes = [100.0 * (1.002**i) for i in range(periods)]
    spot_opens = [100.0, *spot_closes[:-1]]
    perp_opens = [100.0, *perp_closes[:-1]]
    idx = pd.date_range("2026-01-01", periods=periods, freq="8h", tz="UTC")
    spot = pd.DataFrame({"open": spot_opens, "close": spot_closes}, index=idx)
    perp = pd.DataFrame({"open": perp_opens, "close": perp_closes}, index=idx)
    funding = pd.Series(0.0006, index=idx, name="funding_rate")  # one print per 8h bar, ~65.7% annualized
    return {"BTCUSDT": FundingMarket(spot, perp, funding)}


def test_y5_8h_engine_delta_neutral_hedge_captures_only_spot_perp_difference() -> None:
    markets = _diverging_basis_market_8h()
    candidate = FundingCandidate("Y5TEST", 1, 0.05, 1)
    result, _total_cost = run_8h_fixed_fraction(markets, candidate, 0.5)
    # funding_score's rolling window (window_days*3=3 raw 8h periods) first has enough data
    # at bar index 2; carry_position's own shift(1) then delays activation to bar index 3
    # (unlike the daily engine, resample("8h") on an already-8h-native series is a no-op,
    # so there is no same-day "last()" collapse the way research/wave10_carry100's own
    # equivalent test benefits from on daily bars -- see that test's index1 vs this one's index3).
    assert result.positions.iloc[3:].tolist() == pytest.approx([0.5] * (len(result.positions) - 3))
    spot = markets["BTCUSDT"].spot
    perp = markets["BTCUSDT"].perp
    for i in range(4, len(result.equity) - 1):  # interior steady-state bars, skip entry/exit-cost bars
        spot_ret = float(spot["close"].iloc[i] / spot["open"].iloc[i] - 1.0)
        perp_ret = float(perp["close"].iloc[i] / perp["open"].iloc[i] - 1.0)
        expected_factor = 1.0 + 0.5 * (spot_ret - perp_ret + 0.0006)
        actual_factor = float(result.equity.iloc[i] / result.equity.iloc[i - 1])
        assert actual_factor == pytest.approx(expected_factor, rel=1e-9)


# ---------------------------------------------------------------------------
# 5) Total-cost tracking: an independently hand-derived dollar figure for a minimal
#    single-entry, held-to-series-end scenario (entry cost + forced final-unwind cost).
# ---------------------------------------------------------------------------


def test_total_cost_usdt_matches_independently_derived_value() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    frame = lambda values: pd.DataFrame({"BTCUSDT": values}, index=idx)  # noqa: E731
    spot_open_frame, spot_close_frame = frame([100.0, 100.0]), frame([100.0, 100.0])
    perp_open_frame, perp_close_frame = frame([100.0, 100.0]), frame([100.0, 100.0])
    funding_frame = frame([0.0, 0.0])
    score_frame = frame([0.0, 0.20])
    active_frame = frame([0.0, 1.0])  # inactive day0, active day1 -- entry on day1, held open at series end

    result, total_cost = _run_fixed_fraction_loop(
        spot_open_frame, spot_close_frame, perp_open_frame, perp_close_frame, funding_frame, score_frame, active_frame, top_k=1, leg_fraction=0.5
    )
    rate = cost_rate("BTCUSDT")
    entry_cost = ACTIVE_CAPITAL * 0.5 * rate
    capital_after_entry = ACTIVE_CAPITAL - entry_cost
    final_unwind_cost = capital_after_entry * 0.5 * rate
    expected_total_cost = entry_cost + final_unwind_cost
    assert total_cost == pytest.approx(expected_total_cost, rel=1e-9)
    assert result.equity.iloc[-1] == pytest.approx(capital_after_entry - final_unwind_cost, rel=1e-9)


def test_total_cost_usdt_is_zero_when_never_active() -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    frame = lambda values: pd.DataFrame({"BTCUSDT": values}, index=idx)  # noqa: E731
    flat = frame([100.0, 100.0, 100.0])
    zero = frame([0.0, 0.0, 0.0])
    _result, total_cost = _run_fixed_fraction_loop(flat, flat, flat, flat, zero, zero, zero, top_k=1, leg_fraction=0.5)
    assert total_cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6) Gate thresholds: wave11's S2/S3 are registered stricter than wave10's A/C gates.
# ---------------------------------------------------------------------------


def test_s2_s3_thresholds_are_wave11s_own_stricter_bars() -> None:
    assert S2_RUIN_PROBABILITY_MAX == pytest.approx(0.01)  # SPEC: <1% (wave10 gate_b used <5%)
    assert S3_BLOCK_MDD_P95_MAX == pytest.approx(0.10)  # SPEC: <=10% (wave10 gate_c used <=25%)


# ---------------------------------------------------------------------------
# 7) Promotion check: requires BOTH the high-funding-annualized bar AND the MDD bar.
# ---------------------------------------------------------------------------


def test_promotion_check_requires_both_conditions() -> None:
    good_regime = {"high_funding_mean_annualized_return": PROMOTION_HIGH_FUNDING_ANNUALIZED_MIN + 0.01}
    bad_regime = {"high_funding_mean_annualized_return": PROMOTION_HIGH_FUNDING_ANNUALIZED_MIN - 0.01}
    good_gate_s3 = {"mdd_p95": PROMOTION_BLOCK_MDD_MAX - 0.001}
    bad_gate_s3 = {"mdd_p95": PROMOTION_BLOCK_MDD_MAX + 0.001}

    assert promotion_check(good_regime, good_gate_s3).promoted is True
    assert promotion_check(bad_regime, good_gate_s3).promoted is False  # high-funding bar missed
    assert promotion_check(good_regime, bad_gate_s3).promoted is False  # MDD bar missed
    assert promotion_check(bad_regime, bad_gate_s3).promoted is False  # both missed
    assert promotion_check({"high_funding_mean_annualized_return": None}, good_gate_s3).promoted is False  # no data -> not promoted


# ---------------------------------------------------------------------------
# 8) Fail-closed cache checks: a symbol with no cached files must be reported missing,
#    not silently skipped -- this is what makes verify_cache_and_load_symbols_* fail
#    closed instead of quietly running on a smaller universe than intended.
# ---------------------------------------------------------------------------


def test_missing_market_files_reports_all_three_files_for_an_uncached_symbol() -> None:
    missing = missing_market_files(("DEFINITELY_NOT_A_REAL_CACHED_SYMBOL_XYZ",))
    assert len(missing) == 3
    assert any("spot" in name for name in missing)
    assert any("fapi" in name for name in missing)
    assert any("funding" in name for name in missing)
