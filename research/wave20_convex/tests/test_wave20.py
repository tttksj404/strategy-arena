from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave13_liquidity.costs_measured import MeasuredCostMapping
from research.wave20_convex import dataio20, engine20, gates20
from research.wave20_convex.configs20 import GAMBLE_CAPITAL, MAKER_FEE_RATE, STABLE_CAPITAL

UTC = "UTC"


def _mapping(anchor_log_volume: tuple[float, ...] = (3.0, 7.0), anchor_bp: tuple[float, ...] = (10.0, 1.0)) -> MeasuredCostMapping:
    return MeasuredCostMapping(
        anchor_log_volume=np.asarray(anchor_log_volume, dtype=float),
        anchor_bp=np.asarray(anchor_bp, dtype=float),
        bucket_counts=(5,) * len(anchor_log_volume),
        raw_point_count=10,
        source_collected_at_utc="2026-01-01T00:00:00Z",
    )


def _hourly_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1h", tz=UTC)


# ---------------------------------------------------------------------------
# true_range / atr
# ---------------------------------------------------------------------------


def test_true_range_uses_max_of_three_components():
    ohlc = pd.DataFrame(
        {"open": [100.0, 100.0, 100.0], "high": [105.0, 108.0, 101.0], "low": [95.0, 99.0, 90.0], "close": [100.0, 100.0, 100.0]},
        index=_hourly_index(3),
    )
    tr = engine20.true_range(ohlc)
    # bar0: no prior close -> high-low only = 10
    assert tr.iloc[0] == pytest.approx(10.0)
    # bar1: high-low=9, |high-prevclose|=8, |low-prevclose|=1 -> max=9
    assert tr.iloc[1] == pytest.approx(9.0)
    # bar2: high-low=11, |high-prevclose|=1, |low-prevclose|=10 -> max=11
    assert tr.iloc[2] == pytest.approx(11.0)


def test_atr_is_simple_rolling_mean_of_true_range():
    ohlc = pd.DataFrame(
        {"open": [10.0] * 5, "high": [12.0, 12.0, 12.0, 12.0, 12.0], "low": [8.0, 8.0, 8.0, 8.0, 8.0], "close": [10.0] * 5},
        index=_hourly_index(5),
    )
    result = engine20.atr(ohlc, window=2)
    # every bar's TR is exactly 4.0 (high-low, close flat) -> rolling(2).mean() == 4.0 once warmed up
    assert result.iloc[0] != result.iloc[0] or pd.isna(result.iloc[0])  # NaN, not enough history
    assert result.iloc[1] == pytest.approx(4.0)
    assert result.iloc[4] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# trailing_percentile_rank
# ---------------------------------------------------------------------------


def test_trailing_percentile_rank_of_monotonic_increasing_series_is_always_one():
    series = pd.Series(np.arange(1.0, 21.0))
    ranks = engine20.trailing_percentile_rank(series, window=5)
    valid = ranks.dropna()
    assert (valid == 1.0).all()  # the LAST value in any trailing window of a strictly increasing series is always its own max


def test_trailing_percentile_rank_of_monotonic_decreasing_series_is_minimum():
    series = pd.Series(np.arange(20.0, 0.0, -1.0))
    ranks = engine20.trailing_percentile_rank(series, window=5)
    valid = ranks.dropna().to_numpy()
    # the last value in a decreasing window is always the smallest -> rank = 1/window (itself counted as <= itself)
    assert valid == pytest.approx(1.0 / 5.0)


# ---------------------------------------------------------------------------
# Cost model: single-leg, not wave13's 2-leg carry-pair convention.
# ---------------------------------------------------------------------------


def test_worst_case_cost_is_maker_plus_one_times_slippage_not_two_times():
    mapping = _mapping()
    result = engine20.worst_case_cost(mapping, stress_multiplier=1.0)
    expected = MAKER_FEE_RATE + mapping.worst_bp * 0.0001  # ONE leg, not costs_measured.cost_rate_from_bp's 2x
    assert result == pytest.approx(expected)
    assert result == pytest.approx(0.5 * (2.0 * MAKER_FEE_RATE + 2.0 * mapping.worst_bp * 0.0001))


def test_one_leg_cost_rate_series_falls_back_to_worst_when_history_insufficient():
    mapping = _mapping()
    short_volume = pd.Series([1e8] * 5, index=pd.date_range("2024-01-01", periods=5, freq="1D", tz=UTC))
    rates = engine20.one_leg_cost_rate_series(short_volume, mapping)
    worst = engine20.worst_case_cost(mapping)
    assert rates.to_numpy() == pytest.approx(worst)  # < ROLLING_WINDOW_DAYS points of history everywhere -> fail-closed to worst


def test_one_leg_cost_rate_series_cheaper_for_higher_volume():
    mapping = _mapping()
    idx = pd.date_range("2024-01-01", periods=60, freq="1D", tz=UTC)
    low_volume = pd.Series([1e3] * 60, index=idx)
    high_volume = pd.Series([1e9] * 60, index=idx)
    low_rate = engine20.one_leg_cost_rate_series(low_volume, mapping).iloc[-1]
    high_rate = engine20.one_leg_cost_rate_series(high_volume, mapping).iloc[-1]
    assert high_rate < low_rate


# ---------------------------------------------------------------------------
# simulate_breakout_reversal -- the shared V1/V3 core, including the trailing-extreme fix.
# ---------------------------------------------------------------------------


def _flat_cost_array(n: int, rate: float = 0.001) -> np.ndarray:
    return np.full(n, rate, dtype=float)


def test_breakout_enters_at_next_bar_open_not_signal_bar_close():
    n = 6
    index = _hourly_index(n)
    close = np.array([100.0, 100.0, 100.0, 120.0, 121.0, 122.0])  # big jump at bar3 (close 100->120)
    open_ = np.array([100.0, 100.0, 100.0, 100.5, 120.5, 121.5])
    atr_arr = np.full(n, 1.0)  # tiny ATR -> any real move breaks the 2xATR band easily
    cost_arr = _flat_cost_array(n)
    armable = np.array([True, True, True, True, True, True])

    equity_out, trades, total_cost = engine20.simulate_breakout_reversal(
        index=index, open_arr=open_, close_arr=close, atr_arr=atr_arr, cost_arr=cost_arr, worst_cost=0.001,
        atr_multiplier=2.0, armable_arr=armable, forced_initial_direction=None, max_bars_in_position=None,
        starting_equity=GAMBLE_CAPITAL, symbol="TEST",
    )
    assert total_cost > 0.0
    # anchor seeds at bar0's close (100.0); bar3's close (120.0) breaks +2*ATR=+2 -> signal at bar3, filled at bar4's OPEN (120.5)
    opens_seen = {t.entry_price for t in trades}
    assert 120.5 in opens_seen or any(abs(p - 120.5) < 1e-9 for p in opens_seen)


def test_reversal_uses_trailing_extreme_not_fixed_entry_price_regression():
    """Regression test for the bug found during implementation: anchoring the reversal check
    to a FIXED entry price makes a strongly-trending position structurally un-reversible once
    ATR (computed at the current, much higher price level) makes entry_price -/+ N*ATR
    unreachable. A huge favorable run followed by a real pullback of the same relative size
    the entry breakout itself used MUST still trigger a reversal."""
    n = 11
    index = _hourly_index(n)
    # bar0..2 flat (seed anchor), bar3 breaks up hugely (100->1000), bars4-8 stay near 1000,
    # bar9 pulls back by more than 2*ATR from the 1000 extreme (still nowhere near the ORIGINAL
    # entry price of ~1000 -- a fixed-entry anchor would never see this as a reversal), bar10
    # exists solely so bar9's reversal SIGNAL has a next-bar open to execute at (t->t+1 timing).
    close = np.array([100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 960.0, 955.0])
    open_ = np.array([100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 958.0])
    atr_arr = np.full(n, 5.0)  # 2xATR = 10 -- a 40-point pullback (1000->960) should trip it
    cost_arr = _flat_cost_array(n, rate=0.0001)
    armable = np.array([True] * n)

    equity_out, trades, _total_cost = engine20.simulate_breakout_reversal(
        index=index, open_arr=open_, close_arr=close, atr_arr=atr_arr, cost_arr=cost_arr, worst_cost=0.0001,
        atr_multiplier=2.0, armable_arr=armable, forced_initial_direction=None, max_bars_in_position=None,
        starting_equity=GAMBLE_CAPITAL, symbol="TEST",
    )
    reasons = [t.exit_reason for t in trades]
    assert "reversal" in reasons, f"expected a reversal once price pulled back from its trailing extreme, got exit reasons: {reasons}"


def test_forced_initial_direction_opens_at_bar_zero_open_and_marks_intrabar_only():
    n = 3
    index = _hourly_index(n)
    open_ = np.array([100.0, 110.0, 111.0])
    close = np.array([105.0, 110.5, 111.5])
    atr_arr = np.full(n, 1000.0)  # huge ATR -> no reversal ever triggers, isolates the entry-marking behavior
    cost_arr = np.zeros(n)

    equity_out, trades, _total_cost = engine20.simulate_breakout_reversal(
        index=index, open_arr=open_, close_arr=close, atr_arr=atr_arr, cost_arr=cost_arr, worst_cost=0.0,
        atr_multiplier=2.0, armable_arr=None, forced_initial_direction=1.0, max_bars_in_position=None,
        starting_equity=GAMBLE_CAPITAL, symbol="TEST",
    )
    # bar0 is the entry bar: exposure should be close[0]/open[0] - 1 = 105/100 - 1 = 5%, NOT
    # close[0]/close[-1] (no prior close exists) and NOT a full-bar return from some other base.
    expected_bar0 = GAMBLE_CAPITAL * (1.0 + (105.0 / 100.0 - 1.0))
    assert equity_out[0] == pytest.approx(expected_bar0)


def test_max_bars_in_position_forces_close_even_without_a_reversal_signal():
    n = 5
    index = _hourly_index(n)
    open_ = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    close = np.array([101.0, 102.0, 103.0, 104.0, 105.0])  # steadily rising -- never triggers a reversal
    atr_arr = np.full(n, 1000.0)  # huge ATR -> reversal never fires
    cost_arr = np.zeros(n)

    _equity_out, trades, _total_cost = engine20.simulate_breakout_reversal(
        index=index, open_arr=open_, close_arr=close, atr_arr=atr_arr, cost_arr=cost_arr, worst_cost=0.0,
        atr_multiplier=2.0, armable_arr=None, forced_initial_direction=1.0, max_bars_in_position=2,
        starting_equity=GAMBLE_CAPITAL, symbol="TEST",
    )
    window_close = [t for t in trades if t.exit_reason == "window_close"]
    assert len(window_close) == 1
    assert window_close[0].exit_time == index[2]  # entry at bar0, forced close 2 bars later


def test_g1_floor_never_goes_negative_on_catastrophic_single_bar_move():
    n = 4
    index = _hourly_index(n)
    open_ = np.array([100.0, 100.0, 100.0, 100.0])
    close = np.array([101.0, 600.0, 600.0, 600.0])  # bar1: 101->600 (+494% from prior close) while short -> breaches -100% if unclipped
    atr_arr = np.full(n, 1000.0)
    cost_arr = np.zeros(n)

    equity_out, trades, _total_cost = engine20.simulate_breakout_reversal(
        index=index, open_arr=open_, close_arr=close, atr_arr=atr_arr, cost_arr=cost_arr, worst_cost=0.0,
        atr_multiplier=2.0, armable_arr=None, forced_initial_direction=-1.0, max_bars_in_position=None,
        starting_equity=GAMBLE_CAPITAL, symbol="TEST",
    )
    assert (equity_out >= 0.0).all()
    assert any(t.exit_reason == "liquidated" for t in trades)
    assert all(t.pnl_fraction >= -1.0 - 1e-9 for t in trades)


# ---------------------------------------------------------------------------
# skewness / top_decile_contribution
# ---------------------------------------------------------------------------


def test_skewness_zero_for_symmetric_distribution():
    values = np.array([1.0, 2.0, 3.0])  # symmetric around the mean
    assert gates20.skewness(values) == pytest.approx(0.0, abs=1e-9)


def test_skewness_positive_for_right_tailed_distribution():
    many_small_losses = [-1.0] * 20
    one_huge_win = [50.0]
    values = np.array(many_small_losses + one_huge_win)
    skew = gates20.skewness(values)
    assert skew is not None and skew > 0.0


def test_skewness_none_below_minimum_sample_size():
    assert gates20.skewness(np.array([1.0, 2.0])) is None


def test_top_decile_contribution_near_one_when_concentrated():
    values = np.array([-1.0] * 18 + [1.0, 100.0])  # 20 trades, one dominant winner
    contribution = gates20.top_decile_contribution(values, fraction=0.10)
    assert contribution is not None and contribution > 0.95


def test_top_decile_contribution_lower_when_wins_are_even():
    # 20 trades, top 10% BY COUNT = 2 trades. With 10 equal +1.0 winners and 10 -1.0 losers,
    # gross profit is the WINNERS-only pool ($10), and the top 2 (tied) winners are 2 of those
    # 10 -> 20% of gross profit. This is intentionally NOT ~10%: top_decile_contribution is
    # "top 10% of trade COUNT vs gross profit dollars", and only half of these trades are
    # winners at all, so 10%-of-count naturally lands at 20%-of-winners here -- the point of
    # this test is that it is UNAMBIGUOUSLY LOWER than the concentrated-single-winner case
    # above (>95%), not that it hits some particular intuitive number.
    even_values = np.array([-1.0] * 10 + [1.0] * 10)
    concentrated_values = np.array([-1.0] * 18 + [1.0, 100.0])
    even_contribution = gates20.top_decile_contribution(even_values, fraction=0.10)
    concentrated_contribution = gates20.top_decile_contribution(concentrated_values, fraction=0.10)
    assert even_contribution is not None and even_contribution == pytest.approx(0.2)
    assert even_contribution < concentrated_contribution


def test_top_decile_contribution_none_when_no_gross_profit():
    values = np.array([-1.0, -2.0, -3.0])
    assert gates20.top_decile_contribution(values) is None


# ---------------------------------------------------------------------------
# full_period_annualized / calendar_year_return
# ---------------------------------------------------------------------------


def test_full_period_annualized_matches_hand_computed_cagr():
    index = pd.date_range("2020-01-01", periods=2, freq="365D", tz=UTC)  # exactly 365 days apart
    equity = pd.Series([100.0, 121.0], index=index)
    cagr = gates20.full_period_annualized(equity)
    assert cagr == pytest.approx(0.21, abs=1e-9)  # 100 -> 121 over exactly 1 year = +21%


def test_calendar_year_return_uses_prior_year_end_as_base():
    index = pd.date_range("2020-12-01", periods=90, freq="1D", tz=UTC)
    values = np.linspace(100.0, 190.0, 90)
    equity = pd.Series(values, index=index)
    year_2021_return = gates20.calendar_year_return(equity, 2021)
    dec31_value = float(equity[equity.index.year == 2020].iloc[-1])
    end_2021_value = float(equity[equity.index.year == 2021].iloc[-1])
    assert year_2021_return == pytest.approx(end_2021_value / dec31_value - 1.0)


def test_calendar_year_return_none_when_year_not_covered():
    index = pd.date_range("2020-01-01", periods=10, freq="1D", tz=UTC)
    equity = pd.Series(np.linspace(100, 110, 10), index=index)
    assert gates20.calendar_year_return(equity, 2025) is None


# ---------------------------------------------------------------------------
# pad_to_calendar / combine_portfolio
# ---------------------------------------------------------------------------


def test_pad_to_calendar_flat_before_start_then_follows_series():
    calendar = pd.date_range("2020-01-01", periods=10, freq="1D", tz=UTC)
    sparse = pd.Series([50.0, 60.0], index=[calendar[4], calendar[7]])
    padded = engine20.pad_to_calendar(sparse, calendar, initial_value=25.0)
    assert (padded.iloc[:4].to_numpy() == 25.0).all()  # before the sleeve's own data exists: flat at initial capital
    assert padded.iloc[4] == pytest.approx(50.0)
    assert padded.iloc[5:7].to_numpy() == pytest.approx(50.0)  # ffill the gap
    assert padded.iloc[7] == pytest.approx(60.0)
    assert padded.iloc[8:].to_numpy() == pytest.approx(60.0)  # ffill past the sleeve's own last observation


def test_combine_portfolio_is_dollar_additive_and_pads_gamble_leg():
    calendar = pd.date_range("2020-01-01", periods=5, freq="1D", tz=UTC)
    stable = pd.Series([75.0, 76.0, 77.0, 78.0, 79.0], index=calendar)
    gamble = pd.Series([25.0, 30.0], index=[calendar[2], calendar[3]])  # starts mid-way
    combined = engine20.combine_portfolio(stable, gamble, gamble_initial=25.0)
    assert combined["total"].iloc[0] == pytest.approx(75.0 + 25.0)  # gamble padded flat at 25 before its own data starts
    assert combined["total"].iloc[2] == pytest.approx(77.0 + 25.0)
    assert combined["total"].iloc[3] == pytest.approx(78.0 + 30.0)
    assert combined["total"].iloc[4] == pytest.approx(79.0 + 30.0)  # ffill past gamble's own last observation


# ---------------------------------------------------------------------------
# Gate boundary logic (synthetic, independent of any engine run).
# ---------------------------------------------------------------------------


def test_gate_g1_fails_if_equity_ever_negative():
    index = pd.date_range("2020-01-01", periods=3, freq="1D", tz=UTC)
    equity = pd.Series([25.0, -1.0, 5.0], index=index)  # should never happen given the engine's own floor, but the GATE itself must catch it
    outcome = gates20.gate_g1_structural_loss_cap(equity, trades_payload=[])
    assert outcome.status == "FAIL"


def test_gate_g1_passes_when_floored_at_zero():
    index = pd.date_range("2020-01-01", periods=3, freq="1D", tz=UTC)
    equity = pd.Series([25.0, 0.0, 0.0], index=index)
    outcome = gates20.gate_g1_structural_loss_cap(equity, trades_payload=[{"pnl_fraction": -1.0}])
    assert outcome.status == "PASS"


def test_gate_g4_pass_only_when_strictly_beats_reference():
    index = pd.date_range("2020-01-01", periods=2, freq="365D", tz=UTC)
    beats = pd.Series([100.0, 130.0], index=index)
    misses = pd.Series([100.0, 105.0], index=index)
    outcome_pass, _ = gates20.gate_g4_beats_stable_solo(beats, stable_solo_cagr=0.1027)
    outcome_fail, _ = gates20.gate_g4_beats_stable_solo(misses, stable_solo_cagr=0.1027)
    assert outcome_pass.status == "PASS"
    assert outcome_fail.status == "FAIL"


def test_gate_g5_fails_if_either_worst_year_is_worse():
    index = pd.date_range("2021-12-25", periods=800, freq="1D", tz=UTC)
    combined = pd.Series(np.linspace(100.0, 100.0, len(index)), index=index)
    # force 2022 to underperform (flat) while 2025 outperforms heavily
    combined.loc[combined.index.year == 2025] = np.linspace(150.0, 300.0, int((combined.index.year == 2025).sum()))
    solo = pd.Series(np.linspace(100.0, 250.0, len(index)), index=index)  # solo grows steadily every year, including 2022
    outcome, detail = gates20.gate_g5_worst_year_defense(combined, solo, years=(2022, 2025))
    assert outcome.status == "FAIL"
    assert detail["2022"]["no_degradation"] is False


# ---------------------------------------------------------------------------
# Real-cache smoke tests (fast paths only -- V1 and the stable-leg loader run in well under a
# second; the heavier candidates (V2/V3/V5, several seconds each) are exercised by actually
# running research/wave20_convex/run_wave20.py, not repeated here).
# ---------------------------------------------------------------------------


def test_load_stable_leg_rescales_from_i5_active_capital_basis():
    stable, info = engine20.load_stable_leg(STABLE_CAPITAL)
    assert info["source_active_capital_usdt"] == pytest.approx(90.0)
    assert float(stable.iloc[0]) == pytest.approx(STABLE_CAPITAL, rel=0.01)
    assert (stable > 0.0).all()


def test_run_v1_smoke_against_real_cache_never_goes_negative():
    mapping = _mapping(anchor_log_volume=(3.0, 9.0), anchor_bp=(8.0, 0.02))
    result = engine20.run_v1(mapping=mapping)
    assert (result.equity.dropna() >= 0.0).all()
    assert result.symbols_used == ("BTCUSDT",)
    assert result.metadata["n_trades"] == len(result.trades)
    for trade in result.trades:
        assert trade.pnl_fraction >= -1.0 - 1e-6


def test_dataio20_wave1_and_wave3_universes_are_nonempty_and_disjoint_use_cases():
    v2_universe = dataio20.wave1_symbols_with_funding()
    v3_universe = dataio20.wave3_symbols()
    assert len(v2_universe) > 10
    assert len(v3_universe) > 100
    assert "BTCUSDT" in v2_universe
    assert "BTCUSDT" in v3_universe
