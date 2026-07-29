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

from research.wave25_gamble import engine25, gates25
from research.wave25_gamble import indicators25 as ind

UTC = "UTC"
SYMBOL = "TEST"


def _hourly_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1h", tz=UTC)


def _frames(index, open_, high, low, close, risk_atr, cost, signal, symbol: str = SYMBOL) -> dict[str, pd.DataFrame]:
    return {
        "open": pd.DataFrame({symbol: open_}, index=index),
        "high": pd.DataFrame({symbol: high}, index=index),
        "low": pd.DataFrame({symbol: low}, index=index),
        "close": pd.DataFrame({symbol: close}, index=index),
        "risk_atr": pd.DataFrame({symbol: risk_atr}, index=index),
        "cost": pd.DataFrame({symbol: cost}, index=index),
        "signal": pd.DataFrame({symbol: signal}, index=index),
    }


# ===========================================================================
# 1. Indicator accuracy (지표 정확도 검증)
# ===========================================================================


def test_true_range_uses_max_of_three_components():
    ohlc = pd.DataFrame(
        {"open": [100.0, 100.0, 100.0], "high": [105.0, 108.0, 101.0], "low": [95.0, 99.0, 90.0], "close": [100.0, 100.0, 100.0]},
        index=_hourly_index(3),
    )
    tr = ind.true_range(ohlc)
    assert tr.iloc[0] == pytest.approx(10.0)  # no prior close -> high-low only
    assert tr.iloc[1] == pytest.approx(9.0)  # max(high-low=9, |high-prevclose|=8, |low-prevclose|=1)
    assert tr.iloc[2] == pytest.approx(11.0)  # max(high-low=11, |high-prevclose|=1, |low-prevclose|=10)


def test_atr_is_simple_rolling_mean_with_min_periods_warmup():
    ohlc = pd.DataFrame({"open": [10.0] * 5, "high": [12.0] * 5, "low": [8.0] * 5, "close": [10.0] * 5}, index=_hourly_index(5))
    result = ind.atr(ohlc, window=2)
    assert pd.isna(result.iloc[0])  # insufficient history
    assert result.iloc[1] == pytest.approx(4.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_matches_hand_computed_recursive_formula():
    # alpha = 2/(span+1) = 0.5 for span=3; EMA[0]=1, EMA[1]=0.5*2+0.5*1=1.5, EMA[2]=0.5*3+0.5*1.5=2.25, EMA[3]=0.5*4+0.5*2.25=3.125
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = ind.ema(series, span=3)
    assert result.tolist() == pytest.approx([1.0, 1.5, 2.25, 3.125])


def test_macd_histogram_equals_macd_minus_signal_everywhere_defined():
    close = pd.Series(np.sin(np.linspace(0, 10, 200)) * 10 + 100, index=_hourly_index(200))
    frame = ind.macd(close, fast=12, slow=26, signal=9)
    residual = (frame["histogram"] - (frame["macd"] - frame["signal"])).dropna()
    assert residual.abs().max() < 1e-9


def test_macd_masks_warmup_before_slow_minus_one():
    close = pd.Series(np.linspace(100.0, 120.0, 60), index=_hourly_index(60))
    frame = ind.macd(close, fast=12, slow=26, signal=9)
    assert frame["macd"].iloc[: 26 - 1].isna().all()
    assert frame["macd"].iloc[26 - 1 :].notna().all()


def test_wilder_smoothed_adx_stays_within_0_100_bounds_regression():
    """Regression test for a bug found during implementation: the recursive Wilder update
    smoothed[t] = smoothed[t-1] - smoothed[t-1]/N + raw[t] (WITHOUT dividing the new raw[t] by
    N) is correct for accumulating TR/+DM/-DM (their scale cancels inside the +DI/-DI ratio),
    but is WRONG when applied a second time to DX (already a bounded 0-100 percentage) to
    produce ADX -- it diverges unboundedly (observed >1300 in manual testing before the fix).
    ADX must stay within [0, 100] for any input."""
    n = 80
    close = pd.Series(np.linspace(100.0, 160.0, n), index=_hourly_index(n))
    ohlc = pd.DataFrame({"open": close.shift(1).fillna(100.0), "high": close + 1.0, "low": close - 1.0, "close": close})
    adx = ind.adx_dmi(ohlc, window=14)["adx"].dropna()
    assert len(adx) > 0
    assert (adx >= 0.0).all() and (adx <= 100.0 + 1e-9).all()


def test_adx_dmi_strong_monotonic_uptrend_has_dominant_plus_di_and_high_adx():
    n = 60
    close = pd.Series(np.linspace(100.0, 160.0, n), index=_hourly_index(n))
    ohlc = pd.DataFrame({"open": close.shift(1).fillna(100.0), "high": close + 1.0, "low": close - 1.0, "close": close})
    frame = ind.adx_dmi(ohlc, window=14)
    tail = frame.dropna().iloc[-10:]
    assert (tail["plus_di"] > tail["minus_di"]).all()
    assert (tail["minus_di"].to_numpy() == 0.0).all()  # no down-moves at all in a pure monotonic series
    assert (tail["adx"] > 90.0).all()  # a perfectly directional trend should read a near-maximal ADX


def test_supertrend_direction_flips_up_on_a_breakout_through_the_upper_band():
    close = np.array([100.0] * 10 + [200.0] * 10)
    high = close + 2.0
    low = close - 2.0
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    ohlc = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=_hourly_index(20))
    frame = ind.supertrend(ohlc, window=5, multiplier=2.0)
    assert frame["direction"].iloc[9] == pytest.approx(-1.0)  # still flat/pre-breakout
    assert frame["direction"].iloc[10] == pytest.approx(1.0)  # flips the instant close(200) > final_upper (108 at that bar)


def test_supertrend_line_tracks_lower_band_in_uptrend_and_upper_band_in_downtrend():
    close = np.array([100.0] * 10 + [200.0] * 10)
    high = close + 2.0
    low = close - 2.0
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    ohlc = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=_hourly_index(20))
    frame = ind.supertrend(ohlc, window=5, multiplier=2.0).dropna()
    up = frame[frame["direction"] > 0.0]
    down = frame[frame["direction"] < 0.0]
    assert (up["line"] == up["final_lower"]).all()
    assert (down["line"] == down["final_upper"]).all()


def test_keltner_channel_bands_are_symmetric_around_middle_by_multiplier_times_atr():
    n = 40
    close = pd.Series(100.0 + np.cumsum(np.sin(np.linspace(0, 6, n))), index=_hourly_index(n))
    ohlc = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": close + 1.5, "low": close - 1.5, "close": close})
    frame = ind.keltner_channel(ohlc, window=10, atr_window=10, multiplier=2.0).dropna()
    upper_gap = frame["upper"] - frame["middle"]
    lower_gap = frame["middle"] - frame["lower"]
    assert upper_gap.to_numpy() == pytest.approx(lower_gap.to_numpy())  # pytest.approx on arrays already returns a single bool
    assert (upper_gap > 0.0).all()


def test_stochastic_percent_k_matches_hand_computed_values():
    index = _hourly_index(5)
    high = pd.Series([10.0, 12.0, 14.0, 13.0, 11.0], index=index)
    low = pd.Series([8.0, 9.0, 11.0, 10.0, 9.0], index=index)
    close = pd.Series([9.0, 11.0, 13.0, 11.0, 10.0], index=index)
    ohlc = pd.DataFrame({"open": close.shift(1).fillna(9.0), "high": high, "low": low, "close": close})
    result = ind.stochastic(ohlc, k_window=3, d_window=2)
    assert result["percent_k"].iloc[2] == pytest.approx(100.0 * 5.0 / 6.0)
    assert result["percent_k"].iloc[3] == pytest.approx(40.0)
    assert result["percent_k"].iloc[4] == pytest.approx(20.0)


def test_stochastic_percent_k_bounded_0_to_100():
    n = 100
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=_hourly_index(n))
    ohlc = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": close + rng.uniform(0, 2, n), "low": close - rng.uniform(0, 2, n), "close": close})
    k = ind.stochastic(ohlc, k_window=14, d_window=3)["percent_k"].dropna()
    assert (k >= -1e-9).all() and (k <= 100.0 + 1e-9).all()


def test_moving_average_slope_sign_matches_trend_direction():
    up = pd.Series(np.linspace(100.0, 200.0, 30))
    down = pd.Series(np.linspace(200.0, 100.0, 30))
    assert (ind.moving_average_slope(up, ma_window=5, slope_lookback=5).dropna() > 0.0).all()
    assert (ind.moving_average_slope(down, ma_window=5, slope_lookback=5).dropna() < 0.0).all()


# ===========================================================================
# 2. MTF lookahead prevention (다중시간대 정합 룩어헤드 방지)
# ===========================================================================


def test_align_daily_to_intraday_never_leaks_same_day_value_before_day_closes():
    daily = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="1D", tz=UTC))
    hourly_index = pd.date_range("2024-01-02 00:00", periods=25, freq="1h", tz=UTC)  # all 24h of Jan2 + Jan3 00:00
    aligned = ind.align_daily_to_intraday(daily, hourly_index)
    assert (aligned.iloc[:24] == 1.0).all()  # Jan2's own hours only ever see Jan1's value (never Jan2's OWN 2.0)
    assert aligned.iloc[24] == 2.0  # Jan3 00:00 is the first hour allowed to see Jan2's close


def test_mtf_confluence_signal_only_fires_in_daily_trend_direction():
    """B5's own signal: a qualifying 1H momentum breakout must still be suppressed if the
    (lagged) daily trend disagrees with its direction."""
    n = 80
    index = _hourly_index(n)
    close = pd.Series(100.0, index=index)
    close.iloc[40:] = np.linspace(100.0, 140.0, n - 40)  # a sharp upward breakout partway through
    ohlc = pd.DataFrame({"open": close.shift(1).fillna(100.0), "high": close + 0.5, "low": close - 0.5, "close": close})
    # Daily history must start well BEFORE the hourly window so MA(3)+slope(1)+the 1-day lag
    # are already warmed up by index[0] -- otherwise hourly_trend is NaN for the entire hourly
    # window under test and the assertion would pass vacuously for the wrong reason.
    daily_index = pd.date_range(index[0].normalize() - pd.Timedelta(days=10), index[-1].normalize() + pd.Timedelta(days=1), freq="1D", tz=UTC)

    from research.wave25_gamble.configs25 import B5Config

    config = B5Config(candidate_id="B5", daily_ma_window=3, daily_slope_lookback=1, breakout_lookback_bars=5, breakout_atr_window=5, breakout_atr_multiplier=0.5)

    daily_down = pd.DataFrame({"close": np.linspace(200.0, 50.0, len(daily_index))}, index=daily_index)  # daily trend DOWN throughout
    signal_down_trend = engine25.mtf_confluence_signal(ohlc, daily_down, config)
    assert (signal_down_trend > 0.0).sum() == 0  # an upward 1H breakout must never fire long while the daily trend disagrees

    daily_up = pd.DataFrame({"close": np.linspace(50.0, 200.0, len(daily_index))}, index=daily_index)  # daily trend UP throughout
    signal_up_trend = engine25.mtf_confluence_signal(ohlc, daily_up, config)
    assert (signal_up_trend > 0.0).sum() >= 1  # the same breakout DOES fire once the daily trend agrees


# ===========================================================================
# 3. Convexity enforcement / stop trigger / trailing (볼록성 강제, 손절 트리거, 트레일링)
# ===========================================================================


def test_hard_stop_loss_triggers_and_caps_the_loss():
    index = _hourly_index(4)
    open_ = [99.0, 100.0, 99.0, 97.0]
    high = [100.0, 101.0, 99.5, 98.0]
    low = [98.0, 99.0, 97.0, 96.0]
    close = [99.5, 100.5, 97.5, 97.0]
    risk_atr = [2.0] * 4
    cost = [0.0] * 4
    signal = [1.0, 0.0, 0.0, 0.0]
    frames = _frames(index, open_, high, low, close, risk_atr, cost, signal)
    result = engine25.run_multi_symbol_convex(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", starting_equity=25.0, max_bars_in_position=100)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(98.0)  # min(3%*entry=3.0, 1xATR=2.0) -> stop_distance=2.0 -> stop_price=98
    assert trade.pnl_usdt < 0.0
    assert trade.pnl_fraction > -0.05  # loss stayed small/bounded, not catastrophic


def test_hard_stop_distance_picks_the_tighter_of_pct_and_atr():
    """SPEC.md: "-3% 또는 -1xATR 중 가까운 쪽" -- with a LARGE ATR (10.0, implying a 10-point
    stop distance), the 3%-of-entry distance (3.0 points, stop=97) must bind instead, so a
    drop to 96 (below the atr-implied stop of 90 but below the pct-implied stop of 97) must
    still trigger a stop."""
    index = _hourly_index(4)
    open_ = [99.0, 100.0, 99.0, 96.0]
    high = [100.0, 101.0, 99.5, 97.0]
    low = [98.0, 99.0, 96.0, 95.0]
    close = [99.5, 100.5, 96.5, 96.0]
    risk_atr = [10.0] * 4  # atr-implied distance would be 10.0 (stop=90) -- must NOT bind
    cost = [0.0] * 4
    signal = [1.0, 0.0, 0.0, 0.0]
    frames = _frames(index, open_, high, low, close, risk_atr, cost, signal)
    result = engine25.run_multi_symbol_convex(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", starting_equity=25.0, max_bars_in_position=100)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(97.0)  # the TIGHTER pct-based stop (97), not the looser atr-based one (90)


def test_stop_loss_fills_gap_aware_at_the_worse_of_stop_price_and_open():
    """If the bar that breaches the stop ALSO gapped open below the stop level, the fill must
    be the (worse) open price, not the stale stop level -- a stop order cannot fill better
    than the market actually gapped through."""
    index = _hourly_index(4)
    open_ = [99.0, 100.0, 95.0, 90.0]  # bar2 opens at 95, already through the stop_price=98
    high = [100.0, 101.0, 96.0, 91.0]
    low = [98.0, 99.0, 93.0, 89.0]
    close = [99.5, 100.5, 94.0, 90.0]
    risk_atr = [2.0] * 4
    cost = [0.0] * 4
    signal = [1.0, 0.0, 0.0, 0.0]
    frames = _frames(index, open_, high, low, close, risk_atr, cost, signal)
    result = engine25.run_multi_symbol_convex(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", starting_equity=25.0, max_bars_in_position=100)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(95.0)  # the gapped OPEN, not the stale stop_price=98.0


def test_trailing_exit_locks_in_a_gain_not_a_loss():
    """After a large favorable run, a pullback exits via the TRAILING stop at a price well
    above entry -- this is the mechanism SPEC.md requires for all profit-taking (no fixed
    take-profit exists anywhere in engine25)."""
    index = _hourly_index(5)
    open_ = [99.0, 100.0, 104.0, 110.0, 148.0]
    high = [100.0, 105.0, 112.0, 155.0, 149.0]
    low = [98.0, 99.0, 103.0, 109.0, 140.0]
    close = [99.5, 104.0, 110.0, 150.0, 145.0]
    risk_atr = [2.0] * 5
    cost = [0.0] * 5
    signal = [1.0, 0.0, 0.0, 0.0, 0.0]
    frames = _frames(index, open_, high, low, close, risk_atr, cost, signal)
    result = engine25.run_multi_symbol_convex(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", starting_equity=25.0, max_bars_in_position=100)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "trailing_exit"
    assert trade.exit_price == pytest.approx(147.0)  # extreme(150 as of prior bar) - 1.5*ATR(2.0) = 147, gap-checked against this bar's open(148)
    assert trade.exit_price > trade.entry_price  # profit locked in, not a loss
    assert trade.pnl_usdt > 0.0


def test_no_fixed_take_profit_exit_reason_ever_appears():
    """Structural regression guard: the convex lifecycle's entire vocabulary of exit reasons
    must never include a fixed take-profit -- profit-taking is trailing-only by construction
    (SPEC.md: "고정 익절 금지")."""
    allowed = {"stop_loss", "trailing_exit", "max_hold", "signal_reversal", "liquidated", "end_of_data"}
    index = _hourly_index(5)
    open_ = [99.0, 100.0, 104.0, 110.0, 148.0]
    high = [100.0, 105.0, 112.0, 155.0, 149.0]
    low = [98.0, 99.0, 103.0, 109.0, 140.0]
    close = [99.5, 104.0, 110.0, 150.0, 145.0]
    frames = _frames(index, open_, high, low, close, [2.0] * 5, [0.0] * 5, [1.0, 0.0, 0.0, 0.0, 0.0])
    result = engine25.run_multi_symbol_convex(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", starting_equity=25.0, max_bars_in_position=100)
    observed_reasons = {t.exit_reason for t in result.trades}
    assert observed_reasons.issubset(allowed)
    assert "take_profit" not in observed_reasons


def test_max_hold_forces_close_even_without_any_stop_or_trailing_signal():
    n = 6
    index = _hourly_index(n)
    open_ = [99.0] + [100.0 + i for i in range(n - 1)]
    high = [x + 1.0 for x in open_]
    low = [x - 1.0 for x in open_]
    close = [x + 0.2 for x in open_]
    risk_atr = [1000.0] * n  # huge ATR -> stop/trailing never triggers
    signal = [1.0] + [0.0] * (n - 1)
    frames = _frames(index, open_, high, low, close, risk_atr, [0.0] * n, signal)
    result = engine25.run_multi_symbol_convex(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", starting_equity=25.0, max_bars_in_position=2)
    max_hold_trades = [t for t in result.trades if t.exit_reason == "max_hold"]
    assert len(max_hold_trades) == 1
    assert max_hold_trades[0].entry_time == index[1]
    assert max_hold_trades[0].exit_time == index[3]  # entry at bar1, forced close 2 bars later


def test_no_overlapping_positions_across_symbols_single_sleeve_priority_scan():
    """SPEC.md's implicit "gross <= sleeve" (1x leverage, one sleeve): when TWO symbols both
    qualify on the same bar, only the higher-priority symbol opens -- the other's signal is
    simply not acted on while a position is held."""
    index = _hourly_index(4)
    open_a = [99.0, 100.0, 101.0, 102.0]
    open_b = [199.0, 200.0, 201.0, 202.0]
    flat = lambda base: [base] * 4  # noqa: E731
    frames = {
        "open": pd.DataFrame({"A": open_a, "B": open_b}, index=index),
        "high": pd.DataFrame({"A": [x + 1 for x in open_a], "B": [x + 1 for x in open_b]}, index=index),
        "low": pd.DataFrame({"A": [x - 1 for x in open_a], "B": [x - 1 for x in open_b]}, index=index),
        "close": pd.DataFrame({"A": open_a, "B": open_b}, index=index),
        "risk_atr": pd.DataFrame({"A": flat(1000.0), "B": flat(1000.0)}, index=index),
        "cost": pd.DataFrame({"A": flat(0.0), "B": flat(0.0)}, index=index),
        "signal": pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0], "B": [1.0, 0.0, 0.0, 0.0]}, index=index),
    }
    result = engine25.run_multi_symbol_convex(frames, priority=("A", "B"), worst_cost=0.0, candidate_id="TESTC", starting_equity=25.0, max_bars_in_position=100)
    trade_symbols = {t.symbol for t in result.trades}
    assert trade_symbols == {"A"}  # B's simultaneous signal never got a position (A has priority)
    trades_payload = [
        {"entry_time": str(t.entry_time), "exit_time": str(t.exit_time), "entry_equity_usdt": t.entry_equity_usdt} for t in result.trades
    ]
    assert gates25.no_overlapping_positions(trades_payload)


# ===========================================================================
# 4. Ensemble counting (앙상블 카운팅, B7)
# ===========================================================================


def test_sticky_state_holds_last_direction_until_opposite_signal():
    signal = pd.Series([0.0, 1.0, 0.0, 0.0, -1.0, 0.0], index=_hourly_index(6))
    state = engine25.sticky_state(signal)
    assert state.tolist() == [0.0, 1.0, 1.0, 1.0, -1.0, -1.0]


def test_sticky_state_active_mask_zeroes_regime_when_inactive():
    signal = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=_hourly_index(5))
    mask = pd.Series([True, True, True, False, True], index=signal.index)
    state = engine25.sticky_state(signal, mask)
    assert state.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]  # forced to 0 exactly where the mask (e.g. B2's ADX>25) is False


def test_ensemble_signal_fires_only_on_the_bar_agreement_first_reaches_threshold():
    index = _hourly_index(4)
    m1 = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)  # bullish from bar1 onward
    m2 = pd.Series([0.0, 0.0, 1.0, 0.0], index=index)  # bullish from bar2 onward
    m3 = pd.Series([0.0, 0.0, 1.0, 0.0], index=index)  # bullish from bar2 onward
    signal = engine25.ensemble_signal({"M1": m1, "M2": m2, "M3": m3}, {"M1": None, "M2": None, "M3": None}, min_agree=3)
    # 3-way agreement is reached at bar2 and PERSISTS through bar3 -- must fire ONCE, at bar2 only
    assert signal.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_ensemble_signal_requires_the_full_min_agree_threshold():
    index = _hourly_index(4)
    m1 = pd.Series([0.0, 1.0, 0.0, 0.0], index=index)
    m2 = pd.Series([0.0, 0.0, 1.0, 0.0], index=index)
    m3 = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)  # never agrees
    signal = engine25.ensemble_signal({"M1": m1, "M2": m2, "M3": m3}, {"M1": None, "M2": None, "M3": None}, min_agree=3)
    assert (signal == 0.0).all()  # only 2/3 ever agree simultaneously -- below the 3-of-N threshold


def test_ensemble_signal_counts_bearish_agreement_independently_of_bullish():
    index = _hourly_index(3)
    m1 = pd.Series([0.0, -1.0, 0.0], index=index)
    m2 = pd.Series([0.0, -1.0, 0.0], index=index)
    m3 = pd.Series([0.0, -1.0, 0.0], index=index)
    signal = engine25.ensemble_signal({"M1": m1, "M2": m2, "M3": m3}, {"M1": None, "M2": None, "M3": None}, min_agree=3)
    assert signal.tolist() == [0.0, -1.0, 0.0]


def test_ensemble_signal_for_symbol_uses_all_six_members():
    n = 400
    close = pd.Series(100.0 + np.cumsum(np.sin(np.linspace(0, 20, n)) * 2.0), index=_hourly_index(n))
    ohlc = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": close + 1.0, "low": close - 1.0, "close": close})
    daily_index = pd.date_range(ohlc.index[0].normalize(), ohlc.index[-1].normalize() + pd.Timedelta(days=1), freq="1D", tz=UTC)
    daily = pd.DataFrame({"close": 100.0 + np.cumsum(np.sin(np.linspace(0, 20, len(daily_index))))}, index=daily_index)
    signals, masks = engine25._member_signals_and_masks(ohlc, daily)
    assert set(signals.keys()) == {"B1", "B2", "B3", "B4", "B5", "B6"}
    result = engine25.ensemble_signal(signals, masks, min_agree=3)
    assert len(result) == n  # runs end-to-end without raising, aligned to the full hourly index


# ===========================================================================
# 5. Gates P1-P5 (stricter-than-wave20 P1 bootstrap clause, new P2 dollar cap, P4, P5)
# ===========================================================================


def test_gate_p1_undetermined_below_min_trade_count():
    pnls = np.array([1.0, -1.0, 2.0])
    outcome, _ = gates25.gate_p1_convexity(pnls, final_equity=27.0, seed=1)
    assert outcome.status == "UNDETERMINED"


def test_gate_p1_fails_when_bootstrap_lower_bound_is_not_positive_even_if_skew_and_decile_pass():
    """P1 is SPEC.md-stricter than wave20's own G3: skew>0 and top-decile>=50% both pass here
    (a single dominant winner among 9 equal losers), but with only 10 trades the bootstrap's
    5th-percentile skew (P1's third clause) lands at exactly 0.0 -- not >0.0 -- because roughly
    35% of resamples exclude the one winning trade entirely (a flat, zero-skew draw), which is
    well above the 5% cutoff. P1 must FAIL here even though wave20's G3 would have PASSED on
    the same data (G3 never checked the bootstrap bound itself, only disclosed it)."""
    pnls = np.array([-1.0] * 9 + [50.0])
    outcome, diagnostics = gates25.gate_p1_convexity(pnls, final_equity=25.0 + 41.0, seed=12345)
    assert diagnostics["skew"] > 0.0
    assert diagnostics["top_decile_contribution_of_gross_profit"] >= 0.50
    assert diagnostics["bootstrap"]["p05"] <= 0.0
    assert outcome.status == "FAIL"


def test_gate_p2_flags_single_trade_loss_exceeding_25_dollars_even_when_ruin_probability_is_low():
    """The NEW dollar cap (SPEC.md P2's "단일 최대손실 <= $25"): once the sleeve has grown well
    past $25, a single stop/liquidation event can lose more than $25 in absolute terms while
    the system-wide Monte Carlo ruin probability (a portfolio-path measure over many small
    daily moves) stays comfortably low. P2 must catch this even though the ruin-probability
    half of the gate passes."""
    rng = np.random.default_rng(3)
    index = pd.date_range("2020-01-01", periods=200, freq="1D", tz=UTC)
    combined_equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.003, len(index))), index=index)
    trades_payload = [
        {"pnl_usdt": 5.0, "entry_equity_usdt": 40.0},
        {"pnl_usdt": -30.0, "entry_equity_usdt": 120.0},  # a single trade lost $30 > the $25 cap
        {"pnl_usdt": 2.0, "entry_equity_usdt": 90.0},
    ]
    outcome, payload = gates25.gate_p2_bankruptcy(combined_equity, trades_payload, seed=99)
    assert payload["max_single_trade_loss_usdt"] == pytest.approx(30.0)
    assert outcome.status == "FAIL"


def test_max_single_trade_loss_usdt_ignores_winning_trades():
    trades_payload = [{"pnl_usdt": 10.0}, {"pnl_usdt": -4.0}, {"pnl_usdt": -12.5}, {"pnl_usdt": 3.0}]
    assert gates25.max_single_trade_loss_usdt(trades_payload) == pytest.approx(12.5)
    assert gates25.max_single_trade_loss_usdt([{"pnl_usdt": 1.0}, {"pnl_usdt": 2.0}]) == 0.0


def test_gate_p3_requires_strictly_beating_baseline():
    assert gates25.gate_p3_beats_baseline(140.0, 138.48).status == "PASS"
    assert gates25.gate_p3_beats_baseline(138.48, 138.48).status == "FAIL"  # tie does not count
    assert gates25.gate_p3_beats_baseline(100.0, 138.48).status == "FAIL"


def test_rolling_window_return_matches_hand_computation():
    index = pd.date_range("2020-01-01", periods=5, freq="1D", tz=UTC)
    equity = pd.Series([100.0, 110.0, 120.0, 90.0, 150.0], index=index)
    result = gates25.rolling_window_return(equity, window_days=2)
    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(120.0 / 100.0 - 1.0)
    assert result.iloc[3] == pytest.approx(90.0 / 110.0 - 1.0)
    assert result.iloc[4] == pytest.approx(150.0 / 120.0 - 1.0)


def test_top_quartile_mean_averages_only_the_top_fraction():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])  # top 25% (fraction=0.25) of 8 values = top 2
    result = gates25.top_quartile_mean(values, fraction=0.25)
    assert result == pytest.approx((30.0 + 40.0) / 2.0)


def test_gate_p5_detects_overlapping_positions_as_a_violation():
    ok_trades = [
        {"entry_time": "2020-01-01 00:00:00+00:00", "exit_time": "2020-01-02 00:00:00+00:00", "entry_equity_usdt": 25.0},
        {"entry_time": "2020-01-02 00:00:00+00:00", "exit_time": "2020-01-03 00:00:00+00:00", "entry_equity_usdt": 25.0},
    ]
    assert gates25.no_overlapping_positions(ok_trades) is True
    overlapping_trades = [
        {"entry_time": "2020-01-01 00:00:00+00:00", "exit_time": "2020-01-05 00:00:00+00:00", "entry_equity_usdt": 25.0},
        {"entry_time": "2020-01-02 00:00:00+00:00", "exit_time": "2020-01-03 00:00:00+00:00", "entry_equity_usdt": 25.0},  # starts before the first one exits
    ]
    assert gates25.no_overlapping_positions(overlapping_trades) is False


def test_gate_p5_fails_when_leg_below_min_or_stress_flips_sign():
    healthy_trades = [{"entry_time": "2020-01-01 00:00:00+00:00", "exit_time": "2020-01-02 00:00:00+00:00", "entry_equity_usdt": 25.0}]
    outcome_ok, _ = gates25.gate_p5_executable(healthy_trades, base_final_usdt=30.0, stressed_final_usdt=28.0, starting_equity=25.0)
    assert outcome_ok.status == "PASS"

    tiny_leg_trades = [{"entry_time": "2020-01-01 00:00:00+00:00", "exit_time": "2020-01-02 00:00:00+00:00", "entry_equity_usdt": 2.0}]
    outcome_tiny, _ = gates25.gate_p5_executable(tiny_leg_trades, base_final_usdt=30.0, stressed_final_usdt=28.0, starting_equity=25.0)
    assert outcome_tiny.status == "FAIL"

    outcome_sign_flip, _ = gates25.gate_p5_executable(healthy_trades, base_final_usdt=30.0, stressed_final_usdt=20.0, starting_equity=25.0)
    assert outcome_sign_flip.status == "FAIL"  # base was a net gain (30>25), stressed became a net loss (20<25)


def test_promotion_requires_p1_and_p2_plus_either_p3_or_p4():
    """End-to-end orchestration check on evaluate_candidate's promotion boolean (SPEC.md:
    "승격 = P1·P2 필수 + (P3 or P4)")."""
    rng = np.random.default_rng(11)
    index = pd.date_range("2020-01-01", periods=200, freq="1D", tz=UTC)
    combined_equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.004, len(index))), index=index)
    gamble_equity = pd.Series(25.0 * np.cumprod(1.0 + rng.normal(0.002, 0.01, len(index))), index=index)
    baseline_equity = pd.Series(25.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, len(index))), index=index)
    trades_payload = [{"pnl_usdt": (5.0 if i % 3 == 0 else -1.0), "entry_equity_usdt": 25.0, "entry_time": str(index[i]), "exit_time": str(index[i])} for i in range(15)]

    report = gates25.evaluate_candidate(
        "TESTC", gamble_equity, trades_payload, combined_equity, baseline_equity,
        baseline_final_usdt=float(baseline_equity.iloc[-1]), stressed_final_usdt=float(gamble_equity.iloc[-1]), seed_offset=0,
    )
    statuses = {g["gate_id"]: g["status"] for g in report["gates"]}
    expected_promoted = statuses["P1"] == "PASS" and statuses["P2"] == "PASS" and (statuses["P3"] == "PASS" or statuses["P4"] == "PASS")
    assert report["overall"]["promoted"] == expected_promoted


# ===========================================================================
# 6. Real-cache smoke tests (fast paths only)
# ===========================================================================


def test_load_hourly_and_daily_caches_are_available_for_all_symbols():
    from research.wave25_gamble.configs25 import SYMBOLS

    for symbol in SYMBOLS:
        hourly = engine25.load_hourly(symbol)
        daily = engine25.load_daily(symbol)
        assert len(hourly) > 1000
        assert len(daily) > 100
        assert (hourly["close"] > 0.0).all()


def test_run_b0_smoke_reproduces_wave20_v1_exactly():
    """B0 must be a verbatim reproduction of wave20's own V1 (SPEC.md's literal baseline
    anchor) -- not a lookalike reimplementation that could silently drift."""
    from research.wave13_liquidity import costs_measured
    from research.wave20_convex import engine20 as wave20_engine
    from research.wave20_convex.configs20 import V1_CONFIG

    mapping = costs_measured.fit_mapping()
    v1_direct = wave20_engine.run_v1(V1_CONFIG, mapping=mapping)
    b0 = engine25.run_b0(mapping=mapping)
    assert b0.metadata["n_trades"] == v1_direct.metadata["n_trades"]
    assert float(b0.equity.dropna().iloc[-1]) == pytest.approx(float(v1_direct.equity.dropna().iloc[-1]))
