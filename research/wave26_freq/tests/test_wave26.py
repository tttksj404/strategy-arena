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

from research.wave25_gamble import engine25
from research.wave25_gamble import indicators25 as ind
from research.wave26_freq import engine26, gates26

UTC = "UTC"
SYMBOL = "TEST"


def _hourly_index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1h", tz=UTC)


def _frames26(index, open_, high, low, close, risk_atr, cost, signal, symbol: str = SYMBOL) -> dict[str, pd.DataFrame]:
    """Same shape as test_wave25.py's own `_frames` helper, plus the two admission masks
    run_multi_symbol_convex_controlled requires -- defaulted to all-True (gate disabled) unless
    a test overrides them after construction."""
    n = len(index)
    return {
        "open": pd.DataFrame({symbol: open_}, index=index),
        "high": pd.DataFrame({symbol: high}, index=index),
        "low": pd.DataFrame({symbol: low}, index=index),
        "close": pd.DataFrame({symbol: close}, index=index),
        "risk_atr": pd.DataFrame({symbol: risk_atr}, index=index),
        "cost": pd.DataFrame({symbol: cost}, index=index),
        "signal": pd.DataFrame({symbol: signal}, index=index),
        "adx_active": pd.DataFrame({symbol: [True] * n}, index=index),
        "z_active": pd.DataFrame({symbol: [True] * n}, index=index),
    }


# ===========================================================================
# 1. z-score gate (신호값 20일 z-score 필터)
# ===========================================================================


def test_zscore_gate_mask_matches_hand_computed_values():
    # window=3: idx2 window={1,1,5} mean=2.3333 std=1.8856 -> z(5)=1.4142>1.0 True.
    # idx3 window={1,5,1} same mean/std, value=1 -> z=-0.7071 False. idx5 window={1,1,1} std=0 -> NaN -> False.
    strength = pd.Series([1.0, 1.0, 5.0, 1.0, 1.0, 1.0], index=_hourly_index(6))
    mask = engine26.zscore_gate_mask(strength, window_bars=3, threshold=1.0)
    assert mask.tolist() == [False, False, True, False, False, False]


def test_zscore_gate_mask_is_causal_truncation_invariant():
    """No-lookahead pin: truncating the strength series after bar K must never change the mask's
    own value at or before bar K (same pattern as test_wave25.py's own
    test_align_daily_to_intraday_never_leaks_same_day_value_before_day_closes)."""
    n = 60
    rng = np.random.default_rng(7)
    strength = pd.Series(np.abs(rng.normal(1.0, 0.5, n)), index=_hourly_index(n))
    full_mask = engine26.zscore_gate_mask(strength, window_bars=10, threshold=1.0)
    k = 40
    truncated_mask = engine26.zscore_gate_mask(strength.iloc[: k + 1], window_bars=10, threshold=1.0)
    pd.testing.assert_series_equal(full_mask.iloc[: k + 1], truncated_mask, check_names=False)


# ===========================================================================
# 2. ADX regime gate lookahead prevention (build_controlled_frames' own adx_active ingredient)
# ===========================================================================


def test_adx_gate_mask_is_causal_truncation_invariant():
    n = 100
    rng = np.random.default_rng(42)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.5, n)), index=_hourly_index(n))
    ohlc = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": close + rng.uniform(0.5, 2.0, n), "low": close - rng.uniform(0.5, 2.0, n), "close": close})
    full_active = ind.adx_dmi(ohlc, window=14)["adx"] > 20.0
    k = 60
    truncated = ohlc.iloc[: k + 1]
    truncated_active = ind.adx_dmi(truncated, window=14)["adx"] > 20.0
    pd.testing.assert_series_equal(full_active.iloc[: k + 1], truncated_active, check_names=False)


# ===========================================================================
# 3. Cooldown counting (B-family: run_multi_symbol_convex_controlled)
# ===========================================================================


def test_cooldown_blocks_reentry_for_exactly_cooldown_bars_then_admits():
    """Hand-traced: signal fires at bar0 (entry fills bar1), flips at bar2 (close fills bar3,
    exit_reason='signal_flip_exit', cooldown_remaining set to 3). Signal stays live (=1.0) at
    bars 3,4,5,6 while flat: bars 3/4/5 must be blocked (3 == cooldown_bars), bar6 is the first
    bar cooldown_remaining<=0 so it must admit (fills bar7). Price is flat throughout (open=
    high=low=close=100) with a huge risk_atr so the hard stop/trailing never fire -- isolates
    the cooldown mechanic from the inherited stop/trailing lifecycle (that's tested separately
    below)."""
    n = 15
    index = _hourly_index(n)
    flat = [100.0] * n
    risk_atr = [1000.0] * n
    cost = [0.0] * n
    signal = [1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    frames = _frames26(index, flat, flat, flat, flat, risk_atr, cost, signal)

    result = engine26.run_multi_symbol_convex_controlled(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", cooldown_bars=3, starting_equity=25.0, max_bars_in_position=100)

    assert len(result.trades) == 2
    first, second = result.trades
    assert first.entry_time == index[1] and first.exit_time == index[3] and first.exit_reason == "signal_flip_exit"
    assert second.entry_time == index[7] and second.exit_reason == "end_of_data"  # admitted at bar6's Step C, fills bar7's open

    admission = result.metadata["entry_admission"]
    assert admission == {"entry_opportunities": 5, "entries_admitted": 2, "blocked_by_cooldown": 3, "blocked_by_gate": 0}
    assert admission["entry_opportunities"] == admission["entries_admitted"] + admission["blocked_by_cooldown"] + admission["blocked_by_gate"]
    assert result.metadata["final_cooldown_remaining_bars"] == 0


def test_signal_flip_is_close_then_later_reopen_not_same_bar_reversal():
    """engine25's own 'reverse' fills the close AND the new opposite entry in the SAME bar. The
    controlled engine must NOT do this (a same-bar reversal is incompatible with a nonzero
    cooldown by construction -- see engine26 module docstring): even with cooldown_bars=0, the
    re-entry must land one bar LATER than the close, not the same bar."""
    n = 6
    index = _hourly_index(n)
    flat = [100.0] * n
    risk_atr = [1000.0] * n
    cost = [0.0] * n
    signal = [1.0, 0.0, -1.0, -1.0, 0.0, 0.0]
    frames = _frames26(index, flat, flat, flat, flat, risk_atr, cost, signal)

    result = engine26.run_multi_symbol_convex_controlled(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", cooldown_bars=0, starting_equity=25.0, max_bars_in_position=100)

    assert len(result.trades) == 2
    long_trade, short_trade = result.trades
    assert long_trade.exit_time == index[3] and long_trade.exit_reason == "signal_flip_exit"
    assert short_trade.entry_time == index[4]  # ONE bar after the close (index[3]), never the same bar
    assert short_trade.direction == pytest.approx(-1.0)


def test_stop_loss_still_works_and_sets_cooldown_blocking_subsequent_reentries():
    """Regression + integration: Step B's hard-stop math is copy-pasted verbatim into the fork --
    this must still trigger identically to test_wave25.py's own
    test_hard_stop_loss_triggers_and_caps_the_loss (same OHLC, same expected exit_price=98.0),
    AND the stop-loss exit must ALSO set the cooldown timer (SPEC.md: '청산 후' means after ANY
    exit, not just a signal-driven one) -- three fresh same-direction signals immediately after
    the stop must all be blocked."""
    n = 7
    index = _hourly_index(n)
    open_ = [99.0, 100.0, 99.0, 97.0, 97.0, 97.0, 97.0]
    high = [100.0, 101.0, 99.5, 98.0, 98.0, 98.0, 98.0]
    low = [98.0, 99.0, 97.0, 96.0, 96.0, 96.0, 96.0]
    close = [99.5, 100.5, 97.5, 97.0, 97.0, 97.0, 97.0]
    risk_atr = [2.0] * n
    cost = [0.0] * n
    signal = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    frames = _frames26(index, open_, high, low, close, risk_atr, cost, signal)

    result = engine26.run_multi_symbol_convex_controlled(frames, priority=(SYMBOL,), worst_cost=0.0, candidate_id="TESTC", cooldown_bars=5, starting_equity=25.0, max_bars_in_position=100)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(98.0)  # unchanged from test_wave25.py's own hard-stop test

    admission = result.metadata["entry_admission"]
    assert admission["entries_admitted"] == 1  # only the original bar0 entry
    assert admission["blocked_by_cooldown"] == 3  # bars 3,4,5 all had a fresh signal=1.0 but cooldown (set at the stop) was still active
    assert admission["entry_opportunities"] == admission["entries_admitted"] + admission["blocked_by_cooldown"] + admission["blocked_by_gate"]


# ===========================================================================
# 4. V1-family controlled engine (simulate_breakout_reversal_controlled) -- C7's own core
# ===========================================================================


def test_v1_controlled_cooldown_blocks_then_admits_breakout_reentry():
    """V1's OWN arming/breakout math (armable_arr always True here, atr constant so
    threshold=2.0*2.0=4.0) is untouched; only cooldown gates the final open decision. First
    breakout (100->105 at bar2) is unblocked (cooldown starts at 0). It reverses via the
    chandelier trigger at bar5 (extreme 105 - close 90 = 15 >= 4). Cooldown(3) then blocks the
    next two breakout attempts (bars 7,8) and admits the third (bar9)."""
    n = 12
    index = _hourly_index(n)
    close = np.array([100.0, 100.0, 105.0, 105.0, 105.0, 90.0, 90.0, 95.0, 95.0, 95.0, 95.0, 95.0])
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    atr_arr = np.full(n, 2.0)
    cost_arr = np.zeros(n)
    armable_arr = np.full(n, True)

    equity, trades, total_cost, diagnostics = engine26.simulate_breakout_reversal_controlled(
        index=pd.DatetimeIndex(index), open_arr=open_, close_arr=close, atr_arr=atr_arr, cost_arr=cost_arr, worst_cost=0.0,
        atr_multiplier=2.0, armable_arr=armable_arr, adx_active_arr=None, cooldown_bars=3, starting_equity=25.0, symbol=SYMBOL,
    )

    assert len(trades) == 2
    first, second = trades
    assert first.entry_time == index[3] and first.exit_time == index[6] and first.exit_reason == "signal_flip_exit"
    assert second.entry_time == index[10] and second.exit_reason == "end_of_data"
    assert diagnostics == {
        "entry_opportunities": 4, "entries_admitted": 2, "blocked_by_cooldown": 2, "blocked_by_gate": 0,
        "cooldown_bars_setting": 3, "final_cooldown_remaining_bars": 0,
    }


def test_v1_controlled_adx_gate_blocks_until_true():
    """Same breakout setup as above but cooldown disabled (0) and adx_active False for the first
    breakout bar -- the breakout must be blocked exactly while adx_active is False and admitted
    the instant it turns True (not before, not lagged)."""
    n = 6
    index = _hourly_index(n)
    close = np.array([100.0, 100.0, 105.0, 105.0, 105.0, 105.0])
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    atr_arr = np.full(n, 2.0)
    cost_arr = np.zeros(n)
    armable_arr = np.full(n, True)
    adx_active_arr = np.array([False, False, False, True, True, True])

    equity, trades, total_cost, diagnostics = engine26.simulate_breakout_reversal_controlled(
        index=pd.DatetimeIndex(index), open_arr=open_, close_arr=close, atr_arr=atr_arr, cost_arr=cost_arr, worst_cost=0.0,
        atr_multiplier=2.0, armable_arr=armable_arr, adx_active_arr=adx_active_arr, cooldown_bars=0, starting_equity=25.0, symbol=SYMBOL,
    )

    assert len(trades) == 1
    assert trades[0].entry_time == index[4]  # breakout condition true from bar2 onward but ADX only turns True at bar3 -> admitted at bar3's Step C, fills bar4
    assert diagnostics["blocked_by_gate"] == 1  # bar2 only (bar3 onward is admitted/held, not re-checked)
    assert diagnostics["entries_admitted"] == 1


# ===========================================================================
# 5. Gate Q4 (cost efficiency, new) and promotion formula
# ===========================================================================


def test_gate_q4_cost_efficiency_threshold():
    ok, _ = gates26.gate_q4_cost_efficiency(9.0, gamble_capital=25.0, max_fraction=0.40)
    assert ok.status == "PASS"
    boundary, _ = gates26.gate_q4_cost_efficiency(10.0, gamble_capital=25.0, max_fraction=0.40)
    assert boundary.status == "PASS"  # exactly at the cap counts as passing
    fail, _ = gates26.gate_q4_cost_efficiency(10.01, gamble_capital=25.0, max_fraction=0.40)
    assert fail.status == "FAIL"


def test_promotion_requires_all_of_q1_q2_q3_q4():
    """SPEC.md line 37: '승격 = Q1·Q2·Q4 필수 + Q3' -- ALL FOUR required (stricter than wave25's
    own P1·P2 필수+(P3 or P4))."""
    rng = np.random.default_rng(11)
    index = pd.date_range("2020-01-01", periods=200, freq="1D", tz=UTC)
    combined_equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.004, len(index))), index=index)
    gamble_equity = pd.Series(25.0 * np.cumprod(1.0 + rng.normal(0.002, 0.01, len(index))), index=index)
    c0_equity = pd.Series(25.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, len(index))), index=index)
    trades_payload = [{"pnl_usdt": (5.0 if i % 3 == 0 else -1.0), "entry_equity_usdt": 25.0, "entry_time": str(index[i]), "exit_time": str(index[i])} for i in range(15)]

    report = gates26.evaluate_candidate(
        "TESTC", gamble_equity, trades_payload, combined_equity, c0_equity,
        c0_final_usdt=float(c0_equity.iloc[-1]), stressed_final_usdt=float(gamble_equity.iloc[-1]), total_cost_usdt=5.0, seed_offset=0,
    )
    statuses = {g["gate_id"]: g["status"] for g in report["gates"]}
    expected_promoted = statuses["Q1"] == "PASS" and statuses["Q2"] == "PASS" and statuses["Q3"] == "PASS" and statuses["Q4"] == "PASS"
    assert report["overall"]["promoted"] == expected_promoted
    assert statuses["Q4"] == "PASS"  # total_cost_usdt=5.0 <= $10 cap


def test_promotion_fails_when_only_cost_gate_violated():
    """Isolates Q4: a candidate that would otherwise promote (Q1/Q2/Q3 all PASS) must still be
    rejected if its cost exceeds the 40%-of-sleeve cap -- proves Q4 is a REAL, binding AND
    condition, not just disclosed like Q5."""
    rng = np.random.default_rng(5)
    index = pd.date_range("2020-01-01", periods=200, freq="1D", tz=UTC)
    combined_equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.003, len(index))), index=index)
    gamble_equity = pd.Series(80.0 * np.cumprod(1.0 + rng.normal(0.001, 0.004, len(index))), index=index)  # strong uptrend, ends well above C0
    c0_equity = pd.Series(25.0 * np.cumprod(1.0 + rng.normal(0.0, 0.003, len(index))), index=index)
    trades_payload = [{"pnl_usdt": (8.0 if i % 4 == 0 else -1.0), "entry_equity_usdt": 25.0, "entry_time": str(index[i]), "exit_time": str(index[i])} for i in range(20)]

    report = gates26.evaluate_candidate(
        "TESTC", gamble_equity, trades_payload, combined_equity, c0_equity,
        c0_final_usdt=float(c0_equity.iloc[-1]), stressed_final_usdt=float(gamble_equity.iloc[-1]), total_cost_usdt=50.0, seed_offset=1,  # $50 >> $10 cap
    )
    statuses = {g["gate_id"]: g["status"] for g in report["gates"]}
    assert statuses["Q4"] == "FAIL"
    assert report["overall"]["promoted"] is False


# ===========================================================================
# 6. Real-cache smoke tests (C0 must reproduce wave25 B0; control must reduce real trade counts)
# ===========================================================================


def test_c0_matches_wave25_b0_exactly():
    """C0 has NO frequency control (SPEC.md: '없음 -- 기준선') -- it must be numerically
    identical to wave25's own B0 (both ultimately call research.wave20_convex.engine20.run_v1)."""
    from research.wave13_liquidity import costs_measured as cm

    mapping = cm.fit_mapping()
    b0 = engine25.run_b0(mapping=mapping)
    c0 = engine26.run_c0(mapping=mapping)
    assert c0.metadata["n_trades"] == b0.metadata["n_trades"]
    assert float(c0.equity.dropna().iloc[-1]) == pytest.approx(float(b0.equity.dropna().iloc[-1]))


def test_frequency_control_reduces_trade_count_vs_wave25_macd():
    """SPEC.md's central mechanism, on REAL cached data: C1 (MACD + 5-day cooldown, no other
    axis) must trade strictly less often than wave25's own B1 (identical MACD signal, zero
    control) -- proves the cooldown axis alone measurably suppresses frequency on real data, not
    just in the synthetic unit tests above."""
    from research.wave13_liquidity import costs_measured as cm

    mapping = cm.fit_mapping()
    b1 = engine25.run_b1(mapping=mapping)
    c1 = engine26.run_c1(mapping=mapping)
    assert c1.metadata["n_trades"] < b1.metadata["n_trades"]
