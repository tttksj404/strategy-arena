"""Negative look-ahead fixtures."""

import pandas as pd

from src.engine import Costs, run_backtest
from src.strategies import StrategySpec, signal_for


def test_future_signal_cannot_trade_the_same_bar() -> None:
    """A signal discovered at t must not affect t open or t close."""
    ts = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame({"ts": ts, "open": [100, 100, 100, 100], "high": [100, 100, 200, 100], "low": [100, 100, 100, 100], "close": [100, 100, 200, 100]})
    signal = pd.Series([0.0, 0.0, 1.0, 0.0])
    result = run_backtest(frame, signal, pd.DataFrame(columns=["ts", "rate"]), 2, Costs(1))
    assert result.trades == 1
    assert result.net_return < 0.01


def test_daily_signal_waits_for_the_daily_close() -> None:
    """A daily close cannot influence that same day's earlier hourly bars."""
    ts = pd.date_range("2026-01-01", periods=24 * 4, freq="h", tz="UTC")
    close = [100.0] * 24 + [200.0] * 24 + [300.0] * 24 + [400.0] * 24
    frame = pd.DataFrame({"ts": ts, "open": close, "high": close, "low": close, "close": close})
    signal = signal_for(StrategySpec("B1_tsmom_ma", {"fast": 1, "slow": 2}), frame)
    assert signal.iloc[:24].eq(0).all()
    assert signal.iloc[24:48].eq(-1).all()
    assert signal.iloc[48:].eq(1).all()
