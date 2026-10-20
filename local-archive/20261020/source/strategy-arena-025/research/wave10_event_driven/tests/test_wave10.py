import pandas as pd
import pytest

from research.wave10_event_driven.run_wave10 import (
    CANDIDATES,
    Wave10Error,
    _align_markets,
    _positions,
)


def test_registered_event_roster_is_fixed_and_distinct():
    assert tuple(item.candidate_id for item in CANDIDATES) == ("E10a", "E10b", "E10c", "E10d", "E10e", "E10f", "E10g", "E10h")
    assert all(item.entry_days > item.exit_days for item in CANDIDATES if item.candidate_id in {"E10a", "E10c", "E10d", "E10e", "E10f", "E10g", "E10h"})


def test_breakout_uses_prior_window_and_enters_after_observation():
    index = pd.date_range("2024-01-01", periods=35, freq="D", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": 100.0, "ETHUSDT": 100.0, "SOLUSDT": 100.0}, index=index)
    close.loc[index[25], "BTCUSDT"] = 110.0
    close.loc[index[26], "BTCUSDT"] = 120.0
    high = close.copy()
    low = close.copy()
    funding = pd.DataFrame(0.0, index=index, columns=close.columns)
    positions, rebalances, _ = _positions(CANDIDATES[0], high, low, close, funding)
    assert positions.loc[index[24], "BTCUSDT"] == 0.0
    assert positions.loc[index[25], "BTCUSDT"] > 0.0
    assert bool(rebalances.loc[index[25]])


def test_alignment_rejects_invalid_ohlc_range():
    index = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    from research.wave10_event_driven.run_wave10 import Market

    markets = {"BTCUSDT": Market(pd.Series([100.0, 90.0], index=index), pd.Series([110.0, 95.0], index=index), pd.Series([105.0, 100.0], index=index), pd.Series([0.0, 0.0], index=index))}
    markets.update({symbol: markets["BTCUSDT"] for symbol in ("ETHUSDT", "SOLUSDT")})
    markets["ETHUSDT"] = Market(markets["ETHUSDT"].high, pd.Series([120.0, 120.0], index=index), markets["ETHUSDT"].close, markets["ETHUSDT"].funding)
    with pytest.raises(Wave10Error, match="invalid OHLC range"):
        _align_markets(markets)
