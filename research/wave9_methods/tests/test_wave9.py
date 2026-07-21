import numpy as np
import pandas as pd

from research.wave9_methods.run_wave9 import _donchian, _oos, _pair, _top_bottom


def test_top_bottom_is_neutral_and_capped():
    index = pd.date_range("2025-01-01", periods=2, tz="UTC")
    score = pd.DataFrame([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], index=index, columns=list("ABCDEF"))
    positions = _top_bottom(score, 2)
    assert np.allclose(positions.abs().sum(axis=1), 0.6)
    assert np.allclose(positions.sum(axis=1), 0.0)


def test_pair_signal_has_two_legs_and_zero_net():
    index = pd.date_range("2025-01-01", periods=40, tz="UTC")
    close = pd.DataFrame({"BTC": np.linspace(100, 110, 40), "ETH": np.linspace(100, 90, 40), "SOL": np.ones(40), "BNB": np.ones(40), "ADA": np.ones(40), "XRP": np.ones(40), "DOGE": np.ones(40), "AVAX": np.ones(40), "DOT": np.ones(40), "LINK": np.ones(40), "LTC": np.ones(40), "BCH": np.ones(40)}, index=index)
    positions = _pair(close, "P9c")
    assert np.allclose(positions.sum(axis=1), 0.0)
    assert np.all((positions.drop(columns=["BTC", "ETH"]) == 0.0).to_numpy())


def test_donchian_uses_prior_information_only():
    index = pd.date_range("2025-01-01", periods=30, tz="UTC")
    close = pd.DataFrame({"BTC": np.arange(1, 31, dtype=float), "ETH": np.arange(2, 32, dtype=float)}, index=index)
    high = close + 1.0
    low = close - 1.0
    first = _donchian(close, high, low, 5)
    close.iloc[-1] = 10_000.0
    second = _donchian(close, high, low, 5)
    assert np.allclose(first.iloc[:-1].to_numpy(), second.iloc[:-1].to_numpy(), equal_nan=True)


def test_oos_blocks_are_exactly_four_equal_partitions():
    index = pd.date_range("2025-10-01", periods=9, freq="D", tz="UTC")
    returns = pd.Series(np.full(len(index), 0.01), index=index)
    result = _oos(returns, returns, pd.Series(True, index=index))
    assert len(result["block_returns"]) == 4
