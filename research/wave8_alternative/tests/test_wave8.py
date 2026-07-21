from __future__ import annotations

import numpy as np
import pandas as pd

from research.wave8_alternative.run_wave8 import GROSS_CAP, _funding_cash, _invvol_trend, _oos, _top_bottom, _volume_signal, _zscore


def test_cross_sectional_positions_are_market_neutral_and_capped():
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    score = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 2.0, 2.0]], index=index, columns=list("ABCD"))
    positions = _top_bottom(score, 1)
    assert np.allclose(positions.abs().sum(axis=1).to_numpy(), GROSS_CAP)
    assert np.allclose(positions.sum(axis=1).to_numpy(), 0.0)


def test_zscore_is_cross_sectional_per_day():
    index = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    frame = pd.DataFrame([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]], index=index, columns=list("ABC"))
    z = _zscore(frame)
    assert np.isclose(z.iloc[0].mean(), 0.0)
    assert z.iloc[1].isna().all()


def test_no_future_timestamp_is_used_by_position_feature():
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    close = pd.DataFrame({"A": [100.0, 110.0, 90.0, 90.0], "B": [100.0, 100.0, 100.0, 100.0]}, index=index)
    prior = close.pct_change(1).shift(1)
    positions = _top_bottom(prior, 1)
    assert positions.iloc[1].abs().sum() == 0.0
    assert positions.iloc[2, 0] < 0.0


def test_volume_signal_uses_completed_day_only():
    index = pd.date_range("2025-01-01", periods=35, freq="D", tz="UTC")
    columns = list("ABCDEF")
    close = pd.DataFrame({symbol: np.linspace(100.0, 140.0, len(index)) for symbol in columns}, index=index)
    volume = pd.DataFrame(100.0, index=index, columns=columns)
    changed = volume.copy()
    changed.loc[index[25], "A"] = 10_000.0
    first = _volume_signal(close, volume, 1, reversal=False)
    second = _volume_signal(close, changed, 1, reversal=False)
    assert np.allclose(first.loc[index[:26]].to_numpy(), second.loc[index[:26]].to_numpy(), equal_nan=True)


def test_funding_cash_uses_position_held_before_funding_event():
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    positions = pd.DataFrame({"A": [1.0, 1.0, 0.0]}, index=index)
    funding = pd.DataFrame({"A": [0.1, 0.2, 0.3]}, index=index)
    cash = _funding_cash(positions, funding)
    assert np.allclose(cash["A"].to_numpy(), [-0.2, -0.3, 0.0])


def test_major_only_inverse_volatility_leaves_non_major_assets_flat():
    index = pd.date_range("2025-01-01", periods=50, freq="D", tz="UTC")
    close = pd.DataFrame({symbol: np.linspace(100.0, 150.0 + offset, len(index)) for offset, symbol in enumerate(("BTC", "ETH", "SOL", "BNB"))}, index=index)
    positions = _invvol_trend(close, 14, major_only=True)
    assert np.allclose(positions["BNB"].to_numpy(), 0.0)


def test_oos_blocks_are_exactly_four_equal_partitions():
    index = pd.date_range("2025-10-01", periods=9, freq="D", tz="UTC")
    returns = pd.Series(np.full(len(index), 0.01), index=index)
    result = _oos(returns, returns)
    assert len(result["block_returns"]) == 4
