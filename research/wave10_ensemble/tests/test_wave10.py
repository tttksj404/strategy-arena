import numpy as np
import pandas as pd

from research.wave10_ensemble.run_wave10 import _blend, _components


def test_blended_positions_respect_gross_cap():
    index = pd.date_range("2025-01-01", periods=3, tz="UTC")
    columns = ["BTC", "ETH", "BNB", "ADA", "XRP", "DOGE", "SOL", "AVAX", "DOT", "LINK", "LTC", "BCH"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    frames = {"close": close, "open": close, "high": close + 1, "low": close - 1, "volume": close}
    funding = pd.DataFrame(0.0, index=index, columns=columns)
    components = {name: (pd.DataFrame(0.0, index=index, columns=columns), funding_flag) for name, funding_flag in (("D9b", False), ("M10a", False), ("P9b", False), ("F8d", True))}
    components["D9b"] = (pd.DataFrame({symbol: (0.1 if symbol in {"BTC", "ETH", "BNB"} else (-0.1 if symbol in {"ADA", "XRP", "DOGE"} else 0.0)) for symbol in columns}, index=index), False)
    positions, funding_leg = _blend(type("Candidate", (), {"components": (("D9b", 0.5), ("F8d", 0.5)), "throttle": False})(), components, close, funding)
    assert float(positions.abs().sum(axis=1).max()) <= 0.600000001
    assert np.allclose(funding_leg["BTC"].to_numpy(), 0.0)


def test_throttle_uses_actual_equity_drawdown_before_next_position():
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    columns = ["BTC", "ETH", "BNB", "ADA", "XRP", "DOGE", "SOL", "AVAX", "DOT", "LINK", "LTC", "BCH"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    close.loc[index[1], "BTC"] = 50.0
    close.loc[index[2]:, "BTC"] = 116.5
    funding = pd.DataFrame(0.0, index=index, columns=columns)
    funding.loc[index[2], "BTC"] = 0.5
    d9b = pd.DataFrame(0.0, index=index, columns=columns)
    d9b["BTC"] = 0.3
    d9b["ETH"] = -0.3
    components = {name: (pd.DataFrame(0.0, index=index, columns=columns), funding_flag) for name, funding_flag in (("D9b", False), ("M10a", False), ("P9b", False), ("F8d", True))}
    components["D9b"] = (d9b, False)
    candidate = type("Candidate", (), {"components": (("D9b", 1.0),), "throttle": True})()
    positions, _funding_leg = _blend(candidate, components, close, funding)
    assert np.isclose(float(positions.iloc[0].abs().sum()), 0.6)
    assert np.isclose(float(positions.iloc[1].abs().sum()), 0.3)
    assert np.isclose(float(positions.iloc[2].abs().sum()), 0.6)
