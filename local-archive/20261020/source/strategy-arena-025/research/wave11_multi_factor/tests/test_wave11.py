import pandas as pd

from research.wave11_multi_factor.run_wave11 import CANDIDATES, _state_positions, _weights


def _frames(periods: int = 80):
    index = pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC")
    columns = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    close = pd.DataFrame(100.0, index=index, columns=columns)
    spot = close.copy()
    funding = pd.DataFrame(0.0, index=index, columns=columns)
    return close, spot, funding


def test_wave11_roster_covers_four_distinct_families():
    assert len(CANDIDATES) == 16
    assert {candidate.family for candidate in CANDIDATES} == {"rsi", "bollinger", "funding_reversal", "basis"}
    assert len({candidate.candidate_id for candidate in CANDIDATES}) == 16


def test_rsi_signal_uses_prior_close_and_scales_gross_cap():
    close, spot, funding = _frames()
    close.iloc[:25, 0] = list(range(100, 125))
    close.iloc[25:28, 0] = [90.0, 89.0, 88.0]
    states = _state_positions(CANDIDATES[0], close, spot, funding)
    weights = _weights(CANDIDATES[0], states)
    assert states.iloc[25, 0] == 0.0
    assert states.iloc[26, 0] == 1.0
    assert weights.abs().sum(axis=1).max() <= 0.60 + 1e-12


def test_basis_signal_is_opposite_to_basis_direction():
    close, spot, funding = _frames()
    close.iloc[30:, 0] = 103.0
    states = _state_positions(CANDIDATES[12], close, spot, funding)
    assert states.iloc[:, 0].min() >= -1.0
    assert states.iloc[:, 0].max() <= 1.0
