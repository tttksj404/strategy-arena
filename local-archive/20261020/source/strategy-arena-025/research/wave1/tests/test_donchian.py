# Verifies Donchian signals use only completed prior bars.

from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.fam_tsmom import donchian_signal


def test_donchian_breakout_uses_prior_window() -> None:
    # Given
    close = pd.Series([10.0, 11.0, 12.0, 10.0, 9.0])

    # When
    signal = donchian_signal(close, window=2, long_only=False)

    # Then
    assert signal.tolist() == [0.0, 0.0, 1.0, -1.0, -1.0]
