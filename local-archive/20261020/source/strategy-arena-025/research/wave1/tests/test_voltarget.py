# Verifies volatility targeting caps leverage and guards zero volatility.

from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_tsmom import vol_target_fraction


def test_vol_target_caps_leverage_and_guards_zero() -> None:
    # Given
    realized_vol = pd.Series([0.0, 0.005, 0.03])

    # When
    fraction = vol_target_fraction(
        realized_vol,
        target_vol=0.015,
        leverage_cap=2.0,
    )

    # Then
    assert fraction.tolist() == pytest.approx([0.0, 2.0, 0.5])
