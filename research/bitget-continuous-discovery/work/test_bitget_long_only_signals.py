from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_long_only_signals import LongOnlyConfig, LongOnlyStyle, btc_positive_regime_mask, long_only_positions
from selection_bias_stats import SelectionBiasError


def _panel(rows: int = 90) -> MarketData:
    step = np.arange(rows, dtype=float)
    close = np.exp(
        np.column_stack(
            (
                4.5 + 0.008 * step + 0.01 * np.sin(step * 0.2),
                2.0 + 0.015 * step + 0.03 * np.sin(step * 0.4),
                2.0 - 0.007 * step + 0.02 * np.cos(step * 0.3),
                2.0 + 0.004 * step + 0.05 * np.sin(step * 0.16),
            )
        )
    )
    return MarketData(
        dates=tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows)),
        symbols=("BTC", "ALPHA", "BETA", "GAMMA"),
        open_price=close,
        high_price=close * 1.01,
        low_price=close * 0.99,
        close_price=close,
        quote_volume=np.full_like(close, 10_000_000.0),
    )


def _config(style: LongOnlyStyle = LongOnlyStyle.MULTI_HORIZON) -> LongOnlyConfig:
    return LongOnlyConfig(style, 3, 9, 1, 2, 1_000_000.0)


def test_long_only_positions_are_lagged_and_exclude_btc() -> None:
    data = _panel()
    positions = long_only_positions(data, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))
    active = np.flatnonzero(np.abs(positions).sum(axis=1) > 0.0)
    assert active.size > 0
    index = int(active[-1])
    changed_close = data.close_price.copy()
    changed_close[index] *= np.asarray((0.2, 10.0, 0.2, 10.0))
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close, changed_close, changed_close, data.quote_volume)
    assert np.array_equal(positions[index], long_only_positions(changed, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))[index])
    assert positions[index, 0] == pytest.approx(0.0)
    assert positions[index].sum() == pytest.approx(1.0)


def test_btc_positive_regime_is_lagged_and_requires_positive_trend() -> None:
    step = np.arange(48, dtype=float)
    upward = np.exp(4.0 + 0.019 * step + 0.004 * np.sin(step * 0.7))
    mask = btc_positive_regime_mask(upward, trend_bars=6, volatility_bars=6, minimum_trend_z=0.0, volatility_cap=0.1)
    assert mask[-1]
    changed = upward.copy()
    changed[-1] *= 0.1
    assert mask[-1] == btc_positive_regime_mask(changed, trend_bars=6, volatility_bars=6, minimum_trend_z=0.0, volatility_cap=0.1)[-1]
    with pytest.raises(SelectionBiasError, match="parameters"):
        btc_positive_regime_mask(upward, trend_bars=0, volatility_bars=6, minimum_trend_z=0.0, volatility_cap=0.1)
