from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_ctrend_signals import CTrendConfig, CompositeStyle, composite_trend_positions
from selection_bias_stats import SelectionBiasError


def _panel(rows: int = 120) -> MarketData:
    step = np.arange(rows, dtype=float)
    close = np.exp(
        np.column_stack(
            (
                4.5 + 0.003 * step + 0.01 * np.sin(step * 0.2),
                2.0 + 0.012 * step + 0.03 * np.sin(step * 0.4),
                2.0 - 0.009 * step + 0.02 * np.cos(step * 0.3),
                2.0 + 0.004 * step + 0.05 * np.sin(step * 0.16),
                2.0 - 0.002 * step + 0.04 * np.cos(step * 0.26),
            )
        )
    )
    volume = 10_000_000.0 * (1.0 + 0.06 * np.column_stack(tuple(np.sin(step * rate + phase) for rate, phase in ((0.13, 0.0), (0.21, 0.4), (0.18, 0.8), (0.25, 0.2), (0.16, 0.7)))))
    return MarketData(
        dates=tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows)),
        symbols=("BTC", "ALPHA", "BETA", "GAMMA", "DELTA"),
        open_price=close,
        high_price=close * 1.01,
        low_price=close * 0.99,
        close_price=close,
        quote_volume=volume,
    )


def _config(style: CompositeStyle = CompositeStyle.MULTI_SIGNAL) -> CTrendConfig:
    return CTrendConfig(style, 3, 9, 18, 6, 12, 1, 2, 1_000_000.0)


def test_composite_trend_is_lagged_neutral_and_excludes_btc() -> None:
    data = _panel()
    positions = composite_trend_positions(data, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA", "DELTA"))
    active = np.flatnonzero(np.abs(positions).sum(axis=1) > 0.0)
    assert active.size > 0
    index = int(active[-1])
    changed_close = data.close_price.copy()
    changed_close[index] *= np.asarray((0.2, 10.0, 0.2, 10.0, 0.2))
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close, changed_close, changed_close, data.quote_volume)
    assert np.array_equal(positions[index], composite_trend_positions(changed, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA", "DELTA"))[index])
    assert positions[index, 0] == pytest.approx(0.0)
    assert positions[index].sum() == pytest.approx(0.0)


def test_composite_trend_rejects_invalid_cross_section() -> None:
    data = _panel()
    with pytest.raises(SelectionBiasError, match="ordered"):
        composite_trend_positions(data, CTrendConfig(CompositeStyle.MULTI_HORIZON, 9, 3, 18, 6, 12, 1, 1, 1.0), alt_symbols=("ALPHA", "BETA", "GAMMA", "DELTA"))
    with pytest.raises(SelectionBiasError, match="distinct"):
        composite_trend_positions(data, _config(), alt_symbols=("ALPHA", "ALPHA", "GAMMA", "DELTA"))
