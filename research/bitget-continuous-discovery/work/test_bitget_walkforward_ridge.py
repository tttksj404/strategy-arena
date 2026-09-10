from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_walkforward_ridge import RidgeConfig, walkforward_ridge_positions
from selection_bias_stats import SelectionBiasError


def _panel(rows: int = 90) -> MarketData:
    step = np.arange(rows, dtype=float)
    close = np.exp(
        np.column_stack(
            (
                4.5 + 0.003 * step + 0.02 * np.sin(step * 0.3),
                2.0 + 0.015 * step + 0.03 * np.sin(step * 0.5),
                2.0 - 0.010 * step + 0.02 * np.cos(step * 0.4),
                2.0 + 0.004 * step + 0.04 * np.sin(step * 0.2),
            )
        )
    )
    quote_volume = 10_000_000.0 * (
        1.0
        + np.column_stack(
            (
                0.08 * np.sin(step * 0.17),
                0.06 * np.cos(step * 0.23),
                0.05 * np.sin(step * 0.31 + 0.4),
                0.07 * np.cos(step * 0.19 + 0.2),
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
        quote_volume=quote_volume,
    )


def _config() -> RidgeConfig:
    return RidgeConfig(3, 9, 6, 24, 0.1, 1, 2, 1_000_000.0)


def test_walkforward_ridge_is_lagged_neutral_and_excludes_btc() -> None:
    data = _panel()
    positions = walkforward_ridge_positions(data, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))
    active = np.flatnonzero(np.abs(positions).sum(axis=1) > 0.0)
    assert active.size > 0
    index = int(active[-1])
    changed_close = data.close_price.copy()
    changed_close[index] *= np.asarray((0.1, 10.0, 0.1, 10.0))
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close, changed_close, changed_close, data.quote_volume)
    assert np.array_equal(positions[index], walkforward_ridge_positions(changed, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))[index])
    assert positions[index, 0] == pytest.approx(0.0)
    assert positions[index].sum() == pytest.approx(0.0)


def test_walkforward_ridge_requires_btc_and_distinct_cross_section() -> None:
    data = _panel()
    without_btc = MarketData(data.dates, data.symbols[1:], data.open_price[:, 1:], data.high_price[:, 1:], data.low_price[:, 1:], data.close_price[:, 1:], data.quote_volume[:, 1:])
    with pytest.raises(SelectionBiasError, match="BTC"):
        walkforward_ridge_positions(without_btc, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))
    with pytest.raises(SelectionBiasError, match="distinct"):
        walkforward_ridge_positions(data, RidgeConfig(3, 9, 6, 24, 0.1, 2, 2, 1.0), alt_symbols=("ALPHA", "BETA", "GAMMA"))
