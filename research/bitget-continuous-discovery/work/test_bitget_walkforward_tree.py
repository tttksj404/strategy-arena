from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_walkforward_tree import TreeConfig, walkforward_tree_positions
from selection_bias_stats import SelectionBiasError


def _panel(rows: int = 160) -> MarketData:
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
    volume = 10_000_000.0 * (
        1.0
        + 0.05
        * np.column_stack(
            (
                np.sin(step * 0.17),
                np.cos(step * 0.23),
                np.sin(step * 0.31 + 0.4),
                np.cos(step * 0.19 + 0.2),
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
        quote_volume=volume,
    )


def _config() -> TreeConfig:
    return TreeConfig(3, 9, 6, 36, 6, 8, 3, 2, 1, 3, 1_000_000.0)


def test_walkforward_tree_is_lagged_dollar_neutral_and_excludes_btc() -> None:
    data = _panel()
    positions = walkforward_tree_positions(data, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))
    active = np.flatnonzero(np.abs(positions).sum(axis=1) > 0.0)
    assert active.size > 0
    index = int(active[-1])
    changed_close = data.close_price.copy()
    changed_close[index] *= np.asarray((0.1, 10.0, 0.1, 10.0))
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close, changed_close, changed_close, data.quote_volume)
    assert np.array_equal(positions[index], walkforward_tree_positions(changed, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))[index])
    assert positions[index, 0] == pytest.approx(0.0)
    assert positions[index].sum() == pytest.approx(0.0)


def test_walkforward_tree_rejects_nonpositive_refit_interval() -> None:
    with pytest.raises(SelectionBiasError, match="training"):
        walkforward_tree_positions(_panel(), TreeConfig(3, 9, 6, 36, 0, 8, 3, 2, 1, 3, 1.0), alt_symbols=("ALPHA", "BETA", "GAMMA"))
