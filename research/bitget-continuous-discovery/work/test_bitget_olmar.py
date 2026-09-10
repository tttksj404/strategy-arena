from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_olmar import OlmarConfig, walkforward_olmar_positions
from selection_bias_stats import SelectionBiasError


def _market(rows: int = 300) -> MarketData:
    rng = np.random.default_rng(37)
    symbols = ("BTC", "ETH", "SOL", "XRP", "DOGE", "LINK")
    common = rng.normal(0.0, 0.003, rows)
    alt_noise = rng.normal(0.0, 0.008, size=(rows, len(symbols) - 1))
    returns = np.column_stack((common, common[:, None] * 0.3 + alt_noise))
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows))
    return MarketData(dates, symbols, close, close * 1.01, close * 0.99, close, np.full_like(close, 2_000_000.0))


def _config() -> OlmarConfig:
    return OlmarConfig(
        moving_average_lookback=24,
        target_edge=0.003,
        maximum_step=2.0,
        rebalance_bars=3,
        minimum_quote_volume=1_000_000.0,
    )


def test_olmar_positions_are_strictly_lagged_and_market_neutral() -> None:
    data = _market()
    active = 200
    positions = walkforward_olmar_positions(data, _config(), alt_symbols=data.symbols[1:])
    changed_close = data.close_price.copy()
    changed_close[active] *= 2.0
    changed = replace(data, close_price=changed_close, open_price=changed_close, high_price=changed_close * 1.01, low_price=changed_close * 0.99)
    mutated = walkforward_olmar_positions(changed, _config(), alt_symbols=data.symbols[1:])

    assert np.array_equal(positions[: active + 1], mutated[: active + 1])
    assert np.allclose(positions[:, 0], 0.0)
    active_rows = positions[positions.any(axis=1)]
    assert np.allclose(active_rows.sum(axis=1), 0.0)
    assert np.allclose(np.abs(active_rows).sum(axis=1), 1.0)


def test_olmar_rejects_a_nonpositive_target_edge() -> None:
    with pytest.raises(SelectionBiasError, match="target edge"):
        walkforward_olmar_positions(
            _market(),
            replace(_config(), target_edge=0.0),
            alt_symbols=("ETH", "SOL", "XRP", "DOGE", "LINK"),
        )
