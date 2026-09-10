from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_triple_barrier import (
    TripleBarrierConfig,
    triple_barrier_labels,
    walkforward_triple_barrier_positions,
)
from selection_bias_stats import SelectionBiasError


def _market(rows: int = 360) -> MarketData:
    rng = np.random.default_rng(71)
    symbols = ("BTC", "ETH", "SOL", "XRP", "DOGE", "LINK")
    common = rng.normal(0.0, 0.003, rows)
    noise = rng.normal(0.0, 0.009, size=(rows, len(symbols) - 1))
    returns = np.column_stack((common, common[:, None] * 0.2 + noise))
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows))
    return MarketData(dates, symbols, close, close * 1.01, close * 0.99, close, np.full_like(close, 2_000_000.0))


def _config() -> TripleBarrierConfig:
    return TripleBarrierConfig(
        feature_lookback=24,
        barrier_horizon=12,
        profit_take=0.01,
        stop_loss=0.01,
        ridge=1.0,
        rebalance_bars=3,
        minimum_training_samples=20,
        minimum_quote_volume=1_000_000.0,
        prediction_threshold=0.01,
    )


def test_triple_barrier_labels_record_first_hit_and_conservative_same_bar_tie() -> None:
    data = _market(rows=80)
    origin = 40
    high = data.high_price.copy()
    low = data.low_price.copy()
    entry = data.close_price[origin - 1]
    high[origin, 1], low[origin, 1] = entry[1] * 1.02, entry[1] * 0.999
    high[origin, 2], low[origin, 2] = entry[2] * 1.001, entry[2] * 0.98
    high[origin, 3], low[origin, 3] = entry[3] * 1.02, entry[3] * 0.98
    labeled = replace(data, high_price=high, low_price=low)

    labels = triple_barrier_labels(labeled, _config(), alt_symbols=data.symbols[1:])

    assert labels[origin, 0] == 1.0
    assert labels[origin, 1] == -1.0
    assert labels[origin, 2] == 0.0


def test_triple_barrier_positions_are_lagged_and_market_neutral() -> None:
    data = _market()
    active = 240
    positions = walkforward_triple_barrier_positions(data, _config(), alt_symbols=data.symbols[1:])
    high = data.high_price.copy()
    low = data.low_price.copy()
    high[active] *= 1.3
    low[active] *= 0.7
    mutated = walkforward_triple_barrier_positions(replace(data, high_price=high, low_price=low), _config(), alt_symbols=data.symbols[1:])

    assert np.array_equal(positions[: active + 1], mutated[: active + 1])
    assert np.allclose(positions[:, 0], 0.0)
    active_rows = positions[positions.any(axis=1)]
    assert np.allclose(active_rows.sum(axis=1), 0.0)
    assert np.allclose(np.abs(active_rows).sum(axis=1), 1.0)


def test_triple_barrier_rejects_nonpositive_barrier() -> None:
    with pytest.raises(SelectionBiasError, match="barriers"):
        walkforward_triple_barrier_positions(
            _market(),
            replace(_config(), profit_take=0.0),
            alt_symbols=("ETH", "SOL", "XRP", "DOGE", "LINK"),
        )
