from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_directional_change import (
    DirectionalChangeConfig,
    directional_change_scores,
    walkforward_directional_change_positions,
)
from selection_bias_stats import SelectionBiasError


def _market(rows: int = 300) -> MarketData:
    rng = np.random.default_rng(83)
    symbols = ("BTC", "ETH", "SOL", "XRP", "DOGE", "LINK")
    common = rng.normal(0.0, 0.003, rows)
    noise = rng.normal(0.0, 0.008, size=(rows, len(symbols) - 1))
    returns = np.column_stack((common, common[:, None] * 0.25 + noise))
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows))
    return MarketData(dates, symbols, close, close * 1.01, close * 0.99, close, np.full_like(close, 2_000_000.0))


def _config() -> DirectionalChangeConfig:
    return DirectionalChangeConfig(
        threshold=0.01,
        follow_trend=True,
        rebalance_bars=3,
        minimum_quote_volume=1_000_000.0,
    )


def test_directional_change_confirms_only_after_a_threshold_reversal() -> None:
    close = np.asarray([[100.0], [100.0], [98.0], [97.0], [98.5], [99.0]])

    scores = directional_change_scores(close, threshold=0.01)

    assert scores[1, 0] == 0.0
    assert scores[3, 0] < 0.0
    assert scores[5, 0] > 0.0


def test_directional_change_positions_are_lagged_and_market_neutral() -> None:
    data = _market()
    active = 200
    positions = walkforward_directional_change_positions(data, _config(), alt_symbols=data.symbols[1:])
    close = data.close_price.copy()
    close[active] *= 1.4
    mutated = walkforward_directional_change_positions(
        replace(data, close_price=close, open_price=close, high_price=close * 1.01, low_price=close * 0.99),
        _config(),
        alt_symbols=data.symbols[1:],
    )

    assert np.array_equal(positions[: active + 1], mutated[: active + 1])
    assert np.allclose(positions[:, 0], 0.0)
    active_rows = positions[positions.any(axis=1)]
    assert np.allclose(active_rows.sum(axis=1), 0.0)
    assert np.allclose(np.abs(active_rows).sum(axis=1), 1.0)


def test_directional_change_rejects_a_nonpositive_threshold() -> None:
    with pytest.raises(SelectionBiasError, match="threshold"):
        walkforward_directional_change_positions(
            _market(),
            replace(_config(), threshold=0.0),
            alt_symbols=("ETH", "SOL", "XRP", "DOGE", "LINK"),
        )
