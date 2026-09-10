from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_pca_stat_arb import PcaStatArbConfig, walkforward_pca_stat_arb_positions
from selection_bias_stats import SelectionBiasError


def _market(rows: int = 260) -> MarketData:
    rng = np.random.default_rng(17)
    symbols = ("BTC", "ETH", "SOL", "XRP", "DOGE", "LINK")
    innovations = rng.normal(0.0002, 0.008, size=(rows, len(symbols)))
    innovations[:, 1:] += innovations[:, :1] * 0.35
    close = 100.0 * np.exp(np.cumsum(innovations, axis=0))
    dates = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows))
    return MarketData(dates, symbols, close, close * 1.01, close * 0.99, close, np.full_like(close, 2_000_000.0))


def _config() -> PcaStatArbConfig:
    return PcaStatArbConfig(
        training_bars=80,
        factor_count=2,
        signal_bars=6,
        residual_z_window=30,
        entry_z=0.0,
        model_refit_bars=12,
        top_k=1,
        rebalance_bars=3,
        minimum_quote_volume=1_000_000.0,
    )


def test_pca_stat_arb_is_strictly_lagged_and_dollar_neutral() -> None:
    data = _market()
    active = 180
    positions = walkforward_pca_stat_arb_positions(data, _config(), alt_symbols=data.symbols[1:])
    changed_close = data.close_price.copy()
    changed_close[active] *= 3.0
    changed_volume = data.quote_volume.copy()
    changed_volume[active] = 0.0
    changed = replace(
        data,
        open_price=changed_close,
        high_price=changed_close * 1.01,
        low_price=changed_close * 0.99,
        close_price=changed_close,
        quote_volume=changed_volume,
    )
    mutated = walkforward_pca_stat_arb_positions(changed, _config(), alt_symbols=data.symbols[1:])

    assert np.array_equal(positions[: active + 1], mutated[: active + 1])
    assert np.allclose(positions[:, 0], 0.0)
    assert np.allclose(np.abs(positions[positions.any(axis=1)]).sum(axis=1), 1.0)


def test_pca_stat_arb_rejects_an_invalid_factor_dimension() -> None:
    with pytest.raises(SelectionBiasError, match="factor count"):
        walkforward_pca_stat_arb_positions(
            _market(),
            replace(_config(), factor_count=6),
            alt_symbols=("ETH", "SOL", "XRP", "DOGE", "LINK"),
        )
