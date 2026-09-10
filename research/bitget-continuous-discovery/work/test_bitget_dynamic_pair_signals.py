from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_dynamic_pair_signals import DynamicPairConfig, dynamic_pair_positions
from selection_bias_stats import SelectionBiasError


def _pair_panel(rows: int = 80) -> MarketData:
    step = np.arange(rows, dtype=float)
    log_btc = 4.5 + 0.004 * step
    log_beta = 2.0 + 0.006 * step
    residual = 0.15 * np.sin(step * 0.55)
    log_alpha = 0.5 + 1.4 * log_beta + residual
    close = np.exp(np.column_stack((log_btc, log_alpha, log_beta)))
    return MarketData(
        dates=tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows)),
        symbols=("BTC", "ALPHA", "BETA"),
        open_price=close.copy(),
        high_price=close * 1.01,
        low_price=close * 0.99,
        close_price=close,
        quote_volume=np.full_like(close, 10_000_000.0),
    )


def _config() -> DynamicPairConfig:
    return DynamicPairConfig(
        pair=("ALPHA", "BETA"),
        beta_window=24,
        z_window=12,
        entry_z=0.5,
        max_half_life=12.0,
        minimum_quote_volume=1_000_000.0,
    )


def test_dynamic_pair_is_lagged_and_beta_hedged() -> None:
    data = _pair_panel()
    positions = dynamic_pair_positions(data, _config())
    active = np.flatnonzero(np.abs(positions).sum(axis=1) > 0.0)
    assert active.size > 0
    index = int(active[-1])
    changed_close = data.close_price.copy()
    changed_close[index] *= np.asarray((10.0, 0.1, 10.0))
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close, changed_close, changed_close, data.quote_volume)
    mutated = dynamic_pair_positions(changed, _config())
    assert np.array_equal(positions[index], mutated[index])
    assert positions[index, 1] * positions[index, 2] < 0.0
    assert np.abs(positions[index]).sum() == pytest.approx(1.0)


def test_dynamic_pair_rejects_unknown_symbol() -> None:
    with pytest.raises(SelectionBiasError, match="exist"):
        dynamic_pair_positions(_pair_panel(), DynamicPairConfig(("ALPHA", "MISSING"), 24, 12, 1.0, 12.0, 1.0))
