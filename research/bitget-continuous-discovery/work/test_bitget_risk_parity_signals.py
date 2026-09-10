from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_risk_parity_signals import RiskParityConfig, risk_parity_momentum_positions


def _panel(rows: int = 100) -> MarketData:
    step = np.arange(rows, dtype=float)
    close = np.exp(
        np.column_stack(
            (
                4.0 + 0.006 * step + 0.01 * np.sin(step * 0.2),
                2.0 + 0.012 * step + 0.02 * np.sin(step * 0.4),
                2.0 + 0.004 * step + 0.04 * np.cos(step * 0.3),
                2.0 - 0.007 * step + 0.02 * np.cos(step * 0.2),
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


def _config() -> RiskParityConfig:
    return RiskParityConfig(6, 6, 2, 3, 1_000_000.0)


def test_risk_parity_momentum_is_lagged_long_only_and_excludes_btc() -> None:
    data = _panel()
    positions = risk_parity_momentum_positions(data, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))
    active = np.flatnonzero(np.abs(positions).sum(axis=1) > 0.0)
    assert active.size > 0
    index = int(active[-1])
    changed_close = data.close_price.copy()
    changed_close[index] *= np.asarray((0.1, 10.0, 0.1, 10.0))
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close, changed_close, changed_close, data.quote_volume)
    assert np.array_equal(positions[index], risk_parity_momentum_positions(changed, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))[index])
    assert positions[index, 0] == pytest.approx(0.0)
    assert positions[index].sum() == pytest.approx(1.0)
