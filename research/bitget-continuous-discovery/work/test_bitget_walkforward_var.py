from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_walkforward_var import VarConfig, walkforward_var_positions
from selection_bias_stats import SelectionBiasError


def _panel(rows: int = 160) -> MarketData:
    step = np.arange(rows, dtype=float)
    log_btc = 4.5 + 0.003 * step + 0.02 * np.sin(step * 0.3)
    btc_return = np.diff(log_btc, prepend=log_btc[0])
    log_alpha = 2.0 + np.cumsum(0.008 + 0.7 * np.roll(btc_return, 1) + 0.01 * np.sin(step * 0.5))
    log_beta = 2.0 + np.cumsum(-0.004 + 0.4 * np.roll(btc_return, 1) + 0.01 * np.cos(step * 0.4))
    log_gamma = 2.0 + np.cumsum(0.002 - 0.5 * np.roll(btc_return, 1) + 0.01 * np.sin(step * 0.2))
    close = np.exp(np.column_stack((log_btc, log_alpha, log_beta, log_gamma)))
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


def _config() -> VarConfig:
    return VarConfig(2, 8, 36, 0.1, 6, 1, 3, 1_000_000.0)


def test_walkforward_var_is_lagged_dollar_neutral_and_excludes_btc() -> None:
    data = _panel()
    positions = walkforward_var_positions(data, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))
    active = np.flatnonzero(np.abs(positions).sum(axis=1) > 0.0)
    assert active.size > 0
    index = int(active[-1])
    changed_close = data.close_price.copy()
    changed_close[index] *= np.asarray((0.1, 10.0, 0.1, 10.0))
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close, changed_close, changed_close, data.quote_volume)
    assert np.array_equal(positions[index], walkforward_var_positions(changed, _config(), alt_symbols=("ALPHA", "BETA", "GAMMA"))[index])
    assert positions[index, 0] == pytest.approx(0.0)
    assert positions[index].sum() == pytest.approx(0.0)


def test_walkforward_var_rejects_missing_return_lag() -> None:
    with pytest.raises(SelectionBiasError, match="lag"):
        walkforward_var_positions(_panel(), VarConfig(0, 8, 36, 0.1, 6, 1, 3, 1.0), alt_symbols=("ALPHA", "BETA", "GAMMA"))
