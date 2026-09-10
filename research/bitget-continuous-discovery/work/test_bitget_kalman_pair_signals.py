from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_kalman_pair_signals import (
    KalmanPairConfig,
    _rolling_statistics,
    kalman_pair_positions,
)
from selection_bias_stats import SelectionBiasError


def _pair_panel(rows: int = 120) -> MarketData:
    step = np.arange(rows, dtype=float)
    log_btc = 4.5 + 0.004 * step
    log_beta = 2.0 + 0.006 * step + 0.02 * np.sin(step * 0.31)
    dynamic_beta = 1.3 + 0.1 * np.sin(step * 0.07)
    residual = 0.12 * np.sin(step * 0.55)
    log_alpha = 0.5 + dynamic_beta * log_beta + residual
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


def _config() -> KalmanPairConfig:
    return KalmanPairConfig(("ALPHA", "BETA"), 16, 0.5, 1e-5, 1e-3, 12.0, 1_000_000.0)


def test_kalman_pair_is_lagged_and_beta_hedged() -> None:
    data = _pair_panel()
    positions = kalman_pair_positions(data, _config())
    active = np.flatnonzero(np.abs(positions).sum(axis=1) > 0.0)
    assert active.size > 0
    index = int(active[-1])
    changed_close = data.close_price.copy()
    changed_close[index] *= np.asarray((10.0, 0.1, 10.0))
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close, changed_close, changed_close, data.quote_volume)
    assert np.array_equal(positions[index], kalman_pair_positions(changed, _config())[index])
    assert positions[index, 1] * positions[index, 2] < 0.0
    assert np.abs(positions[index]).sum() == pytest.approx(1.0)


def test_kalman_pair_rejects_nonpositive_process_noise() -> None:
    with pytest.raises(SelectionBiasError, match="noise"):
        kalman_pair_positions(_pair_panel(), KalmanPairConfig(("ALPHA", "BETA"), 16, 1.0, 0.0, 1e-3, 12.0, 1.0))


def test_rolling_statistics_matches_windowed_zscore_and_half_life() -> None:
    innovations = np.asarray((0.1, -0.1, 0.0, 0.1, 0.2, 0.2))
    zscores, half_lives = _rolling_statistics(innovations, z_window=4)
    sample = innovations[2:6]
    previous = sample[:-1] - sample[:-1].mean()
    current = sample[1:] - sample[1:].mean()
    coefficient = float(np.dot(previous, current) / np.dot(previous, previous))
    assert zscores[5] == pytest.approx((sample[-1] - sample.mean()) / sample.std(ddof=1))
    assert half_lives[5] == pytest.approx(-np.log(2.0) / np.log(coefficient))
