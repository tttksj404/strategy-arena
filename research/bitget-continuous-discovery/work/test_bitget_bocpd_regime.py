from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_bocpd_regime import (
    BocpdRegimeConfig,
    _initial_state,
    _update_bocpd,
    walkforward_bocpd_regime_position_variants,
    walkforward_bocpd_regime_positions,
)
from selection_bias_stats import SelectionBiasError


def _market(rows: int = 360) -> MarketData:
    rng = np.random.default_rng(31)
    symbols = ("BTC", "ETH", "SOL", "XRP", "DOGE", "LINK")
    btc_returns = np.concatenate((rng.normal(0.001, 0.004, rows // 2), rng.normal(-0.0008, 0.015, rows - rows // 2)))
    alt_noise = rng.normal(0.0, 0.008, size=(rows, len(symbols) - 1))
    returns = np.column_stack((btc_returns, btc_returns[:, None] * 0.3 + alt_noise))
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows))
    return MarketData(dates, symbols, close, close * 1.01, close * 0.99, close, np.full_like(close, 2_000_000.0))


def _config() -> BocpdRegimeConfig:
    return BocpdRegimeConfig(
        hazard_bars=48,
        maximum_run_length=120,
        change_probability_threshold=0.2,
        volatility_lookback=30,
        trend_lookback=24,
        reversal_lookback=6,
        high_volatility_cash=False,
        top_k=1,
        rebalance_bars=3,
        minimum_quote_volume=1_000_000.0,
    )


def test_bocpd_regime_positions_are_strictly_lagged_and_dollar_neutral() -> None:
    data = _market()
    active = 250
    positions = walkforward_bocpd_regime_positions(data, _config(), alt_symbols=data.symbols[1:])
    changed_close = data.close_price.copy()
    changed_close[active] *= 2.0
    changed = replace(data, close_price=changed_close, open_price=changed_close, high_price=changed_close * 1.01, low_price=changed_close * 0.99)
    mutated = walkforward_bocpd_regime_positions(changed, _config(), alt_symbols=data.symbols[1:])

    assert np.array_equal(positions[: active + 1], mutated[: active + 1])
    assert np.allclose(positions[:, 0], 0.0)
    assert np.allclose(np.abs(positions[positions.any(axis=1)]).sum(axis=1), 1.0)


def test_bocpd_variants_reuse_one_causal_changepoint_path() -> None:
    data = _market()
    variants = walkforward_bocpd_regime_position_variants(data, _config(), alt_symbols=data.symbols[1:])

    assert np.array_equal(variants[False], walkforward_bocpd_regime_positions(data, _config(), alt_symbols=data.symbols[1:]))
    assert np.array_equal(
        variants[True],
        walkforward_bocpd_regime_positions(data, replace(_config(), high_volatility_cash=True), alt_symbols=data.symbols[1:]),
    )


def test_bocpd_assigns_more_probability_to_a_large_unexpected_change() -> None:
    state = _initial_state(120)
    for _ in range(80):
        state = _update_bocpd(state, 0.0, hazard_bars=48)
    baseline_probability = state.probabilities[0]
    shocked = _update_bocpd(state, 8.0, hazard_bars=48)

    assert np.isclose(shocked.probabilities.sum(), 1.0)
    assert shocked.probabilities[0] > baseline_probability


def test_bocpd_rejects_an_invalid_probability_threshold() -> None:
    with pytest.raises(SelectionBiasError, match="probability threshold"):
        walkforward_bocpd_regime_positions(
            _market(),
            replace(_config(), change_probability_threshold=1.0),
            alt_symbols=("ETH", "SOL", "XRP", "DOGE", "LINK"),
        )
