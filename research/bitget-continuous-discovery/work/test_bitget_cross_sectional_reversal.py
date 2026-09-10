from __future__ import annotations

import numpy as np

from alt_search_models import MarketData
import run_bitget_cross_sectional_reversal as search


def _market_data() -> MarketData:
    points = 24
    symbols = ("BTC", "ETH", "BNB", "ADA", "SOL", "DOGE", "XRP", "DOT", "LTC", "LINK", "AVAX", "UNI", "ATOM", "NEAR")
    close = np.empty((points, len(symbols)), dtype=float)
    for column in range(len(symbols)):
        close[:, column] = 100.0 + column + np.arange(points) * (column - 6) * 0.1
    return MarketData((), symbols, close, close * 1.01, close * 0.99, close, np.full_like(close, 2_000_000.0))


def test_cross_reversal_signal_does_not_use_future_prices() -> None:
    data = _market_data()
    changed_close = data.close_price.copy()
    changed_close[15:] *= 10.0
    changed = MarketData(data.dates, data.symbols, changed_close, changed_close * 1.01, changed_close * 0.99, changed_close, data.quote_volume)

    baseline = search._cross_reversal_signal(data, lookback=3, top_k=2, rebalance_bars=1, minimum_quote_volume=1_000_000.0)
    future_changed = search._cross_reversal_signal(changed, lookback=3, top_k=2, rebalance_bars=1, minimum_quote_volume=1_000_000.0)

    assert np.array_equal(baseline[:15], future_changed[:15])


def test_one_hour_reversal_grid_is_bounded_and_uses_only_a_five_times_cap() -> None:
    configs = search._all_configs(search.spec_for_interval("1H"))

    assert len(configs) == 288
    assert all(config.overlay.max_leverage == 5.0 for config in configs)
