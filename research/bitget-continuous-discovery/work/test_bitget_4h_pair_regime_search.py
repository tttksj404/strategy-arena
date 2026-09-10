from __future__ import annotations

import numpy as np

import run_bitget_4h_pair_regime_search as search
from alt_search_models import MarketData


def _market_data() -> MarketData:
    points = 80
    btc = np.exp(np.cumsum(np.resize(np.asarray([0.001, -0.002, 0.003, -0.001], dtype=float), points)))
    eth = np.exp(np.cumsum(np.resize(np.asarray([0.01, -0.02, 0.015, -0.005], dtype=float), points)))
    sol = np.exp(np.cumsum(np.resize(np.asarray([-0.008, 0.016, -0.01, 0.004], dtype=float), points)))
    close = np.column_stack((btc, eth, sol))
    return MarketData((), ("BTC", "ETH", "SOL"), close, close * 1.01, close * 0.99, close, np.full_like(close, 2_000_000.0))


def test_btc_regime_mask_does_not_look_into_future_prices() -> None:
    prices = np.exp(np.cumsum(np.asarray([0.0] + [0.001, -0.002, 0.003, -0.001] * 20, dtype=float)))
    changed = prices.copy()
    changed[45:] *= 10.0

    baseline = search.btc_regime_mask(prices, trend_bars=8, volatility_bars=8, trend_limit=2.0, volatility_cap=0.02)
    future_changed = search.btc_regime_mask(changed, trend_bars=8, volatility_bars=8, trend_limit=2.0, volatility_cap=0.02)

    assert np.array_equal(baseline[:45], future_changed[:45])


def test_vectorized_btc_regime_mask_matches_the_lagged_reference_definition() -> None:
    prices = np.exp(np.cumsum(np.resize(np.asarray([0.001, -0.002, 0.003, -0.001], dtype=float), 80)))
    trend_bars = volatility_bars = 8
    actual = search.btc_regime_mask(prices, trend_bars=trend_bars, volatility_bars=volatility_bars, trend_limit=2.0, volatility_cap=0.02)
    expected = np.zeros(prices.size, dtype=bool)
    returns = np.zeros_like(prices)
    returns[1:] = np.log(prices[1:] / prices[:-1])
    for index in range(max(trend_bars, volatility_bars) + 1, prices.size):
        volatility = float(returns[index - volatility_bars : index].std(ddof=1))
        if volatility > 0.0 and volatility <= 0.02:
            trend = float(np.log(prices[index - 1] / prices[index - 1 - trend_bars]))
            expected[index] = abs(trend / (volatility * np.sqrt(trend_bars))) <= 2.0

    assert np.array_equal(actual, expected)


def test_pair_regime_search_grid_is_bounded_and_uses_only_a_five_times_cap() -> None:
    configs = search._all_configs(search.spec_for_interval("4H"))

    assert len(configs) == 7_488
    assert all(config.overlay.max_leverage == 5.0 for config in configs)


def test_vectorized_pair_signal_matches_the_lagged_reference_definition() -> None:
    data = _market_data()
    lookback = 12
    entry_z = 1.0

    actual = search._pair_signal(
        data,
        pair=("ETH", "SOL"),
        spread_lookback=lookback,
        entry_z=entry_z,
        minimum_quote_volume=1_000_000.0,
    )
    expected = np.zeros_like(data.close_price)
    base, quote = 1, 2
    spread = np.log(data.close_price[:, base]) - np.log(data.close_price[:, quote])
    for index in range(lookback + 1, len(expected)):
        window = spread[index - lookback : index]
        standard_deviation = float(window.std(ddof=1))
        zscore = float((spread[index - 1] - window.mean()) / standard_deviation)
        if zscore >= entry_z:
            expected[index, base], expected[index, quote] = -0.5, 0.5
        elif zscore <= -entry_z:
            expected[index, base], expected[index, quote] = 0.5, -0.5

    assert np.array_equal(actual, expected)


def test_interval_spec_preserves_4h_economic_horizons_when_scaled_to_one_hour() -> None:
    four_hour = search.spec_for_interval("4H")
    one_hour = search.spec_for_interval("1H")

    assert four_hour.spread_lookbacks == (30, 60, 120)
    assert one_hour.spread_lookbacks == (120, 240, 480)
    assert one_hour.trend_bars == 120
    assert one_hour.minimum_quote_volume == 250_000.0
