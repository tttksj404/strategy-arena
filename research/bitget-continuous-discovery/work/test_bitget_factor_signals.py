from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_factor_signals import (
    FactorConfig,
    FactorExposure,
    FactorFamily,
    build_factor_position_variants,
    build_factor_positions,
)
from selection_bias_stats import SelectionBiasError


def _market_data(rows: int = 20) -> MarketData:
    step = np.arange(rows, dtype=float)
    close = np.column_stack(
        (
            100.0 * np.exp(0.008 * step + 0.0005 * step * step),
            10.0 * np.exp(0.030 * step),
            10.0 * np.exp(-0.020 * step),
            10.0 * np.exp(0.003 * step),
        )
    )
    return MarketData(
        dates=tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows)),
        symbols=("BTC", "ALPHA", "BETA", "GAMMA"),
        open_price=close.copy(),
        high_price=close * 1.01,
        low_price=close * 0.99,
        close_price=close,
        quote_volume=np.full_like(close, 10_000_000.0),
    )


def _config(family: FactorFamily, **overrides: FactorFamily | float) -> FactorConfig:
    values: dict[str, FactorFamily | float] = {
        "family": family,
        "lookback": 3,
        "top_k": 1,
        "rebalance_bars": 1,
        "minimum_quote_volume": 1_000_000.0,
        "beta_window": 6,
        "volume_multiplier": 1.0,
    }
    values.update(overrides)
    return FactorConfig(**values)


def test_cross_sectional_trend_uses_only_prior_closed_bar() -> None:
    data = _market_data()
    config = _config(FactorFamily.TREND)
    baseline = build_factor_positions(data, config, alt_symbols=("ALPHA", "BETA", "GAMMA"))
    changed_close = data.close_price.copy()
    changed_close[10] *= np.asarray((0.1, 10.0, 0.1, 10.0))
    changed = MarketData(
        data.dates,
        data.symbols,
        changed_close,
        changed_close,
        changed_close,
        changed_close,
        data.quote_volume,
    )
    mutated = build_factor_positions(changed, config, alt_symbols=("ALPHA", "BETA", "GAMMA"))
    assert np.array_equal(baseline[10], mutated[10])
    assert baseline[10, 1] == pytest.approx(0.5)
    assert baseline[10, 2] == pytest.approx(-0.5)
    assert baseline[10].sum() == pytest.approx(0.0)


def test_long_only_factor_exposure_uses_only_prior_top_ranked_alt() -> None:
    data = _market_data()
    positions = build_factor_positions(
        data,
        _config(FactorFamily.TREND),
        alt_symbols=("ALPHA", "BETA", "GAMMA"),
        exposure=FactorExposure.LONG_ONLY,
    )

    assert positions[10, 1] == pytest.approx(1.0)
    assert positions[10, 2] == pytest.approx(0.0)
    assert positions[10].sum() == pytest.approx(1.0)


def test_factor_position_variants_match_each_individual_exposure() -> None:
    data = _market_data()
    config = _config(FactorFamily.TREND)
    variants = build_factor_position_variants(data, config, alt_symbols=("ALPHA", "BETA", "GAMMA"))

    assert np.array_equal(variants.dollar_neutral, build_factor_positions(data, config, alt_symbols=("ALPHA", "BETA", "GAMMA")))
    assert np.array_equal(
        variants.long_only,
        build_factor_positions(data, config, alt_symbols=("ALPHA", "BETA", "GAMMA"), exposure=FactorExposure.LONG_ONLY),
    )


def test_liquidity_guard_uses_prior_volume_and_flattens_positions() -> None:
    data = _market_data()
    volumes = data.quote_volume.copy()
    volumes[9, 1:] = 1.0
    filtered = MarketData(
        data.dates,
        data.symbols,
        data.open_price,
        data.high_price,
        data.low_price,
        data.close_price,
        volumes,
    )
    positions = build_factor_positions(filtered, _config(FactorFamily.VOLUME_TREND), alt_symbols=("ALPHA", "BETA", "GAMMA"))
    assert np.array_equal(positions[10], np.zeros(4))


def test_residual_trend_and_reversal_are_dollar_neutral() -> None:
    data = _market_data()
    for family, expected_alpha in ((FactorFamily.RESIDUAL_TREND, 0.5), (FactorFamily.RESIDUAL_REVERSAL, -0.5)):
        positions = build_factor_positions(data, _config(family), alt_symbols=("ALPHA", "BETA", "GAMMA"))
        assert positions[12, 1] == pytest.approx(expected_alpha)
        assert positions[12].sum() == pytest.approx(0.0)


def test_anchored_reversal_longs_the_alt_closest_to_its_trailing_low_without_future_prices() -> None:
    data = _market_data()
    config = _config(FactorFamily.ANCHORED_REVERSAL)
    baseline = build_factor_positions(data, config, alt_symbols=("ALPHA", "BETA", "GAMMA"))
    changed_close = data.close_price.copy()
    changed_close[10] *= np.asarray((0.1, 0.1, 10.0, 0.1))
    changed = MarketData(
        data.dates,
        data.symbols,
        changed_close,
        changed_close,
        changed_close,
        changed_close,
        data.quote_volume,
    )

    mutated = build_factor_positions(changed, config, alt_symbols=("ALPHA", "BETA", "GAMMA"))

    assert np.array_equal(baseline[10], mutated[10])
    assert baseline[10, 2] == pytest.approx(0.5)
    assert baseline[10, 1] == pytest.approx(-0.5)
    assert baseline[10].sum() == pytest.approx(0.0)


def test_requires_btc_factor_and_sufficient_distinct_alts() -> None:
    data = _market_data()
    data_without_btc = MarketData(
        data.dates,
        ("ALPHA", "BETA", "GAMMA"),
        data.open_price[:, 1:],
        data.high_price[:, 1:],
        data.low_price[:, 1:],
        data.close_price[:, 1:],
        data.quote_volume[:, 1:],
    )
    with pytest.raises(SelectionBiasError, match="BTC"):
        build_factor_positions(data_without_btc, _config(FactorFamily.RESIDUAL_TREND), alt_symbols=("ALPHA", "BETA", "GAMMA"))
    with pytest.raises(SelectionBiasError, match="distinct"):
        build_factor_positions(data, _config(FactorFamily.TREND, top_k=2), alt_symbols=("ALPHA", "BETA", "GAMMA"))
