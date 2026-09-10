from datetime import UTC, datetime, timedelta

import numpy as np

from alt_search_models import Family, MarketData
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_classic_family_search import _candidate_configs, _reinflated_positions


def _data() -> MarketData:
    step = np.arange(16, dtype=float)
    close = np.column_stack((100.0 * np.exp(0.01 * step), 10.0 * np.exp(0.02 * step), 10.0 * np.exp(-0.01 * step)))
    return MarketData(
        dates=tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(len(step))),
        symbols=("BTC", "ALPHA", "BETA"),
        open_price=close,
        high_price=close * 1.01,
        low_price=close * 0.99,
        close_price=close,
        quote_volume=np.full_like(close, 10_000_000.0),
    )


def test_classic_suite_reuses_six_non_factor_signal_families_without_btc_exposure() -> None:
    spec = spec_for_interval("4H")
    candidates = _candidate_configs(spec)
    assert {candidate.family for candidate in candidates} == {
        Family.TIME_SERIES,
        Family.DONCHIAN,
        Family.EMA,
        Family.BOLLINGER,
        Family.RANGE,
        Family.PRICE_VOLUME,
    }
    positions = _reinflated_positions(_data(), candidates[0], alt_symbols=("ALPHA", "BETA"))
    assert positions.shape == (16, 3)
    assert np.array_equal(positions[:, 0], np.zeros(16))
