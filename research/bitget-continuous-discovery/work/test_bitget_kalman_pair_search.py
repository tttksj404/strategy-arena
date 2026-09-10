from itertools import combinations

from run_alt_strategy_search import SYMBOLS
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_kalman_pair_search import _configs


def test_kalman_pair_search_covers_full_alt_pool_and_state_noise_grid_under_five_x_cap() -> None:
    spec = spec_for_interval("4H")
    configs = _configs(spec)
    pair_configs = [config.pair_config for config in configs]
    assert {config.pair for config in pair_configs} == set(combinations(SYMBOLS, 2))
    assert {config.z_window for config in pair_configs} == {30, 60}
    assert {config.process_noise for config in pair_configs} == {1e-5, 1e-4}
    assert {config.entry_z for config in pair_configs} == {1.5, 2.0}
    assert len(configs) == len(set(combinations(SYMBOLS, 2))) * 2 * 2 * 2 * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
