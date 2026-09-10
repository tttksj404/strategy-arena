from itertools import combinations

from run_alt_strategy_search import SYMBOLS
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_dynamic_pair_search import _configs


def test_dynamic_pair_search_covers_full_alt_pair_pool_under_five_x_cap() -> None:
    spec = spec_for_interval("4H")
    configs = _configs(spec)
    expected_pairs = set(combinations(SYMBOLS, 2))
    assert {config.pair_config.pair for config in configs} == expected_pairs
    assert {config.overlay.max_leverage for config in configs} == {5.0}
    assert {config.pair_config.entry_z for config in configs} == {1.5, 2.0}
