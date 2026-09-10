from collections import Counter

from bitget_adaptive_counterfactual import (
    AdaptiveFamily,
    adaptive_configs,
    family,
    identifier,
    overlay,
)
from run_bitget_4h_pair_regime_search import spec_for_interval


def test_grid_contains_unique_olmar_and_directional_change_trials_without_btc_gate_duplicates() -> None:
    # Given: the interval-aware 5-minute search specification.
    spec = spec_for_interval("5m")

    # When: adaptive counterfactual configurations are generated.
    configs = adaptive_configs(spec)

    # Then: each model/volatility pair is counted once and no ignored BTC cap remains.
    assert len(configs) == 48
    assert Counter(family(config) for config in configs) == {
        AdaptiveFamily.OLMAR: 24,
        AdaptiveFamily.DIRECTIONAL_CHANGE: 24,
    }
    assert len({identifier(config) for config in configs}) == len(configs)
    assert all("btc_" not in identifier(config) for config in configs)


def test_every_adaptive_trial_uses_the_five_x_counterfactual_cap() -> None:
    # Given: every adaptive trial in the predeclared grid.
    configs = adaptive_configs(spec_for_interval("5m"))

    # When / Then: the risk overlay caps leverage at five.
    assert {overlay(config).max_leverage for config in configs} == {5.0}
