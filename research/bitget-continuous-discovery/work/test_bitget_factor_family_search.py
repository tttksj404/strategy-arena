from bitget_factor_signals import FactorFamily
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_factor_family_search import _factor_configs, _search_configs


def test_predeclared_suite_covers_distinct_factor_families_under_five_x_cap() -> None:
    spec = spec_for_interval("4H")
    factors = _factor_configs(spec)
    configs = _search_configs(spec)
    assert {factor.family for factor in factors} == set(FactorFamily)
    assert len({factor.identifier for factor in factors}) == len(factors)
    assert len(configs) == len(factors) * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
    assert all(config.factor.minimum_quote_volume == spec.minimum_quote_volume for config in configs)
