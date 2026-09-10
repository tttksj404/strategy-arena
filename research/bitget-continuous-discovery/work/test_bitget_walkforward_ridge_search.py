from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_walkforward_ridge_search import _ridge_configs, _search_configs


def test_predeclared_walkforward_ridge_suite_is_diverse_and_capped_at_five_x() -> None:
    spec = spec_for_interval("4H")
    ridges = _ridge_configs(spec)
    configs = _search_configs(spec)
    assert len(ridges) == 24
    assert len({ridge.identifier for ridge in ridges}) == len(ridges)
    assert {(ridge.short_lookback, ridge.long_lookback) for ridge in ridges} == {(6, 30), (30, 90)}
    assert {ridge.training_bars for ridge in ridges} == {120, 240}
    assert {ridge.penalty for ridge in ridges} == {0.01, 0.1, 1.0}
    assert {ridge.top_k for ridge in ridges} == {1, 2}
    assert len(configs) == len(ridges) * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
    assert all(config.ridge.minimum_quote_volume == spec.minimum_quote_volume for config in configs)
