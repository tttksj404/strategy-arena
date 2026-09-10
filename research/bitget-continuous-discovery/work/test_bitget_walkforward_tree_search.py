from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_walkforward_tree_search import _search_configs, _tree_configs


def test_predeclared_tree_suite_varies_horizons_and_model_depth_under_five_x_cap() -> None:
    spec = spec_for_interval("4H")
    trees = _tree_configs(spec)
    configs = _search_configs(spec)
    assert len(trees) == 8
    assert len({tree.identifier for tree in trees}) == len(trees)
    assert {(tree.short_lookback, tree.long_lookback) for tree in trees} == {(6, 30), (30, 90)}
    assert {tree.training_bars for tree in trees} == {120, 240}
    assert {tree.model_refit_bars for tree in trees} == {24}
    assert {tree.max_depth for tree in trees} == {2, 4}
    assert {tree.trees for tree in trees} == {16}
    assert {tree.min_samples_leaf for tree in trees} == {8}
    assert {tree.top_k for tree in trees} == {1}
    assert len(configs) == len(trees) * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
