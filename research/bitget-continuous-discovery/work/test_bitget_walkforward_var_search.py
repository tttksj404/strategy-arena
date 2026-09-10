from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_walkforward_var_search import _search_configs, _var_configs


def test_predeclared_var_suite_varies_lag_depth_and_regularization_under_five_x_cap() -> None:
    spec = spec_for_interval("4H")
    models = _var_configs(spec)
    configs = _search_configs(spec)
    assert len(models) == 8
    assert len({model.identifier for model in models}) == len(models)
    assert {model.return_lags for model in models} == {1, 2}
    assert {model.training_bars for model in models} == {120, 240}
    assert {model.penalty for model in models} == {0.1, 1.0}
    assert {model.model_refit_bars for model in models} == {24}
    assert {model.top_k for model in models} == {1}
    assert len(configs) == len(models) * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
