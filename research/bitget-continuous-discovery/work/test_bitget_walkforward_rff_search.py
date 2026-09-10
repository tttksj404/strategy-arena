from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_walkforward_rff_search import _rff_configs, _search_configs


def test_predeclared_nonlinear_rff_suite_is_diverse_and_capped_at_five_x() -> None:
    spec = spec_for_interval("4H")
    rffs = _rff_configs(spec)
    configs = _search_configs(spec)
    assert len(rffs) == 32
    assert len({rff.identifier for rff in rffs}) == len(rffs)
    assert {(rff.short_lookback, rff.long_lookback) for rff in rffs} == {(6, 30), (30, 90)}
    assert {rff.training_bars for rff in rffs} == {120, 240}
    assert {rff.kernel_gamma for rff in rffs} == {0.25, 1.0}
    assert {rff.penalty for rff in rffs} == {0.01, 0.1}
    assert {rff.top_k for rff in rffs} == {1, 2}
    assert len(configs) == len(rffs) * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
