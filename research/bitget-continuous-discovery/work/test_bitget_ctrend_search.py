from bitget_ctrend_signals import CompositeStyle
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_ctrend_search import _ctrend_configs, _search_configs


def test_predeclared_composite_trend_suite_has_multiple_algorithm_styles_and_five_x_cap() -> None:
    spec = spec_for_interval("4H")
    signals = _ctrend_configs(spec)
    configs = _search_configs(spec)
    assert len(signals) == 12
    assert {signal.style for signal in signals} == set(CompositeStyle)
    assert {(signal.short_lookback, signal.medium_lookback, signal.long_lookback) for signal in signals} == {(6, 30, 90), (30, 90, 180)}
    assert {signal.top_k for signal in signals} == {1, 2}
    assert len(configs) == len(signals) * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
