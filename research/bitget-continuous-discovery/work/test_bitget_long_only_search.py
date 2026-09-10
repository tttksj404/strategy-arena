from bitget_long_only_signals import LongOnlyStyle
from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_long_only_search import _long_only_configs, _search_configs


def test_predeclared_long_only_suite_uses_positive_btc_regimes_and_five_x_cap() -> None:
    spec = spec_for_interval("4H")
    signals = _long_only_configs(spec)
    configs = _search_configs(spec)
    assert len(signals) == 18
    assert {signal.style for signal in signals} == set(LongOnlyStyle)
    assert {(signal.short_lookback, signal.long_lookback) for signal in signals} == {(6, 30), (30, 90)}
    assert {signal.top_k for signal in signals} == {1, 2, 3}
    assert {config.minimum_btc_trend_z for config in configs} == {0.0, 0.5}
    assert len(configs) == len(signals) * 2 * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
