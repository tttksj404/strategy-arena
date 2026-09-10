from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_low_turnover_long_only_search import _long_only_configs, _search_configs


def test_low_turnover_long_only_grid_extends_holding_periods_without_exceeding_five_x() -> None:
    spec = spec_for_interval("4H")
    signals = _long_only_configs(spec)
    configs = _search_configs(spec)
    assert len(signals) == 72
    assert {signal.rebalance_bars for signal in signals} == {3, 6, 18, 42}
    assert len(configs) == len(signals) * 2 * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
