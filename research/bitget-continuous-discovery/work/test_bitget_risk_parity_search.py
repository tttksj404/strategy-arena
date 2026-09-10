from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_risk_parity_search import _risk_parity_configs, _search_configs


def test_predeclared_risk_parity_grid_has_diversified_horizons_and_five_x_cap() -> None:
    spec = spec_for_interval("4H")
    signals = _risk_parity_configs(spec)
    configs = _search_configs(spec)
    assert len(signals) == 36
    assert {signal.momentum_lookback for signal in signals} == {30, 90, 180}
    assert {signal.volatility_window for signal in signals} == {30, 90}
    assert {signal.top_k for signal in signals} == {2, 3}
    assert {signal.rebalance_bars for signal in signals} == {6, 18, 42}
    assert len(configs) == len(signals) * 2 * len(spec.volatility_caps) * len(spec.target_bar_volatilities)
    assert {config.overlay.max_leverage for config in configs} == {5.0}
