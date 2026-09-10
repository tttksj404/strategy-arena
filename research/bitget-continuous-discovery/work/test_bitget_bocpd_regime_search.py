from __future__ import annotations

from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_bocpd_regime_search import _search_configs


def test_bocpd_grid_spans_changepoint_and_risk_choices() -> None:
    configs = _search_configs(spec_for_interval("4H"))

    assert len(configs) == 64
    assert {config.model.hazard_bars for config in configs} == {24, 96}
    assert {config.model.change_probability_threshold for config in configs} == {0.15, 0.3}
    assert {config.model.high_volatility_cash for config in configs} == {False, True}
