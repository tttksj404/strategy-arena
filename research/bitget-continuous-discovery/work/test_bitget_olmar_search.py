from __future__ import annotations

from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_olmar_search import search_configs


def test_olmar_grid_spans_online_learning_choices() -> None:
    configs = search_configs(spec_for_interval("4H"))

    assert len(configs) == 48
    assert {config.model.moving_average_lookback for config in configs} == {12, 30, 90}
    assert {config.model.maximum_step for config in configs} == {0.5, 2.0}
