from __future__ import annotations

from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_triple_barrier_search import _search_configs


def test_triple_barrier_grid_spans_label_and_regularization_choices() -> None:
    configs = _search_configs(spec_for_interval("4H"))

    assert len(configs) == 64
    assert {config.model.feature_lookback for config in configs} == {12, 30}
    assert {config.model.barrier_horizon for config in configs} == {6, 12}
    assert {config.model.ridge for config in configs} == {0.1, 1.0}
