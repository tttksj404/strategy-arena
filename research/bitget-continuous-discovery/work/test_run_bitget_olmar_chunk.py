from __future__ import annotations

from math import inf

import pytest

from run_bitget_4h_pair_regime_search import spec_for_interval
from run_bitget_olmar_chunk import (
    ChunkRunError,
    _chunk_bounds,
    _model_group_size,
    _stored_validation_score,
)
from run_bitget_olmar_search import search_configs


def test_olmar_chunk_runner_requires_complete_model_groups() -> None:
    configs = search_configs(spec_for_interval("5m"))

    assert _model_group_size(configs) == 4
    assert _chunk_bounds(configs, start_index=4, end_index=8) == (4, 8)
    with pytest.raises(ChunkRunError, match="model-group"):
        _chunk_bounds(configs, start_index=1, end_index=5)


def test_olmar_chunk_runner_stores_a_nonfinite_validation_score_as_null() -> None:
    assert _stored_validation_score(-inf) is None
    assert _stored_validation_score(0.25) == 0.25
