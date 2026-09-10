from __future__ import annotations

import numpy as np

from bitget_olmar_chunk_contract import ChunkRecord
from bitget_olmar_chunk_merge import _moment_arrays, _winner_record


def _record(index: int, score: float | None) -> ChunkRecord:
    return ChunkRecord(
        index=index,
        identifier=f"config-{index}",
        validation_score=score,
        sums=(float(index), float(index + 1)),
        squares=(float(index * index), float((index + 1) * (index + 1))),
    )


def test_merge_selects_the_highest_finite_validation_record() -> None:
    records = (_record(0, None), _record(1, 0.3), _record(2, 0.5))

    winner = _winner_record(records)

    assert winner == _record(2, 0.5)


def test_merge_preserves_configuration_order_in_cscv_moments() -> None:
    records = (_record(0, 0.1), _record(1, 0.2))

    sums, squares = _moment_arrays(records)

    assert np.array_equal(sums, np.array(((0.0, 1.0), (1.0, 2.0))))
    assert np.array_equal(squares, np.array(((0.0, 1.0), (1.0, 4.0))))
