from __future__ import annotations

import json

import pytest

from bitget_olmar_chunk_contract import (
    ChunkArtifact,
    ChunkContractError,
    ChunkRecord,
    complete_records,
)


def _artifact(start: int, end: int) -> ChunkArtifact:
    return ChunkArtifact(
        interval="5m",
        snapshot_ms=1,
        common_rows=100,
        integrity_verified=True,
        source_hashes=("source",),
        train_end=50,
        development_end=75,
        locked_test_rows=25,
        full_configurations=4,
        start_index=start,
        end_index=end,
        bin_counts=(10, 10),
        records=tuple(
            ChunkRecord(index, f"config-{index}", float(index), (float(index), 1.0), (float(index * index), 1.0))
            for index in range(start, end)
        ),
    )


def test_chunk_contract_round_trips_and_merges_complete_disjoint_ranges() -> None:
    first = _artifact(0, 2)
    second = _artifact(2, 4)

    assert ChunkArtifact.from_payload(first.payload()) == first
    assert [record.index for record in complete_records((first, second))] == [0, 1, 2, 3]


def test_chunk_contract_rejects_missing_or_overlapping_configuration_indexes() -> None:
    first = _artifact(0, 2)
    overlapping = _artifact(1, 3)

    with pytest.raises(ChunkContractError, match="coverage"):
        complete_records((first, overlapping))


def test_chunk_contract_round_trips_a_noncandidate_validation_score_as_json_null() -> None:
    artifact = ChunkArtifact(
        interval="5m",
        snapshot_ms=1,
        common_rows=100,
        integrity_verified=True,
        source_hashes=("source",),
        train_end=50,
        development_end=75,
        locked_test_rows=25,
        full_configurations=1,
        start_index=0,
        end_index=1,
        bin_counts=(10, 10),
        records=(ChunkRecord(0, "config-0", None, (0.0, 1.0), (0.0, 1.0)),),
    )

    encoded = json.dumps(artifact.payload(), allow_nan=False)

    assert ChunkArtifact.from_payload(json.loads(encoded)).records[0].validation_score is None
