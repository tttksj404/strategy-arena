"""Durable, typed artifacts for resumable OLMAR configuration chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class ChunkContractError(ValueError):
    """Raised when a partial-search artifact cannot safely be merged."""


def _mapping(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ChunkContractError(f"{label} must be an object")
    return value


def _integer(value: JsonValue, label: str) -> int:
    if type(value) is not int:
        raise ChunkContractError(f"{label} must be an integer")
    return value


def _number(value: JsonValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChunkContractError(f"{label} must be numeric")
    return float(value)


def _nullable_number(value: JsonValue, label: str) -> float | None:
    if value is None:
        return None
    return _number(value, label)


def _string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise ChunkContractError(f"{label} must be a string")
    return value


def _numbers(value: JsonValue, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ChunkContractError(f"{label} must be an array")
    return tuple(_number(item, label) for item in value)


def _strings(value: JsonValue, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ChunkContractError(f"{label} must be an array")
    return tuple(_string(item, label) for item in value)


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    index: int
    identifier: str
    validation_score: float | None
    sums: tuple[float, ...]
    squares: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            self.index < 0
            or not self.identifier
            or (self.validation_score is not None and not isfinite(self.validation_score))
            or len(self.sums) != len(self.squares)
        ):
            raise ChunkContractError("invalid chunk record")

    def payload(self) -> dict[str, JsonValue]:
        return {
            "index": self.index,
            "identifier": self.identifier,
            "validation_score": self.validation_score,
            "sums": [*self.sums],
            "squares": [*self.squares],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JsonValue]) -> ChunkRecord:
        return cls(
            index=_integer(payload.get("index"), "record.index"),
            identifier=_string(payload.get("identifier"), "record.identifier"),
            validation_score=_nullable_number(payload.get("validation_score"), "record.validation_score"),
            sums=_numbers(payload.get("sums"), "record.sums"),
            squares=_numbers(payload.get("squares"), "record.squares"),
        )


@dataclass(frozen=True, slots=True)
class ChunkArtifact:
    interval: str
    snapshot_ms: int
    common_rows: int
    integrity_verified: bool
    source_hashes: tuple[str, ...]
    train_end: int
    development_end: int
    locked_test_rows: int
    full_configurations: int
    start_index: int
    end_index: int
    bin_counts: tuple[int, ...]
    records: tuple[ChunkRecord, ...]

    def __post_init__(self) -> None:
        expected_indexes = tuple(range(self.start_index, self.end_index))
        observed_indexes = tuple(record.index for record in self.records)
        if (
            not self.interval
            or self.full_configurations <= 0
            or self.start_index < 0
            or self.end_index > self.full_configurations
            or self.start_index >= self.end_index
            or self.train_end <= 0
            or self.development_end <= self.train_end
            or not self.integrity_verified
            or not self.source_hashes
            or not self.bin_counts
            or observed_indexes != expected_indexes
        ):
            raise ChunkContractError("invalid chunk artifact")

    def payload(self) -> dict[str, JsonValue]:
        return {
            "schema": "bitget_olmar_chunk_v1",
            "interval": self.interval,
            "snapshot_ms": self.snapshot_ms,
            "common_rows": self.common_rows,
            "integrity_verified": self.integrity_verified,
            "source_hashes": [*self.source_hashes],
            "train_end": self.train_end,
            "development_end": self.development_end,
            "locked_test_rows": self.locked_test_rows,
            "full_configurations": self.full_configurations,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "bin_counts": [*self.bin_counts],
            "records": [record.payload() for record in self.records],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, JsonValue]) -> ChunkArtifact:
        if payload.get("schema") != "bitget_olmar_chunk_v1":
            raise ChunkContractError("unsupported chunk schema")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ChunkContractError("records must be an array")
        records = tuple(ChunkRecord.from_payload(_mapping(item, "record")) for item in raw_records)
        bin_counts = _numbers(payload.get("bin_counts"), "bin_counts")
        if any(value != int(value) for value in bin_counts):
            raise ChunkContractError("bin_counts must contain integers")
        return cls(
            interval=_string(payload.get("interval"), "interval"),
            snapshot_ms=_integer(payload.get("snapshot_ms"), "snapshot_ms"),
            common_rows=_integer(payload.get("common_rows"), "common_rows"),
            integrity_verified=payload.get("integrity_verified") is True,
            source_hashes=_strings(payload.get("source_hashes"), "source_hashes"),
            train_end=_integer(payload.get("train_end"), "train_end"),
            development_end=_integer(payload.get("development_end"), "development_end"),
            locked_test_rows=_integer(payload.get("locked_test_rows"), "locked_test_rows"),
            full_configurations=_integer(payload.get("full_configurations"), "full_configurations"),
            start_index=_integer(payload.get("start_index"), "start_index"),
            end_index=_integer(payload.get("end_index"), "end_index"),
            bin_counts=tuple(int(value) for value in bin_counts),
            records=records,
        )


def complete_records(chunks: Sequence[ChunkArtifact]) -> tuple[ChunkRecord, ...]:
    if not chunks:
        raise ChunkContractError("at least one chunk is required")
    first = chunks[0]
    shared = (
        first.interval,
        first.snapshot_ms,
        first.common_rows,
        first.integrity_verified,
        first.source_hashes,
        first.train_end,
        first.development_end,
        first.locked_test_rows,
        first.full_configurations,
        first.bin_counts,
    )
    records: list[ChunkRecord] = []
    for chunk in chunks:
        candidate = (
            chunk.interval,
            chunk.snapshot_ms,
            chunk.common_rows,
            chunk.integrity_verified,
            chunk.source_hashes,
            chunk.train_end,
            chunk.development_end,
            chunk.locked_test_rows,
            chunk.full_configurations,
            chunk.bin_counts,
        )
        if candidate != shared:
            raise ChunkContractError("chunk metadata differs")
        records.extend(chunk.records)
    ordered = tuple(sorted(records, key=lambda record: record.index))
    expected = tuple(range(first.full_configurations))
    if tuple(record.index for record in ordered) != expected:
        raise ChunkContractError("chunk coverage is incomplete or overlaps")
    return ordered
