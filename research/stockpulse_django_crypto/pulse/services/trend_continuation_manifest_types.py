from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

from .trend_continuation_contract import Direction, stable_manifest_digest


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ForwardIntervalOverlapError(Exception):
    contract_symbol: str
    interval_start: datetime
    interval_end: datetime

    def __str__(self) -> str:
        return f"closed forward interval overlaps for {self.contract_symbol}: {self.interval_start}..{self.interval_end}"


@dataclass(frozen=True, slots=True)
class ManifestLifecycleError(Exception):
    manifest_digest: str
    reason: str

    def __str__(self) -> str:
        return f"manifest {self.manifest_digest} is terminal: {self.reason}"


@dataclass(frozen=True, slots=True)
class TrendContinuationSelectionSnapshot:
    payload: JsonObject

    @property
    def digest(self) -> str:
        return stable_manifest_digest(self.payload)


@dataclass(frozen=True, slots=True)
class ManifestCandidate:
    contract: str
    profile: str
    trend_direction: Direction
    rule_name: str
    candidate_horizon_bars: int
    strategy_version: str
    feature_schema_version: str
