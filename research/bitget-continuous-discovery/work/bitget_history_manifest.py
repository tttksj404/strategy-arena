"""Collector manifest models kept separate from transport and CLI concerns."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import bitget_history_data as history

type _CandleType = Literal["market", "mark"]
_API_ROOT = "https://api.bitget.com"


class ManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    relative_path: str
    sha256: str
    rows: int
    missing_intervals: int = 0
    invalid_ohlc_rows: int = 0
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    window_start_ms: int
    window_end_ms: int

    @classmethod
    def from_entry(cls, entry: history.ArchiveEntry) -> ManifestEntry:
        return cls(
            relative_path=entry.relative_path,
            sha256=entry.sha256,
            rows=entry.rows,
            missing_intervals=entry.missing_intervals,
            invalid_ohlc_rows=entry.invalid_ohlc_rows,
            first_timestamp_ms=entry.first_timestamp_ms,
            last_timestamp_ms=entry.last_timestamp_ms,
            window_start_ms=entry.window_start_ms,
            window_end_ms=entry.window_end_ms,
        )

    def archive_entry(self) -> history.ArchiveEntry:
        return history.ArchiveEntry(**self.model_dump())


class ArchiveManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: int = 1
    source: str = "Bitget public V3 market history"
    endpoint: str = _API_ROOT + history.HISTORY_ROUTE
    category: str = "USDT-FUTURES"
    candle_type: _CandleType = "market"
    created_at: str
    start_ms: int
    snapshot_ms: int
    symbols: tuple[str, ...]
    intervals: tuple[str, ...]
    complete: bool
    partitions: tuple[ManifestEntry, ...]


class CollectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    root: Path
    start_ms: int
    snapshot_ms: int | None
    symbols: tuple[str, ...]
    intervals: tuple[str, ...]
    candle_type: _CandleType = "market"
    active_intervals: tuple[str, ...] | None = None
    max_partitions: int | None
    pacing_seconds: float = Field(default=0.065, ge=0.05, le=60.0)
