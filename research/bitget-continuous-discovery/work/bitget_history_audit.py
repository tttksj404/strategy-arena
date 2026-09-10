#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2"]
# ///

"""Audit hash integrity and chronological continuity of a Bitget history archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

import bitget_history_data as history


class HistoryAuditError(RuntimeError):
    """An archive cannot be trusted as a complete chronological input."""


class ManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    relative_path: str
    sha256: str
    rows: int
    window_start_ms: int
    window_end_ms: int

    def archive_entry(self) -> history.ArchiveEntry:
        return history.ArchiveEntry(
            relative_path=self.relative_path,
            sha256=self.sha256,
            rows=self.rows,
            missing_intervals=0,
            invalid_ohlc_rows=0,
            first_timestamp_ms=None,
            last_timestamp_ms=None,
            window_start_ms=self.window_start_ms,
            window_end_ms=self.window_end_ms,
        )


class ArchiveManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    complete: bool
    start_ms: int = 0
    snapshot_ms: int
    symbols: tuple[str, ...]
    intervals: tuple[str, ...]
    candle_type: Literal["market", "mark"] = "market"
    partitions: tuple[ManifestEntry, ...]


class StoredCandle(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp_ms: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    base_volume: float
    quote_volume_usdt: float

    def candle_row(self) -> history.CandleRow:
        return history.CandleRow.from_api_row(
            (
                str(self.timestamp_ms),
                str(self.open_price),
                str(self.high_price),
                str(self.low_price),
                str(self.close_price),
                str(self.base_volume),
                str(self.quote_volume_usdt),
            )
        )


@dataclass(frozen=True, slots=True)
class GroupAudit:
    symbol: str
    interval: str
    raw_rows: int
    unique_rows: int
    duplicate_rows: int
    missing_intervals: int
    invalid_ohlc_rows: int
    hash_failures: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_path(root: Path, relative_path: str) -> Path:
    candidate = (root.resolve() / relative_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise HistoryAuditError("manifest contains a path outside the archive root")
    return candidate


def _stored_row(line: str) -> history.CandleRow:
    try:
        return StoredCandle.model_validate_json(line).candle_row()
    except ValidationError as error:
        raise HistoryAuditError("stored candle is malformed") from error


def audit_group(*, root: Path, symbol: str, interval: str, entries: Sequence[history.ArchiveEntry]) -> GroupAudit:
    duration = history.interval_duration_ms(interval)
    raw_rows = unique_rows = duplicate_rows = missing_intervals = invalid_ohlc_rows = hash_failures = 0
    previous: history.CandleRow | None = None
    first_timestamp_ms: int | None = None
    recent: dict[int, history.CandleRow] = {}
    for entry in sorted(entries, key=lambda item: (item.window_start_ms, item.window_end_ms)):
        path = _candidate_path(root, entry.relative_path)
        if not path.is_file() or _sha256(path) != entry.sha256:
            hash_failures += 1
            continue
        try:
            with gzip.open(path, mode="rt", encoding="utf-8") as handle:
                for line in handle:
                    row = _stored_row(line)
                    raw_rows += 1
                    invalid_ohlc_rows += int(not row.ohlc_valid)
                    duplicate = recent.get(row.timestamp_ms)
                    if duplicate is not None:
                        if duplicate != row:
                            raise HistoryAuditError("overlapping partitions disagree on a candle")
                        duplicate_rows += 1
                        continue
                    if previous is not None:
                        if row.timestamp_ms <= previous.timestamp_ms:
                            raise HistoryAuditError("partitions are not chronologically mergeable")
                        delta = row.timestamp_ms - previous.timestamp_ms
                        if delta % duration:
                            raise HistoryAuditError("candle timestamps are not interval-aligned")
                        missing_intervals += delta // duration - 1
                    unique_rows += 1
                    if first_timestamp_ms is None:
                        first_timestamp_ms = row.timestamp_ms
                    previous = row
                    recent[row.timestamp_ms] = row
                    if len(recent) > 2:
                        del recent[min(recent)]
        except (OSError, UnicodeDecodeError) as error:
            raise HistoryAuditError("partition cannot be decompressed") from error
    return GroupAudit(
        symbol=symbol,
        interval=interval,
        raw_rows=raw_rows,
        unique_rows=unique_rows,
        duplicate_rows=duplicate_rows,
        missing_intervals=missing_intervals,
        invalid_ohlc_rows=invalid_ohlc_rows,
        hash_failures=hash_failures,
        first_timestamp_ms=first_timestamp_ms,
        last_timestamp_ms=previous.timestamp_ms if previous is not None else None,
    )


def audit_archive(root: Path) -> tuple[ArchiveManifest, tuple[GroupAudit, ...]]:
    try:
        manifest = ArchiveManifest.model_validate_json((root / "MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise HistoryAuditError("manifest is unreadable") from error
    groups: dict[tuple[str, str], list[history.ArchiveEntry]] = {}
    for manifest_entry in manifest.partitions:
        parts = Path(manifest_entry.relative_path).parts
        if len(parts) != 3:
            raise HistoryAuditError("manifest partition path has an unexpected layout")
        groups.setdefault((parts[0], parts[1]), []).append(manifest_entry.archive_entry())
    reports = tuple(audit_group(root=root, symbol=symbol, interval=interval, entries=entries) for (symbol, interval), entries in sorted(groups.items()))
    return manifest, reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    arguments = parser.parse_args(argv)
    try:
        manifest, reports = audit_archive(arguments.archive_root)
    except HistoryAuditError as error:
        print(f"AUDIT_ERROR={error}")
        return 1
    report_path = arguments.archive_root / "ARCHIVE_AUDIT.json"
    report = {"snapshot_ms": manifest.snapshot_ms, "manifest_complete": manifest.complete, "groups": [asdict(item) for item in reports]}
    report_path.write_text(json.dumps(report, allow_nan=False, indent=2), encoding="utf-8")
    eligible = all(item.missing_intervals == item.invalid_ohlc_rows == item.hash_failures == 0 for item in reports)
    print(f"AUDIT={report_path}")
    print(f"ELIGIBLE={str(eligible and manifest.complete).lower()} GROUPS={len(reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
