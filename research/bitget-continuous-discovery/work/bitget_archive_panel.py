#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Materialize one fully verified Bitget candle interval for paper-only research."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError

import bitget_history_audit as audit
import bitget_history_data as history
from alt_search_models import FloatMatrix, MarketData


class ArchivePanelError(RuntimeError):
    """A requested interval lacks the integrity needed for a paper experiment."""


@dataclass(frozen=True, slots=True)
class ArchivePanelMetadata:
    snapshot_ms: int
    interval: str
    common_rows: int
    integrity_verified: bool
    source_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandleBar:
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    quote_volume: float


def _manifest(root: Path) -> audit.ArchiveManifest:
    try:
        return audit.ArchiveManifest.model_validate_json((root / "MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ArchivePanelError("archive manifest is unreadable") from error


def _expected_entries(
    manifest: audit.ArchiveManifest,
    *,
    symbol: str,
    interval: str,
) -> tuple[audit.ManifestEntry, ...]:
    by_path = {entry.relative_path: entry for entry in manifest.partitions}
    end_candle_ms = history.last_closed_candle_start_ms(manifest.snapshot_ms, interval)
    expected: list[audit.ManifestEntry] = []
    for window in history.candle_windows(
        start_ms=manifest.start_ms,
        end_candle_ms=end_candle_ms,
        interval=interval,
        max_query_range_ms=90 * 24 * 60 * 60 * 1000,
    ):
        relative_path = history.archive_relative_path(symbol=symbol, interval=interval, window=window).as_posix()
        entry = by_path.get(relative_path)
        if entry is None:
            raise ArchivePanelError(f"archive interval is incomplete for {symbol}: {relative_path}")
        expected.append(entry)
    return tuple(expected)


def _safe_path(root: Path, relative_path: str) -> Path:
    candidate = (root.resolve() / relative_path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ArchivePanelError("archive contains a path outside its root")
    return candidate


def _read_bars(root: Path, entries: tuple[audit.ManifestEntry, ...]) -> dict[datetime, CandleBar]:
    rows: dict[datetime, CandleBar] = {}
    for entry in entries:
        path = _safe_path(root, entry.relative_path)
        try:
            with gzip.open(path, mode="rt", encoding="utf-8") as handle:
                for line in handle:
                    candle = audit.StoredCandle.model_validate_json(line).candle_row()
                    if not candle.ohlc_valid:
                        raise ArchivePanelError(f"archive contains invalid OHLC in {entry.relative_path}")
                    timestamp = datetime.fromtimestamp(candle.timestamp_ms / 1000.0, UTC)
                    bar = CandleBar(timestamp, candle.open_price, candle.high_price, candle.low_price, candle.close_price, candle.quote_volume_usdt)
                    previous = rows.get(timestamp)
                    if previous is not None and previous != bar:
                        raise ArchivePanelError(f"archive has conflicting duplicate candles in {entry.relative_path}")
                    rows[timestamp] = bar
        except OSError as error:
            raise ArchivePanelError(f"cannot decompress {entry.relative_path}") from error
    if not rows:
        raise ArchivePanelError("archive interval has no usable candles")
    return rows


def _matrix(dates: tuple[datetime, ...], bars: tuple[dict[datetime, CandleBar], ...], field: str) -> FloatMatrix:
    values = np.empty((len(dates), len(bars)), dtype=float)
    fields = {
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
        "close": "close_price",
        "volume": "quote_volume",
    }
    try:
        attribute = fields[field]
    except KeyError as error:
        raise ArchivePanelError(f"unknown matrix field: {field}") from error
    for column, by_date in enumerate(bars):
        for row, date in enumerate(dates):
            values[row, column] = getattr(by_date[date], attribute)
    return values


def load_verified_interval_archive(root: Path, assets: tuple[str, ...], *, interval: str) -> tuple[MarketData, ArchivePanelMetadata]:
    """Load one interval only when every planned source partition is hash-valid and clean."""
    history.interval_duration_ms(interval)
    manifest = _manifest(root)
    wanted = tuple(f"{asset}USDT" for asset in assets)
    entries_by_symbol = tuple(_expected_entries(manifest, symbol=symbol, interval=interval) for symbol in wanted)
    for symbol, entries in zip(wanted, entries_by_symbol, strict=True):
        report = audit.audit_group(root=root, symbol=symbol, interval=interval, entries=tuple(entry.archive_entry() for entry in entries))
        if report.missing_intervals or report.invalid_ohlc_rows or report.hash_failures:
            raise ArchivePanelError(f"archive interval is not eligible for {symbol}")
    bar_maps = tuple(_read_bars(root, entries) for entries in entries_by_symbol)
    common = set(bar_maps[0])
    for by_date in bar_maps[1:]:
        common.intersection_update(by_date)
    dates = tuple(sorted(common))
    if not dates:
        raise ArchivePanelError("no common interval history across requested assets")
    source_hashes = tuple(entry.sha256 for entries in entries_by_symbol for entry in entries)
    data = MarketData(
        dates=dates,
        symbols=assets,
        open_price=_matrix(dates, bar_maps, "open"),
        high_price=_matrix(dates, bar_maps, "high"),
        low_price=_matrix(dates, bar_maps, "low"),
        close_price=_matrix(dates, bar_maps, "close"),
        quote_volume=_matrix(dates, bar_maps, "volume"),
    )
    return data, ArchivePanelMetadata(manifest.snapshot_ms, interval, len(dates), True, source_hashes)
