#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Load only audit-eligible Bitget daily candles into the paper-search model."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import bitget_history_audit as audit
import bitget_history_data as history
from alt_search_models import FloatMatrix, MarketData


@dataclass(frozen=True, slots=True)
class ArchiveDailyMetadata:
    snapshot_ms: int
    common_rows: int
    integrity_verified: bool
    source_hashes: tuple[str, ...]


class ArchiveDailyError(RuntimeError):
    """A public archive cannot support a reproducible daily search."""


@dataclass(frozen=True, slots=True)
class DailyBar:
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    quote_volume: float


def _symbol_entries(manifest: audit.ArchiveManifest, symbol: str) -> tuple[audit.ManifestEntry, ...]:
    matching = tuple(entry for entry in manifest.partitions if Path(entry.relative_path).parts[:2] == (symbol, "1D"))
    if not matching:
        raise ArchiveDailyError(f"archive has no daily partitions for {symbol}")
    return matching


def _read_bars(root: Path, entries: tuple[audit.ManifestEntry, ...]) -> dict[datetime, DailyBar]:
    rows: dict[datetime, DailyBar] = {}
    for entry in sorted(entries, key=lambda item: (item.window_start_ms, item.window_end_ms)):
        path = root / entry.relative_path
        try:
            with gzip.open(path, mode="rt", encoding="utf-8") as handle:
                for line in handle:
                    candle = audit.StoredCandle.model_validate_json(line).candle_row()
                    if not candle.ohlc_valid:
                        raise ArchiveDailyError(f"archive contains invalid OHLC in {entry.relative_path}")
                    timestamp = datetime.fromtimestamp(candle.timestamp_ms / 1000.0, UTC)
                    bar = DailyBar(timestamp, candle.open_price, candle.high_price, candle.low_price, candle.close_price, candle.quote_volume_usdt)
                    previous = rows.get(timestamp)
                    if previous is not None and previous != bar:
                        raise ArchiveDailyError(f"archive has conflicting duplicate daily bars in {entry.relative_path}")
                    rows[timestamp] = bar
        except OSError as error:
            raise ArchiveDailyError(f"cannot decompress {entry.relative_path}") from error
    if not rows:
        raise ArchiveDailyError("archive does not contain any daily bars")
    return rows


def _matrix(dates: tuple[datetime, ...], bars: tuple[dict[datetime, DailyBar], ...], field: str) -> FloatMatrix:
    values = np.empty((len(dates), len(bars)), dtype=float)
    for column, by_date in enumerate(bars):
        for row, date in enumerate(dates):
            bar = by_date[date]
            if field == "open":
                values[row, column] = bar.open_price
            elif field == "high":
                values[row, column] = bar.high_price
            elif field == "low":
                values[row, column] = bar.low_price
            elif field == "close":
                values[row, column] = bar.close_price
            elif field == "volume":
                values[row, column] = bar.quote_volume
            else:
                raise ArchiveDailyError(f"unknown matrix field: {field}")
    return values


def load_verified_daily_archive(root: Path, assets: tuple[str, ...]) -> tuple[MarketData, ArchiveDailyMetadata]:
    """Require a complete, hash-valid, gap-free archive before materializing data."""
    manifest, audits = audit.audit_archive(root)
    wanted = tuple(f"{asset}USDT" for asset in assets)
    by_group = {(item.symbol, item.interval): item for item in audits}
    for symbol in wanted:
        group = by_group.get((symbol, "1D"))
        if group is None or group.missing_intervals or group.invalid_ohlc_rows or group.hash_failures:
            raise ArchiveDailyError(f"daily archive is not eligible for {symbol}")
    if not manifest.complete:
        raise ArchiveDailyError("daily archive collection is incomplete")
    bar_maps = tuple(_read_bars(root, _symbol_entries(manifest, symbol)) for symbol in wanted)
    common = set(bar_maps[0])
    for by_date in bar_maps[1:]:
        common.intersection_update(by_date)
    dates = tuple(sorted(common))
    if not dates:
        raise ArchiveDailyError("no common daily history across the requested assets")
    source_hashes = tuple(entry.sha256 for entry in manifest.partitions if Path(entry.relative_path).parts[:2] in {(symbol, "1D") for symbol in wanted})
    data = MarketData(
        dates=dates,
        symbols=assets,
        open_price=_matrix(dates, bar_maps, "open"),
        high_price=_matrix(dates, bar_maps, "high"),
        low_price=_matrix(dates, bar_maps, "low"),
        close_price=_matrix(dates, bar_maps, "close"),
        quote_volume=_matrix(dates, bar_maps, "volume"),
    )
    return data, ArchiveDailyMetadata(manifest.snapshot_ms, len(dates), True, source_hashes)
