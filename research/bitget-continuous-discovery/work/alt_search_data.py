"""Verified loading of the frozen long-horizon daily altcoin cache."""

from __future__ import annotations

import csv
import gzip
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from alt_search_models import FloatMatrix, MarketData


DAILY_TEMPLATE: Final = "binance_fapi_{symbol}USDT_1d.csv.gz"


@dataclass(frozen=True, slots=True)
class CacheIntegrityError(Exception):
    """A frozen-cache input cannot support a reproducible research result."""

    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    """Evidence about the exact frozen source used in the search."""

    frozen_end: str
    common_rows: int
    integrity_verified: bool
    source_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DailyBar:
    """One parsed and closed daily bar."""

    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    quote_volume: float


class ManifestEntry(BaseModel):
    """One checksum row in the cache manifest."""

    model_config = ConfigDict(frozen=True)
    file: str
    sha256: str


class CacheManifest(BaseModel):
    """The minimal manifest fields needed for a trusted input boundary."""

    model_config = ConfigDict(frozen=True)
    frozen_end: str
    files: tuple[ManifestEntry, ...]


class RawDailyBar(BaseModel):
    """The CSV boundary representation before conversion to an internal bar."""

    model_config = ConfigDict(frozen=True)
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    quote_volume: float


def _sha256(path: Path) -> str:
    """Return a file digest without exposing its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bar(row: dict[str, str | None], path: Path) -> DailyBar:
    """Parse a validated CSV row into an immutable closed bar."""

    try:
        raw = RawDailyBar.model_validate(row)
        timestamp = datetime.fromisoformat(raw.timestamp)
    except (ValidationError, ValueError) as error:
        raise CacheIntegrityError(f"invalid daily bar in {path.name}: {error}") from error
    if raw.open <= 0.0 or raw.high <= 0.0 or raw.low <= 0.0 or raw.close <= 0.0 or raw.quote_volume < 0.0:
        raise CacheIntegrityError(f"non-positive price or negative volume in {path.name}")
    return DailyBar(timestamp, raw.open, raw.high, raw.low, raw.close, raw.quote_volume)


def _read_daily_file(path: Path) -> dict[datetime, DailyBar]:
    """Read a gzip CSV into timestamp-indexed internal values."""

    rows: dict[datetime, DailyBar] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                bar = _parse_bar(row, path)
                if bar.timestamp in rows:
                    raise CacheIntegrityError(f"duplicate timestamp in {path.name}: {bar.timestamp.isoformat()}")
                rows[bar.timestamp] = bar
    except OSError as error:
        raise CacheIntegrityError(f"cannot read gzip cache {path.name}: {error}") from error
    if not rows:
        raise CacheIntegrityError(f"empty cache file {path.name}")
    return rows


def _matrix(dates: tuple[datetime, ...], bars: tuple[dict[datetime, DailyBar], ...], field: str) -> FloatMatrix:
    """Materialize one aligned numeric field in the symbol order."""

    values = np.empty((len(dates), len(bars)), dtype=float)
    for column, symbol_bars in enumerate(bars):
        for row, date in enumerate(dates):
            bar = symbol_bars[date]
            match field:
                case "open":
                    values[row, column] = bar.open_price
                case "high":
                    values[row, column] = bar.high_price
                case "low":
                    values[row, column] = bar.low_price
                case "close":
                    values[row, column] = bar.close_price
                case "volume":
                    values[row, column] = bar.quote_volume
                case unreachable:
                    raise CacheIntegrityError(f"unknown requested bar field {unreachable}")
    return values


def load_frozen_daily_cache(cache_dir: Path, symbols: tuple[str, ...]) -> tuple[MarketData, CacheMetadata]:
    """Load only hash-verified rows shared by every fixed altcoin symbol."""

    manifest_path = cache_dir / "manifest.json"
    try:
        manifest = CacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise CacheIntegrityError(f"cannot parse cache manifest: {error}") from error
    expected_hashes = {entry.file: entry.sha256.lower() for entry in manifest.files}
    bar_maps: list[dict[datetime, DailyBar]] = []
    observed_hashes: list[str] = []
    for symbol in symbols:
        path = cache_dir / DAILY_TEMPLATE.format(symbol=symbol)
        expected = expected_hashes.get(path.name)
        if expected is None:
            raise CacheIntegrityError(f"manifest has no checksum for {path.name}")
        actual = _sha256(path)
        if actual != expected:
            raise CacheIntegrityError(f"checksum mismatch for {path.name}")
        observed_hashes.append(actual)
        bar_maps.append(_read_daily_file(path))
    common = set(bar_maps[0])
    for bars in bar_maps[1:]:
        common.intersection_update(bars)
    dates = tuple(sorted(common))
    if not dates:
        raise CacheIntegrityError("no common daily bars across the fixed universe")
    aligned = tuple(bar_maps)
    data = MarketData(
        dates=dates,
        symbols=symbols,
        open_price=_matrix(dates, aligned, "open"),
        high_price=_matrix(dates, aligned, "high"),
        low_price=_matrix(dates, aligned, "low"),
        close_price=_matrix(dates, aligned, "close"),
        quote_volume=_matrix(dates, aligned, "volume"),
    )
    return data, CacheMetadata(manifest.frozen_end, len(dates), True, tuple(observed_hashes))
