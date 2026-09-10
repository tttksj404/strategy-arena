#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Load a verified archive without giving later listings artificial early history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import bitget_history_audit as audit
from alt_search_models import FloatMatrix, MarketData
from bitget_archive_panel import (
    ArchivePanelError,
    CandleBar,
    _expected_entries,
    _manifest,
    _read_bars,
)


@dataclass(frozen=True, slots=True)
class DynamicArchivePanelMetadata:
    snapshot_ms: int
    interval: str
    reference_rows: int
    integrity_verified: bool
    minimum_listing_bars: int
    source_hashes: tuple[str, ...]


def _verify_groups(
    root: Path,
    *,
    assets: tuple[str, ...],
    interval: str,
) -> tuple[tuple[audit.ManifestEntry, ...], ...]:
    manifest = _manifest(root)
    entries_by_symbol = tuple(_expected_entries(manifest, symbol=f"{asset}USDT", interval=interval) for asset in assets)
    for asset, entries in zip(assets, entries_by_symbol, strict=True):
        report = audit.audit_group(root=root, symbol=f"{asset}USDT", interval=interval, entries=tuple(entry.archive_entry() for entry in entries))
        if report.missing_intervals or report.invalid_ohlc_rows or report.hash_failures:
            raise ArchivePanelError(f"dynamic archive interval is not eligible for {asset}")
    return entries_by_symbol


def _available_index(dates: tuple[datetime, ...], bars: dict[datetime, CandleBar]) -> int:
    available = np.fromiter((date in bars for date in dates), dtype=bool, count=len(dates))
    indices = np.flatnonzero(available)
    if not len(indices):
        raise ArchivePanelError("dynamic archive asset has no bars on the BTC reference calendar")
    first = int(indices[0])
    if not available[first:].all():
        raise ArchivePanelError("dynamic archive asset has an unverified gap after its first bar")
    return first


def _dynamic_matrix(
    dates: tuple[datetime, ...],
    bars: tuple[dict[datetime, CandleBar], ...],
    *,
    field: str,
    minimum_listing_bars: int,
) -> FloatMatrix:
    attributes = {
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
        "close": "close_price",
        "volume": "quote_volume",
    }
    try:
        attribute = attributes[field]
    except KeyError as error:
        raise ArchivePanelError(f"unknown dynamic matrix field: {field}") from error
    values = np.empty((len(dates), len(bars)), dtype=float)
    for column, by_date in enumerate(bars):
        first = _available_index(dates, by_date)
        first_value = float(getattr(by_date[dates[first]], attribute))
        values[:first, column] = 0.0 if field == "volume" else first_value
        for row in range(first, len(dates)):
            values[row, column] = float(getattr(by_date[dates[row]], attribute))
        if field == "volume":
            values[: first + minimum_listing_bars, column] = 0.0
    return values


def load_verified_dynamic_interval_archive(
    root: Path,
    assets: tuple[str, ...],
    *,
    interval: str,
    minimum_listing_bars: int,
) -> tuple[MarketData, DynamicArchivePanelMetadata]:
    """Materialize a BTC-calendar panel with causal post-listing trade eligibility.

    Every asset must pass hash and continuity checks.  Pre-listing prices are a
    harmless flat fill solely to keep the matrix finite; quote volume is zero until
    the listing has accumulated ``minimum_listing_bars`` of its own observed history.
    """
    if "BTC" not in assets or len(set(assets)) != len(assets) or minimum_listing_bars < 0:
        raise ArchivePanelError("dynamic archive requires distinct assets including BTC and a non-negative listing delay")
    manifest = _manifest(root)
    entries_by_symbol = _verify_groups(root, assets=assets, interval=interval)
    bars = tuple(_read_bars(root, entries) for entries in entries_by_symbol)
    dates = tuple(sorted(bars[assets.index("BTC")]))
    if not dates:
        raise ArchivePanelError("dynamic archive BTC reference calendar is empty")
    data = MarketData(
        dates=dates,
        symbols=assets,
        open_price=_dynamic_matrix(dates, bars, field="open", minimum_listing_bars=minimum_listing_bars),
        high_price=_dynamic_matrix(dates, bars, field="high", minimum_listing_bars=minimum_listing_bars),
        low_price=_dynamic_matrix(dates, bars, field="low", minimum_listing_bars=minimum_listing_bars),
        close_price=_dynamic_matrix(dates, bars, field="close", minimum_listing_bars=minimum_listing_bars),
        quote_volume=_dynamic_matrix(dates, bars, field="volume", minimum_listing_bars=minimum_listing_bars),
    )
    source_hashes = tuple(entry.sha256 for entries in entries_by_symbol for entry in entries)
    return data, DynamicArchivePanelMetadata(
        snapshot_ms=manifest.snapshot_ms,
        interval=interval,
        reference_rows=len(dates),
        integrity_verified=True,
        minimum_listing_bars=minimum_listing_bars,
        source_hashes=source_hashes,
    )
