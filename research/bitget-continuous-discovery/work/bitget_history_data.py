#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2"]
# ///

"""Validated, resumable storage primitives for Bitget public candle history."""

from __future__ import annotations

import gzip
import hashlib
import json
from math import isfinite
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError


HISTORY_ROUTE: Final = "/api/v3/market/history-candles"
MAX_PAGE_ROWS: Final = 100
INTERVAL_MS: Final = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1H": 3_600_000,
    "4H": 14_400_000,
    "6H": 21_600_000,
    "12H": 43_200_000,
    "1D": 86_400_000,
}


class HistoryDataError(RuntimeError):
    """A public-history response or local artifact that cannot be trusted."""


class CandleEnvelope(BaseModel):
    """Strict V3 market-history boundary model."""

    model_config = ConfigDict(frozen=True)
    code: str
    msg: str
    data: tuple[tuple[str, str, str, str, str, str, str], ...]


@dataclass(frozen=True, slots=True)
class CandleRow:
    timestamp_ms: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    base_volume: float
    quote_volume_usdt: float
    ohlc_valid: bool

    @classmethod
    def from_api_row(cls, row: Sequence[str]) -> CandleRow:
        if len(row) != 7:
            raise HistoryDataError(f"expected seven candle fields, received {len(row)}")
        try:
            timestamp_ms = int(row[0])
            open_price = float(row[1])
            high_price = float(row[2])
            low_price = float(row[3])
            close_price = float(row[4])
            base_volume = float(row[5])
            quote_volume_usdt = float(row[6])
        except ValueError as error:
            raise HistoryDataError("candle row has a non-numeric field") from error
        values = (open_price, high_price, low_price, close_price, base_volume, quote_volume_usdt)
        if timestamp_ms < 0 or not all(isfinite(value) for value in values):
            raise HistoryDataError("candle timestamp or numeric value is invalid")
        ohlc_valid = (
            min(open_price, high_price, low_price, close_price) > 0.0
            and base_volume >= 0.0
            and quote_volume_usdt >= 0.0
            and high_price >= max(open_price, close_price)
            and low_price <= min(open_price, close_price)
        )
        return cls(timestamp_ms, open_price, high_price, low_price, close_price, base_volume, quote_volume_usdt, ohlc_valid)


@dataclass(frozen=True, slots=True)
class HistoryWindow:
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise HistoryDataError("history window is invalid")


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    relative_path: str
    sha256: str
    rows: int
    missing_intervals: int
    invalid_ohlc_rows: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    window_start_ms: int
    window_end_ms: int


RequestText = Callable[[str, dict[str, str]], str]


def closed_end_ms(now_ms: int, interval: str) -> int:
    """Return the last completed candle's exclusive-boundary millisecond."""
    duration = interval_duration_ms(interval)
    return (now_ms // duration) * duration - 1


def interval_duration_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError as error:
        raise HistoryDataError(f"unsupported V3 history interval: {interval}") from error


def partition_windows(*, start_ms: int, end_ms: int, span_ms: int) -> tuple[HistoryWindow, ...]:
    if span_ms <= 0:
        raise HistoryDataError("partition span must be positive")
    if end_ms < start_ms:
        raise HistoryDataError("partition range is invalid")
    windows: list[HistoryWindow] = []
    cursor = start_ms
    while cursor <= end_ms:
        window_end = min(end_ms, cursor + span_ms - 1)
        windows.append(HistoryWindow(start_ms=cursor, end_ms=window_end))
        cursor = window_end + 1
    return tuple(windows)


def last_closed_candle_start_ms(now_ms: int, interval: str) -> int:
    duration = interval_duration_ms(interval)
    if now_ms < duration:
        raise HistoryDataError("exchange snapshot precedes one full candle")
    return (now_ms // duration) * duration - duration


def candle_windows(
    *,
    start_ms: int,
    end_candle_ms: int,
    interval: str,
    max_query_range_ms: int,
) -> tuple[HistoryWindow, ...]:
    """Partition aligned candle starts so each exclusive API range stays bounded."""
    duration = interval_duration_ms(interval)
    if max_query_range_ms < duration or max_query_range_ms % duration:
        raise HistoryDataError("query range must be a positive whole number of candles")
    if start_ms < 0 or end_candle_ms < start_ms or start_ms % duration or end_candle_ms % duration:
        raise HistoryDataError("candle window endpoints must be aligned timestamps")
    candles_per_window = max_query_range_ms // duration
    windows: list[HistoryWindow] = []
    cursor = start_ms
    while cursor <= end_candle_ms:
        window_end = min(end_candle_ms, cursor + (candles_per_window - 1) * duration)
        windows.append(HistoryWindow(start_ms=cursor, end_ms=window_end))
        cursor = window_end + duration
    return tuple(windows)


def parse_candle_page(raw_text: str, *, route: str) -> tuple[CandleRow, ...]:
    try:
        payload = CandleEnvelope.model_validate_json(raw_text)
    except ValidationError as error:
        raise HistoryDataError(f"{route}: malformed JSON payload") from error
    if payload.code != "00000":
        raise HistoryDataError(f"{route}: {payload.msg}")
    rows = tuple(CandleRow.from_api_row(row) for row in payload.data)
    timestamps = tuple(row.timestamp_ms for row in rows)
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise HistoryDataError(f"{route}: timestamps are not strictly increasing")
    return rows


def fetch_window(
    *,
    request: RequestText,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    max_request_end_ms: int | None = None,
) -> tuple[CandleRow, ...]:
    """Read a bounded time window backwards; reject API pagination anomalies."""
    window = HistoryWindow(start_ms=start_ms, end_ms=end_ms)
    duration = interval_duration_ms(interval)
    if max_request_end_ms is not None and max_request_end_ms <= window.start_ms:
        raise HistoryDataError("exchange snapshot precedes the requested history window")
    rows_by_timestamp: dict[int, CandleRow] = {}
    cursor = window.end_ms
    while cursor >= window.start_ms:
        response = request(
            HISTORY_ROUTE,
            {
                "category": "USDT-FUTURES",
                "symbol": symbol,
                "interval": interval,
                "startTime": str(window.start_ms),
                "endTime": str(min(cursor + duration, max_request_end_ms) if max_request_end_ms is not None else cursor + duration),
                "limit": str(MAX_PAGE_ROWS),
            },
        )
        page = parse_candle_page(response, route=f"history-candles/{symbol}/{interval}")
        if not page:
            break
        earliest_allowed = max(0, window.start_ms - duration)
        if any(row.timestamp_ms < earliest_allowed or row.timestamp_ms > cursor for row in page):
            raise HistoryDataError("history page exceeded its documented one-candle boundary allowance")
        usable = tuple(row for row in page if earliest_allowed <= row.timestamp_ms <= cursor)
        if not usable:
            raise HistoryDataError("history page has no usable rows inside its requested window")
        for row in usable:
            previous = rows_by_timestamp.get(row.timestamp_ms)
            if previous is not None and previous != row:
                raise HistoryDataError("history page changed a previously observed candle")
            rows_by_timestamp[row.timestamp_ms] = row
        oldest = min(row.timestamp_ms for row in usable)
        if oldest <= window.start_ms:
            break
        cursor = oldest if oldest < cursor else oldest - duration
    return tuple(rows_by_timestamp[timestamp] for timestamp in sorted(rows_by_timestamp))


def archive_relative_path(*, symbol: str, interval: str, window: HistoryWindow) -> Path:
    if not symbol.isalnum() or interval not in INTERVAL_MS:
        raise HistoryDataError("unsafe archive path component")
    return Path(symbol) / interval / f"{window.start_ms}_{window.end_ms}.jsonl.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_partition(
    *,
    root: Path,
    symbol: str,
    interval: str,
    window: HistoryWindow,
    rows: Sequence[CandleRow],
) -> ArchiveEntry:
    """Atomically persist one immutable, hash-addressed history partition."""
    ordered = tuple(sorted(rows, key=lambda row: row.timestamp_ms))
    timestamps = tuple(row.timestamp_ms for row in ordered)
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise HistoryDataError("partition rows are not strictly increasing")
    relative = archive_relative_path(symbol=symbol, interval=interval, window=window)
    duration = interval_duration_ms(interval)
    missing_intervals = sum((later - earlier) // duration - 1 for earlier, later in zip(timestamps, timestamps[1:]))
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with partial.open("wb") as binary_handle:
        with gzip.GzipFile(fileobj=binary_handle, mode="wb", mtime=0) as compressed:
            for row in ordered:
                line = json.dumps(asdict(row), allow_nan=False, separators=(",", ":")) + "\n"
                compressed.write(line.encode("utf-8"))
    os.replace(partial, target)
    return ArchiveEntry(
        relative_path=relative.as_posix(),
        sha256=_sha256(target),
        rows=len(ordered),
        missing_intervals=missing_intervals,
        invalid_ohlc_rows=sum(not row.ohlc_valid for row in ordered),
        first_timestamp_ms=timestamps[0] if timestamps else None,
        last_timestamp_ms=timestamps[-1] if timestamps else None,
        window_start_ms=window.start_ms,
        window_end_ms=window.end_ms,
    )


def valid_partition(*, root: Path, entry: ArchiveEntry) -> bool:
    root_path = root.resolve()
    candidate = (root_path / entry.relative_path).resolve()
    if not candidate.is_relative_to(root_path) or not candidate.is_file() or _sha256(candidate) != entry.sha256:
        return False
    timestamps: list[int] = []
    invalid_ohlc_rows = 0
    try:
        with gzip.open(candidate, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                record = CandleRow(**json.loads(line))
                timestamps.append(record.timestamp_ms)
                invalid_ohlc_rows += int(not record.ohlc_valid)
    except (OSError, TypeError, ValueError):
        return False
    return (
        len(timestamps) == entry.rows
        and sum((later - earlier) // interval_duration_ms(candidate.parent.name) - 1 for earlier, later in zip(timestamps, timestamps[1:])) == entry.missing_intervals
        and invalid_ohlc_rows == entry.invalid_ohlc_rows
        and (timestamps[0] if timestamps else None) == entry.first_timestamp_ms
        and (timestamps[-1] if timestamps else None) == entry.last_timestamp_ms
        and all(later > earlier for earlier, later in zip(timestamps, timestamps[1:]))
    )
