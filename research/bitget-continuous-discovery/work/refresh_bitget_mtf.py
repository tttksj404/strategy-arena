# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx2[http2,brotli,zstd]", "pydantic>=2", "typer", "rich"]
# ///
# How to run: uv run work/refresh_bitget_mtf.py
"""Collect a bounded recent update from public market endpoints, without credentials."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Literal, Self

import typer
from pydantic import BaseModel, ConfigDict, model_validator
from rich.logging import RichHandler

from bitget_history_audit import GroupAudit, audit_group
from bitget_history_data import (
    INTERVAL_MS,
    CandleRow,
    HistoryDataError,
    HistoryWindow,
    parse_candle_page,
    write_partition,
)
from bitget_history_manifest import ArchiveManifest, ManifestEntry
from fetch_bitget_history import (
    DEFAULT_SYMBOLS,
    create_client,
    fetch_server_time,
    get_text,
)

WORK = Path(__file__).resolve().parent
ROUTE = "/api/v3/market/candles"
MARK_INTERVALS = ("5m", "1H", "4H", "1D")
LOGGER = logging.getLogger(__name__)
type Kind = Literal["market", "mark"]


class RecentSettingsError(ValueError):
    """Reject unsupported recent-history ranges through Pydantic's value boundary."""


class Settings(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    start_ms: int
    snapshot_ms: int
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    intervals: tuple[str, ...] = tuple(INTERVAL_MS)
    mark_intervals: tuple[str, ...] = MARK_INTERVALS
    credentials_read_or_sent: bool = False
    private_endpoints_used: bool = False
    source: str = "https://api.bitget.com/api/v3/market/candles"

    @model_validator(mode="after")
    def validate_recent_range(self) -> Self:
        if not 0 < self.snapshot_ms - self.start_ms <= 30 * 86_400_000:
            raise RecentSettingsError("This recent-only adapter accepts a positive interval of at most 30 days")
        return self


class GroupResult(BaseModel):
    kind: Kind
    symbol: str
    interval: str
    partition: ManifestEntry
    audit: dict[str, int | str | None]
    boundary_complete: bool

    @property
    def complete(self) -> bool:
        return self.boundary_complete and all(
            self.audit[field] == 0 for field in
            ("hash_failures", "missing_intervals", "duplicate_rows", "invalid_ohlc_rows")
        )


class RefreshReport(BaseModel):
    settings: Settings
    groups: tuple[GroupResult, ...]
    complete: bool
    live_strategy_promotable: bool = False
    limitations: tuple[str, ...] = (
        "Recent update only; does not fill every older archive or delisted universe",
        "Mark candles are not historical margin tiers, funding settlements, or orderbook fills",
        "Collected after strategy research; never silently appended to the old sealed evaluation",
    )


def closed_rows(request: Callable[[dict[str, str]], str], *, symbol: str, interval: str,
                kind: Kind, start: int, snapshot: int) -> tuple[CandleRow, ...]:
    duration = INTERVAL_MS[interval]
    cursor = snapshot
    stored: dict[int, CandleRow] = {}
    while cursor >= start:
        page = parse_candle_page(request({"category": "USDT-FUTURES", "symbol": symbol,
                                         "interval": interval, "type": kind, "limit": "1000",
                                         "startTime": str(max(0, start - duration)), "endTime": str(cursor)}), route=ROUTE)
        if not page:
            break
        if page[-1].timestamp_ms > cursor:
            raise HistoryDataError("Recent page did not move backwards inside its requested boundary")
        for row in page:
            if start <= row.timestamp_ms and row.timestamp_ms + duration <= snapshot:
                if row.timestamp_ms in stored and stored[row.timestamp_ms] != row:
                    raise HistoryDataError("Closed candle changed across pagination")
                stored[row.timestamp_ms] = row
        if page[0].timestamp_ms <= start:
            break
        cursor = page[0].timestamp_ms - duration
    return tuple(stored[stamp] for stamp in sorted(stored))


def complete_coverage(rows: tuple[CandleRow, ...], *, start: int, snapshot: int, duration: int) -> bool:
    if not rows:
        return False
    offset = rows[0].timestamp_ms % duration
    first = ((start - offset + duration - 1) // duration) * duration + offset
    last = ((snapshot - offset) // duration - 1) * duration + offset
    return (rows[0].timestamp_ms == first and rows[-1].timestamp_ms == last
            and len(rows) == (last - first) // duration + 1
            and all(row.ohlc_valid for row in rows))


def audit_values(audit: GroupAudit) -> dict[str, int | str | None]:
    return {"unique_rows": audit.unique_rows, "duplicate_rows": audit.duplicate_rows,
            "missing_intervals": audit.missing_intervals, "invalid_ohlc_rows": audit.invalid_ohlc_rows,
            "hash_failures": audit.hash_failures, "first_timestamp_ms": audit.first_timestamp_ms,
            "last_timestamp_ms": audit.last_timestamp_ms}


def collect_group(root: Path, settings: Settings, job: tuple[Kind, str, str]) -> GroupResult:
    kind, symbol, interval = job
    with create_client() as client:
        # Four independent workers, each paced to at most 3.33 requests/s; no credential loading.
        def request(parameters: dict[str, str]) -> str:
            return get_text(client, ROUTE, parameters, pacing_seconds=.30)

        rows = closed_rows(request, symbol=symbol, interval=interval, kind=kind,
                           start=settings.start_ms, snapshot=settings.snapshot_ms)
    duration = INTERVAL_MS[interval]
    window = HistoryWindow(settings.start_ms, settings.snapshot_ms - duration)
    entry = write_partition(root=root / kind, symbol=symbol, interval=interval, window=window, rows=rows)
    audit = audit_group(root=root / kind, symbol=symbol, interval=interval, entries=(entry,))
    boundary = complete_coverage(rows, start=settings.start_ms, snapshot=settings.snapshot_ms, duration=duration)
    LOGGER.info("REFRESH %s %s %s rows=%s complete=%s", kind, symbol, interval, len(rows), boundary)
    return GroupResult(kind=kind, symbol=symbol, interval=interval, partition=ManifestEntry.from_entry(entry),
                       audit=audit_values(audit), boundary_complete=boundary)


def main(output: Path = WORK / "bitget_history_refresh_20260910",
         start: str = "2026-08-24T00:00:00+00:00") -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)])
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "SETTINGS.json"
    if settings_path.exists():
        settings = Settings.model_validate_json(settings_path.read_bytes())
    else:
        with create_client() as client:
            snapshot = fetch_server_time(client)
        parsed_start = datetime.fromisoformat(start)
        if parsed_start.tzinfo is None:
            raise RecentSettingsError("Start must include an explicit timezone")
        settings = Settings(start_ms=int(parsed_start.timestamp() * 1000), snapshot_ms=snapshot)
        _ = settings_path.write_bytes(settings.model_dump_json(indent=2).encode())
    jobs: tuple[tuple[Kind, str, str], ...] = tuple(
        (kind, symbol, interval) for kind in ("market", "mark") for symbol in settings.symbols
        for interval in (settings.intervals if kind == "market" else settings.mark_intervals))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(partial(collect_group, output, settings), jobs))
    complete = all(item.complete for item in results)
    for kind in ("market", "mark"):
        selected = tuple(item for item in results if item.kind == kind)
        manifest = ArchiveManifest(source="Bitget public V3 recent candles; no authentication", endpoint=settings.source,
                                   candle_type=kind, created_at=datetime.now(UTC).isoformat(),
                                   start_ms=settings.start_ms, snapshot_ms=settings.snapshot_ms,
                                   symbols=settings.symbols, intervals=settings.intervals if kind == "market" else settings.mark_intervals,
                                   complete=all(item.complete for item in selected),
                                   partitions=tuple(item.partition for item in selected))
        _ = (output / kind / "MANIFEST.json").write_bytes(manifest.model_dump_json(indent=2).encode())
    report = RefreshReport(settings=settings, groups=results, complete=complete)
    report_path = output / "REFRESH_REPORT.json"
    _ = report_path.write_bytes(report.model_dump_json(indent=2).encode())
    LOGGER.info("MTF_REFRESH_COMPLETE groups=%s rows=%s coverage=%s sha256=%s", len(results),
                sum(item.partition.rows for item in results), complete, sha256(report_path.read_bytes()).hexdigest())
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    typer.run(main)
