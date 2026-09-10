#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx2[http2,brotli,zstd]", "pydantic>=2"]
# ///

"""Download the predeclared Bitget futures history into verified public-only files."""

from __future__ import annotations

import argparse
import os
import socket
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError

import bitget_history_data as history
from bitget_history_manifest import ArchiveManifest, CollectionConfig, ManifestEntry

API_ROOT: Final = "https://api.bitget.com"
DAY_MS: Final = 86_400_000
WINDOW_MS: Final = 90 * DAY_MS
PACING_SECONDS: Final = 0.065
RETRIES: Final = 3
DEFAULT_START_MS: Final = int(datetime(2018, 1, 1, tzinfo=UTC).timestamp() * 1000)
DEFAULT_SYMBOLS: Final = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT",
    "LTCUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT",
)
DEFAULT_INTERVALS: Final = tuple(history.INTERVAL_MS)
CANDLE_TYPES: Final = ("market", "mark")
type CandleType = Literal["market", "mark"]


class BitgetHistoryRequestError(RuntimeError):
    """A read-only public endpoint failure that invalidates the collection."""


class ServerTimeData(BaseModel):
    model_config = ConfigDict(frozen=True)
    serverTime: str


class ServerTimeEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: str
    msg: str
    data: ServerTimeData


def create_client() -> httpx2.Client:
    limits = httpx2.Limits(max_connections=8, max_keepalive_connections=4, keepalive_expiry=30.0)
    timeout = httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=2,
        limits=limits,
        socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
    )
    return httpx2.Client(base_url=API_ROOT, transport=transport, timeout=timeout, follow_redirects=True)


def get_text(
    client: httpx2.Client,
    route: str,
    params: dict[str, str],
    *,
    pacing_seconds: float = PACING_SECONDS,
) -> str:
    for attempt in range(RETRIES + 1):
        try:
            response = client.get(route, params=params)
            response.raise_for_status()
        except httpx2.HTTPStatusError as error:
            if error.response.status_code == 429 and attempt < RETRIES:
                time.sleep(2.0 * 2**attempt)
                continue
            raise BitgetHistoryRequestError(f"{route}: HTTP {error.response.status_code}") from error
        except httpx2.HTTPError as error:
            if attempt < RETRIES:
                time.sleep(1.0 * 2**attempt)
                continue
            raise BitgetHistoryRequestError(f"{route}: {error}") from error
        time.sleep(pacing_seconds)
        return response.text
    raise AssertionError("bounded request retry loop must return or raise")


def parse_server_time(raw_text: str) -> int:
    try:
        payload = ServerTimeEnvelope.model_validate_json(raw_text)
        server_time_ms = int(payload.data.serverTime)
    except (ValidationError, ValueError) as error:
        raise history.HistoryDataError("Bitget server time payload is malformed") from error
    if payload.code != "00000" or server_time_ms < DEFAULT_START_MS:
        raise history.HistoryDataError(f"Bitget server time is invalid: {payload.msg}")
    return server_time_ms


def fetch_server_time(client: httpx2.Client, *, pacing_seconds: float = PACING_SECONDS) -> int:
    return parse_server_time(get_text(client, "/api/v2/public/time", {}, pacing_seconds=pacing_seconds))


def _csv_values(value: str, *, label: str, uppercase: bool) -> tuple[str, ...]:
    values = tuple(item.strip().upper() if uppercase else item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise history.HistoryDataError(f"{label} must not be empty")
    return values


def parse_config(argv: Sequence[str] | None = None) -> CollectionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("bitget_history_archive")))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--intervals", default=",".join(DEFAULT_INTERVALS))
    parser.add_argument("--candle-type", choices=CANDLE_TYPES, default="market")
    parser.add_argument("--active-intervals", default=None, help="comma-separated subset to collect now; preserves the archive's full interval scope")
    parser.add_argument("--start-ms", type=int, default=DEFAULT_START_MS)
    parser.add_argument("--snapshot-ms", type=int, default=None)
    parser.add_argument("--max-partitions", type=int, default=None)
    parser.add_argument("--pacing-seconds", type=float, default=PACING_SECONDS)
    arguments = parser.parse_args(argv)
    symbols = _csv_values(arguments.symbols, label="symbols", uppercase=True)
    intervals = _csv_values(arguments.intervals, label="intervals", uppercase=False)
    for interval in intervals:
        history.interval_duration_ms(interval)
    active_intervals = None
    if arguments.active_intervals is not None:
        active_intervals = _csv_values(arguments.active_intervals, label="active intervals", uppercase=False)
        for interval in active_intervals:
            history.interval_duration_ms(interval)
        if not set(active_intervals).issubset(intervals):
            raise history.HistoryDataError("active intervals must be a subset of the archive interval scope")
    snapshot_ms = arguments.snapshot_ms
    if arguments.start_ms < 0 or (snapshot_ms is not None and snapshot_ms <= arguments.start_ms):
        raise history.HistoryDataError("snapshot must be after a non-negative start")
    if arguments.max_partitions is not None and arguments.max_partitions <= 0:
        raise history.HistoryDataError("max-partitions must be positive")
    if not 0.05 <= arguments.pacing_seconds <= 60.0:
        raise history.HistoryDataError("request pacing must be between 0.05 and 60 seconds")
    return CollectionConfig(
        root=Path(arguments.output_dir),
        start_ms=arguments.start_ms,
        snapshot_ms=snapshot_ms,
        symbols=symbols,
        intervals=intervals,
        candle_type=arguments.candle_type,
        active_intervals=active_intervals,
        max_partitions=arguments.max_partitions,
        pacing_seconds=arguments.pacing_seconds,
    )


def _manifest_path(root: Path) -> Path:
    return root / "MANIFEST.json"


def load_manifest(config: CollectionConfig) -> ArchiveManifest:
    path = _manifest_path(config.root)
    if not path.exists():
        return ArchiveManifest(
            created_at=datetime.now(UTC).isoformat(),
            start_ms=config.start_ms,
            snapshot_ms=config.snapshot_ms,
            symbols=config.symbols,
            intervals=config.intervals,
            candle_type=config.candle_type,
            complete=False,
            partitions=(),
        )
    try:
        manifest = ArchiveManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise history.HistoryDataError("existing manifest is unreadable") from error
    if (
        manifest.start_ms != config.start_ms
        or manifest.snapshot_ms != config.snapshot_ms
        or manifest.symbols != config.symbols
        or manifest.intervals != config.intervals
        or manifest.candle_type != config.candle_type
    ):
        raise history.HistoryDataError("existing manifest scope differs; choose a separate output directory")
    return manifest


def save_manifest(*, root: Path, manifest: ArchiveManifest) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = _manifest_path(root)
    partial = target.with_suffix(".json.partial")
    partial.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(partial, target)


def _updated_manifest(manifest: ArchiveManifest, entries: dict[str, ManifestEntry], *, complete: bool) -> ArchiveManifest:
    return manifest.model_copy(update={"complete": complete, "partitions": tuple(entries[path] for path in sorted(entries))})


def _scope_is_complete(root: Path, manifest: ArchiveManifest, entries: dict[str, ManifestEntry]) -> bool:
    """Verify every partition in the declared archive scope before marking it complete."""
    for interval in manifest.intervals:
        end_candle_ms = history.last_closed_candle_start_ms(manifest.snapshot_ms, interval)
        for symbol in manifest.symbols:
            for window in history.candle_windows(
                start_ms=manifest.start_ms,
                end_candle_ms=end_candle_ms,
                interval=interval,
                max_query_range_ms=WINDOW_MS,
            ):
                relative_path = history.archive_relative_path(symbol=symbol, interval=interval, window=window).as_posix()
                existing = entries.get(relative_path)
                if existing is None or not history.valid_partition(root=root, entry=existing.archive_entry()):
                    return False
    return True


def collect(config: CollectionConfig) -> ArchiveManifest:
    with create_client() as client:
        snapshot_ms = (
            config.snapshot_ms
            if config.snapshot_ms is not None
            else fetch_server_time(client, pacing_seconds=config.pacing_seconds)
        )
        if snapshot_ms <= config.start_ms:
            raise history.HistoryDataError("exchange snapshot must be after the requested start")
        resolved_config = config.model_copy(update={"snapshot_ms": snapshot_ms})
        manifest = load_manifest(resolved_config)
        entries = {entry.relative_path: entry for entry in manifest.partitions}
        completed_this_run = 0
        active_intervals = resolved_config.active_intervals or resolved_config.intervals
        def request(route: str, params: dict[str, str]) -> str:
            return get_text(
                client,
                route,
                {**params, "type": resolved_config.candle_type},
                pacing_seconds=resolved_config.pacing_seconds,
            )

        for interval in active_intervals:
            end_candle_ms = history.last_closed_candle_start_ms(manifest.snapshot_ms, interval)
            for symbol in resolved_config.symbols:
                for window in history.candle_windows(
                    start_ms=manifest.start_ms,
                    end_candle_ms=end_candle_ms,
                    interval=interval,
                    max_query_range_ms=WINDOW_MS,
                ):
                    relative_path = history.archive_relative_path(symbol=symbol, interval=interval, window=window).as_posix()
                    existing = entries.get(relative_path)
                    if existing is not None and history.valid_partition(root=resolved_config.root, entry=existing.archive_entry()):
                        continue
                    rows = history.fetch_window(
                        request=request,
                        symbol=symbol,
                        interval=interval,
                        start_ms=window.start_ms,
                        end_ms=window.end_ms,
                        max_request_end_ms=manifest.snapshot_ms,
                    )
                    entry = history.write_partition(root=resolved_config.root, symbol=symbol, interval=interval, window=window, rows=rows)
                    entries[entry.relative_path] = ManifestEntry.from_entry(entry)
                    manifest = _updated_manifest(manifest, entries, complete=False)
                    save_manifest(root=resolved_config.root, manifest=manifest)
                    completed_this_run += 1
                    print(
                        f"PARTITION {symbol} {interval} {window.start_ms}:{window.end_ms} "
                        f"rows={entry.rows} invalid_ohlc={entry.invalid_ohlc_rows}",
                        flush=True,
                    )
                    if resolved_config.max_partitions is not None and completed_this_run >= resolved_config.max_partitions:
                        return manifest
    manifest = _updated_manifest(manifest, entries, complete=_scope_is_complete(resolved_config.root, manifest, entries))
    save_manifest(root=resolved_config.root, manifest=manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_config(argv)
        manifest = collect(config)
    except (BitgetHistoryRequestError, history.HistoryDataError) as error:
        print(f"COLLECTION_ERROR={error}")
        return 1
    print(f"MANIFEST={_manifest_path(config.root)}")
    print(f"COMPLETE={str(manifest.complete).lower()} PARTITIONS={len(manifest.partitions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
