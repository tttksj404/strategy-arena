from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Self

import pytest

import bitget_history_data as history
import fetch_bitget_history as runner

MINUTE_MS = 60_000
type HistoryRequest = Callable[[str, dict[str, str]], str]


def _payload(rows: list[list[str]]) -> str:
    return json.dumps({"code": "00000", "msg": "success", "data": rows})


def _row(timestamp_ms: int) -> list[str]:
    return [str(timestamp_ms), "1", "1.1", "0.9", "1.05", "2", "2.1"]


def test_parse_candle_page_rejects_non_increasing_timestamps() -> None:
    payload = _payload([_row(MINUTE_MS), _row(MINUTE_MS)])

    with pytest.raises(history.HistoryDataError, match="strictly increasing"):
        history.parse_candle_page(payload, route="history-candles")


def test_parse_candle_page_preserves_but_flags_invalid_exchange_ohlc() -> None:
    payload = _payload([["0", "100", "101", "101", "100.5", "2", "201"]])

    rows = history.parse_candle_page(payload, route="history-candles")

    assert len(rows) == 1
    assert rows[0].ohlc_valid is False


def test_fetch_window_pages_backwards_without_gaps() -> None:
    calls: list[int] = []
    pages = {
        419_999: [_row(240_000), _row(300_000)],
        300_000: [_row(120_000), _row(180_000)],
        180_000: [_row(0), _row(60_000)],
    }

    def request(_route: str, params: dict[str, str]) -> str:
        cursor = int(params["endTime"])
        calls.append(cursor)
        return _payload(pages[cursor])

    rows = history.fetch_window(
        request=request,
        symbol="ETHUSDT",
        interval="1m",
        start_ms=0,
        end_ms=359_999,
    )

    assert calls == [419_999, 300_000, 180_000]
    assert [row.timestamp_ms for row in rows] == [0, 60_000, 120_000, 180_000, 240_000, 300_000]


def test_fetch_window_retains_one_exchange_boundary_candle_before_start() -> None:
    def request(_route: str, _params: dict[str, str]) -> str:
        return _payload([_row(0), _row(60_000), _row(120_000)])

    rows = history.fetch_window(
        request=request,
        symbol="ETHUSDT",
        interval="1m",
        start_ms=60_000,
        end_ms=179_999,
    )

    assert [row.timestamp_ms for row in rows] == [0, 60_000, 120_000]


def test_partition_windows_cover_every_millisecond_once() -> None:
    windows = history.partition_windows(start_ms=0, end_ms=299, span_ms=100)

    assert windows == (
        history.HistoryWindow(start_ms=0, end_ms=99),
        history.HistoryWindow(start_ms=100, end_ms=199),
        history.HistoryWindow(start_ms=200, end_ms=299),
    )


def test_candle_windows_are_aligned_and_never_exceed_the_api_range() -> None:
    windows = history.candle_windows(
        start_ms=0,
        end_candle_ms=480_000,
        interval="1m",
        max_query_range_ms=180_000,
    )

    assert windows == (
        history.HistoryWindow(start_ms=0, end_ms=120_000),
        history.HistoryWindow(start_ms=180_000, end_ms=300_000),
        history.HistoryWindow(start_ms=360_000, end_ms=480_000),
    )


def test_partition_round_trip_is_hashed_and_resumable(tmp_path: Path) -> None:
    window = history.HistoryWindow(start_ms=0, end_ms=119_999)
    rows = tuple(history.CandleRow.from_api_row(_row(timestamp)) for timestamp in (0, 60_000))

    entry = history.write_partition(root=tmp_path, symbol="ETHUSDT", interval="1m", window=window, rows=rows)

    assert history.valid_partition(root=tmp_path, entry=entry)
    stored = tmp_path / entry.relative_path
    with gzip.open(stored, mode="rt", encoding="utf-8") as handle:
        restored = [json.loads(line) for line in handle]
    assert [record["timestamp_ms"] for record in restored] == [0, 60_000]


def test_partition_records_internal_missing_candles(tmp_path: Path) -> None:
    window = history.HistoryWindow(start_ms=0, end_ms=179_999)
    rows = tuple(history.CandleRow.from_api_row(_row(timestamp)) for timestamp in (0, 120_000))

    entry = history.write_partition(root=tmp_path, symbol="ETHUSDT", interval="1m", window=window, rows=rows)

    assert entry.missing_intervals == 1
    assert history.valid_partition(root=tmp_path, entry=entry)


def test_manifest_entry_preserves_a_slotted_archive_entry(tmp_path: Path) -> None:
    window = history.HistoryWindow(start_ms=0, end_ms=59_999)
    entry = history.write_partition(
        root=tmp_path,
        symbol="ETHUSDT",
        interval="1m",
        window=window,
        rows=(history.CandleRow.from_api_row(_row(0)),),
    )

    restored = runner.ManifestEntry.from_entry(entry).archive_entry()

    assert restored == entry


def test_parse_server_time_uses_the_exchange_timestamp_not_the_local_clock() -> None:
    payload = json.dumps({"code": "00000", "msg": "success", "data": {"serverTime": "1704067200000"}})

    assert runner.parse_server_time(payload) == 1_704_067_200_000


def test_get_text_retries_a_transient_transport_error_without_accepting_bad_data(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        text = '{"code":"00000"}'

        def raise_for_status(self) -> None:
            return None

    class Client:
        calls = 0

        def get(self, _route: str, *, params: dict[str, str]) -> Response:
            self.calls += 1
            if self.calls < 3:
                raise runner.httpx2.ConnectError("temporary connection loss")
            return Response()

    sleeps: list[float] = []
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    client = Client()

    assert runner.get_text(
        client,
        "/api/v3/market/history-candles",
        {"symbol": "ETHUSDT"},
        pacing_seconds=0.5,
    ) == '{"code":"00000"}'
    assert client.calls == 3
    assert sleeps == [1.0, 2.0, 0.5]


def test_parse_config_normalizes_symbols_but_preserves_interval_case() -> None:
    config = runner.parse_config(
        [
            "--symbols", "ethusdt",
            "--intervals", "1m,1H",
            "--start-ms", "0",
            "--snapshot-ms", "1704067200000",
            "--max-partitions", "1",
        ]
    )

    assert config.symbols == ("ETHUSDT",)
    assert config.intervals == ("1m", "1H")


def test_parse_config_keeps_archive_scope_when_a_subset_is_prioritized() -> None:
    config = runner.parse_config(
        [
            "--intervals", "1m,5m,1H",
            "--active-intervals", "1H,5m",
            "--start-ms", "0",
            "--snapshot-ms", "1704067200000",
        ]
    )

    assert config.intervals == ("1m", "5m", "1H")
    assert config.active_intervals == ("1H", "5m")


def test_parse_config_records_mark_candle_type() -> None:
    config = runner.parse_config(
        [
            "--intervals", "1m,1H",
            "--candle-type", "mark",
            "--start-ms", "0",
            "--snapshot-ms", "1704067200000",
        ]
    )

    assert config.candle_type == "mark"


def test_parse_config_accepts_a_conservative_request_pacing() -> None:
    config = runner.parse_config(
        [
            "--intervals", "1H",
            "--start-ms", "0",
            "--snapshot-ms", "1704067200000",
            "--pacing-seconds", "0.5",
        ]
    )

    assert config.pacing_seconds == 0.5


def test_parse_config_rejects_unsafe_request_pacing() -> None:
    with pytest.raises(history.HistoryDataError, match="pacing"):
        runner.parse_config(
            [
                "--intervals", "1H",
                "--start-ms", "0",
                "--snapshot-ms", "1704067200000",
                "--pacing-seconds", "0.01",
            ]
        )


def test_parse_config_rejects_a_priority_interval_outside_the_archive_scope() -> None:
    with pytest.raises(history.HistoryDataError, match="subset"):
        runner.parse_config(
            [
                "--intervals", "1m,5m",
                "--active-intervals", "1H",
                "--start-ms", "0",
                "--snapshot-ms", "1704067200000",
            ]
        )


def test_declared_scope_stays_incomplete_when_only_a_priority_interval_exists(tmp_path: Path) -> None:
    entry = history.write_partition(
        root=tmp_path,
        symbol="ETHUSDT",
        interval="1H",
        window=history.HistoryWindow(0, 3_599_999),
        rows=(history.CandleRow.from_api_row(_row(0)),),
    )
    manifest = runner.ArchiveManifest(
        created_at="2026-08-24T00:00:00+00:00",
        start_ms=0,
        snapshot_ms=7_200_000,
        symbols=("ETHUSDT",),
        intervals=("1m", "1H"),
        complete=False,
        partitions=(runner.ManifestEntry.from_entry(entry),),
    )

    assert runner._scope_is_complete(tmp_path, manifest, {entry.relative_path: runner.ManifestEntry.from_entry(entry)}) is False


def test_collect_adds_requested_candle_type_to_history_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested_params: list[dict[str, str]] = []

    class Client:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, _kind: type[BaseException] | None, _value: BaseException | None, _traceback: TracebackType | None) -> None:
            return None

    def fake_get_text(
        _client: Client,
        _route: str,
        params: dict[str, str],
        *,
        pacing_seconds: float = runner.PACING_SECONDS,
    ) -> str:
        assert pacing_seconds == runner.PACING_SECONDS
        requested_params.append(params)
        return "{}"

    def fake_fetch_window(
        *,
        request: HistoryRequest,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        max_request_end_ms: int,
    ) -> tuple[history.CandleRow, ...]:
        request(history.HISTORY_ROUTE, {"symbol": symbol, "interval": interval})
        return (history.CandleRow.from_api_row(_row(start_ms)),)

    monkeypatch.setattr(runner, "create_client", Client)
    monkeypatch.setattr(runner, "get_text", fake_get_text)
    monkeypatch.setattr(history, "fetch_window", fake_fetch_window)
    config = runner.CollectionConfig(
        root=tmp_path,
        start_ms=0,
        snapshot_ms=120_000,
        symbols=("ETHUSDT",),
        intervals=("1m",),
        active_intervals=None,
        max_partitions=1,
        candle_type="mark",
    )

    manifest = runner.collect(config)

    assert manifest.candle_type == "mark"
    assert requested_params == [{"symbol": "ETHUSDT", "interval": "1m", "type": "mark"}]


def test_fetch_window_fails_closed_when_the_api_repeats_an_out_of_window_row() -> None:
    def request(_route: str, _params: dict[str, str]) -> str:
        return _payload([_row(300_000)])

    with pytest.raises(history.HistoryDataError, match="boundary allowance"):
        history.fetch_window(
            request=request,
            symbol="ETHUSDT",
            interval="1m",
            start_ms=0,
            end_ms=120_000,
        )


def test_fetch_window_caps_the_last_request_at_the_exchange_snapshot() -> None:
    calls: list[int] = []

    def request(_route: str, params: dict[str, str]) -> str:
        calls.append(int(params["endTime"]))
        return _payload([_row(300_000)])

    rows = history.fetch_window(
        request=request,
        symbol="ETHUSDT",
        interval="1m",
        start_ms=300_000,
        end_ms=359_999,
        max_request_end_ms=360_000,
    )

    assert calls == [360_000]
    assert [row.timestamp_ms for row in rows] == [300_000]
