"""Public-only collection boundaries, including incomplete candles and gaps."""

import json
from pathlib import Path

import pytest

import refresh_bitget_mtf as refresh
from bitget_history_data import CandleRow, HistoryDataError
from bitget_history_manifest import ManifestEntry
from refresh_bitget_mtf import GroupResult, Settings, closed_rows, complete_coverage


def test_pagination_excludes_open_candles_and_moves_backwards() -> None:
    requests: list[dict[str, str]] = []

    def request(parameters: dict[str, str]) -> str:
        requests.append(parameters)
        times = (120_000, 180_000, 240_000) if len(requests) == 1 else (0, 60_000)
        return json.dumps({"code": "00000", "msg": "ok", "data":
                           [[str(t), "10", "11", "9", "10", "1", "10"] for t in times]})

    rows = closed_rows(request, symbol="SOLUSDT", interval="1m", kind="market",
                       start=0, snapshot=250_000)
    assert [row.timestamp_ms for row in rows] == [0, 60_000, 120_000, 180_000]
    assert requests[1]["endTime"] == "60000"
    assert all(set(item) == {"category", "symbol", "interval", "type", "limit", "startTime", "endTime"}
               for item in requests)


def test_future_page_and_repeated_pagination_fail_closed() -> None:
    def request(_: dict[str, str]) -> str:
        return json.dumps({"code": "00000", "msg": "ok", "data":
                           [["999999", "10", "11", "9", "10", "1", "10"]]})

    with pytest.raises(HistoryDataError):
        closed_rows(request, symbol="SOLUSDT", interval="1m", kind="market", start=0, snapshot=250_000)


def test_coverage_checks_edges_as_well_as_internal_gaps() -> None:
    row = CandleRow.from_api_row(("60000", "10", "11", "9", "10", "1", "10"))
    assert not complete_coverage((row,), start=0, snapshot=180_000, duration=60_000)
    assert complete_coverage((row,), start=60_000, snapshot=120_000, duration=60_000)


def test_one_row_page_at_the_requested_cursor_is_valid() -> None:
    def request(parameters: dict[str, str]) -> str:
        stamp = min(int(parameters["endTime"]), 120_000)
        return json.dumps({"code": "00000", "msg": "ok", "data":
                           [[str(stamp), "10", "11", "9", "10", "1", "10"]]})

    rows = closed_rows(request, symbol="SOLUSDT", interval="1m", kind="mark", start=0, snapshot=180_000)
    assert len(rows) == 3


def test_last_page_never_sends_equal_start_and_end_to_bitget() -> None:
    def request(parameters: dict[str, str]) -> str:
        assert int(parameters["startTime"]) < int(parameters["endTime"])
        stamp = min(int(parameters["endTime"]), 120_000)
        return json.dumps({"code": "00000", "msg": "ok", "data":
                           [[str(stamp), "10", "11", "9", "10", "1", "10"]]})

    rows = closed_rows(request, symbol="SOLUSDT", interval="1m", kind="market", start=60_000, snapshot=180_000)
    assert len(rows) == 2


@pytest.mark.parametrize("duration", [-1, 0, 30 * 86_400_000 + 1])
def test_resumed_settings_reject_invalid_ranges_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: int,
) -> None:
    _ = (tmp_path / "SETTINGS.json").write_text(
        json.dumps({"start_ms": 0, "snapshot_ms": duration}), encoding="utf-8",
    )

    def forbidden_collection(*_: object) -> None:
        pytest.fail("Invalid saved settings reached the collection boundary")

    monkeypatch.setattr(refresh, "collect_group", forbidden_collection)
    with pytest.raises(ValueError, match="at most 30 days"):
        refresh.main(output=tmp_path)


@pytest.mark.parametrize("audit_field", ["hash_failures", "missing_intervals", "duplicate_rows", "invalid_ohlc_rows"])
def test_per_kind_manifest_and_top_report_share_all_audit_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audit_field: str,
) -> None:
    settings = Settings(start_ms=0, snapshot_ms=60_000, symbols=("SOLUSDT",),
                        intervals=("1m",), mark_intervals=("1m",))
    _ = (tmp_path / "SETTINGS.json").write_bytes(settings.model_dump_json().encode())

    def collect(root: Path, _: Settings, job: tuple[refresh.Kind, str, str]) -> GroupResult:
        kind, symbol, interval = job
        (root / kind).mkdir()
        audit: dict[str, int | str | None] = dict.fromkeys(
            ("hash_failures", "missing_intervals", "duplicate_rows", "invalid_ohlc_rows"), 0,
        )
        audit[audit_field] = 1
        entry = ManifestEntry(relative_path="fixture.jsonl.gz", sha256="0" * 64, rows=1,
                              first_timestamp_ms=0, last_timestamp_ms=0, window_start_ms=0, window_end_ms=0)
        return GroupResult(kind=kind, symbol=symbol, interval=interval, partition=entry,
                           audit=audit, boundary_complete=True)

    monkeypatch.setattr(refresh, "collect_group", collect)
    with pytest.raises(SystemExit, match="2"):
        refresh.main(output=tmp_path)
    for kind in ("market", "mark"):
        assert json.loads((tmp_path / kind / "MANIFEST.json").read_bytes())["complete"] is False
    assert json.loads((tmp_path / "REFRESH_REPORT.json").read_bytes())["complete"] is False
