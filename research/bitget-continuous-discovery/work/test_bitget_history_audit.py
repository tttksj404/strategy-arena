from __future__ import annotations

from pathlib import Path

import bitget_history_audit as audit
import bitget_history_data as history


def _row(timestamp_ms: int, *, invalid: bool = False) -> history.CandleRow:
    values = [str(timestamp_ms), "100", "101", "99", "100.5", "2", "201"]
    if invalid:
        values[3] = "101"
    return history.CandleRow.from_api_row(values)


def test_audit_group_detects_missing_and_invalid_rows(tmp_path: Path) -> None:
    first = history.write_partition(
        root=tmp_path,
        symbol="ETHUSDT",
        interval="1m",
        window=history.HistoryWindow(0, 59_999),
        rows=(_row(0),),
    )
    second = history.write_partition(
        root=tmp_path,
        symbol="ETHUSDT",
        interval="1m",
        window=history.HistoryWindow(60_000, 179_999),
        rows=(_row(120_000, invalid=True),),
    )

    report = audit.audit_group(root=tmp_path, symbol="ETHUSDT", interval="1m", entries=(first, second))

    assert report.raw_rows == 2
    assert report.unique_rows == 2
    assert report.missing_intervals == 1
    assert report.invalid_ohlc_rows == 1
    assert report.hash_failures == 0
    assert report.first_timestamp_ms == 0
    assert report.last_timestamp_ms == 120_000
