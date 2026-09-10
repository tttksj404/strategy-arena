from __future__ import annotations

from pathlib import Path

import pytest

import bitget_history_data as history
import fetch_bitget_history as collector
import resume_bitget_history as supervisor


def _config(tmp_path: Path) -> collector.CollectionConfig:
    return collector.CollectionConfig(
        root=tmp_path,
        start_ms=0,
        snapshot_ms=120_000,
        symbols=("ETHUSDT",),
        intervals=("1m",),
        active_intervals=None,
        max_partitions=1,
    )


def _manifest() -> collector.ArchiveManifest:
    return collector.ArchiveManifest(
        created_at="2026-08-25T00:00:00+00:00",
        start_ms=0,
        snapshot_ms=120_000,
        symbols=("ETHUSDT",),
        intervals=("1m",),
        complete=True,
        partitions=(),
    )


def test_collector_supervisor_retries_request_failure_with_a_fresh_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0
    sleeps: list[float] = []
    expected = _manifest()

    def fake_collect(_config: collector.CollectionConfig) -> collector.ArchiveManifest:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise collector.BitgetHistoryRequestError("temporary DNS failure")
        return expected

    monkeypatch.setattr(collector, "collect", fake_collect)
    monkeypatch.setattr(supervisor.time, "sleep", sleeps.append)

    result = supervisor.collect_with_restarts(_config(tmp_path), restart_attempts=2)

    assert result == expected
    assert calls == 2
    assert sleeps == [supervisor.restart_delay_seconds(0)]


def test_collector_supervisor_fails_closed_for_invalid_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0

    def fake_collect(_config: collector.CollectionConfig) -> collector.ArchiveManifest:
        nonlocal calls
        calls += 1
        raise history.HistoryDataError("invalid history payload")

    monkeypatch.setattr(collector, "collect", fake_collect)

    with pytest.raises(history.HistoryDataError, match="invalid history payload"):
        supervisor.collect_with_restarts(_config(tmp_path), restart_attempts=2)

    assert calls == 1
