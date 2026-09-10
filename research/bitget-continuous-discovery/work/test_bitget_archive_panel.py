from __future__ import annotations

import json
from pathlib import Path

import bitget_archive_panel as panel
import bitget_history_data as history


def _candle(timestamp_ms: int, close_price: float) -> history.CandleRow:
    return history.CandleRow.from_api_row(
        [str(timestamp_ms), str(close_price), str(close_price + 1.0), str(close_price - 1.0), str(close_price), "2", "200"]
    )


def test_loader_accepts_a_complete_verified_interval_while_another_interval_is_pending(tmp_path: Path) -> None:
    duration = history.interval_duration_ms("4H")
    window = history.HistoryWindow(0, duration)
    eth = history.write_partition(
        root=tmp_path,
        symbol="ETHUSDT",
        interval="4H",
        window=window,
        rows=(_candle(0, 100.0), _candle(duration, 101.0)),
    )
    sol = history.write_partition(
        root=tmp_path,
        symbol="SOLUSDT",
        interval="4H",
        window=window,
        rows=(_candle(0, 50.0), _candle(duration, 51.0)),
    )
    manifest = {
        "complete": False,
        "snapshot_ms": 2 * duration + 1,
        "symbols": ["ETHUSDT", "SOLUSDT"],
        "intervals": ["1m", "4H"],
        "partitions": [
            {"relative_path": eth.relative_path, "sha256": eth.sha256, "rows": eth.rows, "window_start_ms": eth.window_start_ms, "window_end_ms": eth.window_end_ms},
            {"relative_path": sol.relative_path, "sha256": sol.sha256, "rows": sol.rows, "window_start_ms": sol.window_start_ms, "window_end_ms": sol.window_end_ms},
        ],
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    data, metadata = panel.load_verified_interval_archive(tmp_path, ("ETH", "SOL"), interval="4H")

    assert data.symbols == ("ETH", "SOL")
    assert data.close_price.shape == (2, 2)
    assert metadata.common_rows == 2
    assert metadata.integrity_verified is True
