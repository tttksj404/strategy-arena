from __future__ import annotations

import json
from pathlib import Path

import bitget_archive_daily as archive
import bitget_history_data as history


def _candle(timestamp_ms: int, close_price: float) -> history.CandleRow:
    return history.CandleRow.from_api_row(
        [str(timestamp_ms), str(close_price), str(close_price + 1.0), str(close_price - 1.0), str(close_price), "2", "200"]
    )


def test_loader_requires_hash_valid_common_daily_history(tmp_path: Path) -> None:
    window = history.HistoryWindow(0, 172_800_000)
    eth = history.write_partition(
        root=tmp_path,
        symbol="ETHUSDT",
        interval="1D",
        window=window,
        rows=tuple(_candle(timestamp, close) for timestamp, close in ((0, 100.0), (86_400_000, 101.0), (172_800_000, 102.0))),
    )
    sol = history.write_partition(
        root=tmp_path,
        symbol="SOLUSDT",
        interval="1D",
        window=window,
        rows=tuple(_candle(timestamp, close) for timestamp, close in ((0, 50.0), (86_400_000, 51.0), (172_800_000, 52.0))),
    )
    manifest = {
        "complete": True,
        "snapshot_ms": 200_000_000,
        "symbols": ["ETHUSDT", "SOLUSDT"],
        "intervals": ["1D"],
        "partitions": [
            {"relative_path": eth.relative_path, "sha256": eth.sha256, "rows": eth.rows, "window_start_ms": eth.window_start_ms, "window_end_ms": eth.window_end_ms},
            {"relative_path": sol.relative_path, "sha256": sol.sha256, "rows": sol.rows, "window_start_ms": sol.window_start_ms, "window_end_ms": sol.window_end_ms},
        ],
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    data, metadata = archive.load_verified_daily_archive(tmp_path, ("ETH", "SOL"))

    assert data.symbols == ("ETH", "SOL")
    assert data.close_price.shape == (3, 2)
    assert metadata.common_rows == 3
    assert metadata.integrity_verified is True
