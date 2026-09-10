from __future__ import annotations

import json
from pathlib import Path

import bitget_dynamic_archive_panel as panel
import bitget_history_data as history


def _candle(timestamp_ms: int, close_price: float, quote_volume: float = 200.0) -> history.CandleRow:
    return history.CandleRow.from_api_row(
        [str(timestamp_ms), str(close_price), str(close_price + 1.0), str(close_price - 1.0), str(close_price), "2", str(quote_volume)]
    )


def test_dynamic_loader_delays_new_listing_eligibility_until_its_own_history_exists(tmp_path: Path) -> None:
    duration = history.interval_duration_ms("4H")
    window = history.HistoryWindow(0, 2 * duration)
    btc = history.write_partition(
        root=tmp_path,
        symbol="BTCUSDT",
        interval="4H",
        window=window,
        rows=(_candle(0, 100.0), _candle(duration, 101.0), _candle(2 * duration, 102.0)),
    )
    eth = history.write_partition(
        root=tmp_path,
        symbol="ETHUSDT",
        interval="4H",
        window=window,
        rows=(_candle(0, 50.0), _candle(duration, 51.0), _candle(2 * duration, 52.0)),
    )
    sol = history.write_partition(
        root=tmp_path,
        symbol="SOLUSDT",
        interval="4H",
        window=window,
        rows=(_candle(duration, 25.0), _candle(2 * duration, 26.0)),
    )
    manifest = {
        "complete": True,
        "snapshot_ms": 3 * duration + 1,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "intervals": ["4H"],
        "partitions": [
            {"relative_path": entry.relative_path, "sha256": entry.sha256, "rows": entry.rows, "window_start_ms": entry.window_start_ms, "window_end_ms": entry.window_end_ms}
            for entry in (btc, eth, sol)
        ],
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    data, metadata = panel.load_verified_dynamic_interval_archive(
        tmp_path,
        ("BTC", "ETH", "SOL"),
        interval="4H",
        minimum_listing_bars=1,
    )

    sol_index = data.symbols.index("SOL")
    assert data.close_price[:, sol_index].tolist() == [25.0, 25.0, 26.0]
    assert data.quote_volume[:, sol_index].tolist() == [0.0, 0.0, 200.0]
    assert metadata.reference_rows == 3
    assert metadata.integrity_verified is True
