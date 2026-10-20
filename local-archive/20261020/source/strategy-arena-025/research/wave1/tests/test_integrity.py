# Verifies integrity checks expose unsorted and duplicate timestamps.

from __future__ import annotations

from datetime import timedelta

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import integrity_report, normalize_market_frame


def test_normalization_does_not_hide_timestamp_defects() -> None:
    # Given
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "close": [2.0, 1.0, 1.1],
            "volume": [1.0, 1.0, 1.0],
        }
    )

    # When
    report = integrity_report(normalize_market_frame(frame), timedelta(days=1))

    # Then
    assert report.monotonic is False
    assert report.duplicate_count == 1
    assert report.valid is False
