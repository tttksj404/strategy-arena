# Regression test for the spot-klines pagination truncation bug: fetch_klines() used to send
# limit=1500 for every market, but Binance spot klines (/api/v3/klines) actually caps a page at
# 1000 rows. A request for 1500 silently comes back with 1000 rows, len(page)=1000 < 1500 read
# as "no more data", and pagination stopped after one page -- even when far more history exists.
# fapi (perp) klines genuinely accept 1500, so that path must keep requesting 1500 unchanged.

from __future__ import annotations

import json
from typing import cast
import warnings

import requests

from research.wave1.common import JsonValue
from research.wave1.fetch_binance import BinanceKlineRequest, fetch_klines


DAY_MS = 86_400_000


def _kline_row(open_time: int) -> list[JsonValue]:
    # [openTime, open, high, low, close, volume, closeTime, quoteVolume] -- 8 fields, matching
    # KLINE_COLUMNS; enough for fetch_klines' trimming/typing logic.
    return [open_time, "1", "2", "0.5", "1.5", "10", open_time + DAY_MS - 1, "15"]


class _Response:
    def __init__(self, payload: JsonValue) -> None:
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None


class _PagedSession:
    """Records every request's params and serves one canned page per call."""

    def __init__(self, pages: list[list[list[JsonValue]]]) -> None:
        self.headers: dict[str, str] = {}
        self._pages = list(pages)
        self.requests: list[dict[str, str | int]] = []

    def get(self, _url: str, params: dict[str, str | int], **_kwargs: object) -> _Response:
        self.requests.append(dict(params))
        payload = self._pages.pop(0) if self._pages else []
        return _Response(payload)


def test_spot_market_requests_limit_1000_and_paginates_past_a_full_first_page() -> None:
    # Given: two pages of spot data -- a full 1000-row page (must trigger another request)
    # followed by a shorter 500-row page (must stop pagination there).
    page1 = [_kline_row(i * DAY_MS) for i in range(1000)]
    page2 = [_kline_row((1000 + i) * DAY_MS) for i in range(500)]
    session = _PagedSession([page1, page2])
    end_ms = 3000 * DAY_MS

    # When
    frame = fetch_klines(
        BinanceKlineRequest("BTCUSDT", "1d", 0, end_ms, "spot"),
        cast(requests.Session, session),
    )

    # Then: exactly two requests were made (a third would mean the short page didn't stop it,
    # or a missing third would mean it stopped after page 1 without noticing the full page).
    assert len(session.requests) == 2
    assert session.requests[0]["limit"] == 1000
    assert session.requests[1]["limit"] == 1000
    # The second request's startTime must resume right after the last row of page 1
    # (open_time of row 999, i.e. 999 * DAY_MS, plus 1ms).
    assert session.requests[1]["startTime"] == 999 * DAY_MS + 1
    # All 1500 rows across both pages made it into the result (regression: the old bug
    # returned only 1000).
    assert len(frame) == 1500


def test_fapi_market_still_requests_limit_1500() -> None:
    # Given: a single full 1500-row fapi page followed by a short page that ends pagination.
    page1 = [_kline_row(i * DAY_MS) for i in range(1500)]
    page2 = [_kline_row((1500 + i) * DAY_MS) for i in range(10)]
    session = _PagedSession([page1, page2])
    end_ms = 3000 * DAY_MS

    # When
    frame = fetch_klines(
        BinanceKlineRequest("BTCUSDT", "1d", 0, end_ms, "fapi"),
        cast(requests.Session, session),
    )

    # Then: fapi's real 1500 cap must be untouched by the spot fix.
    assert len(session.requests) == 2
    assert session.requests[0]["limit"] == 1500
    assert len(frame) == 1510


def test_spot_single_short_page_does_not_paginate_further() -> None:
    # Given: a spot response smaller than the 1000-row cap -- must be treated as complete.
    page1 = [_kline_row(i * DAY_MS) for i in range(200)]
    session = _PagedSession([page1])
    end_ms = 3000 * DAY_MS

    # When
    frame = fetch_klines(
        BinanceKlineRequest("BTCUSDT", "1d", 0, end_ms, "spot"),
        cast(requests.Session, session),
    )

    # Then
    assert len(session.requests) == 1
    assert len(frame) == 200


def test_string_millisecond_timestamp_is_numeric_before_datetime_conversion() -> None:
    # Mirrors test_fetch_bitget's compatibility guard: even a single-row spot page must
    # convert cleanly without pandas FutureWarnings.
    timestamp = 1_767_225_600_000
    session = _PagedSession([[_kline_row(timestamp)]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = fetch_klines(
            BinanceKlineRequest("BTCUSDT", "1d", timestamp, timestamp + DAY_MS, "spot"),
            cast(requests.Session, session),
        )

    assert len(frame) == 1
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]
