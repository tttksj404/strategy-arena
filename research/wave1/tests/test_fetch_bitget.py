# Verifies Bitget string millisecond timestamps parse without compatibility warnings.

from __future__ import annotations

import json
from typing import cast
import warnings

import requests

from research.wave1.common import JsonValue
from research.wave1.fetch_bitget import BitgetCandleRequest, fetch_candles


class _Response:
    def __init__(self, payload: JsonValue) -> None:
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, payload: JsonValue) -> None:
        self.headers: dict[str, str] = {}
        self.payload = payload

    def get(self, _url: str, **_kwargs: str | int | float | dict[str, str | int] | tuple[float, float]) -> _Response:
        return _Response(self.payload)


def test_string_millisecond_timestamp_is_numeric_before_datetime_conversion() -> None:
    # Given
    timestamp = 1_767_225_600_000
    payload = {"code": "00000", "data": [[str(timestamp), "1", "2", "0.5", "1.5", "10", "15"]]}
    session = _Session(payload)

    # When
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = fetch_candles(BitgetCandleRequest("BTCUSDT", "1H", timestamp, timestamp + 3_600_000), cast(requests.Session, session))

    # Then
    assert len(frame) == 1
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]
