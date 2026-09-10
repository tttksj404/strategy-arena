from __future__ import annotations

import json

import bitget_mtf_paper_experiment as experiment
import httpx2


DAY_MS = 86_400_000


def test_long_timeframe_uses_multiple_90_day_requests_when_a_page_is_not_full(monkeypatch) -> None:
    now_ms = 200 * DAY_MS
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(experiment, "HISTORY_DAYS_BY_INTERVAL", {"3D": 180}, raising=False)

    def fake_get_text(_client, _route: str, params: dict[str, str]) -> str:
        calls.append(dict(params))
        offset = len(calls)
        rows = [
            [now_ms - (3 * offset + 2) * DAY_MS, "1", "1.1", "0.9", "1", "10", "20"],
            [now_ms - (3 * offset + 1) * DAY_MS, "1", "1.1", "0.9", "1", "10", "20"],
        ]
        return json.dumps({"code": "00000", "msg": "success", "data": rows})

    monkeypatch.setattr(experiment, "_get_text", fake_get_text)

    candles = experiment.fetch_candles(client=None, symbol="SOLUSDT", interval="3D", now_ms=now_ms)

    assert len(calls) == 2
    assert min(int(call["startTime"]) for call in calls) == now_ms - 180 * DAY_MS
    assert all(int(call["endTime"]) - int(call["startTime"]) <= 90 * DAY_MS for call in calls)
    assert len(candles) == 3


def test_get_text_when_rate_limited_then_backs_off_once_and_retries(monkeypatch) -> None:
    request = httpx2.Request("GET", "https://api.bitget.com/test")
    responses = [httpx2.Response(429, request=request), httpx2.Response(200, content=b"{}", request=request)]
    pauses: list[float] = []

    class Client:
        def get(self, _route: str, params: dict[str, str]) -> httpx2.Response:
            assert params == {"one": "1"}
            return responses.pop(0)

    monkeypatch.setattr(experiment.time, "sleep", pauses.append)

    payload = experiment._get_text(Client(), "/test", {"one": "1"})

    assert payload == "{}"
    assert pauses == [experiment.RATE_LIMIT_BACKOFF_SECONDS, experiment.PACING_SECONDS]
