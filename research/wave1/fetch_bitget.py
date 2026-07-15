# Bitget market-data and Yahoo baseline fetchers with local caching.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import JsonValue, PipelineError, request_json, validate_symbol


BITGET_BASE: Final = "https://api.bitget.com"
YAHOO_BASE: Final = "https://query1.finance.yahoo.com"


@dataclass(frozen=True, slots=True)
class BitgetCandleRequest:
    symbol: str
    granularity: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class YahooDailyRequest:
    symbol: str
    start_seconds: int
    end_seconds: int


def _data(payload: JsonValue) -> JsonValue:
    if not isinstance(payload, dict) or "data" not in payload:
        raise PipelineError("Bitget response is missing data")
    code = payload.get("code")
    if code is not None and code != "00000":
        raise PipelineError(f"Bitget error code: {code}")
    return payload["data"]


def fetch_candles(request: BitgetCandleRequest, session: requests.Session | None = None) -> pd.DataFrame:
    validate_symbol(request.symbol)
    owned_session = session is None
    client = session or requests.Session()
    cursor = request.end_ms
    collected: list[list[JsonValue]] = []
    try:
        current_params: dict[str, str | int] = {
            "symbol": request.symbol,
            "productType": "usdt-futures",
            "granularity": request.granularity,
            "limit": 500,
        }
        current_raw = _data(request_json(client, BITGET_BASE + "/api/v2/mix/market/candles", current_params))
        if not isinstance(current_raw, list):
            raise PipelineError("Bitget candles data must be a list")
        current_page = [row for row in current_raw if isinstance(row, list)]
        collected.extend(current_page)
        if current_page:
            cursor = min(cursor, min(int(row[0]) for row in current_page) - 1)
        while cursor >= request.start_ms:
            params: dict[str, str | int] = {
                "symbol": request.symbol,
                "productType": "usdt-futures",
                "granularity": request.granularity,
                "limit": 200,
                "endTime": cursor,
            }
            raw = _data(request_json(client, BITGET_BASE + "/api/v2/mix/market/history-candles", params))
            if not isinstance(raw, list):
                raise PipelineError("Bitget candles data must be a list")
            page = [row for row in raw if isinstance(row, list)]
            if not page:
                break
            collected.extend(page)
            earliest = min(int(row[0]) for row in page)
            next_cursor = earliest - 1
            if next_cursor >= cursor or earliest <= request.start_ms:
                break
            cursor = next_cursor
    finally:
        if owned_session:
            client.close()
    columns = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    frame = pd.DataFrame([row[:7] for row in collected if len(row) >= 7], columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=columns[1:], index=pd.DatetimeIndex([], name="timestamp"))
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    frame[columns[1:]] = frame[columns[1:]].apply(pd.to_numeric, errors="coerce")
    result = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index().loc[lambda item: ~item.index.duplicated()]
    start = pd.to_datetime(request.start_ms, unit="ms", utc=True)
    end = pd.to_datetime(request.end_ms, unit="ms", utc=True)
    return result[(result.index >= start) & (result.index < end)]


def fetch_funding(symbol: str, session: requests.Session | None = None) -> pd.DataFrame:
    validate_symbol(symbol)
    owned_session = session is None
    client = session or requests.Session()
    collected: list[dict[str, JsonValue]] = []
    previous_oldest: int | None = None
    try:
        for page_number in range(1, 101):
            params: dict[str, str | int] = {
                "symbol": symbol,
                "productType": "usdt-futures",
                "pageSize": 100,
                "pageNo": page_number,
            }
            raw = _data(request_json(client, BITGET_BASE + "/api/v2/mix/market/history-fund-rate", params))
            if not isinstance(raw, list):
                raise PipelineError("Bitget funding data must be a list")
            page = [row for row in raw if isinstance(row, dict)]
            if not page:
                break
            funding_times = [int(row["fundingTime"]) for row in page if "fundingTime" in row]
            if not funding_times:
                raise PipelineError("Bitget funding page has no timestamps")
            oldest = min(funding_times)
            if previous_oldest is not None and oldest >= previous_oldest:
                raise PipelineError("Bitget funding pagination did not advance")
            previous_oldest = oldest
            collected.extend(page)
        else:
            raise PipelineError("Bitget funding pagination exceeded 100 pages")
    finally:
        if owned_session:
            client.close()
    frame = pd.DataFrame(collected)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "funding_rate"])
    frame = frame.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate"})
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="coerce")
    return frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index().loc[lambda item: ~item.index.duplicated()]


def fetch_contracts(session: requests.Session) -> JsonValue:
    params: dict[str, str | int] = {"productType": "usdt-futures"}
    return _data(request_json(session, BITGET_BASE + "/api/v2/mix/market/contracts", params))


def contract_symbols(payload: JsonValue) -> set[str]:
    if not isinstance(payload, list):
        raise PipelineError("Bitget contracts must be a list")
    return {
        item["symbol"]
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("symbol"), str)
    }


def fetch_yahoo_daily(request: YahooDailyRequest, session: requests.Session | None = None) -> pd.DataFrame:
    validate_symbol(request.symbol)
    owned_session = session is None
    client = session or requests.Session()
    params: dict[str, str | int] = {
        "period1": request.start_seconds,
        "period2": request.end_seconds,
        "interval": "1d",
        "events": "history",
    }
    try:
        payload = request_json(client, YAHOO_BASE + f"/v8/finance/chart/{request.symbol}", params)
    finally:
        if owned_session:
            client.close()
    if not isinstance(payload, dict):
        raise PipelineError("Yahoo response must be an object")
    chart = payload.get("chart")
    if not isinstance(chart, dict) or not isinstance(chart.get("result"), list) or not chart["result"]:
        raise PipelineError("Yahoo response has no result")
    result = chart["result"][0]
    if not isinstance(result, dict):
        raise PipelineError("Yahoo result must be an object")
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise PipelineError("Yahoo result is missing timestamps or indicators")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or not quotes or not isinstance(quotes[0], dict):
        raise PipelineError("Yahoo result is missing quote data")
    quote = quotes[0]
    frame = pd.DataFrame({"timestamp": timestamps, **quote})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame.set_index("timestamp").sort_index()
