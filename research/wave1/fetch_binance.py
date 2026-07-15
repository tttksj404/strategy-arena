# Binance price, funding, contract, volume, and cache fetchers.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Final

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import JsonValue, PipelineError, load_frame, request_json, save_frame, validate_symbol


FAPI_BASE: Final = "https://fapi.binance.com"
SPOT_BASE: Final = "https://api.binance.com"
KLINE_COLUMNS: Final = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
)


@dataclass(frozen=True, slots=True)
class BinanceKlineRequest:
    symbol: str
    interval: str
    start_ms: int
    end_ms: int | None = None
    market: str = "fapi"


@dataclass(frozen=True, slots=True)
class BinanceFundingRequest:
    symbol: str
    start_ms: int
    end_ms: int | None = None


def _rows(payload: JsonValue) -> list[list[JsonValue]]:
    if not isinstance(payload, list):
        raise PipelineError("Binance response must be a list")
    return [row for row in payload if isinstance(row, list)]


def _records(payload: JsonValue) -> list[dict[str, JsonValue]]:
    if not isinstance(payload, list):
        raise PipelineError("Binance response must be a list")
    return [row for row in payload if isinstance(row, dict)]


def fetch_klines(request: BinanceKlineRequest, session: requests.Session | None = None) -> pd.DataFrame:
    validate_symbol(request.symbol)
    owned_session = session is None
    client = session or requests.Session()
    endpoint = "/fapi/v1/klines" if request.market == "fapi" else "/api/v3/klines"
    base = FAPI_BASE if request.market == "fapi" else SPOT_BASE
    cursor = request.start_ms
    collected: list[list[JsonValue]] = []
    try:
        while request.end_ms is None or cursor <= request.end_ms:
            params: dict[str, str | int] = {
                "symbol": request.symbol,
                "interval": request.interval,
                "limit": 1500,
                "startTime": cursor,
            }
            if request.end_ms is not None:
                params["endTime"] = request.end_ms
            page = _rows(request_json(client, base + endpoint, params))
            if not page:
                break
            collected.extend(page)
            next_cursor = int(page[-1][0]) + 1
            if next_cursor <= cursor or len(page) < 1500:
                break
            cursor = next_cursor
    finally:
        if owned_session:
            client.close()
    trimmed = [row[: len(KLINE_COLUMNS)] for row in collected if len(row) >= len(KLINE_COLUMNS)]
    frame = pd.DataFrame(trimmed, columns=KLINE_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS[1:], index=pd.DatetimeIndex([], name="timestamp"))
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    result = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index().loc[lambda item: ~item.index.duplicated()]
    if request.end_ms is not None:
        result = result[result.index < pd.to_datetime(request.end_ms, unit="ms", utc=True)]
    return result


def fetch_funding(request: BinanceFundingRequest, session: requests.Session | None = None) -> pd.DataFrame:
    validate_symbol(request.symbol)
    owned_session = session is None
    client = session or requests.Session()
    cursor = request.start_ms
    collected: list[dict[str, JsonValue]] = []
    try:
        while request.end_ms is None or cursor <= request.end_ms:
            params: dict[str, str | int] = {
                "symbol": request.symbol,
                "startTime": cursor,
                "limit": 1000,
            }
            if request.end_ms is not None:
                params["endTime"] = request.end_ms
            page = _records(request_json(client, FAPI_BASE + "/fapi/v1/fundingRate", params))
            if not page:
                break
            collected.extend(page)
            next_cursor = int(page[-1]["fundingTime"]) + 1
            if next_cursor <= cursor or len(page) < 1000:
                break
            cursor = next_cursor
    finally:
        if owned_session:
            client.close()
    frame = pd.DataFrame(collected)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "funding_rate", "mark_price"])
    frame = frame.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate", "markPrice": "mark_price"})
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    frame[["funding_rate", "mark_price"]] = frame[["funding_rate", "mark_price"]].apply(pd.to_numeric, errors="coerce")
    result = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index().loc[lambda item: ~item.index.duplicated()]
    if request.end_ms is not None:
        result = result[result.index < pd.to_datetime(request.end_ms, unit="ms", utc=True)]
    return result


def fetch_exchange_info(session: requests.Session) -> JsonValue:
    return request_json(session, FAPI_BASE + "/fapi/v1/exchangeInfo", {})


def fetch_spot_exchange_info(session: requests.Session) -> JsonValue:
    # /api/v3/exchangeInfo exceeds the 16 MB response guard; ticker/price is a light proxy for listed spot symbols.
    payload = request_json(session, SPOT_BASE + "/api/v3/ticker/price", {})
    if not isinstance(payload, list):
        raise PipelineError("spot ticker response must be a list")
    return {"symbols": [{"symbol": item.get("symbol"), "status": "TRADING"} for item in payload if isinstance(item, dict)]}


def fetch_quote_volumes(session: requests.Session) -> JsonValue:
    return request_json(session, FAPI_BASE + "/fapi/v1/ticker/24hr", {})


def exchange_symbols(payload: JsonValue) -> set[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise PipelineError("exchange info is missing symbols")
    symbols: set[str] = set()
    for item in payload["symbols"]:
        if isinstance(item, dict) and item.get("status") == "TRADING" and isinstance(item.get("symbol"), str):
            symbols.add(item["symbol"])
    return symbols


def quote_volumes(payload: JsonValue) -> dict[str, float]:
    if not isinstance(payload, list):
        raise PipelineError("ticker response must be a list")
    volumes: dict[str, float] = {}
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("symbol"), str):
            volumes[item["symbol"]] = float(item.get("quoteVolume", 0.0))
    return volumes


def cached_frame(path: Path, force: bool, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if path.exists() and not force:
        return load_frame(path)
    frame = loader()
    save_frame(path, frame)
    return frame
