from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Final, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .long_forecast import Candle
from .market_timeframes import CandleGranularity


BITGET_API_BASE: Final = "https://api.bitget.com"
PUBLIC_REQUEST_INTERVAL_SECONDS: Final = 0.07
REQUEST_TIMEOUT_SECONDS: Final = 8
UNDERLYING_ALIASES: Final = {"STXSTOCK": "STX"}
DEFAULT_TAKER_FEE_RATE: Final = 0.0006
LIVE_CANDLE_LIMIT: Final = 200
HISTORICAL_CANDLE_LIMIT: Final = 200


@dataclass(frozen=True, slots=True)
class BitgetContract:
    symbol: str
    underlying_symbol: str
    max_leverage: int
    funding_interval_hours: int
    taker_fee_rate: float = DEFAULT_TAKER_FEE_RATE


class BitgetError(Exception):
    """Base error for the public, read-only Bitget adapter."""


class BitgetNetworkError(BitgetError):
    def __init__(self, url: str, detail: str) -> None:
        self.url = url
        self.detail = detail
        super().__init__(f"Bitget public request failed for {url}: {detail}")


class BitgetPayloadError(BitgetError):
    def __init__(self, endpoint: str, detail: str) -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"Bitget payload was invalid for {endpoint}: {detail}")


class TextFetcher(Protocol):
    def __call__(self, url: str) -> str: ...


class BitgetPublicClient:
    """Mutable only to pace public requests below Bitget's published IP limit."""

    def __init__(self) -> None:
        self._last_request_at = 0.0
        self._request_lock = Lock()

    def get_text(self, url: str) -> str:
        with self._request_lock:
            requested_at = time.monotonic()
            wait_seconds = max(PUBLIC_REQUEST_INTERVAL_SECONDS - (requested_at - self._last_request_at), 0.0)
            self._last_request_at = requested_at + wait_seconds
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        request = Request(url, headers={"User-Agent": "StockPulseAI/bitget-stock-screen"})
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            raise BitgetNetworkError(url=url, detail=f"HTTP {error.code}") from error
        except URLError as error:
            raise BitgetNetworkError(url=url, detail=str(error.reason)) from error
        except TimeoutError as error:
            raise BitgetNetworkError(url=url, detail="timeout") from error
        return body


def _endpoint(path: str, **query: str) -> str:
    return f"{BITGET_API_BASE}{path}?{urlencode(query)}"


def _underlying_symbol(symbol: str) -> str:
    raw_symbol = symbol.removesuffix("USDT")
    if raw_symbol == symbol:
        raise BitgetPayloadError(endpoint="contracts", detail=f"symbol lacks USDT suffix: {symbol}")
    return UNDERLYING_ALIASES.get(raw_symbol, raw_symbol)


def _positive_integer(value: str, endpoint: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise BitgetPayloadError(endpoint=endpoint, detail=f"{field} was not an integer") from error
    if parsed <= 0:
        raise BitgetPayloadError(endpoint=endpoint, detail=f"{field} was not positive")
    return parsed


def _nonnegative_float(value: str, endpoint: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise BitgetPayloadError(endpoint=endpoint, detail=f"{field} was not numeric") from error
    if parsed < 0:
        raise BitgetPayloadError(endpoint=endpoint, detail=f"{field} was negative")
    return parsed


def list_rwa_contracts(fetch_text: TextFetcher) -> tuple[BitgetContract, ...]:
    """Fetch normal RWA contracts from Bitget's public USDT-futures registry."""

    endpoint = _endpoint("/api/v2/mix/market/contracts", productType="USDT-FUTURES")
    try:
        response = json.loads(fetch_text(endpoint))
    except json.JSONDecodeError as error:
        raise BitgetPayloadError(endpoint=endpoint, detail="response was not JSON") from error
    match response:  # noqa: MATCH_OK — untrusted Bitget JSON is exhaustively parsed into typed contracts here.
        case {"code": "00000", "data": list(rows)}:
            contracts: list[BitgetContract] = []
            for row in rows:
                match row:  # noqa: MATCH_OK — malformed external rows are deliberately skipped at this boundary.
                    case {
                        "symbol": str(symbol),
                        "symbolStatus": "normal",
                        "maxLever": str(raw_max_leverage),
                        "fundInterval": str(raw_funding_interval),
                        "isRwa": "YES",
                        "takerFeeRate": str(raw_taker_fee_rate),
                    }:
                        contracts.append(
                            BitgetContract(
                                symbol=symbol,
                                underlying_symbol=_underlying_symbol(symbol),
                                max_leverage=_positive_integer(raw_max_leverage, endpoint, "maxLever"),
                                funding_interval_hours=_positive_integer(raw_funding_interval, endpoint, "fundInterval"),
                                taker_fee_rate=_nonnegative_float(raw_taker_fee_rate, endpoint, "takerFeeRate"),
                            )
                        )
                    case {
                        "symbol": str(symbol),
                        "symbolStatus": "normal",
                        "maxLever": str(raw_max_leverage),
                        "fundInterval": str(raw_funding_interval),
                        "isRwa": "YES",
                    }:
                        contracts.append(
                            BitgetContract(
                                symbol=symbol,
                                underlying_symbol=_underlying_symbol(symbol),
                                max_leverage=_positive_integer(raw_max_leverage, endpoint, "maxLever"),
                                funding_interval_hours=_positive_integer(raw_funding_interval, endpoint, "fundInterval"),
                            )
                        )
                    case _:
                        continue
            return tuple(sorted(contracts, key=lambda contract: contract.symbol))
        case {"code": str(code), "msg": str(message)}:
            raise BitgetPayloadError(endpoint=endpoint, detail=f"API {code}: {message}")
        case _:
            raise BitgetPayloadError(endpoint=endpoint, detail="unexpected contract response shape")


def select_stock_contracts(
    contracts: tuple[BitgetContract, ...],
    local_us_symbols: frozenset[str],
) -> tuple[BitgetContract, ...]:
    """Intersect exchange RWA contracts with the locally tracked US stock/ETF universe."""

    return tuple(contract for contract in contracts if contract.underlying_symbol in local_us_symbols)


def _candle_from_row(row: list[str]) -> Candle | None:
    match row:  # noqa: MATCH_OK — external candle rows can be malformed and are rejected at this boundary.
        case [str(raw_opened_at), str(raw_open), str(raw_high), str(raw_low), str(raw_close), str(_), str(raw_quote_volume)]:
            try:
                candle = Candle(
                    opened_at=datetime.fromtimestamp(int(raw_opened_at) / 1000, tz=UTC),
                    open=float(raw_open),
                    high=float(raw_high),
                    low=float(raw_low),
                    close=float(raw_close),
                    quote_volume=float(raw_quote_volume),
                )
            except (OverflowError, OSError, ValueError) as error:
                return None
            if min(candle.open, candle.high, candle.low, candle.close) <= 0:
                return None
            return candle
        case _:
            return None


def fetch_completed_candles(
    contract: BitgetContract,
    granularity: CandleGranularity,
    fetch_text: TextFetcher,
    now: datetime | None = None,
) -> tuple[Candle, ...]:
    """Return only completed, ascending market candles for one requested interval."""

    endpoint = _endpoint(
        "/api/v2/mix/market/candles",
        symbol=contract.symbol,
        productType="USDT-FUTURES",
        granularity=granularity.value,
        limit=str(LIVE_CANDLE_LIMIT),
    )
    try:
        response = json.loads(fetch_text(endpoint))
    except json.JSONDecodeError as error:
        raise BitgetPayloadError(endpoint=endpoint, detail="response was not JSON") from error
    match response:  # noqa: MATCH_OK — untrusted Bitget JSON is parsed at the market-data boundary.
        case {"code": "00000", "data": list(rows)}:
            reference_time = now or datetime.now(UTC)
            completed = tuple(
                candle
                for row in rows
                if (candle := _candle_from_row(row)) is not None
                and candle.opened_at + granularity.interval <= reference_time
            )
            return tuple(sorted(completed, key=lambda candle: candle.opened_at))
        case {"code": str(code), "msg": str(message)}:
            raise BitgetPayloadError(endpoint=endpoint, detail=f"API {code}: {message}")
        case _:
            raise BitgetPayloadError(endpoint=endpoint, detail="unexpected candle response shape")


@dataclass(frozen=True, slots=True)
class HistoricalCandleQuery:
    """Bounded completed-candle query used only to settle already-recorded signals."""

    contract: BitgetContract
    granularity: CandleGranularity
    starts_at: datetime
    ends_at: datetime


def fetch_historical_completed_candles(
    query: HistoricalCandleQuery,
    fetch_text: TextFetcher,
) -> tuple[Candle, ...]:
    """Fetch completed historical candles strictly inside a recorded forecast window."""

    endpoint = _endpoint(
        "/api/v2/mix/market/history-candles",
        symbol=query.contract.symbol,
        productType="USDT-FUTURES",
        granularity=query.granularity.value,
        startTime=str(int(query.starts_at.timestamp() * 1000)),
        endTime=str(int(query.ends_at.timestamp() * 1000)),
        limit=str(HISTORICAL_CANDLE_LIMIT),
    )
    try:
        response = json.loads(fetch_text(endpoint))
    except json.JSONDecodeError as error:
        raise BitgetPayloadError(endpoint=endpoint, detail="response was not JSON") from error
    match response:  # noqa: MATCH_OK — untrusted Bitget JSON is parsed at the historical-data boundary.
        case {"code": "00000", "data": list(rows)}:
            completed = tuple(
                candle
                for row in rows
                if (candle := _candle_from_row(row)) is not None
                and query.starts_at <= candle.opened_at < query.ends_at
            )
            return tuple(sorted(completed, key=lambda candle: candle.opened_at))
        case {"code": str(code), "msg": str(message)}:
            raise BitgetPayloadError(endpoint=endpoint, detail=f"API {code}: {message}")
        case _:
            raise BitgetPayloadError(endpoint=endpoint, detail="unexpected historical candle response shape")


def fetch_completed_4h_candles(
    contract: BitgetContract,
    fetch_text: TextFetcher,
    now: datetime | None = None,
) -> tuple[Candle, ...]:
    """Compatibility wrapper for the legacy 4-hour research-only screen."""

    return fetch_completed_candles(contract, CandleGranularity.FOUR_HOURS, fetch_text, now)
