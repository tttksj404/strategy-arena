from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final
from urllib.parse import urlencode

from .bitget_market import BITGET_API_BASE, BitgetContract, BitgetPayloadError, TextFetcher


MAX_TICKER_AGE: Final = timedelta(seconds=15)
MAX_TICKER_FUTURE_SKEW: Final = timedelta(seconds=5)


@dataclass(frozen=True, slots=True)
class BitgetTicker:
    """A fresh public quote used only to decide whether a manual entry is still valid."""

    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    observed_at: datetime

    @property
    def bid_ask_spread_pct(self) -> float:
        """Return the executable ask-over-bid spread as a percent of the bid."""

        return (self.ask_price / self.bid_price - 1) * 100


def _positive_price(value: str, endpoint: str, field: str) -> float:
    try:
        price = float(value)
    except ValueError as error:
        raise BitgetPayloadError(endpoint=endpoint, detail=f"{field} was not numeric") from error
    if price <= 0:
        raise BitgetPayloadError(endpoint=endpoint, detail=f"{field} was not positive")
    return price


def _observed_at(value: str, endpoint: str) -> datetime:
    try:
        timestamp = int(value)
        observed_at = datetime.fromtimestamp(timestamp / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise BitgetPayloadError(endpoint=endpoint, detail="ts was not a valid millisecond timestamp") from error
    return observed_at


def _assert_current(quote: BitgetTicker, now: datetime) -> BitgetTicker:
    """Reject quotes that cannot safely represent a current manual entry decision."""

    if quote.observed_at < now - MAX_TICKER_AGE:
        raise BitgetPayloadError(endpoint="ticker", detail="ticker timestamp was stale")
    if quote.observed_at > now + MAX_TICKER_FUTURE_SKEW:
        raise BitgetPayloadError(endpoint="ticker", detail="ticker timestamp was implausibly in the future")
    return quote


def fetch_live_ticker(
    contract: BitgetContract,
    fetch_text: TextFetcher,
    now: datetime | None = None,
) -> BitgetTicker:
    """Fetch one fresh public bid/ask quote for a normal USDT-futures contract."""

    endpoint = f"{BITGET_API_BASE}/api/v2/mix/market/ticker?{urlencode({'symbol': contract.symbol, 'productType': 'USDT-FUTURES'})}"
    try:
        response = json.loads(fetch_text(endpoint))
    except json.JSONDecodeError as error:
        raise BitgetPayloadError(endpoint=endpoint, detail="response was not JSON") from error
    match response:  # noqa: MATCH_OK — external ticker data is parsed once at this boundary.
        case {
            "code": "00000",
            "data": [
                {
                    "symbol": str(symbol),
                    "lastPr": str(raw_last_price),
                    "bidPr": str(raw_bid_price),
                    "askPr": str(raw_ask_price),
                    "ts": str(raw_timestamp),
                },
                *_,
            ],
        }:
            if symbol != contract.symbol:
                raise BitgetPayloadError(endpoint=endpoint, detail="ticker symbol did not match requested contract")
            quote = BitgetTicker(
                symbol=symbol,
                last_price=_positive_price(raw_last_price, endpoint, "lastPr"),
                bid_price=_positive_price(raw_bid_price, endpoint, "bidPr"),
                ask_price=_positive_price(raw_ask_price, endpoint, "askPr"),
                observed_at=_observed_at(raw_timestamp, endpoint),
            )
            return _assert_current(quote, now or datetime.now(UTC))
        case {"code": str(code), "msg": str(message)}:
            raise BitgetPayloadError(endpoint=endpoint, detail=f"API {code}: {message}")
        case _:
            raise BitgetPayloadError(endpoint=endpoint, detail="unexpected ticker response shape")
