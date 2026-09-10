"""Public Bitget funding-history parsing and closed-bar alignment for the paper study."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from alt_search_models import MarketData
from bitget_mtf_core import Candle


PACING_SECONDS = 0.055


@dataclass(frozen=True, slots=True)
class FundingStudyError(Exception):
    """A public-data condition that prevents a trustworthy study result."""

    route: str
    detail: str

    def __str__(self) -> str:
        return f"{self.route}: {self.detail}"


@dataclass(frozen=True, slots=True)
class FundingFetchRequest:
    """One public history request fixed to a contract in the study universe."""

    symbol: str


@dataclass(frozen=True, slots=True)
class TimeframeMarket:
    """Closed, aligned Bitget bars with their precise open timestamps."""

    timestamps_ms: tuple[int, ...]
    market: MarketData


class FundingEvent(BaseModel):
    """A parsed Bitget funding settlement at the external-data boundary."""

    model_config = ConfigDict(frozen=True)
    funding_time_ms: int = Field(alias="fundingTime")
    funding_rate: float = Field(alias="fundingRate")


class FundingEnvelope(BaseModel):
    """The public Bitget funding-history response boundary."""

    model_config = ConfigDict(frozen=True)
    code: str
    msg: str
    data: tuple[FundingEvent, ...]


def funding_history(client: httpx2.Client, request: FundingFetchRequest) -> tuple[FundingEvent, ...]:
    """Page public funding settlements until Bitget exhausts its retained history."""

    events: list[FundingEvent] = []
    seen: set[int] = set()
    for page_number in range(1, 101):
        try:
            response = client.get(
                "/api/v2/mix/market/history-fund-rate",
                params={"symbol": request.symbol, "productType": "USDT-FUTURES", "pageSize": "100", "pageNo": str(page_number)},
            )
            response.raise_for_status()
            envelope = FundingEnvelope.model_validate_json(response.text)
        except (httpx2.HTTPError, ValidationError) as error:
            raise FundingStudyError(request.symbol, str(error)) from error
        if envelope.code != "00000":
            raise FundingStudyError(request.symbol, envelope.msg)
        if not envelope.data:
            return tuple(sorted(events, key=lambda event: event.funding_time_ms))
        for event in envelope.data:
            if event.funding_time_ms not in seen:
                seen.add(event.funding_time_ms)
                events.append(event)
        if len(envelope.data) < 100:
            return tuple(sorted(events, key=lambda event: event.funding_time_ms))
        time.sleep(PACING_SECONDS)
    raise FundingStudyError(request.symbol, "funding pagination exceeded 100 pages")


def timeframe_market(candles: dict[str, tuple[Candle, ...]]) -> TimeframeMarket:
    """Intersect each symbol's closed candles into one chronological price matrix."""

    timestamp_sets = tuple({candle.timestamp_ms for candle in rows} for rows in candles.values())
    common = set.intersection(*timestamp_sets)
    timestamps = tuple(sorted(common))
    if not timestamps:
        raise FundingStudyError("candles", "no common closed bars")
    by_symbol = {symbol: {candle.timestamp_ms: candle for candle in rows} for symbol, rows in candles.items()}
    symbols = tuple(candles)
    values = tuple(tuple(by_symbol[symbol][timestamp] for symbol in symbols) for timestamp in timestamps)
    market = MarketData(
        tuple(datetime.fromtimestamp(timestamp / 1000.0, UTC) for timestamp in timestamps),
        symbols,
        np.array([[candle.open_price for candle in row] for row in values], dtype=float),
        np.array([[candle.high_price for candle in row] for row in values], dtype=float),
        np.array([[candle.low_price for candle in row] for row in values], dtype=float),
        np.array([[candle.close_price for candle in row] for row in values], dtype=float),
        np.array([[candle.quote_volume_usdt for candle in row] for row in values], dtype=float),
    )
    return TimeframeMarket(timestamps, market)


def trim_to_funding_coverage(timeframe: TimeframeMarket, histories: dict[str, tuple[FundingEvent, ...]]) -> TimeframeMarket:
    """Keep only bars after every fixed-universe contract has a funding observation."""

    earliest: list[int] = []
    for symbol in timeframe.market.symbols:
        events = histories[symbol]
        if not events:
            raise FundingStudyError(symbol, "no retained funding observations")
        earliest.append(events[0].funding_time_ms)
    first_usable = max(earliest)
    start = next((index for index, timestamp in enumerate(timeframe.timestamps_ms) if timestamp > first_usable), len(timeframe.timestamps_ms))
    if len(timeframe.timestamps_ms) - start < 24:
        raise FundingStudyError("coverage", "fewer than 24 aligned bars after funding coverage starts")
    market = timeframe.market
    return TimeframeMarket(
        timeframe.timestamps_ms[start:],
        MarketData(
            market.dates[start:], market.symbols, market.open_price[start:], market.high_price[start:],
            market.low_price[start:], market.close_price[start:], market.quote_volume[start:],
        ),
    )


def funding_matrix(timeframe: TimeframeMarket, histories: dict[str, tuple[FundingEvent, ...]]) -> np.ndarray:
    """Apply each settled rate to the next bar so no same-bar signal sees it early."""

    timestamps = np.asarray(timeframe.timestamps_ms, dtype=np.int64)
    rates = np.zeros_like(timeframe.market.close_price)
    for column, symbol in enumerate(timeframe.market.symbols):
        for event in histories[symbol]:
            row = int(np.searchsorted(timestamps, event.funding_time_ms, side="right"))
            if row < len(timestamps):
                rates[row, column] += event.funding_rate
    return rates
