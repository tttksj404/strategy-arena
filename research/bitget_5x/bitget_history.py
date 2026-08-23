from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

from .bitget_market import (
    BitgetContract,
    BitgetPayloadError,
    HistoricalCandleQuery,
    TextFetcher,
    fetch_historical_completed_candles,
)
from .long_forecast import Candle
from .market_timeframes import CandleGranularity


MAX_HISTORICAL_PAGES: Final = 720


@dataclass(frozen=True, slots=True)
class HistoricalRangeQuery:
    """A bounded public Bitget candle window that may require several 200-row pages."""

    contract: BitgetContract
    granularity: CandleGranularity
    starts_at: datetime
    ends_at: datetime


def fetch_historical_candle_range(query: HistoricalRangeQuery, fetch_text: TextFetcher) -> tuple[Candle, ...]:
    """Page backwards through one historical range and return unique completed candles in ascending order."""

    if query.starts_at >= query.ends_at:
        raise BitgetPayloadError(endpoint="history-candles", detail="range start must precede range end")
    cursor = query.ends_at
    candles_by_opened_at: dict[datetime, Candle] = {}
    for _ in range(MAX_HISTORICAL_PAGES):
        page = fetch_historical_completed_candles(
            HistoricalCandleQuery(
                contract=query.contract,
                granularity=query.granularity,
                starts_at=query.starts_at,
                ends_at=cursor,
            ),
            fetch_text,
        )
        if not page:
            break
        for candle in page:
            if query.starts_at <= candle.opened_at < query.ends_at:
                candles_by_opened_at[candle.opened_at] = candle
        earliest = page[0].opened_at
        if earliest <= query.starts_at:
            break
        if earliest >= cursor:
            raise BitgetPayloadError(endpoint="history-candles", detail="history page did not move backwards")
        cursor = earliest
    else:
        raise BitgetPayloadError(endpoint="history-candles", detail="history page limit exhausted")
    return tuple(candles_by_opened_at[opened_at] for opened_at in sorted(candles_by_opened_at))
