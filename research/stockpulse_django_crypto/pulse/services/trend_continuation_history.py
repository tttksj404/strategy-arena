from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from .bitget_market import BitgetContract, BitgetError, TextFetcher
from .long_forecast import Candle
from .market_timeframes import CandleGranularity
from .trend_continuation_contract import PublicMarketRequest, stable_manifest_digest
from .trend_continuation_manifest import JsonObject, JsonValue
from .trend_continuation_provenance import hash_raw_body, public_bitget_url, require_v2_public_market_request


MAX_HISTORY_PAGES: Final = 720
HISTORY_PAGE_SIZE: Final = 200


@dataclass(frozen=True, slots=True)
class HistoricalCollectionError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class HistoricalCollectionQuery:
    contract: BitgetContract
    granularity: CandleGranularity
    starts_at: datetime
    ends_at: datetime
    max_pages: int


@dataclass(frozen=True, slots=True)
class HistoricalPageProvenance:
    page_number: int
    request: PublicMarketRequest
    raw_sha256: str
    row_count: int
    first_open: datetime | None
    last_open: datetime | None

    def payload(self) -> JsonObject:
        return {
            "page_number": self.page_number,
            "query": [[key, value] for key, value in self.request.query],
            "raw_sha256": self.raw_sha256,
            "row_count": self.row_count,
            "first_open": self.first_open.isoformat() if self.first_open else None,
            "last_open": self.last_open.isoformat() if self.last_open else None,
        }


@dataclass(frozen=True, slots=True)
class HistoricalCandleResult:
    granularity: CandleGranularity
    candles: tuple[Candle, ...]
    pages: tuple[HistoricalPageProvenance, ...]
    complete: bool
    continuous: bool
    rejection_code: str | None
    terminal_reason: str
    aggregate_raw_sha256: str
    first_open: datetime | None
    last_open: datetime | None
    continuity_gaps: tuple[datetime, ...]

    def payload(self) -> JsonObject:
        return {
            "granularity": self.granularity.value,
            "complete": self.complete,
            "continuous": self.continuous,
            "rejection_code": self.rejection_code,
            "terminal_reason": self.terminal_reason,
            "aggregate_raw_sha256": self.aggregate_raw_sha256,
            "page_count": len(self.pages),
            "row_count": len(self.candles),
            "first_open": self.first_open.isoformat() if self.first_open else None,
            "last_open": self.last_open.isoformat() if self.last_open else None,
            "continuity_gaps": [gap.isoformat() for gap in self.continuity_gaps],
            "pages": [page.payload() for page in self.pages],
        }


def required_profile_intervals(profile: str) -> tuple[CandleGranularity, ...]:
    if profile == "standard_15m_5m":
        return (CandleGranularity.FIFTEEN_MINUTES, CandleGranularity.FIVE_MINUTES)
    if profile == "triple_leveraged_15m_5m_1m":
        return (
            CandleGranularity.FIFTEEN_MINUTES,
            CandleGranularity.FIVE_MINUTES,
            CandleGranularity.ONE_MINUTE,
        )
    raise HistoricalCollectionError(f"unsupported historical profile: {profile}")


def collect_historical_candles(
    query: HistoricalCollectionQuery,
    fetch_text: TextFetcher,
) -> HistoricalCandleResult:
    _validate_query(query)
    cursor = query.ends_at
    pages: list[HistoricalPageProvenance] = []
    collected: list[Candle] = []
    rejection: str | None = None
    terminal = "page_limit_exhausted"
    for page_number in range(1, query.max_pages + 1):
        request = _page_request(query, cursor)
        try:
            raw_body = fetch_text(public_bitget_url(request.path, request.query))
        except BitgetError:
            rejection = "history_unavailable"
            terminal = "source_unavailable"
            break
        raw_sha256 = hash_raw_body(raw_body)
        parsed = _parse_page(raw_body)
        pages.append(_page_provenance(page_number, request, raw_sha256, parsed.candles))
        if parsed.rejection_code is not None:
            rejection = parsed.rejection_code
            terminal = "malformed_page"
            break
        if not parsed.candles:
            terminal = "empty_page"
            break
        collected.extend(
            candle for candle in parsed.candles if query.starts_at <= candle.opened_at < query.ends_at
        )
        earliest = parsed.candles[0].opened_at
        if earliest <= query.starts_at:
            terminal = "start_reached"
            break
        if earliest >= cursor:
            rejection = "history_cursor_stalled"
            terminal = "cursor_stalled"
            break
        cursor = earliest
    expected = _expected_opens(query)
    actual_times = tuple(candle.opened_at for candle in collected)
    duplicates = len(frozenset(actual_times)) != len(actual_times)
    unique_by_open = {candle.opened_at: candle for candle in collected}
    ordered = tuple(unique_by_open[opened_at] for opened_at in sorted(unique_by_open))
    gaps = tuple(opened_at for opened_at in expected if opened_at not in unique_by_open)
    if rejection is None and terminal == "page_limit_exhausted":
        rejection = "history_page_limit"
    if rejection is None and duplicates:
        rejection = "duplicate_candle"
    if rejection is None and gaps:
        rejection = "missing_candle"
    terminal_complete = terminal in {"start_reached", "empty_page"}
    complete = rejection is None and terminal_complete and len(ordered) == len(expected)
    continuous = complete and not gaps
    return HistoricalCandleResult(
        granularity=query.granularity,
        candles=ordered if complete else (),
        pages=tuple(pages),
        complete=complete,
        continuous=continuous,
        rejection_code=rejection,
        terminal_reason=terminal,
        aggregate_raw_sha256=stable_manifest_digest([page.payload() for page in pages]),
        first_open=ordered[0].opened_at if ordered else None,
        last_open=ordered[-1].opened_at if ordered else None,
        continuity_gaps=gaps,
    )


@dataclass(frozen=True, slots=True)
class ParsedCandlePage:
    candles: tuple[Candle, ...]
    rejection_code: str | None


def _validate_query(query: HistoricalCollectionQuery) -> None:
    if not 1 <= query.max_pages <= MAX_HISTORY_PAGES:
        raise HistoricalCollectionError("history max_pages must be between 1 and 720")
    if query.starts_at.tzinfo is None or query.ends_at.tzinfo is None:
        raise HistoricalCollectionError("historical range must be timezone-aware")
    if query.starts_at >= query.ends_at:
        raise HistoricalCollectionError("historical range start must precede end")
    interval_seconds = int(query.granularity.interval.total_seconds())
    if int(query.starts_at.timestamp()) % interval_seconds or int(query.ends_at.timestamp()) % interval_seconds:
        raise HistoricalCollectionError("historical range must align to candle boundaries")


def _page_request(query: HistoricalCollectionQuery, cursor: datetime) -> PublicMarketRequest:
    pairs = (
        ("symbol", query.contract.symbol),
        ("productType", "USDT-FUTURES"),
        ("granularity", query.granularity.value),
        ("startTime", str(int(query.starts_at.timestamp() * 1000))),
        ("endTime", str(int(cursor.timestamp() * 1000))),
        ("limit", str(HISTORY_PAGE_SIZE)),
    )
    return require_v2_public_market_request("GET", "/api/v2/mix/market/history-candles", pairs)


def _parse_page(raw_body: str) -> ParsedCandlePage:
    try:
        response: JsonValue = json.loads(raw_body)
    except json.JSONDecodeError:
        return ParsedCandlePage((), "malformed_json")
    match response:  # noqa: MATCH_OK - untrusted page JSON is converted at this boundary.
        case {"code": "00000", "data": list(rows)}:
            parsed = tuple(candle for row in rows if (candle := _candle_from_row(row)) is not None)
            if len(parsed) != len(rows):
                return ParsedCandlePage((), "malformed_candle")
            return ParsedCandlePage(tuple(sorted(parsed, key=lambda candle: candle.opened_at)), None)
        case _:
            return ParsedCandlePage((), "blocked_or_malformed_source")


def _candle_from_row(row: JsonValue) -> Candle | None:
    match row:  # noqa: MATCH_OK - malformed provider rows are rejected.
        case [str(raw_time), str(raw_open), str(raw_high), str(raw_low), str(raw_close), str(_), str(raw_volume)]:
            try:
                candle = Candle(
                    opened_at=datetime.fromtimestamp(int(raw_time) / 1000, tz=UTC),
                    open=float(raw_open),
                    high=float(raw_high),
                    low=float(raw_low),
                    close=float(raw_close),
                    quote_volume=float(raw_volume),
                )
            except (OverflowError, OSError, ValueError):
                return None
            return candle if min(candle.open, candle.high, candle.low, candle.close) > 0 else None
        case _:
            return None


def _page_provenance(
    page_number: int,
    request: PublicMarketRequest,
    raw_sha256: str,
    candles: tuple[Candle, ...],
) -> HistoricalPageProvenance:
    return HistoricalPageProvenance(
        page_number=page_number,
        request=request,
        raw_sha256=raw_sha256,
        row_count=len(candles),
        first_open=candles[0].opened_at if candles else None,
        last_open=candles[-1].opened_at if candles else None,
    )


def _expected_opens(query: HistoricalCollectionQuery) -> tuple[datetime, ...]:
    expected: list[datetime] = []
    cursor = query.starts_at
    while cursor < query.ends_at:
        expected.append(cursor)
        cursor += query.granularity.interval
    return tuple(expected)
