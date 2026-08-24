from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from .bitget_market import BitgetContract, TextFetcher
from .bitget_ticker import BitgetTicker
from .long_forecast import Candle
from .market_timeframes import CandleGranularity
from .trend_continuation_contract import PublicMarketRequest
from .trend_continuation_provenance import QuoteProvenance, RawMarketProvenance, hash_raw_body, provenance_request, public_bitget_url


QUOTE_WINDOW: Final = timedelta(seconds=15)


@dataclass(frozen=True, slots=True)
class CandleWindowQuery:
    contract: BitgetContract
    granularity: CandleGranularity
    starts_at: datetime
    ends_at: datetime
    endpoint_path: str


@dataclass(frozen=True, slots=True)
class CandleWindowResult:
    eligible: bool
    rejection_code: str | None
    candles: tuple[Candle, ...]
    provenance: RawMarketProvenance


@dataclass(frozen=True, slots=True)
class QuoteObservationQuery:
    contract: BitgetContract
    evidence_time: datetime
    watcher_collected_at: datetime
    settlement_lookup_started_at: datetime
    taker_fee_rate: float


@dataclass(frozen=True, slots=True)
class QuoteObservationResult:
    eligible: bool
    rejection_code: str | None
    quote: BitgetTicker
    provenance: QuoteProvenance


@dataclass(frozen=True, slots=True)
class RejectedMarketObservation:
    eligible: bool
    rejection_code: str
    blocked_path: str | None = None


def collect_completed_candle_window(
    query: CandleWindowQuery,
    fetch_text: TextFetcher,
    *,
    collected_at: datetime,
) -> CandleWindowResult:
    request_query = _candle_query(query)
    raw_body = fetch_text(public_bitget_url(query.endpoint_path, request_query))
    raw_sha256 = hash_raw_body(raw_body)
    parsed = _parse_candles(raw_body, query.granularity)
    expected_times = _expected_opens(query)
    candles = parsed.candles
    reason = parsed.rejection_code or _window_rejection(candles, expected_times)
    eligible = reason is None
    accepted = candles if eligible else ()
    provenance = provenance_request(
        PublicMarketRequest(method="GET", path=query.endpoint_path, query=request_query),
        contract=query.contract.symbol,
        raw_sha256=raw_sha256,
        collected_at=collected_at,
        row_count=len(candles),
        complete=eligible,
        continuous=eligible,
    )
    return CandleWindowResult(
        eligible=eligible,
        rejection_code=reason,
        candles=accepted,
        provenance=provenance,
    )


def collect_forward_quote(query: QuoteObservationQuery, fetch_text: TextFetcher) -> QuoteObservationResult:
    rejected = _quote_rejection(query)
    if rejected is not None:
        return _empty_quote_result(query, rejected)

    request_query = (("productType", "USDT-FUTURES"), ("symbol", query.contract.symbol))
    raw_body = fetch_text(public_bitget_url("/api/v2/mix/market/ticker", request_query))
    raw_sha256 = hash_raw_body(raw_body)
    parsed = _parse_quote(raw_body, query.contract.symbol)
    if parsed.rejection_code is not None:
        return _empty_quote_result(query, parsed.rejection_code, raw_body=raw_body)
    quote = parsed.quote
    provenance = QuoteProvenance(
        request=PublicMarketRequest(method="GET", path="/api/v2/mix/market/ticker", query=tuple(sorted(request_query))),
        normalized_query=tuple(sorted(request_query)),
        contract=query.contract.symbol,
        collected_at=query.watcher_collected_at,
        raw_sha256=raw_sha256,
        row_count=1,
        complete=True,
        continuous=True,
        bid_price=quote.bid_price,
        ask_price=quote.ask_price,
        observed_at=quote.observed_at,
        fee_rate=query.taker_fee_rate,
    )
    return QuoteObservationResult(eligible=True, rejection_code=None, quote=quote, provenance=provenance)


def reject_historical_quote_fixture(payload: object) -> RejectedMarketObservation:
    return _reject_historical_quote_fixture(payload, path="$")


def _reject_historical_quote_fixture(payload: object, *, path: str) -> RejectedMarketObservation:
    blocked = frozenset({"bid", "ask", "spread", "bid_ask_spread_pct"})
    match payload:
        case dict():
            for key, value in payload.items():
                child_path = f"{path}.{key}"
                if key in blocked:
                    return RejectedMarketObservation(eligible=False, rejection_code="historical_quote_fixture", blocked_path=child_path)
                result = _reject_historical_quote_fixture(value, path=child_path)
                if not result.eligible:
                    return result
            return RejectedMarketObservation(eligible=True, rejection_code=None)
        case list() | tuple():
            for index, value in enumerate(payload):
                result = _reject_historical_quote_fixture(value, path=f"{path}[{index}]")
                if not result.eligible:
                    return result
            return RejectedMarketObservation(eligible=True, rejection_code=None)
        case _:
            return RejectedMarketObservation(eligible=True, rejection_code=None)


@dataclass(frozen=True, slots=True)
class ParsedCandles:
    candles: tuple[Candle, ...]
    rejection_code: str | None


@dataclass(frozen=True, slots=True)
class ParsedQuote:
    quote: BitgetTicker
    rejection_code: str | None


def _candle_query(query: CandleWindowQuery) -> tuple[tuple[str, str], ...]:
    items = [
        ("symbol", query.contract.symbol),
        ("productType", "USDT-FUTURES"),
        ("granularity", query.granularity.value),
        ("limit", "200"),
    ]
    if query.endpoint_path == "/api/v2/mix/market/history-candles":
        items.extend(
            [
                ("startTime", str(int(query.starts_at.timestamp() * 1000))),
                ("endTime", str(int(query.ends_at.timestamp() * 1000))),
            ]
        )
    return tuple(sorted(items))


def _parse_candles(raw_body: str, granularity: CandleGranularity) -> ParsedCandles:
    try:
        response = json.loads(raw_body)
    except json.JSONDecodeError:
        return ParsedCandles(candles=(), rejection_code="malformed_json")
    match response:  # noqa: MATCH_OK
        case {"code": "00000", "data": list(rows)}:
            candles = tuple(candle for row in rows if (candle := _candle_from_row(row)) is not None)
            if len(candles) != len(rows):
                return ParsedCandles(candles=(), rejection_code="malformed_row")
            ordered = tuple(sorted(candles, key=lambda candle: candle.opened_at))
            return ParsedCandles(candles=ordered, rejection_code=_duplicate_rejection(ordered))
        case {"code": str(), "msg": str()}:
            return ParsedCandles(candles=(), rejection_code="blocked_source")
        case _:
            return ParsedCandles(candles=(), rejection_code="malformed_json")


def _candle_from_row(row: object) -> Candle | None:
    match row:  # noqa: MATCH_OK
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
            except (OverflowError, OSError, ValueError):
                return None
            if min(candle.open, candle.high, candle.low, candle.close) <= 0:
                return None
            return candle
        case _:
            return None


def _duplicate_rejection(candles: tuple[Candle, ...]) -> str | None:
    opened = tuple(candle.opened_at for candle in candles)
    if len(frozenset(opened)) != len(opened):
        return "duplicate_candle"
    return None


def _expected_opens(query: CandleWindowQuery) -> tuple[datetime, ...]:
    expected: list[datetime] = []
    cursor = query.starts_at
    while cursor < query.ends_at:
        expected.append(cursor)
        cursor += query.granularity.interval
    return tuple(expected)


def _window_rejection(candles: tuple[Candle, ...], expected_times: tuple[datetime, ...]) -> str | None:
    actual = tuple(candle.opened_at for candle in candles)
    if len(actual) > 200:
        return "page_limit"
    if not actual:
        return "missing_candle"
    if actual != expected_times:
        if expected_times and any(opened_at > expected_times[-1] for opened_at in actual):
            return "open_candle"
        return "missing_candle"
    return None


def _quote_rejection(query: QuoteObservationQuery) -> str | None:
    if query.watcher_collected_at < query.evidence_time:
        return "early_quote"
    if query.watcher_collected_at > query.evidence_time + QUOTE_WINDOW:
        return "late_quote"
    if query.settlement_lookup_started_at <= query.watcher_collected_at:
        return "quote_after_settlement_lookup"
    return None


def _parse_quote(raw_body: str, symbol: str) -> ParsedQuote:
    empty = BitgetTicker(symbol=symbol, last_price=0.0, bid_price=0.0, ask_price=0.0, observed_at=datetime.fromtimestamp(0, tz=UTC))
    try:
        response = json.loads(raw_body)
    except json.JSONDecodeError:
        return ParsedQuote(quote=empty, rejection_code="malformed_json")
    match response:  # noqa: MATCH_OK
        case {
            "code": "00000",
            "data": [
                {
                    "symbol": str(raw_symbol),
                    "lastPr": str(raw_last),
                    "bidPr": str(raw_bid),
                    "askPr": str(raw_ask),
                    "ts": str(raw_ts),
                },
                *_,
            ],
        }:
            if raw_symbol != symbol:
                return ParsedQuote(quote=empty, rejection_code="symbol_mismatch")
            try:
                quote = BitgetTicker(
                    symbol=raw_symbol,
                    last_price=float(raw_last),
                    bid_price=float(raw_bid),
                    ask_price=float(raw_ask),
                    observed_at=datetime.fromtimestamp(int(raw_ts) / 1000, tz=UTC),
                )
            except (OverflowError, OSError, ValueError):
                return ParsedQuote(quote=empty, rejection_code="malformed_quote")
            if min(quote.last_price, quote.bid_price, quote.ask_price) <= 0:
                return ParsedQuote(quote=empty, rejection_code="malformed_quote")
            return ParsedQuote(quote=quote, rejection_code=None)
        case _:
            return ParsedQuote(quote=empty, rejection_code="malformed_json")


def _empty_quote_result(query: QuoteObservationQuery, rejection_code: str, raw_body: str = "{}") -> QuoteObservationResult:
    raw_sha256 = hash_raw_body(raw_body)
    empty = BitgetTicker(
        symbol=query.contract.symbol,
        last_price=0.0,
        bid_price=0.0,
        ask_price=0.0,
        observed_at=datetime.fromtimestamp(0, tz=UTC),
    )
    provenance = QuoteProvenance(
        request=PublicMarketRequest(method="GET", path="/api/v2/mix/market/ticker", query=(("productType", "USDT-FUTURES"), ("symbol", query.contract.symbol))),
        normalized_query=(("productType", "USDT-FUTURES"), ("symbol", query.contract.symbol)),
        contract=query.contract.symbol,
        collected_at=query.watcher_collected_at,
        raw_sha256=raw_sha256,
        row_count=0,
        complete=False,
        continuous=False,
        bid_price=0.0,
        ask_price=0.0,
        observed_at=empty.observed_at,
        fee_rate=query.taker_fee_rate,
    )
    return QuoteObservationResult(eligible=False, rejection_code=rejection_code, quote=empty, provenance=provenance)
