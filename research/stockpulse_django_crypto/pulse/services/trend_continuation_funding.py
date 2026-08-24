from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, assert_never

from .bitget_market import BitgetContract, BitgetError, TextFetcher
from .trend_continuation_contract import Direction, PublicMarketRequest, stable_manifest_digest
from .trend_continuation_evaluation import FundingProof, FundingRateEvent
from .trend_continuation_manifest import JsonValue
from .trend_continuation_provenance import (
    RawMarketProvenance,
    hash_raw_body,
    provenance_request,
    public_bitget_url,
    require_v2_public_market_request,
)


MAX_FUNDING_PAGES: Final = 720
FUNDING_PAGE_SIZE: Final = 100


@dataclass(frozen=True, slots=True)
class FundingCollectionError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class FundingEvent:
    funding_time: datetime
    funding_rate: float


@dataclass(frozen=True, slots=True)
class FundingWindow:
    contract: BitgetContract
    evidence_time: datetime
    label_end_time: datetime


@dataclass(frozen=True, slots=True)
class FundingPageProvenance:
    page_number: int
    request: PublicMarketRequest
    raw_sha256: str
    row_count: int


@dataclass(frozen=True, slots=True)
class FundingWindowResult:
    eligible: bool
    rejection_code: str | None
    events: tuple[FundingEvent, ...]
    provenance: RawMarketProvenance
    pages: tuple[FundingPageProvenance, ...]
    terminal_reason: str
    aggregate_raw_sha256: str
    coverage_complete: bool

    def to_evaluation_proof(self, window: FundingWindow) -> FundingProof:
        return FundingProof(
            complete=self.coverage_complete and self.rejection_code is None,
            events=tuple(FundingRateEvent(event.funding_time, event.funding_rate) for event in self.events),
            coverage_start=window.evidence_time,
            coverage_end=window.label_end_time,
            provenance_digest=self.aggregate_raw_sha256,
            page_count=len(self.pages),
            row_count=len(self.events),
        )


@dataclass(frozen=True, slots=True)
class DirectionalFundingCost:
    direction: Direction
    cost_pct: float

    @classmethod
    def from_events(cls, events: tuple[FundingEvent, ...], direction: Direction) -> DirectionalFundingCost:
        match direction:
            case Direction.UP:
                cost = sum(max(event.funding_rate, 0.0) * 100 for event in events)
            case Direction.DOWN:
                cost = sum(max(-event.funding_rate, 0.0) * 100 for event in events)
            case Direction.NEUTRAL:
                cost = 0.0
            case unreachable:
                assert_never(unreachable)
        return cls(direction=direction, cost_pct=round(cost, 10))


def collect_funding_window(
    window: FundingWindow,
    fetch_text: TextFetcher,
    *,
    collected_at: datetime,
    max_pages: int = MAX_FUNDING_PAGES,
) -> FundingWindowResult:
    if not 1 <= max_pages <= MAX_FUNDING_PAGES:
        raise FundingCollectionError("funding max_pages must be between 1 and 720")
    base_query = _base_query(window)
    pages: list[FundingPageProvenance] = []
    collected: list[FundingEvent] = []
    terminal_reason = "page_limit_exhausted"
    rejection: str | None = None
    for page_number in range(1, max_pages + 1):
        query = tuple(sorted((*base_query, ("pageNo", str(page_number)))))
        request = require_v2_public_market_request("GET", "/api/v2/mix/market/history-fund-rate", query)
        try:
            raw_body = fetch_text(public_bitget_url(request.path, request.query))
        except BitgetError:
            rejection = "funding_unavailable"
            terminal_reason = "source_unavailable"
            break
        raw_sha256 = hash_raw_body(raw_body)
        parsed = _parse_funding(raw_body)
        pages.append(FundingPageProvenance(page_number, request, raw_sha256, len(parsed.events)))
        if parsed.rejection_code is not None:
            rejection = parsed.rejection_code
            terminal_reason = "malformed_page"
            break
        collected.extend(parsed.events)
        if len(parsed.events) < FUNDING_PAGE_SIZE:
            terminal_reason = "short_page"
            break
    coverage_complete = terminal_reason == "short_page" and rejection is None
    aggregate = _aggregate_digest(tuple(pages))
    filtered = tuple(
        sorted(
            (event for event in collected if window.evidence_time < event.funding_time <= window.label_end_time),
            key=lambda event: event.funding_time,
        )
    )
    if coverage_complete and _has_duplicate_time(tuple(collected)):
        rejection = "duplicate_funding"
        coverage_complete = False
        terminal_reason = "duplicate_event"
    if not coverage_complete and rejection is None:
        rejection = "funding_page_limit"
    eligible = coverage_complete and rejection is None
    provenance = provenance_request(
        PublicMarketRequest(method="GET", path="/api/v2/mix/market/history-fund-rate", query=base_query),
        contract=window.contract.symbol,
        raw_sha256=aggregate,
        collected_at=collected_at,
        row_count=len(filtered),
        complete=eligible,
        continuous=coverage_complete,
    )
    return FundingWindowResult(
        eligible=eligible,
        rejection_code=rejection,
        events=filtered if eligible else (),
        provenance=provenance,
        pages=tuple(pages),
        terminal_reason=terminal_reason,
        aggregate_raw_sha256=aggregate,
        coverage_complete=coverage_complete,
    )


@dataclass(frozen=True, slots=True)
class ParsedFunding:
    events: tuple[FundingEvent, ...]
    rejection_code: str | None


def _base_query(window: FundingWindow) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("symbol", window.contract.symbol),
                ("productType", "USDT-FUTURES"),
                ("startTime", str(int(window.evidence_time.timestamp() * 1000))),
                ("endTime", str(int(window.label_end_time.timestamp() * 1000))),
                ("pageSize", str(FUNDING_PAGE_SIZE)),
            )
        )
    )


def _parse_funding(raw_body: str) -> ParsedFunding:
    try:
        response: JsonValue = json.loads(raw_body)
    except json.JSONDecodeError:
        return ParsedFunding((), "malformed_json")
    match response:  # noqa: MATCH_OK - untrusted provider JSON is converted to typed rows here.
        case {"code": "00000", "data": list(rows)}:
            parsed = tuple(event for row in rows if (event := _event_from_row(row)) is not None)
            if len(parsed) != len(rows):
                return ParsedFunding((), "malformed_funding")
            return ParsedFunding(tuple(sorted(parsed, key=lambda event: event.funding_time)), None)
        case _:
            return ParsedFunding((), "malformed_json")


def _event_from_row(row: JsonValue) -> FundingEvent | None:
    match row:  # noqa: MATCH_OK - malformed provider rows are rejected, never defaulted.
        case {"fundingTime": str(raw_time), "fundingRate": str(raw_rate)}:
            try:
                return FundingEvent(
                    funding_time=datetime.fromtimestamp(int(raw_time) / 1000, tz=UTC),
                    funding_rate=float(raw_rate),
                )
            except (OverflowError, OSError, ValueError):
                return None
        case _:
            return None


def _has_duplicate_time(events: tuple[FundingEvent, ...]) -> bool:
    event_times = tuple(event.funding_time for event in events)
    return len(frozenset(event_times)) != len(event_times)


def _aggregate_digest(pages: tuple[FundingPageProvenance, ...]) -> str:
    payload: list[JsonValue] = [
        {
            "page_number": page.page_number,
            "query": [[key, value] for key, value in page.request.query],
            "raw_sha256": page.raw_sha256,
            "row_count": page.row_count,
        }
        for page in pages
    ]
    return stable_manifest_digest(payload)
