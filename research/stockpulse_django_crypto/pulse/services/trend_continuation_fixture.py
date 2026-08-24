from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .bitget_market import BitgetContract
from .long_forecast import Candle
from .market_timeframes import CandleGranularity
from .trend_continuation_evaluation import FeePolicySnapshot, FundingProof, FundingRateEvent
from .trend_continuation_history import HistoricalCandleResult, required_profile_intervals
from .trend_continuation_manifest import JsonObject, JsonValue
from .trend_continuation_market import reject_historical_quote_fixture
from .trend_continuation_provenance import hash_raw_body


@dataclass(frozen=True, slots=True)
class HistoricalResearchError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class HistoricalResearchRequest:
    symbol: str
    profile: str
    starts_at: datetime
    ends_at: datetime
    max_pages: int
    input_fixture: Path | None = None


@dataclass(frozen=True, slots=True)
class HistoricalDataset:
    contract: BitgetContract
    series: tuple[HistoricalCandleResult, ...]
    funding: FundingProof
    funding_evidence: JsonObject
    fee_policy: FeePolicySnapshot | None
    contract_raw_sha256: str
    rejection_codes: tuple[str, ...]


def validate_research_request(request: HistoricalResearchRequest) -> None:
    required_profile_intervals(request.profile)
    if not 1 <= request.max_pages <= 720:
        raise HistoricalResearchError("--max-pages must be between 1 and 720")
    if request.starts_at.tzinfo is None or request.ends_at.tzinfo is None:
        raise HistoricalResearchError("start/end must include timezone info")
    if request.starts_at >= request.ends_at:
        raise HistoricalResearchError("start must precede end")


def load_fixture_dataset(request: HistoricalResearchRequest) -> HistoricalDataset:
    path = request.input_fixture
    if path is None or not path.is_file():
        raise HistoricalResearchError("input fixture does not exist")
    raw = path.read_text(encoding="utf-8")
    try:
        payload: JsonValue = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HistoricalResearchError("input fixture is not valid JSON") from error
    rejected = reject_historical_quote_fixture(payload)
    if not rejected.eligible:
        raise HistoricalResearchError(
            f"historical fixture cannot contain bid/ask/spread fields: {rejected.blocked_path}"
        )
    if not isinstance(payload, dict) or set(payload) != {"candles", "funding", "fee_policy"}:
        raise HistoricalResearchError("fixture must contain only candles, funding, and fee_policy")
    raw_sha256 = hash_raw_body(raw)
    contract = BitgetContract(request.symbol, request.symbol.removesuffix("USDT"), 1, 8, 0.0)
    series = tuple(
        _fixture_series(request, interval, payload["candles"], raw_sha256)
        for interval in required_profile_intervals(request.profile)
    )
    funding = _fixture_funding(payload["funding"])
    return HistoricalDataset(
        contract=contract,
        series=series,
        funding=funding,
        funding_evidence=_fixture_funding_evidence(funding),
        fee_policy=_fixture_fee_policy(payload["fee_policy"]),
        contract_raw_sha256=raw_sha256,
        rejection_codes=(),
    )


def _fixture_series(
    request: HistoricalResearchRequest,
    interval: CandleGranularity,
    raw_candles: JsonValue,
    raw_sha256: str,
) -> HistoricalCandleResult:
    if not isinstance(raw_candles, dict) or interval.value not in raw_candles:
        raise HistoricalResearchError(f"fixture is missing {interval.value} candles")
    rows = raw_candles[interval.value]
    if not isinstance(rows, list):
        raise HistoricalResearchError(f"fixture {interval.value} candles must be a list")
    candles = tuple(candle for row in rows if (candle := _fixture_candle(row)) is not None)
    if len(candles) != len(rows):
        raise HistoricalResearchError(f"fixture {interval.value} candle is malformed")
    expected: list[datetime] = []
    cursor = request.starts_at
    while cursor < request.ends_at:
        expected.append(cursor)
        cursor += interval.interval
    ordered = tuple(sorted(candles, key=lambda candle: candle.opened_at))
    actual = tuple(candle.opened_at for candle in ordered)
    complete = actual == tuple(expected)
    gaps = tuple(value for value in expected if value not in frozenset(actual))
    return HistoricalCandleResult(
        granularity=interval,
        candles=ordered if complete else (),
        pages=(),
        complete=complete,
        continuous=complete,
        rejection_code=None if complete else "missing_candle",
        terminal_reason="fixture_complete",
        aggregate_raw_sha256=raw_sha256,
        first_open=ordered[0].opened_at if ordered else None,
        last_open=ordered[-1].opened_at if ordered else None,
        continuity_gaps=gaps,
    )


def _fixture_candle(row: JsonValue) -> Candle | None:
    match row:  # noqa: MATCH_OK - deterministic fixture rows are validated into candles.
        case [
            str(raw_time),
            int() | float() as raw_open,
            int() | float() as raw_high,
            int() | float() as raw_low,
            int() | float() as raw_close,
            int() | float() as raw_volume,
        ]:
            try:
                candle = Candle(
                    datetime.fromisoformat(raw_time),
                    float(raw_open),
                    float(raw_high),
                    float(raw_low),
                    float(raw_close),
                    float(raw_volume),
                )
            except ValueError:
                return None
            return candle if min(candle.open, candle.high, candle.low, candle.close) > 0 else None
        case _:
            return None


def _fixture_funding(payload: JsonValue) -> FundingProof:
    if not isinstance(payload, dict):
        raise HistoricalResearchError("fixture funding must be an object")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise HistoricalResearchError("fixture funding proof is malformed")
    events = tuple(_fixture_funding_event(row) for row in raw_events)
    if any(event is None for event in events):
        raise HistoricalResearchError("fixture funding proof is malformed")
    try:
        return FundingProof(
            complete=payload["complete"] is True,
            events=tuple(event for event in events if event is not None),
            coverage_start=datetime.fromisoformat(str(payload["coverage_start"])),
            coverage_end=datetime.fromisoformat(str(payload["coverage_end"])),
            provenance_digest=str(payload["provenance_digest"]),
            page_count=int(payload["page_count"]),
            row_count=len(events),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HistoricalResearchError("fixture funding proof is malformed") from error


def _fixture_funding_event(payload: JsonValue) -> FundingRateEvent | None:
    match payload:  # noqa: MATCH_OK - malformed fixture events are rejected.
        case {"funding_time": str(raw_time), "funding_rate": int() | float() as raw_rate}:
            try:
                return FundingRateEvent(datetime.fromisoformat(raw_time), float(raw_rate))
            except ValueError:
                return None
        case _:
            return None


def _fixture_funding_evidence(proof: FundingProof) -> JsonObject:
    return {
        "complete": proof.complete,
        "rejection_code": None if proof.complete else "fixture_funding_incomplete",
        "terminal_reason": "fixture_complete" if proof.complete else "fixture_incomplete",
        "aggregate_raw_sha256": proof.provenance_digest,
        "page_count": proof.page_count,
        "row_count": proof.row_count,
        "coverage_start": proof.coverage_start.isoformat() if proof.coverage_start else None,
        "coverage_end": proof.coverage_end.isoformat() if proof.coverage_end else None,
        "pages": [
            {
                "page_number": 1,
                "raw_sha256": proof.provenance_digest,
                "row_count": proof.row_count,
            }
        ] if proof.page_count == 1 else [],
    }


def _fixture_fee_policy(payload: JsonValue) -> FeePolicySnapshot:
    if not isinstance(payload, dict):
        raise HistoricalResearchError("fixture fee_policy must be an object")
    try:
        return FeePolicySnapshot(
            version=str(payload["version"]),
            recorded_at=datetime.fromisoformat(str(payload["recorded_at"])),
            source_url=str(payload["source_url"]),
            raw_sha256=str(payload["raw_sha256"]),
            effective_from=datetime.fromisoformat(str(payload["effective_from"])),
            effective_until=datetime.fromisoformat(str(payload["effective_until"])),
            taker_fee_rate=float(payload["taker_fee_rate"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HistoricalResearchError("fixture fee_policy is malformed") from error
