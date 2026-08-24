from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qsl, urlparse

from pulse.models import Stock

from .bitget_market import BitgetPublicClient, TextFetcher, list_rwa_contracts, select_stock_contracts
from .trend_continuation_evaluation import select_historical_candidate
from .trend_continuation_fixture import (
    HistoricalDataset,
    HistoricalResearchError,
    HistoricalResearchRequest,
    load_fixture_dataset,
    validate_research_request,
)
from .trend_continuation_funding import (
    FundingWindow,
    FundingWindowResult,
    collect_funding_window,
)
from .trend_continuation_historical_selection import (
    build_historical_developments,
    fee_covers_history,
    freeze_historical_selection,
)
from .trend_continuation_history import (
    HistoricalCollectionQuery,
    collect_historical_candles,
    required_profile_intervals,
)
from .trend_continuation_manifest import JsonObject
from .trend_continuation_provenance import hash_raw_body, require_v2_public_market_request


@dataclass(frozen=True, slots=True)
class HistoricalResearchOutcome:
    evidence: JsonObject
    failure_code: str | None

    @property
    def successful(self) -> bool:
        return self.failure_code is None

    def payload(self) -> JsonObject:
        return self.evidence


def eligible_local_us_symbols() -> frozenset[str]:
    return frozenset(
        Stock.objects.filter(market="US", asset_type__in=("stock", "etf")).values_list(
            "code", flat=True
        )
    )


def evaluate_historical_research(
    request: HistoricalResearchRequest,
    *,
    fetch_text: TextFetcher | None = None,
    collected_at: datetime,
) -> HistoricalResearchOutcome:
    validate_research_request(request)
    dataset = (
        load_fixture_dataset(request)
        if request.input_fixture is not None
        else _network_dataset(request, fetch_text or BitgetPublicClient().get_text, collected_at)
    )
    candle_complete = all(series.complete and series.continuous for series in dataset.series)
    funding_complete = dataset.funding.covers(request.starts_at, request.ends_at)
    fee_complete = fee_covers_history(dataset, request, collected_at)
    developments, unresolved_count = build_historical_developments(
        request,
        dataset,
        eligible=candle_complete and funding_complete and fee_complete,
    )
    selection = select_historical_candidate(developments)
    manifest = freeze_historical_selection(selection, dataset, collected_at)
    failures = _failure_codes(dataset, candle_complete, funding_complete, fee_complete)
    payload: JsonObject = {
        "mode": "research_only",
        "execution": {"enabled": False},
        "exploratory_only": True,
        "decision": "NO_RELIABLE_WINNER",
        "symbol": request.symbol,
        "profile": request.profile,
        "history": {
            "start": request.starts_at.isoformat(),
            "end": request.ends_at.isoformat(),
            "max_pages": request.max_pages,
            "required_intervals": [
                interval.value for interval in required_profile_intervals(request.profile)
            ],
            "contract_raw_sha256": dataset.contract_raw_sha256,
            "series": [series.payload() for series in dataset.series],
        },
        "funding": dataset.funding_evidence,
        "fee_policy": (
            dataset.fee_policy.to_payload()
            if dataset.fee_policy is not None
            else {"coverage_proven": False, "rejection_code": "historical_fee_policy_coverage_unproven"}
        ),
        "data_contracts": {
            "candles_complete": candle_complete,
            "funding_complete": funding_complete,
            "fee_policy_complete": fee_complete,
            "failure_codes": list(failures),
        },
        "research": {
            "candidate_count": len(developments),
            "unresolved_rows": unresolved_count,
            "candidate_frozen": manifest is not None,
            "manifest_digest": manifest.digest if manifest is not None else None,
        },
    }
    return HistoricalResearchOutcome(payload, failures[0] if failures else None)


def _network_dataset(
    request: HistoricalResearchRequest,
    fetch_text: TextFetcher,
    collected_at: datetime,
) -> HistoricalDataset:
    captured_hash = ""

    def capture_contract(url: str) -> str:
        nonlocal captured_hash
        parsed = urlparse(url)
        query = tuple(parse_qsl(parsed.query, keep_blank_values=True))
        require_v2_public_market_request("GET", parsed.path, query)
        raw_body = fetch_text(url)
        captured_hash = hash_raw_body(raw_body)
        return raw_body

    contracts = select_stock_contracts(
        list_rwa_contracts(capture_contract), eligible_local_us_symbols()
    )
    contract = next((item for item in contracts if item.symbol == request.symbol), None)
    if contract is None:
        raise HistoricalResearchError(f"unknown public RWA contract: {request.symbol}")
    series = tuple(
        collect_historical_candles(
            HistoricalCollectionQuery(
                contract, interval, request.starts_at, request.ends_at, request.max_pages
            ),
            fetch_text,
        )
        for interval in required_profile_intervals(request.profile)
    )
    funding_window = FundingWindow(contract, request.starts_at, request.ends_at)
    funding_result = collect_funding_window(
        funding_window,
        fetch_text,
        collected_at=collected_at,
        max_pages=request.max_pages,
    )
    codes = tuple(
        code
        for code in (
            *(item.rejection_code for item in series),
            funding_result.rejection_code,
        )
        if code is not None
    )
    return HistoricalDataset(
        contract=contract,
        series=series,
        funding=funding_result.to_evaluation_proof(funding_window),
        funding_evidence=_funding_evidence(funding_result, funding_window),
        fee_policy=None,
        contract_raw_sha256=captured_hash,
        rejection_codes=codes,
    )


def _failure_codes(
    dataset: HistoricalDataset,
    candles_complete: bool,
    funding_complete: bool,
    fee_complete: bool,
) -> tuple[str, ...]:
    codes = list(dataset.rejection_codes)
    if not candles_complete:
        codes.append("historical_candle_coverage_incomplete")
    if not funding_complete:
        codes.append("historical_funding_coverage_incomplete")
    if not fee_complete:
        codes.append("historical_fee_policy_coverage_unproven")
    return tuple(sorted(frozenset(codes)))


def _funding_evidence(
    result: FundingWindowResult,
    window: FundingWindow,
) -> JsonObject:
    return {
        "complete": result.coverage_complete,
        "rejection_code": result.rejection_code,
        "terminal_reason": result.terminal_reason,
        "aggregate_raw_sha256": result.aggregate_raw_sha256,
        "page_count": len(result.pages),
        "row_count": len(result.events),
        "coverage_start": window.evidence_time.isoformat(),
        "coverage_end": window.label_end_time.isoformat(),
        "pages": [
            {
                "page_number": page.page_number,
                "query": [[key, value] for key, value in page.request.query],
                "raw_sha256": page.raw_sha256,
                "row_count": page.row_count,
            }
            for page in result.pages
        ],
    }
