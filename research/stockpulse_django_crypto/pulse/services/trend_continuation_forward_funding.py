from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from .bitget_market import BitgetContract, TextFetcher
from .trend_continuation_contract import PublicMarketRequest
from .trend_continuation_manifest import JsonObject, JsonValue
from .trend_continuation_provenance import (
    hash_raw_body,
    public_bitget_url,
    require_v2_public_market_request,
)


@dataclass(frozen=True, slots=True)
class CurrentFundingError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CurrentFundingSnapshot:
    request: PublicMarketRequest
    contract: str
    collected_at: datetime
    funding_rate: float
    raw_sha256: str

    def payload(self) -> JsonObject:
        return {
            "complete": True,
            "scope": "capture_time_current_rate",
            "contract": self.contract,
            "collected_at": self.collected_at.isoformat(),
            "funding_rate": self.funding_rate,
            "raw_sha256": self.raw_sha256,
            "query": [[key, value] for key, value in self.request.query],
        }


def collect_current_funding(
    contract: BitgetContract,
    fetch_text: TextFetcher,
    *,
    collected_at: datetime,
) -> CurrentFundingSnapshot:
    query = (("productType", "USDT-FUTURES"), ("symbol", contract.symbol))
    request = require_v2_public_market_request(
        "GET", "/api/v2/mix/market/current-fund-rate", query
    )
    raw_body = fetch_text(public_bitget_url(request.path, request.query))
    raw_sha256 = hash_raw_body(raw_body)
    try:
        response: JsonValue = json.loads(raw_body)
    except json.JSONDecodeError as error:
        raise CurrentFundingError("current funding response is not JSON") from error
    funding_rate = _funding_rate(response, contract.symbol)
    if funding_rate is None:
        raise CurrentFundingError("current funding response is incomplete")
    return CurrentFundingSnapshot(
        request=request,
        contract=contract.symbol,
        collected_at=collected_at,
        funding_rate=funding_rate,
        raw_sha256=raw_sha256,
    )


def _funding_rate(response: JsonValue, contract: str) -> float | None:
    match response:  # noqa: MATCH_OK - the provider payload is rejected unless exact fields parse.
        case {"code": "00000", "data": list(rows)}:
            matches = tuple(_row_rate(row, contract) for row in rows)
            accepted = tuple(rate for rate in matches if rate is not None)
            return accepted[0] if len(accepted) == 1 else None
        case _:
            return None


def _row_rate(row: JsonValue, contract: str) -> float | None:
    match row:  # noqa: MATCH_OK - malformed current-rate rows are rejected.
        case {"symbol": str(symbol), "fundingRate": str(raw_rate)} if symbol == contract:
            try:
                return float(raw_rate)
            except ValueError:
                return None
        case _:
            return None
