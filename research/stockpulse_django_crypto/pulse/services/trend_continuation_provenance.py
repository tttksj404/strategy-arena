from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Final
from urllib.parse import urlencode

from .trend_continuation_contract import PublicMarketRequest


RAW_SHA_256_EMPTY_JSON: Final = hashlib.sha256(b"{}").hexdigest()


@dataclass(frozen=True, slots=True)
class ProvenanceBlockedSource(Exception):
    method: str
    path: str

    def __str__(self) -> str:
        return f"blocked trend-continuation public source: {self.method} {self.path}"


@dataclass(frozen=True, slots=True)
class RawMarketProvenance:
    request: PublicMarketRequest
    normalized_query: tuple[tuple[str, str], ...]
    contract: str
    collected_at: datetime
    raw_sha256: str
    row_count: int
    complete: bool
    continuous: bool


@dataclass(frozen=True, slots=True)
class QuoteProvenance(RawMarketProvenance):
    bid_price: float
    ask_price: float
    observed_at: datetime
    fee_rate: float


def hash_raw_body(raw_body: str) -> str:
    return hashlib.sha256(raw_body.encode("utf-8")).hexdigest()


def normalize_query(query: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(query, key=lambda item: (item[0], item[1])))


def require_v2_public_market_request(
    method: str,
    path: str,
    query: tuple[tuple[str, str], ...] = (),
) -> PublicMarketRequest:
    normalized_method = method.upper()
    if normalized_method != "GET":
        raise ProvenanceBlockedSource(method=normalized_method, path=path)
    try:
        return PublicMarketRequest(method=normalized_method, path=path, query=normalize_query(query))
    except ValueError as error:
        raise ProvenanceBlockedSource(method=normalized_method, path=path) from error


def provenance_request(
    request: PublicMarketRequest,
    *,
    contract: str,
    raw_sha256: str,
    collected_at: datetime,
    row_count: int,
    complete: bool,
    continuous: bool,
) -> RawMarketProvenance:
    checked = require_v2_public_market_request(request.method, request.path, request.query)
    return RawMarketProvenance(
        request=checked,
        normalized_query=checked.query,
        contract=contract,
        collected_at=collected_at,
        raw_sha256=raw_sha256,
        row_count=row_count,
        complete=complete,
        continuous=continuous,
    )


def public_bitget_url(path: str, query: tuple[tuple[str, str], ...]) -> str:
    request = require_v2_public_market_request("GET", path, query)
    return f"https://api.bitget.com{request.path}?{urlencode(request.query)}"
