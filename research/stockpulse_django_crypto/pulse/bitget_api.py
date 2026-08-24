from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Final

from django.core.cache import cache
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .services.bitget_market import BitgetError
from .services.bitget_intraday_screener import JsonObject, JsonValue
from .services.live_long_signal_feed import screen_bitget_live_long_alerts
from .services.bitget_ticker import MAX_TICKER_AGE, MAX_TICKER_FUTURE_SKEW
from .services.trend_continuation_contract import (
    FORBIDDEN_PAYLOAD_KEYS,
    FORBIDDEN_PAYLOAD_VALUES,
    stable_manifest_digest,
)
from .services.trend_continuation_research import build_trend_continuation_report


BITGET_INTRADAY_SCREEN_CACHE_KEY: Final = "bitget:intraday-long-screen:v6"
BITGET_INTRADAY_SCREEN_CACHE_SECONDS: Final = 20
BITGET_TREND_CONTINUATION_CACHE_PREFIX: Final = "bitget:trend-continuation:v1:"
BITGET_TREND_CONTINUATION_CACHE_SECONDS: Final = 60


def _cached_enter_instruction_is_current(signal: JsonValue, now: datetime) -> bool:
    """Permit a cached instruction only while its public quote is still usable for entry."""

    match signal:
        case (
            {
                "action": "ENTER_LONG",
                "research_status": "forward_evidence_ready",
                "quote": {"observed_at": str(raw_observed_at)},
            }
            | {
                "action": "WATCH_LONG_SETUP",
                "research_status": "forward_evidence_pending",
                "quote": {"observed_at": str(raw_observed_at)},
            }
        ):
            try:
                observed_at = datetime.fromisoformat(raw_observed_at)
            except ValueError:
                return False
            if observed_at.tzinfo is None:
                return False
            return now - MAX_TICKER_AGE <= observed_at <= now + MAX_TICKER_FUTURE_SKEW
        case _:
            return False


def _cached_payload_has_current_entries(payload: JsonObject, now: datetime) -> bool:
    """Reject malformed or stale cached verified and provisional quote signals."""

    for key in ("enter_now_signals", "watch_candidates"):
        match payload.get(key, []):
            case list(signals):
                if not all(_cached_enter_instruction_is_current(signal, now) for signal in signals):
                    return False
            case _:
                return False
    return True


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def stock_long_candidates(request) -> Response:
    """Return current manual entry/exit instructions from public Bitget quotes, never an order path."""

    cached_payload = cache.get(BITGET_INTRADAY_SCREEN_CACHE_KEY)
    if isinstance(cached_payload, dict) and _cached_payload_has_current_entries(cached_payload, datetime.now(UTC)):
        return Response(cached_payload)
    if cached_payload is not None:
        cache.delete(BITGET_INTRADAY_SCREEN_CACHE_KEY)
    try:
        payload = screen_bitget_live_long_alerts().payload()
    except BitgetError:
        return Response(
            {"error": {"code": "BITGET_MARKET_UNAVAILABLE", "message": "Bitget 공개 시세를 지금 확인하지 못했습니다. 주문이나 계정 정보는 조회하지 않았습니다."}},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    cache.set(BITGET_INTRADAY_SCREEN_CACHE_KEY, payload, BITGET_INTRADAY_SCREEN_CACHE_SECONDS)
    return Response(payload)


def _trend_continuation_cache_key(now: datetime) -> str:
    boundary = _trend_continuation_completed_1m_boundary(now)
    return f"{BITGET_TREND_CONTINUATION_CACHE_PREFIX}{boundary.isoformat()}"


def _trend_continuation_completed_1m_boundary(now: datetime) -> datetime:
    return now.replace(second=0, microsecond=0) - timedelta(minutes=1)


def _trend_continuation_payload_identity(payload: JsonObject) -> str:
    diagnostics = payload.get("diagnostics", [])
    return stable_manifest_digest(
        {
            "mode": payload.get("mode"),
            "decision_status": payload.get("decision_status"),
            "diagnostics": diagnostics,
            "rejections": payload.get("rejections", []),
        }
    )


def _assert_trend_continuation_research_only(value: JsonValue) -> None:
    match value:
        case dict():
            for key, nested in value.items():
                if key in FORBIDDEN_PAYLOAD_KEYS:
                    raise ValueError(f"forbidden trend continuation payload key: {key}")
                _assert_trend_continuation_research_only(nested)
        case list() | tuple():
            for nested in value:
                _assert_trend_continuation_research_only(nested)
        case str() as text:
            if text in FORBIDDEN_PAYLOAD_VALUES:
                raise ValueError(f"forbidden trend continuation payload value: {text}")
        case _:
            return


def _public_trend_continuation_payload(report_payload: JsonObject, boundary: datetime) -> JsonObject:
    payload = dict(report_payload)
    payload["decision_status"] = "NO_TRADE"
    payload["execution"] = {"enabled": False}
    payload["cache"] = {"boundary": boundary.isoformat()}
    _assert_trend_continuation_research_only(payload)
    return payload


def _cached_trend_continuation_payload(cached_payload: JsonValue, boundary: datetime) -> JsonObject | None:
    match cached_payload:
        case {
            "schema": "trend-continuation-api-cache.v1",
            "boundary": str(raw_boundary),
            "payload_identity": str(payload_identity),
            "payload": dict(payload),
        }:
            if raw_boundary != boundary.isoformat():
                return None
            if _trend_continuation_payload_identity(payload) != payload_identity:
                return None
            try:
                _assert_trend_continuation_research_only(payload)
            except ValueError:
                return None
            return payload
        case _:
            return None


def _trend_continuation_cache_wrapper(payload: JsonObject, boundary: datetime) -> JsonObject:
    return {
        "schema": "trend-continuation-api-cache.v1",
        "boundary": boundary.isoformat(),
        "payload_identity": _trend_continuation_payload_identity(payload),
        "payload": payload,
    }


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def trend_continuation_research(request) -> Response:
    now = datetime.now(UTC)
    boundary = _trend_continuation_completed_1m_boundary(now)
    cache_key = _trend_continuation_cache_key(now)
    cached_payload = cache.get(cache_key)
    cached_public_payload = _cached_trend_continuation_payload(cached_payload, boundary)
    if cached_public_payload is not None:
        return Response(cached_public_payload)
    if cached_payload is not None:
        cache.delete(cache_key)
    try:
        report = build_trend_continuation_report(now)
    except BitgetError:
        return Response(
            {"error": {"code": "DATA_UNAVAILABLE", "message": "Bitget 공개 시장 데이터를 지금 확인하지 못했습니다."}},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:
        payload = _public_trend_continuation_payload(report.payload(), boundary)
    except ValueError:
        return Response(
            {"error": {"code": "DATA_UNAVAILABLE", "message": "추세 지속 연구 결과가 공개 연구 전용 계약을 통과하지 못했습니다."}},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    cache.set(cache_key, _trend_continuation_cache_wrapper(payload, boundary), BITGET_TREND_CONTINUATION_CACHE_SECONDS)
    return Response(payload)
