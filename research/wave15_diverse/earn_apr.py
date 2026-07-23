# B1's Binance Simple Earn Flexible historical-APR feasibility probe (SPEC.md: "과거 APR
# 시계열 수집 가능한 범위, 불가시 보수적 고정 2% 가정 명시"). Every historical/current-rate
# Simple Earn REST route (`/sapi/v1/simple-earn/flexible/*`) requires an HMAC-signed request
# with an account API key (interactively confirmed below: `/sapi/v1/simple-earn/flexible/list`
# returns HTTP 400 `{"code":-2014,"msg":"API-key format invalid."}` with NO key supplied, and
# this repo/session holds no Binance API credentials -- entering exchange API keys is outside
# what this task is authorized to do, and no user-provided key exists to try anyway). Several
# unauthenticated "public homepage widget" endpoint guesses were also tried and 404'd. This
# module records that probe result (so it is reproducible/auditable, not just asserted in
# prose) and exposes the resulting constant fallback -- ASSUMED, never "measured".
#
# Consequence for B1 (and B2's USDT collateral leg, which reuses the same fallback -- see
# common15.ASSUMED_FLEXIBLE_EARN_APR): every wave15_report.md figure that includes this yield
# must be labeled as resting on an ASSUMED rate, and the carry-only vs carry+earn columns must
# stay separated so a reader can discount the assumed portion entirely if they choose to.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import requests

from research.wave15_diverse.common15 import ASSUMED_FLEXIBLE_EARN_APR

PROBED_ENDPOINTS: Final[tuple[str, ...]] = (
    "https://api.binance.com/sapi/v1/simple-earn/flexible/list",
    "https://www.binance.com/bapi/earn/v1/friendly/finance-earn/simple/homepage/details",
    "https://www.binance.com/bapi/earn/v2/public/simple-earn/flexible/list",
    "https://www.binance.com/bapi/earn/v2/public/simple-earn/flexible/product/list",
    "https://www.binance.com/bapi/earn/v1/public/simple-earn/flexible/product/list",
)


@dataclass(frozen=True, slots=True)
class EarnAprProbeResult:
    endpoint: str
    status_code: int | None
    outcome: str  # "auth_required" | "not_found" | "network_error" | "unexpected_success"
    detail: str


def probe_endpoint(url: str, session: requests.Session) -> EarnAprProbeResult:
    try:
        response = session.get(url, params={"asset": "USDT"}, timeout=(5.0, 15.0))
    except requests.RequestException as error:
        return EarnAprProbeResult(url, None, "network_error", str(error))
    if response.status_code == 400 and "-2014" in response.text:
        return EarnAprProbeResult(url, response.status_code, "auth_required", response.text[:200])
    if response.status_code == 404:
        return EarnAprProbeResult(url, response.status_code, "not_found", response.text[:200])
    if response.status_code == 200:
        return EarnAprProbeResult(url, response.status_code, "unexpected_success", response.text[:500])
    return EarnAprProbeResult(url, response.status_code, "network_error", response.text[:200])


def probe_all(session: requests.Session | None = None) -> tuple[EarnAprProbeResult, ...]:
    owned = session is None
    client = session or requests.Session()
    try:
        return tuple(probe_endpoint(url, client) for url in PROBED_ENDPOINTS)
    finally:
        if owned:
            client.close()


def resolve_flexible_earn_apr(force_probe: bool = False, session: requests.Session | None = None) -> tuple[float, bool, tuple[EarnAprProbeResult, ...]]:
    """Returns (apr, is_verified, probe_results). is_verified is always False today (every
    probed route requires auth or 404s) -- kept as an explicit bool rather than silently
    hardcoding 2% so a future run with real credentials available has a place to flip this
    without touching every call site."""
    results: tuple[EarnAprProbeResult, ...] = ()
    if force_probe:
        results = probe_all(session)
    return ASSUMED_FLEXIBLE_EARN_APR, False, results


__all__ = ["EarnAprProbeResult", "PROBED_ENDPOINTS", "probe_all", "probe_endpoint", "resolve_flexible_earn_apr"]
