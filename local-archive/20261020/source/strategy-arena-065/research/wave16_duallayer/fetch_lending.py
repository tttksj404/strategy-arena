# Wave-16 work item 1: current-snapshot lending/funding collection (SPEC.md "방법 1": "현재
# 단면: 전 심볼의 (펀딩 APR, 대여이자, 유동성, 실측 슬리피지) 수집·저장"). This is the ONLY
# network-touching module in wave16_duallayer -- the historical funding/OHLCV time series the
# backtest itself runs over is NOT refetched here (it is borrowed read-only from
# research/wave12_frontier/cache via research.wave13_liquidity.universe_liquidity, same as every
# E0-E4 candidate; see engine16.py's module docstring). What this module fetches is exactly the
# THREE live things SPEC.md's "발견" section names:
#   1. OKX public `finance/savings/lending-rate-summary` -- current average lending (savings/
#      peer-to-peer margin-pool) rate per currency, ~170 coins.
#   2. Bitget public `mix/market/current-fund-rate` -- current perp funding rate, ALL USDT-M
#      contracts (one call, no per-symbol looping needed, unlike wave13's collect_spreads.py).
#   3. Bitget public `earn/loan/public/coinInfos` -- Bitget's OWN (borrower-side) loan book rate,
#      used only as a cross-check/secondary data point (see LENDER_SIDE_UNCERTAINTY_NOTE below),
#      plus a probe of Bitget's `earn/savings/*` routes to check whether Bitget itself exposes a
#      public (unauthenticated) SPOT-asset lend-out product -- SPEC.md 치명적 한계 3 asks this be
#      checked BEFORE accepting a cross-exchange (OKX-lend / Bitget-execute) structure.
#
# Everything downstream (engine16.py's ranking/PnL construction, reporting16.py's snapshot table)
# consumes cache/lending_snapshot.json with no further network access -- the same fail-closed,
# collect-then-cache convention as research/wave13_liquidity/collect_spreads.py.

from __future__ import annotations

from pathlib import Path
import re
import sys
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import requests

from research.wave1.common import save_json

BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR / "cache"

OKX_BASE_URL: Final = "https://www.okx.com"
OKX_LENDING_SUMMARY_PATH: Final = "/api/v5/finance/savings/lending-rate-summary"

BITGET_BASE_URL: Final = "https://api.bitget.com"
BITGET_CURRENT_FUNDING_PATH: Final = "/api/v2/mix/market/current-fund-rate"
BITGET_LOAN_COININFOS_PATH: Final = "/api/v2/earn/loan/public/coinInfos"
# Probed, not assumed -- SPEC.md 치명적 한계 3: "Bitget 자체 대여 가능 여부를 우선 확인". Every
# route below was interactively confirmed (2026-07-23, this task) to require an authenticated
# ACCESS_KEY (or 404) with NO key supplied -- mirrors research/wave15_diverse/earn_apr.py's own
# probe-and-record convention for an analogous "can't get past auth publicly" finding.
BITGET_SAVINGS_PROBE_PATHS: Final[tuple[str, ...]] = (
    "/api/v2/earn/savings/product",
    "/api/v2/earn/savings/account",
)
BITGET_PRODUCT_TYPE: Final = "usdt-futures"

MAX_RETRIES: Final = 3
REQUEST_SLEEP_SECONDS: Final = 0.15

# SPEC.md "발견": "BETH 365%는 이상치이므로 제외" -- OKX's wrapped-staked-ETH savings listing is a
# promotional/staking-linked rate, not a spot altcoin the L4 futures universe could ever hold (no
# "BETHUSDT" perp exists), so it can never actually join to a wave13 symbol anyway -- excluded here
# purely so descriptive stats (median/percentiles) reported alongside the snapshot aren't skewed by
# a value that isn't a real candidate. A general (not just BETH-specific) sanity cap catches any
# FUTURE anomaly of the same shape rather than only ever guarding this one hardcoded name.
HARDCODED_OUTLIER_CCY: Final[tuple[str, ...]] = ("BETH",)
OUTLIER_APR_CAP: Final = 1.0  # 100% APR -- anything above this is flagged+excluded from descriptive stats, never silently averaged in

# SPEC.md "치명적 한계 2": "대여이자가 대여자 수취분인지 차입자 지불분인지... 전부 미확인." This
# task attempted to resolve it (2026-07-23) via OKX's public API docs (`#financial-product-savings`
# anchor) and a web search; neither yielded an explicit "lender receives X net of fees" statement.
# Structural evidence found: OKX's Savings market is peer-to-peer (a private, authenticated
# `POST /api/v5/finance/savings/set-lending-rate` lets a LENDER quote their own rate; the public
# `lending-rate-summary` is the resulting system-wide matched-rate average/estimate/previous-period
# triplet: avgRate/estRate/preRate), which is *consistent* with avgRate approximating what lenders
# actually clear at (a matched peer-to-peer book, not a bank-style borrow/deposit spread product) --
# but no authoritative text confirming "net of platform fee" was retrievable through the tools
# available to this task. Bitget's OWN `earn/loan/public/coinInfos` (a *different*, borrower-facing
# loan book) shows a same-order-of-magnitude ETH rate (1.26% 7D) vs OKX's ETH avgRate (1.50%) --
# suggestive that OKX's number is not wildly detached from a comparable market, but this is
# circumstantial, not confirmation, and the two are different products on different venues.
# CONCLUSION: UNCONFIRMED. Treated as uncertain, not asserted either way -- E3's 50% discount
# candidate is this wave's mitigation for exactly this uncertainty, not a resolution of it.
LENDER_SIDE_UNCERTAINTY_NOTE: Final = (
    "OKX avgRate가 '대여자(lender) 실수취 순액'인지 확인 시도(2026-07-23): OKX 공개 API 문서 "
    "(#financial-product-savings 앵커) 및 웹 검색 결과, avgRate/estRate/preRate에 대한 명시적 "
    "'lender net-of-fee' 정의 문구는 확보하지 못했다. 구조적 정황: OKX Savings는 P2P 매칭 "
    "시장으로 보인다(인증 필요한 POST set-lending-rate로 대여자가 직접 자기 대여금리를 호가하는 "
    "구조 -- 은행식 예대마진 상품이 아니라 매칭북), 공개 lending-rate-summary는 그 매칭 결과의 "
    "시스템 평균/추정/직전값 3종(avgRate/estRate/preRate)이다. Bitget 자체 대출북(borrower 측, "
    "earn/loan/public/coinInfos)의 ETH 7일 금리(1.26%)가 OKX ETH avgRate(1.50%)와 같은 자릿수인 "
    "점은 정황상 참고할 만하나 확증은 아니다(플랫폼·상품이 다름). 결론: 미확인(UNCONFIRMED). "
    "이 불확실성 자체가 E3(50% 할인) 후보의 존재 이유다 -- 확인이 아니라 대비책."
)


class FetchError(Exception):
    pass


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "strategy-arena-wave16/1.0"})
    return session


def _get_json(session: requests.Session, base_url: str, path: str, params: dict[str, Any], *, ok_code: str) -> dict[str, Any]:
    url = base_url + path
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=(5.0, 20.0))
            response.raise_for_status()
            payload = response.json()
            time.sleep(REQUEST_SLEEP_SECONDS)
            if not isinstance(payload, dict) or str(payload.get("code")) != ok_code:
                raise FetchError(f"non-ok payload from {path} {params}: code={payload.get('code')!r} msg={payload.get('msg')!r}")
            return payload
        except (requests.RequestException, ValueError, FetchError) as error:
            last_error = error
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_SLEEP_SECONDS * (2**attempt))
    raise FetchError(f"failed after {MAX_RETRIES} attempts: {path} {params}: {last_error}")


# ---------------------------------------------------------------------------
# 1. OKX lending-rate-summary
# ---------------------------------------------------------------------------


def fetch_okx_lending_summary(session: requests.Session) -> list[dict[str, Any]]:
    payload = _get_json(session, OKX_BASE_URL, OKX_LENDING_SUMMARY_PATH, {}, ok_code="0")
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise FetchError("OKX lending-rate-summary response had no data")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            ccy = str(row["ccy"])
            avg_rate = float(row["avgRate"])
        except (KeyError, TypeError, ValueError):
            continue

        def _opt_float(key: str) -> float | None:
            value = row.get(key)
            try:
                return float(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None

        parsed.append({"ccy": ccy, "avg_rate": avg_rate, "est_rate": _opt_float("estRate"), "pre_rate": _opt_float("preRate")})
    return parsed


def split_outliers(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, excluded) -- excluded = hardcoded named outliers (BETH, SPEC.md 발견) UNION anything
    over OUTLIER_APR_CAP (general fail-safe, see module docstring)."""
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        if row["ccy"] in HARDCODED_OUTLIER_CCY:
            excluded.append({**row, "reason": "hardcoded_spec_outlier (BETH: wrapped-staked-ETH promotional rate, not a spot altcoin the futures universe can hold)"})
        elif row["avg_rate"] > OUTLIER_APR_CAP:
            excluded.append({**row, "reason": f"avg_rate {row['avg_rate']:.4f} exceeds sanity cap {OUTLIER_APR_CAP:.2f} (100% APR)"})
        else:
            kept.append(row)
    return kept, excluded


# ---------------------------------------------------------------------------
# 2. Bitget current funding rate (ALL USDT-M perps, one call)
# ---------------------------------------------------------------------------


def fetch_bitget_current_funding(session: requests.Session) -> list[dict[str, Any]]:
    payload = _get_json(session, BITGET_BASE_URL, BITGET_CURRENT_FUNDING_PATH, {"productType": BITGET_PRODUCT_TYPE}, ok_code="00000")
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise FetchError("Bitget current-fund-rate response had no data")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            symbol = str(row["symbol"])
            funding_rate_8h = float(row["fundingRate"])
        except (KeyError, TypeError, ValueError):
            continue
        if not symbol.endswith("USDT"):
            continue
        parsed.append({"symbol": symbol, "funding_rate_8h": funding_rate_8h, "funding_apr": funding_rate_8h * 3.0 * 365.0})
    return parsed


# ---------------------------------------------------------------------------
# 3. Bitget loan coinInfos (borrower-side, secondary cross-check) + savings probe
# ---------------------------------------------------------------------------


def fetch_bitget_loan_coininfos(session: requests.Session) -> list[dict[str, Any]]:
    payload = _get_json(session, BITGET_BASE_URL, BITGET_LOAN_COININFOS_PATH, {}, ok_code="00000")
    data = payload.get("data")
    rows = data.get("loanInfos") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise FetchError("Bitget loan coinInfos response had no loanInfos")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            coin = str(row["coin"])
            rate_7d = float(row["rate7D"])
            rate_30d = float(row["rate30D"])
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append({"coin": coin, "rate_7d": rate_7d, "rate_30d": rate_30d})
    return parsed


def probe_bitget_savings(session: requests.Session) -> list[dict[str, Any]]:
    """Mirrors research/wave15_diverse/earn_apr.py's probe_endpoint convention: record the
    ACTUAL HTTP outcome (never assert access exists without trying). Expected/documented outcome
    as of 2026-07-23: HTTP 400 `{"code":"40006","msg":"Invalid ACCESS_KEY"}` for every route below
    -- i.e. Bitget's own spot-asset savings/lend-out product cannot be publicly confirmed to
    exist or be usable without an authenticated account (see LENDER_SIDE_UNCERTAINTY_NOTE /
    SPEC.md 치명적 한계 3)."""
    results: list[dict[str, Any]] = []
    for path in BITGET_SAVINGS_PROBE_PATHS:
        try:
            response = session.get(BITGET_BASE_URL + path, params={"coin": "USDT"}, timeout=(5.0, 15.0))
            time.sleep(REQUEST_SLEEP_SECONDS)
        except requests.RequestException as error:
            results.append({"path": path, "status_code": None, "outcome": "network_error", "detail": str(error)})
            continue
        if response.status_code == 400 and "40006" in response.text:
            outcome = "auth_required"
        elif response.status_code == 404:
            outcome = "not_found"
        elif response.status_code == 200:
            outcome = "unexpected_success"
        else:
            outcome = "other"
        results.append({"path": path, "status_code": response.status_code, "outcome": outcome, "detail": response.text[:200]})
    return results


# ---------------------------------------------------------------------------
# Symbol <-> ccy join helpers
# ---------------------------------------------------------------------------

_LEADING_DIGITS: Final = re.compile(r"^\d+")


def base_ccy_candidates(symbol: str) -> tuple[str, ...]:
    """Best-effort futures-symbol -> spot-ccy candidates, tried in order. Plain
    'strip trailing USDT' covers every wave13 L4 symbol observed in this run (including
    digit-leading real tickers like 1INCH, which OKX itself lists as '1INCH' -- not stripped
    further). The SECOND candidate (also strip a leading digit-run, e.g. Binance's
    1000-rebased-token convention '1000SHIBUSDT' -> 'SHIB') is a fallback for symbols whose
    exact stripped form has no OKX listing; if NEITHER matches, the caller marks the symbol as
    lending_available=False (fail-closed -- never invents a rate)."""
    if not symbol.endswith("USDT"):
        return (symbol,)
    stripped = symbol[:-4]
    candidates = [stripped]
    digit_stripped = _LEADING_DIGITS.sub("", stripped)
    if digit_stripped and digit_stripped != stripped:
        candidates.append(digit_stripped)
    return tuple(candidates)


def resolve_lending_apr(symbol: str, lending_by_ccy: dict[str, float]) -> tuple[float | None, str | None]:
    """Returns (avg_rate_or_None, matched_ccy_or_None). None means genuinely unavailable --
    engine16.py's own join treats that as lending_apr=0.0 AND lending_available=False (fail
    -closed; see configs16.py/engine16.py)."""
    for candidate in base_ccy_candidates(symbol):
        if candidate in lending_by_ccy:
            return lending_by_ccy[candidate], candidate
    return None, None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _pd_now_iso() -> str:
    import pandas as pd  # noqa: PANDAS_OK -- local import, matches collect_spreads.py's own pd_now_iso convention

    return pd.Timestamp.now(tz="UTC").isoformat()


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return float(ordered[mid]) if n % 2 else float((ordered[mid - 1] + ordered[mid]) / 2.0)


def collect_lending_snapshot(l4_symbols: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Fetches all three live sources, joins OKX lending + Bitget current funding onto
    `l4_symbols` (defaults to wave13's own frozen L4 200-symbol universe, read-only, via
    research.wave13_liquidity.universe_liquidity -- SPEC.md "top200 유니버스(L4 승계)"), and
    saves the result to cache/lending_snapshot.json. Never modifies wave13's own cache/results."""
    if l4_symbols is None:
        from research.wave13_liquidity import universe_liquidity as ul
        from research.wave16_duallayer.configs16 import L4_CONFIG

        l4_symbols = ul.verify_cache_and_load_symbols(L4_CONFIG)

    with _session() as session:
        okx_rows = fetch_okx_lending_summary(session)
        bitget_funding_rows = fetch_bitget_current_funding(session)
        bitget_loan_rows = fetch_bitget_loan_coininfos(session)
        savings_probe = probe_bitget_savings(session)

    okx_kept, okx_excluded = split_outliers(okx_rows)
    lending_by_ccy = {row["ccy"]: row["avg_rate"] for row in okx_kept}
    funding_by_symbol = {row["symbol"]: row for row in bitget_funding_rows}

    by_symbol: dict[str, dict[str, Any]] = {}
    for symbol in l4_symbols:
        lending_apr, matched_ccy = resolve_lending_apr(symbol, lending_by_ccy)
        funding_row = funding_by_symbol.get(symbol)
        by_symbol[symbol] = {
            "base_ccy_matched": matched_ccy,
            "lending_apr": lending_apr if lending_apr is not None else 0.0,
            "lending_available": lending_apr is not None,
            "bitget_funding_apr_current": funding_row["funding_apr"] if funding_row is not None else None,
            "bitget_funding_available": funding_row is not None,
        }

    kept_rates = [row["avg_rate"] for row in okx_kept]
    matched_funding_aprs = [row["bitget_funding_apr_current"] for row in by_symbol.values() if row["bitget_funding_available"]]
    n_with_lending = sum(1 for row in by_symbol.values() if row["lending_available"])

    payload: dict[str, Any] = {
        "collected_at_utc": _pd_now_iso(),
        "snapshot_limitation": (
            "단일 시점 스냅샷(대여이자·현재펀딩 둘 다). 대여이자는 과거 시계열이 공개되지 않아 "
            "이 스냅샷 하나가 전부다 -- SPEC.md 치명적 한계 1 참조. 과거 펀딩 시계열에 이 상수를 "
            "얹은 값은 추정이며 검증이 아니다."
        ),
        "okx_lending": {
            "source": OKX_BASE_URL + OKX_LENDING_SUMMARY_PATH,
            "raw_count": len(okx_rows),
            "kept_count": len(okx_kept),
            "excluded_outliers": okx_excluded,
            "median_avg_rate_excl_outliers": _median(kept_rates),
            "lender_side_uncertainty_note": LENDER_SIDE_UNCERTAINTY_NOTE,
            "rows": okx_kept,
        },
        "bitget_current_funding": {
            "source": BITGET_BASE_URL + BITGET_CURRENT_FUNDING_PATH,
            "raw_count": len(bitget_funding_rows),
            "l4_universe_matched_count": len(matched_funding_aprs),
            "l4_universe_median_apr": _median(matched_funding_aprs),
            "rows": bitget_funding_rows,
        },
        "bitget_loan_coininfos": {
            "source": BITGET_BASE_URL + BITGET_LOAN_COININFOS_PATH,
            "note": (
                "Bitget 자체 USDT등 담보대출 상품의 차입자(borrower) 지불금리 -- lender 수취 "
                "확인용이 아니라 OKX avgRate와의 자릿수 교차확인용 보조 데이터."
            ),
            "rows": bitget_loan_rows,
        },
        "bitget_savings_probe": {
            "endpoints_tried": list(BITGET_SAVINGS_PROBE_PATHS),
            "results": savings_probe,
            "conclusion": (
                "auth_required (또는 not_found) -- Bitget 자체 스팟자산 대여(savings) 상품의 "
                "공개(비인증) 확인 불가. SPEC.md 치명적 한계 3: 거래소간 분리(OKX 대여/Bitget "
                "실행) 노출을 해소하지 못했다 -- 미해결 리스크로 리포트에 명시."
            )
            if all(r["outcome"] in {"auth_required", "not_found"} for r in savings_probe)
            else "one or more probe endpoints returned an unexpected result -- see results[] detail",
        },
        "by_symbol": by_symbol,
        "universe_summary": {
            "n_symbols": len(l4_symbols),
            "n_with_lending": n_with_lending,
            "n_with_lending_pct": (n_with_lending / len(l4_symbols) * 100.0) if l4_symbols else 0.0,
            "n_with_current_funding": len(matched_funding_aprs),
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(CACHE_DIR / "lending_snapshot.json", payload)
    print(
        f"fetch: OKX lending {len(okx_kept)} ccy kept ({len(okx_excluded)} excluded), Bitget funding "
        f"{len(bitget_funding_rows)} symbols, L4 universe {len(l4_symbols)} symbols -> "
        f"{n_with_lending} with lending data ({payload['universe_summary']['n_with_lending_pct']:.1f}%) "
        f"-> {CACHE_DIR / 'lending_snapshot.json'}"
    )
    return payload


def load_lending_snapshot() -> dict[str, Any]:
    path = CACHE_DIR / "lending_snapshot.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing -- run collect_lending_snapshot() / `--stage fetch` first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CACHE_DIR",
    "HARDCODED_OUTLIER_CCY",
    "LENDER_SIDE_UNCERTAINTY_NOTE",
    "OUTLIER_APR_CAP",
    "FetchError",
    "base_ccy_candidates",
    "collect_lending_snapshot",
    "fetch_bitget_current_funding",
    "fetch_bitget_loan_coininfos",
    "fetch_okx_lending_summary",
    "load_lending_snapshot",
    "probe_bitget_savings",
    "resolve_lending_apr",
    "split_outliers",
]


if __name__ == "__main__":
    collect_lending_snapshot()
