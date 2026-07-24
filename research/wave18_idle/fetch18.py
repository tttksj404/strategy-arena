# Wave-18 work item: OKX `lending-rate-history` collection for USDT specifically (I1's
# idle-capital lending candidate). wave17_lending_verified's OWN target universe explicitly
# EXCLUDES USDT (verified: research/wave17_lending_verified/cache/lending_realized.json has
# 112 target ccys and no USDT entry -- USDT is the quote/margin currency of every pair in that
# wave's universe, never a `base_ccy_matched` alt, so wave17 never had a reason to fetch it).
# This module is a NEW, wave18-local fetch that reuses (imports, does not reimplement)
# wave17's own fetch_lending_rate_history / summarize_history and wave16's own _session /
# fetch_okx_lending_summary / split_outliers -- the exact same OKX endpoints, same parsing,
# same rate-vs-lendingRate field split fetch17.py's own module docstring documents (`rate` =
# borrower-facing, `lendingRate` = lender-facing, which is what this wave needs).
#
# research/wave18_idle/ 밖 수정 금지 -- this module reads wave16/wave17's own modules
# (import only) and never writes to their cache/results directories.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import save_json
from research.wave16_duallayer import fetch_lending
from research.wave17_lending_verified import fetch17

BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR / "cache"

TARGET_CCY: Final = "USDT"
MARGIN_LOAN_QUOTA_PATH: Final = "/api/v5/public/interest-rate-loan-quota"  # public, unauthenticated -- same access tier as lending-rate-summary/history
MARGIN_SAMPLE_CCYS: Final[tuple[str, ...]] = ("BTC", "ETH", "SOL", "USDT")  # representative sample for I4's short-spot feasibility research, not the full ~200-symbol universe


def collect_usdt_lending() -> dict[str, Any]:
    """Fetches lending-rate-history(ccy=USDT, limit=100, ~4 days hourly) + a fresh
    lending-rate-summary (for the same avgRate-vs-lendingRate diagnostic ratio wave17
    reports), saves cache/usdt_lending.json. I1's actual constant APR is the CONSERVATIVE
    (minimum observed) `lending_rate` in this window -- SPEC.md task instruction "보수적으로
    하한 사용" -- computed downstream by load_usdt_lending_apr, not baked in here (this
    function only saves the raw + summarized observations, exactly wave17's own
    fetch-vs-recompute separation)."""
    with fetch_lending._session() as session:
        history = fetch17.fetch_lending_rate_history(session, TARGET_CCY)
        summary_rows = fetch_lending.fetch_okx_lending_summary(session)
    stats = fetch17.summarize_history(history)
    kept, excluded = fetch_lending.split_outliers(summary_rows)
    avg_rate_fresh = next((row["avg_rate"] for row in kept if row["ccy"] == TARGET_CCY), None)
    if avg_rate_fresh is None:
        avg_rate_fresh = next((row["avg_rate"] for row in summary_rows if row["ccy"] == TARGET_CCY), None)
    ratio_vs_avgrate = (
        stats["lending_rate_median"] / avg_rate_fresh if (stats.get("n_samples", 0) > 0 and avg_rate_fresh not in (None, 0.0)) else None
    )

    payload: dict[str, Any] = {
        "collected_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "ccy": TARGET_CCY,
        "source_history": fetch_lending.OKX_BASE_URL + fetch17.OKX_LENDING_HISTORY_PATH,
        "source_summary": fetch_lending.OKX_BASE_URL + fetch_lending.OKX_LENDING_SUMMARY_PATH,
        "history_limit": fetch17.HISTORY_LIMIT,
        "avg_rate_fresh": avg_rate_fresh,
        "ratio_lendingrate_median_over_avgrate_fresh": ratio_vs_avgrate,
        "summary_excluded_as_outlier": TARGET_CCY not in {row["ccy"] for row in kept} and TARGET_CCY in {row["ccy"] for row in excluded},
        "scope_note": (
            "wave17_lending_verified의 타깃 유니버스(112종, base_ccy_matched)는 USDT를 포함하지 "
            "않는다(USDT는 대상 페어들의 quote/margin 통화이지 base가 아니므로) -- 이 모듈은 wave18 "
            "I1 전용으로 USDT만 별도 수집한다. fetch17.py의 세션/파서 함수를 재사용(임포트)했을 뿐, "
            "wave17 캐시는 건드리지 않는다."
        ),
        "depth_limitation_note": (
            "limit=100, 시간당 1건 -> 약 4일. wave17과 동일한 제약: 과거(4일 이전) lendingRate "
            "시계열은 이 엔드포인트에서 얻을 수 없다 -- I1의 상수 적용은 이 4일 스냅샷 하나에 근거한다."
        ),
        **stats,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(CACHE_DIR / "usdt_lending.json", payload)
    print(
        f"fetch18: USDT lending-rate-history n_samples={stats.get('n_samples', 0)} "
        f"median={stats.get('lending_rate_median')} min={stats.get('lending_rate_min')} "
        f"max={stats.get('lending_rate_max')} avg_rate_fresh={avg_rate_fresh} -> {CACHE_DIR / 'usdt_lending.json'}"
    )
    return payload


def load_usdt_lending() -> dict[str, Any]:
    path = CACHE_DIR / "usdt_lending.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing -- run collect_usdt_lending() / `--stage fetch` first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_usdt_lending_apr(conservative: bool = True) -> float:
    """I1's constant APR input: the MINIMUM observed `lending_rate` in the ~4-day window when
    conservative=True (SPEC.md task instruction: '보수적으로 하한 사용'), else the median (the
    same field wave17's own F1 candidate uses as its point estimate). Raises if n_samples == 0
    (fetch failed/empty -- never silently substitutes a made-up number)."""
    payload = load_usdt_lending()
    if payload.get("n_samples", 0) == 0:
        raise RuntimeError("usdt_lending.json has n_samples=0 -- OKX returned no usable USDT history")
    field = "lending_rate_min" if conservative else "lending_rate_median"
    value = payload.get(field)
    if value is None:
        raise RuntimeError(f"usdt_lending.json missing {field}")
    return float(value)


def collect_margin_borrow_rates() -> dict[str, Any]:
    """I4 feasibility research (task instruction: '숏현물 차입 비용·가능성 확인 시도').
    OKX's PUBLIC (unauthenticated) margin interest-rate-loan-quota endpoint returns a `rate`
    field per currency; this wave could NOT confirm from public OKX documentation whether that
    field is already an annualized rate or an hourly figure needing x(24*365) to annualize --
    an OKX help-center article describes interest as 'Liability x (Annualized interest rate /
    365 / 24)', charged hourly, which is consistent with EITHER `rate` being that stored
    'Annualized interest rate' directly, or being the already-divided hourly figure. BOTH
    readings are saved here, unresolved -- reported as UNCONFIRMED, never silently picked one
    way (see gates18.I4_SHORT_SPOT_INFEASIBLE_REASONS, which does not depend on resolving this
    -- the structural blockers hold regardless of which reading is correct)."""
    response = requests.get(f"https://www.okx.com{MARGIN_LOAN_QUOTA_PATH}", timeout=(5.0, 20.0))
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or str(body.get("code")) != "0":
        raise RuntimeError(f"non-ok payload from {MARGIN_LOAN_QUOTA_PATH}: {body}")

    rows_by_ccy: dict[str, float] = {}
    for entry in body.get("data", []):
        for row in entry.get("basic", []):
            ccy, rate = row.get("ccy"), row.get("rate")
            if ccy is not None and rate is not None:
                rows_by_ccy[str(ccy)] = float(rate)

    sample: dict[str, Any] = {}
    for ccy in MARGIN_SAMPLE_CCYS:
        raw_rate = rows_by_ccy.get(ccy)
        sample[ccy] = {
            "raw_rate_field": raw_rate,
            "reading_a_raw_as_already_annualized_pct": (raw_rate * 100.0) if raw_rate is not None else None,
            "reading_b_hourly_times_8760_annualized_pct": (raw_rate * 8_760.0 * 100.0) if raw_rate is not None else None,
        }

    payload: dict[str, Any] = {
        "collected_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": f"https://www.okx.com{MARGIN_LOAN_QUOTA_PATH}",
        "unit_confirmed": False,
        "unit_note": (
            "OKX 도움말(borrowing-and-repaying-in-multi-currency-and-portfolio-margin-account-modes): "
            "'Interest = Liability exceeding the interest-free quota x (Annualized interest rate / "
            "365 / 24)', 매시 정산. `rate` 필드가 이 '연환산 금리' 원값인지 이미 시간당으로 나뉜 값인지 "
            "API 필드 문서 원문으로 확정하지 못했다 -- reading_a(원값을 그대로 연환산으로 해석)와 "
            "reading_b(시간당으로 보고 x8760 연환산) 둘 다 병기, 어느 쪽도 확정 아님(UNCONFIRMED)."
        ),
        "n_ccys_in_basic_tier": len(rows_by_ccy),
        "sample": sample,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(CACHE_DIR / "margin_borrow_rates.json", payload)
    print(f"fetch18: OKX margin loan-quota sample={sample} -> {CACHE_DIR / 'margin_borrow_rates.json'}")
    return payload


def load_margin_borrow_rates() -> dict[str, Any]:
    path = CACHE_DIR / "margin_borrow_rates.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing -- run collect_margin_borrow_rates() / `--stage fetch` first")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CACHE_DIR",
    "MARGIN_SAMPLE_CCYS",
    "TARGET_CCY",
    "collect_margin_borrow_rates",
    "collect_usdt_lending",
    "load_margin_borrow_rates",
    "load_usdt_lending",
    "load_usdt_lending_apr",
]


if __name__ == "__main__":
    collect_usdt_lending()
    collect_margin_borrow_rates()
