# Wave-17 work item 1: OKX `lending-rate-history` collection (SPEC.md 방법 1). This is the
# ONLY network-touching module in wave17_lending_verified.
#
# wave16's own fetch_lending.py only ever called `lending-rate-summary` (`avgRate` -- a
# CURRENT-BORROWER-facing average, per this wave's own SPEC.md 발견). This module calls the
# SIBLING endpoint, `lending-rate-history`, which additionally returns `lendingRate` -- a
# lower, apparently lender-side number -- at hourly granularity for the last <=100 hours
# (~4 days; OKX does not expose more via this endpoint, so the "시계열 백테스트 불가" limit
# from wave16 SPEC.md 치명적 한계 1 is unchanged here, just re-confirmed for a different field).
#
# Target coin set: the wave16 snapshot's OWN universe -- read-only from
# research/wave16_duallayer/cache/lending_snapshot.json's `by_symbol`, taking the unique
# `base_ccy_matched` values where `lending_available` is true (112 ccys as of 2026-07-23's
# wave16 fetch). wave16's own cache is NEVER written to by this module.
#
# Reuses (imports, does not reimplement) research.wave16_duallayer.fetch_lending's own
# `_session`, `fetch_okx_lending_summary`, `split_outliers`, `FetchError`, `OKX_BASE_URL`,
# `OKX_LENDING_SUMMARY_PATH` -- the summary re-fetch below exists so this session's `avgRate`
# is TIME-MATCHED to this session's `lendingRate` history fetch (wave16's cached avgRate is
# from a different, earlier collection run; reusing it here would compare two different
# points in time under one "ratio" number, which SPEC.md's own honesty constraints forbid
# doing silently).
#
# `lending-rate-history` itself has no existing wrapper in fetch_lending.py (wave16 never
# called it), so `_get_json_paced` below is a deliberately-close adaptation of
# fetch_lending._get_json's own retry/backoff shape -- NOT a reuse-by-import, because this
# task's own instruction fixes a DIFFERENT inter-request sleep (0.12s, vs fetch_lending's
# 0.15s) that must actually take effect per request, which a shared module-level constant
# closed over in the other module cannot be overridden from here.

from __future__ import annotations

from pathlib import Path
import statistics
import sys
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import save_json
from research.wave16_duallayer import fetch_lending

BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR / "cache"
WAVE16_SNAPSHOT_PATH: Final = BASE_DIR.parent / "wave16_duallayer" / "cache" / "lending_snapshot.json"

OKX_LENDING_HISTORY_PATH: Final = "/api/v5/finance/savings/lending-rate-history"
HISTORY_LIMIT: Final = 100  # SPEC.md instruction: "코인당 limit=100" -- also OKX's own apparent per-call ceiling
REQUEST_SLEEP_SECONDS: Final = 0.12  # SPEC.md instruction: "sleep 0.12s"
MAX_RETRIES: Final = 3


class Wave17FetchError(Exception):
    pass


def _get_json_paced(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = fetch_lending.OKX_BASE_URL + path
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=(5.0, 20.0))
            response.raise_for_status()
            payload = response.json()
            time.sleep(REQUEST_SLEEP_SECONDS)
            if not isinstance(payload, dict) or str(payload.get("code")) != "0":
                raise Wave17FetchError(f"non-ok payload from {path} {params}: code={payload.get('code')!r} msg={payload.get('msg')!r}")
            return payload
        except (requests.RequestException, ValueError, Wave17FetchError) as error:
            last_error = error
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_SLEEP_SECONDS * (2**attempt))
    raise Wave17FetchError(f"failed after {MAX_RETRIES} attempts: {path} {params}: {last_error}")


# ---------------------------------------------------------------------------
# Target universe (read-only peek at wave16's own cache).
# ---------------------------------------------------------------------------


def _load_wave16_snapshot() -> dict[str, Any]:
    if not WAVE16_SNAPSHOT_PATH.exists():
        raise Wave17FetchError(f"{WAVE16_SNAPSHOT_PATH} missing -- run wave16's fetch stage first (research/wave16_duallayer/run_wave16.py --stage fetch)")
    import json

    return json.loads(WAVE16_SNAPSHOT_PATH.read_text(encoding="utf-8"))


def load_wave16_target_ccys() -> tuple[str, ...]:
    """Unique OKX ccy codes wave16 actually matched into its L4 200-symbol universe
    (`lending_available=True` entries' `base_ccy_matched`) -- read-only, raises if wave16's
    own fetch stage never ran (this module never fetches wave16's snapshot itself)."""
    wave16_snapshot = _load_wave16_snapshot()
    ccys = {
        info["base_ccy_matched"]
        for info in wave16_snapshot["by_symbol"].values()
        if info.get("lending_available") and info.get("base_ccy_matched")
    }
    return tuple(sorted(ccys))


def load_wave16_avg_rate_by_ccy() -> dict[str, float]:
    """wave16's OWN (stale, collected 2026-07-23) `lending-rate-summary` avgRate per ccy --
    exactly the number wave16's E0-E4 pipeline actually fed into `lending_apr` (see
    research/wave16_duallayer/fetch_lending.py: `lending_by_ccy = {row['ccy']: row['avg_rate']
    ...}`). Read-only; used ONLY as a labeled comparison point (SPEC.md: 'wave16 자체 대비 몇 %
    과대평가였나'), never as this wave's own F1 input (F1 uses THIS session's realized
    lendingRate directly, not any avgRate)."""
    wave16_snapshot = _load_wave16_snapshot()
    return {row["ccy"]: float(row["avg_rate"]) for row in wave16_snapshot["okx_lending"]["rows"]}


# ---------------------------------------------------------------------------
# Per-ccy history fetch + descriptive stats.
# ---------------------------------------------------------------------------


def _parse_history_rows(rows: Any) -> list[dict[str, Any]]:
    """Pure parsing step (no network) -- kept separate from `fetch_lending_rate_history` so it
    is directly unit-testable without mocking HTTP. `rate` (OKX's borrower-facing field) and
    `lendingRate` (the lender-facing field this whole wave exists to use instead) are read into
    DISTINCTLY NAMED keys (`rate` / `lending_rate`) right here, at the parse boundary -- every
    downstream consumer (summarize_history, recompute17.py) only ever reads `lending_rate` for
    yield purposes, never `rate` (see tests/test_wave17.py's own field-confusion regression
    test)."""
    if not isinstance(rows, list):
        return []
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            ts_ms = int(row["ts"])
            rate = float(row["rate"])
            lending_rate = float(row["lendingRate"])
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append({"ts_ms": ts_ms, "rate": rate, "lending_rate": lending_rate})
    parsed.sort(key=lambda item: item["ts_ms"])
    return parsed


def fetch_lending_rate_history(session: requests.Session, ccy: str) -> list[dict[str, Any]]:
    """One `lending-rate-history` call (limit=100, ~hourly cadence). Returns rows sorted
    ascending by timestamp: [{'ts_ms': int, 'rate': float, 'lending_rate': float}, ...].
    Raises Wave17FetchError on a non-ok payload; returns [] (never invents data) if OKX
    returns an ok payload with an empty/malformed `data` list for this ccy."""
    payload = _get_json_paced(session, OKX_LENDING_HISTORY_PATH, {"ccy": ccy, "limit": str(HISTORY_LIMIT)})
    return _parse_history_rows(payload.get("data"))


def _iso(ts_ms: int) -> str:
    return pd.Timestamp(ts_ms, unit="ms", tz="UTC").isoformat()


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure function (no network) -- descriptive stats over one ccy's raw history rows.
    SPEC.md 방법 3 (변동성 정량화): std/range/CV over `lending_rate` is the whole point of
    this function; `rate` stats are kept alongside only as a same-sample cross-check against
    the separately-fetched `lending-rate-summary` avgRate (see collect_lending_realized)."""
    if not history:
        return {"n_samples": 0}
    rates = pd.Series([row["rate"] for row in history], dtype=float)
    lending_rates = pd.Series([row["lending_rate"] for row in history], dtype=float)
    ts_values = [row["ts_ms"] for row in history]
    span_hours = (max(ts_values) - min(ts_values)) / 3_600_000.0
    lend_mean = float(lending_rates.mean())
    lend_std = float(lending_rates.std()) if len(lending_rates) > 1 else 0.0
    return {
        "n_samples": len(history),
        "span_hours": span_hours,
        "span_days": span_hours / 24.0,
        "earliest_ts_utc": _iso(min(ts_values)),
        "latest_ts_utc": _iso(max(ts_values)),
        "rate_median": float(rates.median()),
        "rate_mean": float(rates.mean()),
        "rate_std": float(rates.std()) if len(rates) > 1 else 0.0,
        "rate_min": float(rates.min()),
        "rate_max": float(rates.max()),
        "lending_rate_median": float(lending_rates.median()),
        "lending_rate_mean": lend_mean,
        "lending_rate_std": lend_std,
        "lending_rate_min": float(lending_rates.min()),
        "lending_rate_max": float(lending_rates.max()),
        "lending_rate_range": float(lending_rates.max() - lending_rates.min()),
        "lending_rate_cv": (lend_std / lend_mean) if lend_mean > 0.0 else None,
        "sample_first_3": history[:3],
        "sample_last_3": history[-3:],
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def _median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def collect_lending_realized(target_ccys: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Fetches (1) a FRESH `lending-rate-summary` (this session's avgRate, time-matched to
    the history fetch below -- see module docstring) and (2) `lending-rate-history` for every
    ccy in `target_ccys` (defaults to wave16's own matched-universe ccys). Saves
    cache/lending_realized.json. Never modifies wave16's own cache/results."""
    if target_ccys is None:
        target_ccys = load_wave16_target_ccys()
    wave16_avg_rate_by_ccy = load_wave16_avg_rate_by_ccy()

    with fetch_lending._session() as session:
        fresh_summary_rows = fetch_lending.fetch_okx_lending_summary(session)
        fresh_kept, fresh_excluded = fetch_lending.split_outliers(fresh_summary_rows)
        fresh_avg_by_ccy = {row["ccy"]: row["avg_rate"] for row in fresh_kept}

        by_ccy: dict[str, Any] = {}
        n_ok = 0
        n_failed = 0
        for ccy in target_ccys:
            try:
                history = fetch_lending_rate_history(session, ccy)
            except fetch_lending.FetchError as error:
                by_ccy[ccy] = {"history_available": False, "error": str(error), "n_samples": 0}
                n_failed += 1
                continue
            except Wave17FetchError as error:
                by_ccy[ccy] = {"history_available": False, "error": str(error), "n_samples": 0}
                n_failed += 1
                continue
            if not history:
                by_ccy[ccy] = {"history_available": False, "error": "empty/malformed data from OKX", "n_samples": 0}
                n_failed += 1
                continue
            stats = summarize_history(history)
            avg_rate_fresh = fresh_avg_by_ccy.get(ccy)
            avg_rate_wave16 = wave16_avg_rate_by_ccy.get(ccy)
            ratio_vs_fresh_avg = (
                stats["lending_rate_median"] / avg_rate_fresh if (avg_rate_fresh is not None and avg_rate_fresh > 0.0) else None
            )
            ratio_vs_wave16_avg = (
                stats["lending_rate_median"] / avg_rate_wave16 if (avg_rate_wave16 is not None and avg_rate_wave16 > 0.0) else None
            )
            ratio_vs_history_rate = (
                stats["lending_rate_median"] / stats["rate_median"] if stats["rate_median"] > 0.0 else None
            )
            by_ccy[ccy] = {
                "history_available": True,
                "avg_rate_fresh": avg_rate_fresh,  # THIS session's lending-rate-summary avgRate; None if ccy absent/excluded this session
                "avg_rate_wave16_snapshot": avg_rate_wave16,  # wave16's OWN stale avgRate (what E0-E4 actually used); None if ccy wasn't in wave16's kept rows
                "ratio_lendingrate_over_avgrate_fresh": ratio_vs_fresh_avg,
                "ratio_lendingrate_over_avgrate_wave16": ratio_vs_wave16_avg,  # "wave16이 실제로 얼마나 과대평가했나"에 답하는 지표
                "ratio_lendingrate_over_historyrate": ratio_vs_history_rate,  # same-call cross-check, avoids cross-session timing noise
                **stats,
            }
            n_ok += 1

    ratios_fresh = [row["ratio_lendingrate_over_avgrate_fresh"] for row in by_ccy.values() if row.get("ratio_lendingrate_over_avgrate_fresh") is not None]
    ratios_wave16 = [row["ratio_lendingrate_over_avgrate_wave16"] for row in by_ccy.values() if row.get("ratio_lendingrate_over_avgrate_wave16") is not None]
    ratios_history = [row["ratio_lendingrate_over_historyrate"] for row in by_ccy.values() if row.get("ratio_lendingrate_over_historyrate") is not None]
    span_days_all = [row["span_days"] for row in by_ccy.values() if row.get("history_available")]

    payload: dict[str, Any] = {
        "collected_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_history": fetch_lending.OKX_BASE_URL + OKX_LENDING_HISTORY_PATH,
        "source_summary": fetch_lending.OKX_BASE_URL + fetch_lending.OKX_LENDING_SUMMARY_PATH,
        "history_limit_per_ccy": HISTORY_LIMIT,
        "request_sleep_seconds": REQUEST_SLEEP_SECONDS,
        "target_universe_note": (
            "타깃 코인 = wave16 cache/lending_snapshot.json by_symbol 중 lending_available=True 항목의 "
            "base_ccy_matched 고유집합 (read-only 참조, wave16 파일은 손대지 않음)."
        ),
        "depth_limitation_note": (
            "limit=100, 시간당 1건 -> 약 4일(96~100시간)이 이 엔드포인트에서 얻을 수 있는 전부다. "
            "이는 wave16 SPEC.md 치명적 한계 1과 동일한 제약이며, 이 wave가 해소한 것이 아니다 -- "
            "avgRate를 lendingRate로 바꿔도 '시계열 백테스트 불가'는 그대로다."
        ),
        "n_target_ccys": len(target_ccys),
        "n_history_available": n_ok,
        "n_history_failed": n_failed,
        "ratio_median_across_universe_vs_fresh_avgrate": _median_or_none(ratios_fresh),
        "ratio_mean_across_universe_vs_fresh_avgrate": _mean_or_none(ratios_fresh),
        "ratio_min_vs_fresh_avgrate": min(ratios_fresh) if ratios_fresh else None,
        "ratio_max_vs_fresh_avgrate": max(ratios_fresh) if ratios_fresh else None,
        "ratio_median_across_universe_vs_wave16_avgrate": _median_or_none(ratios_wave16),
        "ratio_mean_across_universe_vs_wave16_avgrate": _mean_or_none(ratios_wave16),
        "ratio_median_across_universe_vs_history_rate": _median_or_none(ratios_history),
        "span_days_median": _median_or_none(span_days_all),
        "span_days_min": min(span_days_all) if span_days_all else None,
        "fresh_summary_excluded_outliers": fresh_excluded,
        "by_ccy": by_ccy,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(CACHE_DIR / "lending_realized.json", payload)
    print(
        f"fetch17: {len(target_ccys)} target ccys -> {n_ok} history OK / {n_failed} failed, "
        f"ratio_median vs fresh avgRate={payload['ratio_median_across_universe_vs_fresh_avgrate']}, "
        f"vs wave16 avgRate={payload['ratio_median_across_universe_vs_wave16_avgrate']}, "
        f"vs same-call rate={payload['ratio_median_across_universe_vs_history_rate']} "
        f"-> {CACHE_DIR / 'lending_realized.json'}"
    )
    return payload


def load_lending_realized() -> dict[str, Any]:
    path = CACHE_DIR / "lending_realized.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing -- run collect_lending_realized() / `--stage fetch` first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CACHE_DIR",
    "HISTORY_LIMIT",
    "REQUEST_SLEEP_SECONDS",
    "Wave17FetchError",
    "collect_lending_realized",
    "fetch_lending_rate_history",
    "load_lending_realized",
    "load_wave16_target_ccys",
    "summarize_history",
]


if __name__ == "__main__":
    collect_lending_realized()
