# Wave-12 universe construction: builds the point-in-time candidate pool that every one
# of U0-U6 draws its (breadth, funding-history-floor) membership from, and fetches the
# market data that pool needs.
#
# Two distinct "point-in-time" concerns live in this module, and they are NOT the same
# mechanism -- conflating them would either make the pipeline intractable (fully
# time-varying set membership, symbols entering/leaving mid-backtest) or reintroduce
# lookahead (using a single current-day snapshot for a per-day cost decision):
#
#   1. SET MEMBERSHIP (which symbols U0/U1/.../U6 even consider trading) is a STATIC list
#      per config, chosen once from a reference volume measured as of the frozen backtest
#      end date (FROZEN_END, 2026-07-14 -- identical to research/wave11_yield's Y4 cutoff,
#      so U0 reproduces Y4's universe bit-for-bit apart from the cost model). This mirrors
#      research.wave11_yield.fetch_y11.expand_universe_y4's own methodology exactly
#      (rank live candidates by volume, walk down keeping every one that clears a history
#      floor, until a target breadth is reached or the candidate list is exhausted) --
#      generalized here to seven different (breadth, history-months) pairs sharing one
#      probed candidate pool instead of Y4's single (100, 12mo) pair. This is a coarser,
#      known limitation (a symbol's SET membership does not itself walk-forward over the
#      backtest's 7 years the way its cost tier does below) -- disclosed in the wave12
#      report rather than silently assumed away; SPEC.md's own "볼륨 랭크" column reads as
#      exactly this kind of static breadth cap, the same way Y4's "top-100" did.
#   2. COST TIER + LIQUIDITY FLOOR (what each already-a-member symbol actually costs to
#      trade on a given day) genuinely walks forward day by day with no lookahead --
#      that piece lives in research/wave12_frontier/costs_tiered.py, using the raw
#      per-symbol daily quote_volume this module fetches (load_quote_volume_frame) as its
#      only input, recomputed fresh every day of the backtest via a shifted rolling
#      window. SPEC.md's explicit "미래 볼륨 사용 금지 — 이건 룩어헤드다" instruction is
#      about THIS piece, not the coarser set-membership piece above.
#
# Data-quality note (carried forward from research/wave11_yield/fetch_y11.py): Binance's
# spot klines endpoint caps at 1000 rows/request; research/wave1/fetch_binance.py used to
# hardcode 1500 (correct only for perp/fapi), silently truncating every spot series to
# 1000 rows. That bug is FIXED in research/wave1/fetch_binance.py itself as of this wave
# (see its fetch_klines docstring) -- so, unlike wave11, this module fetches spot directly
# via research.wave1.fetch_binance.fetch_klines with no local pagination workaround.  What
# is NOT fixed automatically is data already sitting in research/wave1/cache/*.csv.gz from
# before the fix: those files are frozen on disk and stay truncated until something
# refetches them. ensure_verified_spot below is that something: for every symbol this wave
# actually uses, it loads whatever spot is already cached, checks its end date against
# that same symbol's own perp end date (SPOT_TRUNCATION_TOLERANCE_DAYS), and refetches
# with the corrected fetcher into research/wave12_frontier/cache/ if the gap is
# suspiciously large. research/wave11_yield/cache/'s ~100 symbols are already
# individually corrected (wave11 refetched them all) and pass this check without a new
# network call; research/wave1/cache/'s original ~70 mostly will not, and get refetched.

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path
import sys
from typing import Callable, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import CACHE_DIR as WAVE1_CACHE_DIR
from research.wave1.common import PipelineError, SYMBOL_PATTERN, integrity_report, load_frame, load_json, save_frame, save_json
from research.wave1.fam_funding import FundingMarket
from research.wave1.fetch_binance import (
    BinanceFundingRequest,
    BinanceKlineRequest,
    exchange_symbols,
    fetch_exchange_info,
    fetch_funding,
    fetch_klines,
    fetch_quote_volumes,
    fetch_spot_exchange_info,
    quote_volumes,
)
from research.wave12_frontier.configs12 import CONFIGS
from research.wave12_frontier.costs_tiered import ROLLING_WINDOW_DAYS

BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR / "cache"
WAVE11_CACHE_DIR: Final = BASE_DIR.parent / "wave11_yield" / "cache"

START_MS: Final = int(pd.Timestamp("2019-09-01T00:00:00Z").timestamp() * 1000)  # matches wave1/wave11's own window
END_MS: Final = int(pd.Timestamp("2026-07-15T00:00:00Z").timestamp() * 1000)  # frozen_end (2026-07-14) + 1 day
FROZEN_END: Final = pd.Timestamp("2026-07-14T00:00:00Z")

# Identical to research/wave11_yield/fetch_y11.py's HISTORY_CUTOFF_12MO -- U0 must
# reproduce Y4's universe-membership rule exactly (only the cost model may differ).
HISTORY_CUTOFF_12MO: Final = pd.Timestamp("2025-07-15T00:00:00Z")
HISTORY_CUTOFF_6MO: Final = pd.Timestamp("2026-01-15T00:00:00Z")
HISTORY_CUTOFF_3MO: Final = pd.Timestamp("2026-04-15T00:00:00Z")
HISTORY_CUTOFFS_MONTHS: Final[dict[float, pd.Timestamp]] = {
    12.0: HISTORY_CUTOFF_12MO,
    6.0: HISTORY_CUTOFF_6MO,
    3.0: HISTORY_CUTOFF_3MO,
}
LOOSEST_HISTORY_MONTHS: Final = min(HISTORY_CUTOFFS_MONTHS)  # 3.0 -- U6's floor; nothing looser is ever fetched

SPOT_TRUNCATION_TOLERANCE_DAYS: Final = 10  # the real bug produces year-scale gaps; this only guards against false negatives, not false positives on incidental short gaps
REFERENCE_VOLUME_WINDOW_DAYS: Final = ROLLING_WINDOW_DAYS  # 30d, same window as the daily point-in-time tier


# ---------------------------------------------------------------------------
# Cache resolution (CACHE_DIR -> wave11_yield's corrected cache -> wave1's original cache)
# ---------------------------------------------------------------------------


def _first_existing(filename: str) -> Path | None:
    for base in (CACHE_DIR, WAVE11_CACHE_DIR, WAVE1_CACHE_DIR):
        path = base / filename
        if path.exists():
            return path
    return None


def _resolved_or_fetch(filename: str, force: bool, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    existing = _first_existing(filename)
    if existing is not None and not force:
        return load_frame(existing)
    frame = loader()
    save_frame(CACHE_DIR / filename, frame)
    return frame


def _required_files(symbol: str) -> tuple[str, str, str]:
    return (f"binance_spot_{symbol}_1d.csv.gz", f"binance_fapi_{symbol}_1d.csv.gz", f"binance_funding_{symbol}.csv.gz")


def missing_market_files(symbols: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for symbol in symbols:
        missing.extend(name for name in _required_files(symbol) if _first_existing(name) is None)
    return missing


# ---------------------------------------------------------------------------
# Candidate discovery (walk order only -- current live volume; does not decide any
# backtest-time cost or eligibility, only which order to probe candidates in).
# ---------------------------------------------------------------------------


def _ranked_candidates(session: requests.Session) -> list[str]:
    futures = exchange_symbols(fetch_exchange_info(session))
    spot = exchange_symbols(fetch_spot_exchange_info(session))
    volumes = quote_volumes(fetch_quote_volumes(session))
    pool = {symbol for symbol in (futures & spot) if symbol.endswith("USDT") and SYMBOL_PATTERN.fullmatch(symbol)}
    return sorted(pool, key=lambda symbol: volumes.get(symbol, 0.0), reverse=True)


def _reference_volume_30d(perp: pd.DataFrame) -> float:
    daily = perp["quote_volume"].resample("1D").sum()
    window = daily.loc[:FROZEN_END].tail(REFERENCE_VOLUME_WINDOW_DAYS)
    return float(window.mean()) if len(window) else 0.0


# ---------------------------------------------------------------------------
# Phase 1: shared candidate pool (research/wave12_frontier/cache/candidate_pool.json).
# One probing pass serves every one of U0-U6 -- each just filters/ranks this same pool
# differently (symbols_for_breadth_history below), so a symbol already resolved for one
# config's fetch never needs a second network round-trip for another's.
# ---------------------------------------------------------------------------


def build_candidate_pool(force: bool = False) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    detail: dict[str, dict] = {}
    with requests.Session() as session:
        ranked = _ranked_candidates(session)
        total = len(ranked)
        for count, symbol in enumerate(ranked, start=1):
            try:
                funding = _resolved_or_fetch(
                    f"binance_funding_{symbol}.csv.gz", force, lambda: fetch_funding(BinanceFundingRequest(symbol, START_MS, END_MS), session)
                )
            except (PipelineError, requests.RequestException) as error:
                detail[symbol] = {"ok": False, "reason": f"funding_fetch_error: {error}"}
                continue
            if funding.empty:
                detail[symbol] = {"ok": False, "reason": "no_funding_history"}
                continue
            history_start = pd.Timestamp(funding.index.min())
            if history_start > HISTORY_CUTOFF_3MO:  # fails even U6's loosest floor -- never worth fetching perp/spot for
                detail[symbol] = {"ok": False, "reason": "insufficient_history", "history_start": history_start.isoformat()}
                continue
            try:
                perp = _resolved_or_fetch(
                    f"binance_fapi_{symbol}_1d.csv.gz",
                    force,
                    lambda: fetch_klines(BinanceKlineRequest(symbol=symbol, interval="1d", start_ms=START_MS, end_ms=END_MS, market="fapi"), session),
                )
            except (PipelineError, requests.RequestException) as error:
                detail[symbol] = {"ok": False, "reason": f"perp_fetch_error: {error}", "history_start": history_start.isoformat()}
                continue
            funding_ok = integrity_report(funding, timedelta(hours=8)).valid
            perp_ok = not perp.empty and integrity_report(perp, timedelta(days=1)).valid
            if not (funding_ok and perp_ok):
                detail[symbol] = {"ok": False, "reason": "integrity_fail", "history_start": history_start.isoformat()}
                continue
            months_history = (FROZEN_END - history_start).days / 30.4368  # descriptive only -- membership filters use the literal cutoff dates above
            detail[symbol] = {
                "ok": True,
                "history_start": history_start.isoformat(),
                "months_history": months_history,
                "reference_volume_30d_usdt": _reference_volume_30d(perp),
                "perp_end": pd.Timestamp(perp.index.max()).isoformat(),
            }
            if count % 20 == 0 or count == total:
                valid_so_far = sum(1 for info in detail.values() if info["ok"])
                print(f"fetch: pool {count}/{total} candidates examined, {valid_so_far} valid so far (last={symbol})")
    payload = {
        "frozen_end": FROZEN_END.date().isoformat(),
        "candidate_count": len(detail),
        "valid_count": sum(1 for info in detail.values() if info["ok"]),
        "history_cutoffs_iso": {str(months): cutoff.isoformat() for months, cutoff in HISTORY_CUTOFFS_MONTHS.items()},
        "reference_volume_window_days": REFERENCE_VOLUME_WINDOW_DAYS,
        "symbols": detail,
    }
    save_json(CACHE_DIR / "candidate_pool.json", payload)
    print(f"fetch: pool done, {payload['valid_count']}/{payload['candidate_count']} candidates valid at >= {LOOSEST_HISTORY_MONTHS:.0f}mo history")
    return payload


def load_candidate_pool() -> dict:
    path = CACHE_DIR / "candidate_pool.json"
    if not path.exists():
        raise RuntimeError("research/wave12_frontier/cache/candidate_pool.json missing -- run `--stage fetch` first")
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), dict):
        raise RuntimeError("research/wave12_frontier/cache/candidate_pool.json is invalid")
    return payload


# ---------------------------------------------------------------------------
# Phase 2: per-config membership (pure filter/sort over the pool -- no network).
# ---------------------------------------------------------------------------


def symbols_for_breadth_history(pool: dict, breadth: int | None, history_months: float) -> tuple[str, ...]:
    """Replicates research.wave11_yield.fetch_y11.expand_universe_y4's own selection
    rule, generalized: rank every symbol that clears `history_months` of funding history
    by its reference_volume_30d_usdt (measured as of FROZEN_END), keep the top `breadth`
    (or all of them if breadth is None -- U3's "unlimited, liquidity-floor-only" reading;
    bounded in practice by however many symbols in the whole Binance USDT perp market
    actually clear a 12-month history floor, not by an arbitrary extra cap this module
    imposes -- see the module docstring's point 1)."""
    if history_months not in HISTORY_CUTOFFS_MONTHS:
        raise ValueError(f"unregistered history_months floor: {history_months} (expected one of {sorted(HISTORY_CUTOFFS_MONTHS)})")
    cutoff = HISTORY_CUTOFFS_MONTHS[history_months]
    candidates = [
        (symbol, info["reference_volume_30d_usdt"])
        for symbol, info in pool["symbols"].items()
        if info.get("ok") is True and pd.Timestamp(info["history_start"]) <= cutoff
    ]
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    selected = ranked if breadth is None else ranked[:breadth]
    return tuple(symbol for symbol, _volume in selected)


def tier_reference_symbols(pool: dict) -> tuple[str, ...]:
    """The shared cross-sectional pool costs_tiered's daily rank is computed against:
    every symbol that clears the loosest history floor used by ANY of U0-U6 (3 months,
    U6's), uncapped by breadth. Using one fixed reference set for every config's cost
    calculation is what makes "rank 1-50" mean the same 50 names in every one of the 7
    runs -- see this module's docstring."""
    return symbols_for_breadth_history(pool, None, LOOSEST_HISTORY_MONTHS)


def build_all_memberships(pool: dict) -> dict[str, tuple[str, ...]]:
    return {config.candidate.candidate_id: symbols_for_breadth_history(pool, config.breadth, config.history_months) for config in CONFIGS}


# ---------------------------------------------------------------------------
# Phase 3: verified (non-truncated) spot, fetched only for symbols actually selected
# into at least one config's membership.
# ---------------------------------------------------------------------------


def _spot_is_truncated(spot: pd.DataFrame, reference_end: pd.Timestamp) -> bool:
    if spot.empty:
        return True
    return (reference_end - pd.Timestamp(spot.index.max())).days > SPOT_TRUNCATION_TOLERANCE_DAYS


def ensure_verified_spot(symbols: tuple[str, ...], force: bool = False) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    verification: dict[str, dict] = {}
    with requests.Session() as session:
        for count, symbol in enumerate(symbols, start=1):
            perp_path = _first_existing(f"binance_fapi_{symbol}_1d.csv.gz")
            if perp_path is None:
                verification[symbol] = {"source": "missing_perp", "truncated_after_check": True}
                continue
            perp_end = pd.Timestamp(load_frame(perp_path).index.max())
            existing_spot_path = _first_existing(f"binance_spot_{symbol}_1d.csv.gz")
            spot = load_frame(existing_spot_path) if (existing_spot_path is not None and not force) else pd.DataFrame()
            source = "none"
            if existing_spot_path is not None and not force:
                if existing_spot_path.parent == CACHE_DIR:
                    source = "wave12_cache"
                elif existing_spot_path.parent == WAVE11_CACHE_DIR:
                    source = "wave11_yield_corrected_cache"
                else:
                    source = "wave1_cache_unverified"
            truncated = _spot_is_truncated(spot, perp_end)
            # Refetch on ANY truncation, regardless of source -- a symbol resolved from
            # wave11_yield's "corrected" cache is not exempt from this run's own check:
            # wave11 corrected the pagination bug for the 100 symbols ITS OWN Y4 universe
            # used, but a gap can also be genuine (e.g. Binance delisting that symbol's
            # SPOT pair while its perp stayed listed) rather than the old pagination bug.
            # Attempting a fresh fetch either recovers real data the old cache missed, or
            # (if Binance's API itself now returns the same short series) confirms the gap
            # is a real listing/delisting fact, not a stale artifact -- either way this
            # wave's own cache ends up holding a freshly-verified copy, not a trusted-blind
            # carry-forward. See spot_truncation_remaining in universe_frontier.json /
            # the report's spot-verification table for whatever remains genuinely short
            # after this refetch attempt.
            needed_fetch = existing_spot_path is None or force or truncated
            if needed_fetch:
                pre_refetch_source = source
                spot = fetch_klines(BinanceKlineRequest(symbol=symbol, interval="1d", start_ms=START_MS, end_ms=END_MS, market="spot"), session)
                save_frame(CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz", spot)
                source = "refetched_corrected" if pre_refetch_source == "none" else f"refetched_corrected(was={pre_refetch_source})"
                truncated = _spot_is_truncated(spot, perp_end)
            verification[symbol] = {
                "source": source,
                "spot_end": pd.Timestamp(spot.index.max()).isoformat() if not spot.empty else None,
                "perp_end": perp_end.isoformat(),
                "gap_days": (perp_end - pd.Timestamp(spot.index.max())).days if not spot.empty else None,
                "truncated_after_check": truncated,
            }
            if count % 20 == 0 or count == len(symbols):
                print(f"fetch: spot verify {count}/{len(symbols)} ({symbol})")
    save_json(CACHE_DIR / "spot_verification.json", verification)
    return verification


# ---------------------------------------------------------------------------
# Top-level fetch-stage orchestration (used by run_wave12.py --stage fetch).
# ---------------------------------------------------------------------------


def run_fetch_stage(force: bool = False) -> dict:
    pool = build_candidate_pool(force)
    memberships = build_all_memberships(pool)
    tier_reference = tier_reference_symbols(pool)
    union_symbols = tuple(sorted(set(tier_reference).union(*memberships.values())))
    print(f"fetch: union of all 7 configs' memberships + tier-reference pool = {len(union_symbols)} symbols; verifying/refetching spot")
    verification = ensure_verified_spot(union_symbols, force)
    truncated_remaining = sorted(symbol for symbol, info in verification.items() if info.get("truncated_after_check"))
    payload = {
        "frozen_end": FROZEN_END.date().isoformat(),
        "history_cutoffs_iso": {str(months): cutoff.isoformat() for months, cutoff in HISTORY_CUTOFFS_MONTHS.items()},
        "memberships": {config_id: list(symbols) for config_id, symbols in memberships.items()},
        "membership_sizes": {config_id: len(symbols) for config_id, symbols in memberships.items()},
        "tier_reference_symbols": list(tier_reference),
        "tier_reference_size": len(tier_reference),
        "union_symbol_count": len(union_symbols),
        "spot_truncation_remaining": truncated_remaining,
    }
    save_json(CACHE_DIR / "universe_frontier.json", payload)
    print(f"fetch: wave12 universe done. sizes={payload['membership_sizes']}, spot_truncation_remaining={len(truncated_remaining)}")
    if truncated_remaining:
        print(f"fetch: WARNING -- {len(truncated_remaining)} symbols still look spot-truncated after refetch: {truncated_remaining[:8]}")
    return payload


def load_universe_frontier() -> dict:
    path = CACHE_DIR / "universe_frontier.json"
    if not path.exists():
        raise RuntimeError("research/wave12_frontier/cache/universe_frontier.json missing -- run `--stage fetch` first")
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("memberships"), dict):
        raise RuntimeError("research/wave12_frontier/cache/universe_frontier.json is invalid")
    return payload


def verify_cache_and_load_symbols(candidate_id: str) -> tuple[str, ...]:
    """Fail-closed cache check for a single config, mirroring
    research.wave11_yield.engine_y.verify_cache_and_load_symbols_y4: performs no network
    access, requires universe_frontier.json (written by `--stage fetch`) plus every
    symbol's three market files to already be on disk."""
    payload = load_universe_frontier()
    members = payload["memberships"].get(candidate_id)
    if members is None:
        raise RuntimeError(f"universe_frontier.json has no membership recorded for {candidate_id}")
    symbols = tuple(str(symbol) for symbol in members)
    missing = missing_market_files(symbols)
    if missing:
        raise RuntimeError(f"wave-12 cache incomplete for {candidate_id}: {', '.join(sorted(set(missing))[:8])}")
    return symbols


def verify_cache_and_load_tier_reference_symbols() -> tuple[str, ...]:
    payload = load_universe_frontier()
    symbols = tuple(str(symbol) for symbol in payload["tier_reference_symbols"])
    missing = [name for name in symbols if _first_existing(f"binance_fapi_{name}_1d.csv.gz") is None]
    if missing:
        raise RuntimeError(f"wave-12 tier-reference cache incomplete, missing perp for: {', '.join(missing[:8])}")
    return symbols


# ---------------------------------------------------------------------------
# Market/volume loading (run stage; cache-only, no network).
# ---------------------------------------------------------------------------


def load_markets_for_symbols(symbols: tuple[str, ...]) -> dict[str, FundingMarket]:
    markets: dict[str, FundingMarket] = {}
    for symbol in symbols:
        spot_name, perp_name, funding_name = _required_files(symbol)
        spot_path, perp_path, funding_path = _first_existing(spot_name), _first_existing(perp_name), _first_existing(funding_name)
        if spot_path is not None and perp_path is not None and funding_path is not None:
            markets[symbol] = FundingMarket(spot=load_frame(spot_path), perp=load_frame(perp_path), funding=load_frame(funding_path)["funding_rate"])
    return markets


@lru_cache(maxsize=4)
def load_quote_volume_frame(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Daily perp quote_volume per symbol -- the sole input to
    research.wave12_frontier.costs_tiered's rolling point-in-time rank. Perp (not spot)
    is used for consistency with how this module itself ranks candidates for set
    membership (_reference_volume_30d, also perp-sourced) and with
    research.wave11_yield.fetch_y11.expand_universe_y4's own volume source
    (fetch_quote_volumes hits Binance's futures 24hr ticker).

    Cached: this reads and resamples up to ~360 gzip'd CSVs (the whole tier-reference
    pool), and research/wave12_frontier/engine12.py calls it once per run_candidate call
    (14 times across all 7 configs' base+stress runs within a single `--stage run`
    process) with the EXACT SAME `symbols` tuple every time (the tier-reference pool is
    fixed for the whole pipeline run) -- recomputing it from disk on every call was pure
    waste. Safe to cache: the returned frame is never mutated in place anywhere in this
    codebase (costs_tiered's rolling/rank functions all return new frames)."""
    series: dict[str, pd.Series] = {}
    for symbol in symbols:
        perp_path = _first_existing(f"binance_fapi_{symbol}_1d.csv.gz")
        if perp_path is not None:
            perp = load_frame(perp_path)
            series[symbol] = perp["quote_volume"].resample("1D").sum()
    return pd.DataFrame(series).sort_index()


__all__ = [
    "CACHE_DIR",
    "FROZEN_END",
    "HISTORY_CUTOFFS_MONTHS",
    "LOOSEST_HISTORY_MONTHS",
    "SPOT_TRUNCATION_TOLERANCE_DAYS",
    "build_all_memberships",
    "build_candidate_pool",
    "ensure_verified_spot",
    "load_candidate_pool",
    "load_markets_for_symbols",
    "load_quote_volume_frame",
    "load_universe_frontier",
    "missing_market_files",
    "run_fetch_stage",
    "symbols_for_breadth_history",
    "tier_reference_symbols",
    "verify_cache_and_load_symbols",
    "verify_cache_and_load_tier_reference_symbols",
]
