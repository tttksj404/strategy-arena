# Wave-11 data acquisition, scoped to exactly the two axes that need data beyond what
# research/wave1's existing cache already has:
#
#   - Y4 (universe expansion): wave1's cache/universe.json enforces a 24-month funding
#     -history floor and stops as soon as 40 volume-ranked symbols pass. Y4 relaxes the
#     floor to 12 months and raises the target to 100 symbols, so it needs (a) the 30
#     already-cached-but-rejected symbols re-evaluated under the relaxed rule (zero
#     network cost -- their spot/fapi/funding CSVs are already on disk from wave1's own
#     probing pass, research/wave1/cache/ has 70 fully-cached candidate symbols even
#     though only 40 were selected), and (b) new candidates fetched further down the
#     current volume ranking until 100 valid symbols are found or a bounded candidate
#     budget is exhausted (whichever first -- see MAX_NEW_CANDIDATES_CHECKED_Y4).
#   - Y5 (8h rebalance): needs spot+perp OHLC at <=8h resolution for BTC/ETH/SOL.
#     research/wave6/cache/binance_fapi_{BTC,ETH,SOL}USDT_1h.csv.gz already has the perp
#     side. No cache anywhere in the repo has spot sub-daily data (wave6 never needed it
#     -- its strategies are single-leg), so this module fetches spot 1h for exactly
#     those three symbols.
#
# Y1/Y2/Y3/Y6 need no NEW symbols (baseline-40 universe), but see the data-quality note
# below: their spot *prices* are re-fetched here too, into wave11_yield's own cache.
#
# ---------------------------------------------------------------------------------------
# DISCOVERED DATA-QUALITY BUG (research/wave1/fetch_binance.py, out of this wave's scope
# to fix in place -- flagged for a dedicated follow-up, see spawn_task at the bottom of
# the wave11 report):
#
# research.wave1.fetch_binance.fetch_klines() hardcodes `"limit": 1500` for every request
# and treats `len(page) < 1500` as "no more data, stop paginating". That is correct for
# the fapi (perp) klines endpoint, which genuinely accepts limit=1500. It is WRONG for
# spot klines (`market="spot"`, `/api/v3/klines`), whose real server-side cap is 1000: a
# request for 1500 silently comes back with exactly 1000 rows, `len(page)=1000 < 1500`
# reads as "done", and pagination stops after a single page -- even though the symbol's
# real history is much longer. Verified against research/wave1/cache/: 32 of the 40
# universe symbols have `binance_spot_{symbol}_1d.csv.gz` frozen at exactly 1000 rows,
# ending on whatever calendar date happened to be 1000 days after that symbol's Binance
# listing (2022-05-27 for BTC/ETH/XRP/DOGE/ADA/LINK/LTC/XLM/TRX/FETUSDT/ETCUSDT, later for
# symbols listed more recently) -- NOT at the frozen_end (2026-07-14) every perp/funding
# file correctly reaches. `research.wave1.common.integrity_report` never catches this
# because it only checks internal consistency (monotonic/no-dup/no-gap/positive-volume)
# of whatever rows exist, not whether the series reaches the intended end date.
#
# Net effect on every prior carry-family wave (wave1 F1, wave2 W2, wave10 C1-C4) that
# consumes research.wave1.fam_funding.load_markets: for roughly 3/4 of the nominal
# "top-40 by volume" universe -- including BTC/ETH/XRP/DOGE/ADA -- the *spot* leg goes
# NaN (and the symbol therefore silently drops out of `eligible`) partway through the
# backtest, years before OOS (2025-10-01) even starts. This is NOT something this wave
# introduced and this wave does not repair research/wave1/cache/*.csv.gz or
# research/wave1/fetch_binance.py in place (outside this wave's "research/wave11_yield/
# 밖 수정 금지" contract) -- but it WOULD reproduce inside every fresh fetch this wave
# performs if it reused fetch_klines(market="spot") verbatim, which would make Y4's newly
# -added symbols (and Y5's freshly-fetched spot 1h) *silently* just as broken as the
# ones already in wave1's cache. Since none of those fetches have any preexisting
# baseline to preserve bit-for-bit compatibility with, there is no reason to inherit the
# bug going forward: `_fetch_spot_klines_paged` below is a small local, correctly
# -paginated replacement (real per-market page cap, continues for as long as a full page
# comes back) used for every spot fetch this module performs. wave11's own cache
# (research/wave11_yield/cache/binance_spot_*_1d.csv.gz) additionally holds a *corrected*
# refetch of all 40 baseline symbols' spot series (see `ensure_corrected_spot_daily`),
# and research/wave11_yield/engine_y.py prefers that corrected copy over
# research/wave1/cache's truncated one whenever both exist. Every wave11 candidate (Y1-Y6)
# therefore runs on the corrected spot series uniformly -- this is a shared data-quality
# fix applied identically underneath all 6 candidates, not a per-candidate axis change, so
# the SPEC's "바꾸는 건 6개 축뿐" contract (comparability *among* Y1-Y6) still holds. The
# wave10 C1 baseline this report compares against is reused as-already-computed (its own
# JSON result, not recomputed), so it still reflects the old truncated data; the report
# discloses this vintage mismatch explicitly rather than silently recomputing C1.
# ---------------------------------------------------------------------------------------

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import CACHE_DIR as WAVE1_CACHE_DIR
from research.wave1.common import PipelineError, SYMBOL_PATTERN, integrity_report, load_frame, load_json, request_json, save_frame, save_json, validate_symbol
from research.wave1.fetch_binance import (
    KLINE_COLUMNS,
    BinanceFundingRequest,
    FAPI_BASE,
    SPOT_BASE,
    cached_frame,
    exchange_symbols,
    fetch_exchange_info,
    fetch_funding,
    fetch_quote_volumes,
    fetch_spot_exchange_info,
    quote_volumes,
)

CACHE_DIR: Final = Path(__file__).resolve().parent / "cache"
START_MS: Final = int(pd.Timestamp("2019-09-01T00:00:00Z").timestamp() * 1000)  # matches wave1/run_wave1.py START_MS
END_MS: Final = int(pd.Timestamp("2026-07-15T00:00:00Z").timestamp() * 1000)  # matches wave1/run_wave1.py END_MS (frozen_end 2026-07-14 + 1d)
FROZEN_END: Final = pd.Timestamp("2026-07-14T00:00:00Z")  # matches research/wave1/cache/universe.json's frozen_end
HISTORY_CUTOFF_12MO: Final = pd.Timestamp("2025-07-15T00:00:00Z")  # Y4 axis: 24mo->12mo (wave1's 24mo cutoff was 2024-07-15, exactly one more year back)
TARGET_VALID_Y4: Final = 100
MAX_NEW_CANDIDATES_CHECKED_Y4: Final = 260  # bounded network budget; if exhausted before reaching 100, universe_y4.json discloses target_reached=False

MAJOR_SYMBOLS_Y5: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SPOT_PAGE_LIMIT: Final = 1000  # Binance spot klines' real server-side cap (see module docstring)


def _fetch_klines_paged_fixed(symbol: str, interval: str, start_ms: int, end_ms: int, market: str, session: requests.Session) -> pd.DataFrame:
    """Correctly-paginated replacement for research.wave1.fetch_binance.fetch_klines:
    same request shape and response parsing, but pages with a limit that matches the
    endpoint's real cap and continues for as long as a full page comes back (instead of
    assuming every endpoint's cap is 1500)."""
    validate_symbol(symbol)
    endpoint = "/fapi/v1/klines" if market == "fapi" else "/api/v3/klines"
    base = FAPI_BASE if market == "fapi" else SPOT_BASE
    limit = 1500 if market == "fapi" else SPOT_PAGE_LIMIT
    cursor = start_ms
    collected: list[list] = []
    while cursor <= end_ms:
        params: dict[str, str | int] = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": cursor, "endTime": end_ms}
        page = request_json(session, base + endpoint, params)
        if not isinstance(page, list) or not page:
            break
        rows = [row for row in page if isinstance(row, list)]
        collected.extend(rows)
        next_cursor = int(rows[-1][0]) + 1
        if next_cursor <= cursor or len(rows) < limit:
            break
        cursor = next_cursor
    trimmed = [row[: len(KLINE_COLUMNS)] for row in collected if len(row) >= len(KLINE_COLUMNS)]
    frame = pd.DataFrame(trimmed, columns=KLINE_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS[1:], index=pd.DatetimeIndex([], name="timestamp"))
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    result = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index().loc[lambda item: ~item.index.duplicated()]
    return result[result.index < pd.to_datetime(end_ms, unit="ms", utc=True)]


def _cached_spot(path: Path, interval: str, symbol: str, session: requests.Session, force: bool) -> pd.DataFrame:
    if path.exists() and not force:
        return load_frame(path)
    frame = _fetch_klines_paged_fixed(symbol, interval, START_MS, END_MS, "spot", session)
    save_frame(path, frame)
    return frame


def ensure_corrected_spot_daily(symbols: list[str], force: bool = False) -> dict[str, int]:
    """Refetches spot 1d for every given symbol with the corrected pager, into
    research/wave11_yield/cache/ (never touching research/wave1/cache/). Every wave11
    candidate's market loader prefers this copy over wave1's when both exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}
    with requests.Session() as session:
        for count, symbol in enumerate(symbols, start=1):
            path = CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz"
            frame = _cached_spot(path, "1d", symbol, session, force)
            rows[symbol] = len(frame)
            if count % 10 == 0 or count == len(symbols):
                print(f"fetch: corrected spot 1d {count}/{len(symbols)} done ({symbol} rows={len(frame)})")
    return rows


def fetch_spot_1h_majors(force: bool = False) -> dict[str, int]:
    """Y5 axis: spot 1h klines for BTC/ETH/SOL, the one data type genuinely missing
    from every existing cache in the repo (wave6 only ever fetched perp 1h)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}
    with requests.Session() as session:
        for symbol in MAJOR_SYMBOLS_Y5:
            path = CACHE_DIR / f"binance_spot_{symbol}_1h.csv.gz"
            frame = _cached_spot(path, "1h", symbol, session, force)
            rows[symbol] = len(frame)
            print(f"fetch: spot 1h {symbol} rows={len(frame)} range=({frame.index.min()}..{frame.index.max()})" if len(frame) else f"fetch: spot 1h {symbol} EMPTY")
    return rows


def _binance_triplet_ok(funding: pd.DataFrame, perp: pd.DataFrame, spot: pd.DataFrame) -> bool:
    if funding.empty or perp.empty or spot.empty:
        return False
    return (
        integrity_report(funding, timedelta(hours=8)).valid
        and integrity_report(perp, timedelta(days=1)).valid
        and integrity_report(spot, timedelta(days=1)).valid
    )


def _reevaluate_cached_symbol(symbol: str) -> dict:
    """Phase A of Y4: symbols wave1 already probed (spot/fapi/funding CSVs already on
    disk in research/wave1/cache/) but did NOT select into the 40 -- re-check them under
    Y4's relaxed rule (12mo history, no bitget cross-check) at zero network cost. This
    only checks FUNDING history + PERP/FUNDING integrity (both unaffected by the spot
    -truncation bug); the corrected spot series for any symbol that flips valid here is
    fetched afterwards by ensure_corrected_spot_daily, same as every other symbol."""
    spot_path = WAVE1_CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz"
    perp_path = WAVE1_CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz"
    funding_path = WAVE1_CACHE_DIR / f"binance_funding_{symbol}.csv.gz"
    if not (spot_path.exists() and perp_path.exists() and funding_path.exists()):
        return {"ok": False, "reason": "missing_cached_files", "source": "wave1_cache_reeval"}
    funding = load_frame(funding_path)
    perp = load_frame(perp_path)
    has_12mo = not funding.empty and funding.index.min() <= HISTORY_CUTOFF_12MO
    binance_ok = not funding.empty and not perp.empty and integrity_report(funding, timedelta(hours=8)).valid and integrity_report(perp, timedelta(days=1)).valid
    ok = bool(has_12mo and binance_ok)
    return {
        "ok": ok,
        "reason": "ok" if ok else ("insufficient_history" if not has_12mo else "integrity_fail"),
        "source": "wave1_cache_reeval",
        "history_start": funding.index.min().isoformat() if not funding.empty else None,
    }


def _fetch_and_evaluate_new_symbol(symbol: str, session: requests.Session, force: bool) -> dict:
    """Phase B of Y4: a symbol wave1 never probed. Funding is fetched first (cheapest
    reliable history-length signal, cached under wave11_yield/cache/, never
    wave1/cache/); perp+spot klines are only fetched if funding clears the 12mo floor.
    Spot uses the corrected pager (see module docstring); perp uses the existing
    fetch_klines (fapi is not affected by the truncation bug)."""
    from research.wave1.fetch_binance import BinanceKlineRequest, fetch_klines

    funding_path = CACHE_DIR / f"binance_funding_{symbol}.csv.gz"
    funding = cached_frame(funding_path, force, lambda: fetch_funding(BinanceFundingRequest(symbol, START_MS, END_MS), session))
    has_12mo = not funding.empty and funding.index.min() <= HISTORY_CUTOFF_12MO
    if not has_12mo:
        return {
            "ok": False,
            "reason": "insufficient_history",
            "source": "y4_new_fetch",
            "history_start": funding.index.min().isoformat() if not funding.empty else None,
        }
    perp_path = CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz"
    spot_path = CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz"
    perp = cached_frame(perp_path, force, lambda: fetch_klines(BinanceKlineRequest(symbol, "1d", START_MS, END_MS), session))
    spot = _cached_spot(spot_path, "1d", symbol, session, force)
    ok = _binance_triplet_ok(funding, perp, spot)
    return {
        "ok": ok,
        "reason": "ok" if ok else "integrity_fail",
        "source": "y4_new_fetch",
        "history_start": funding.index.min().isoformat(),
    }


def _ranked_candidates(session: requests.Session) -> list[str]:
    futures = exchange_symbols(fetch_exchange_info(session))
    spot = exchange_symbols(fetch_spot_exchange_info(session))
    volumes = quote_volumes(fetch_quote_volumes(session))
    # Defensive: Binance's live symbol lists occasionally include entries that don't
    # match the [A-Z0-9]{1,32} contract validate_symbol() enforces everywhere else in
    # this codebase (observed once during this wave's own fetch: a garbled/CJK symbol
    # name in the raw exchangeInfo payload). Filter here so one malformed entry can't
    # abort the whole ranking; validate_symbol() is still the authority used downstream.
    pool = {symbol for symbol in (futures & spot) if symbol.endswith("USDT") and SYMBOL_PATTERN.fullmatch(symbol)}
    return sorted(pool, key=lambda symbol: volumes.get(symbol, 0.0), reverse=True)


def expand_universe_y4(force: bool = False) -> dict:
    """Builds research/wave11_yield/cache/universe_y4.json: the 40 wave1-selected
    symbols (all trivially still qualify -- a 24mo history floor implies a 12mo one)
    plus every additional symbol that clears Y4's relaxed rule, up to TARGET_VALID_Y4.
    Bitget cross-correlation is deliberately NOT required (Y4's registered axis is
    "펀딩히스토리 요건 24mo->12mo + 볼륨 top-40->top-100" only; the wave11_yield engine,
    like wave10's, only ever reads Binance spot/fapi/funding CSVs -- bitget was purely a
    wave1 data-quality cross-check on a source the carry engine never touches). After the
    symbol list is settled, fetches a *corrected* (non-truncated) spot-1d series for every
    member -- see module docstring."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_universe = load_json(WAVE1_CACHE_DIR / "universe.json")
    if not isinstance(base_universe, dict) or not isinstance(base_universe.get("symbols"), list):
        raise RuntimeError("wave1 cache universe.json is invalid or missing; cannot expand for Y4")
    base_symbols: list[str] = [str(s) for s in base_universe["symbols"]]
    already_checked: set[str] = {str(s) for s in base_universe.get("integrity", {})}

    valid: list[str] = list(base_symbols)
    detail: dict[str, dict] = {}

    unselected = sorted(s for s in already_checked if s not in base_symbols)
    for symbol in unselected:
        result = _reevaluate_cached_symbol(symbol)
        detail[symbol] = result
        if result["ok"]:
            valid.append(symbol)
    print(f"fetch: y4 phase-a reeval done, {len(valid) - len(base_symbols)}/{len(unselected)} unselected symbols flipped valid under 12mo rule")

    checked_new = 0
    target_reached = len(valid) >= TARGET_VALID_Y4
    if not target_reached:
        with requests.Session() as session:
            ranked = _ranked_candidates(session)
            for symbol in ranked:
                if len(valid) >= TARGET_VALID_Y4 or checked_new >= MAX_NEW_CANDIDATES_CHECKED_Y4:
                    break
                if symbol in already_checked:
                    continue
                checked_new += 1
                try:
                    result = _fetch_and_evaluate_new_symbol(symbol, session, force)
                except (PipelineError, requests.RequestException) as error:
                    result = {"ok": False, "reason": f"fetch_error: {error}", "source": "y4_new_fetch"}
                detail[symbol] = result
                if result["ok"]:
                    valid.append(symbol)
                if checked_new % 10 == 0:
                    print(f"fetch: y4 phase-b checked {checked_new} new candidates, valid so far {len(valid)}/{TARGET_VALID_Y4}")
        target_reached = len(valid) >= TARGET_VALID_Y4

    print(f"fetch: y4 refetching corrected (non-truncated) spot 1d for all {len(valid)} valid symbols")
    ensure_corrected_spot_daily(valid, force)

    payload = {
        "symbols": valid,
        "base_symbols_count": len(base_symbols),
        "added_symbols_count": len(valid) - len(base_symbols),
        "phase_a_unselected_checked": len(unselected),
        "phase_b_new_candidates_checked": checked_new,
        "target_valid": TARGET_VALID_Y4,
        "target_reached": target_reached,
        "criteria": {
            "history_cutoff_iso": HISTORY_CUTOFF_12MO.isoformat(),
            "history_requirement": "12 months (relaxed from wave1's 24 months)",
            "volume_rank_target": TARGET_VALID_Y4,
            "bitget_cross_check_required": False,
        },
        "detail": detail,
        "frozen_end": FROZEN_END.date().isoformat(),
    }
    save_json(CACHE_DIR / "universe_y4.json", payload)
    print(f"fetch: y4 universe done, {len(valid)}/{TARGET_VALID_Y4} valid symbols (target_reached={target_reached})")
    return payload


def run_fetch_stage(force: bool = False) -> None:
    fetch_spot_1h_majors(force)
    expand_universe_y4(force)


__all__ = [
    "CACHE_DIR",
    "MAJOR_SYMBOLS_Y5",
    "TARGET_VALID_Y4",
    "ensure_corrected_spot_daily",
    "expand_universe_y4",
    "fetch_spot_1h_majors",
    "run_fetch_stage",
]
