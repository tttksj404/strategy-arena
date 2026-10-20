# Wave-14 network-touching data collection. The ONLY module in wave14_multivenue that
# hits the network (mirrors research/wave13_liquidity/collect_spreads.py's own
# "collect is network-bound, run/gates/report are cache-only" split). Four jobs, all under
# SPEC.md's "신규 데이터 (검증 필요)" section:
#
#   1. Discover which of wave13 L4's 200 Binance symbols are ALSO tradable on Bybit (both
#      spot AND linear/USDT-perpetual listed -- the carry structure needs both legs on the
#      SAME venue) -- `discover_bybit_universe`. Deliberately restricted to L4's own 200,
#      not Bybit's full independent listing -- wave13 already found top200 is the symbol-
#      breadth peak and top358 hurts (SPEC.md "심볼 축 소진"); letting Bybit silently widen
#      the SYMBOL axis again would recontaminate the wave13 finding this wave is supposed to
#      hold fixed while it isolates the VENUE axis instead. One symbol
#      (HOMEUSDT, fundingInterval=60min) is additionally excluded purely for fetch
#      tractability -- a 1-hour funding cadence needs ~4x the funding/history API pages of
#      this universe's 4h/8h majority for the same historical span, for exactly one symbol's
#      worth of extra opportunity; excluded and logged, not silently dropped.
#   2. Fetch Bybit spot + linear daily klines and linear funding-rate history for that
#      universe, from ~2023-09-01 (a safety margin before Bybit's own ~2023-10-28 kline
#      floor observed at collection time) through FROZEN_END (2026-07-14, wave12/13's own
#      frozen boundary -- reused so the Binance and Bybit legs of every M-config end on
#      exactly the same day). A symbol whose Bybit fetch fails outright (bad response,
#      delisted mid-collection, etc.) is EXCLUDED and logged in cache/bybit_universe.json's
#      `fetch_failures`, never backfilled with an assumption (task contract: "Bybit 데이터
#      수집 실패 심볼은 가정으로 채우지 말고 제외 기록").
#   3. Measure Bybit order books (spot AND linear, separately) for that same universe,
#      using the EXACT SAME half-spread/$45-book-walk formula as
#      research.wave13_liquidity.collect_spreads (compute_walk_cost_bp is imported from
#      there, not reimplemented) -- feeds research/wave14_multivenue/costs_venue.py's
#      per-venue mapping fit.
#   4. Probe Hyperliquid's public /info endpoint for feasibility only (SPEC.md: "Hyperliquid
#      /info(fundingHistory) — 접근 가능 여부부터 프로브, 불가 시 제외 기록"). Hyperliquid
#      does not appear in ANY of SPEC.md's 8 frozen M-configs (only Binance/Bitget and Bybit
#      do), so this is a scoped feasibility record for a future wave, not a dataset this
#      wave's backtest consumes.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import threading
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import save_frame, save_json
from research.wave13_liquidity.collect_spreads import ORDER_SIZE_USDT, compute_walk_cost_bp
from research.wave13_liquidity.universe_liquidity import FROZEN_END

BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR / "cache"

BYBIT_BASE_URL: Final = "https://api.bybit.com"
INSTRUMENTS_PATH: Final = "/v5/market/instruments-info"
TICKERS_PATH: Final = "/v5/market/tickers"
KLINE_PATH: Final = "/v5/market/kline"
FUNDING_PATH: Final = "/v5/market/funding/history"
ORDERBOOK_PATH: Final = "/v5/market/orderbook"
HYPERLIQUID_URL: Final = "https://api.hyperliquid.xyz/info"

MIN_FUNDING_INTERVAL_MINUTES: Final = 240  # excludes sub-4h-cadence symbols; see module docstring
KLINE_START: Final = pd.Timestamp("2023-09-01T00:00:00Z")
FUNDING_START: Final = pd.Timestamp("2023-09-01T00:00:00Z")
END_TS: Final = FROZEN_END + pd.Timedelta(days=1)  # matches wave12's own END_MS = frozen_end + 1 day convention
KLINE_LIMIT: Final = 1000  # Bybit v5 kline max page size; (END_TS - KLINE_START) < 1000 rows so one request/symbol/category suffices
FUNDING_PAGE_LIMIT: Final = 200  # Bybit v5 funding-history max page size
MAX_FUNDING_PAGES: Final = 250  # safety cap; ~4h-cadence symbol over the full window needs ~35 pages, generous headroom
ORDERBOOK_LIMIT: Final = 50  # Bybit spot orderbook depth cap (linear allows more; 50 already exceeds wave13's own 15-level walk)

REQUEST_SLEEP_SECONDS: Final = 0.12
MAX_RETRIES: Final = 4
DEFAULT_MAX_WORKERS: Final = 4
MINIMUM_UNIVERSE_SYMBOLS: Final = 20  # fail loud if Bybit intersection collapses to near-nothing (probe result sanity floor)

_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": "strategy-arena-wave14/1.0"})
        _thread_local.session = session
    return _thread_local.session


class VenueFetchError(Exception):
    pass


def _get_json(path: str, params: dict[str, Any], base_url: str = BYBIT_BASE_URL) -> dict[str, Any]:
    session = _session()
    url = base_url + path
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=(5.0, 30.0))
            response.raise_for_status()
            payload = response.json()
            time.sleep(REQUEST_SLEEP_SECONDS)
            if not isinstance(payload, dict) or payload.get("retCode") != 0:
                raise VenueFetchError(f"bybit error {path} {params}: retCode={payload.get('retCode') if isinstance(payload, dict) else '?'} {payload.get('retMsg') if isinstance(payload, dict) else payload}")
            return payload
        except (requests.RequestException, ValueError, VenueFetchError) as error:
            last_error = error
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_SLEEP_SECONDS * (2**attempt) + 0.05)
    raise VenueFetchError(f"failed after {MAX_RETRIES} attempts: {path} {params}: {last_error}")


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Universe discovery.
# ---------------------------------------------------------------------------


def fetch_instruments(category: str) -> dict[str, dict[str, Any]]:
    payload = _get_json(INSTRUMENTS_PATH, {"category": category, "limit": 1000})
    rows = payload["result"]["list"]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("quoteCoin") != "USDT" or row.get("status") != "Trading":
            continue
        if category == "linear" and row.get("contractType") != "LinearPerpetual":
            continue
        out[row["symbol"]] = row
    return out


def discover_bybit_universe(l4_symbols: tuple[str, ...]) -> dict[str, Any]:
    spot = fetch_instruments("spot")
    linear = fetch_instruments("linear")
    both = set(spot) & set(linear)
    candidate = sorted(set(l4_symbols) & both)
    excluded_low_interval = sorted(sym for sym in candidate if int(linear[sym].get("fundingInterval", 0)) < MIN_FUNDING_INTERVAL_MINUTES)
    universe = tuple(sym for sym in candidate if sym not in set(excluded_low_interval))
    if len(universe) < MINIMUM_UNIVERSE_SYMBOLS:
        raise VenueFetchError(f"bybit universe collapsed to {len(universe)} symbols (< {MINIMUM_UNIVERSE_SYMBOLS}) -- probe looks broken, not just narrow")
    return {
        "probed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "l4_symbol_count": len(l4_symbols),
        "bybit_spot_usdt_count": len(spot),
        "bybit_linear_usdt_count": len(linear),
        "bybit_spot_and_linear_count": len(both),
        "l4_intersect_bybit_count": len(candidate),
        "excluded_low_funding_interval": excluded_low_interval,
        "min_funding_interval_minutes": MIN_FUNDING_INTERVAL_MINUTES,
        "universe": list(universe),
        "universe_count": len(universe),
        "funding_interval_minutes": {sym: int(linear[sym].get("fundingInterval", 0)) for sym in universe},
        "fetch_failures": [],  # populated by run_fetch_stage after the per-symbol fetch pass
    }


# ---------------------------------------------------------------------------
# Per-symbol market data (klines + funding history).
# ---------------------------------------------------------------------------


def fetch_kline_frame(symbol: str, category: str) -> pd.DataFrame:
    payload = _get_json(KLINE_PATH, {"category": category, "symbol": symbol, "interval": "D", "end": _ms(END_TS), "limit": KLINE_LIMIT})
    rows = payload["result"]["list"]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"], index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="ms", utc=True)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated()]
    return frame.loc[(frame.index >= KLINE_START) & (frame.index < END_TS)]


def fetch_funding_frame(symbol: str, start: pd.Timestamp = FUNDING_START) -> pd.DataFrame:
    """endTime walkback pagination -- same methodology
    research.validation.cross_venue_funding.fetch_bybit already validated (2024-01-01
    reachable), generalized here to a caller-supplied `start` (this module's own
    FUNDING_START is ~2 months earlier, for score-warmup headroom before the 2024-01-01
    backtest window) and to this module's own retry/cache plumbing rather than that
    script's standalone one (that file is read-only per the task's own "밖 수정 금지")."""
    start_ms = _ms(start)
    rows: list[tuple[int, float]] = []
    cursor: int | None = None
    for _ in range(MAX_FUNDING_PAGES):
        params: dict[str, Any] = {"category": "linear", "symbol": symbol, "limit": FUNDING_PAGE_LIMIT}
        if cursor is not None:
            params["endTime"] = cursor
        payload = _get_json(FUNDING_PATH, params)
        entries = payload["result"]["list"]
        page = [(int(entry["fundingRateTimestamp"]), float(entry["fundingRate"])) for entry in entries if "fundingRateTimestamp" in entry and "fundingRate" in entry]
        if not page:
            break
        rows.extend(page)
        oldest = min(timestamp for timestamp, _ in page)
        if oldest <= start_ms:
            break
        next_cursor = oldest - 1
        if cursor is not None and next_cursor >= cursor:
            raise VenueFetchError(f"bybit funding {symbol}: pagination did not advance")
        cursor = next_cursor
    else:
        raise VenueFetchError(f"bybit funding {symbol}: exceeded {MAX_FUNDING_PAGES} pages")
    if not rows:
        return pd.DataFrame(columns=["funding_rate"], index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.dropna().sort_values("timestamp").set_index("timestamp")
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame.loc[(frame.index >= start) & (frame.index < END_TS), ["funding_rate"]]


# ---------------------------------------------------------------------------
# Order-book spread/depth measurement -- same formula as
# research.wave13_liquidity.collect_spreads, imported (compute_walk_cost_bp), applied to
# Bybit's own order book instead of Bitget's.
# ---------------------------------------------------------------------------


def measure_orderbook(symbol: str, category: str, usdt_volume_24h: float) -> dict[str, Any] | None:
    payload = _get_json(ORDERBOOK_PATH, {"category": category, "symbol": symbol, "limit": ORDERBOOK_LIMIT})
    result = payload["result"]
    asks_raw, bids_raw = result.get("a", []), result.get("b", [])
    if not asks_raw or not bids_raw:
        return None
    try:
        ask_levels = [(float(p), float(q)) for p, q in asks_raw]
        bid_levels = [(float(p), float(q)) for p, q in bids_raw]
    except (TypeError, ValueError):
        return None
    ask1_price, ask1_size = ask_levels[0]
    bid1_price, _bid1_size = bid_levels[0]
    if ask1_price <= bid1_price or ask1_price <= 0.0:
        return None
    mid = (ask1_price + bid1_price) / 2.0
    half_spread_bp = (ask1_price - bid1_price) / 2.0 / mid * 1.0e4
    depth_usdt_top = ask1_price * ask1_size
    walk_cost_bp, insufficient_depth, filled_fraction = compute_walk_cost_bp(ask_levels, mid, ORDER_SIZE_USDT)
    effective_slippage_bp = max(half_spread_bp, walk_cost_bp)
    return {
        "symbol": symbol,
        "category": category,
        "usdt_volume_24h": usdt_volume_24h,
        "mid": mid,
        "half_spread_bp": half_spread_bp,
        "depth_usdt_top": depth_usdt_top,
        "order_size_usdt": ORDER_SIZE_USDT,
        "walk_cost_bp": walk_cost_bp,
        "effective_slippage_bp": effective_slippage_bp,
        "insufficient_depth": insufficient_depth,
        "filled_fraction_at_levels": filled_fraction,
        "ask_levels_returned": len(ask_levels),
    }


def fetch_all_tickers(category: str) -> dict[str, float]:
    payload = _get_json(TICKERS_PATH, {"category": category})
    rows = payload["result"]["list"]
    out: dict[str, float] = {}
    for row in rows:
        try:
            out[row["symbol"]] = float(row.get("turnover24h", 0.0))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Per-symbol orchestration (one worker task -- own thread-local session via _session()).
# ---------------------------------------------------------------------------


def _cache_paths(symbol: str) -> dict[str, Path]:
    return {
        "spot": CACHE_DIR / f"bybit_spot_{symbol}_1d.csv.gz",
        "linear": CACHE_DIR / f"bybit_linear_{symbol}_1d.csv.gz",
        "funding": CACHE_DIR / f"bybit_funding_{symbol}.csv.gz",
    }


def fetch_symbol_market_data(symbol: str, force: bool = False) -> dict[str, Any]:
    paths = _cache_paths(symbol)
    if not force and all(path.exists() for path in paths.values()):
        return {"symbol": symbol, "status": "cached"}
    try:
        spot = fetch_kline_frame(symbol, "spot")
        linear = fetch_kline_frame(symbol, "linear")
        funding = fetch_funding_frame(symbol)
        if spot.empty or linear.empty or funding.empty:
            return {"symbol": symbol, "status": "failed", "reason": f"empty frame(s): spot={len(spot)} linear={len(linear)} funding={len(funding)}"}
        save_frame(paths["spot"], spot)
        save_frame(paths["linear"], linear)
        save_frame(paths["funding"], funding)
        return {
            "symbol": symbol,
            "status": "ok",
            "spot_rows": len(spot),
            "linear_rows": len(linear),
            "funding_rows": len(funding),
            "spot_start": spot.index.min().isoformat(),
            "funding_start": funding.index.min().isoformat(),
        }
    except (VenueFetchError, requests.RequestException, ValueError, KeyError) as error:
        return {"symbol": symbol, "status": "failed", "reason": str(error)}


def run_market_data_fetch(universe: tuple[str, ...], force: bool = False, max_workers: int = DEFAULT_MAX_WORKERS) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_symbol_market_data, symbol, force): symbol for symbol in universe}
        done = 0
        for future in as_completed(futures):
            symbol = futures[future]
            results[symbol] = future.result()
            done += 1
            if done % 10 == 0 or done == len(universe):
                ok = sum(1 for r in results.values() if r["status"] in {"ok", "cached"})
                print(f"fetch: market data {done}/{len(universe)} symbols processed ({ok} ok/cached so far, last={symbol})")
    return results


def run_orderbook_measurement(universe: tuple[str, ...], max_workers: int = DEFAULT_MAX_WORKERS) -> dict[str, Any]:
    spot_turnover = fetch_all_tickers("spot")
    linear_turnover = fetch_all_tickers("linear")

    def _measure_both(symbol: str) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        spot_measurement = measure_orderbook(symbol, "spot", spot_turnover.get(symbol, float("nan")))
        linear_measurement = measure_orderbook(symbol, "linear", linear_turnover.get(symbol, float("nan")))
        return symbol, spot_measurement, linear_measurement

    spot_measurements: list[dict[str, Any]] = []
    linear_measurements: list[dict[str, Any]] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_measure_both, symbol): symbol for symbol in universe}
        done = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _, spot_m, linear_m = future.result()
            except (VenueFetchError, requests.RequestException, ValueError) as error:
                failed.append(f"{symbol}: {error}")
                continue
            if spot_m is not None:
                spot_measurements.append(spot_m)
            if linear_m is not None:
                linear_measurements.append(linear_m)
            done += 1
            if done % 20 == 0 or done == len(universe):
                print(f"fetch: orderbook measurement {done}/{len(universe)} symbols")
    spot_measurements.sort(key=lambda item: -item["usdt_volume_24h"] if item["usdt_volume_24h"] == item["usdt_volume_24h"] else 0.0)
    linear_measurements.sort(key=lambda item: -item["usdt_volume_24h"] if item["usdt_volume_24h"] == item["usdt_volume_24h"] else 0.0)
    return {
        "collected_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": "bybit",
        "order_size_usdt": ORDER_SIZE_USDT,
        "orderbook_limit": ORDERBOOK_LIMIT,
        "universe_count": len(universe),
        "measurement_failures": failed,
        "snapshot_limitation": (
            "단일 시점 스냅샷(수집 시각 시장 상황). research.wave13_liquidity와 동일하게 고변동기 "
            "스프레드 확대를 반영하지 않음 -- costs_venue.py의 stress_multiplier=3.0(engine14의 S5)로만 보정."
        ),
        "spot": {"measurements": spot_measurements, "measured_count": len(spot_measurements)},
        "linear": {"measurements": linear_measurements, "measured_count": len(linear_measurements)},
    }


# ---------------------------------------------------------------------------
# Hyperliquid feasibility probe -- see module docstring point 4. Not part of the M0-M7
# dataset; recorded for the record only.
# ---------------------------------------------------------------------------


def probe_hyperliquid() -> dict[str, Any]:
    session = requests.Session()
    session.headers.update({"User-Agent": "strategy-arena-wave14/1.0", "Content-Type": "application/json"})
    try:
        meta_response = session.post(HYPERLIQUID_URL, json={"type": "meta"}, timeout=(5.0, 20.0))
        meta_response.raise_for_status()
        universe = meta_response.json().get("universe", [])
        time.sleep(0.2)
        start_ms = _ms(pd.Timestamp("2024-01-01T00:00:00Z"))
        history_response = session.post(HYPERLIQUID_URL, json={"type": "fundingHistory", "coin": "BTC", "startTime": start_ms}, timeout=(5.0, 20.0))
        history_response.raise_for_status()
        history = history_response.json()
        accessible = True
        note = "meta + fundingHistory(BTC) 둘 다 200 OK로 접근 가능 확인."
    except (requests.RequestException, ValueError) as error:
        accessible, universe, history = False, [], []
        note = f"probe failed: {error}"
    history_list = history if isinstance(history, list) else []
    return {
        "probed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "endpoint": HYPERLIQUID_URL,
        "accessible": accessible,
        "note": note,
        "perp_universe_count": len(universe) if isinstance(universe, list) else 0,
        "funding_history_btc_sample_count": len(history_list),
        "funding_history_btc_sample_first": history_list[0] if history_list else None,
        "funding_history_btc_sample_last": history_list[-1] if history_list else None,
        "in_scope_for_backtest": False,
        "scope_note": (
            "SPEC.md의 8개 동결 구성(M0-M7)에는 Hyperliquid가 없다 (Binance/Bitget 단일 또는 "
            "+Bybit 뿐) -- 이 프로브는 '접근 가능 여부' 확인용으로, 접근 가능하더라도 이번 wave의 "
            "백테스트에는 사용하지 않는다. 향후 wave 후보로만 기록."
        ),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration (run_wave14.py --stage fetch).
# ---------------------------------------------------------------------------


def run_fetch_stage(l4_symbols: tuple[str, ...], force: bool = False, max_workers: int = DEFAULT_MAX_WORKERS) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"fetch: discovering Bybit universe against {len(l4_symbols)} wave13 L4 symbols...")
    universe_payload = discover_bybit_universe(l4_symbols)
    universe = tuple(universe_payload["universe"])
    print(f"fetch: Bybit universe = {len(universe)} symbols (spot+linear intersection with L4, >= {MIN_FUNDING_INTERVAL_MINUTES}min funding interval)")

    market_results = run_market_data_fetch(universe, force=force, max_workers=max_workers)
    failures = sorted(symbol for symbol, result in market_results.items() if result["status"] == "failed")
    ok_universe = tuple(sorted(symbol for symbol in universe if symbol not in set(failures)))
    universe_payload["fetch_failures"] = [{"symbol": symbol, "reason": market_results[symbol].get("reason", "unknown")} for symbol in failures]
    universe_payload["universe_after_fetch"] = list(ok_universe)
    universe_payload["universe_after_fetch_count"] = len(ok_universe)
    save_json(CACHE_DIR / "bybit_universe.json", universe_payload)
    if failures:
        print(f"fetch: WARNING -- {len(failures)} symbols failed Bybit market-data fetch, excluded (not backfilled): {failures[:10]}")

    print(f"fetch: measuring Bybit order books for {len(ok_universe)} symbols (spot + linear)...")
    spread_payload = run_orderbook_measurement(ok_universe, max_workers=max_workers)
    save_json(CACHE_DIR / "bybit_spreads.json", spread_payload)
    print(
        f"fetch: order-book measurement done. spot={spread_payload['spot']['measured_count']}, "
        f"linear={spread_payload['linear']['measured_count']}, failures={len(spread_payload['measurement_failures'])}"
    )

    print("fetch: probing Hyperliquid /info...")
    hyperliquid_payload = probe_hyperliquid()
    save_json(CACHE_DIR / "hyperliquid_probe.json", hyperliquid_payload)
    print(f"fetch: Hyperliquid accessible={hyperliquid_payload['accessible']} (out of scope for M0-M7 backtest either way)")

    return {
        "universe": universe_payload,
        "market_data_results": market_results,
        "spread_payload_summary": {"spot": spread_payload["spot"]["measured_count"], "linear": spread_payload["linear"]["measured_count"]},
        "hyperliquid": hyperliquid_payload,
    }


__all__ = [
    "CACHE_DIR",
    "END_TS",
    "FUNDING_START",
    "KLINE_START",
    "MIN_FUNDING_INTERVAL_MINUTES",
    "VenueFetchError",
    "discover_bybit_universe",
    "fetch_all_tickers",
    "fetch_funding_frame",
    "fetch_instruments",
    "fetch_kline_frame",
    "fetch_symbol_market_data",
    "measure_orderbook",
    "probe_hyperliquid",
    "run_fetch_stage",
    "run_market_data_fetch",
    "run_orderbook_measurement",
]
