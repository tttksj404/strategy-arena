# Wave-13 work item 1: measured Bitget spread/depth collection (SPEC.md "작업 1").
#
# This is the ONLY network-touching module in wave13_liquidity -- everything downstream
# (costs_measured.py's mapping, engine13.py's backtest) consumes cache/measured_spreads.json
# with no further network access, matching every prior wave's fail-closed cache convention
# (research/wave12_frontier/universe_frontier.py's own "fetch is network-bound, run/gates/
# report are cache-only" split).
#
# What this measures, per symbol, from Bitget's live USDT-M futures order book:
#   - half_spread_bp: (ask1-bid1)/2/mid * 1e4 -- the cost of crossing the touch once.
#   - walk_cost_bp: the USDT-notional-weighted-average execution price deviation from mid
#     (in bp) for a $45 order (this repo's own per-leg order size, see SPEC.md "공통 고정"),
#     walking up to 15 ask levels from merge-depth. When $45 fills entirely at level 1 this
#     is mathematically identical to half_spread_bp (the execution price IS ask1); it only
#     exceeds half_spread_bp when the order has to walk past level 1, which is exactly the
#     "$45 order can't just assume top-of-book pricing" case SPEC.md asks this wave to stop
#     assuming away.
#   - effective_slippage_bp = max(half_spread_bp, walk_cost_bp) -- SPEC.md's literal formula.
#
# Deliberately ASK-side only (SPEC.md: "depth_usdt_top = ask1 가격×수량"), not a separate
# bid-side walk -- a delta-neutral carry pair always has one leg crossing each side of its
# own book, and SPEC.md's own worked table (BTC/ETH/NVDA/AMC/BIO/AIO) is ask-side-only, so
# this module matches that literal instruction rather than inventing a bid+ask average that
# SPEC.md never asked for.
#
# Snapshot limitation (disclosed again in reporting13.py, not just here): this is ONE live
# read at collection time, not a time series. SPEC.md requires the x3 stress variant
# (costs_measured.py's stress_multiplier) specifically to compensate for this being a
# calm-market snapshot that would otherwise understate high-volatility-regime spreads.

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import requests

from research.wave1.common import save_json

BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR / "cache"

BITGET_BASE_URL: Final = "https://api.bitget.com"
TICKERS_PATH: Final = "/api/v2/mix/market/tickers"
DEPTH_PATH: Final = "/api/v2/mix/market/merge-depth"
PRODUCT_TYPE: Final = "usdt-futures"
DEPTH_LIMIT: Final = 15
DEPTH_PRECISION: Final = "scale0"  # finest/unmerged price ticks -- see module docstring

ORDER_SIZE_USDT: Final = 45.0  # this repo's own per-leg order size (SPEC.md "$45/$45")
REQUEST_SLEEP_SECONDS: Final = 0.12
MAX_RETRIES: Final = 3
MINIMUM_SYMBOLS: Final = 60

# SPEC.md's literal anchor ranks, extended with a log-spaced fill so the union always
# clears MINIMUM_SYMBOLS regardless of how many contracts Bitget currently lists.
EXPLICIT_ANCHOR_RANKS: Final[tuple[int, ...]] = (1, 2, 5, 10, 20, 35, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350)
FILL_TARGET_COUNT: Final = 70  # geomspace fill target BEFORE de-dup against explicit anchors; >= MINIMUM_SYMBOLS after dedup


class SpreadCollectionError(Exception):
    pass


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "strategy-arena-wave13/1.0"})
    return session


def _get_json(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = BITGET_BASE_URL + path
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=(5.0, 20.0))
            response.raise_for_status()
            payload = response.json()
            time.sleep(REQUEST_SLEEP_SECONDS)
            if not isinstance(payload, dict) or payload.get("code") != "00000":
                raise SpreadCollectionError(f"bitget error for {path} {params}: {payload.get('code')} {payload.get('msg')}")
            return payload
        except (requests.RequestException, ValueError, SpreadCollectionError) as error:
            last_error = error
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_SLEEP_SECONDS * (2**attempt))
    raise SpreadCollectionError(f"failed after {MAX_RETRIES} attempts: {path} {params}: {last_error}")


# ---------------------------------------------------------------------------
# Phase 1: rank every USDT-M futures contract by 24h quote volume (one call).
# ---------------------------------------------------------------------------


def fetch_ranked_tickers(session: requests.Session) -> list[dict[str, Any]]:
    payload = _get_json(session, TICKERS_PATH, {"productType": PRODUCT_TYPE})
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise SpreadCollectionError("bitget tickers response had no data")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            symbol = str(row["symbol"])
            usdt_volume_24h = float(row["usdtVolume"])
        except (KeyError, TypeError, ValueError):
            continue
        if not symbol.endswith("USDT") or usdt_volume_24h <= 0.0:
            continue
        parsed.append({"symbol": symbol, "usdt_volume_24h": usdt_volume_24h})
    parsed.sort(key=lambda item: item["usdt_volume_24h"], reverse=True)
    for rank, item in enumerate(parsed, start=1):
        item["rank"] = rank
    return parsed


def build_target_ranks(total: int, minimum: int = MINIMUM_SYMBOLS) -> list[int]:
    """SPEC.md: "랭크 1,2,5,10,20,35,50,75,100,125,150,175,200,250,300,350... 고르게",
    "최소 60개 심볼". Explicit anchors are always included (when they exist); a log-spaced
    fill on top guarantees the union clears `minimum` distinct ranks even after de-dup,
    spread evenly across the FULL available range (not just out to 350) so the tail beyond
    SPEC.md's last literal anchor is still sampled -- SPEC.md's "..." reads as "continue the
    pattern", not "stop at 350"."""
    if total < 1:
        raise SpreadCollectionError("no ranked contracts available to sample")
    anchors = {rank for rank in EXPLICIT_ANCHOR_RANKS if rank <= total}
    anchors.add(1)
    anchors.add(total)
    fill_count = max(minimum, FILL_TARGET_COUNT)
    geo = np.unique(np.round(np.geomspace(1, total, num=min(fill_count, total))).astype(int))
    anchors.update(int(rank) for rank in geo if 1 <= rank <= total)
    ranks = sorted(anchors)
    if len(ranks) < minimum:
        # extremely small universe (should not happen with a real exchange, but fail loud
        # rather than silently shipping < minimum) -- pad with every remaining rank in order.
        remaining = [rank for rank in range(1, total + 1) if rank not in anchors]
        ranks = sorted(set(ranks) | set(remaining[: minimum - len(ranks)]))
    return ranks


# ---------------------------------------------------------------------------
# Phase 2: per-symbol order-book measurement.
# ---------------------------------------------------------------------------


def fetch_depth(session: requests.Session, symbol: str) -> dict[str, Any]:
    payload = _get_json(
        session,
        DEPTH_PATH,
        {"symbol": symbol, "productType": PRODUCT_TYPE, "precision": DEPTH_PRECISION, "limit": str(DEPTH_LIMIT)},
    )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise SpreadCollectionError(f"bitget depth response for {symbol} had no data")
    return data


def _parse_levels(raw_levels: Any) -> list[tuple[float, float]]:
    levels: list[tuple[float, float]] = []
    if not isinstance(raw_levels, list):
        return levels
    for level in raw_levels:
        try:
            price, size = float(level[0]), float(level[1])
        except (IndexError, TypeError, ValueError):
            continue
        if price > 0.0 and size > 0.0:
            levels.append((price, size))
    return levels


def compute_walk_cost_bp(ask_levels: list[tuple[float, float]], mid: float, order_size_usdt: float = ORDER_SIZE_USDT) -> tuple[float, bool, float]:
    """Notional-weighted-average execution price deviation from `mid`, in bp, for an
    order_size_usdt BUY that walks up through `ask_levels` (best price first). If the full
    15 levels merge-depth returns still don't cover order_size_usdt, the shortfall is
    conservatively assumed to fill at the WORST level actually observed (fail-closed --
    never assumes the order gets cheaper fills than what was actually quoted), and
    `insufficient_depth=True` is returned so the caller can flag it (SPEC.md's own AIO/rank
    ~300 example: "$18 depth < $45 주문"). Returns (walk_cost_bp, insufficient_depth,
    filled_fraction)."""
    if not ask_levels or mid <= 0.0:
        return float("nan"), True, 0.0
    remaining = order_size_usdt
    weighted_price_sum = 0.0
    filled_usdt = 0.0
    for price, size in ask_levels:
        if remaining <= 1e-12:
            break
        level_usdt = price * size
        take_usdt = min(level_usdt, remaining)
        weighted_price_sum += price * take_usdt
        filled_usdt += take_usdt
        remaining -= take_usdt
    insufficient = remaining > 1e-9
    if insufficient:
        worst_price = ask_levels[-1][0]
        weighted_price_sum += worst_price * remaining
        filled_usdt += remaining
    avg_execution_price = weighted_price_sum / filled_usdt if filled_usdt > 0.0 else mid
    walk_cost_bp = (avg_execution_price - mid) / mid * 1.0e4
    filled_fraction = min(1.0, (order_size_usdt - max(remaining, 0.0)) / order_size_usdt) if insufficient else 1.0
    return float(walk_cost_bp), insufficient, float(filled_fraction)


def measure_symbol(session: requests.Session, symbol: str, rank: int, usdt_volume_24h: float) -> dict[str, Any] | None:
    try:
        depth = fetch_depth(session, symbol)
    except SpreadCollectionError:
        return None
    asks = _parse_levels(depth.get("asks"))
    bids = _parse_levels(depth.get("bids"))
    if not asks or not bids:
        return None
    ask1_price, ask1_size = asks[0]
    bid1_price, _bid1_size = bids[0]
    if ask1_price <= bid1_price:
        return None  # crossed/invalid book -- skip rather than record a nonsensical negative spread
    mid = (ask1_price + bid1_price) / 2.0
    half_spread_bp = (ask1_price - bid1_price) / 2.0 / mid * 1.0e4
    depth_usdt_top = ask1_price * ask1_size
    walk_cost_bp, insufficient_depth, filled_fraction = compute_walk_cost_bp(asks, mid, ORDER_SIZE_USDT)
    effective_slippage_bp = max(half_spread_bp, walk_cost_bp)
    return {
        "symbol": symbol,
        "rank": rank,
        "usdt_volume_24h": usdt_volume_24h,
        "mid": mid,
        "bid1": bid1_price,
        "ask1": ask1_price,
        "half_spread_bp": half_spread_bp,
        "depth_usdt_top": depth_usdt_top,
        "order_size_usdt": ORDER_SIZE_USDT,
        "walk_cost_bp": walk_cost_bp,
        "effective_slippage_bp": effective_slippage_bp,
        "insufficient_depth": insufficient_depth,
        "filled_fraction_at_15_levels": filled_fraction,
        "ask_levels_returned": len(asks),
        "ts": depth.get("ts"),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------


def collect_measured_spreads(minimum_symbols: int = MINIMUM_SYMBOLS) -> dict[str, Any]:
    with _session() as session:
        ranked = fetch_ranked_tickers(session)
        total = len(ranked)
        by_rank = {item["rank"]: item for item in ranked}
        target_ranks = build_target_ranks(total, minimum_symbols)
        print(f"collect: {total} live USDT-M contracts on Bitget; sampling {len(target_ranks)} target ranks (min={minimum_symbols})")

        measurements: list[dict[str, Any]] = []
        failed_ranks: list[int] = []
        for target_rank in target_ranks:
            # If the exact target rank's own symbol fails (delisted mid-run, empty book,
            # etc.), fail over to the next-lower-volume rank rather than silently shrinking
            # the sample below `minimum_symbols` -- bounded to a handful of hops so one dead
            # patch of the ranking can't spiral into scanning the whole exchange.
            resolved = None
            for hop in range(6):
                candidate_rank = target_rank + hop
                if candidate_rank > total or candidate_rank in {m["rank"] for m in measurements}:
                    continue
                candidate = by_rank.get(candidate_rank)
                if candidate is None:
                    continue
                result = measure_symbol(session, candidate["symbol"], candidate_rank, candidate["usdt_volume_24h"])
                if result is not None:
                    resolved = result
                    break
            if resolved is None:
                failed_ranks.append(target_rank)
                continue
            measurements.append(resolved)
            if len(measurements) % 10 == 0:
                print(f"collect: {len(measurements)}/{len(target_ranks)} symbols measured (last={resolved['symbol']} rank={resolved['rank']})")

    measurements.sort(key=lambda item: item["rank"])
    payload = {
        "collected_at_utc": pd_now_iso(),
        "source": "bitget",
        "product_type": PRODUCT_TYPE,
        "depth_endpoint": DEPTH_PATH,
        "depth_limit": DEPTH_LIMIT,
        "depth_precision": DEPTH_PRECISION,
        "order_size_usdt": ORDER_SIZE_USDT,
        "total_live_contracts": total,
        "target_rank_count": len(target_ranks),
        "measured_count": len(measurements),
        "failed_ranks": failed_ranks,
        "snapshot_limitation": (
            "단일 시점 스냅샷(평온장). 고펀딩/고변동기 스프레드 확대를 반영하지 않음 -- "
            "costs_measured.py의 stress_multiplier=3.0 변형으로만 보정한다. 과거 시점 "
            "백테스트에 이 실측을 균일 적용하는 것은 근사다 (SPEC.md 한계 절 참조)."
        ),
        "measurements": measurements,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(CACHE_DIR / "measured_spreads.json", payload)
    print(
        f"collect: done. {len(measurements)} symbols measured, {len(failed_ranks)} target ranks unresolved "
        f"-> {CACHE_DIR / 'measured_spreads.json'}"
    )
    return payload


def pd_now_iso() -> str:
    import pandas as pd  # noqa: PANDAS_OK -- local import keeps this module's top-level import list network/collection-focused

    return pd.Timestamp.now(tz="UTC").isoformat()


def load_measured_spreads() -> dict[str, Any]:
    path = CACHE_DIR / "measured_spreads.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing -- run collect_measured_spreads() / `--stage collect` first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CACHE_DIR",
    "MINIMUM_SYMBOLS",
    "ORDER_SIZE_USDT",
    "SpreadCollectionError",
    "build_target_ranks",
    "collect_measured_spreads",
    "compute_walk_cost_bp",
    "fetch_depth",
    "fetch_ranked_tickers",
    "load_measured_spreads",
    "measure_symbol",
]


if __name__ == "__main__":
    collect_measured_spreads()
