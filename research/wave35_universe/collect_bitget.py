#!/usr/bin/env python3
# Wave-35 data collection: 1h OHLCV for many Bitget USDT-M perpetuals.
#
# ---------------------------------------------------------------------------------------
# Why this exists
# ---------------------------------------------------------------------------------------
# Every wave from 30 to 34 reported "universe = BTC/ETH/SOL" as a hard data constraint, because
# 20x leverage needs 1h bars to resolve a 4.5% liquidation band and the only local 1h cache
# (research/wave6/cache) holds three symbols. That constraint was real but it was never the whole
# truth: Binance is region-blocked here (HTTP 451), yet Bitget answers normally and exposes
# 731 USDT-M perpetuals with 1h history paginating back past 2019-12.
#
# So the narrow universe was a limit of what had been COLLECTED, not of what is available. This
# module removes it. Widening the universe is also the one axis where widening is scientifically
# legitimate rather than p-hacking: symbols never previously touched supply a genuinely fresh
# holdout, whereas running more parameter search over the same three symbols would just be
# reopening a holdout that has already been used four times.
#
# ---------------------------------------------------------------------------------------
# Design constraints this respects
# ---------------------------------------------------------------------------------------
# * Public GET endpoints only. No key, no signing, no order/account endpoint (guardrail).
# * RESUMABLE: one gzip CSV per symbol, already-complete symbols are skipped, so collection can
#   be run in several short invocations instead of one long one.
# * history-candles caps `limit` at 200, so deep history needs ~285 requests per symbol for the
#   full span. Requests are paced and retried; a symbol that fails mid-way is left incomplete and
#   picked up on the next run rather than written half-formed under a final name.
# * Bars are stored as CLOSED bars only: the most recent (still forming) bar is dropped, matching
#   the t->t+1 discipline every engine in this repo uses.

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

BITGET: Final = "https://api.bitget.com"
PRODUCT_TYPE: Final = "usdt-futures"
PAGE_LIMIT: Final = 200  # hard API cap on history-candles

# Every granularity paginates back to at least 2020-12 (measured), but collection cost scales
# inversely with bar size: 200 bars per request means 1h needs ~285 requests per symbol while 1m
# needs ~14,500 (about 13 minutes per symbol). The dict below is the cost ladder that decides how
# wide each timeframe can practically go.
BAR_MINUTES: Final = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240, "1D": 1440}
DEFAULT_GRANULARITY: Final = "1H"
CACHE_DIR: Final = Path(__file__).resolve().parent / "cache"
MANIFEST_PATH: Final = CACHE_DIR / "manifest.json"
REQUEST_PAUSE: Final = 0.055  # ~18 req/s, inside Bitget's public market-data allowance
MAX_RETRIES: Final = 4
EARLIEST: Final = pd.Timestamp("2019-09-01", tz="UTC")


class CollectionError(RuntimeError):
    pass


def _get(path: str, params: dict[str, Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{BITGET}{path}", params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            payload = response.json()
            if payload.get("code") != "00000":
                raise CollectionError(f"{path} -> {payload.get('code')}: {payload.get('msg')}")
            return payload["data"]
        except (requests.RequestException, ValueError, CollectionError) as error:
            last_error = error
            time.sleep(0.4 * (attempt + 1))
    raise CollectionError(f"{path} failed after {MAX_RETRIES} attempts: {last_error}")


def liquid_symbols(min_volume_usdt: float) -> list[dict[str, Any]]:
    rows = _get("/api/v2/mix/market/tickers", {"productType": PRODUCT_TYPE})
    frame = pd.DataFrame(rows)
    frame["volume"] = pd.to_numeric(frame.get("usdtVolume"), errors="coerce").fillna(0.0)
    frame = frame[frame["volume"] >= min_volume_usdt].sort_values("volume", ascending=False)
    return [{"symbol": row.symbol, "volume_usdt_24h": float(row.volume)} for row in frame.itertuples()]


def _page(symbol: str, end_ms: int, granularity: str) -> pd.DataFrame:
    """One page of history ending strictly before `end_ms`."""
    rows = _get(
        "/api/v2/mix/market/history-candles",
        {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "granularity": granularity,
            "limit": PAGE_LIMIT,
            "endTime": end_ms,
        },
    )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"])
    frame = frame.astype({column: float for column in ("open", "high", "low", "close", "volume", "quote_volume")})
    frame["timestamp"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True)
    return frame.drop(columns=["ts"])


def collect_symbol(
    symbol: str,
    earliest: pd.Timestamp = EARLIEST,
    granularity: str = DEFAULT_GRANULARITY,
    max_pages: int = 400,
) -> pd.DataFrame:
    """Page backwards from now until the API stops returning older bars or `earliest` is reached."""
    bar_ms = BAR_MINUTES[granularity] * 60_000
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    collected: list[pd.DataFrame] = []
    seen_oldest = None
    for _page_index in range(max_pages):
        frame = _page(symbol, end_ms, granularity)
        time.sleep(REQUEST_PAUSE)
        if frame.empty:
            break
        oldest = frame["timestamp"].min()
        collected.append(frame)
        if seen_oldest is not None and oldest >= seen_oldest:
            break  # API stopped going further back
        seen_oldest = oldest
        if oldest <= earliest:
            break
        end_ms = int(oldest.timestamp() * 1000) - bar_ms
    if not collected:
        return pd.DataFrame()
    out = pd.concat(collected, ignore_index=True)
    out = out.drop_duplicates(subset="timestamp").sort_values("timestamp")
    out = out[out["timestamp"] >= earliest]
    # Drop the most recent bar: it may still be forming.
    if len(out):
        out = out.iloc[:-1]
    return out[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].reset_index(drop=True)


def symbol_path(symbol: str, granularity: str = DEFAULT_GRANULARITY) -> Path:
    return CACHE_DIR / f"bitget_{symbol}_{granularity}.csv.gz"


def has_history_before(symbol: str, cutoff: pd.Timestamp, granularity: str = DEFAULT_GRANULARITY) -> bool:
    """One cheap request: does this symbol have any bar older than `cutoff`?

    Used to build the universe, because 24h volume is a terrible proxy for history depth here --
    the highest-volume Bitget perpetuals right now are tokenised equities (SNDK, SOXL, MU,
    SKHYNIX) listed weeks ago with under 4,000 hourly bars. Screening on volume alone produced a
    universe that looked wide and was unusable.
    """
    rows = _get(
        "/api/v2/mix/market/history-candles",
        {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "granularity": granularity,
            "limit": 1,
            "endTime": int(cutoff.timestamp() * 1000),
        },
    )
    return bool(rows)


def discover_universe(
    min_volume_usdt: float, scan_symbols: int, min_history_start: pd.Timestamp
) -> list[dict[str, Any]]:
    """Volume screen first (liquidity is required for the measured cost model to mean anything),
    then a history-depth screen. Both are necessary; neither alone is sufficient."""
    candidates = liquid_symbols(min_volume_usdt)[:scan_symbols]
    kept: list[dict[str, Any]] = []
    for entry in candidates:
        try:
            deep = has_history_before(entry["symbol"], min_history_start)
        except CollectionError:
            deep = False
        time.sleep(REQUEST_PAUSE)
        if deep:
            kept.append(entry)
    print(f"유니버스 선별: 거래대금 ${min_volume_usdt/1e6:.0f}M+ {len(candidates)}종목 중 "
          f"{min_history_start.date()} 이전 이력 보유 **{len(kept)}종목**", flush=True)
    return kept


def collect(
    min_volume_usdt: float,
    limit_symbols: int,
    budget_seconds: float,
    granularity: str = DEFAULT_GRANULARITY,
    min_history_start: pd.Timestamp = pd.Timestamp("2024-01-01", tz="UTC"),
    only: tuple[str, ...] = (),
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if only:
        universe = [{"symbol": symbol, "volume_usdt_24h": float("nan")} for symbol in only]
    else:
        universe = discover_universe(min_volume_usdt, limit_symbols, min_history_start)
    manifest: dict[str, Any] = (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {"symbols": {}}
    )
    manifest.setdefault("symbols", {})
    started = time.time()
    done = skipped = failed = 0
    minimum_rows = int(180 * 24 * 60 / BAR_MINUTES[granularity])  # ~6 months of bars
    for entry in universe:
        symbol = entry["symbol"]
        if symbol_path(symbol, granularity).exists():
            skipped += 1
            continue
        if time.time() - started > budget_seconds:
            print("시간 예산 소진 — 남은 종목은 다음 실행에서 이어서 수집", flush=True)
            break
        try:
            frame = collect_symbol(symbol, granularity=granularity)
        except CollectionError as error:
            print(f"  {symbol}: 실패 {error}", flush=True)
            failed += 1
            continue
        if len(frame) < minimum_rows:
            print(f"  {symbol}: 이력 부족 {len(frame)}봉 (<{minimum_rows}) — 건너뜀", flush=True)
            manifest["symbols"].setdefault(symbol, {})[granularity] = {"rows": int(len(frame)), "usable": False}
            failed += 1
            continue
        with gzip.open(symbol_path(symbol, granularity), "wt", encoding="utf-8", newline="\n") as stream:
            frame.to_csv(stream, index=False)
        manifest["symbols"].setdefault(symbol, {})[granularity] = {
            "rows": int(len(frame)),
            "first": str(frame["timestamp"].iloc[0]),
            "last": str(frame["timestamp"].iloc[-1]),
            "volume_usdt_24h": entry["volume_usdt_24h"],
            "usable": True,
        }
        done += 1
        print(
            f"  {symbol:14s} {len(frame):6d}봉 {frame['timestamp'].iloc[0].date()} → "
            f"{frame['timestamp'].iloc[-1].date()} (${entry['volume_usdt_24h']/1e6:.0f}M)",
            flush=True,
        )
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["collected_at_utc"] = str(pd.Timestamp.now(tz="UTC"))
    manifest["source"] = "Bitget public /api/v2/mix/market/history-candles"
    manifest["min_volume_usdt_24h"] = min_volume_usdt
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    usable = sum(
        1 for entry in manifest["symbols"].values() if entry.get(granularity, {}).get("usable")
    )
    print(f"\n[{granularity}] 신규 {done} · 기존 {skipped} · 제외/실패 {failed} · "
          f"누적 사용가능 {usable}종목 ({time.time()-started:.0f}s)")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect Bitget perpetual OHLCV history at a chosen granularity (public, read-only)"
    )
    parser.add_argument("--granularity", default=DEFAULT_GRANULARITY, choices=sorted(BAR_MINUTES))
    parser.add_argument("--min-volume", type=float, default=1e6, help="minimum 24h USDT volume")
    parser.add_argument("--symbols", type=int, default=200, help="how many top-volume symbols to screen")
    parser.add_argument("--budget-seconds", type=float, default=900.0, help="stop starting new symbols after this")
    parser.add_argument("--min-history", default="2024-01-01", help="require a bar older than this date")
    parser.add_argument("--only", default="", help="comma-separated symbols, skipping discovery")
    args = parser.parse_args(argv)
    try:
        collect(
            args.min_volume,
            args.symbols,
            args.budget_seconds,
            granularity=args.granularity,
            min_history_start=pd.Timestamp(args.min_history, tz="UTC"),
            only=tuple(s for s in args.only.split(",") if s),
        )
    except (CollectionError, requests.RequestException) as error:
        print(f"수집 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
