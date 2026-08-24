"""Paginated historical kline downloader for Bitget."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def download_klines_range(
    client: Any,
    *,
    market: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    page_limit: int = 200,
    sleep_between: float = 1.5,
    use_history: bool = False,
) -> list[dict[str, Any]]:
    """Paginate backward from end_ms to start_ms, return klines sorted ascending.

    When *use_history* is True, uses the ``history-candles`` endpoint which
    supports data older than ~31 days on Bitget.
    """
    all_klines: dict[int, dict[str, Any]] = {}
    cursor_end = end_ms
    pages = 0

    fetch = client.get_history_klines if use_history else client.get_klines

    while cursor_end > start_ms:
        batch = fetch(
            market=market,
            symbol=symbol,
            interval=interval,
            limit=page_limit,
            end_time=cursor_end,
        )
        if not batch:
            break
        for row in batch:
            ot = int(row["open_time"])
            if ot >= start_ms:
                all_klines[ot] = row
        oldest = min(int(row["open_time"]) for row in batch)
        if oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        pages += 1
        if pages % 10 == 0:
            print(f"  [{symbol}/{interval}] {pages} pages, {len(all_klines)} klines...", flush=True)
        time.sleep(sleep_between)

    result = sorted(all_klines.values(), key=lambda r: int(r["open_time"]))
    print(f"  [{symbol}/{interval}] done: {len(result)} klines in {pages + 1} pages", flush=True)
    return result


def save_historical_klines(
    klines: list[dict[str, Any]],
    *,
    output_dir: Path,
    symbol: str,
    interval: str,
) -> Path:
    """Write klines to {output_dir}/{symbol}/{interval}.json."""
    target_dir = output_dir / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{interval}.json"
    path.write_text(json.dumps(klines, indent=None, sort_keys=False), encoding="utf-8")
    return path


def load_historical_klines(
    *,
    data_dir: Path,
    symbol: str,
    interval: str,
) -> list[dict[str, Any]]:
    """Load cached klines from disk."""
    path = data_dir / symbol / f"{interval}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def download_funding_rates(
    client: Any,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    page_size: int = 100,
    sleep_between: float = 1.0,
    max_pages: int = 10,
) -> list[dict[str, Any]]:
    """Paginate funding rate history for *symbol* within [start_ms, end_ms].

    Bitget settles funding every 8h, so 90 days ≈ 270 records ≈ 3 pages.
    Uses pageNo pagination.
    """
    all_rates: dict[int, dict[str, Any]] = {}  # dedup by funding_time
    page_no = 1

    while page_no <= max_pages:
        batch = client.get_historical_funding_rates(
            symbol=symbol,
            page_size=page_size,
            page_no=page_no,
        )
        if not batch:
            break

        reached_start = False
        for row in batch:
            ft = int(row["funding_time"])
            if ft < start_ms:
                reached_start = True
                continue
            if ft <= end_ms:
                all_rates[ft] = row

        if reached_start:
            break

        page_no += 1
        time.sleep(sleep_between)

    result = sorted(all_rates.values(), key=lambda r: int(r["funding_time"]))
    print(f"  [{symbol}/funding] done: {len(result)} rates in {page_no} pages", flush=True)
    return result


def save_funding_rates(
    rates: list[dict[str, Any]],
    *,
    output_dir: Path,
    symbol: str,
) -> Path:
    """Write funding rates to {output_dir}/{symbol}/funding_rates.json."""
    target_dir = output_dir / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "funding_rates.json"
    path.write_text(json.dumps(rates, indent=None, sort_keys=False), encoding="utf-8")
    return path


def load_funding_rates(
    *,
    data_dir: Path,
    symbol: str,
) -> list[dict[str, Any]]:
    """Load cached funding rates from disk."""
    path = data_dir / symbol / "funding_rates.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def download_spot_klines(
    client: Any,
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    page_limit: int = 200,
    sleep_between: float = 1.5,
) -> list[dict[str, Any]]:
    """Download spot klines using download_klines_range with market='spot'.

    Saves to {output_dir}/{symbol}/spot_{interval}.json when combined with
    ``save_spot_klines``.
    """
    return download_klines_range(
        client,
        market="spot",
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        page_limit=page_limit,
        sleep_between=sleep_between,
    )


def save_spot_klines(
    klines: list[dict[str, Any]],
    *,
    output_dir: Path,
    symbol: str,
    interval: str,
) -> Path:
    """Write spot klines to {output_dir}/{symbol}/spot_{interval}.json."""
    target_dir = output_dir / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"spot_{interval}.json"
    path.write_text(json.dumps(klines, indent=None, sort_keys=False), encoding="utf-8")
    return path


def load_spot_klines(
    *,
    data_dir: Path,
    symbol: str,
    interval: str,
) -> list[dict[str, Any]]:
    """Load cached spot klines from disk."""
    path = data_dir / symbol / f"spot_{interval}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def download_all_symbols(
    client: Any,
    *,
    symbols: list[str],
    intervals: list[str],
    days: int,
    market: str = "futures",
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """Download historical klines for all symbols/intervals.
    Returns {symbol: {interval: count}} counts."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86_400_000
    counts: dict[str, dict[str, int]] = {}

    for symbol in symbols:
        counts[symbol] = {}
        for interval in intervals:
            cached = load_historical_klines(data_dir=output_dir, symbol=symbol, interval=interval)
            if cached:
                cached_range_days = (int(cached[-1]["open_time"]) - int(cached[0]["open_time"])) / 86_400_000
                if cached_range_days >= days * 0.9:
                    print(f"  [{symbol}/{interval}] cached ({len(cached)} klines, {cached_range_days:.0f}d), skipping", flush=True)
                    counts[symbol][interval] = len(cached)
                    continue

            klines = download_klines_range(
                client,
                market=market,
                symbol=symbol,
                interval=interval,
                start_ms=start_ms,
                end_ms=now_ms,
            )
            if klines:
                save_historical_klines(klines, output_dir=output_dir, symbol=symbol, interval=interval)
            counts[symbol][interval] = len(klines)

    # --- Additional downloads: spot klines, funding rates, 1m klines ---

    for symbol in symbols:
        # Spot 1h klines (for basis = futures_close - spot_close)
        spot_key = "spot_1h"
        cached_spot = load_spot_klines(data_dir=output_dir, symbol=symbol, interval="1h")
        if cached_spot:
            cached_range_days = (int(cached_spot[-1]["open_time"]) - int(cached_spot[0]["open_time"])) / 86_400_000
            if cached_range_days >= days * 0.9:
                print(f"  [{symbol}/spot_1h] cached ({len(cached_spot)} klines, {cached_range_days:.0f}d), skipping", flush=True)
                counts[symbol][spot_key] = len(cached_spot)
            else:
                cached_spot = []  # force re-download
        if not cached_spot:
            spot_klines = download_spot_klines(
                client,
                symbol=symbol,
                interval="1h",
                start_ms=start_ms,
                end_ms=now_ms,
            )
            if spot_klines:
                save_spot_klines(spot_klines, output_dir=output_dir, symbol=symbol, interval="1h")
            counts[symbol][spot_key] = len(spot_klines)

        # Funding rate history
        funding_key = "funding_rates"
        cached_funding = load_funding_rates(data_dir=output_dir, symbol=symbol)
        if cached_funding:
            cached_range_days = (int(cached_funding[-1]["funding_time"]) - int(cached_funding[0]["funding_time"])) / 86_400_000
            if cached_range_days >= days * 0.9:
                print(f"  [{symbol}/funding] cached ({len(cached_funding)} rates, {cached_range_days:.0f}d), skipping", flush=True)
                counts[symbol][funding_key] = len(cached_funding)
            else:
                cached_funding = []  # force re-download
        if not cached_funding:
            rates = download_funding_rates(
                client,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=now_ms,
            )
            if rates:
                save_funding_rates(rates, output_dir=output_dir, symbol=symbol)
            counts[symbol][funding_key] = len(rates)

        # 1m klines — only last 7 days to limit API calls
        one_min_key = "1m"
        start_1m = now_ms - 7 * 86_400_000
        cached_1m = load_historical_klines(data_dir=output_dir, symbol=symbol, interval="1m")
        if cached_1m:
            cached_range_days = (int(cached_1m[-1]["open_time"]) - int(cached_1m[0]["open_time"])) / 86_400_000
            if cached_range_days >= 6:  # 7d * ~0.85
                print(f"  [{symbol}/1m] cached ({len(cached_1m)} klines, {cached_range_days:.0f}d), skipping", flush=True)
                counts[symbol][one_min_key] = len(cached_1m)
            else:
                cached_1m = []  # force re-download
        if not cached_1m:
            klines_1m = download_klines_range(
                client,
                market=market,
                symbol=symbol,
                interval="1m",
                start_ms=start_1m,
                end_ms=now_ms,
            )
            if klines_1m:
                save_historical_klines(klines_1m, output_dir=output_dir, symbol=symbol, interval="1m")
            counts[symbol][one_min_key] = len(klines_1m)

    return counts
