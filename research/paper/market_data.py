from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from typing import Final

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.scanner.scan_bitget import STOCK_BASE_COINS
from research.wave1.common import JsonValue, PipelineError, request_json, validate_symbol
from research.wave1.fetch_binance import BinanceFundingRequest, BinanceKlineRequest, exchange_symbols, fetch_exchange_info, fetch_funding, fetch_klines, fetch_quote_volumes, fetch_spot_exchange_info, quote_volumes
from research.wave1.fetch_bitget import BITGET_BASE, BitgetCandleRequest, contract_symbols, fetch_candles, fetch_contracts, fetch_funding as fetch_bitget_funding
from research.wave3.engine import AssetMarket
from research.wave3.universe import AssetListing, AssetType, parse_binance_um_listings, parse_bitget_stock_listings

HISTORY_DAYS: Final = 230
CRYPTO_VOLUME_LIMIT: Final = 150
CARRY_VOLUME_LIMIT: Final = 40

# Bitget-ranked carry universe (bug fix, see research/paper/fidelity.py): the block above this
# (Binance UM perp AND Binance spot AND Bitget contract, triple-intersected) is what
# CRYPTO_VOLUME_LIMIT/CARRY_VOLUME_LIMIT ration -- it silently excludes any symbol missing just
# ONE of those three listings, which turns out to include most of Bitget's highest-funding
# perpetuals (tokenized stocks/commodities, Bitget-only alts). research/STRATEGY_CARD.md's own
# lineage documents the REAL declared universes: G1 = top100 (wave21_ga genome
# universe_breadth=100), W2c = top200 (the wave13 "L4" breadth-sweep saturation point that W2c's
# params match exactly). 200 covers both. Ranked straight off Bitget's own tickers (the actual
# execution venue this whole card models), independent of whether Binance lists the symbol at all.
BITGET_UNIVERSE_LIMIT: Final = 200
# Bitget's own max page size for history-fund-rate. At the standard 8h funding interval (3/day)
# one page spans ~33 days -- comfortably over G1's 14-day scoring window (+margin) and W2c/F1e's
# 7-day window. Deliberately NOT research.wave1.fetch_bitget.fetch_funding, which paginates up
# to 100 pages per symbol (years of history) -- fine for one symbol, far too slow for 200.
BITGET_FUNDING_PAGE_SIZE: Final = 100
# Soft cutoff for the per-symbol Bitget funding-history loop: leaves ~1 minute of the 5-minute
# runtime target for the tickers calls, the pre-existing Binance-sourced path, and everything
# else. A slow-network day degrades to "fewer symbols collected, clearly reported as failed"
# rather than silently blowing past the target -- see run_once()'s STATUS.md reporting.
FUNDING_LOOP_BUDGET_SECONDS: Final = 240.0
FUNDING_CACHE_DIR: Final = Path(__file__).resolve().parent / "cache" / "bitget_funding"


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    observed_at: pd.Timestamp
    funding_series: dict[str, pd.Series]
    funding_rates: dict[str, float]
    perp_prices: dict[str, float]
    spot_prices: dict[str, float]
    wave3_markets: dict[str, AssetMarket]
    source_names: tuple[str, ...]
    # Collection transparency (task: "수집 실패 심볼은 제외하고 기록(가정으로 채우지 말 것)" +
    # "5분 초과하면 그 사실을 STATUS.md에 명시"). Defaulted so every pre-existing constructor
    # (this module's own return statement aside) keeps working unchanged.
    universe_failed_symbols: tuple[str, ...] = ()
    collection_seconds: float = 0.0


def _listing_maps(
    futures_payload: JsonValue,
    spot_payload: JsonValue,
    bitget_payload: JsonValue,
    now: pd.Timestamp,
) -> tuple[dict[str, AssetListing], tuple[AssetListing, ...]]:
    futures = parse_binance_um_listings(
        futures_payload,
        exchange_symbols(spot_payload),
        contract_symbols(bitget_payload),
    )
    fallback = {
        str(item["symbol"]): now - pd.Timedelta(days=365)
        for item in bitget_payload
        if isinstance(item, dict) and isinstance(item.get("symbol"), str) and str(item.get("baseCoin", "")).upper() in STOCK_BASE_COINS
    } if isinstance(bitget_payload, list) else {}
    stocks = parse_bitget_stock_listings(bitget_payload, set(STOCK_BASE_COINS), fallback)
    return ({listing.symbol: listing for listing in futures}, stocks)


def _latest_price(frame: pd.DataFrame) -> float:
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        raise PipelineError("live market frame has no close price")
    return float(close.iloc[-1])


def _fetch_crypto_markets(
    session: requests.Session,
    symbols: tuple[str, ...],
    listings: dict[str, AssetListing],
    funding_symbols: set[str],
    start_ms: int,
    end_ms: int,
) -> tuple[dict[str, AssetMarket], dict[str, pd.Series], dict[str, float]]:
    markets: dict[str, AssetMarket] = {}
    funding_series: dict[str, pd.Series] = {}
    prices: dict[str, float] = {}
    for symbol in symbols:
        try:
            perp = fetch_klines(BinanceKlineRequest(symbol, "1d", start_ms, end_ms), session)
            if perp.empty or symbol not in listings:
                continue
            funding = fetch_funding(BinanceFundingRequest(symbol, start_ms, end_ms), session) if symbol in funding_symbols else pd.DataFrame()
            series = funding["funding_rate"].sort_index() if not funding.empty else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
            markets[symbol] = AssetMarket(listings[symbol], perp.sort_index(), None, series)
            prices[symbol] = _latest_price(perp)
            if not series.empty:
                funding_series[symbol] = series
        except (PipelineError, requests.RequestException) as error:
            print(f"paper: skipped {symbol}: {error}")
    return markets, funding_series, prices


def _fetch_stock_markets(
    session: requests.Session,
    listings: tuple[AssetListing, ...],
    start_ms: int,
    end_ms: int,
) -> tuple[dict[str, AssetMarket], dict[str, pd.Series], dict[str, float]]:
    markets: dict[str, AssetMarket] = {}
    funding_series: dict[str, pd.Series] = {}
    prices: dict[str, float] = {}
    for listing in listings:
        try:
            candles = fetch_candles(BitgetCandleRequest(listing.symbol, "1D", start_ms, end_ms), session)
            if candles.empty:
                continue
            funding = fetch_bitget_funding(listing.symbol, session)
            series = funding["funding_rate"].sort_index() if not funding.empty else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
            markets[listing.symbol] = AssetMarket(listing, candles.sort_index(), None, series)
            prices[listing.symbol] = _latest_price(candles)
            if not series.empty:
                funding_series[listing.symbol] = series
        except (PipelineError, requests.RequestException) as error:
            print(f"paper: skipped {listing.symbol}: {error}")
    return markets, funding_series, prices


def _bitget_payload(payload: JsonValue) -> JsonValue:
    """Mirrors research.wave1.fetch_bitget's private `_data()` helper. Duplicated (not
    imported): that name is private to its module, and research/paper/ may only READ other
    wave modules -- never edit them to export a new name."""
    if not isinstance(payload, dict) or "data" not in payload:
        raise PipelineError("Bitget response is missing data")
    code = payload.get("code")
    if code is not None and code != "00000":
        raise PipelineError(f"Bitget error code: {code}")
    return payload["data"]


def _as_float(value: JsonValue, default: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bitget_mix_volumes(payload: JsonValue) -> dict[str, float]:
    """symbol -> 24h USDT turnover, parsed from ONE bulk /api/v2/mix/market/tickers response
    (no per-symbol request). Ranking source of truth for BITGET_UNIVERSE_LIMIT."""
    if not isinstance(payload, list):
        raise PipelineError("Bitget mix tickers payload must be a list")
    volumes: dict[str, float] = {}
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("symbol"), str):
            raw = item["usdtVolume"] if item.get("usdtVolume") is not None else item.get("quoteVolume")
            volumes[item["symbol"]] = _as_float(raw, 0.0) or 0.0
    return volumes


def bitget_mix_prices(payload: JsonValue) -> dict[str, float]:
    """symbol -> last perp trade price, parsed from the SAME bulk tickers response used for
    ranking -- `lastPr` rides along for free, no extra per-symbol price request needed."""
    if not isinstance(payload, list):
        raise PipelineError("Bitget mix tickers payload must be a list")
    prices: dict[str, float] = {}
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("symbol"), str):
            price = _as_float(item.get("lastPr"), None)
            if price is not None and price > 0.0:
                prices[item["symbol"]] = price
    return prices


def bitget_spot_prices(payload: JsonValue) -> dict[str, float]:
    """symbol -> last spot trade price, parsed from ONE bulk /api/v2/spot/market/tickers
    response. Only symbols with an actual Bitget spot market appear -- a carry candidate whose
    top pick is perp-only (common for tokenized-stock/commodity perpetuals) correctly gets no
    spot leg here and is dropped downstream by track.py's existing "spot_price is None" guard;
    never fabricated."""
    if not isinstance(payload, list):
        raise PipelineError("Bitget spot tickers payload must be a list")
    prices: dict[str, float] = {}
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("symbol"), str):
            price = _as_float(item.get("lastPr"), None)
            if price is not None and price > 0.0:
                prices[item["symbol"]] = price
    return prices


def rank_bitget_universe(volumes: dict[str, float], limit: int = BITGET_UNIVERSE_LIMIT) -> tuple[str, ...]:
    """Top `limit` symbols by 24h USDT turnover, descending, symbol as a deterministic
    tie-break -- same convention the pre-existing Binance ranking a few lines up uses
    (`sorted(common, key=lambda symbol: (-volumes.get(symbol, 0.0), symbol))`)."""
    return tuple(sorted(volumes, key=lambda symbol: (-volumes[symbol], symbol))[:limit])


def _fetch_bitget_mix_tickers(session: requests.Session) -> JsonValue:
    return _bitget_payload(request_json(session, BITGET_BASE + "/api/v2/mix/market/tickers", {"productType": "usdt-futures"}))


def _fetch_bitget_spot_tickers(session: requests.Session) -> JsonValue:
    return _bitget_payload(request_json(session, BITGET_BASE + "/api/v2/spot/market/tickers", {}))


def _fetch_bitget_funding_recent(symbol: str, session: requests.Session, page_size: int = BITGET_FUNDING_PAGE_SIZE) -> pd.Series:
    """Single-page Bitget history-fund-rate fetch. See BITGET_FUNDING_PAGE_SIZE for why one
    page (not research.wave1.fetch_bitget.fetch_funding's full pagination) is both sufficient
    and required to keep 200 symbols inside the runtime budget."""
    validate_symbol(symbol)
    raw = _bitget_payload(
        request_json(
            session,
            BITGET_BASE + "/api/v2/mix/market/history-fund-rate",
            {"symbol": symbol, "productType": "usdt-futures", "pageSize": page_size, "pageNo": 1},
        )
    )
    if not isinstance(raw, list):
        raise PipelineError("Bitget funding data must be a list")
    rows = [row for row in raw if isinstance(row, dict) and "fundingTime" in row and "fundingRate" in row]
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame["fundingTime"], errors="coerce"), unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "funding_rate"])
    return frame.set_index("timestamp")["funding_rate"].sort_index().loc[lambda item: ~item.index.duplicated()]


def _funding_cache_path(symbol: str) -> Path:
    return FUNDING_CACHE_DIR / f"{symbol}.json"


def load_cached_funding(symbol: str, today: str) -> pd.Series | None:
    """A cached funding series if -- and only if -- it was fetched on THIS UTC calendar date;
    otherwise None (forces a fresh fetch). Implements "당일 이미 받은 건 스킵": reruns later the
    same day skip the network call entirely; a new day always refetches. A symbol cached
    yesterday but outside today's ranked top-N is simply never looked up (harmless leftover)."""
    path = _funding_cache_path(symbol)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("fetched_date") != today:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.DataFrame(rows)
    if "timestamp" not in frame.columns or "funding_rate" not in frame.columns:
        return None
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    rates = pd.to_numeric(frame["funding_rate"], errors="coerce")
    series = pd.Series(rates.to_numpy(dtype=float), index=pd.DatetimeIndex(timestamps)).dropna()
    return series.sort_index().loc[lambda item: ~item.index.duplicated()]


def save_cached_funding(symbol: str, today: str, series: pd.Series) -> None:
    FUNDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"timestamp": pd.Timestamp(index).isoformat(), "funding_rate": float(value)} for index, value in series.items()]
    payload = {"fetched_date": today, "rows": rows}
    path = _funding_cache_path(symbol)
    tmp_path = path.with_suffix(f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def extend_with_bitget_carry_universe(
    session: requests.Session,
    funding_series: dict[str, pd.Series],
    perp_prices: dict[str, float],
    spot_prices: dict[str, float],
    today: str,
    loop_deadline: float,
) -> tuple[str, ...]:
    """Fills in funding history (+ perp/spot price where available) for the Bitget top-N
    volume-ranked universe, for every symbol the Binance-intersected path above did NOT already
    cover. This is the actual bug fix: G1 (top100) and W2c (top200) need a broader universe than
    "listed on Binance perp AND Binance spot AND Bitget contract" -- see BITGET_UNIVERSE_LIMIT's
    comment. Returns the symbols whose collection failed or was skipped this run (time budget or
    request error) -- excluded from funding_series, never assumed/filled with a guess."""
    mix_tickers = _fetch_bitget_mix_tickers(session)
    volumes = bitget_mix_volumes(mix_tickers)
    perp_lookup = bitget_mix_prices(mix_tickers)
    ranked = rank_bitget_universe(volumes, BITGET_UNIVERSE_LIMIT)
    spot_tickers = _fetch_bitget_spot_tickers(session)
    spot_lookup = bitget_spot_prices(spot_tickers)

    failed: list[str] = []
    for symbol in ranked:
        if symbol in funding_series:
            continue  # already covered by the Binance-sourced path above; keep that data as-is, don't refetch/overwrite
        if time.monotonic() > loop_deadline:
            failed.append(symbol)  # time-budget cutoff -- reported, not silently dropped
            continue
        cached = load_cached_funding(symbol, today)
        if cached is not None:
            series = cached
        else:
            try:
                series = _fetch_bitget_funding_recent(symbol, session)
            except (PipelineError, requests.RequestException) as error:
                print(f"paper: bitget funding skipped {symbol}: {error}")
                failed.append(symbol)
                continue
            save_cached_funding(symbol, today, series)
        if series.empty:
            failed.append(symbol)
            continue
        funding_series[symbol] = series
        if symbol in perp_lookup:
            perp_prices.setdefault(symbol, perp_lookup[symbol])
        if symbol in spot_lookup:
            spot_prices.setdefault(symbol, spot_lookup[symbol])
    return tuple(failed)


def collect_live_snapshot(now: pd.Timestamp | None = None) -> LiveSnapshot:
    started = time.monotonic()
    observed_at = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    observed_at = observed_at.tz_localize("UTC") if observed_at.tzinfo is None else observed_at.tz_convert("UTC")
    end_ms = int(observed_at.timestamp() * 1000)
    start_ms = int((observed_at - timedelta(days=HISTORY_DAYS)).timestamp() * 1000)
    today = observed_at.date().isoformat()
    with requests.Session() as session:
        futures_payload = fetch_exchange_info(session)
        spot_payload = fetch_spot_exchange_info(session)
        bitget_payload = fetch_contracts(session)
        volume_payload = fetch_quote_volumes(session)
        futures = exchange_symbols(futures_payload)
        spot = exchange_symbols(spot_payload)
        bitget = contract_symbols(bitget_payload)
        volumes = quote_volumes(volume_payload)
        common = futures & spot & bitget
        crypto_symbols = tuple(sorted(common, key=lambda symbol: (-volumes.get(symbol, 0.0), symbol))[:CRYPTO_VOLUME_LIMIT])
        carry_symbols = set(crypto_symbols[:CARRY_VOLUME_LIMIT]) | {"BTCUSDT", "ETHUSDT"}
        crypto_listings, stock_listings = _listing_maps(futures_payload, spot_payload, bitget_payload, observed_at)
        crypto_markets, funding_series, perp_prices = _fetch_crypto_markets(session, crypto_symbols, crypto_listings, carry_symbols, start_ms, end_ms)
        stock_markets, stock_funding, stock_prices = _fetch_stock_markets(session, stock_listings, start_ms, end_ms)
        funding_series.update(stock_funding)
        perp_prices.update(stock_prices)
        spot_prices: dict[str, float] = {}
        for symbol in sorted(funding_series):
            if symbol not in crypto_markets:
                continue
            try:
                spot = fetch_klines(BinanceKlineRequest(symbol, "1d", max(start_ms, end_ms - 3 * 86_400_000), end_ms, "spot"), session)
                if not spot.empty:
                    spot_prices[symbol] = _latest_price(spot)
            except (PipelineError, requests.RequestException) as error:
                print(f"paper: spot unavailable {symbol}: {error}")
        if not crypto_markets:
            raise PipelineError("live Binance market snapshot is empty")
        wave3_markets = {**crypto_markets, **stock_markets}

        # Bug fix: broaden the carry-relevant funding universe to what G1 (top100) and W2c
        # (top200) actually require -- see BITGET_UNIVERSE_LIMIT and
        # extend_with_bitget_carry_universe's docstrings.
        loop_deadline = time.monotonic() + FUNDING_LOOP_BUDGET_SECONDS
        failed_symbols = extend_with_bitget_carry_universe(session, funding_series, perp_prices, spot_prices, today, loop_deadline)

        funding_rates = {symbol: float(series.iloc[-1]) for symbol, series in funding_series.items() if not series.empty}
    collection_seconds = time.monotonic() - started
    return LiveSnapshot(
        observed_at,
        funding_series,
        funding_rates,
        perp_prices,
        spot_prices,
        wave3_markets,
        (
            "Binance UM public klines/funding/exchangeInfo",
            "Binance spot public klines",
            "Bitget public contracts/candles/funding",
            "Bitget public mix/spot tickers + history-fund-rate",
        ),
        failed_symbols,
        collection_seconds,
    )


def current_funding_rates(symbols: tuple[str, ...], markets: dict[str, AssetMarket]) -> dict[str, float]:
    rates: dict[str, float] = {}
    crypto = tuple(symbol for symbol in symbols if symbol in markets and markets[symbol].listing.asset_type is AssetType.CRYPTO)
    stocks = tuple(symbol for symbol in symbols if symbol in markets and markets[symbol].listing.asset_type is AssetType.STOCK_TOKEN)
    with requests.Session() as session:
        for symbol in crypto:
            funding = fetch_funding(BinanceFundingRequest(validate_symbol(symbol), int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).timestamp() * 1000), int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)), session)
            if not funding.empty:
                rates[symbol] = float(funding["funding_rate"].iloc[-1])
        for symbol in stocks:
            funding = fetch_bitget_funding(validate_symbol(symbol), session)
            if not funding.empty:
                rates[symbol] = float(funding["funding_rate"].iloc[-1])
    return rates


__all__ = [
    "BITGET_FUNDING_PAGE_SIZE",
    "BITGET_UNIVERSE_LIMIT",
    "FUNDING_LOOP_BUDGET_SECONDS",
    "LiveSnapshot",
    "bitget_mix_prices",
    "bitget_mix_volumes",
    "bitget_spot_prices",
    "collect_live_snapshot",
    "current_funding_rates",
    "extend_with_bitget_carry_universe",
    "load_cached_funding",
    "rank_bitget_universe",
    "save_cached_funding",
]
