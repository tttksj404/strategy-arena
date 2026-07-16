from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Final

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.scanner.scan_bitget import STOCK_BASE_COINS
from research.wave1.common import JsonValue, PipelineError, validate_symbol
from research.wave1.fetch_binance import BinanceFundingRequest, BinanceKlineRequest, exchange_symbols, fetch_exchange_info, fetch_funding, fetch_klines, fetch_quote_volumes, fetch_spot_exchange_info, quote_volumes
from research.wave1.fetch_bitget import BitgetCandleRequest, contract_symbols, fetch_candles, fetch_contracts, fetch_funding as fetch_bitget_funding
from research.wave3.engine import AssetMarket
from research.wave3.universe import AssetListing, AssetType, parse_binance_um_listings, parse_bitget_stock_listings

HISTORY_DAYS: Final = 230
CRYPTO_VOLUME_LIMIT: Final = 150
CARRY_VOLUME_LIMIT: Final = 40


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    observed_at: pd.Timestamp
    funding_series: dict[str, pd.Series]
    funding_rates: dict[str, float]
    perp_prices: dict[str, float]
    spot_prices: dict[str, float]
    wave3_markets: dict[str, AssetMarket]
    source_names: tuple[str, ...]


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


def collect_live_snapshot(now: pd.Timestamp | None = None) -> LiveSnapshot:
    observed_at = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    observed_at = observed_at.tz_localize("UTC") if observed_at.tzinfo is None else observed_at.tz_convert("UTC")
    end_ms = int(observed_at.timestamp() * 1000)
    start_ms = int((observed_at - timedelta(days=HISTORY_DAYS)).timestamp() * 1000)
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
        funding_rates = {symbol: float(series.iloc[-1]) for symbol, series in funding_series.items() if not series.empty}
    return LiveSnapshot(
        observed_at,
        funding_series,
        funding_rates,
        perp_prices,
        spot_prices,
        wave3_markets,
        ("Binance UM public klines/funding/exchangeInfo", "Binance spot public klines", "Bitget public contracts/candles/funding"),
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


__all__ = ["LiveSnapshot", "collect_live_snapshot", "current_funding_rates"]
