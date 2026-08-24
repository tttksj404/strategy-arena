from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from quant_binance.data.market_store import MarketStateStore
from quant_binance.data.state import KlineBar, SymbolMarketState, TopOfBook


def _parse_kline(symbol: str, interval: str, row: object) -> KlineBar:
    if isinstance(row, dict):
        open_time = int(row["open_time"])
        quote_volume = float(row.get("quote_volume", row.get("base_volume", 0.0)))
        close_time = open_time
        if interval.endswith("m"):
            close_time += int(interval[:-1]) * 60 * 1000
        elif interval.endswith("h"):
            close_time += int(interval[:-1]) * 60 * 60 * 1000
        elif interval.endswith("d"):
            close_time += int(interval[:-1]) * 24 * 60 * 60 * 1000
        return KlineBar(
            symbol=symbol,
            interval=interval,
            start_time=datetime.fromtimestamp(open_time / 1000, tz=timezone.utc),
            close_time=datetime.fromtimestamp(close_time / 1000, tz=timezone.utc),
            open_price=float(row["open_price"]),
            high_price=float(row["high_price"]),
            low_price=float(row["low_price"]),
            close_price=float(row["close_price"]),
            volume=float(row["base_volume"]),
            quote_volume=quote_volume,
            is_closed=True,
        )
    assert isinstance(row, list)
    return KlineBar(
        symbol=symbol,
        interval=interval,
        start_time=datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
        close_time=datetime.fromtimestamp(int(row[6]) / 1000, tz=timezone.utc),
        open_price=float(row[1]),
        high_price=float(row[2]),
        low_price=float(row[3]),
        close_price=float(row[4]),
        volume=float(row[5]),
        quote_volume=float(row[7]),
        is_closed=True,
    )


def seed_market_store_from_rest(
    *,
    client: Any,
    symbols: tuple[str, ...],
    intervals: tuple[str, ...],
) -> MarketStateStore:
    store = MarketStateStore()
    spot_exchange_info = client.get_exchange_info(market="spot")
    futures_exchange_info = client.get_exchange_info(market="futures")
    spot_symbols = {item["symbol"] for item in spot_exchange_info["symbols"]}
    futures_symbols = {item["symbol"] for item in futures_exchange_info["symbols"]}
    for symbol in symbols:
        if symbol not in spot_symbols:
            continue
        spot_book = client.get_book_ticker(market="spot", symbol=symbol)
        now = datetime.now(tz=timezone.utc)
        if symbol in futures_symbols:
            futures_mark = client.get_mark_price(symbol=symbol)
            open_interest = client.get_open_interest(symbol=symbol)
            mark_price = float(futures_mark["markPrice"])
            index_price = float(futures_mark["indexPrice"])
            funding_rate = float(futures_mark["lastFundingRate"])
            open_interest_value = float(open_interest["openInterest"])
            basis_bps = ((mark_price - index_price) / max(index_price, 1e-9)) * 10000.0
        else:
            mark_price = float(spot_book["bidPrice"])
            funding_rate = 0.0
            open_interest_value = 0.0
            basis_bps = 0.0
        state = SymbolMarketState(
            symbol=symbol,
            top_of_book=TopOfBook(
                bid_price=float(spot_book["bidPrice"]),
                bid_qty=float(spot_book["bidQty"]),
                ask_price=float(spot_book["askPrice"]),
                ask_qty=float(spot_book["askQty"]),
                updated_at=now,
            ),
            last_trade_price=mark_price,
            funding_rate=funding_rate,
            open_interest=open_interest_value,
            basis_bps=basis_bps,
            last_update_time=now,
        )
        state.funding_rate_samples.append(state.funding_rate)
        state.basis_bps_samples.append(state.basis_bps)
        state.open_interest_samples.append(state.open_interest)
        for interval in intervals:
            limit = 140 if interval in {"1h", "4h"} else 100
            for row in client.get_klines(market="spot", symbol=symbol, interval=interval, limit=limit):
                state.klines.setdefault(interval, []).append(_parse_kline(symbol, interval, row))
        store.put(state)
    return store
