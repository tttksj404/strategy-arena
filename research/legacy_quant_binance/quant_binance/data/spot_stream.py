from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from quant_binance.data.state import KlineBar, SpotTrade, TopOfBook


BINANCE_SPOT_WS_URL = "wss://stream.binance.com:9443/stream"


def build_spot_streams(symbol: str, intervals: tuple[str, ...]) -> list[str]:
    base = symbol.lower()
    streams = [f"{base}@trade", f"{base}@bookTicker"]
    streams.extend(f"{base}@kline_{interval}" for interval in intervals)
    return streams


def build_subscribe_message(streams: list[str], request_id: int = 1) -> dict[str, Any]:
    return {"method": "SUBSCRIBE", "params": streams, "id": request_id}


def parse_spot_trade_event(payload: dict[str, Any]) -> SpotTrade:
    return SpotTrade(
        symbol=payload["s"],
        price=float(payload["p"]),
        quantity=float(payload["q"]),
        event_time=datetime.fromtimestamp(payload["E"] / 1000, tz=timezone.utc),
        is_buyer_maker=bool(payload["m"]),
    )


def parse_book_ticker_event(payload: dict[str, Any]) -> TopOfBook:
    event_time_ms = payload.get("E", payload.get("u"))
    return TopOfBook(
        bid_price=float(payload["b"]),
        bid_qty=float(payload["B"]),
        ask_price=float(payload["a"]),
        ask_qty=float(payload["A"]),
        updated_at=datetime.fromtimestamp(event_time_ms / 1000, tz=timezone.utc),
    )


def parse_kline_event(payload: dict[str, Any]) -> KlineBar:
    kline = payload["k"]
    return KlineBar(
        symbol=payload["s"],
        interval=kline["i"],
        start_time=datetime.fromtimestamp(kline["t"] / 1000, tz=timezone.utc),
        close_time=datetime.fromtimestamp(kline["T"] / 1000, tz=timezone.utc),
        open_price=float(kline["o"]),
        high_price=float(kline["h"]),
        low_price=float(kline["l"]),
        close_price=float(kline["c"]),
        volume=float(kline["v"]),
        quote_volume=float(kline["q"]),
        is_closed=bool(kline["x"]),
    )
