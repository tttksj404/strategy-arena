from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

try:
    import websockets
except ImportError:  # pragma: no cover - optional dependency
    websockets = None


BITGET_PUBLIC_WS_URL = "wss://ws.bitget.com/v2/ws/public"
BITGET_SPOT_PUBLIC_WS_URL = BITGET_PUBLIC_WS_URL
BITGET_FUTURES_PUBLIC_WS_URL = BITGET_PUBLIC_WS_URL
BITGET_MAX_CHANNELS_PER_CONNECTION = 40
BITGET_MAX_CHANNELS_PER_SUBSCRIBE = 40

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]

_INTERVAL_TO_CHANNEL = {
    "1h": "candle1H",
    "4h": "candle4H",
}

_CHANNEL_TO_INTERVAL = {
    "candle1H": "1h",
    "candle4H": "4h",
}


def _coerce_timestamp_ms(value: object, fallback_ms: int) -> int:
    if value is None:
        return fallback_ms
    try:
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return fallback_ms
        return int(float(text))
    except (TypeError, ValueError):
        return fallback_ms


def _interval_to_channel(interval: str) -> str:
    return _INTERVAL_TO_CHANNEL.get(interval, f"candle{interval}")


def _channel_to_interval(channel: str) -> str:
    return _CHANNEL_TO_INTERVAL.get(channel, channel.removeprefix("candle").replace("D", "d").replace("H", "h"))


def _interval_to_milliseconds(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60 * 1000
    if interval.endswith("h"):
        return int(interval[:-1]) * 60 * 60 * 1000
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60 * 60 * 1000
    raise RuntimeError(f"unsupported Bitget interval '{interval}'")


def translate_bitget_ws_payload(
    payload: dict[str, Any],
    *,
    market: str,
    received_at: datetime | None = None,
) -> list[dict[str, Any]]:
    return BitgetWebSocketClient(
        market=market,
        symbols=(),
        intervals=(),
    ).normalize_payload(payload, received_at=received_at)


def build_spot_streams(symbol: str, intervals: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {"instType": "SPOT", "channel": "trade", "instId": symbol},
        {"instType": "SPOT", "channel": "ticker", "instId": symbol},
        *(
            {"instType": "SPOT", "channel": _interval_to_channel(interval), "instId": symbol}
            for interval in intervals
        ),
    ]


def build_futures_streams(symbol: str, intervals: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {"instType": "USDT-FUTURES", "channel": "trade", "instId": symbol},
        {"instType": "USDT-FUTURES", "channel": "ticker", "instId": symbol},
        *(
            {"instType": "USDT-FUTURES", "channel": _interval_to_channel(interval), "instId": symbol}
            for interval in intervals
        ),
    ]


def build_subscribe_message(streams: list[dict[str, str]]) -> dict[str, Any]:
    return {"op": "subscribe", "args": streams}


def translate_public_payload(
    *,
    market: str,
    payload: dict[str, Any],
    received_at: datetime | None = None,
) -> list[dict[str, Any]]:
    return BitgetWebSocketClient(market=market, symbols=(), intervals=()).normalize_payload(
        payload,
        received_at=received_at,
    )


class BitgetWebSocketClient:
    def __init__(
        self,
        *,
        market: str,
        symbols: tuple[str, ...],
        intervals: tuple[str, ...],
        allow_insecure_ssl: bool = False,
        label: str = "",
    ) -> None:
        self.market = market
        self.symbols = symbols
        self.intervals = intervals
        self.allow_insecure_ssl = allow_insecure_ssl
        self.label = label or market
        self._closed_candle_cache: set[tuple[str, str, int]] = set()

    @property
    def url(self) -> str:
        return BITGET_PUBLIC_WS_URL

    @property
    def inst_type(self) -> str:
        return "SPOT" if self.market == "spot" else "USDT-FUTURES"

    def subscription_args(self) -> list[dict[str, str]]:
        args: list[dict[str, str]] = []
        for symbol in self.symbols:
            builder = build_spot_streams if self.market == "spot" else build_futures_streams
            args.extend(builder(symbol, self.intervals))
        return args

    def build_subscribe_message(self) -> dict[str, Any]:
        return build_subscribe_message(self.subscription_args())

    def build_subscribe_messages(self) -> list[dict[str, Any]]:
        args = self.subscription_args()
        if not args:
            return []
        messages: list[dict[str, Any]] = []
        for index in range(0, len(args), BITGET_MAX_CHANNELS_PER_SUBSCRIBE):
            messages.append(build_subscribe_message(args[index : index + BITGET_MAX_CHANNELS_PER_SUBSCRIBE]))
        return messages

    def _connect_kwargs(self, *, ssl_context: ssl.SSLContext | None) -> dict[str, Any]:
        # Let the runtime stall watchdog own reconnect decisions instead of the
        # websockets client's ping/pong lifecycle, which has been causing
        # false-positive disconnects and unclean close errors on long-lived
        # Bitget sessions.
        kwargs: dict[str, Any] = {
            "ping_interval": None,
            "ping_timeout": None,
            "close_timeout": 1,
        }
        if ssl_context is not None:
            kwargs["ssl"] = ssl_context
        return kwargs

    async def run(self, handler: MessageHandler) -> None:
        if websockets is None:
            raise RuntimeError("websockets package is required for live websocket consumption")
        ssl_context = ssl._create_unverified_context() if self.allow_insecure_ssl else None
        async with websockets.connect(self.url, **self._connect_kwargs(ssl_context=ssl_context)) as connection:
            for message in self.build_subscribe_messages():
                await connection.send(json.dumps(message))
            async for raw in connection:
                if raw == "ping":
                    await connection.send("pong")
                    continue
                if raw == "pong":
                    continue
                payload = json.loads(raw)
                for normalized in self.normalize_payload(payload):
                    await handler(normalized)

    async def collect(self, limit: int = 1) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        async def handler(payload: dict[str, Any]) -> None:
            items.append(payload)
            if len(items) >= limit:
                raise asyncio.CancelledError

        try:
            await self.run(handler)
        except asyncio.CancelledError:
            pass
        return items

    def normalize_payload(
        self,
        payload: dict[str, Any],
        *,
        received_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or payload.get("event"):
            return []
        arg = payload.get("arg")
        if not isinstance(arg, dict):
            return []
        channel = str(arg.get("channel") or "")
        symbol = str(arg.get("instId") or "")
        if not channel or not symbol:
            return []
        rows = payload.get("data", [])
        if isinstance(rows, dict):
            row_items: list[object] = [rows]
        elif isinstance(rows, list):
            row_items = rows
        else:
            return []
        normalized: list[dict[str, Any]] = []
        for row in row_items:
            if channel == "trade":
                trade_payload = self._normalize_trade(symbol, row, received_at=received_at)
                if trade_payload is not None:
                    normalized.append(trade_payload)
                continue
            if channel == "ticker":
                normalized.extend(self._normalize_ticker(symbol, row, received_at=received_at))
                continue
            if channel.startswith("candle"):
                kline_payload = self._normalize_candle(
                    symbol,
                    channel,
                    row,
                    received_at=received_at,
                )
                if kline_payload is not None:
                    normalized.append(kline_payload)
        return normalized

    def _normalize_trade(
        self,
        symbol: str,
        row: object,
        *,
        received_at: datetime | None,
    ) -> dict[str, Any] | None:
        fallback_ms = int((received_at or datetime.now(tz=timezone.utc)).timestamp() * 1000)
        if isinstance(row, dict):
            timestamp_ms = _coerce_timestamp_ms(
                row.get("ts") or row.get("cTime") or row.get("uTime"),
                fallback_ms,
            )
            price = row.get("price") or row.get("px") or row.get("p")
            quantity = row.get("size") or row.get("sz") or row.get("q")
            side = str(row.get("side") or row.get("takerSide") or "").lower()
        elif isinstance(row, list) and len(row) >= 4:
            timestamp_ms = _coerce_timestamp_ms(row[0], fallback_ms)
            price = row[1]
            quantity = row[2]
            side = str(row[3]).lower()
        else:
            return None
        if price is None or quantity is None:
            return None
        data: dict[str, Any] = {
            "s": symbol,
            "p": str(price),
            "q": str(quantity),
            "E": timestamp_ms,
            "m": side == "sell",
        }
        if self.market == "futures":
            data["ps"] = symbol
        return {"stream": f"{symbol.lower()}@trade", "data": data}

    def _normalize_ticker(
        self,
        symbol: str,
        row: object,
        *,
        received_at: datetime | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(row, dict):
            return []
        fallback_ms = int((received_at or datetime.now(tz=timezone.utc)).timestamp() * 1000)
        timestamp_ms = _coerce_timestamp_ms(
            row.get("ts") or row.get("systemTime") or row.get("uTime") or row.get("cTime"),
            fallback_ms,
        )
        bid_price = row.get("bidPr") or row.get("bidPrice") or row.get("bestBid")
        bid_qty = row.get("bidSz") or row.get("bidQty") or row.get("bestBidSize")
        ask_price = row.get("askPr") or row.get("askPrice") or row.get("bestAsk")
        ask_qty = row.get("askSz") or row.get("askQty") or row.get("bestAskSize")
        normalized: list[dict[str, Any]] = []
        if bid_price is not None and bid_qty is not None and ask_price is not None and ask_qty is not None:
            data: dict[str, Any] = {
                "s": symbol,
                "b": str(bid_price),
                "B": str(bid_qty),
                "a": str(ask_price),
                "A": str(ask_qty),
                "E": timestamp_ms,
            }
            if self.market == "futures":
                data["ps"] = symbol
            normalized.append({"stream": f"{symbol.lower()}@bookTicker", "data": data})
        if self.market != "futures":
            return normalized
        mark_price = row.get("markPrice") or row.get("markPr") or row.get("lastPr")
        index_price = row.get("indexPrice") or row.get("indexPr") or row.get("lastPr")
        funding_rate = row.get("fundingRate") or row.get("capitalRate")
        if mark_price is not None and index_price is not None:
            normalized.append(
                {
                    "stream": f"{symbol.lower()}@markPrice",
                    "data": {
                        "s": symbol,
                        "p": str(mark_price),
                        "i": str(index_price),
                        "r": str(funding_rate or "0"),
                        "E": timestamp_ms,
                    },
                }
            )
        open_interest = row.get("openInterest") or row.get("holdingAmount") or row.get("size")
        if open_interest is not None:
            normalized.append(
                {
                    "stream": f"{symbol.lower()}@openInterest",
                    "data": {
                        "symbol": symbol,
                        "openInterest": str(open_interest),
                        "time": timestamp_ms,
                    },
                }
            )
        return normalized

    def _normalize_candle(
        self,
        symbol: str,
        channel: str,
        row: object,
        *,
        received_at: datetime | None,
    ) -> dict[str, Any] | None:
        interval = _channel_to_interval(channel)
        interval_ms = _interval_to_milliseconds(interval)
        fallback_ms = int((received_at or datetime.now(tz=timezone.utc)).timestamp() * 1000)
        if isinstance(row, dict):
            start_ms = _coerce_timestamp_ms(
                row.get("ts") or row.get("openTime") or row.get("t"),
                fallback_ms,
            )
            open_price = row.get("open") or row.get("o")
            high_price = row.get("high") or row.get("h")
            low_price = row.get("low") or row.get("l")
            close_price = row.get("close") or row.get("c")
            volume = row.get("baseVolume") or row.get("volume") or row.get("vol") or "0"
            quote_volume = row.get("quoteVolume") or row.get("quoteVol") or row.get("amount") or "0"
            event_time_ms = _coerce_timestamp_ms(
                row.get("ts") or row.get("uTime") or row.get("cTime"),
                fallback_ms,
            )
        elif isinstance(row, list) and len(row) >= 6:
            start_ms = _coerce_timestamp_ms(row[0], fallback_ms)
            open_price = row[1]
            high_price = row[2]
            low_price = row[3]
            close_price = row[4]
            volume = row[5]
            quote_volume = row[6] if len(row) > 6 else "0"
            event_time_ms = fallback_ms
        else:
            return None
        if None in (open_price, high_price, low_price, close_price):
            return None
        close_time_ms = start_ms + interval_ms - 1
        is_closed = event_time_ms >= close_time_ms
        cache_key = (symbol, interval, start_ms)
        if is_closed:
            if cache_key in self._closed_candle_cache:
                return None
            self._closed_candle_cache.add(cache_key)
            if len(self._closed_candle_cache) > 2048:
                self._closed_candle_cache = set(sorted(self._closed_candle_cache)[-1024:])
        data: dict[str, Any] = {
            "s": symbol,
            "k": {
                "i": interval,
                "t": start_ms,
                "T": close_time_ms,
                "o": str(open_price),
                "h": str(high_price),
                "l": str(low_price),
                "c": str(close_price),
                "v": str(volume),
                "q": str(quote_volume),
                "x": is_closed,
            },
        }
        if self.market == "futures":
            data["ps"] = symbol
        return {"stream": f"{symbol.lower()}@kline_{interval}", "data": data}
