from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from quant_binance.data.futures_stream import BINANCE_FUTURES_WS_URL
from quant_binance.data.spot_stream import BINANCE_SPOT_WS_URL

try:
    import websockets
except ImportError:  # pragma: no cover - optional dependency
    websockets = None


MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


class BinanceWebSocketClient:
    def __init__(
        self,
        *,
        market: str,
        streams: list[str],
        allow_insecure_ssl: bool = False,
        label: str = "",
    ) -> None:
        self.market = market
        self.streams = streams
        self.allow_insecure_ssl = allow_insecure_ssl
        self.label = label or market

    @property
    def url(self) -> str:
        base = BINANCE_FUTURES_WS_URL if self.market == "futures" else BINANCE_SPOT_WS_URL
        return f"{base}?streams={'/'.join(self.streams)}"

    async def run(self, handler: MessageHandler) -> None:
        if websockets is None:
            raise RuntimeError("websockets package is required for live websocket consumption")
        ssl_context = ssl._create_unverified_context() if self.allow_insecure_ssl else None
        async with websockets.connect(self.url, ssl=ssl_context) as connection:
            async for raw in connection:
                payload = json.loads(raw)
                await handler(payload)

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
