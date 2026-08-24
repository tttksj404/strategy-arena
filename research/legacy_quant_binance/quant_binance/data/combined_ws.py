from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


class CombinedWebSocketClient:
    def __init__(self, clients: list[Any]) -> None:
        self.clients = clients

    async def run(self, handler: MessageHandler) -> None:
        await asyncio.gather(*(client.run(handler) for client in self.clients))
