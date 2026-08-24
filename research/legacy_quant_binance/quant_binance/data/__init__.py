"""Binance market data adapter scaffolding."""

from .futures_stream import BINANCE_FUTURES_WS_URL, build_futures_streams
from .market_store import MarketStateStore
from .snapshot_builder import SnapshotBuilder
from .spot_stream import BINANCE_SPOT_WS_URL, build_spot_streams
from .state import KlineBar, SpotTrade, SymbolMarketState, TopOfBook

__all__ = [
    "BINANCE_FUTURES_WS_URL",
    "BINANCE_SPOT_WS_URL",
    "KlineBar",
    "MarketStateStore",
    "SnapshotBuilder",
    "SpotTrade",
    "SymbolMarketState",
    "TopOfBook",
    "build_futures_streams",
    "build_spot_streams",
]
