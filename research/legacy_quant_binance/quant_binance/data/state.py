from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class TopOfBook:
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    updated_at: datetime


@dataclass(frozen=True)
class SpotTrade:
    symbol: str
    price: float
    quantity: float
    event_time: datetime
    is_buyer_maker: bool


@dataclass(frozen=True)
class KlineBar:
    symbol: str
    interval: str
    start_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    is_closed: bool


@dataclass
class SymbolMarketState:
    symbol: str
    top_of_book: TopOfBook
    last_trade_price: float
    funding_rate: float
    open_interest: float
    basis_bps: float
    last_update_time: datetime
    trades: list[SpotTrade] = field(default_factory=list)
    klines: dict[str, list[KlineBar]] = field(default_factory=dict)
    order_book_imbalance_samples: list[float] = field(default_factory=list)
    funding_rate_samples: list[float] = field(default_factory=list)
    basis_bps_samples: list[float] = field(default_factory=list)
    open_interest_samples: list[float] = field(default_factory=list)

    # Memory bounds — trimmed in prune_samples()
    _MAX_TRADES: int = field(default=500, repr=False)
    _MAX_KLINES_PER_INTERVAL: int = field(default=500, repr=False)
    _MAX_SAMPLES: int = field(default=1000, repr=False)

    def prune_samples(self) -> None:
        """Trim all unbounded lists to prevent memory leaks during long runs."""
        if len(self.trades) > self._MAX_TRADES:
            self.trades[:] = self.trades[-self._MAX_TRADES:]
        for interval, bars in self.klines.items():
            if len(bars) > self._MAX_KLINES_PER_INTERVAL:
                self.klines[interval] = bars[-self._MAX_KLINES_PER_INTERVAL:]
        for samples in (self.order_book_imbalance_samples, self.funding_rate_samples,
                        self.basis_bps_samples, self.open_interest_samples):
            if len(samples) > self._MAX_SAMPLES:
                samples[:] = samples[-self._MAX_SAMPLES:]

    def freshness_ms(self, now: datetime) -> int:
        return int((now - self.last_update_time).total_seconds() * 1000)
