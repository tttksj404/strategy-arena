from __future__ import annotations

from datetime import datetime

from quant_binance.data.state import KlineBar, SpotTrade, SymbolMarketState, TopOfBook


class MissingMarketStateError(KeyError):
    def __init__(self, *, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(f"missing market state for symbol={symbol}")


class MarketStateStore:
    def __init__(self) -> None:
        self._states: dict[str, SymbolMarketState] = {}
        self._event_counter: int = 0

    def get(self, symbol: str) -> SymbolMarketState | None:
        return self._states.get(symbol)

    def put(self, state: SymbolMarketState) -> None:
        self._states[state.symbol] = state

    def apply_trade(self, trade: SpotTrade) -> SymbolMarketState:
        state = self._require_state(trade.symbol)
        state.last_trade_price = trade.price
        state.last_update_time = trade.event_time
        state.trades.append(trade)
        return state

    def apply_top_of_book(self, symbol: str, top_of_book: TopOfBook) -> SymbolMarketState:
        state = self._require_state(symbol)
        state.top_of_book = top_of_book
        state.last_update_time = top_of_book.updated_at
        imbalance = (top_of_book.bid_qty - top_of_book.ask_qty) / max(
            top_of_book.bid_qty + top_of_book.ask_qty,
            1e-9,
        )
        state.order_book_imbalance_samples.append(imbalance)
        return state

    def apply_kline(self, bar: KlineBar) -> SymbolMarketState:
        state = self._require_state(bar.symbol)
        bucket = state.klines.setdefault(bar.interval, [])
        bucket.append(bar)
        state.last_update_time = bar.close_time
        if bar.close_price > 0:
            state.last_trade_price = bar.close_price
        # Periodic memory pruning (every 500 kline events)
        self._event_counter += 1
        if self._event_counter % 500 == 0:
            state.prune_samples()
        return state

    def apply_mark_price(
        self,
        *,
        symbol: str,
        funding_rate: float,
        mark_price: float,
        index_price: float,
        event_time: datetime,
    ) -> SymbolMarketState:
        state = self._require_state(symbol)
        state.funding_rate = funding_rate
        state.last_trade_price = mark_price
        state.basis_bps = ((mark_price - index_price) / max(index_price, 1e-9)) * 10000.0
        state.last_update_time = event_time
        state.funding_rate_samples.append(funding_rate)
        state.basis_bps_samples.append(state.basis_bps)
        return state

    def apply_open_interest(
        self,
        *,
        symbol: str,
        open_interest: float,
        event_time: datetime,
    ) -> SymbolMarketState:
        state = self._require_state(symbol)
        state.open_interest = open_interest
        state.last_update_time = event_time
        state.open_interest_samples.append(open_interest)
        return state

    def _require_state(self, symbol: str) -> SymbolMarketState:
        if symbol not in self._states:
            raise MissingMarketStateError(symbol=symbol)
        return self._states[symbol]
