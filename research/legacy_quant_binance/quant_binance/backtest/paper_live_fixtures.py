from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quant_binance.data.state import KlineBar, SpotTrade, SymbolMarketState, TopOfBook
from quant_binance.features.primitive import FeatureHistoryContext, PrimitiveInputs


@dataclass(frozen=True)
class PaperLiveCycle:
    decision_time: datetime
    symbol: str
    state: SymbolMarketState
    primitive_inputs: PrimitiveInputs
    history: FeatureHistoryContext


def load_paper_live_cycles(path: str | Path) -> list[PaperLiveCycle]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cycles: list[PaperLiveCycle] = []
    for item in payload["cycles"]:
        decision_time = datetime.fromisoformat(item["decision_time"])
        state_payload = item["state"]
        top = state_payload["top_of_book"]
        trades = [
            SpotTrade(
                symbol=trade["symbol"],
                price=trade["price"],
                quantity=trade["quantity"],
                event_time=datetime.fromisoformat(trade["event_time"]),
                is_buyer_maker=trade["is_buyer_maker"],
            )
            for trade in state_payload.get("trades", [])
        ]
        klines = {
            interval: [
                KlineBar(
                    symbol=bar["symbol"],
                    interval=bar["interval"],
                    start_time=datetime.fromisoformat(bar["start_time"]),
                    close_time=datetime.fromisoformat(bar["close_time"]),
                    open_price=bar["open_price"],
                    high_price=bar["high_price"],
                    low_price=bar["low_price"],
                    close_price=bar["close_price"],
                    volume=bar["volume"],
                    quote_volume=bar["quote_volume"],
                    is_closed=bar["is_closed"],
                )
                for bar in bars
            ]
            for interval, bars in state_payload.get("klines", {}).items()
        }
        state = SymbolMarketState(
            symbol=item["symbol"],
            top_of_book=TopOfBook(
                bid_price=top["bid_price"],
                bid_qty=top["bid_qty"],
                ask_price=top["ask_price"],
                ask_qty=top["ask_qty"],
                updated_at=datetime.fromisoformat(top["updated_at"]),
            ),
            last_trade_price=state_payload["last_trade_price"],
            funding_rate=state_payload["funding_rate"],
            open_interest=state_payload["open_interest"],
            basis_bps=state_payload["basis_bps"],
            last_update_time=datetime.fromisoformat(state_payload["last_update_time"]),
            trades=trades,
            klines=klines,
            order_book_imbalance_samples=list(state_payload.get("order_book_imbalance_samples", [])),
            funding_rate_samples=list(state_payload.get("funding_rate_samples", [])),
            basis_bps_samples=list(state_payload.get("basis_bps_samples", [])),
            open_interest_samples=list(state_payload.get("open_interest_samples", [])),
        )
        cycles.append(
            PaperLiveCycle(
                decision_time=decision_time,
                symbol=item["symbol"],
                state=state,
                primitive_inputs=PrimitiveInputs(**item["primitive_inputs"]),
                history=FeatureHistoryContext(**item["history"]),
            )
        )
    return cycles
