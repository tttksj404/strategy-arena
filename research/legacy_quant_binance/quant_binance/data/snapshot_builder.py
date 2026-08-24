from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import uuid4

from quant_binance.data.state import SymbolMarketState
from quant_binance.models import FeatureVector, MarketSnapshot
from quant_binance.settings import Settings
from quant_binance.snapshots import validate_snapshot


class SnapshotBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_decision_boundary(self, timestamp: datetime) -> bool:
        interval_seconds = max(
            int(
                self.settings.decision_engine.decision_interval_seconds
                or self.settings.decision_engine.decision_interval_minutes * 60
            ),
            1,
        )
        return timestamp.microsecond == 0 and int(timestamp.timestamp()) % interval_seconds == 0

    def build(
        self,
        state: SymbolMarketState,
        features: FeatureVector,
        decision_time: datetime,
    ) -> MarketSnapshot:
        if not self.is_decision_boundary(decision_time):
            raise ValueError("decision_time must align to the configured decision boundary")

        snapshot = MarketSnapshot(
            snapshot_id=str(uuid4()),
            config_version=self.settings.config_version,
            snapshot_schema_version=self.settings.snapshot_schema_version,
            symbol=state.symbol,
            decision_time=decision_time,
            last_trade_price=state.last_trade_price,
            best_bid=state.top_of_book.bid_price,
            best_ask=state.top_of_book.ask_price,
            funding_rate=state.funding_rate,
            open_interest=state.open_interest,
            basis_bps=state.basis_bps,
            data_freshness_ms=state.freshness_ms(decision_time),
            feature_values=features,
        )
        validate_snapshot(snapshot)
        return snapshot
