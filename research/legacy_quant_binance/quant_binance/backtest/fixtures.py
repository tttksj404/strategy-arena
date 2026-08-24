from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from quant_binance.models import FeatureVector, MarketSnapshot


def load_snapshot_fixture(path: str | Path) -> list[MarketSnapshot]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshots: list[MarketSnapshot] = []
    for item in data["snapshots"]:
        features = FeatureVector(**item["feature_values"])
        snapshots.append(
            MarketSnapshot(
                snapshot_id=item["snapshot_id"],
                config_version=item["config_version"],
                snapshot_schema_version=item["snapshot_schema_version"],
                symbol=item["symbol"],
                decision_time=datetime.fromisoformat(item["decision_time"]),
                last_trade_price=item["last_trade_price"],
                best_bid=item["best_bid"],
                best_ask=item["best_ask"],
                funding_rate=item["funding_rate"],
                open_interest=item["open_interest"],
                basis_bps=item["basis_bps"],
                data_freshness_ms=item["data_freshness_ms"],
                feature_values=features,
            )
        )
    return snapshots
