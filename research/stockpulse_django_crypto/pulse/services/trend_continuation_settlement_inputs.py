from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..trend_continuation_models import TrendContinuationObservation
from .market_timeframes import CandleGranularity
from .trend_continuation_evaluation import FeePolicySnapshot, ForwardQuoteSnapshot
from .trend_continuation_manifest import JsonObject


@dataclass(frozen=True, slots=True)
class ForwardCollectionError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


def interval_from_seconds(seconds: int) -> CandleGranularity:
    mapping = {
        60: CandleGranularity.ONE_MINUTE,
        300: CandleGranularity.FIVE_MINUTES,
        900: CandleGranularity.FIFTEEN_MINUTES,
    }
    try:
        return mapping[seconds]
    except KeyError as error:
        raise ForwardCollectionError(f"unsupported trigger interval seconds: {seconds}") from error


def fee_policy_from_payload(payload: JsonObject) -> FeePolicySnapshot:
    try:
        return FeePolicySnapshot(
            version=str(payload["version"]),
            recorded_at=datetime.fromisoformat(str(payload["recorded_at"])),
            source_url=str(payload["source_url"]),
            raw_sha256=str(payload["raw_sha256"]),
            effective_from=datetime.fromisoformat(str(payload["effective_from"])),
            effective_until=datetime.fromisoformat(str(payload["effective_until"])),
            taker_fee_rate=float(payload["taker_fee_rate"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ForwardCollectionError("malformed fee policy snapshot") from error


def quote_snapshot_from_observation(
    observation: TrendContinuationObservation,
) -> ForwardQuoteSnapshot:
    lookup = observation.quote_snapshot.get("settlement_lookup_started_at")
    if (
        observation.quote_observed_at is None
        or observation.bid_price is None
        or observation.ask_price is None
        or observation.bid_ask_spread_pct is None
    ):
        raise ForwardCollectionError("missing quote snapshot")
    return ForwardQuoteSnapshot(
        observed_at=observation.quote_observed_at,
        bid_price=float(observation.bid_price),
        ask_price=float(observation.ask_price),
        spread_pct=observation.bid_ask_spread_pct,
        raw_sha256=str(observation.quote_snapshot.get("raw_sha256", "")),
        settlement_lookup_started_at=datetime.fromisoformat(str(lookup)) if lookup else None,
    )
