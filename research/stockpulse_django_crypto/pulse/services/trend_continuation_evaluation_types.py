from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite
from typing import Callable, Final

from .market_timeframes import CandleGranularity
from .trend_continuation_contract import CandidateIdentity, Direction
from .trend_continuation_features import TrendContinuationFeatures


Z_95: Final = 1.959963984540054
HISTORICAL_EXPLORATORY_COST_FLOOR_PCT: Final = 0.20
MIN_SELECTION_OBSERVATIONS: Final = 80
FINAL_HOLDOUT_OBSERVATIONS: Final = 20
MIN_WILSON_LOWER_95_PCT: Final = 50.0
STANDARD_EMBARGO: Final = timedelta(hours=14)
TRIPLE_EMBARGO: Final = timedelta(hours=12, minutes=45)
FORWARD_QUOTE_MAX_DELAY: Final = timedelta(seconds=15)


class ObservationSource(StrEnum):
    HISTORICAL_EXPLORATORY = "historical_exploratory"
    FORWARD = "forward"


@dataclass(frozen=True, slots=True)
class FeePolicySnapshot:
    version: str
    recorded_at: datetime
    source_url: str
    raw_sha256: str
    effective_from: datetime
    effective_until: datetime
    taker_fee_rate: float

    def valid_at(self, evidence_time: datetime) -> bool:
        timestamps = (self.recorded_at, self.effective_from, self.effective_until, evidence_time)
        return (
            bool(self.version)
            and self.source_url.startswith("https://")
            and len(self.raw_sha256) == 64
            and all(character in "0123456789abcdefABCDEF" for character in self.raw_sha256)
            and all(value.tzinfo is not None for value in timestamps)
            and self.recorded_at <= evidence_time
            and self.effective_from <= evidence_time < self.effective_until
            and isfinite(self.taker_fee_rate)
            and self.taker_fee_rate >= 0
        )

    def to_payload(self) -> dict[str, str | float]:
        return {
            "version": self.version,
            "recorded_at": self.recorded_at.isoformat(),
            "source_url": self.source_url,
            "raw_sha256": self.raw_sha256,
            "effective_from": self.effective_from.isoformat(),
            "effective_until": self.effective_until.isoformat(),
            "taker_fee_rate": self.taker_fee_rate,
        }


@dataclass(frozen=True, slots=True)
class FundingRateEvent:
    funding_time: datetime
    funding_rate: float


@dataclass(frozen=True, slots=True)
class FundingProof:
    complete: bool
    events: tuple[FundingRateEvent, ...]
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None
    provenance_digest: str | None = None
    page_count: int = 0
    row_count: int = 0

    def covers(self, evidence_time: datetime, label_end_time: datetime) -> bool:
        return (
            self.complete
            and self.coverage_start is not None
            and self.coverage_end is not None
            and self.provenance_digest is not None
            and len(self.provenance_digest) == 64
            and all(character in "0123456789abcdefABCDEF" for character in self.provenance_digest)
            and self.page_count >= 1
            and self.row_count == len(self.events)
            and self.coverage_start.tzinfo is not None
            and self.coverage_end.tzinfo is not None
            and self.coverage_start <= evidence_time
            and self.coverage_end >= label_end_time
        )


@dataclass(frozen=True, slots=True)
class ForwardQuoteSnapshot:
    observed_at: datetime
    bid_price: float
    ask_price: float
    spread_pct: float
    raw_sha256: str
    settlement_lookup_started_at: datetime | None = None

    def valid_for(self, evidence_time: datetime) -> bool:
        return (
            self.observed_at.tzinfo is not None
            and evidence_time <= self.observed_at <= evidence_time + FORWARD_QUOTE_MAX_DELAY
            and isfinite(self.bid_price)
            and isfinite(self.ask_price)
            and isfinite(self.spread_pct)
            and self.bid_price > 0
            and self.ask_price >= self.bid_price
            and self.spread_pct >= 0
            and len(self.raw_sha256) == 64
            and all(character in "0123456789abcdefABCDEF" for character in self.raw_sha256)
            and self.settlement_lookup_started_at is not None
            and self.settlement_lookup_started_at.tzinfo is not None
            and self.settlement_lookup_started_at > self.observed_at
        )


@dataclass(frozen=True, slots=True)
class LabelRequest:
    evidence_time: datetime
    trigger_interval: CandleGranularity
    horizon_bars: int
    direction: Direction
    reference_price: float
    invalidation_price: float
    source: ObservationSource
    fee_policy: FeePolicySnapshot | None
    funding: FundingProof
    quote: ForwardQuoteSnapshot | None = None

    @property
    def label_end_time(self) -> datetime:
        return self.evidence_time + self.trigger_interval.interval * self.horizon_bars


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    evidence_time: datetime
    label_end_time: datetime
    reference_price: float
    invalidation_price: float
    outcome_success: bool | None
    net_return_pct: float | None
    terminal_close: float | None = None
    cost_pct: float | None = None
    source: ObservationSource = ObservationSource.HISTORICAL_EXPLORATORY
    rejection_code: str | None = None
    features: TrendContinuationFeatures | None = None

    @property
    def resolved(self) -> bool:
        values = (self.net_return_pct, self.terminal_close, self.cost_pct)
        return (
            self.outcome_success is not None
            and all(value is not None and isfinite(value) for value in values)
            and self.rejection_code is None
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    resolved_count: int
    success_count: int
    wilson_lower_95: float
    mean_return_lower_95: float
    mean_return_pct: float
    statistics_valid: bool


@dataclass(frozen=True, slots=True)
class DevelopmentFold:
    boundary: datetime
    next_boundary: datetime
    embargo_duration: timedelta
    training: tuple[EvaluationRow, ...]
    validation: tuple[EvaluationRow, ...]


@dataclass(frozen=True, slots=True)
class CandidateDevelopment:
    identity: CandidateIdentity
    metrics: EvaluationResult
    split_boundaries: tuple[datetime, ...]
    historical_audit_start: datetime


@dataclass(frozen=True, slots=True)
class HistoricalSelection:
    identity: CandidateIdentity
    metrics: EvaluationResult
    split_boundaries: tuple[datetime, ...]
    historical_audit_start: datetime
    exploratory_only: bool = True
    promotable: bool = False


FundingProvider = Callable[[datetime, datetime], FundingProof]
