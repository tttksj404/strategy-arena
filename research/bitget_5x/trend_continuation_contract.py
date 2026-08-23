from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final


SCHEMA_VERSION: Final = "trend-continuation.v1"
DEFAULT_TRIGGER_INTERVAL_SECONDS: Final = 300
FORBIDDEN_PAYLOAD_KEYS: Final = frozenset(
    {
        "action",
        "entry",
        "enter_now",
        "estimated_20x",
        "estimated_5x",
        "long",
        "order",
        "position",
        "short",
        "sell",
        "stop_loss",
        "target",
        "take_profit",
        "target_price",
    }
)
FORBIDDEN_PAYLOAD_VALUES: Final = frozenset(
    {
        "ENTER_LONG",
        "WATCH_LONG_SETUP",
        "BUY",
        "SELL",
        "buy",
        "sell",
    }
)


class Direction(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class ResearchDecision(StrEnum):
    NO_TRADE = "NO_TRADE"
    CONTINUATION_OBSERVE = "CONTINUATION_OBSERVE"


class HorizonStatus(StrEnum):
    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    contract: str
    profile: str
    trend_direction: Direction
    rule_name: str
    candidate_horizon_bars: int
    strategy_version: str
    feature_schema_version: str

    def __post_init__(self) -> None:
        if self.candidate_horizon_bars <= 0:
            raise ValueError("candidate_horizon_bars must be positive")


@dataclass(frozen=True, slots=True)
class PublicMarketRequest:
    method: str
    path: str
    query: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    ALLOWED_PATHS: Final = frozenset(
        {
            "/api/v2/mix/market/contracts",
            "/api/v2/mix/market/candles",
            "/api/v2/mix/market/history-candles",
            "/api/v2/mix/market/ticker",
            "/api/v2/mix/market/current-fund-rate",
            "/api/v2/mix/market/history-fund-rate",
        }
    )

    def __post_init__(self) -> None:
        if self.method.upper() != "GET":
            raise ValueError("only public GET requests are allowed")
        if self.path not in self.ALLOWED_PATHS:
            raise ValueError(f"blocked public market path: {self.path}")


def stable_manifest_digest(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class TrendContinuationDiagnostic:
    identity: CandidateIdentity
    as_of: datetime
    evidence_time: datetime
    trigger_interval_seconds: int
    reference_price: float
    invalidation_price: float
    decision: ResearchDecision
    horizon_status: HorizonStatus
    rejection_code: str | None
    all_observe_gates_passed: bool = False
    holdout_empirical_continuation_rate_pct: float | None = None
    wilson_lower_95: float | None = None
    mean_return_lower_95: float | None = None
    holdout_sample_count: int | None = None
    forward_provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.trigger_interval_seconds <= 0:
            raise ValueError("trigger_interval_seconds must be positive")
        if self.reference_price <= 0 or self.invalidation_price <= 0:
            raise ValueError("prices must be positive")
        if self.decision is ResearchDecision.CONTINUATION_OBSERVE and not self.all_observe_gates_passed:
            raise ValueError("CONTINUATION_OBSERVE requires every gate")
        if self.decision is ResearchDecision.NO_TRADE and self.horizon_status is HorizonStatus.VALIDATED:
            raise ValueError("NO_TRADE diagnostics cannot be validated")

    @property
    def label_end_time(self) -> datetime:
        return self.evidence_time + timedelta(seconds=self.trigger_interval_seconds * self.identity.candidate_horizon_bars)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SCHEMA_VERSION,
            "mode": "research_only",
            "decision": self.decision.value,
            "trend_direction": self.identity.trend_direction.value,
            "horizon_status": self.horizon_status.value,
            "candidate_horizon_bars": self.identity.candidate_horizon_bars,
            "strategy_version": self.identity.strategy_version,
            "feature_schema_version": self.identity.feature_schema_version,
            "rule_name": self.identity.rule_name,
            "contract": self.identity.contract,
            "profile": self.identity.profile,
            "reference_price": self.reference_price,
            "invalidation_price": self.invalidation_price,
            "evidence_time": self.evidence_time.isoformat(),
            "label_end_time": self.label_end_time.isoformat(),
            "as_of": self.as_of.isoformat(),
            "execution": {"enabled": False},
            "data_policy": {"closed_candles_only": True},
            "research_safety": {"public_market_data_only": True},
            "provenance": {
                "source": "public_bitget_market",
                "rejection_code": self.rejection_code,
            },
        }
        if self.forward_provenance:
            payload["provenance"]["forward"] = self.forward_provenance
        if self.decision is ResearchDecision.CONTINUATION_OBSERVE and self.holdout_sample_count is not None:
            payload["holdout_empirical_continuation_rate_pct"] = self.holdout_empirical_continuation_rate_pct
            payload["wilson_lower_95"] = self.wilson_lower_95
            payload["mean_return_lower_95"] = self.mean_return_lower_95
            payload["holdout_sample_count"] = self.holdout_sample_count
        _assert_research_only_payload(payload)
        return payload


def default_no_trade_diagnostic(
    *,
    identity: CandidateIdentity,
    as_of: datetime,
    evidence_time: datetime,
    trigger_interval_seconds: int = DEFAULT_TRIGGER_INTERVAL_SECONDS,
    reference_price: float,
    invalidation_price: float,
    rejection_code: str,
) -> TrendContinuationDiagnostic:
    return TrendContinuationDiagnostic(
        identity=identity,
        as_of=as_of,
        evidence_time=evidence_time,
        trigger_interval_seconds=trigger_interval_seconds,
        reference_price=reference_price,
        invalidation_price=invalidation_price,
        decision=ResearchDecision.NO_TRADE,
        horizon_status=HorizonStatus.PENDING_VALIDATION,
        rejection_code=rejection_code,
    )


def _assert_research_only_payload(payload: object) -> None:
    match payload:
        case dict():
            for key, value in payload.items():
                if key in FORBIDDEN_PAYLOAD_KEYS:
                    raise ValueError(f"forbidden payload key: {key}")
                _assert_research_only_payload(value)
        case list() | tuple():
            for item in payload:
                _assert_research_only_payload(item)
        case str() as value:
            if value in FORBIDDEN_PAYLOAD_VALUES:
                raise ValueError(f"forbidden payload value: {value}")
        case _:
            return
