from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ..trend_continuation_models import TrendContinuationSelectionManifest
from .bitget_market import BitgetContract, TextFetcher
from .long_forecast import Candle
from .market_timeframes import CandleGranularity
from .trend_continuation_contract import (
    CandidateIdentity,
    Direction,
    ResearchDecision,
    stable_manifest_digest,
)
from .trend_continuation_evaluation import FeePolicySnapshot
from .trend_continuation_features import analyze_trend_continuation
from .trend_continuation_forward_funding import collect_current_funding
from .trend_continuation_history import required_profile_intervals
from .trend_continuation_manifest import JsonObject
from .trend_continuation_market import (
    CandleWindowQuery,
    CandleWindowResult,
    collect_completed_candle_window,
)
from .trend_continuation_settlement import ForwardObservationDraft


@dataclass(frozen=True, slots=True)
class ForwardEvidenceError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ForwardEvidenceCollection:
    draft: ForwardObservationDraft | None
    evidence_time: datetime
    rejection_code: str | None


def collect_manifest_forward_evidence(
    manifest: TrendContinuationSelectionManifest,
    fetch_text: TextFetcher,
    *,
    collected_at: datetime,
) -> ForwardEvidenceCollection:
    if collected_at.tzinfo is None:
        raise ForwardEvidenceError("watcher collection time must be timezone-aware")
    trigger = _trigger_interval(manifest.profile)
    evidence_time = _floor_time(collected_at, trigger)
    delay_seconds = (collected_at - evidence_time).total_seconds()
    if delay_seconds < 0 or delay_seconds > 15:
        return ForwardEvidenceCollection(None, evidence_time, "outside_forward_quote_window")
    fee_policy = _fee_policy(manifest)
    if not fee_policy.valid_at(evidence_time):
        raise ForwardEvidenceError("manifest fee policy is not valid at evidence time")
    contract = BitgetContract(
        manifest.contract_symbol,
        manifest.contract_symbol.removesuffix("USDT"),
        1,
        8,
        fee_policy.taker_fee_rate,
    )
    windows = tuple(
        _collect_window(contract, interval, evidence_time, fetch_text, collected_at)
        for interval in required_profile_intervals(manifest.profile)
    )
    incomplete = next((window.rejection_code for window in windows if not window.eligible), None)
    if incomplete is not None:
        raise ForwardEvidenceError(incomplete)
    funding = collect_current_funding(contract, fetch_text, collected_at=collected_at)
    context_15m = _candles_for(windows, CandleGranularity.FIFTEEN_MINUTES)
    confirmation_5m = _candles_for(windows, CandleGranularity.FIVE_MINUTES)
    trigger_1m = (
        _candles_for(windows, CandleGranularity.ONE_MINUTE)
        if manifest.profile == "triple_leveraged_15m_5m_1m"
        else ()
    )
    decision = analyze_trend_continuation(
        profile=manifest.profile,
        context_15m=context_15m,
        confirmation_5m=confirmation_5m,
        trigger_1m=trigger_1m,
        evidence_time=evidence_time,
        rule_name=manifest.rule_name,
    )
    if (
        decision.decision is not ResearchDecision.CONTINUATION_OBSERVE
        or decision.trend_direction.value != manifest.trend_direction
        or manifest.candidate_horizon_bars not in decision.candidate_horizons
        or decision.reference_price is None
        or decision.invalidation_price is None
    ):
        return ForwardEvidenceCollection(
            None,
            evidence_time,
            decision.rejection_code or "manifest_candidate_not_qualified",
        )
    provenance_rows = [
        {
            "interval": window.provenance.request.query,
            "raw_sha256": window.provenance.raw_sha256,
            "row_count": window.provenance.row_count,
        }
        for window in windows
    ]
    evidence_raw_sha256 = stable_manifest_digest(
        [*provenance_rows, funding.payload(), fee_policy.to_payload()]
    )
    identity = CandidateIdentity(
        manifest.contract_symbol,
        manifest.profile,
        Direction(manifest.trend_direction),
        manifest.rule_name,
        manifest.candidate_horizon_bars,
        manifest.strategy_version,
        manifest.feature_schema_version,
    )
    feature_snapshot: JsonObject = {
        "schema": manifest.feature_schema_version,
        "complete": True,
        "evidence_time": evidence_time.isoformat(),
        "technical_score": decision.score,
        "records": [
            {
                "name": feature.name,
                "formula": feature.formula,
                "window": feature.window,
                "timeframe": feature.timeframe,
                "value": feature.value,
                "direction": feature.direction.value,
                "evidence_time": feature.evidence_time.isoformat(),
                "complete": feature.complete,
            }
            for feature in decision.features
        ],
        "raw_pages": provenance_rows,
    }
    draft = ForwardObservationDraft(
        contract=contract,
        identity=identity,
        evidence_time=evidence_time,
        reference_price=decision.reference_price,
        invalidation_price=decision.invalidation_price,
        trigger_interval_seconds=int(trigger.interval.total_seconds()),
        evidence_raw_sha256=evidence_raw_sha256,
        feature_snapshot=feature_snapshot,
        funding_snapshot=funding.payload(),
        fee_policy_snapshot=fee_policy.to_payload(),
    )
    return ForwardEvidenceCollection(draft, evidence_time, None)


def _collect_window(
    contract: BitgetContract,
    interval: CandleGranularity,
    evidence_time: datetime,
    fetch_text: TextFetcher,
    collected_at: datetime,
) -> CandleWindowResult:
    ends_at = _floor_time(evidence_time, interval)
    starts_at = ends_at - interval.interval * 200
    return collect_completed_candle_window(
        CandleWindowQuery(
            contract=contract,
            granularity=interval,
            starts_at=starts_at,
            ends_at=ends_at,
            endpoint_path="/api/v2/mix/market/candles",
        ),
        fetch_text,
        collected_at=collected_at,
    )


def _candles_for(
    windows: tuple[CandleWindowResult, ...],
    interval: CandleGranularity,
) -> tuple[Candle, ...]:
    return next(
        window.candles
        for window in windows
        if ("granularity", interval.value) in window.provenance.request.query
    )


def _trigger_interval(profile: str) -> CandleGranularity:
    if profile == "standard_15m_5m":
        return CandleGranularity.FIVE_MINUTES
    if profile == "triple_leveraged_15m_5m_1m":
        return CandleGranularity.ONE_MINUTE
    raise ForwardEvidenceError(f"unsupported watcher profile: {profile}")


def _floor_time(value: datetime, interval: CandleGranularity) -> datetime:
    seconds = int(interval.interval.total_seconds())
    timestamp = int(value.timestamp())
    return datetime.fromtimestamp(timestamp - timestamp % seconds, tz=UTC)


def _fee_policy(manifest: TrendContinuationSelectionManifest) -> FeePolicySnapshot:
    payload = manifest.payload.get("fee_policy")
    if not isinstance(payload, dict):
        raise ForwardEvidenceError("manifest fee policy is missing")
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
        raise ForwardEvidenceError("manifest fee policy is malformed") from error
