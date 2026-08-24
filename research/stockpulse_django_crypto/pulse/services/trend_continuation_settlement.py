from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from django.core.exceptions import ValidationError
from django.db import transaction

from ..trend_continuation_models import (
    SOURCE_FORWARD,
    TrendContinuationObservation,
    TrendContinuationSelectionManifest,
    TrendContinuationSettlement,
)
from .bitget_market import BitgetContract, TextFetcher
from .trend_continuation_contract import CandidateIdentity, Direction
from .trend_continuation_decimal import decimal_field_string
from .trend_continuation_evaluation import (
    ForwardQuoteSnapshot,
    LabelRequest,
    ObservationSource,
    evaluate_label_window,
)
from .trend_continuation_funding import FundingWindow, collect_funding_window
from .trend_continuation_history import HistoricalCollectionQuery, collect_historical_candles
from .trend_continuation_manifest import JsonObject
from .trend_continuation_market import QuoteObservationQuery, collect_forward_quote
from .trend_continuation_settlement_inputs import (
    ForwardCollectionError,
    fee_policy_from_payload,
    interval_from_seconds,
    quote_snapshot_from_observation,
)
from .trend_continuation_watcher_lifecycle import (
    WatcherLifecyclePhase,
    advance_watcher_lifecycle,
    identity_records,
    manifest_records,
    watcher_lifecycle_phase,
)


SettlementLookup = Callable[[], int]


@dataclass(frozen=True, slots=True)
class SettlementIncompleteError(Exception):
    observation_id: int
    reason: str

    def __str__(self) -> str:
        return f"observation {self.observation_id}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ForwardObservationDraft:
    contract: BitgetContract
    identity: CandidateIdentity
    evidence_time: datetime
    reference_price: float
    invalidation_price: float
    trigger_interval_seconds: int
    evidence_raw_sha256: str
    feature_snapshot: JsonObject
    funding_snapshot: JsonObject
    fee_policy_snapshot: JsonObject


@dataclass(frozen=True, slots=True)
class ForwardObservationCapture:
    observation: TrendContinuationObservation
    created: bool
    settled_count: int


def capture_forward_observation(
    draft: ForwardObservationDraft,
    fetch_text: TextFetcher,
    *,
    watcher_collected_at: datetime,
    settlement_lookup_started_at: datetime,
    settlement_lookup: SettlementLookup,
) -> ForwardObservationCapture:
    existing = identity_records(draft.identity).filter(evidence_time=draft.evidence_time).first()
    if existing is not None:
        return ForwardObservationCapture(existing, False, 0)
    quote_result = collect_forward_quote(
        QuoteObservationQuery(
            contract=draft.contract,
            evidence_time=draft.evidence_time,
            watcher_collected_at=watcher_collected_at,
            settlement_lookup_started_at=settlement_lookup_started_at,
            taker_fee_rate=draft.contract.taker_fee_rate,
        ),
        fetch_text,
    )
    quote = quote_result.quote
    snapshot = ForwardQuoteSnapshot(
        observed_at=quote.observed_at,
        bid_price=quote.bid_price,
        ask_price=quote.ask_price,
        spread_pct=quote.bid_ask_spread_pct,
        raw_sha256=quote_result.provenance.raw_sha256,
        settlement_lookup_started_at=settlement_lookup_started_at,
    )
    if not quote_result.eligible or not snapshot.valid_for(draft.evidence_time):
        raise ForwardCollectionError(quote_result.rejection_code or "invalid_forward_quote")
    defaults = _observation_defaults(draft, snapshot)
    try:
        observation, created = TrendContinuationObservation.objects.get_or_create(
            contract_symbol=draft.identity.contract,
            profile=draft.identity.profile,
            trend_direction=draft.identity.trend_direction.value,
            rule_name=draft.identity.rule_name,
            candidate_horizon_bars=draft.identity.candidate_horizon_bars,
            strategy_version=draft.identity.strategy_version,
            feature_schema_version=draft.identity.feature_schema_version,
            evidence_time=draft.evidence_time,
            defaults=defaults,
        )
    except ValidationError as error:
        raise ForwardCollectionError("invalid_forward_observation") from error
    settled_count = settlement_lookup() if created else 0
    return ForwardObservationCapture(observation, created, settled_count)


def settle_ready_observations(
    manifest: TrendContinuationSelectionManifest,
    fetch_text: TextFetcher,
    *,
    collected_at: datetime,
) -> int:
    settled = 0
    for observation in manifest_records(manifest):
        if observation.expected_label_end_time <= collected_at and not hasattr(observation, "settlement"):
            settle_observation(observation, fetch_text, collected_at=collected_at)
            settled += 1
    return settled


def settle_observation(
    observation: TrendContinuationObservation,
    fetch_text: TextFetcher,
    *,
    collected_at: datetime,
) -> TrendContinuationSettlement:
    if hasattr(observation, "settlement"):
        return observation.settlement
    interval = interval_from_seconds(observation.trigger_interval_seconds)
    contract = BitgetContract(
        observation.contract_symbol,
        observation.underlying_symbol,
        1,
        8,
        observation.fee_rate or 0.0,
    )
    history = collect_historical_candles(
        HistoricalCollectionQuery(contract, interval, observation.evidence_time, observation.expected_label_end_time, 1),
        fetch_text,
    )
    funding_window = FundingWindow(contract, observation.evidence_time, observation.expected_label_end_time)
    funding = collect_funding_window(funding_window, fetch_text, collected_at=collected_at)
    if not history.complete or not funding.eligible:
        raise SettlementIncompleteError(observation.pk or 0, history.rejection_code or funding.rejection_code or "incomplete_provenance")
    request = LabelRequest(
        evidence_time=observation.evidence_time,
        trigger_interval=interval,
        horizon_bars=observation.candidate_horizon_bars,
        direction=Direction(observation.trend_direction),
        reference_price=float(observation.reference_price),
        invalidation_price=float(observation.invalidation_price),
        source=ObservationSource.FORWARD,
        fee_policy=fee_policy_from_payload(observation.fee_policy_snapshot),
        funding=funding.to_evaluation_proof(funding_window),
        quote=quote_snapshot_from_observation(observation),
    )
    row = evaluate_label_window(request, history.candles, settled_as_of=collected_at)
    if not row.resolved or row.terminal_close is None or row.cost_pct is None or row.net_return_pct is None:
        raise SettlementIncompleteError(observation.pk or 0, row.rejection_code or "unresolved_evaluation")
    return record_settlement(
        observation=observation,
        outcome="success" if row.outcome_success else "failure",
        label_end_time=row.label_end_time,
        terminal_price=row.terminal_close,
        cost_pct=row.cost_pct,
        net_return_pct=row.net_return_pct,
        settled_at=collected_at,
        provenance={
            "complete": True,
            "history": history.payload(),
            "funding_raw_sha256": funding.aggregate_raw_sha256,
            "invalidated": row.outcome_success is False and row.net_return_pct > 0,
        },
    )


def record_settlement(
    *,
    observation: TrendContinuationObservation,
    outcome: str,
    label_end_time: datetime,
    terminal_price: float,
    cost_pct: float,
    net_return_pct: float,
    settled_at: datetime,
    provenance: JsonObject,
) -> TrendContinuationSettlement:
    with transaction.atomic():
        return TrendContinuationSettlement.objects.create(
            observation=observation,
            outcome=outcome,
            label_end_time=label_end_time,
            terminal_price=terminal_price,
            cost_pct=cost_pct,
            net_return_pct=net_return_pct,
            settled_at=settled_at,
            provenance=provenance,
        )


def _observation_defaults(draft: ForwardObservationDraft, quote: ForwardQuoteSnapshot) -> JsonObject:
    return {
        "underlying_symbol": draft.contract.underlying_symbol,
        "reference_price": decimal_field_string(draft.reference_price),
        "invalidation_price": decimal_field_string(draft.invalidation_price),
        "trigger_interval_seconds": draft.trigger_interval_seconds,
        "source_kind": SOURCE_FORWARD,
        "quote_observed_at": quote.observed_at,
        "bid_price": decimal_field_string(quote.bid_price),
        "ask_price": decimal_field_string(quote.ask_price),
        "bid_ask_spread_pct": quote.spread_pct,
        "fee_rate": draft.contract.taker_fee_rate,
        "raw_response_sha256": draft.evidence_raw_sha256,
        "provenance": {"complete": True, "evidence_time": draft.evidence_time.isoformat()},
        "feature_snapshot": draft.feature_snapshot,
        "quote_snapshot": {
            "observed_at": quote.observed_at.isoformat(),
            "raw_sha256": quote.raw_sha256,
            "settlement_lookup_started_at": quote.settlement_lookup_started_at.isoformat() if quote.settlement_lookup_started_at else None,
        },
        "funding_snapshot": draft.funding_snapshot,
        "fee_policy_snapshot": draft.fee_policy_snapshot,
    }
