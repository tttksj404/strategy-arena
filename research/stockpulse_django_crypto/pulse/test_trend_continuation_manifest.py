from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from django.core.exceptions import ValidationError
from django.db import close_old_connections
from django.test import TestCase, TransactionTestCase

from .services.trend_continuation_contract import CandidateIdentity, Direction, ResearchDecision
from .services.trend_continuation_evaluation import EvaluationResult
from .services.trend_continuation_manifest import (
    ForwardIntervalOverlapError,
    ManifestLifecycleError,
    TrendContinuationSelectionSnapshot,
    append_holdout_result,
    append_validation_result,
    freeze_selection,
    register_forward_consumption,
)
from .services.trend_continuation_evaluation import choose_diagnostic
from .trend_continuation_models import (
    TrendContinuationForwardConsumption,
    TrendContinuationHoldoutResult,
    TrendContinuationManifestRejection,
    TrendContinuationObservation,
    TrendContinuationSelectionManifest,
    TrendContinuationSettlement,
    TrendContinuationValidationResult,
)


BASE = datetime(2026, 2, 1, tzinfo=UTC)


def _identity(contract: str = "TSLAUSDT", *, version: str = "trend-v1") -> CandidateIdentity:
    return CandidateIdentity(
        contract=contract,
        profile="standard_15m_5m",
        trend_direction=Direction.UP,
        rule_name="technical_score_v1",
        candidate_horizon_bars=1,
        strategy_version=version,
        feature_schema_version="features-v1",
    )


def _snapshot(identity: CandidateIdentity, *, nonce: str = "a") -> TrendContinuationSelectionSnapshot:
    return TrendContinuationSelectionSnapshot(
        payload={
            "schema": "trend-continuation-selection.v1",
            "exploratory_only": True,
            "nonce": nonce,
            "candidate": {
                "contract": identity.contract,
                "profile": identity.profile,
                "trend_direction": identity.trend_direction.value,
                "rule_name": identity.rule_name,
                "candidate_horizon_bars": identity.candidate_horizon_bars,
                "strategy_version": identity.strategy_version,
                "feature_schema_version": identity.feature_schema_version,
            },
            "fee_policy": {
                "version": "fee-v1",
                "recorded_at": (BASE - timedelta(days=2)).isoformat(),
                "source_url": "https://www.bitget.com/api-doc/contract/market/Get-All-Symbols-Contracts",
                "raw_sha256": "a" * 64,
                "effective_from": (BASE - timedelta(days=2)).isoformat(),
                "effective_until": (BASE + timedelta(days=30)).isoformat(),
                "taker_fee_rate": 0.0006,
            },
        }
    )


def _observation(identity: CandidateIdentity, evidence: datetime, *, historical: bool = False) -> TrendContinuationObservation:
    quote = {} if historical else {"raw_sha256": "b" * 64, "observed_at": evidence.isoformat()}
    return TrendContinuationObservation.objects.create(
        contract_symbol=identity.contract,
        underlying_symbol=identity.contract.removesuffix("USDT"),
        profile=identity.profile,
        trend_direction=identity.trend_direction.value,
        rule_name=identity.rule_name,
        candidate_horizon_bars=identity.candidate_horizon_bars,
        strategy_version=identity.strategy_version,
        feature_schema_version=identity.feature_schema_version,
        evidence_time=evidence,
        reference_price=Decimal("100"),
        invalidation_price=Decimal("99"),
        trigger_interval_seconds=300,
        source_kind="historical_exploratory" if historical else "forward",
        quote_observed_at=None if historical else evidence,
        bid_price=None if historical else Decimal("99.9"),
        ask_price=None if historical else Decimal("100.1"),
        bid_ask_spread_pct=None if historical else 0.2,
        fee_rate=0.0006,
        raw_response_sha256="c" * 64,
        provenance={"complete": True},
        feature_snapshot={"schema": "features-v1"},
        quote_snapshot=quote,
        funding_snapshot={"complete": True, "events": []},
        fee_policy_snapshot={"version": "fee-v1", "raw_sha256": "a" * 64},
    )


def _settle(observation: TrendContinuationObservation, *, success: bool = True) -> TrendContinuationSettlement:
    return TrendContinuationSettlement.objects.create(
        observation=observation,
        outcome="success" if success else "failure",
        label_end_time=observation.evidence_time + timedelta(minutes=5),
        terminal_price=Decimal("101" if success else "99"),
        cost_pct=0.32,
        net_return_pct=1.0 if success else -1.0,
        settled_at=observation.evidence_time + timedelta(minutes=6),
        provenance={"complete": True},
    )


class ImmutablePersistenceTests(TestCase):
    def _resolved_records(
        self,
        identity: CandidateIdentity,
        count: int,
        *,
        starts_at: datetime,
        success: bool = True,
    ) -> tuple[TrendContinuationObservation, ...]:
        records = []
        for index in range(count):
            observation = _observation(identity, starts_at + timedelta(minutes=index * 10))
            _settle(observation, success=success)
            records.append(observation)
        return tuple(records)

    def test_manifest_freeze_detaches_payload_and_rejects_invalid_fee_policy(self):
        # Given: a valid snapshot and a second snapshot with an unproven fee-policy hash.
        identity = _identity()
        snapshot = _snapshot(identity)
        invalid = _snapshot(_identity("RGTIUSDT"))
        invalid.payload["fee_policy"]["raw_sha256"] = "short"

        # When: the valid snapshot is frozen and the caller mutates its original nested dict.
        manifest = freeze_selection(snapshot, frozen_at=BASE)
        snapshot.payload["candidate"]["rule_name"] = "mutated_after_freeze"

        # Then: persisted bytes and digest stay detached, while invalid policy evidence cannot freeze.
        manifest.refresh_from_db()
        self.assertEqual(manifest.payload["candidate"]["rule_name"], "technical_score_v1")
        with self.assertRaises(ValidationError):
            freeze_selection(invalid, frozen_at=BASE)

    def test_historical_observation_rejects_quote_or_spread_and_json_round_trips(self):
        # Given: an otherwise valid historical observation with a forbidden observed spread.
        identity = _identity()
        invalid = TrendContinuationObservation(
            contract_symbol=identity.contract,
            underlying_symbol="TSLA",
            profile=identity.profile,
            trend_direction=identity.trend_direction.value,
            rule_name=identity.rule_name,
            candidate_horizon_bars=1,
            strategy_version=identity.strategy_version,
            feature_schema_version=identity.feature_schema_version,
            evidence_time=BASE + timedelta(days=1),
            reference_price=100,
            invalidation_price=99,
            trigger_interval_seconds=300,
            source_kind="historical_exploratory",
            bid_ask_spread_pct=0.1,
            provenance={"nested": {"rows": [1, 2]}},
            feature_snapshot={"schema": "features-v1"},
            funding_snapshot={"complete": True},
            fee_policy_snapshot={"version": "fee-v1"},
        )

        # When / Then: the source boundary rejects forward quote fields.
        with self.assertRaises(ValidationError):
            invalid.save()

        # And: a clean historical record preserves nested provenance exactly.
        saved = _observation(identity, BASE + timedelta(days=1), historical=True)
        saved.refresh_from_db()
        self.assertEqual(saved.feature_snapshot, {"schema": "features-v1"})
        self.assertEqual(saved.funding_snapshot, {"complete": True, "events": []})

    def test_instance_and_queryset_mutations_and_deletes_are_blocked(self):
        # Given: a persisted immutable observation and settlement.
        observation = _observation(_identity(), BASE + timedelta(days=1))
        settlement = _settle(observation)

        # When / Then: instance save/delete and bulk domain paths cannot rewrite audit history.
        observation.reference_price = Decimal("200")
        with self.assertRaises(ValueError):
            observation.save()
        with self.assertRaises(ValueError):
            settlement.delete()
        with self.assertRaises(ValueError):
            TrendContinuationObservation.objects.filter(pk=observation.pk).update(reference_price=200)
        with self.assertRaises(ValueError):
            TrendContinuationObservation.objects.filter(pk=observation.pk).delete()
