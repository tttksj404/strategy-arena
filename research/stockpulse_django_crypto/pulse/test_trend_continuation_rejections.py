from __future__ import annotations

from datetime import timedelta

from django.core.exceptions import ValidationError
from django.test import TestCase

from .services.trend_continuation_manifest import (
    append_holdout_result,
    append_validation_result,
    freeze_selection,
    register_forward_consumption,
)
from .test_trend_continuation_manifest import BASE, _identity, _observation, _settle, _snapshot
from .trend_continuation_models import (
    TrendContinuationForwardConsumption,
    TrendContinuationObservation,
)


class RejectedBatchConsumptionTests(TestCase):
    def _records(
        self,
        contract: str,
        count: int,
        *,
        historical: bool = False,
        settled: bool = True,
        spacing_minutes: int = 10,
        version: str = "trend-v1",
        success: bool = True,
    ) -> tuple[TrendContinuationObservation, ...]:
        identity = _identity(contract, version=version)
        records = []
        for index in range(count):
            observation = _observation(
                identity,
                BASE + timedelta(days=1, minutes=index * spacing_minutes),
                historical=historical,
            )
            if settled:
                _settle(observation, success=success)
            records.append(observation)
        return tuple(records)

    def test_candidate_mismatch_leaves_consumption_ledger_empty(self):
        # Given: a v1 manifest and 80 resolved records from a different strategy version.
        manifest = freeze_selection(_snapshot(_identity("IBMUSDT")), frozen_at=BASE)
        records = self._records("IBMUSDT", 80, version="trend-v2")

        # When: the mismatched batch is submitted to validation.
        result = append_validation_result(manifest, records)

        # Then: the terminal rejection reserves no accepted forward interval.
        self.assertEqual(result.rejection_code, "candidate_identity_mismatch")
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 0)

    def test_historical_source_leaves_consumption_ledger_empty(self):
        # Given: 80 resolved historical exploratory records.
        manifest = freeze_selection(_snapshot(_identity("QQQUSDT")), frozen_at=BASE)
        records = self._records("QQQUSDT", 80, historical=True)

        # When: the historical batch is submitted to forward validation.
        result = append_validation_result(manifest, records)

        # Then: historical evidence is rejected without reserving forward intervals.
        self.assertEqual(result.rejection_code, "forward_result_requires_forward_source")
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 0)

    def test_unresolved_source_leaves_consumption_ledger_empty(self):
        # Given: 80 forward observations with no settlement rows.
        manifest = freeze_selection(_snapshot(_identity("AMDUSDT")), frozen_at=BASE)
        records = self._records("AMDUSDT", 80, settled=False)

        # When: the unresolved batch is submitted to validation.
        result = append_validation_result(manifest, records)

        # Then: unresolved rows produce a terminal result without any consumption.
        self.assertEqual(result.rejection_code, "unresolved_observation")
        self.assertEqual(result.resolved_count, 0)
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 0)

    def test_stream_overlap_leaves_consumption_ledger_empty(self):
        # Given: consecutive closed intervals sharing every five-minute endpoint.
        manifest = freeze_selection(_snapshot(_identity("NVDAUSDT")), frozen_at=BASE)
        records = self._records("NVDAUSDT", 80, spacing_minutes=5)

        # When: the internally overlapping stream is submitted to validation.
        result = append_validation_result(manifest, records)

        # Then: the whole invalid batch is rejected before global registration.
        self.assertEqual(result.rejection_code, "stream_interval_overlap")
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 0)

    def test_wrong_count_leaves_consumption_ledger_empty(self):
        # Given: only 79 otherwise admissible forward records.
        manifest = freeze_selection(_snapshot(_identity("PLTRUSDT")), frozen_at=BASE)
        records = self._records("PLTRUSDT", 79)

        # When: the under-sized validation batch is accessed.
        result = append_validation_result(manifest, records)

        # Then: count rejection does not poison the global interval ledger.
        self.assertEqual(result.rejection_code, "validation_requires_exactly_80")
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 0)

    def test_failed_metrics_leave_consumption_ledger_empty(self):
        manifest = freeze_selection(_snapshot(_identity("METAUSDT")), frozen_at=BASE)
        records = self._records("METAUSDT", 80, success=False)

        result = append_validation_result(manifest, records)

        self.assertEqual(result.rejection_code, "validation_fixed_metrics_failed")
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 0)

    def test_rejected_holdout_leaves_holdout_consumption_empty(self):
        # Given: a passed validation followed by only 19 later records.
        manifest = freeze_selection(_snapshot(_identity("MSTRUSDT")), frozen_at=BASE)
        validation_records = self._records("MSTRUSDT", 80)
        append_validation_result(manifest, validation_records)
        holdout_records = tuple(
            _observation(
                _identity("MSTRUSDT"),
                BASE + timedelta(days=2, minutes=index * 10),
            )
            for index in range(19)
        )
        for observation in holdout_records:
            _settle(observation)

        # When: the under-sized holdout is accessed.
        result = append_holdout_result(manifest, holdout_records)

        # Then: validation proof remains, but no rejected holdout interval is registered.
        self.assertEqual(result.rejection_code, "holdout_requires_exactly_20")
        self.assertEqual(
            TrendContinuationForwardConsumption.objects.filter(phase="holdout").count(),
            0,
        )


class ConsumptionMetadataTests(TestCase):
    def test_persisted_metadata_exactly_matches_manifest_observation_and_settlement(self):
        identity = _identity("COINUSDT")
        manifest = freeze_selection(_snapshot(identity), frozen_at=BASE)
        observation = _observation(identity, BASE + timedelta(days=1))
        settlement = _settle(observation)

        consumption = register_forward_consumption(
            contract_symbol=identity.contract,
            interval_start=observation.evidence_time,
            interval_end=settlement.label_end_time,
            manifest_digest=manifest.digest,
            profile=identity.profile,
            trend_direction=identity.trend_direction.value,
            rule_name=identity.rule_name,
            candidate_horizon_bars=identity.candidate_horizon_bars,
            strategy_version=identity.strategy_version,
            feature_schema_version=identity.feature_schema_version,
            observation=observation,
            phase="validation",
        )

        self.assertEqual(consumption.manifest_id, manifest.pk)
        self.assertEqual(consumption.observation_id, observation.pk)
        self.assertEqual(consumption.interval_start, observation.evidence_time)
        self.assertEqual(consumption.interval_end, settlement.label_end_time)
        self.assertEqual(consumption.manifest_digest, manifest.digest)
        self.assertEqual(consumption.profile, manifest.profile)
        self.assertEqual(consumption.trend_direction, manifest.trend_direction)
        self.assertEqual(consumption.rule_name, manifest.rule_name)
        self.assertEqual(consumption.candidate_horizon_bars, manifest.candidate_horizon_bars)
        self.assertEqual(consumption.strategy_version, manifest.strategy_version)
        self.assertEqual(consumption.feature_schema_version, manifest.feature_schema_version)

    def test_manifest_identity_mismatch_is_rejected_before_persistence(self):
        # Given: a frozen standard-profile UP manifest and a disjoint interval.
        manifest = freeze_selection(_snapshot(_identity("TSLAUSDT")), frozen_at=BASE)
        start = BASE + timedelta(days=1)

        # When / Then: contradictory non-key audit metadata is rejected and not persisted.
        with self.assertRaises(ValidationError):
            register_forward_consumption(
                contract_symbol="TSLAUSDT",
                interval_start=start,
                interval_end=start + timedelta(minutes=5),
                manifest_digest=manifest.digest,
                profile="triple_leveraged_15m_5m_1m",
                trend_direction="DOWN",
                rule_name="baseline_mtf_v1",
                candidate_horizon_bars=15,
                strategy_version="trend-v2",
                feature_schema_version="other-features",
                phase="validation",
            )
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 0)

    def test_linked_observation_interval_mismatch_is_rejected(self):
        # Given: a resolved forward observation and its exact manifest identity.
        identity = _identity("RGTIUSDT")
        manifest = freeze_selection(_snapshot(identity), frozen_at=BASE)
        observation = _observation(identity, BASE + timedelta(days=1))
        settlement = _settle(observation)

        # When / Then: a consumption interval that does not equal its settlement is rejected.
        with self.assertRaises(ValidationError):
            register_forward_consumption(
                contract_symbol=identity.contract,
                interval_start=observation.evidence_time,
                interval_end=settlement.label_end_time + timedelta(minutes=1),
                manifest_digest=manifest.digest,
                profile=identity.profile,
                trend_direction=identity.trend_direction.value,
                rule_name=identity.rule_name,
                candidate_horizon_bars=identity.candidate_horizon_bars,
                strategy_version=identity.strategy_version,
                feature_schema_version=identity.feature_schema_version,
                observation=observation,
                phase="validation",
            )
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 0)
