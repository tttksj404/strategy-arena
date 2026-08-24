from __future__ import annotations

from datetime import datetime, timedelta

from django.test import TestCase

from .services.trend_continuation_contract import CandidateIdentity, ResearchDecision
from .services.trend_continuation_evaluation import EvaluationResult, choose_diagnostic
from .services.trend_continuation_manifest import (
    ManifestLifecycleError,
    append_holdout_result,
    append_validation_result,
    freeze_selection,
)
from .test_trend_continuation_manifest import BASE, _identity, _observation, _settle, _snapshot
from .trend_continuation_models import (
    TrendContinuationForwardConsumption,
    TrendContinuationManifestRejection,
    TrendContinuationObservation,
)


class ForwardLifecycleTests(TestCase):
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

    def test_exact_80_then_later_20_is_the_only_promotable_lifecycle(self):
        # Given: a digest-frozen historical candidate and 100 later resolved forward records.
        identity = _identity()
        manifest = freeze_selection(_snapshot(identity), frozen_at=BASE)
        observations = []
        for index in range(100):
            observation = _observation(identity, BASE + timedelta(days=1, minutes=index * 10))
            _settle(observation)
            observations.append(observation)

        # When: exactly 80 validation records and a strictly later 20-record holdout are appended once.
        validation = append_validation_result(manifest, tuple(observations[:80]))
        holdout = append_holdout_result(manifest, tuple(observations[80:]))
        diagnostic = choose_diagnostic(
            identity=identity,
            as_of=BASE + timedelta(days=3),
            evidence_time=BASE + timedelta(days=3),
            trigger_interval_seconds=300,
            reference_price=100,
            invalidation_price=99,
            validation_result=EvaluationResult(80, 80, 95, 1, 1, True),
            manifest_digest=manifest.digest,
        )

        # Then: both immutable gates pass, all 100 intervals are consumed, and only DB evidence promotes.
        self.assertTrue(validation.passed)
        self.assertTrue(holdout.passed)
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 100)
        self.assertEqual(diagnostic.decision, ResearchDecision.CONTINUATION_OBSERVE)
        self.assertEqual(diagnostic.holdout_sample_count, 20)
        with self.assertRaises(ManifestLifecycleError):
            append_holdout_result(manifest, tuple(observations[80:]))
        self.assertTrue(TrendContinuationManifestRejection.objects.filter(manifest=manifest).exists())
        rejected_diagnostic = choose_diagnostic(
            identity=identity,
            as_of=BASE + timedelta(days=3),
            evidence_time=BASE + timedelta(days=3),
            trigger_interval_seconds=300,
            reference_price=100,
            invalidation_price=99,
            manifest_digest=manifest.digest,
        )
        self.assertEqual(rejected_diagnostic.decision, ResearchDecision.NO_TRADE)

    def test_wrong_validation_count_is_terminal_and_never_promotes(self):
        # Given: a fresh manifest with only 79 resolved forward observations.
        identity = _identity("RGTIUSDT")
        manifest = freeze_selection(_snapshot(identity), frozen_at=BASE)
        observations = []
        for index in range(79):
            observation = _observation(identity, BASE + timedelta(days=1, minutes=index * 10))
            _settle(observation)
            observations.append(observation)

        # When: validation is accessed with the wrong exact sample count.
        result = append_validation_result(manifest, tuple(observations))

        # Then: one terminal failed result is preserved and cannot be retried or open holdout.
        self.assertFalse(result.passed)
        self.assertEqual(result.rejection_code, "validation_requires_exactly_80")
        with self.assertRaises(ManifestLifecycleError):
            append_validation_result(manifest, tuple(observations))
        with self.assertRaises(ManifestLifecycleError):
            append_holdout_result(manifest, ())

    def test_wrong_holdout_count_is_terminal_after_a_passing_validation(self):
        # Given: a passed exact-80 validation and only 19 strictly later holdout rows.
        identity = _identity("PLTRUSDT")
        manifest = freeze_selection(_snapshot(identity), frozen_at=BASE)
        validation_records = self._resolved_records(
            identity,
            80,
            starts_at=BASE + timedelta(days=1),
        )
        validation = append_validation_result(manifest, validation_records)
        holdout_records = self._resolved_records(
            identity,
            19,
            starts_at=BASE + timedelta(days=2),
        )

        # When: the one-time holdout is accessed at the wrong exact count.
        holdout = append_holdout_result(manifest, holdout_records)

        # Then: the failed result is terminal and cannot produce or retry validated output.
        self.assertTrue(validation.passed)
        self.assertFalse(holdout.passed)
        self.assertEqual(holdout.rejection_code, "holdout_requires_exactly_20")
        with self.assertRaises(ManifestLifecycleError):
            append_holdout_result(manifest, holdout_records)

    def test_81_validation_and_21_holdout_are_terminal(self):
        # Given: one manifest receives 81 rows, while another reaches validation then receives 21 holdout rows.
        validation_identity = _identity("IBMUSDT")
        validation_manifest = freeze_selection(_snapshot(validation_identity), frozen_at=BASE)
        validation_records = self._resolved_records(
            validation_identity,
            81,
            starts_at=BASE + timedelta(days=1),
        )
        holdout_identity = _identity("QQQUSDT")
        holdout_manifest = freeze_selection(_snapshot(holdout_identity), frozen_at=BASE)
        passing_validation = self._resolved_records(
            holdout_identity,
            80,
            starts_at=BASE + timedelta(days=1),
        )
        append_validation_result(holdout_manifest, passing_validation)
        holdout_records = self._resolved_records(
            holdout_identity,
            21,
            starts_at=BASE + timedelta(days=2),
        )

        # When: each oversized one-time gate is accessed.
        validation = append_validation_result(validation_manifest, validation_records)
        holdout = append_holdout_result(holdout_manifest, holdout_records)

        # Then: both exact-count violations are immutable terminal failures.
        self.assertFalse(validation.passed)
        self.assertEqual(validation.rejection_code, "validation_requires_exactly_80")
        self.assertFalse(holdout.passed)
        self.assertEqual(holdout.rejection_code, "holdout_requires_exactly_20")

    def test_exact_20_holdout_metric_failure_is_terminal(self):
        # Given: a passing validation followed by exactly 20 losing forward rows.
        identity = _identity("MSTRUSDT")
        manifest = freeze_selection(_snapshot(identity), frozen_at=BASE)
        validation_records = self._resolved_records(identity, 80, starts_at=BASE + timedelta(days=1))
        append_validation_result(manifest, validation_records)
        holdout_records = self._resolved_records(
            identity,
            20,
            starts_at=BASE + timedelta(days=2),
            success=False,
        )

        # When: the fixed holdout statistics are evaluated.
        holdout = append_holdout_result(manifest, holdout_records)

        # Then: the metric failure is terminal despite the exact count.
        self.assertFalse(holdout.passed)
        self.assertEqual(holdout.rejection_code, "holdout_fixed_metrics_failed")
        with self.assertRaises(ManifestLifecycleError):
            append_holdout_result(manifest, holdout_records)

    def test_exact_80_metric_failure_is_terminal(self):
        # Given: exactly 80 resolved forward rows that all lose after cost.
        identity = _identity("NVDAUSDT")
        manifest = freeze_selection(_snapshot(identity), frozen_at=BASE)
        records = self._resolved_records(
            identity,
            80,
            starts_at=BASE + timedelta(days=1),
            success=False,
        )

        # When: fixed validation metrics are appended.
        result = append_validation_result(manifest, records)

        # Then: exact count alone cannot pass and the failed metrics cannot be retried.
        self.assertFalse(result.passed)
        self.assertEqual(result.rejection_code, "validation_fixed_metrics_failed")
        with self.assertRaises(ManifestLifecycleError):
            append_validation_result(manifest, records)
