from __future__ import annotations

from datetime import timedelta

from django.test import TestCase

from .services.trend_continuation_contract import ResearchDecision
from .services.trend_continuation_evaluation import EvaluationResult, choose_diagnostic
from .services.trend_continuation_manifest import (
    ForwardIntervalOverlapError,
    freeze_selection,
    register_forward_consumption,
)
from .test_trend_continuation_manifest import BASE, _identity, _snapshot
from .trend_continuation_models import (
    TrendContinuationForwardConsumption,
    TrendContinuationHoldoutResult,
    TrendContinuationValidationResult,
)


class ForwardProofTests(TestCase):
    def test_global_closed_interval_conflicts_ignore_clone_version_and_identity_metadata(self):
        # Given: one consumed TSLA interval and cloned/new-version manifests.
        original = freeze_selection(_snapshot(_identity(), nonce="original"), frozen_at=BASE)
        clone = freeze_selection(_snapshot(_identity(version="trend-v2"), nonce="clone"), frozen_at=BASE)
        start = BASE + timedelta(days=1)
        end = start + timedelta(minutes=5)
        register_forward_consumption(
            contract_symbol="TSLAUSDT",
            interval_start=start,
            interval_end=end,
            manifest_digest=original.digest,
            profile="standard_15m_5m",
            trend_direction="UP",
            rule_name="technical_score_v1",
            candidate_horizon_bars=1,
            phase="validation",
        )

        # When / Then: equal endpoint, partial, full, and new-version reuse all reject globally.
        for overlap_start, overlap_end in (
            (end, end + timedelta(minutes=5)),
            (start + timedelta(minutes=1), end + timedelta(minutes=1)),
            (start - timedelta(minutes=1), end + timedelta(minutes=1)),
        ):
            with self.assertRaises(ForwardIntervalOverlapError):
                register_forward_consumption(
                    contract_symbol="TSLAUSDT",
                    interval_start=overlap_start,
                    interval_end=overlap_end,
                    manifest_digest=clone.digest,
                    profile="triple_leveraged_15m_5m_1m",
                    trend_direction="DOWN",
                    rule_name="baseline_mtf_v1",
                    candidate_horizon_bars=15,
                    phase="holdout",
                )
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 1)

    def test_historical_metrics_cannot_promote_without_immutable_forward_results(self):
        # Given: excellent caller-supplied metrics but no frozen manifest results.
        identity = _identity()

        # When: the diagnostic boundary receives those historical metrics.
        diagnostic = choose_diagnostic(
            identity=identity,
            as_of=BASE,
            evidence_time=BASE,
            trigger_interval_seconds=300,
            reference_price=100,
            invalidation_price=99,
            validation_result=EvaluationResult(80, 80, 95, 1, 1, True),
        )

        # Then: historical-only performance remains NO_TRADE.
        self.assertEqual(diagnostic.decision, ResearchDecision.NO_TRADE)

    def test_fabricated_pass_rows_without_100_consumptions_cannot_promote(self):
        # Given: directly inserted pass-shaped result rows but no linked forward interval consumptions.
        identity = _identity("AMDUSDT")
        manifest = freeze_selection(_snapshot(identity), frozen_at=BASE)
        validation_end = BASE + timedelta(days=2)
        TrendContinuationValidationResult.objects.create(
            manifest=manifest,
            resolved_count=80,
            success_count=80,
            wilson_lower_95=95,
            mean_return_lower_95=1,
            mean_return_pct=1,
            passed=True,
            interval_start=BASE + timedelta(days=1),
            interval_end=validation_end,
            sample_digest="e" * 64,
            provenance={"complete": True},
        )
        TrendContinuationHoldoutResult.objects.create(
            manifest=manifest,
            resolved_count=20,
            success_count=20,
            wilson_lower_95=83,
            mean_return_lower_95=1,
            mean_return_pct=1,
            passed=True,
            interval_start=validation_end + timedelta(minutes=1),
            interval_end=validation_end + timedelta(hours=4),
            sample_digest="f" * 64,
            provenance={"complete": True},
        )

        # When: the public diagnostic boundary checks the digest.
        diagnostic = choose_diagnostic(
            identity=identity,
            as_of=BASE + timedelta(days=3),
            evidence_time=BASE + timedelta(days=3),
            trigger_interval_seconds=300,
            reference_price=100,
            invalidation_price=99,
            manifest_digest=manifest.digest,
        )

        # Then: result-shaped rows cannot replace exact linked 80+20 forward consumption proof.
        self.assertEqual(diagnostic.decision, ResearchDecision.NO_TRADE)
