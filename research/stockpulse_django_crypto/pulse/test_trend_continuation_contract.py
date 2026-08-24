from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from django.test import SimpleTestCase

from pulse.services.trend_continuation_contract import (
    FORBIDDEN_PAYLOAD_KEYS,
    FORBIDDEN_PAYLOAD_VALUES,
    SCHEMA_VERSION,
    CandidateIdentity,
    Direction,
    HorizonStatus,
    PublicMarketRequest,
    ResearchDecision,
    TrendContinuationDiagnostic,
    default_no_trade_diagnostic,
    stable_manifest_digest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrendContinuationContractTests(SimpleTestCase):
    def test_default_payload_is_research_only_no_trade_with_complete_time_contract(self):
        # Given: a complete trigger candle with no validation evidence yet.
        evidence_time = datetime(2026, 8, 18, 3, 5, tzinfo=UTC)
        identity = CandidateIdentity(
            contract="TSLAUSDT",
            profile="standard_15m_5m",
            trend_direction=Direction.UP,
            rule_name="baseline_mtf_v1",
            candidate_horizon_bars=12,
            strategy_version="trend-continuation-strategy.v1",
            feature_schema_version="trend-continuation-features.v1",
        )

        # When: the contract serializes a default observation candidate.
        diagnostic = default_no_trade_diagnostic(
            identity=identity,
            as_of=evidence_time,
            evidence_time=evidence_time,
            trigger_interval_seconds=300,
            reference_price=23.0,
            invalidation_price=21.9,
            rejection_code="pending_forward_validation",
        )
        payload = diagnostic.to_payload()

        # Then: it is explicit research-only output, never a trading instruction.
        self.assertEqual(payload["schema"], SCHEMA_VERSION)
        self.assertEqual(payload["mode"], "research_only")
        self.assertEqual(payload["decision"], ResearchDecision.NO_TRADE.value)
        self.assertEqual(payload["trend_direction"], Direction.UP.value)
        self.assertEqual(payload["horizon_status"], HorizonStatus.PENDING_VALIDATION.value)
        self.assertEqual(payload["candidate_horizon_bars"], 12)
        self.assertEqual(payload["evidence_time"], evidence_time.isoformat())
        self.assertEqual(payload["label_end_time"], (evidence_time + timedelta(minutes=60)).isoformat())
        self.assertEqual(payload["execution"], {"enabled": False})
        self.assertTrue(payload["data_policy"]["closed_candles_only"])
        self.assertTrue(payload["research_safety"]["public_market_data_only"])
        self.assertEqual(payload["provenance"]["source"], "public_bitget_market")
        self.assertNotIn("holdout_empirical_continuation_rate_pct", payload)
        self.assertFalse(FORBIDDEN_PAYLOAD_KEYS.intersection(payload))
        self.assertFalse(FORBIDDEN_PAYLOAD_VALUES.intersection(json.dumps(payload).split()))

    def test_observe_decision_requires_every_gate_and_rejects_zero_horizon(self):
        # Given: an otherwise complete diagnostic with a missing forward gate.
        evidence_time = datetime(2026, 8, 18, 3, 5, tzinfo=UTC)
        identity = CandidateIdentity(
            contract="TSLAUSDT",
            profile="standard_15m_5m",
            trend_direction=Direction.UP,
            rule_name="baseline_mtf_v1",
            candidate_horizon_bars=12,
            strategy_version="trend-continuation-strategy.v1",
            feature_schema_version="trend-continuation-features.v1",
        )

        # When / Then: construction fails closed unless every observe gate is true.
        with self.assertRaisesMessage(ValueError, "CONTINUATION_OBSERVE requires every gate"):
            TrendContinuationDiagnostic(
                identity=identity,
                as_of=evidence_time,
                evidence_time=evidence_time,
                trigger_interval_seconds=300,
                reference_price=23.0,
                invalidation_price=21.9,
                decision=ResearchDecision.CONTINUATION_OBSERVE,
                horizon_status=HorizonStatus.PENDING_VALIDATION,
                rejection_code=None,
                all_observe_gates_passed=False,
            )

        with self.assertRaisesMessage(ValueError, "candidate_horizon_bars must be positive"):
            CandidateIdentity(
                contract="TSLAUSDT",
                profile="standard_15m_5m",
                trend_direction=Direction.UP,
                rule_name="baseline_mtf_v1",
                candidate_horizon_bars=0,
                strategy_version="trend-continuation-strategy.v1",
                feature_schema_version="trend-continuation-features.v1",
            )

    def test_public_request_allowlist_rejects_private_order_and_legacy_action_vocabulary(self):
        # Given / When / Then: only explicit public market GET paths can be represented.
        allowed_paths = (
            "/api/v2/mix/market/contracts",
            "/api/v2/mix/market/candles",
            "/api/v2/mix/market/history-candles",
            "/api/v2/mix/market/ticker",
            "/api/v2/mix/market/current-fund-rate",
            "/api/v2/mix/market/history-fund-rate",
        )
        for path in allowed_paths:
            with self.subTest(allowed_path=path):
                allowed = PublicMarketRequest(method="GET", path=path)
                self.assertEqual(allowed.path, path)

        blocked_inputs = [
            ("POST", "/api/v2/mix/market/candles"),
            ("GET", "/api/v2/mix/order/place-order"),
            ("GET", "/api/v2/mix/account/accounts"),
            ("GET", "/api/v2/mix/position/single-position"),
            ("GET", "/api/v2/mix/market/leverage"),
            ("GET", "/api/v3/market/instruments"),
            ("GET", "/api/v3/market/tickers"),
            ("GET", "/api/v3/market/candles"),
            ("GET", "/api/v3/market/history-candles"),
            ("GET", "/api/v3/market/history-fund-rate"),
        ]
        for method, path in blocked_inputs:
            with self.subTest(method=method, path=path):
                with self.assertRaises(ValueError):
                    PublicMarketRequest(method=method, path=path)

        self.assertIn("ENTER_LONG", FORBIDDEN_PAYLOAD_VALUES)
        self.assertIn("WATCH_LONG_SETUP", FORBIDDEN_PAYLOAD_VALUES)
        self.assertIn("estimated_20x", FORBIDDEN_PAYLOAD_KEYS)
        self.assertIn("estimated_5x", FORBIDDEN_PAYLOAD_KEYS)
        self.assertIn("target", FORBIDDEN_PAYLOAD_KEYS)

    def test_payload_guard_rejects_nested_trade_keys_and_allows_validated_empirical_rate(self):
        # Given: a validated research observation with empirical holdout fields.
        evidence_time = datetime(2026, 8, 18, 3, 5, tzinfo=UTC)
        identity = CandidateIdentity(
            contract="TSLAUSDT",
            profile="standard_15m_5m",
            trend_direction=Direction.UP,
            rule_name="baseline_mtf_v1",
            candidate_horizon_bars=12,
            strategy_version="trend-continuation-strategy.v1",
            feature_schema_version="trend-continuation-features.v1",
        )

        diagnostic = TrendContinuationDiagnostic(
            identity=identity,
            as_of=evidence_time,
            evidence_time=evidence_time,
            trigger_interval_seconds=300,
            reference_price=23.0,
            invalidation_price=21.9,
            decision=ResearchDecision.CONTINUATION_OBSERVE,
            horizon_status=HorizonStatus.VALIDATED,
            rejection_code=None,
            all_observe_gates_passed=True,
            holdout_empirical_continuation_rate_pct=55.0,
            wilson_lower_95=50.1,
            mean_return_lower_95=0.2,
            holdout_sample_count=20,
        )

        # When / Then: validated empirical rate is allowed, but trade vocabulary is not.
        payload = diagnostic.to_payload()
        self.assertEqual(payload["holdout_empirical_continuation_rate_pct"], 55.0)

        for forbidden_key in ("estimated_20x", "estimated_5x", "target"):
            with self.subTest(forbidden_key=forbidden_key):
                blocked = TrendContinuationDiagnostic(
                    identity=identity,
                    as_of=evidence_time,
                    evidence_time=evidence_time,
                    trigger_interval_seconds=300,
                    reference_price=23.0,
                    invalidation_price=21.9,
                    decision=ResearchDecision.NO_TRADE,
                    horizon_status=HorizonStatus.PENDING_VALIDATION,
                    rejection_code="malformed_input",
                    forward_provenance={forbidden_key: 1},
                )
                with self.assertRaisesMessage(ValueError, f"forbidden payload key: {forbidden_key}"):
                    blocked.to_payload()

        blocked_value = TrendContinuationDiagnostic(
            identity=identity,
            as_of=evidence_time,
            evidence_time=evidence_time,
            trigger_interval_seconds=300,
            reference_price=23.0,
            invalidation_price=21.9,
            decision=ResearchDecision.NO_TRADE,
            horizon_status=HorizonStatus.PENDING_VALIDATION,
            rejection_code="malformed_input",
            forward_provenance={"state": "ENTER_LONG"},
        )
        with self.assertRaisesMessage(ValueError, "forbidden payload value: ENTER_LONG"):
            blocked_value.to_payload()

    def test_manifest_digest_is_stable_and_changes_on_payload_mutation(self):
        # Given: the same manifest data with different insertion order.
        left = {"schema": SCHEMA_VERSION, "candidate": {"contract": "TSLAUSDT", "horizon": 12}}
        right = {"candidate": {"horizon": 12, "contract": "TSLAUSDT"}, "schema": SCHEMA_VERSION}

        # When / Then: canonical JSON hashing is deterministic and mutation-sensitive.
        self.assertEqual(stable_manifest_digest(left), stable_manifest_digest(right))
        self.assertNotEqual(
            stable_manifest_digest(left),
            stable_manifest_digest({"schema": SCHEMA_VERSION, "candidate": {"contract": "TSLAUSDT", "horizon": 18}}),
        )


class OwnedDiffVerifierTests(SimpleTestCase):
    def test_owned_diff_verifier_accepts_exact_reconstruction_and_rejects_tampering(self):
        # Given: a tiny owned path with an empty baseline and a patch that recreates final bytes.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_path = root / "owned.txt"
            baseline_path = root / "baseline.empty"
            patch_path = root / "owned.patch"
            manifest_path = root / "manifest.json"
            baseline_path.write_text("", encoding="utf-8")
            final_path.write_text("alpha\n", encoding="utf-8")
            patch_path.write_text(
                "diff --git a/owned.txt b/owned.txt\n"
                "new file mode 100644\n"
                "index 0000000..4a58007\n"
                "--- /dev/null\n"
                "+++ b/owned.txt\n"
                "@@ -0,0 +1 @@\n"
                "+alpha\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "paths": [
                            {
                                "path": "owned.txt",
                                "baseline": "baseline.empty",
                                "patch": "owned.patch",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            # When: the verifier checks the final file against baseline plus patch.
            ok = subprocess.run(
                ["node", str(REPO_ROOT / "scripts/verify_trend_continuation_owned_diff.mjs"), "--manifest", str(manifest_path), "--repo-root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            # Then: exact reconstruction succeeds with machine-readable output.
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn('"status":"PASS"', ok.stdout)

            final_path.write_text("tampered\n", encoding="utf-8")
            bad = subprocess.run(
                ["node", str(REPO_ROOT / "scripts/verify_trend_continuation_owned_diff.mjs"), "--manifest", str(manifest_path), "--repo-root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("reconstructed bytes differ", bad.stderr)
