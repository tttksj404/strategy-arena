from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.prediction_candidate_manifest import build_manifest


class PredictionCandidateManifestTestCase(unittest.TestCase):
    def test_manifest_classifies_known_candidates_without_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)

            manifest = build_manifest(
                kra_meta_gate_path=base_path / "missing_meta_gate.json",
                kra_context_rerank_path=base_path / "missing_context_rerank.json",
            )

        by_id = {candidate["candidate_id"]: candidate for candidate in manifest["candidates"]}
        self.assertEqual(
            by_id["kcycle_trifecta_ensemble_v1"]["decision"],
            "validation_only_baseline",
        )
        self.assertEqual(by_id["kcycle_trifecta_ensemble_v1"]["reason"], "baseline_retained")
        self.assertEqual(by_id["gen2_mut_436"]["decision"], "rejected")
        self.assertEqual(by_id["gen2_mut_436"]["reason"], "selection_scope_must_be_train_val")
        self.assertEqual(
            by_id["kcycle_trifecta_ensemble_v2_candidate"]["decision"],
            "data_blocked",
        )
        self.assertEqual(
            by_id["kcycle_trifecta_ensemble_v2_candidate"]["reason"],
            "audit_pending",
        )
        self.assertEqual(by_id["kcycle_timed_market_lane"]["decision"], "data_blocked")
        self.assertEqual(
            by_id["kcycle_timed_market_lane"]["reason"],
            "outcome_linked_timed_snapshot_unavailable",
        )
        self.assertEqual(by_id["kra_nested_meta_gate"]["decision"], "data_blocked")
        self.assertEqual(by_id["kra_nested_meta_gate"]["reason"], "evidence_unavailable")
        self.assertEqual(by_id["kra_context_rerank"]["decision"], "data_blocked")
        self.assertEqual(by_id["kra_context_rerank"]["reason"], "evidence_unavailable")

        for candidate in by_id.values():
            self.assertEqual(
                set(candidate),
                {"candidate_id", "lane", "provenance", "decision", "reason", "evidence_path"},
            )

    def test_manifest_rejects_kra_reports_when_promotion_pass_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            meta_gate_path = base_path / "meta_gate.json"
            context_rerank_path = base_path / "context_rerank.json"
            meta_gate_path.write_text(json.dumps({"promotion_pass": False}), encoding="utf-8")
            context_rerank_path.write_text(
                json.dumps({"promotion_pass": False}),
                encoding="utf-8",
            )

            manifest = build_manifest(
                kra_meta_gate_path=meta_gate_path,
                kra_context_rerank_path=context_rerank_path,
            )

        by_id = {candidate["candidate_id"]: candidate for candidate in manifest["candidates"]}
        self.assertEqual(by_id["kra_nested_meta_gate"]["decision"], "rejected")
        self.assertEqual(by_id["kra_nested_meta_gate"]["reason"], "promotion_pass_false")
        self.assertEqual(by_id["kra_nested_meta_gate"]["evidence_path"], str(meta_gate_path))
        self.assertEqual(by_id["kra_context_rerank"]["decision"], "rejected")
        self.assertEqual(by_id["kra_context_rerank"]["reason"], "promotion_pass_false")
        self.assertEqual(by_id["kra_context_rerank"]["evidence_path"], str(context_rerank_path))

    def test_manifest_reads_default_context_rerank_report_from_repo_root(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                os.chdir(tmp_dir)

                manifest = build_manifest()
            finally:
                os.chdir(original_cwd)

        by_id = {candidate["candidate_id"]: candidate for candidate in manifest["candidates"]}
        self.assertEqual(by_id["kra_context_rerank"]["decision"], "rejected")
        self.assertEqual(by_id["kra_context_rerank"]["reason"], "promotion_pass_false")
        self.assertEqual(
            by_id["kra_context_rerank"]["evidence_path"],
            "runs/kra_context_rerank_v6_results.json",
        )

    def test_manifest_cli_writes_only_explicit_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "manifest.json"

            without_report = subprocess.run(
                [sys.executable, "tools/prediction_candidate_manifest.py"],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            with_report = subprocess.run(
                [
                    sys.executable,
                    "tools/prediction_candidate_manifest.py",
                    "--report",
                    str(report_path),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

            self.assertFalse(
                (
                    Path(__file__).resolve().parents[1]
                    / "prediction_candidate_manifest.json"
                ).exists()
            )
            self.assertEqual(
                json.loads(without_report.stdout)["schema"],
                "prediction_candidate_manifest_v1",
            )
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8")),
                json.loads(with_report.stdout),
            )
