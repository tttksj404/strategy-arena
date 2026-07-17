import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_kra_autoresearch import validate_report


class ValidateKraAutoresearchTestCase(unittest.TestCase):
    def test_validator_passes_only_promoted_selected_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps({
                "selected": "recent_form",
                "promotion_pass": True,
                "selected_result": {
                    "promotion_pass": True,
                    "pooled_bootstrap": {"mean_pp": 5.0, "ci95_low_pp": 0.2},
                    "pooled_logloss_delta": -0.01,
                    "fresh_holdout": {
                        "bootstrap": {"mean_pp": 5.0, "ci95_low_pp": 0.2},
                    },
                },
            }), encoding="utf-8")

            result = validate_report(str(report_path))

        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "passed")

    def test_validator_rejects_report_without_significant_lift(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps({
                "selected": "recent_form",
                "promotion_pass": False,
                "selected_result": {
                    "promotion_pass": False,
                    "pooled_bootstrap": {"ci95_low_pp": -0.1},
                    "pooled_logloss_delta": -0.01,
                },
            }), encoding="utf-8")

            result = validate_report(str(report_path))

        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "failed")

    def test_validator_rejects_statistically_significant_lift_below_five_points(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps({
                "selected": "recent_form",
                "promotion_pass": True,
                "selected_result": {
                    "promotion_pass": True,
                    "pooled_bootstrap": {"mean_pp": 4.99, "ci95_low_pp": 0.2},
                    "pooled_logloss_delta": -0.01,
                    "fresh_holdout": {
                        "bootstrap": {"mean_pp": 5.0, "ci95_low_pp": 0.2},
                    },
                },
            }), encoding="utf-8")

            result = validate_report(str(report_path))

        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "failed")

    def test_validator_rejects_candidate_without_fresh_five_point_lift(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps({
                "selected": "recent_form",
                "promotion_pass": True,
                "selected_result": {
                    "promotion_pass": True,
                    "pooled_bootstrap": {"mean_pp": 5.5, "ci95_low_pp": 0.2},
                    "pooled_logloss_delta": -0.01,
                    "fresh_holdout": {
                        "bootstrap": {"mean_pp": 4.99, "ci95_low_pp": 0.2},
                    },
                },
            }), encoding="utf-8")

            result = validate_report(str(report_path))

        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
