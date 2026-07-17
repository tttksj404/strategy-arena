import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_kcycle_grade_routing as experiment


class GradeRoutingExperimentTest(unittest.TestCase):
    def test_grade_proxy_maps_race_bands(self):
        self.assertEqual(experiment.grade_proxy_from_race_no("01"), "선발")
        self.assertEqual(experiment.grade_proxy_from_race_no("6"), "우수")
        self.assertEqual(experiment.grade_proxy_from_race_no("11"), "특선")

    def test_routed_metric_uses_grade_specific_method(self):
        df = pd.DataFrame([
            {
                "year": "2026",
                "race_no": "1",
                "grade_proxy": "선발",
                "actual_order": "1-2-3",
                "pred_board_min": "2-1-3",
                "pred_pair_mass": "1-2-3",
            },
            {
                "year": "2026",
                "race_no": "11",
                "grade_proxy": "특선",
                "actual_order": "4-5-6",
                "pred_board_min": "4-5-6",
                "pred_pair_mass": "6-5-4",
            },
        ])

        metric = experiment.routed_metric(
            df,
            "route",
            {"선발": "pair_mass", "우수": "board_min", "특선": "board_min"},
            "test",
        )

        self.assertEqual(metric.races, 2)
        self.assertEqual(metric.exact, 1.0)
        self.assertEqual(metric.board_exact, 0.5)
        self.assertEqual(metric.exact_lift_pp, 50.0)


if __name__ == "__main__":
    unittest.main()
