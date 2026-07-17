import unittest

import pandas as pd

from kra_candidate_features import add_candidate_features
from tools import kra_autoresearch_v5


class KraCandidateFeaturesTestCase(unittest.TestCase):
    def test_candidate_features_use_only_prior_dates(self):
        frame = pd.DataFrame([
            {"hrNo": "A", "jkNo": "J1", "trNo": "T1", "meet": "서울", "rcDate": "20250101", "rcDist": 1200, "ord": 1, "field_size": 5, "win": 1, "place": 1, "rating": 30, "wgBudam": 52},
            {"hrNo": "A", "jkNo": "J1", "trNo": "T1", "meet": "서울", "rcDate": "20250108", "rcDist": 1200, "ord": 5, "field_size": 5, "win": 0, "place": 0, "rating": 32, "wgBudam": 53},
            {"hrNo": "A", "jkNo": "J2", "trNo": "T1", "meet": "부경", "rcDate": "20250115", "rcDist": 1600, "ord": 3, "field_size": 5, "win": 0, "place": 1, "rating": 31, "wgBudam": 51},
            {"hrNo": "NEW", "jkNo": "J3", "trNo": "T2", "meet": "서울", "rcDate": "20250115", "rcDist": 1200, "ord": 2, "field_size": 5, "win": 0, "place": 1, "rating": 20, "wgBudam": 50},
        ])

        result = add_candidate_features(frame)

        self.assertAlmostEqual(result.loc[0, "hr_recent_finish"], 0.5)
        self.assertAlmostEqual(result.loc[1, "hr_recent_finish"], 0.0)
        self.assertAlmostEqual(result.loc[2, "hr_recent_finish"], 0.5)
        self.assertAlmostEqual(result.loc[1, "hr_distance_finish"], 0.0)
        self.assertAlmostEqual(result.loc[2, "hr_distance_finish"], 0.5)
        self.assertAlmostEqual(result.loc[2, "hr_meet_finish"], 0.5)
        self.assertAlmostEqual(result.loc[2, "hr_rating_change"], -1.0)
        self.assertAlmostEqual(result.loc[3, "hr_recent_finish"], 0.5)

    def test_same_day_rows_do_not_leak_into_each_other(self):
        frame = pd.DataFrame([
            {"hrNo": "A", "jkNo": "J1", "trNo": "T1", "meet": "서울", "rcDate": "20250101", "rcDist": 1200, "ord": 1, "field_size": 5, "win": 1, "place": 1, "rating": 30, "wgBudam": 52},
            {"hrNo": "A", "jkNo": "J1", "trNo": "T1", "meet": "서울", "rcDate": "20250101", "rcDist": 1200, "ord": 5, "field_size": 5, "win": 0, "place": 0, "rating": 30, "wgBudam": 52},
        ])

        result = add_candidate_features(frame)

        self.assertEqual(result["hr_recent_starts"].tolist(), [0.0, 0.0])
        self.assertEqual(result["hr_recent_finish"].tolist(), [0.5, 0.5])

    def test_research_frame_contains_relative_candidate_columns_before_fold_copy(self):
        frame = pd.DataFrame({
            "rk": ["A", "A"],
            "hr_recent_win": [0.1, 0.3],
        })

        prepared = kra_autoresearch_v5._prepare_candidate_frame(frame)

        self.assertIn("hr_recent_win_rel", prepared.columns)
        self.assertAlmostEqual(prepared.loc[0, "hr_recent_win_rel"], -0.1)


if __name__ == "__main__":
    unittest.main()
