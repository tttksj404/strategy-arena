import math
import unittest

import pandas as pd

from kra_history_features import (
    add_horse_history_features,
    apply_horse_history_snapshot,
    build_horse_history_snapshot,
)


class KraHistoryFeaturesTestCase(unittest.TestCase):
    def test_training_features_use_only_prior_dates(self):
        frame = pd.DataFrame(
            [
                {"hrNo": "A", "rcDate": "20250101", "ord": 1, "field_size": 10, "win": 1, "place": 1},
                {"hrNo": "B", "rcDate": "20250101", "ord": 2, "field_size": 10, "win": 0, "place": 1},
                {"hrNo": "A", "rcDate": "20250108", "ord": 5, "field_size": 10, "win": 0, "place": 0},
            ]
        )

        result = add_horse_history_features(frame)

        self.assertAlmostEqual(result.loc[0, "hr_win_prior"], 0.10)
        self.assertAlmostEqual(result.loc[2, "hr_win_prior"], 0.25)
        self.assertAlmostEqual(result.loc[2, "hr_place_prior"], 2.25 / 6.0)
        self.assertAlmostEqual(result.loc[2, "hr_starts_log"], math.log1p(1))
        self.assertEqual(result.loc[2, "hr_days_since"], 7.0)

    def test_snapshot_applies_full_history_to_future_starters(self):
        frame = pd.DataFrame(
            [
                {"hrNo": "A", "rcDate": "20250101", "ord": 1, "field_size": 10, "win": 1, "place": 1},
                {"hrNo": "A", "rcDate": "20250108", "ord": 5, "field_size": 10, "win": 0, "place": 0},
            ]
        )
        snapshot = build_horse_history_snapshot(frame)
        starters = pd.DataFrame([{"hrNo": "A"}, {"hrNo": "NEW"}])

        result = apply_horse_history_snapshot(starters, snapshot, "20250115")

        self.assertAlmostEqual(result.loc[0, "hr_win_prior"], 1.5 / 7.0)
        self.assertAlmostEqual(result.loc[1, "hr_win_prior"], 0.10)
        self.assertEqual(result.loc[0, "hr_days_since"], 7.0)
        self.assertTrue(pd.isna(result.loc[1, "hr_days_since"]))


if __name__ == "__main__":
    unittest.main()
