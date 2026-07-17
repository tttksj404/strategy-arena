from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from kra_diversified_rankers import (
    _race_favorite_target,
    add_pace_interactions,
    fit_market_distillation,
)


class DiversifiedRankerTests(unittest.TestCase):
    def test_pace_features_are_race_relative(self) -> None:
        frame = pd.DataFrame({
            "rk": ["a", "a", "b", "b"],
            "rating": [10, 20, 30, 25],
            "hr_elo": [1400, 1500, 1600, 1550],
            "hr_speed_mean_3": [1.0, 2.0, 4.0, 3.0],
            "hr_recent_finish_mean_3": [0.8, 0.2, 0.1, 0.6],
            "hr_early_position_mean_3": [0.2, 0.8, 0.3, 0.7],
            "hr_finish_gain_mean_3": [0.1, 0.2, 0.3, 0.4],
        })
        result, columns = add_pace_interactions(frame)
        self.assertEqual(10, len(columns))
        self.assertEqual([1.0, 1.0, 1.0, 1.0], result["pace_pressure"].tolist())
        self.assertTrue((result.groupby("rk")["rating_percentile"].max() == 1.0).all())

    def test_market_teacher_does_not_read_test_odds(self) -> None:
        train = pd.DataFrame({
            "rk": ["a", "a", "b", "b", "c", "c"],
            "field_size": [2] * 6,
            "winOdds": [2.0, 4.0, 5.0, 1.5, 2.5, 3.5],
            "x": [1.0, 0.0, 0.2, 0.9, 0.8, 0.1],
        })
        model = fit_market_distillation(train, ["x"])
        test = pd.DataFrame({
            "rk": ["z", "z"],
            "field_size": [2, 2],
            "winOdds": [1.1, 100.0],
            "x": [0.7, 0.3],
        })
        changed = test.assign(winOdds=[100.0, 1.1])
        np.testing.assert_allclose(model.predict(test), model.predict(changed))

    def test_market_favorite_target_has_one_label_per_race(self) -> None:
        frame = pd.DataFrame({
            "rk": ["a", "a", "a", "b", "b"],
            "winOdds": [3.0, 1.5, 2.0, 4.0, 2.0],
        })
        target = _race_favorite_target(frame, "winOdds")
        self.assertEqual([0, 1, 0, 0, 1], target.tolist())


if __name__ == "__main__":
    unittest.main()
