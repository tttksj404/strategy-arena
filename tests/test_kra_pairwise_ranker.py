import unittest

import numpy as np
import pandas as pd

from kra_pairwise_ranker import PairwiseModel, build_winner_pairs, pairwise_scores


class FirstFeatureEstimator:
    def predict_proba(self, values):
        positive = (values.iloc[:, 0].to_numpy() > 0).astype(float)
        return np.column_stack((1.0 - positive, positive))


class KraPairwiseRankerTestCase(unittest.TestCase):
    def test_winner_pairs_are_symmetric_and_balanced(self):
        frame = pd.DataFrame({
            "rk": ["R1", "R1", "R1"],
            "win": [1, 0, 0],
            "strength": [3.0, 2.0, 1.0],
        })

        pairs = build_winner_pairs(frame, ["strength"])

        self.assertEqual(pairs.values["strength"].tolist(), [1.0, -1.0, 2.0, -2.0])
        self.assertEqual(pairs.targets.tolist(), [1, 0, 1, 0])

    def test_pairwise_scores_rank_stronger_entry_first(self):
        frame = pd.DataFrame({
            "rk": ["R1", "R1", "R1"],
            "strength": [3.0, 2.0, 1.0],
        })
        model = PairwiseModel(FirstFeatureEstimator(), pd.Series({"strength": 0.0}))

        scores = pairwise_scores(model, frame, ["strength"])

        self.assertEqual(scores.tolist(), [2.0, 1.0, 0.0])

    def test_boolean_dummy_features_are_converted_before_subtraction(self):
        frame = pd.DataFrame({
            "rk": ["R1", "R1"],
            "win": [1, 0],
            "sex_male": [True, False],
        })

        pairs = build_winner_pairs(frame, ["sex_male"])

        self.assertEqual(pairs.values["sex_male"].tolist(), [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
