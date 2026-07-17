from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from kra_market_residual import (
    _challenger_race_frame,
    add_market_features,
    restricted_market_rerank,
    uncertainty_gate,
)
from kra_model_evaluation import market_probability


class MarketResidualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({
            "rk": ["a", "a", "a", "b", "b"],
            "winOdds": [2.0, 3.0, 6.0, 1.5, 4.0],
            "plcOdds": [1.4, 1.7, 2.2, 1.2, 2.0],
            "consensus_strength": [0.8, 0.5, 0.2, 0.9, 0.3],
            "consensus_disagreement": [0.1, 0.2, 0.3, 0.1, 0.4],
        })

    def test_market_features_are_race_relative(self) -> None:
        result, columns = add_market_features(self.frame)
        self.assertEqual(12, len(columns))
        np.testing.assert_allclose(
            result.groupby("rk")["market_win_probability"].sum().to_numpy(),
            [1.0, 1.0],
        )
        self.assertEqual(1, int(result[result["rk"] == "a"]["market_is_favorite"].sum()))

    def test_restricted_rerank_only_changes_market_top_k(self) -> None:
        market = market_probability(self.frame)
        candidate = np.array([0.1, 0.8, 0.9, 0.2, 0.8])
        result = restricted_market_rerank(self.frame, candidate, top_k=2)
        self.assertEqual(1, int(np.argmax(result[:3])))
        self.assertAlmostEqual(market[2], result[2])

    def test_uncertainty_gate_preserves_confident_race(self) -> None:
        market = market_probability(self.frame)
        candidate = np.array([0.1, 0.8, 0.1, 0.1, 0.9])
        result = uncertainty_gate(
            self.frame,
            candidate,
            maximum_favorite_probability=0.55,
            maximum_gap=0.1,
        )
        np.testing.assert_allclose(result[3:], market[3:])

    def test_challenger_frame_target_is_market_rank_or_other(self) -> None:
        frame = self.frame.assign(
            win=[0, 1, 0, 0, 1],
            rating=[10.0, 12.0, 8.0, 20.0, 15.0],
        )
        result, keys = _challenger_race_frame(frame, ["rating"], 2, include_target=True)
        self.assertEqual(["a", "b"], keys)
        self.assertEqual([1, 1], result["target"].tolist())
        self.assertIn("d1_rating", result)


if __name__ == "__main__":
    unittest.main()
