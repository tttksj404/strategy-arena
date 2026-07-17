import unittest

import numpy as np
import pandas as pd

from kra_pairwise_reranker import RaceScores, RerankPolicy, restricted_rerank


class KraPairwiseRerankerTestCase(unittest.TestCase):
    def test_reranker_switches_only_to_baseline_top_k_entry(self):
        frame = pd.DataFrame({"rk": ["R1", "R1", "R1", "R2", "R2", "R2"]})
        baseline = np.array([0.5, 0.3, 0.2, 0.5, 0.3, 0.2])
        pairwise = np.array([0.2, 0.7, 0.1, 0.2, 0.1, 0.7])

        result = restricted_rerank(
            frame,
            RaceScores(baseline, pairwise),
            RerankPolicy(weight=0.75, top_k=2),
        )

        leaders = [
            race["score"].idxmax()
            for _, race in frame.assign(score=result.scores).groupby("rk", sort=False)
        ]
        self.assertEqual(leaders, [1, 3])
        self.assertEqual(result.switches, 1)

    def test_reranker_preserves_baseline_top_three_membership(self):
        frame = pd.DataFrame({"rk": ["R1", "R1", "R1", "R1"]})
        baseline = np.array([0.4, 0.3, 0.2, 0.1])
        pairwise = np.array([0.1, 0.2, 0.6, 0.1])

        result = restricted_rerank(
            frame,
            RaceScores(baseline, pairwise),
            RerankPolicy(weight=1.0, top_k=3),
        )

        top_three = set(np.argsort(-result.scores)[:3].tolist())
        self.assertEqual(top_three, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
