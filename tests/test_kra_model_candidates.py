import unittest

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — compact estimator fixtures match production contract

from kra_model_candidates import ModelSpec, fit_candidate
from tools import kra_dual_phase_experiment


class KraModelCandidatesTestCase(unittest.TestCase):
    def test_pairwise_artifact_locks_validated_rerank_policy(self):
        frame = pd.DataFrame({
            "rk": ["R1", "R1", "R2", "R2"],
            "win": [1, 0, 0, 1],
            "strength": [2.0, 1.0, 1.0, 2.0],
        })

        artifact = kra_dual_phase_experiment.build_pairwise_artifact(frame, ["strength"])

        self.assertEqual(artifact["weight"], 0.5)
        self.assertEqual(artifact["top_k"], 3)
        self.assertFalse(artifact["enabled"])
        self.assertIn("estimator", artifact)
        self.assertIn("median", artifact)

    def test_each_model_family_produces_binary_probabilities(self):
        frame = pd.DataFrame({
            "a": np.arange(40, dtype=float),
            "b": np.tile([0.0, 1.0], 20),
            "win": np.tile([0, 1], 20),
        })
        specs = (
            ModelSpec("hgb", "hgb", 2, 20, 2, 1.0),
            ModelSpec("extra", "extra_trees", 4, 20, 2, 1.0),
            ModelSpec("forest", "random_forest", 4, 20, 2, 1.0),
            ModelSpec("linear", "logistic", None, 200, 2, 1.0),
        )

        for spec in specs:
            with self.subTest(spec=spec.name):
                model, median = fit_candidate(frame, ["a", "b"], "win", spec)
                probability = model.predict_proba(frame[["a", "b"]].fillna(median))[:, 1]

                self.assertEqual(probability.shape, (40,))
                self.assertTrue(np.all((probability >= 0) & (probability <= 1)))


if __name__ == "__main__":
    unittest.main()
