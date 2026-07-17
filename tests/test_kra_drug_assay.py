import unittest

import numpy as np
import pandas as pd

from kra_drug_assay import build_fold_tensor, screen_genomes, screen_top1_matrix
from kra_drug_discovery import Genome


class KraDrugAssayTestCase(unittest.TestCase):
    def test_screening_compares_candidates_with_market_and_v4_by_race(self):
        frame = pd.DataFrame({
            "rk": ["r1", "r1", "r2", "r2"],
            "win": [1, 0, 0, 1],
        })
        v4 = np.asarray([0.6, 0.4, 0.7, 0.3])
        challenger = np.asarray([0.8, 0.2, 0.2, 0.8])
        market = np.asarray([0.7, 0.3, 0.4, 0.6])
        fold = build_fold_tensor(
            "fold",
            frame,
            np.vstack([v4, challenger]),
            market,
            v4_model_index=0,
        )
        genomes = (
            Genome((1.0, 0.0), 0.0),
            Genome((0.0, 1.0), 0.0),
        )

        scores = screen_genomes(genomes, (fold, fold, fold, fold), batch_size=2)

        self.assertEqual(scores[0].discovery_market_gaps_pp, (-50.0, -50.0, -50.0))
        self.assertEqual(scores[1].discovery_market_gaps_pp, (0.0, 0.0, 0.0))
        self.assertEqual(scores[1].confirmation_v4_lift_pp, 50.0)

    def test_top1_matrix_keeps_candidate_by_fold_orientation(self):
        frame = pd.DataFrame({"rk": ["r1", "r1"], "win": [1, 0]})
        fold = build_fold_tensor(
            "fold",
            frame,
            np.asarray([[0.8, 0.2], [0.2, 0.8]]),
            np.asarray([0.7, 0.3]),
            v4_model_index=0,
        )
        matrix = screen_top1_matrix((Genome((1.0, 0.0), 0.0), Genome((0.0, 1.0), 0.0)), (fold,), 2)
        self.assertEqual(matrix.shape, (2, 1))
        self.assertEqual(matrix[:, 0].tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
