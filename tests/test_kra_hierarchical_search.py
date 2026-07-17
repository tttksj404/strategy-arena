import unittest

import numpy as np
import pandas as pd

from kra_drug_assay import build_fold_tensor
from kra_drug_discovery import Genome
from kra_hierarchical_search import hierarchical_screen


class KraHierarchicalSearchTestCase(unittest.TestCase):
    def test_confirmation_is_only_used_after_discovery_stages(self):
        frame = pd.DataFrame({"rk": ["r1", "r1"], "win": [1, 0]})
        model_scores = np.asarray([[0.8, 0.2], [0.2, 0.8]])
        market = np.asarray([0.7, 0.3])
        folds = tuple(
            build_fold_tensor(f"fold-{index}", frame, model_scores, market, 0)
            for index in range(4)
        )
        scores, reports = hierarchical_screen(
            (Genome((1.0, 0.0), 0.0), Genome((0.0, 1.0), 0.0)),
            folds,
            batch_size=2,
            stage_beams=(1, 1, 1),
        )
        self.assertEqual(len(scores), 1)
        self.assertEqual([report.fold_count for report in reports], [1, 2, 3])
        self.assertTrue(all(report.output_count == 1 for report in reports))

    def test_stage_reports_account_for_multifidelity_assays(self):
        frame = pd.DataFrame({"rk": ["r1", "r1"], "win": [1, 0]})
        model_scores = np.asarray([[0.8, 0.2], [0.2, 0.8]])
        market = np.asarray([0.7, 0.3])
        folds = tuple(
            build_fold_tensor(f"fold-{index}", frame, model_scores, market, 0)
            for index in range(4)
        )
        genomes = (Genome((1.0, 0.0), 0.0),) * 3
        _, reports = hierarchical_screen(
            genomes,
            folds,
            batch_size=2,
            stage_beams=(2, 1, 1),
        )
        self.assertEqual(
            [report.assay_evaluations for report in reports],
            [3, 4, 3],
        )


if __name__ == "__main__":
    unittest.main()
