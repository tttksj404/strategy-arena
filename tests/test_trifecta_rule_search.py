#!/usr/bin/env python3
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import search_kcycle_trifecta_rules as search
from test_live_decision import make_trifecta_candidate_board


class TrifectaRuleSearchTestCase(unittest.TestCase):
    def test_frame_builds_market_features_and_method_hits(self):
        board = make_trifecta_candidate_board()
        df = search.frame_from_records([
            {
                "date": "20260628",
                "stnd_yr": "2026",
                "actual_order": "5-1-7",
                "board_count": len(board),
                "board": board,
            }
        ])

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "year"], "2026")
        self.assertAlmostEqual(df.loc[0, "gap12"], 2.3)
        self.assertGreater(df.loc[0, "first_mass_best"], 0.0)
        for method in search.METHODS:
            self.assertTrue(bool(df.loc[0, f"hit_{method}"]))

    def test_predicates_are_deduped(self):
        board = make_trifecta_candidate_board()
        records = [
            {
                "date": f"201801{i + 1:02d}",
                "stnd_yr": "2018",
                "actual_order": "5-1-7",
                "board_count": len(board),
                "board": board,
            }
            for i in range(60)
        ]
        df = search.frame_from_records(records)

        preds = search.predicates(df, df["year"].isin(search.TRAIN_YEARS))
        names = [name for name, _ in preds]

        self.assertEqual(len(names), len(set(names)))
        self.assertIn("all", names)
        self.assertTrue(any(name.startswith("xdom_") for name in names))

    def test_xdom_methods_are_real_prediction_methods(self):
        board = make_trifecta_candidate_board()
        df = search.frame_from_records([
            {
                "date": "20260628",
                "stnd_yr": "2026",
                "actual_order": "5-1-7",
                "board_count": len(board),
                "board": board,
            }
        ])

        self.assertGreaterEqual(len(search.XDOM_METHODS), 5)
        for method in search.XDOM_METHODS:
            self.assertIn(f"hit_{method}", df.columns)
            self.assertTrue(bool(df.loc[0, f"hit_{method}"]))

    def test_eval_candidate_requires_holdout_year_coverage(self):
        hits = np.array([True] * 60 + [True] * 5 + [True] * 5 + [True] * 5)
        train_base = np.array([True] * 60 + [False] * 15)
        holdout_base = ~train_base
        holdout_year_masks = {
            "2024": np.array([False] * 60 + [True] * 5 + [False] * 10),
            "2025": np.array([False] * 65 + [True] * 5 + [False] * 5),
            "2026": np.array([False] * 70 + [True] * 5),
        }

        row = search.eval_candidate(
            hits,
            train_base,
            holdout_base,
            holdout_year_masks,
            "board_min",
            "all",
            np.ones(75, dtype=bool),
        )

        self.assertIsNotNone(row)
        self.assertEqual(row.status, "PASS_HOLDOUT_50")
        self.assertEqual(row.holdout_n, 15)


if __name__ == "__main__":
    unittest.main()
