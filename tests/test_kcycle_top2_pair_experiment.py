#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import experiment_kcycle_top2_pair as top2


def make_pair_advantage_board():
    board = {}
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) == 3:
                    board[f"{a}-{b}-{c}"] = 10000.0
    board["3-4-5"] = 8.0
    for third in (3, 4, 5, 6, 7):
        board[f"1-2-{third}"] = 18.0 + third / 10
        board[f"2-1-{third}"] = 19.0 + third / 10
    return board


class KcycleTop2PairExperimentTestCase(unittest.TestCase):
    def test_method_predictions_include_pair_mass_that_can_override_best_combo_top2(self):
        board = make_pair_advantage_board()

        predictions = top2.method_predictions(board)

        self.assertEqual(predictions["best_combo_top2"], [3, 4])
        self.assertEqual(predictions["unordered_pair_mass"], [1, 2])

    def test_analyze_records_scores_top2_slot_pair_and_ordered_pair_metrics(self):
        records = [
            {
                "date": "20240105",
                "stnd_yr": "2024",
                "actual_order": "1-2-3",
                "board_count": 210,
                "board": make_pair_advantage_board(),
            }
        ]

        result = top2.analyze_records(records)
        by_method = {row["method"]: row for row in result["methods"]}

        self.assertEqual(by_method["best_combo_top2"]["top2_slot_hits"], 0.0)
        self.assertEqual(by_method["unordered_pair_mass"]["top2_slot_hits"], 1.0)
        self.assertEqual(by_method["unordered_pair_mass"]["unordered_pair_hits"], 1.0)
        self.assertEqual(by_method["ordered_pair_mass"]["ordered_pair_hits"], 1.0)

    def test_run_writes_machine_and_markdown_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = os.path.join(tmp, "snapshots.jsonl")
            json_path = os.path.join(tmp, "out.json")
            md_path = os.path.join(tmp, "out.md")
            with open(snapshot_path, "w", encoding="utf-8") as f:
                f.write(top2.json_dumps({
                    "date": "20240105",
                    "stnd_yr": "2024",
                    "actual_order": "1-2-3",
                    "board_count": 210,
                    "board": make_pair_advantage_board(),
                }) + "\n")

            result = top2.run(snapshot_path, json_path, md_path)

            self.assertEqual(result["summary"]["race_count"], 1)
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))

    def test_hybrid_search_can_use_pair_method_only_on_lift_cohort(self):
        records = []
        for year in ("2018", "2024"):
            records.append({
                "date": f"{year}0105",
                "stnd_yr": year,
                "race_no": "1",
                "actual_order": "1-2-3",
                "board_count": 210,
                "board": make_pair_advantage_board(),
            })

        candidates = top2.search_hybrid_candidates(records, min_train_lift=0.0)

        self.assertGreater(candidates[0]["holdout_top2_lift_vs_best_combo"], 0.0)
        self.assertTrue(any("unordered_pair_mass" in row["method"] for row in candidates))


if __name__ == "__main__":
    unittest.main()
