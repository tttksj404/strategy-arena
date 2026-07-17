#!/usr/bin/env python3
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_kcycle_surprise_top2 as surprise


def _snapshot(date, race_no, actual_order, board):
    return {
        "schema": "kcycle_trifecta_snapshot_v1",
        "date": date,
        "meet": "광명",
        "race_no": str(race_no),
        "actual_order": "-".join(str(item) for item in actual_order),
        "board": board,
        "best20": sorted(board.items(), key=lambda item: item[1])[:20],
    }


class SurpriseTop2ExperimentTestCase(unittest.TestCase):
    def test_snapshot_analysis_extracts_unexpected_top2_common_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.jsonl"
            rows = [
                _snapshot("20260701", 1, [4, 2, 1], {
                    "1-3-2": 2.0,
                    "3-1-2": 2.4,
                    "1-2-3": 4.0,
                    "2-1-3": 5.0,
                    "4-2-1": 80.0,
                    "2-4-1": 95.0,
                }),
                _snapshot("20260701", 2, [1, 3, 2], {
                    "1-3-2": 2.0,
                    "3-1-2": 2.3,
                    "1-2-3": 4.0,
                    "3-2-1": 6.0,
                    "4-1-3": 70.0,
                }),
            ]
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            result = surprise.analyze_snapshot_surprises(path, surprise_rank_min=4)

        self.assertEqual(result["summary"]["races"], 2)
        self.assertEqual(result["summary"]["actual_top2_slots"], 4)
        self.assertEqual(result["summary"]["surprise_top2_count"], 1)
        self.assertEqual(result["surprises"][0]["runner"], 4)
        self.assertEqual(result["surprises"][0]["actual_position"], 1)
        self.assertEqual(result["surprises"][0]["market_rank"], 4)
        self.assertEqual(result["common_gates"][0]["gate"], 4)
        self.assertEqual(result["common_gates"][0]["count"], 1)

    def test_walk_forward_gate_boost_reports_baseline_and_candidate_rates(self):
        records = [
            {
                "key": ("keirin", "20260701", "광명", "1"),
                "actual_top2": [4, 2],
                "scores": {1: 0.6, 2: 0.5, 3: 0.4, 4: 0.1},
            },
            {
                "key": ("keirin", "20260702", "광명", "1"),
                "actual_top2": [4, 1],
                "scores": {1: 0.6, 2: 0.55, 3: 0.45, 4: 0.2},
            },
            {
                "key": ("keirin", "20260703", "광명", "1"),
                "actual_top2": [4, 1],
                "scores": {1: 0.6, 2: 0.55, 3: 0.45, 4: 0.2},
            },
        ]

        result = surprise.evaluate_walk_forward_gate_boost(records, surprise_rank_min=4, min_gate_starts=1, boost_weight=2.0)

        self.assertEqual(result["races"], 3)
        self.assertEqual(result["baseline_top2_slots_hit"], 3)
        self.assertGreater(result["candidate_top2_slots_hit"], result["baseline_top2_slots_hit"])
        self.assertGreater(result["candidate_top2_slot_hit_rate"], result["baseline_top2_slot_hit_rate"])
        self.assertGreaterEqual(result["adjusted_races"], 1)

    def test_prediction_analysis_uses_model_podium_scores_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "strategy.sqlite"
            outcome_path = Path(tmp) / "outcomes.jsonl"
            payload = {
                "rows": [
                    {"bno": 1, "name": "축선수", "pwin": 0.60, "pplc": 0.82},
                    {"bno": 2, "name": "상위권", "pwin": 0.30, "pplc": 0.70},
                    {"bno": 3, "name": "중위권", "pwin": 0.20, "pplc": 0.52},
                    {"bno": 4, "name": "놓친선수", "pwin": 0.05, "pplc": 0.20},
                ],
            }
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    "CREATE TABLE prediction__predictions ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, sport TEXT, race_date TEXT, meet TEXT, "
                    "race_no TEXT, status TEXT, market_used INTEGER, payload_json TEXT, created_at TEXT)"
                )
                conn.execute(
                    "INSERT INTO prediction__predictions "
                    "(sport, race_date, meet, race_no, status, market_used, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("keirin", "2026-07-03", "광명", "1", "ready", 1, json.dumps(payload, ensure_ascii=False), "2026-07-03T06:00:00+00:00"),
                )
                conn.commit()
            outcome_path.write_text(
                json.dumps({"date": "20260703", "meet": "광명", "race_no": "1", "actual_order": [4, 1, 2]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            result = surprise.analyze_prediction_surprises(db_path, outcome_path, surprise_rank_min=4)

        self.assertEqual(result["summary"]["source"], "prediction_model")
        self.assertEqual(result["summary"]["races"], 1)
        self.assertEqual(result["summary"]["surprise_top2_count"], 1)
        self.assertEqual(result["surprises"][0]["runner"], 4)
        self.assertEqual(result["common_gates"][0]["gate"], 4)


if __name__ == "__main__":
    unittest.main()
