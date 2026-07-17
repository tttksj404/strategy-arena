#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import engine
import update_prediction_feedback as feedback


def _prediction_payload(rows):
    return {
        "ok": True,
        "status": "ready",
        "rows": rows,
        "top": rows[0],
    }


class PredictionFeedbackTestCase(unittest.TestCase):
    def test_build_feedback_links_latest_prediction_to_actual_result_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "strategy.sqlite"
            snapshot_path = Path(tmp) / "snapshots.jsonl"
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
                    (
                        "keirin",
                        "2026-06-28",
                        "광명",
                        "5",
                        "ready",
                        0,
                        json.dumps(_prediction_payload([
                            {"bno": 2, "name": "강석호", "pwin": 0.55, "pplc": 0.75},
                            {"bno": 4, "name": "이승원", "pwin": 0.35, "pplc": 0.70},
                        ]), ensure_ascii=False),
                        "2026-07-02T23:50:00+00:00",
                    ),
                )
                conn.execute(
                    "INSERT INTO prediction__predictions "
                    "(sport, race_date, meet, race_no, status, market_used, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "keirin",
                        "2026-06-28",
                        "광명",
                        "5",
                        "ready",
                        1,
                        json.dumps(_prediction_payload([
                            {"bno": 4, "name": "이승원", "pwin": 0.62, "pplc": 0.82},
                            {"bno": 2, "name": "강석호", "pwin": 0.30, "pplc": 0.61},
                            {"bno": 6, "name": "박민수", "pwin": 0.08, "pplc": 0.44},
                        ]), ensure_ascii=False),
                        "2026-07-02T23:58:00+00:00",
                    ),
                )
                conn.commit()

            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema": "kcycle_trifecta_snapshot_v1",
                        "date": "20260628",
                        "meet": "광명",
                        "race_no": "5",
                        "actual_order": "4-2-6",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = feedback.build_feedback(db_path, snapshot_path, min_starts_for_live_adjustment=3)

        self.assertEqual(result["summary"]["matched_races"], 1)
        self.assertEqual(result["summary"]["raw_prediction_rows"], 2)
        self.assertEqual(result["summary"]["prediction_rows"], 1)
        self.assertEqual(result["summary"]["duplicate_prediction_rows_ignored"], 1)
        self.assertEqual(result["summary"]["oos_status"], "insufficient_matched_races")
        self.assertFalse(result["summary"]["learning_enabled"])
        self.assertEqual(result["summary"]["top1_hit_rate"], 1.0)
        self.assertEqual(result["summary"]["exact_trifecta_hit_rate"], 1.0)
        racers = result["sports"]["keirin"]["participants"]
        self.assertEqual(racers["이승원"]["starts"], 1)
        self.assertEqual(racers["이승원"]["wins"], 1)
        self.assertEqual(racers["이승원"]["podiums"], 1)
        self.assertEqual(racers["강석호"]["starts"], 1)
        self.assertEqual(racers["강석호"]["wins"], 0)
        self.assertEqual(racers["강석호"]["podiums"], 1)

    def test_build_feedback_ignores_races_without_actual_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "strategy.sqlite"
            snapshot_path = Path(tmp) / "snapshots.jsonl"
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
                    (
                        "keirin",
                        "2026-06-28",
                        "광명",
                        "5",
                        "ready",
                        1,
                        json.dumps(_prediction_payload([
                            {"bno": 4, "name": "이승원", "pwin": 0.62, "pplc": 0.82},
                        ]), ensure_ascii=False),
                        "2026-07-02T23:58:00+00:00",
                    ),
                )
                conn.commit()

            snapshot_path.write_text(
                json.dumps({"date": "20260628", "meet": "광명", "race_no": "5", "actual_order": None}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            result = feedback.build_feedback(db_path, snapshot_path)

        self.assertEqual(result["summary"]["matched_races"], 0)
        self.assertEqual(result["sports"]["keirin"]["participants"], {})

    def test_build_feedback_links_kra_result_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "strategy.sqlite"
            snapshot_path = Path(tmp) / "snapshots.jsonl"
            kra_db_path = Path(tmp) / "kra.sqlite"
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
                    (
                        "horse",
                        "2026-06-28",
                        "서울",
                        "3",
                        "ready",
                        0,
                        json.dumps(_prediction_payload([
                            {"bno": 7, "name": "라이트닝", "pwin": 0.34, "pplc": 0.61},
                            {"bno": 2, "name": "윈드", "pwin": 0.24, "pplc": 0.55},
                            {"bno": 9, "name": "스톤", "pwin": 0.18, "pplc": 0.49},
                        ]), ensure_ascii=False),
                        "2026-07-02T23:58:00+00:00",
                    ),
                )
                conn.commit()
            snapshot_path.write_text("", encoding="utf-8")
            with closing(sqlite3.connect(kra_db_path)) as conn:
                conn.execute("CREATE TABLE race_result (meet TEXT, rcDate TEXT, rcNo TEXT, chulNo TEXT, ord TEXT)")
                conn.executemany(
                    "INSERT INTO race_result (meet, rcDate, rcNo, chulNo, ord) VALUES (?, ?, ?, ?, ?)",
                    [
                        ("서울", "20260628", "3", "7", "1"),
                        ("서울", "20260628", "3", "2", "2"),
                        ("서울", "20260628", "3", "9", "3"),
                    ],
                )
                conn.commit()

            result = feedback.build_feedback(db_path, snapshot_path, kra_db_path=kra_db_path)

        self.assertEqual(result["summary"]["matched_races"], 1)
        self.assertEqual(result["sports"]["horse"]["participants"]["라이트닝"]["wins"], 1)
        self.assertEqual(result["sports"]["horse"]["participants"]["윈드"]["podiums"], 1)

    def test_build_feedback_links_kcycle_outcome_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "strategy.sqlite"
            snapshot_path = Path(tmp) / "snapshots.jsonl"
            outcome_path = Path(tmp) / "kcycle_outcomes.jsonl"
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
                    (
                        "keirin",
                        "2026-07-03",
                        "광명",
                        "5",
                        "ready",
                        1,
                        json.dumps(_prediction_payload([
                            {"bno": 4, "name": "김로운", "pwin": 0.64, "pplc": 0.84},
                            {"bno": 1, "name": "박훈재", "pwin": 0.34, "pplc": 0.66},
                            {"bno": 2, "name": "이정민", "pwin": 0.09, "pplc": 0.16},
                        ]), ensure_ascii=False),
                        "2026-07-03T06:26:13+00:00",
                    ),
                )
                conn.commit()
            snapshot_path.write_text("", encoding="utf-8")
            outcome_path.write_text(
                json.dumps(
                    {
                        "schema": "kcycle_result_outcome_v1",
                        "date": "20260703",
                        "meet": "광명",
                        "race_no": "05",
                        "actual_order": [1, 2, 4],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = feedback.build_feedback(
                db_path,
                snapshot_path,
                kcycle_outcome_path=outcome_path,
            )

        self.assertEqual(result["summary"]["matched_races"], 1)
        self.assertEqual(result["summary"]["top1_hits"], 0)
        self.assertEqual(result["summary"]["exact_trifecta_hits"], 0)
        self.assertEqual(result["sports"]["keirin"]["participants"]["박훈재"]["wins"], 1)
        self.assertEqual(result["sports"]["keirin"]["participants"]["김로운"]["podiums"], 1)

    def test_participant_learning_adjusts_only_when_prior_has_enough_starts(self):
        priors = {
            "enabled": True,
            "min_starts_for_live_adjustment": 3,
            "sports": {
                "keirin": {
                    "participants": {
                        "강석호": {"starts": 6, "win_delta": 0.08, "podium_delta": 0.04},
                        "이승원": {"starts": 2, "win_delta": -0.08, "podium_delta": -0.04},
                    }
                }
            },
        }
        rows = [
            {"bno": 4, "name": "이승원", "grade": "선발", "pwin": 0.52, "pplc": 0.80},
            {"bno": 2, "name": "강석호", "grade": "선발", "pwin": 0.50, "pplc": 0.77},
        ]

        with patch.object(engine, "load_participant_learning_priors", return_value=priors), \
             patch.dict(os.environ, {"PARTICIPANT_LEARNING_ENABLED": "1", "PARTICIPANT_LEARNING_WEIGHT": "0.5"}, clear=False):
            adjusted = engine.apply_participant_learning(rows, "keirin")

        by_name = {row["name"]: row for row in adjusted}
        self.assertEqual(by_name["이승원"]["pwin"], 0.52)
        self.assertGreater(by_name["강석호"]["pwin"], 0.52)
        self.assertEqual(by_name["강석호"]["learning_starts"], 6)
        self.assertAlmostEqual(by_name["강석호"]["learning_win_delta"], 0.08)

    def test_walk_forward_candidate_is_deployable_only_on_oos_lift(self):
        records = []
        for offset in range(5):
            records.append({
                "key": ("keirin", f"202606{20 + offset}", "광명", "1"),
                "actual": [1, 3, 4],
                "baseline_order": [2, 1, 3, 4],
                "rows": [
                    {"bno": 2, "name": "과대평가", "pwin": 0.45, "pplc": 0.52},
                    {"bno": 1, "name": "저평가", "pwin": 0.40, "pplc": 0.50},
                    {"bno": 3, "name": "안정권", "pwin": 0.15, "pplc": 0.49},
                    {"bno": 4, "name": "추입권", "pwin": 0.10, "pplc": 0.48},
                ],
            })
        candidate = {
            "name": "test_activity_recovery",
            "family": "lead_optimization",
            "min_starts": 1,
            "alpha": 0.0,
            "weight": 1.0,
            "delta_limit": 0.08,
        }

        result = feedback.evaluate_learning_candidates(records, candidates=[candidate], min_oos_races=3)

        self.assertEqual(result["status"], "deployable")
        self.assertTrue(result["deployable"])
        self.assertEqual(result["baseline"]["top1_hits"], 0)
        self.assertGreater(result["best_candidate"]["top1_hits"], 0)
        self.assertGreater(result["best_candidate"]["top1_lift"], 0)

    def test_walk_forward_candidate_stays_blocked_without_enough_matched_races(self):
        records = [{
            "key": ("keirin", "20260620", "광명", "1"),
            "actual": [1, 2, 3],
            "baseline_order": [2, 1, 3],
            "rows": [
                {"bno": 2, "name": "과대평가", "pwin": 0.45, "pplc": 0.55},
                {"bno": 1, "name": "저평가", "pwin": 0.40, "pplc": 0.50},
            ],
        }]
        candidate = {
            "name": "test_activity_recovery",
            "family": "lead_optimization",
            "min_starts": 1,
            "alpha": 0.0,
            "weight": 1.0,
            "delta_limit": 0.08,
        }

        result = feedback.evaluate_learning_candidates(records, candidates=[candidate], min_oos_races=3)

        self.assertEqual(result["status"], "insufficient_matched_races")
        self.assertFalse(result["deployable"])

    def test_grade_policy_deploys_only_when_precision_beats_baseline(self):
        records = [
            {
                "key": ("keirin", "20260620", "광명", "13"),
                "actual": [2, 1, 3],
                "baseline_order": [1, 2, 3],
                "rows": [
                    {"bno": 1, "name": "특선과신", "grade": "특선", "pwin": 0.46, "pplc": 0.75},
                    {"bno": 2, "name": "특선역전", "grade": "특선", "pwin": 0.30, "pplc": 0.68},
                    {"bno": 3, "name": "상대", "grade": "특선", "pwin": 0.12, "pplc": 0.42},
                ],
            },
            {
                "key": ("keirin", "20260621", "광명", "14"),
                "actual": [2, 1, 3],
                "baseline_order": [1, 2, 3],
                "rows": [
                    {"bno": 1, "name": "특선과신2", "grade": "특선", "pwin": 0.47, "pplc": 0.76},
                    {"bno": 2, "name": "특선역전2", "grade": "특선", "pwin": 0.31, "pplc": 0.69},
                    {"bno": 3, "name": "상대2", "grade": "특선", "pwin": 0.11, "pplc": 0.41},
                ],
            },
            {
                "key": ("keirin", "20260622", "광명", "3"),
                "actual": [1, 2, 3],
                "baseline_order": [1, 2, 3],
                "rows": [
                    {"bno": 1, "name": "선발수치", "grade": "선발", "pwin": 0.46, "pplc": 0.75},
                    {"bno": 2, "name": "선발상대", "grade": "선발", "pwin": 0.30, "pplc": 0.68},
                    {"bno": 3, "name": "선발3", "grade": "선발", "pwin": 0.12, "pplc": 0.42},
                ],
            },
        ]

        result = feedback.evaluate_grade_policy(records, min_oos_races=3, min_recommendations=1)

        self.assertEqual(result["status"], "deployable")
        self.assertTrue(result["deployable"])
        self.assertEqual(result["selected_policy"], "grade_context")
        self.assertEqual(result["baseline"]["recommended_races"], 3)
        self.assertEqual(result["candidate"]["recommended_races"], 1)
        self.assertGreater(result["candidate"]["top1_precision"], result["baseline"]["top1_precision"])

    def test_grade_policy_stays_baseline_without_precision_lift(self):
        records = [
            {
                "key": ("keirin", f"2026062{idx}", "광명", "3"),
                "actual": [1, 2, 3],
                "baseline_order": [1, 2, 3],
                "rows": [
                    {"bno": 1, "name": f"선발수치{idx}", "grade": "선발", "pwin": 0.46, "pplc": 0.75},
                    {"bno": 2, "name": f"선발상대{idx}", "grade": "선발", "pwin": 0.30, "pplc": 0.68},
                    {"bno": 3, "name": f"선발3-{idx}", "grade": "선발", "pwin": 0.12, "pplc": 0.42},
                ],
            }
            for idx in range(3)
        ]

        result = feedback.evaluate_grade_policy(records, min_oos_races=3, min_recommendations=1)

        self.assertEqual(result["status"], "no_precision_improvement")
        self.assertFalse(result["deployable"])
        self.assertEqual(result["selected_policy"], "baseline")

    def test_participant_learning_uses_oos_selected_weight_when_env_is_unset(self):
        priors = {
            "enabled": True,
            "min_starts_for_live_adjustment": 3,
            "learning_weight": 0.25,
            "sports": {
                "keirin": {
                    "participants": {
                        "강석호": {"starts": 6, "win_delta": 0.08, "podium_delta": 0.04},
                    }
                }
            },
        }
        rows = [{"bno": 2, "name": "강석호", "grade": "선발", "pwin": 0.50, "pplc": 0.70}]

        with patch.object(engine, "load_participant_learning_priors", return_value=priors), \
             patch.dict(os.environ, {}, clear=True):
            adjusted = engine.apply_participant_learning(rows, "keirin")

        self.assertAlmostEqual(adjusted[0]["pwin"], 0.52)
        self.assertAlmostEqual(adjusted[0]["pplc"], 0.71)


if __name__ == "__main__":
    unittest.main()
