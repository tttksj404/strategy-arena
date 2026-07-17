import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import engine
import numpy as np


class CapturingProbabilityModel:
    def __init__(self):
        self.values = None

    def predict_proba(self, values):
        self.values = values.copy()
        return np.tile([0.5, 0.5], (len(values), 1))


class FixedProbabilityModel:
    def __init__(self, probability):
        self.probability = np.asarray(probability, dtype=float)

    def predict_proba(self, values):
        probability = self.probability[:len(values)]
        return np.column_stack((1.0 - probability, probability))


class KraPredictionPhaseTestCase(unittest.TestCase):
    def test_parse_kra_race_starts_reads_official_weekly_table(self):
        page = (
            "<table><tr><th>순</th><th>지역</th><th>경주일자</th><th>경주</th>"
            "<th>등급</th><th>거리</th><th>출전</th><th>구분</th><th>출발시각</th></tr>"
            "<tr><td>30</td><td>서울</td><td>2026/07/11(토)</td><td>7</td>"
            "<td>혼4</td><td>1400</td><td>11</td><td>일반</td><td>16:05</td></tr></table>"
        )
        starts = engine.parse_kra_race_starts(page)
        self.assertEqual(datetime(2026, 7, 11, 16, 5), starts[("20260711", "서울", 7)])

    def test_kra_odds_snapshot_is_fresh_only_before_official_start(self):
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]
        kst = timezone(timedelta(hours=9))
        start = datetime(2026, 7, 11, 16, 5)
        with patch.object(engine, "_kra_official_race_start", return_value=start):
            before = engine.kra_odds_snapshot_metadata(
                starters,
                "20260711",
                "서울",
                "7",
                datetime(2026, 7, 11, 15, 55, tzinfo=kst),
            )
            after = engine.kra_odds_snapshot_metadata(
                starters,
                "20260711",
                "서울",
                "7",
                datetime(2026, 7, 11, 16, 6, tzinfo=kst),
            )
        self.assertTrue(before["odds_snapshot_fresh"])
        self.assertFalse(after["odds_snapshot_fresh"])

    def test_kra_odds_snapshot_rejects_rows_with_results(self):
        starters = [
            {"chulNo": "1", "winOdds": "2.0", "ord": "1"},
            {"chulNo": "2", "winOdds": "4.0", "ord": "2"},
            {"chulNo": "3", "winOdds": "8.0", "ord": "3"},
        ]
        kst = timezone(timedelta(hours=9))
        with patch.object(
            engine,
            "_kra_official_race_start",
            return_value=datetime(2026, 7, 11, 16, 5),
        ):
            metadata = engine.kra_odds_snapshot_metadata(
                starters,
                "20260711",
                "서울",
                "7",
                datetime(2026, 7, 11, 15, 55, tzinfo=kst),
            )
        self.assertFalse(metadata["odds_snapshot_fresh"])

    def test_score_kra_applies_pairwise_reranker_only_to_pre_race_ranking(self):
        model = {
            "win": FixedProbabilityModel([0.40, 0.35, 0.25]),
            "plc": FixedProbabilityModel([0.70, 0.65, 0.55]),
            "cols": ["chulNo"],
            "med": {"chulNo": 2.0},
            "num": ["chulNo"],
            "rel": [],
            "jk_prior": {},
            "tr_prior": {},
            "global_win_rate": 0.1,
            "horse_history": {"records": {}},
            "pairwise": {
                "estimator": FixedProbabilityModel([0.10, 0.90, 0.90]),
                "median": {"chulNo": 2.0},
                "weight": 0.5,
                "top_k": 3,
                "enabled": True,
            },
            "live_market_weight": 1.0,
        }
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
            {"chulNo": "3", "winOdds": "8.0"},
        ]

        with patch.object(engine, "load_kra_model", return_value=(model, None)):
            pre_rows, pre_error = engine.score_kra(starters)
            live_rows, live_error = engine.score_kra(starters, use_market=True)

        self.assertIsNone(pre_error)
        self.assertIsNone(live_error)
        self.assertEqual(pre_rows[0]["bno"], 2)
        self.assertAlmostEqual(next(row for row in pre_rows if row["bno"] == 1)["pwin"], 0.4)
        self.assertTrue(pre_rows[0]["rerank_applied"])
        self.assertEqual(live_rows[0]["bno"], 1)
        self.assertFalse(live_rows[0]["rerank_applied"])

    def test_score_kra_keeps_baseline_order_when_pairwise_candidate_is_disabled(self):
        model = {
            "win": FixedProbabilityModel([0.40, 0.35, 0.25]),
            "plc": FixedProbabilityModel([0.70, 0.65, 0.55]),
            "cols": ["chulNo"],
            "med": {"chulNo": 2.0},
            "num": ["chulNo"],
            "rel": [],
            "jk_prior": {},
            "tr_prior": {},
            "global_win_rate": 0.1,
            "horse_history": {"records": {}},
            "pairwise": {
                "estimator": FixedProbabilityModel([0.10, 0.90, 0.90]),
                "median": {"chulNo": 2.0},
                "weight": 0.5,
                "top_k": 3,
                "enabled": False,
            },
        }
        starters = [{"chulNo": "1"}, {"chulNo": "2"}, {"chulNo": "3"}]

        with patch.object(engine, "load_kra_model", return_value=(model, None)):
            rows, error = engine.score_kra(starters)

        self.assertIsNone(error)
        self.assertEqual(rows[0]["bno"], 1)
        self.assertFalse(rows[0]["rerank_applied"])

    def test_score_kra_applies_horse_history_snapshot_at_race_date(self):
        win_model = CapturingProbabilityModel()
        place_model = CapturingProbabilityModel()
        model = {
            "win": win_model,
            "plc": place_model,
            "cols": ["chulNo", "hr_win_prior", "hr_days_since"],
            "med": {},
            "num": [],
            "rel": [],
            "jk_prior": {},
            "tr_prior": {},
            "global_win_rate": 0.1,
            "horse_history": {
                "records": {
                    "A": {
                        "win_prior": 0.4,
                        "place_prior": 0.6,
                        "finish_prior": 0.2,
                        "starts_log": 2.0,
                        "last_date": "20250108",
                    }
                }
            },
        }
        starters = [
            {"chulNo": "1", "hrNo": "A"},
            {"chulNo": "2", "hrNo": "NEW"},
        ]

        with patch.object(engine, "load_kra_model", return_value=(model, None)):
            rows, error = engine.score_kra(starters, meta={"ymd": "20250115"})

        self.assertIsNone(error)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(win_model.values.iloc[0]["hr_win_prior"], 0.4)
        self.assertAlmostEqual(win_model.values.iloc[0]["hr_days_since"], 7.0)
        self.assertAlmostEqual(win_model.values.iloc[1]["hr_win_prior"], 0.1)

    def test_market_probability_requires_a_complete_board_and_removes_overround(self):
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]

        probability = engine._kra_market_probabilities(starters)

        self.assertAlmostEqual(probability[1], 2.0 / 3.0)
        self.assertAlmostEqual(probability[2], 1.0 / 3.0)
        self.assertIsNone(engine._kra_market_probabilities(starters[:1]))
        self.assertIsNone(engine._kra_market_probabilities([starters[0], {"chulNo": "2"}]))

    def test_predict_kra_exposes_win_probability_leader_as_top(self):
        rows = [
            {"bno": 1, "name": "연대우세", "grade": "", "pwin": 0.30, "pplc": 0.80},
            {"bno": 2, "name": "우승우세", "grade": "", "pwin": 0.55, "pplc": 0.65},
            {"bno": 3, "name": "기타", "grade": "", "pwin": 0.15, "pplc": 0.40},
        ]

        with patch.object(engine, "score_kra", return_value=(rows, None)):
            result = engine.predict_kra([{"chulNo": "1"}], meta={"race_no": 1})

        self.assertEqual(result["top"]["bno"], 2)
        self.assertEqual(result["prediction_phase"], "pre_race")

    def test_predict_kra_uses_restricted_rerank_for_top_and_win_pick(self):
        rows = [
            {"bno": 1, "name": "재정렬", "grade": "", "pwin": 0.35, "pplc": 0.70, "rank_score": 0.51, "rerank_applied": True},
            {"bno": 2, "name": "기존선두", "grade": "", "pwin": 0.40, "pplc": 0.75, "rank_score": 0.50, "rerank_applied": True},
            {"bno": 3, "name": "기타", "grade": "", "pwin": 0.25, "pplc": 0.60, "rank_score": 0.25, "rerank_applied": True},
        ]

        with patch.object(engine, "score_kra", return_value=(rows, None)):
            result = engine.predict_kra([{"chulNo": "1"}], meta={"race_no": 1})

        win_pick = next(pick for pick in result["picks"] if pick["code"] == "단승")
        self.assertEqual(result["top"]["bno"], 1)
        self.assertTrue(win_pick["pick"][0].startswith("1번"))
        self.assertEqual(result["selective_conf"]["tier"], "normal")

    def test_predict_kra_marks_live_phase_only_with_complete_positive_odds(self):
        rows = [
            {"bno": 1, "name": "일번", "grade": "", "pwin": 0.60, "pplc": 0.80},
            {"bno": 2, "name": "이번", "grade": "", "pwin": 0.40, "pplc": 0.70},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]

        with patch.object(engine, "score_kra", return_value=(rows, None)):
            result = engine.predict_kra(starters, meta={"odds_snapshot_fresh": True})

        self.assertEqual(result["prediction_phase"], "live_odds")
        self.assertTrue(result["market_used"])

    def test_predict_kra_reports_market_pick_source_for_fresh_complete_odds(self):
        rows = [
            {"bno": 1, "name": "일번", "grade": "", "pwin": 0.60, "pplc": 0.80},
            {"bno": 2, "name": "이번", "grade": "", "pwin": 0.40, "pplc": 0.70},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]

        with patch.object(engine, "score_kra", return_value=(rows, None)):
            result = engine.predict_kra(starters, meta={"odds_snapshot_fresh": True})

        self.assertEqual(result["pick_source"], "market")

    def test_market_if_odds_uses_lowest_place_odds_for_place_pick(self):
        rows = [
            {"bno": 2, "name": "모델연승", "grade": "", "pwin": 0.50, "pplc": 0.90},
            {"bno": 3, "name": "모델상대", "grade": "", "pwin": 0.30, "pplc": 0.85},
            {"bno": 1, "name": "시장연승", "grade": "", "pwin": 0.20, "pplc": 0.40},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0", "plcOdds": "1.1"},
            {"chulNo": "2", "winOdds": "3.0", "plcOdds": "1.8"},
            {"chulNo": "3", "winOdds": "4.0", "plcOdds": "2.4"},
        ]

        with patch.dict(os.environ, {"KRA_PICK_POLICY": "market_if_odds"}, clear=False), \
             patch.object(engine, "score_kra", return_value=(rows, None)):
            result = engine.predict_kra(starters, meta={"odds_snapshot_fresh": True})

        place_pick = next(pick for pick in result["picks"] if pick["code"] == "연승")
        self.assertEqual(result["pick_source"], "market")
        self.assertEqual(place_pick["pick_source"], "market")
        self.assertEqual(place_pick["pick"], ["1번 시장연승"])
        self.assertEqual(place_pick["prob"], "plcOdds 최저 기준 픽")

    def test_market_if_odds_keeps_model_place_pick_when_place_odds_missing(self):
        rows = [
            {"bno": 2, "name": "모델연승", "grade": "", "pwin": 0.50, "pplc": 0.90},
            {"bno": 3, "name": "모델상대", "grade": "", "pwin": 0.30, "pplc": 0.85},
            {"bno": 1, "name": "시장연승", "grade": "", "pwin": 0.20, "pplc": 0.40},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "3.0"},
            {"chulNo": "3", "winOdds": "4.0"},
        ]

        with patch.dict(os.environ, {"KRA_PICK_POLICY": "market_if_odds"}, clear=False), \
             patch.object(engine, "score_kra", return_value=(rows, None)):
            result = engine.predict_kra(starters, meta={"odds_snapshot_fresh": True})

        place_pick = next(pick for pick in result["picks"] if pick["code"] == "연승")
        self.assertEqual(result["pick_source"], "market")
        self.assertEqual(place_pick["pick_source"], "model")
        self.assertEqual(place_pick["pick"], ["2번 모델연승", "3번 모델상대"])

    def test_predict_kra_reports_model_pick_source_without_fresh_snapshot_proof(self):
        rows = [
            {"bno": 1, "name": "일번", "grade": "", "pwin": 0.60, "pplc": 0.80},
            {"bno": 2, "name": "이번", "grade": "", "pwin": 0.40, "pplc": 0.70},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]

        with patch.dict(os.environ, {"KRA_PICK_POLICY": "current_gate"}, clear=False), \
             patch.object(engine, "score_kra", return_value=(rows, None)):
            result = engine.predict_kra(starters)

        self.assertEqual(result["pick_source"], "model")

    def test_current_gate_does_not_use_complete_odds_without_fresh_snapshot_proof(self):
        rows = [
            {"bno": 1, "name": "일번", "grade": "", "pwin": 0.60, "pplc": 0.80},
            {"bno": 2, "name": "이번", "grade": "", "pwin": 0.40, "pplc": 0.70},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]

        with patch.dict(os.environ, {"KRA_PICK_POLICY": "current_gate"}, clear=False), \
             patch.object(engine, "score_kra", return_value=(rows, None)):
            result = engine.predict_kra(starters)

        self.assertEqual(result["prediction_phase"], "pre_race")
        self.assertFalse(result["market_used"])

    def test_default_policy_rejects_complete_stale_odds_as_a_market_pick(self):
        rows = [
            {"bno": 1, "name": "시장선두", "grade": "", "pwin": 0.70, "pplc": 0.80},
            {"bno": 2, "name": "기타", "grade": "", "pwin": 0.30, "pplc": 0.70},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]

        with patch.dict(os.environ, {}, clear=False), \
             patch.object(engine, "score_kra", return_value=(rows, None)) as scorer:
            os.environ.pop("KRA_PICK_POLICY", None)
            result = engine.predict_kra(starters, meta={"odds_snapshot_fresh": False})

        self.assertEqual(result["prediction_phase"], "pre_race")
        self.assertFalse(result["market_used"])
        self.assertEqual(result["pick_source"], "model")
        self.assertFalse(scorer.call_args.kwargs["use_market"])

    def test_market_if_odds_cannot_bypass_the_fresh_snapshot_requirement(self):
        rows = [
            {"bno": 1, "name": "시장선두", "grade": "", "pwin": 0.70, "pplc": 0.80},
            {"bno": 2, "name": "기타", "grade": "", "pwin": 0.30, "pplc": 0.70},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]

        with patch.dict(os.environ, {"KRA_PICK_POLICY": "market_if_odds"}, clear=False), \
             patch.object(engine, "score_kra", return_value=(rows, None)) as scorer:
            result = engine.predict_kra(starters, meta={"odds_snapshot_fresh": False})

        self.assertFalse(result["market_used"])
        self.assertEqual(result["pick_source"], "model")
        self.assertFalse(scorer.call_args.kwargs["use_market"])

    def test_weak_disagreement_policy_cannot_bypass_the_fresh_snapshot_requirement(self):
        model_rows = [
            {"bno": 1, "name": "모델선두", "grade": "", "pwin": 0.70, "pplc": 0.80},
            {"bno": 2, "name": "기타", "grade": "", "pwin": 0.30, "pplc": 0.70},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "2.0"},
            {"chulNo": "2", "winOdds": "4.0"},
        ]

        with patch.dict(os.environ, {"KRA_PICK_POLICY": "market_except_weak_disagree"}, clear=False), \
             patch.object(engine, "score_kra", return_value=(model_rows, None)) as scorer:
            result = engine.predict_kra(starters, meta={"odds_snapshot_fresh": False})

        self.assertFalse(result["market_used"])
        self.assertEqual(result["pick_source"], "model")
        self.assertFalse(scorer.call_args.kwargs["use_market"])

    def test_default_policy_keeps_model_when_odds_are_missing_or_partial(self):
        rows = [
            {"bno": 1, "name": "일번", "grade": "", "pwin": 0.60, "pplc": 0.80},
            {"bno": 2, "name": "이번", "grade": "", "pwin": 0.40, "pplc": 0.70},
        ]

        with patch.dict(os.environ, {}, clear=False), \
             patch.object(engine, "score_kra", return_value=(rows, None)):
            os.environ.pop("KRA_PICK_POLICY", None)
            no_odds = engine.predict_kra([{"chulNo": "1"}, {"chulNo": "2"}])
            partial = engine.predict_kra([{"chulNo": "1", "winOdds": "2.0"}, {"chulNo": "2"}])

        self.assertFalse(no_odds["market_used"])
        self.assertEqual(no_odds["pick_source"], "model")
        self.assertFalse(partial["market_used"])
        self.assertEqual(partial["pick_source"], "model")

    def test_weak_open_disagreement_policy_keeps_model_pick(self):
        model_rows = [
            {"bno": 2, "name": "모델선두", "grade": "", "pwin": 0.55, "pplc": 0.80, "rank_score": 0.55},
            {"bno": 1, "name": "시장선두", "grade": "", "pwin": 0.45, "pplc": 0.70, "rank_score": 0.45},
        ]
        starters = [
            {"chulNo": "1", "winOdds": "5.0"},
            {"chulNo": "2", "winOdds": "5.4"},
        ]

        with patch.dict(os.environ, {"KRA_PICK_POLICY": "market_except_weak_disagree"}, clear=False), \
             patch.object(engine, "score_kra", return_value=(model_rows, None)) as scorer:
            result = engine.predict_kra(starters)

        self.assertFalse(result["market_used"])
        self.assertEqual(result["pick_source"], "model")
        scorer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
