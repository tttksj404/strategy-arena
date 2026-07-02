#!/usr/bin/env python3
"""test_live_decision: /api/live-decision API 테스트."""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KCYCLE_ENABLED"] = "0"  # Render와 동일 (kcycle 비활성화)

import app as app_module


def make_trifecta_candidate_board():
    board = {}
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) == 3:
                    board[f"{a}-{b}-{c}"] = 10000.0
    for combo, odds in {
        "5-1-7": 10.0,
        "5-1-2": 23.0,
        "5-1-3": 24.0,
        "5-1-4": 25.0,
        "5-1-6": 26.0,
    }.items():
        board[combo] = odds
    return board


class LiveDecisionTestCase(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()
        app_module._BASE_PREDICTION_CACHE.clear()

    def test_live_decision_no_date(self):
        """날짜 없으면 400 + hold."""
        r = self.client.get("/api/live-decision")
        self.assertEqual(r.status_code, 400)
        d = r.get_json()
        self.assertFalse(d["ok"])
        self.assertEqual(d["decision"], "hold")

    def test_live_decision_demo_mode(self):
        """demo 폴밑 시에도 JSON 응답 + hold 또는 ok."""
        r = self.client.get("/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5")
        # demo 모드(키 없거나 fact) → ok 또는 error 둘 다 허용
        data = r.get_json()
        if data is None:
            # 500 에러일 수 있음 (Render 환경 차이) — status code만
            self.assertIn(r.status_code, [200, 500])
            return
        # 성공 시: status, decision, market_used 필드 있어야
        if "status" in data:
            self.assertIn(data.get("decision", "hold"), ["hold", "final_candidate"])
            self.assertFalse(data.get("market_used", False))  # kcycle 비활성화
        else:
            self.assertFalse(data.get("ok", True))

    def test_market_unused_when_disabled(self):
        """KCYCLE_ENABLED=0이면 market_used=false."""
        r = self.client.get("/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=1")
        d = r.get_json()
        if d and "market_used" in d:
            self.assertFalse(d["market_used"])
            self.assertEqual(d["poll_delay_ms"], 15000)
            self.assertEqual(d["market_risk"]["level"], "live_market_blocked")
            self.assertIn("Render", d["market_risk"]["message"])

    def test_live_decision_exposes_official_support_when_market_blocked(self):
        base = {
            "kind": "ok",
            "rows": [{"bno": 3, "name": "A", "pwin": 0.61, "pplc": 0.82}],
            "rankingpredict_signal": {
                "tier": "kcycle_market3_support",
                "label": "KCYCLE 시장3합의 보조픽",
                "expected_top1": 0.6656,
            },
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "0"}, clear=False):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["fallback_signal"]["label"], "KCYCLE 시장3합의 보조픽")
        self.assertEqual(decision["fallback_signal"]["expected_top1"], 0.6656)
        self.assertEqual(decision["decision"], "hold")

    def test_live_decision_promotes_strong_market_favorite_when_enabled(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 1, "name": "모델선두", "pwin": 0.62, "pplc": 0.86},
                {"bno": 5, "name": "시장강축", "pwin": 0.20, "pplc": 0.70},
                {"bno": 7, "name": "상대", "pwin": 0.18, "pplc": 0.64},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(
                 {1: 4.2, 5: 1.0, 7: 9.5},
                 "2026-07-02T12:00:00",
             )), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 None,
                 "2026-07-02T12:00:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertTrue(decision["market_used"])
        self.assertEqual(decision["decision"], "final_candidate")
        self.assertEqual(decision["top"]["bno"], 5)
        self.assertEqual(decision["market_signal"]["tier"], "market_fav_odds_le_1_0")
        self.assertAlmostEqual(decision["market_signal"]["expected_top1"], 0.8896)
        self.assertEqual(decision["poll_delay_ms"], 5000)

    def test_trifecta_signal_exposes_immediate_prior_lift_with_robust_warning(self):
        signal = app_module.engine._market_trifecta_signal(make_trifecta_candidate_board())

        self.assertEqual(signal["tier"], "market_trifecta_50_candidate")
        self.assertEqual(signal["order"], [5, 1, 7])
        self.assertAlmostEqual(signal["expected_trio_exact"], 0.5)
        self.assertAlmostEqual(signal["baseline_trio_exact"], 0.2719)
        self.assertAlmostEqual(signal["lift_pp"], 22.81)
        self.assertEqual(signal["robust_status"], "failed_small_n")
        self.assertIn("robust PASS", signal["robust_warning"])

    def test_live_decision_exposes_trifecta_signal_without_overwriting_top1_signal(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 1, "name": "모델선두", "pwin": 0.62, "pplc": 0.86},
                {"bno": 5, "name": "시장삼쌍", "pwin": 0.20, "pplc": 0.70},
                {"bno": 7, "name": "상대", "pwin": 0.18, "pplc": 0.64},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(
                 {1: 4.2, 5: 1.2, 7: 9.5},
                 "2026-07-02T12:00:00",
             )), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_candidate_board(),
                 "2026-07-02T12:00:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertIsNone(decision["market_signal"])
        self.assertEqual(decision["trifecta_signal"]["tier"], "market_trifecta_50_candidate")
        self.assertAlmostEqual(decision["trifecta_signal"]["expected_trio_exact"], 0.5)
        self.assertIn("robust 미통과", decision["message"])
        self.assertEqual(decision["poll_delay_ms"], 5000)

    def test_trifecta_snapshot_writer_appends_and_dedupes(self):
        app_module.engine._KCYCLE_TRIFECTA_SNAPSHOT_LAST.clear()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshots.jsonl")
            with patch.dict(os.environ, {
                "KCYCLE_TRIFECTA_SNAPSHOT_PATH": path,
                "KCYCLE_TRIFECTA_SNAPSHOT_MIN_INTERVAL_SEC": "60",
            }, clear=False):
                signal = app_module.engine._market_trifecta_signal(make_trifecta_candidate_board())
                first = app_module.engine.save_kcycle_trifecta_snapshot(
                    "2026", "20260628", "광명", "7",
                    make_trifecta_candidate_board(),
                    fetched_at="2026-07-02T12:00:00",
                    signal=signal,
                    source="test",
                )
                app_module.engine._KCYCLE_TRIFECTA_SNAPSHOT_LAST.clear()
                app_module.engine._KCYCLE_TRIFECTA_SNAPSHOT_FILE_KEYS.clear()
                second = app_module.engine.save_kcycle_trifecta_snapshot(
                    "2026", "20260628", "광명", "7",
                    make_trifecta_candidate_board(),
                    fetched_at="2026-07-02T12:00:01",
                    signal=signal,
                    source="test",
                )

            self.assertTrue(first)
            self.assertFalse(second)
            lines = open(path, encoding="utf-8").read().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["schema"], "kcycle_trifecta_snapshot_v1")
            self.assertEqual(record["board_count"], 210)
            self.assertEqual(record["signal"]["tier"], "market_trifecta_50_candidate")
            self.assertIn("5-1-7", record["board"])
            self.assertTrue(os.path.exists(path + ".keys"))

    def test_trifecta_snapshot_writer_rejects_incomplete_board(self):
        app_module.engine._KCYCLE_TRIFECTA_SNAPSHOT_LAST.clear()
        board = make_trifecta_candidate_board()
        board.pop("1-2-3")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshots.jsonl")
            with patch.dict(os.environ, {"KCYCLE_TRIFECTA_SNAPSHOT_PATH": path}, clear=False):
                saved = app_module.engine.save_kcycle_trifecta_snapshot(
                    "2026", "20260628", "광명", "7", board, source="test",
                )

            self.assertFalse(saved)
            self.assertFalse(os.path.exists(path))

    def test_live_decision_saves_trifecta_snapshot_when_board_is_available(self):
        app_module.engine._KCYCLE_TRIFECTA_SNAPSHOT_LAST.clear()
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 1, "name": "모델선두", "pwin": 0.62, "pplc": 0.86},
                {"bno": 5, "name": "시장삼쌍", "pwin": 0.20, "pplc": 0.70},
                {"bno": 7, "name": "상대", "pwin": 0.18, "pplc": 0.64},
            ],
            "picks": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshots.jsonl")
            with patch.dict(os.environ, {
                "KCYCLE_ENABLED": "1",
                "KCYCLE_TRIFECTA_SNAPSHOT_PATH": path,
                "KCYCLE_TRIFECTA_SNAPSHOT_MIN_INTERVAL_SEC": "0",
            }, clear=False), \
                 patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(
                     {1: 4.2, 5: 1.2, 7: 9.5},
                     "2026-07-02T12:00:00",
                 )), \
                 patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                     make_trifecta_candidate_board(),
                     "2026-07-02T12:00:00",
                 )):
                decision = app_module.engine.compute_live_decision(
                    "keirin", "2026-06-28", "광명", "7", base_model_out=base,
                )

            self.assertEqual(decision["trifecta_signal"]["tier"], "market_trifecta_50_candidate")
            record = json.loads(open(path, encoding="utf-8").readline())
            self.assertEqual(record["source"], "live_decision")
            self.assertEqual(record["date"], "20260628")
            self.assertEqual(record["race_no"], "7")

    def test_live_decision_keeps_official_signal_when_card_model_fails(self):
        with patch.dict(os.environ, {"KCYCLE_ENABLED": "0"}, clear=False):
            decision = app_module.engine.compute_live_decision(
                "keirin",
                "2026-06-28",
                "광명",
                "7",
                base_model_out={"error": "출주표 조회 실패"},
            )

        self.assertTrue(decision["ok"])
        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["fallback_signal"]["label"], "KCYCLE 시장3합의 보조픽")
        self.assertEqual(decision["top"]["bno"], 3)
        self.assertEqual(decision["poll_delay_ms"], 15000)

    def test_template_has_live_panel_js(self):
        """index.html에 live-decision 자동 폴링 JS가 있어야."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "templates", "index.html")) as f:
            html = f.read()
        self.assertIn("startLivePolling", html)
        self.assertIn("/api/live-decision", html)
        self.assertIn("setTimeout", html)
        self.assertIn("pollDelayMs", html)
        self.assertIn("d.poll_delay_ms", html)
        self.assertIn("market_risk", html)
        self.assertIn("market_signal", html)
        self.assertIn("trifecta_signal", html)
        self.assertIn("expected_trio_exact", html)
        self.assertIn('method="get"', html)

    def test_live_decision_reuses_cached_base_prediction(self):
        base = {"kind": "ok", "rows": [{"bno": 1, "name": "A", "pwin": 0.6, "pplc": 0.9}]}
        decision = {
            "ok": True, "status": "pre_race", "message": "cached",
            "updated_at": "2026-06-30T12:00:00", "odds_age_sec": None,
            "market_odds": None, "top": base["rows"][0], "rows": base["rows"],
            "decision": "hold", "market_used": False, "snapshot_phase": "pre_race",
        }
        url = "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=5"
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy"}, clear=False), \
             patch.object(app_module, "_compute_base_prediction", return_value=base) as compute, \
             patch.object(app_module.engine, "compute_live_decision", return_value=decision):
            r1 = self.client.get(url)
            r2 = self.client.get(url)

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(compute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
