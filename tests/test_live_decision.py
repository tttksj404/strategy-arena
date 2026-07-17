#!/usr/bin/env python3
"""test_live_decision: /api/live-decision API 테스트."""
import datetime as dt
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import ANY, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KCYCLE_ENABLED"] = "0"
os.environ["KEIRIN_PREWARM_ENABLED"] = "0"

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


def make_trifecta_axis_board(order="1-4-6"):
    board = {}
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) == 3:
                    board[f"{a}-{b}-{c}"] = 10000.0
    board[order] = 10.0
    board["2-3-5"] = 11.0
    return board


def make_trifecta_late_pull_board(order="1-4-6"):
    board = {}
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) == 3:
                    board[f"{a}-{b}-{c}"] = 10000.0
    board[order] = 2.8
    board["2-3-5"] = 3.5
    return board


def make_trifecta_lift_board():
    board = {}
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) == 3:
                    board[f"{a}-{b}-{c}"] = 10000.0
                    if b == 2:
                        board[f"{a}-{b}-{c}"] = 24.0
                    if c == 3:
                        board[f"{a}-{b}-{c}"] = min(board[f"{a}-{b}-{c}"], 28.0)
    board["1-2-3"] = 10.0
    board["4-2-3"] = 18.0
    return board


def make_trifecta_top2_hybrid_board():
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


def make_trifecta_global_rerank_board():
    board = {}
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) == 3:
                    board[f"{a}-{b}-{c}"] = 10000.0
    for combo, odds in {
        "4-6-7": 9.1,
        "4-7-6": 10.0,
        "4-5-6": 10.6,
        "4-6-5": 13.3,
        "4-5-7": 16.8,
        "7-4-6": 17.4,
        "4-7-5": 19.7,
        "4-6-2": 21.2,
        "6-4-7": 24.1,
        "6-4-5": 30.4,
    }.items():
        board[combo] = odds
    return board


class LiveDecisionTestCase(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self._db_tmp = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._db_tmp.name, "strategy.sqlite")
        self._env_patch = patch.dict(os.environ, {
            "DATABASE_URL": f"sqlite:///{db_path}",
            "KCYCLE_TRIFECTA_SNAPSHOT_ENABLED": "0",
        }, clear=False)
        self._env_patch.start()
        self._result_patch = patch.object(app_module.engine, "fetch_kcycle_result_outcome", return_value=None)
        self._result_patch.start()
        self.client = app_module.app.test_client()
        app_module._BASE_PREDICTION_CACHE.clear()
        app_module._NEGATIVE_BASE_PREDICTION_CACHE.clear()
        app_module._LIVE_DECISION_RESULT_CACHE.clear()
        app_module._LIVE_DECISION_PREWARM_STATUS.clear()

    def tearDown(self):
        self._result_patch.stop()
        self._env_patch.stop()
        self._db_tmp.cleanup()

    def test_live_decision_no_date(self):
        """날짜 없으면 400 + hold."""
        r = self.client.get("/api/live-decision")
        self.assertEqual(r.status_code, 400)
        d = r.get_json()
        self.assertFalse(d["ok"])
        self.assertEqual(d["decision"], "hold")

    def test_preload_endpoint_queues_every_race_before_user_selects_one(self):
        with patch.object(app_module, "_enqueue_live_decision_prewarm", return_value={"state": "warming"}) as enqueue:
            response = self.client.post(
                "/api/live-decisions/preload?sport=keirin&date=2026-07-17&meet=광명&race_count=3&priority_race_no=3",
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 202)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["race_nos"], [3, 1, 2])
        self.assertEqual(payload["prewarm"], {"state": "warming"})
        enqueue.assert_called_once_with("keirin", "2026-07-17", "광명", ("3", "1", "2"), ANY)

    def test_preload_status_reports_current_cache_and_task_state(self):
        response = self.client.get(
            "/api/live-decisions/preload?sport=keirin&date=2026-07-17&meet=광명&race_count=3",
        )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["prewarm"]["state"], "idle")
        self.assertIn("pages", payload["prewarm"]["cache"])

    def test_pending_copy_distinguishes_card_fetch_from_analysis(self):
        result = app_module._live_decision_pending_response(
            {}, {}, "2026-07-17", "pending", entry_cards_ready=True,
        )

        self.assertEqual(result["market_risk"]["title"], "분석 결과 계산 중")
        self.assertEqual(result["market_risk"]["level"], "analysis_in_progress")
        self.assertIn("출전표는 준비", result["message"])

    def test_keirin_prewarm_only_fetches_the_shared_official_card_page(self):
        with patch.object(app_module.engine, "prewarm_keirin_card_pages") as prewarm, \
             patch.object(app_module, "_run_live_decision_with_budget") as decision:
            app_module._prewarm_official_entry_cards(
                "keirin", "2026-07-17", "광명", ("3", "1", "2"), "dummy"
            )

        prewarm.assert_called_once_with(2026, "dummy", max_pages=1)
        decision.assert_not_called()

    def test_live_decision_never_uses_demo_racers_without_api_key(self):
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": ""}, clear=False), \
             patch.object(app_module.engine, "_kcycle_rankingpredict_signal", return_value=None):
            r = self.client.get("/api/live-decision?sport=keirin&date=2026-07-03&meet=광명&race_no=1")
        data = r.get_json()

        self.assertEqual(r.status_code, 200)
        self.assertFalse(data["ok"])
        self.assertEqual(data["rows"], [])
        self.assertEqual(data["participants"], [])
        self.assertFalse(data["market_used"])
        self.assertEqual(data["error_kind"], "missing_api_key")
        self.assertIn("DATAGOKR_SERVICE_KEY", data["message"])

    def test_base_prediction_rejects_unsupported_keirin_meet(self):
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy"}, clear=False):
            out = app_module._compute_base_prediction("keirin", "2026-07-03", "창원", "1", "dummy")

        self.assertEqual(out["error_kind"], "unsupported_meet")
        self.assertEqual(out["message"], "지원하지 않는 경주장입니다")

    def test_live_decision_does_not_apply_official_signal_to_unsupported_meet(self):
        with patch.dict(os.environ, {"KCYCLE_ENABLED": "0"}, clear=False):
            decision = app_module.engine.compute_live_decision(
                "keirin",
                "2026-07-03",
                "창원",
                "1",
                base_model_out={"error": "unsupported_meet", "error_kind": "unsupported_meet", "message": "지원하지 않는 경주장입니다"},
            )

        self.assertFalse(decision["ok"])
        self.assertEqual(decision["rows"], [])
        self.assertIsNone(decision["top"])
        self.assertIsNone(decision["fallback_signal"])
        self.assertEqual(decision["error_kind"], "unsupported_meet")

    def test_top2_hybrid_signal_is_separate_from_trifecta_order(self):
        signal = app_module.engine._market_trifecta_axis_signal(make_trifecta_top2_hybrid_board())

        self.assertEqual(signal["order"], [3, 4, 5])
        self.assertEqual(signal["top2_hybrid"]["order_pair"], [2, 1])
        self.assertEqual(signal["top2_hybrid"]["usage"], "top2_pair_signal")

    def test_mobile_picks_use_top2_hybrid_only_for_qnl(self):
        rows = [
            {"bno": bno, "pwin": 0.2 if bno == 3 else 0.1}
            for bno in range(1, 8)
        ]
        signal = app_module.engine._market_trifecta_axis_signal(make_trifecta_top2_hybrid_board())

        picks = app_module.engine._mobile_live_picks(rows, signal)
        by_code = {pick["code"]: pick for pick in picks}

        self.assertEqual(by_code["QNL"]["selection"], "2-1")
        self.assertEqual(by_code["TRI"]["selection"], "3-4-5")

    def test_settled_keirin_result_blocks_final_odds_as_prediction_signal(self):
        base_model = {
            "rows": [
                {"bno": 4, "name": "김로운", "grade": "선발", "pwin": 0.64, "pplc": 0.84},
                {"bno": 1, "name": "박훈재", "grade": "선발", "pwin": 0.34, "pplc": 0.66},
            ],
            "picks": [],
        }
        settled = {
            "actual_order": [1, 2, 4],
            "racers": [{"bno": 1, "name": "박훈재", "rank": 1}],
            "payouts": {"삼쌍승식": {"winner": "1 · 2 · 4", "odds": 35.4}},
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_result_outcome", return_value=settled) as result_fetch, \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", side_effect=AssertionError("final odds leaked into prediction")):
            decision = app_module.engine.compute_live_decision(
                "keirin",
                "2026-07-03",
                "광명",
                "5",
                base_model_out=base_model,
            )

        self.assertEqual(decision["status"], "settled")
        self.assertEqual(decision["decision"], "settled")
        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["poll_delay_ms"], 0)
        self.assertEqual(decision["actual_result"]["actual_order"], [1, 2, 4])
        self.assertEqual(decision["top"]["bno"], 4)
        result_fetch.assert_called_once_with("2026", "2026-07-03", "광명", "5", timeout=0.75, max_attempts=1)

    def test_past_keirin_result_is_settled_even_when_live_market_is_disabled(self):
        base_model = {
            "rows": [
                {"bno": 4, "name": "김로운", "grade": "선발", "pwin": 0.64, "pplc": 0.84},
                {"bno": 1, "name": "박훈재", "grade": "선발", "pwin": 0.34, "pplc": 0.66},
            ],
            "picks": [],
        }
        settled = {
            "actual_order": [1, 2, 4],
            "racers": [{"bno": 1, "name": "박훈재", "rank": 1}],
            "payouts": {"삼쌍승식": {"winner": "1 · 2 · 4", "odds": 35.4}},
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": ""}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_result_outcome", return_value=settled), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", side_effect=AssertionError("live odds should remain disabled")):
            decision = app_module.engine.compute_live_decision(
                "keirin",
                "2026-07-03",
                "광명",
                "5",
                base_model_out=base_model,
            )

        self.assertEqual(decision["status"], "settled")
        self.assertEqual(decision["decision"], "settled")
        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["snapshot_phase"], "settled_result")
        self.assertEqual(decision["actual_result"]["actual_order"], [1, 2, 4])

    def test_today_keirin_result_popup_is_ignored_before_settlement_buffer(self):
        base_model = {
            "rows": [
                {"bno": 1, "name": "모델선두", "grade": "특선", "pwin": 0.52, "pplc": 0.82},
                {"bno": 2, "name": "상대", "grade": "특선", "pwin": 0.30, "pplc": 0.70},
            ],
            "picks": [],
        }
        bogus_result = {
            "actual_order": [7, 6, 5],
            "racers": [{"bno": 7, "name": "미래결과", "rank": 1}],
            "payouts": {"삼쌍승식": {"winner": "7 · 6 · 5", "odds": 77.7}},
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "_kcycle_should_accept_result_outcome", return_value=False), \
             patch.object(app_module.engine, "fetch_kcycle_result_outcome", return_value=bogus_result) as result_fetch, \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(
                 {1: 2.0, 2: 3.0},
                 "2026-07-05T17:44:00",
             )), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(None, "2026-07-05T17:44:00")):
            decision = app_module.engine.compute_live_decision(
                "keirin",
                "2026-07-05",
                "광명",
                "14",
                base_model_out=base_model,
            )

        result_fetch.assert_not_called()
        self.assertNotEqual(decision["status"], "settled")
        self.assertNotIn("actual_result", decision)
        self.assertGreater(len(decision["rows"]), 0)

    def test_base_prediction_never_falls_back_to_horse_demo_on_fetch_error(self):
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy"}, clear=False), \
             patch.object(app_module.engine, "fetch_kra_card", return_value=(None, "API 호출 실패")):
            out = app_module._base_predict_horse("2026-07-03", "부경", "1", {})

        self.assertEqual(out["error_kind"], "upstream_api_error")
        self.assertEqual(out["message"], "상류 API 장애로 출주표를 가져오지 못했습니다")

    def test_base_horse_prediction_passes_verified_odds_snapshot_metadata(self):
        starters = [
            {"chulNo": "1", "hrName": "선두", "winOdds": "2.0"},
            {"chulNo": "2", "hrName": "상대", "winOdds": "4.0"},
        ]
        metadata = {
            "odds_snapshot_fresh": True,
            "odds_snapshot_fetched_at": "2026-07-11T15:55:00+09:00",
            "race_start_at": "2026-07-11T16:05:00+09:00",
        }
        prediction = {"rows": [], "picks": [], "top": None}
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy"}, clear=False), \
             patch.object(app_module.engine, "fetch_kra_card", return_value=(starters, None)), \
             patch.object(app_module.engine, "kra_odds_snapshot_metadata", return_value=metadata), \
             patch.object(app_module, "_verify_roster_or_block", return_value=({"state": "verified"}, None)), \
             patch.object(app_module.engine, "predict_kra", return_value=prediction) as predict:
            out = app_module._base_predict_horse("2026-07-11", "서울", "7", {})

        passed_meta = predict.call_args.kwargs["meta"]
        self.assertTrue(passed_meta["odds_snapshot_fresh"])
        self.assertEqual("2026-07-11T16:05:00+09:00", passed_meta["race_start_at"])
        self.assertEqual("verified", out["roster_verification"]["state"])

    def test_base_prediction_error_kinds_are_distinct(self):
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": ""}, clear=False):
            missing = app_module._compute_base_prediction("keirin", "2026-07-03", "광명", "1", None)
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy"}, clear=False), \
             patch.object(app_module.engine, "fetch_race_card", return_value=(None, "해당 경주를 찾지 못했습니다")):
            no_race = app_module._compute_base_prediction("keirin", "2026-07-03", "광명", "1", "dummy")
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy"}, clear=False), \
             patch.object(app_module.engine, "fetch_race_card", return_value=(None, "API 호출 실패: timeout")):
            upstream = app_module._compute_base_prediction("keirin", "2026-07-03", "광명", "1", "dummy")

        self.assertEqual(missing["error_kind"], "missing_api_key")
        self.assertEqual(no_race["error_kind"], "no_race")
        self.assertEqual(upstream["error_kind"], "upstream_api_error")
        self.assertEqual(no_race["message"], "해당 날짜에는 경주가 없습니다")

    def test_live_decision_slow_provider_returns_hold_within_budget(self):
        def slow_page(stnd_yr, page, rows, key, timeout=8):
            time.sleep(float(timeout) + 0.05)
            raise TimeoutError("provider timeout")

        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy", "KCYCLE_ENABLED": "0"}, clear=False), \
             patch.object(app_module, "_today_kst", return_value=dt.date(2026, 7, 7)), \
             patch.object(app_module.engine, "_api_page_cached", side_effect=slow_page) as page_mock:
            started = time.monotonic()
            response = self.client.get(
                "/api/live-decision?sport=keirin&date=2026-07-07&meet=광명&race_no=1",
                headers={"X-RaceLens-Device-Id": "device-slow-provider"},
            )
            elapsed = time.monotonic() - started

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 4.0)
        self.assertEqual(payload["status"], "hold")
        self.assertEqual(payload["decision"], "hold")
        self.assertEqual(payload["error_kind"], "upstream_api_error")
        self.assertFalse(payload["market_used"])
        self.assertTrue(
            all(call.kwargs.get("timeout", 8) <= 1.5 for call in page_mock.call_args_list),
            page_mock.call_args_list,
        )

    def test_production_live_decision_returns_pending_without_waiting_for_provider(self):
        base = {"kind": "ok", "rows": [{"bno": 1, "name": "A", "pwin": 0.6, "pplc": 0.9}]}
        decision = {
            "ok": True, "status": "ready", "message": "ready",
            "updated_at": "2026-07-07T12:00:00", "odds_age_sec": None,
            "market_odds": [], "top": base["rows"][0], "rows": base["rows"],
            "decision": "hold", "market_used": False, "snapshot_phase": "pre_race",
        }

        def slow_live_decision(*_args, **_kwargs):
            time.sleep(0.5)
            return decision

        previous_testing = app_module.app.config["TESTING"]
        app_module.app.config["TESTING"] = False
        cached = None
        pending_reason = None
        try:
            with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy"}, clear=False), \
                 patch.object(app_module, "_compute_base_prediction_cached", return_value=base), \
                 patch.object(app_module.engine, "compute_live_decision", side_effect=slow_live_decision):
                started = time.monotonic()
                response = self.client.get(
                    "/api/live-decision?sport=keirin&date=2026-07-07&meet=광명&race_no=1",
                    headers={"X-RaceLens-Device-Id": "device-production-pending"},
                )
                elapsed = time.monotonic() - started
                task_key = app_module._live_decision_task_key("keirin", "2026-07-07", "광명", "1")
                app_module._LIVE_DECISION_FUTURES[task_key].result(timeout=1)
                cached, pending_reason = app_module._run_live_decision_with_budget(
                    "keirin", "2026-07-07", "광명", "1", "dummy"
                )
        finally:
            app_module.app.config["TESTING"] = previous_testing

        payload = response.get_json()
        self.assertLess(elapsed, 0.25)
        self.assertEqual(payload["snapshot_phase"], "pending")
        self.assertEqual(payload["poll_delay_ms"], 3000)
        self.assertIsNone(pending_reason)
        self.assertEqual(cached["snapshot_phase"], "pre_race")

    def test_live_decision_past_date_uses_legacy_budget_and_skips_negative_cache(self):
        """과거 날짜는 fast path 제외: 축소 예산 미적용 + negative cache 미기록 (settled 회귀 방지)."""
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy", "KCYCLE_ENABLED": "0"}, clear=False), \
             patch.object(app_module, "_today_kst", return_value=dt.date(2026, 7, 7)), \
             patch.object(
                 app_module.engine,
                 "fetch_race_card",
                 return_value=(None, "해당 경주를 찾지 못했습니다"),
             ) as fetch_mock:
            response = self.client.get(
                "/api/live-decision?sport=keirin&date=2026-07-03&meet=광명&race_no=1",
                headers={"X-RaceLens-Device-Id": "device-past-date"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fetch_mock.call_count, 1)
        call_kwargs = fetch_mock.mock_calls[0].kwargs
        self.assertGreater(call_kwargs.get("timeout", 8), 1.5)
        self.assertGreater(call_kwargs.get("max_pages", 6), 1)
        cache_key = app_module._base_cache_key("keirin", "2026-07-03", "광명", "1")
        self.assertNotIn(cache_key, app_module._NEGATIVE_BASE_PREDICTION_CACHE)

    def test_live_decision_negative_cache_skips_repeated_no_card_fetch(self):
        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy", "KCYCLE_ENABLED": "0"}, clear=False), \
             patch.object(app_module, "_today_kst", return_value=dt.date(2026, 7, 7)), \
             patch.object(
                 app_module.engine,
                 "fetch_race_card",
                 return_value=(None, "해당 경주를 찾지 못했습니다"),
             ) as fetch_mock:
            first = self.client.get(
                "/api/live-decision?sport=keirin&date=2026-07-07&meet=광명&race_no=1",
                headers={"X-RaceLens-Device-Id": "device-negative-cache"},
            )
            started = time.monotonic()
            second = self.client.get(
                "/api/live-decision?sport=keirin&date=2026-07-07&meet=광명&race_no=1",
                headers={"X-RaceLens-Device-Id": "device-negative-cache"},
            )
            elapsed = time.monotonic() - started

        first_payload = first.get_json()
        second_payload = second.get_json()
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(fetch_mock.call_count, 1)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(second_payload["error_kind"], "no_race")
        self.assertIn("출주표 미공개", second_payload["message"])
        self.assertFalse(first_payload["market_used"])
        self.assertFalse(second_payload["market_used"])

    def test_settled_horse_result_attaches_actual_result_without_kcycle_wording(self):
        base_model = {
            "rows": [
                {"bno": 2, "name": "테스트말", "grade": "", "pwin": 0.4, "pplc": 0.8},
                {"bno": 1, "name": "상대말", "grade": "", "pwin": 0.3, "pplc": 0.6},
            ],
            "picks": [],
            "_kra_result": {
                "actual_order": [2, 1, 3],
                "racers": [{"bno": 2, "name": "테스트말", "rank": 1}],
                "payouts": {},
                "source": "kra_race_detail_result",
            },
        }

        decision = app_module.engine.compute_live_decision(
            "horse",
            "2026-07-03",
            "부경",
            "1",
            base_model_out=base_model,
        )

        self.assertEqual(decision["status"], "settled")
        self.assertEqual(decision["snapshot_phase"], "settled_result")
        self.assertEqual(decision["actual_result"]["actual_order"], [2, 1, 3])
        self.assertNotIn("KCYCLE", decision["market_risk"]["message"])

    def test_live_horse_market_anchor_survives_live_decision_projection(self):
        base_model = {
            "rows": [
                {"bno": 1, "name": "시장본선", "grade": "", "pwin": 0.67, "pplc": 0.8},
                {"bno": 2, "name": "상대말", "grade": "", "pwin": 0.33, "pplc": 0.6},
            ],
            "picks": [],
            "market_used": True,
            "pick_source": "market",
            "prediction_phase": "live_odds",
        }

        decision = app_module.engine.compute_live_decision(
            "horse",
            "2026-07-12",
            "서울",
            "1",
            base_model_out=base_model,
        )

        self.assertEqual(decision["status"], "odds_live")
        self.assertTrue(decision["market_used"])
        self.assertEqual(decision["pick_source"], "market")
        self.assertEqual(decision["snapshot_phase"], "live_odds")
        self.assertEqual(decision["market_risk"]["level"], "odds_live")
        self.assertEqual(decision["poll_delay_ms"], 3000)
        self.assertEqual(decision["decision"], "hold")

    def test_market_unused_when_disabled(self):
        """KCYCLE_ENABLED=0이면 market_used=false."""
        r = self.client.get(
            "/api/live-decision?sport=keirin&date=2026-06-28&meet=광명&race_no=1",
            headers={"X-RaceLens-Device-Id": "device-market-disabled"},
        )
        d = r.get_json()
        if d and "market_used" in d:
            self.assertFalse(d["market_used"])
            self.assertEqual(d["poll_delay_ms"], 15000)
            self.assertEqual(d["market_risk"]["level"], "kcycle_disabled")
            self.assertIn("KCYCLE_ENABLED=0", d["market_risk"]["message"])
            self.assertEqual(d["market_odds"], [])

    def test_live_market_fetch_uses_single_short_attempt(self):
        base_model = {
            "kind": "ok",
            "rows": [
                {"bno": 1, "name": "테스트 선수", "pwin": 0.6, "pplc": 0.8},
                {"bno": 2, "name": "상대 선수", "pwin": 0.4, "pplc": 0.7},
            ],
        }
        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "KCYCLE_TRIFECTA_ENABLED", True), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, None)) as odds_mock, \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(None, None)) as trifecta_mock:
            app_module.engine.compute_live_decision(
                "keirin", "2999-01-01", "광명", "1", base_model_out=base_model,
            )

        odds_mock.assert_called_once_with("2999", "2999-01-01", "1", timeout=0.75, max_attempts=1)
        trifecta_mock.assert_called_once_with("2999", "2999-01-01", "1", timeout=0.75, max_attempts=1)

    def test_participant_algorithm_explanations_are_pro_gated(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 4, "name": "이승원", "grade": "선발", "pwin": 0.42, "pplc": 0.74, "pwin_base": 0.39, "learning_starts": 8, "learning_win_delta": 0.06},
                {"bno": 2, "name": "강석호", "grade": "선발", "pwin": 0.34, "pplc": 0.68},
            ],
        }

        free_result = app_module.engine.attach_participant_explanations(
            {"rows": [dict(row) for row in base["rows"]]},
            "keirin",
            pro_enabled=False,
        )
        pro_result = app_module.engine.attach_participant_explanations(
            {"rows": [dict(row) for row in base["rows"]]},
            "keirin",
            pro_enabled=True,
        )

        self.assertTrue(free_result["participants"][0]["algorithm_locked"])
        self.assertEqual(free_result["participants"][0]["algorithm_reasons"], [])
        self.assertFalse(pro_result["participants"][0]["algorithm_locked"])
        self.assertGreaterEqual(len(pro_result["participants"][0]["algorithm_reasons"]), 4)
        self.assertIn("모델 1착", pro_result["participants"][0]["algorithm_note"])
        self.assertTrue(any(item["label"] == "누적학습" for item in pro_result["participants"][0]["algorithm_reasons"]))

    def test_keirin_participant_explanation_reads_racer_card_metrics(self):
        result = {
            "rows": [
                {"bno": 5, "name": "최강우", "grade": "특선", "pwin": 0.62, "pplc": 0.84},
                {"bno": 1, "name": "김태훈", "grade": "우수", "pwin": 0.31, "pplc": 0.66},
            ],
            "_participant_sources": app_module.engine.participant_sources_from_starters([
                {
                    "back_no": "5",
                    "racer_nm": "최강우",
                    "racer_grd_cd": "특선",
                    "racer_age": "32",
                    "trng_plc_nm": "광명",
                    "tot_tms_avg_scr": "91.2",
                    "rec_200m_scr": "11\"21",
                    "gear_rate": "4.00",
                    "high_rate": "68",
                    "pre_win_cnt": "18",
                    "brk_win_cnt": "43",
                    "pas_win_cnt": "27",
                    "mrk_win_cnt": "12",
                    "bf1_day1_rank": "특선5-1",
                    "bf1_day2_rank": "특선5-1",
                    "bf1_day3_rank": "특선5-1",
                },
                {
                    "back_no": "1",
                    "racer_nm": "김태훈",
                    "racer_grd_cd": "우수",
                    "racer_age": "34",
                    "trng_plc_nm": "광명",
                    "tot_tms_avg_scr": "87.4",
                    "rec_200m_scr": "11\"42",
                    "gear_rate": "3.92",
                    "high_rate": "42",
                    "pre_win_cnt": "46",
                    "brk_win_cnt": "22",
                    "pas_win_cnt": "14",
                    "mrk_win_cnt": "18",
                    "bf1_day1_rank": "우수1-1",
                    "bf1_day2_rank": "우수1-2",
                    "bf1_day3_rank": "우수1-4",
                },
            ], "keirin"),
        }

        explained = app_module.engine.attach_participant_explanations(result, "keirin", pro_enabled=True)
        by_number = {item["number"]: item for item in explained["participants"]}

        self.assertIn("평균득점 91.2점", by_number[5]["algorithm_note"])
        self.assertIn("순간 가속 우위", by_number[5]["algorithm_note"])
        self.assertIn("젖히기 43%", by_number[5]["algorithm_note"])
        self.assertTrue(any(item["label"] == "200m" and "11.21초" in item["value"] for item in by_number[5]["algorithm_reasons"]))
        self.assertIn("평균득점 87.4점", by_number[1]["algorithm_note"])
        self.assertIn("선행 46%", by_number[1]["algorithm_note"])

    def test_participant_algorithm_note_rotates_first_sentence_by_number_and_date(self):
        result = {
            "race_date": "2026-07-06",
            "rows": [
                {"bno": 1, "name": "김하나", "grade": "특선", "pwin": 0.62, "pplc": 0.84},
                {"bno": 2, "name": "박두리", "grade": "특선", "pwin": 0.24, "pplc": 0.52},
            ],
        }

        explained = app_module.engine.attach_participant_explanations(result, "keirin", pro_enabled=True)
        notes = [participant["algorithm_note"] for participant in explained["participants"]]

        self.assertTrue(notes[0].startswith("김하나의 가장 특이한 지표는"))
        self.assertTrue(notes[1].startswith("경주 내 편차가 가장 큰 항목은"))
        self.assertNotEqual(notes[0].split(" ", 5)[:5], notes[1].split(" ", 5)[:5])
        self.assertIn("가장", notes[0])
        self.assertIn("모델 1착", notes[0])

    def test_horse_participant_explanation_reads_horse_card_metrics(self):
        result = {
            "rows": [
                {"bno": 5, "name": "골든포커스", "grade": "", "pwin": 0.38, "pplc": 0.67},
            ],
            "_participant_sources": app_module.engine.participant_sources_from_starters([
                {
                    "chulNo": "5",
                    "hrName": "골든포커스",
                    "jkName": "문태오",
                    "wgBudam": "57",
                    "wgHr": "474(-1)",
                    "age": "5",
                    "sex": "거",
                    "rating": "58",
                    "rcDist": "1200",
                    "winOdds": "2.4",
                    "plcOdds": "1.3",
                },
            ], "horse"),
        }

        explained = app_module.engine.attach_participant_explanations(result, "horse", pro_enabled=True)
        note = explained["participants"][0]["algorithm_note"]
        reasons = explained["participants"][0]["algorithm_reasons"]

        self.assertIn("부담중량 57kg", note)
        self.assertIn("마체 -1kg", note)
        self.assertIn("게이트 5번", note)
        self.assertTrue(any(item["label"] == "마체" and "-1kg" in item["value"] for item in reasons))

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

    def test_live_decision_picks_follow_actual_row_order(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 2, "name": "이정민", "pwin": 0.41, "pplc": 0.72},
                {"bno": 4, "name": "김로운", "pwin": 0.34, "pplc": 0.68},
                {"bno": 5, "name": "한기봉", "pwin": 0.21, "pplc": 0.58},
                {"bno": 1, "name": "박훈재", "pwin": 0.11, "pplc": 0.42},
            ],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "0"}, clear=False):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-07-03", "광명", "5", base_model_out=base,
            )

        picks = {pick["code"]: pick for pick in decision["picks"]}
        self.assertEqual(picks["TOP1"]["selection"], "2")
        self.assertEqual(picks["TRI"]["selection"], "2-4-5")
        self.assertNotEqual(picks["TRI"]["selection"], "5-1-7")

    def test_live_decision_keeps_strong_market_favorite_advisory_when_model_disagrees(self):
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
        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["top"]["bno"], 1)
        self.assertEqual(decision["market_signal"]["tier"], "market_fav_odds_le_1_0")
        self.assertFalse(decision["market_signal"]["applied"])
        self.assertTrue(decision["market_signal"]["blocked_by_order_conflict"])
        self.assertAlmostEqual(decision["market_signal"]["expected_top1"], 0.8896)
        self.assertEqual(decision["poll_delay_ms"], 3000)
        self.assertEqual(decision["market_risk"]["level"], "odds_live")
        self.assertIsInstance(decision["market_odds"], list)
        self.assertEqual(decision["market_odds"][0]["label"], "단승")
        self.assertEqual(decision["market_odds"][0]["selection"], "5")
        self.assertEqual(decision["market_odds"][0]["odds"], 1.0)

    def test_live_decision_does_not_overweight_ambiguous_market_odds(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 1, "name": "모델선두", "pwin": 0.62, "pplc": 0.86},
                {"bno": 5, "name": "배당관심", "pwin": 0.20, "pplc": 0.70},
                {"bno": 7, "name": "상대", "pwin": 0.18, "pplc": 0.64},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(
                 {1: 4.2, 5: 2.4, 7: 9.5},
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
        self.assertIsNone(decision["market_signal"])
        self.assertEqual(decision["top"]["bno"], 1)
        self.assertIn("저의존 배당 블렌드", decision["message"])
        self.assertEqual(decision["market_timing"]["policy"], "low_odds_dependency_blend_w0.30")

    def test_market_timing_uses_robust_low_odds_dependency_blend_policy(self):
        app_module.engine._KCYCLE_LOW_ODDS_BLEND_CACHE.clear()
        policy_payload = {
            "years": {
                "2025": {
                    "test": [
                        {
                            "name": "blend_w0.70",
                            "top1": 0.6145,
                            "market_flip_rate": 0.112,
                            "rule": "(1-w)*model+w*market, w=0.70",
                        },
                        {
                            "name": "blend_w0.30",
                            "top1": 0.6217,
                            "market_flip_rate": 0.050,
                            "rule": "(1-w)*model+w*market, w=0.30",
                        },
                    ],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "market_blend.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(policy_payload, f)
            with patch.dict(os.environ, {
                "KCYCLE_MARKET_BLEND_RESULTS_PATH": path,
                "KCYCLE_RACE_START_1_TIME": "11:00",
                "KCYCLE_RACE_INTERVAL_MIN": "23",
            }, clear=False):
                timing = app_module.engine._kcycle_market_timing_policy(
                    "2026-07-04",
                    "5",
                    "2026-07-04T12:27:00",
                )

        self.assertEqual(timing["phase"], "late")
        self.assertAlmostEqual(timing["market_weight"], 0.30)
        self.assertEqual(timing["policy"], "low_odds_dependency_blend_w0.30")
        self.assertAlmostEqual(timing["policy_top1"], 0.6217)
        self.assertAlmostEqual(timing["policy_flip_rate"], 0.050)

    def test_trifecta_signal_exposes_immediate_prior_lift_with_robust_warning(self):
        signal = app_module.engine._market_trifecta_signal(make_trifecta_candidate_board())

        self.assertEqual(signal["tier"], "market_trifecta_watch_low_sample")
        self.assertEqual(signal["order"], [5, 1, 7])
        self.assertFalse(signal["deployable"])
        self.assertEqual(signal["usage"], "research_watch_only")
        self.assertIsNone(signal["expected_trio_exact"])
        self.assertAlmostEqual(signal["observed_trio_exact"], 0.5)
        self.assertAlmostEqual(signal["baseline_trio_exact"], 0.2719)
        self.assertAlmostEqual(signal["lift_pp"], 22.81)
        self.assertEqual(signal["robust_status"], "failed_small_n")
        self.assertIn("배포 금지", signal["robust_warning"])

    def test_trifecta_axis_signal_uses_full_board_first_number(self):
        signal = app_module.engine._market_trifecta_axis_signal(make_trifecta_axis_board("1-4-6"))

        self.assertEqual(signal["tier"], "market_trifecta_axis")
        self.assertEqual(signal["leader"], 1)
        self.assertEqual(signal["order"], [1, 4, 6])
        self.assertAlmostEqual(signal["expected_top1"], 0.6370721789223992)
        self.assertAlmostEqual(signal["expected_pair_board"], 0.33310742121314807)
        self.assertAlmostEqual(signal["expected_trio_exact"], 0.16062351745171127)
        self.assertTrue(signal["deployable"])
        self.assertEqual(signal["usage"], "prediction_signal")
        self.assertEqual(signal["robust_status"], "passed_oos")

    def test_trifecta_axis_signal_promotes_late_market_pull_when_board_is_extreme(self):
        signal = app_module.engine._market_trifecta_axis_signal(make_trifecta_late_pull_board("1-4-6"))

        self.assertEqual(signal["tier"], "market_trifecta_late_pull_strong")
        self.assertEqual(signal["leader"], 1)
        self.assertEqual(signal["order"], [1, 4, 6])
        self.assertAlmostEqual(signal["expected_top1"], 0.8347107438016529)
        self.assertAlmostEqual(signal["expected_pair_board"], 0.5702479338842975)
        self.assertAlmostEqual(signal["expected_trio_exact"], 0.3347107438016529)
        self.assertTrue(signal["deployable"])
        self.assertEqual(signal["robust_status"], "passed_late_market_pull_oos")

    def test_market_timing_phase_uses_configurable_race_clock(self):
        with patch.dict(os.environ, {
            "KCYCLE_RACE_START_1_TIME": "11:00",
            "KCYCLE_RACE_INTERVAL_MIN": "23",
        }, clear=False):
            timing = app_module.engine._kcycle_market_timing_policy(
                "2026-07-04",
                "5",
                "2026-07-04T12:20:00",
            )

        self.assertEqual(timing["phase"], "mid")
        self.assertAlmostEqual(timing["minutes_to_start"], 12.0)
        self.assertAlmostEqual(timing["market_weight"], 0.15)
        self.assertFalse(timing["allow_late_pull"])
        self.assertFalse(timing["allow_trifecta_axis"])

    def test_market_timing_phase_prefers_official_card_clock(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KCYCLE_RACE_START_1_TIME", None)
            os.environ.pop("KCYCLE_RACE_INTERVAL_MIN", None)
            with patch.object(
                app_module.engine,
                "_kcycle_official_race_start",
                return_value=app_module.engine.datetime(2026, 7, 4, 12, 32),
            ):
                timing = app_module.engine._kcycle_market_timing_policy(
                    "2026-07-04",
                    "5",
                    "2026-07-04T12:27:00",
                )

        self.assertEqual(timing["phase"], "late")
        self.assertEqual(timing["race_start_at"], "2026-07-04T12:32:00")
        self.assertTrue(timing["allow_late_pull"])
        self.assertTrue(timing["allow_trifecta_axis"])

    def test_early_market_does_not_override_model_with_late_pull_proxy(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 4, "name": "모델선두", "pwin": 0.91, "pplc": 0.94},
                {"bno": 1, "name": "초반저배당", "pwin": 0.34, "pplc": 0.70},
                {"bno": 6, "name": "삼쌍3번", "pwin": 0.18, "pplc": 0.62},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {
            "KCYCLE_ENABLED": "1",
            "KCYCLE_RACE_START_1_TIME": "11:00",
            "KCYCLE_RACE_INTERVAL_MIN": "23",
        }, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-04T11:30:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_late_pull_board("1-4-6"),
                 "2026-07-04T11:30:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-07-04", "광명", "5", base_model_out=base,
            )

        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["top"]["bno"], 4)
        self.assertEqual(decision["market_timing"]["phase"], "early")
        self.assertEqual(decision["trifecta_axis_signal"]["tier"], "market_trifecta_late_pull_strong")
        self.assertFalse(decision["trifecta_axis_signal"]["applied"])
        self.assertIn("초반 배당 관찰", decision["message"])

    def test_late_market_pull_stays_advisory_when_it_conflicts_with_model(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 4, "name": "모델선두", "pwin": 0.91, "pplc": 0.94},
                {"bno": 1, "name": "마감강축", "pwin": 0.34, "pplc": 0.70},
                {"bno": 6, "name": "삼쌍3번", "pwin": 0.18, "pplc": 0.62},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {
            "KCYCLE_ENABLED": "1",
            "KCYCLE_RACE_START_1_TIME": "11:00",
            "KCYCLE_RACE_INTERVAL_MIN": "23",
        }, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-04T12:27:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_late_pull_board("1-4-6"),
                 "2026-07-04T12:27:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-07-04", "광명", "5", base_model_out=base,
            )

        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["top"]["bno"], 4)
        self.assertEqual(decision["market_timing"]["phase"], "late")
        self.assertFalse(decision["trifecta_axis_signal"]["applied"])
        self.assertTrue(decision["trifecta_axis_signal"]["blocked_by_order_conflict"])
        self.assertAlmostEqual(decision["top"]["pwin"], 0.91)

    def test_trifecta_lift_signal_exposes_stat_strict_rerank_candidate(self):
        signal = app_module.engine._market_trifecta_lift_signal(make_trifecta_lift_board())

        self.assertEqual(signal["tier"], "market_trifecta_stat_strict_lift")
        self.assertTrue(signal["deployable"])
        self.assertEqual(signal["usage"], "selective_trifecta_rerank_signal")
        self.assertEqual(len(signal["order"]), 3)
        self.assertGreaterEqual(signal["second_mass_best"], 0.380599)
        self.assertGreaterEqual(signal["third_mass_best"], 0.263285)
        self.assertAlmostEqual(signal["expected_trio_exact"], 0.26090342679127726)
        self.assertAlmostEqual(signal["current_axis_trio_exact"], 0.16062351745171127)
        self.assertAlmostEqual(signal["current_axis_lift_pp"], 10.0279909339566)
        self.assertAlmostEqual(signal["baseline_trio_exact"], 0.2554517133956386)
        self.assertAlmostEqual(signal["lift_pp"], 0.5451713395638658)
        self.assertIn("current full-board-axis 16.06% 대비 +10.03%p", signal["validation_split"])
        self.assertEqual(signal["robust_status"], "passed_directional_lift_oos")

    def test_trifecta_global_rerank_signal_exposes_incremental_oos_candidate(self):
        missing_path = os.path.join(self._db_tmp.name, "missing_global_champion.json")
        app_module.engine._KCYCLE_GLOBAL_RERANK_CACHE.clear()
        with patch.dict(os.environ, {"KCYCLE_GLOBAL_RERANK_RESULTS_PATH": missing_path}, clear=False):
            signal = app_module.engine._market_trifecta_global_rerank_signal(make_trifecta_global_rerank_board())

        self.assertEqual(signal["tier"], "market_trifecta_global_incremental_rerank")
        self.assertTrue(signal["deployable"])
        self.assertEqual(signal["usage"], "selective_trifecta_rerank_signal")
        self.assertEqual(signal["order"], [4, 7, 6])
        self.assertEqual(signal["favorite_odds"], 9.1)
        self.assertEqual(signal["selected_odds"], 10.0)
        self.assertEqual(signal["selected_board_rank"], 2)
        self.assertAlmostEqual(signal["expected_trio_exact"], 0.18063583970069885)
        self.assertAlmostEqual(signal["baseline_trio_exact"], 0.17196531791907516)
        self.assertAlmostEqual(signal["lift_pp"], 0.8670521781623697)
        self.assertIn("10pp breakthrough 아님", signal["validation_split"])
        self.assertEqual(signal["robust_status"], "passed_incremental_oos")

    def test_trifecta_global_rerank_loads_deployable_champion_from_search_json(self):
        path = os.path.join(self._db_tmp.name, "global_champion.json")
        payload = {
            "feature_stats_by_top_k": {
                "10": {
                    "mu": app_module.engine._GLOBAL_RERANK_MU,
                    "sigma": app_module.engine._GLOBAL_RERANK_SIGMA,
                },
            },
            "candidates": [
                {
                    "name": "dynamic_probe_candidate",
                    "top_k": 10,
                    "deployable": True,
                    "weights": app_module.engine._GLOBAL_RERANK_WEIGHTS,
                    "test_exact": 0.199,
                    "test_board_exact": 0.171,
                    "test_board_lift_pp": 2.8,
                    "test_current_axis_lift_pp": 3.9,
                    "formula": "probe formula",
                },
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        app_module.engine._KCYCLE_GLOBAL_RERANK_CACHE.clear()

        with patch.dict(os.environ, {"KCYCLE_GLOBAL_RERANK_RESULTS_PATH": path}, clear=False):
            signal = app_module.engine._market_trifecta_global_rerank_signal(make_trifecta_global_rerank_board())

        self.assertEqual(signal["order"], [4, 7, 6])
        self.assertAlmostEqual(signal["expected_trio_exact"], 0.199)
        self.assertAlmostEqual(signal["baseline_trio_exact"], 0.171)
        self.assertAlmostEqual(signal["lift_pp"], 2.8)
        self.assertIn("dynamic_probe_candidate", signal["rule"])
        self.assertIn("probe formula", signal["rule"])

    def test_live_decision_uses_trifecta_axis_when_full_board_is_available(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 4, "name": "모델선두", "pwin": 0.91, "pplc": 0.94},
                {"bno": 1, "name": "삼쌍축", "pwin": 0.34, "pplc": 0.70},
                {"bno": 6, "name": "삼쌍3번", "pwin": 0.18, "pplc": 0.62},
                {"bno": 7, "name": "상대", "pwin": 0.12, "pplc": 0.48},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1", "KEIRIN_PICK_POLICY": "model_always"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-02T12:00:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_axis_board("1-4-6"),
                 "2026-07-02T12:00:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["top"]["bno"], 4)
        self.assertAlmostEqual(decision["top"]["pwin"], 0.91)
        self.assertEqual(decision["trifecta_axis_signal"]["tier"], "market_trifecta_axis")
        self.assertFalse(decision["trifecta_axis_signal"]["applied"])
        self.assertTrue(decision["trifecta_axis_signal"]["blocked_by_order_conflict"])
        self.assertEqual(decision["picks"][0]["selection"], "4")
        self.assertEqual(decision["picks"][2]["selection"], "4-1-6")

    def test_live_decision_uses_late_market_pull_confidence_when_board_is_extreme(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 4, "name": "모델선두", "pwin": 0.91, "pplc": 0.94},
                {"bno": 1, "name": "마감강축", "pwin": 0.34, "pplc": 0.70},
                {"bno": 6, "name": "삼쌍3번", "pwin": 0.18, "pplc": 0.62},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-02T12:00:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_late_pull_board("1-4-6"),
                 "2026-07-02T12:00:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["trifecta_axis_signal"]["tier"], "market_trifecta_late_pull_strong")
        self.assertFalse(decision["trifecta_axis_signal"]["applied"])
        self.assertTrue(decision["trifecta_axis_signal"]["blocked_by_order_conflict"])
        self.assertEqual(decision["top"]["bno"], 4)
        self.assertAlmostEqual(decision["top"]["pwin"], 0.91)
        self.assertEqual(decision["picks"][2]["selection"], "4-1-6")

    def test_live_decision_exposes_trifecta_lift_signal_when_full_board_matches(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 1, "name": "축", "pwin": 0.43, "pplc": 0.70},
                {"bno": 2, "name": "상대1", "pwin": 0.31, "pplc": 0.62},
                {"bno": 3, "name": "상대2", "pwin": 0.26, "pplc": 0.58},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-02T12:00:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_lift_board(),
                 "2026-07-02T12:00:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertEqual(decision["trifecta_lift_signal"]["tier"], "market_trifecta_stat_strict_lift")
        self.assertTrue(decision["trifecta_lift_signal"]["deployable"])
        self.assertTrue(decision["trifecta_lift_signal"]["applied"])
        self.assertAlmostEqual(decision["trifecta_lift_signal"]["current_axis_lift_pp"], 10.0279909339566)
        self.assertGreaterEqual(decision["trifecta_lift_signal"]["second_mass_best"], 0.380599)
        self.assertEqual(decision["picks"][2]["selection"], "-".join(str(x) for x in decision["trifecta_lift_signal"]["order"]))
        self.assertEqual(decision["picks"][2]["basis"], "market_trifecta_stat_strict_lift")
        self.assertIn("삼쌍 순서 재랭킹 반영", decision["message"])

    def test_live_decision_applies_global_rerank_when_stat_lift_is_absent(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 4, "name": "축", "pwin": 0.43, "pplc": 0.70},
                {"bno": 6, "name": "상대1", "pwin": 0.31, "pplc": 0.62},
                {"bno": 7, "name": "상대2", "pwin": 0.26, "pplc": 0.58},
            ],
            "picks": [],
        }

        missing_path = os.path.join(self._db_tmp.name, "missing_global_champion.json")
        app_module.engine._KCYCLE_GLOBAL_RERANK_CACHE.clear()
        with patch.dict(os.environ, {
            "KCYCLE_ENABLED": "1",
            "KCYCLE_GLOBAL_RERANK_RESULTS_PATH": missing_path,
        }, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-02T12:00:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_global_rerank_board(),
                 "2026-07-02T12:00:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertIsNone(decision["trifecta_lift_signal"])
        self.assertEqual(decision["trifecta_global_signal"]["tier"], "market_trifecta_global_incremental_rerank")
        self.assertTrue(decision["trifecta_global_signal"]["applied"])
        self.assertAlmostEqual(decision["trifecta_global_signal"]["lift_pp"], 0.8670521781623697)
        self.assertEqual(decision["picks"][2]["selection"], "4-7-6")
        self.assertEqual(decision["picks"][2]["basis"], "market_trifecta_global_incremental_rerank")
        self.assertIn("삼쌍 전체보드 재랭킹 반영", decision["message"])

    def test_global_rerank_signal_is_blocked_when_first_pick_conflicts(self):
        conflict_global_signal = dict(app_module.engine._market_trifecta_global_rerank_signal(make_trifecta_global_rerank_board()))
        conflict_global_signal["leader"] = 7
        conflict_global_signal["order"] = [7, 4, 6]
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 4, "name": "축", "pwin": 0.43, "pplc": 0.70},
                {"bno": 6, "name": "상대1", "pwin": 0.31, "pplc": 0.62},
                {"bno": 7, "name": "상대2", "pwin": 0.26, "pplc": 0.58},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-02T12:00:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_global_rerank_board(),
                 "2026-07-02T12:00:00",
             )), \
             patch.object(app_module.engine, "_market_trifecta_global_rerank_signal", return_value=conflict_global_signal):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertFalse(decision["trifecta_global_signal"]["applied"])
        self.assertTrue(decision["trifecta_global_signal"]["blocked_by_order_conflict"])
        self.assertEqual(decision["picks"][2]["selection"], "4-6-7")
        self.assertIn("1착 후보 충돌", decision["message"])

    def test_trifecta_lift_signal_is_blocked_when_first_pick_conflicts(self):
        conflict_lift_signal = dict(app_module.engine._market_trifecta_lift_signal(make_trifecta_lift_board()))
        conflict_lift_signal["leader"] = 4
        conflict_lift_signal["order"] = [4, 2, 3]
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 3, "name": "모델축", "pwin": 0.43, "pplc": 0.70},
                {"bno": 2, "name": "상대1", "pwin": 0.31, "pplc": 0.62},
                {"bno": 1, "name": "상대2", "pwin": 0.26, "pplc": 0.58},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-02T12:00:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_lift_board(),
                 "2026-07-02T12:00:00",
             )), \
             patch.object(app_module.engine, "_market_trifecta_lift_signal", return_value=conflict_lift_signal):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertFalse(decision["trifecta_lift_signal"]["applied"])
        self.assertTrue(decision["trifecta_lift_signal"]["blocked_by_order_conflict"])
        self.assertEqual(decision["picks"][2]["selection"], "3-2-1")
        self.assertIn("1착 후보 충돌", decision["message"])

    def test_trifecta_lift_signal_is_blocked_before_late_timing(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 3, "name": "모델축", "pwin": 0.43, "pplc": 0.70},
                {"bno": 2, "name": "상대1", "pwin": 0.31, "pplc": 0.62},
                {"bno": 1, "name": "상대2", "pwin": 0.26, "pplc": 0.58},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {
            "KCYCLE_ENABLED": "1",
            "KCYCLE_RACE_START_1_TIME": "11:00",
            "KCYCLE_RACE_INTERVAL_MIN": "23",
        }, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-04T12:05:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_lift_board(),
                 "2026-07-04T12:05:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-07-04", "광명", "5", base_model_out=base,
            )

        self.assertEqual(decision["market_timing"]["phase"], "mid")
        self.assertFalse(decision["trifecta_lift_signal"]["applied"])
        self.assertTrue(decision["trifecta_lift_signal"]["blocked_by_timing"])
        self.assertEqual(decision["picks"][2]["selection"], "3-2-1")
        self.assertIn("삼쌍 순서 재랭킹 관찰 중", decision["message"])

    def test_strong_win_market_keeps_priority_over_conflicting_trifecta_axis(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 1, "name": "단승강축", "pwin": 0.32, "pplc": 0.74},
                {"bno": 4, "name": "삼쌍축", "pwin": 0.31, "pplc": 0.72},
                {"bno": 6, "name": "상대", "pwin": 0.18, "pplc": 0.62},
                {"bno": 7, "name": "상대2", "pwin": 0.12, "pplc": 0.48},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1", "KEIRIN_PICK_POLICY": "model_always"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(
                 {1: 1.0, 4: 2.2, 6: 8.0, 7: 15.0},
                 "2026-07-02T12:00:00",
             )), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_axis_board("4-1-6"),
                 "2026-07-02T12:00:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertEqual(decision["market_signal"]["tier"], "market_fav_odds_le_1_0")
        self.assertEqual(decision["top"]["bno"], 1)
        self.assertNotIn("trifecta_axis_pwin", decision["top"])
        self.assertEqual(decision["picks"][0]["selection"], "1")

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
        self.assertEqual(decision["trifecta_signal"]["tier"], "market_trifecta_watch_low_sample")
        self.assertFalse(decision["trifecta_signal"]["deployable"])
        self.assertIsNone(decision["trifecta_signal"]["expected_trio_exact"])
        self.assertIn("50% 배포 금지", decision["message"])
        self.assertEqual(decision["poll_delay_ms"], 3000)
        self.assertTrue(any(item["code"] == "TRI" and item["selection"] == "5-1-7" for item in decision["market_odds"]))

    def test_trifecta_watch_signal_does_not_count_as_live_market_when_not_applied(self):
        base = {
            "kind": "ok",
            "rows": [
                {"bno": 1, "name": "모델선두", "pwin": 0.62, "pplc": 0.86},
                {"bno": 7, "name": "상대", "pwin": 0.18, "pplc": 0.64},
                {"bno": 2, "name": "상대2", "pwin": 0.12, "pplc": 0.54},
            ],
            "picks": [],
        }

        with patch.dict(os.environ, {"KCYCLE_ENABLED": "1"}, clear=False), \
             patch.object(app_module.engine, "fetch_kcycle_result_outcome", return_value=None), \
             patch.object(app_module.engine, "fetch_kcycle_odds_with_ts", return_value=(None, "2026-07-02T12:00:00")), \
             patch.object(app_module.engine, "fetch_kcycle_trifecta_board_with_ts", return_value=(
                 make_trifecta_candidate_board(),
                 "2026-07-02T12:00:00",
             )):
            decision = app_module.engine.compute_live_decision(
                "keirin", "2026-06-28", "광명", "7", base_model_out=base,
            )

        self.assertFalse(decision["market_used"])
        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["status"], "odds_unavailable")
        self.assertEqual(decision["poll_delay_ms"], 15000)
        self.assertEqual(decision["market_risk"]["level"], "odds_unavailable")
        self.assertFalse(decision["trifecta_signal"]["deployable"])
        self.assertIn("50% 배포 금지", decision["message"])

    def test_trifecta_snapshot_writer_appends_and_dedupes(self):
        app_module.engine._KCYCLE_TRIFECTA_SNAPSHOT_LAST.clear()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshots.jsonl")
            with patch.dict(os.environ, {
                "KCYCLE_TRIFECTA_SNAPSHOT_ENABLED": "1",
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
            self.assertEqual(record["snapshot_phase"], "pre_result_market_snapshot")
            self.assertEqual(record["board_count"], 210)
            self.assertEqual(record["signal"]["tier"], "market_trifecta_watch_low_sample")
            self.assertFalse(record["signal"]["deployable"])
            self.assertIn("5-1-7", record["board"])
            self.assertTrue(os.path.exists(path + ".keys"))

    def test_trifecta_snapshot_writer_rejects_incomplete_board(self):
        app_module.engine._KCYCLE_TRIFECTA_SNAPSHOT_LAST.clear()
        board = make_trifecta_candidate_board()
        board.pop("1-2-3")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshots.jsonl")
            with patch.dict(os.environ, {
                "KCYCLE_TRIFECTA_SNAPSHOT_ENABLED": "1",
                "KCYCLE_TRIFECTA_SNAPSHOT_PATH": path,
            }, clear=False):
                saved = app_module.engine.save_kcycle_trifecta_snapshot(
                    "2026", "20260628", "광명", "7", board, source="test",
                )

            self.assertFalse(saved)
            self.assertFalse(os.path.exists(path))

    def test_trifecta_snapshot_writer_records_market_timing_and_signal_bundle(self):
        app_module.engine._KCYCLE_TRIFECTA_SNAPSHOT_LAST.clear()
        board = make_trifecta_candidate_board()
        signal = app_module.engine._market_trifecta_signal(board)
        timing = {
            "phase": "late",
            "minutes_to_start": 4.2,
            "market_weight": 0.30,
            "allow_late_pull": True,
            "allow_trifecta_axis": True,
            "race_start_at": "2026-06-28T17:10:00",
            "ignored_field": "drop-me",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshots.jsonl")
            with patch.dict(os.environ, {
                "KCYCLE_TRIFECTA_SNAPSHOT_ENABLED": "1",
                "KCYCLE_TRIFECTA_SNAPSHOT_PATH": path,
                "KCYCLE_TRIFECTA_SNAPSHOT_MIN_INTERVAL_SEC": "0",
            }, clear=False):
                saved = app_module.engine.save_kcycle_trifecta_snapshot(
                    "2026", "20260628", "광명", "7",
                    board,
                    fetched_at="2026-06-28T17:05:48",
                    signal=signal,
                    source="test",
                    market_timing=timing,
                    signals={"watch": signal, "axis": None},
                )

            self.assertTrue(saved)
            record = json.loads(open(path, encoding="utf-8").readline())
            self.assertEqual(record["schema"], "kcycle_trifecta_snapshot_v2")
            self.assertEqual(record["market_timing"]["phase"], "late")
            self.assertNotIn("ignored_field", record["market_timing"])
            self.assertEqual(record["signals"]["watch"]["tier"], "market_trifecta_watch_low_sample")
            self.assertIsNone(record["signals"]["axis"])

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
                "KCYCLE_TRIFECTA_SNAPSHOT_ENABLED": "1",
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

            self.assertEqual(decision["trifecta_signal"]["tier"], "market_trifecta_watch_low_sample")
            record = json.loads(open(path, encoding="utf-8").readline())
            self.assertEqual(record["schema"], "kcycle_trifecta_snapshot_v2")
            self.assertEqual(record["source"], "live_decision")
            self.assertEqual(record["date"], "20260628")
            self.assertEqual(record["race_no"], "7")
            self.assertEqual(record["market_timing"]["phase"], "unknown")
            self.assertEqual(record["signals"]["watch"]["tier"], "market_trifecta_watch_low_sample")
            self.assertIn("axis", record["signals"])
            self.assertIn("lift", record["signals"])

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
                               "templates", "index.html"), encoding="utf-8") as f:
            html = f.read()
        self.assertIn("startLivePolling", html)
        self.assertIn("/api/live-decision", html)
        self.assertIn("setTimeout", html)
        self.assertIn("pollDelayMs", html)
        self.assertIn("d.poll_delay_ms", html)
        self.assertIn("market_risk", html)
        self.assertIn("market_signal", html)
        self.assertIn("trifecta_signal", html)
        self.assertIn("trifecta_lift_signal", html)
        self.assertIn("삼쌍 개선 신호", html)
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
