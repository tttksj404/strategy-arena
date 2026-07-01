import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
import engine


class CrossDomainModelTestCase(unittest.TestCase):
    def test_healthz_reports_cross_domain_model(self):
        app_module.app.config["TESTING"] = True
        client = app_module.app.test_client()

        response = client.get("/healthz")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["keirin_cross_model"], "present")
        self.assertIsNone(data["keirin_cross_err"])

    def test_deep_healthz_loads_cross_domain_model(self):
        app_module.app.config["TESTING"] = True
        client = app_module.app.test_client()

        response = client.get("/healthz?deep=1")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["keirin_cross_model"], "loaded")
        self.assertIsNone(data["keirin_cross_err"])

    def test_demo_keirin_uses_cross_domain_fallback(self):
        demo = engine.load_demo_race()

        output = engine.predict(
            demo["items"],
            meta={"ymd": demo.get("race_ymd"), "meet": "광명", "race_no": "5"},
        )

        self.assertNotIn("error", output)
        self.assertTrue(output.get("model_cross_domain"))
        self.assertEqual(len(output["rows"]), 7)

    def test_keirin_top_pick_matches_win_probability_leader(self):
        demo = engine.load_demo_race()

        output = engine.predict(
            demo["items"],
            meta={"ymd": "1900.01.01", "meet": "광명", "race_no": "5"},
        )
        win_leader = max(output["rows"], key=lambda row: row["pwin"])

        self.assertEqual(output["top"]["bno"], win_leader["bno"])

    def test_keirin_predict_without_meta_does_not_crash(self):
        demo = engine.load_demo_race()

        output = engine.predict(demo["items"])

        self.assertNotIn("error", output)
        self.assertEqual(len(output["rows"]), 7)

    def test_keirin_selective_confidence_extreme_tier(self):
        tier = engine._keirin_selective_confidence({"pwin": 0.55, "pplc": 0.91})

        self.assertEqual(tier["tier"], "extreme")
        self.assertAlmostEqual(tier["expected_top1"], 0.8175)

    def test_keirin_selective_confidence_ultra_tier_uses_gap_first(self):
        rows = [
            {"pwin": 0.82, "pplc": 0.88},
            {"pwin": 0.18, "pplc": 0.61},
        ]

        tier = engine._keirin_selective_confidence(rows[0], rows)

        self.assertEqual(tier["tier"], "ultra")
        self.assertAlmostEqual(tier["expected_top1"], 0.8467)

    def test_keirin_selective_confidence_fixed_ultra_86_tier(self):
        rows = [
            {"pwin": 0.80, "pplc": 0.93},
            {"pwin": 0.30, "pplc": 0.70},
        ]

        tier = engine._keirin_selective_confidence(rows[0], rows)

        self.assertEqual(tier["tier"], "ultra_fixed_86")
        self.assertAlmostEqual(tier["expected_top1"], 0.8593)
        self.assertEqual(tier["validation_n"], 2111)

    def test_keirin_selective_confidence_gap_extends_extreme_coverage(self):
        rows = [
            {"pwin": 0.75, "pplc": 0.86},
            {"pwin": 0.18, "pplc": 0.60},
        ]

        tier = engine._keirin_selective_confidence(rows[0], rows)

        self.assertEqual(tier["tier"], "extreme_gap")
        self.assertAlmostEqual(tier["coverage"], 0.3029)

    def test_keirin_selective_confidence_broad_tier(self):
        tier = engine._keirin_selective_confidence({"pwin": 0.61, "pplc": 0.80})

        self.assertEqual(tier["tier"], "broad")
        self.assertAlmostEqual(tier["expected_top1"], 0.7287)

    def test_keirin_selective_confidence_uses_win_leader_for_pwin_gate(self):
        rows = [
            {"pwin": 0.34, "pplc": 0.88},
            {"pwin": 0.61, "pplc": 0.78},
            {"pwin": 0.22, "pplc": 0.60},
        ]

        tier = engine._keirin_selective_confidence(rows[0], rows)

        self.assertEqual(tier["tier"], "broad")

    def test_ordered_exotic_picks_are_not_graded_like_single_win(self):
        rows = [
            {"bno": 1, "name": "A", "pwin": 0.80, "pplc": 0.94},
            {"bno": 2, "name": "B", "pwin": 0.12, "pplc": 0.62},
            {"bno": 3, "name": "C", "pwin": 0.08, "pplc": 0.55},
        ]

        picks = {pick["code"]: pick for pick in engine.build_picks(rows)}

        self.assertEqual(picks["단승"]["grade"], "강")
        self.assertEqual(picks["쌍승"]["grade"], "약")
        self.assertEqual(picks["삼쌍"]["grade"], "약")
        self.assertIn("순서권 리스크", picks["삼쌍"]["prob"])

    def test_kcycle_rankingpredict_cache_signal_uses_official_consensus(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        signal = engine._kcycle_rankingpredict_signal(
            {"ymd": "2024.12.08", "meet": "광명", "race_no": "16"},
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal["tier"], "kcycle_all_first_agree")
        self.assertEqual(signal["leader"], 7)
        self.assertAlmostEqual(signal["expected_top1"], 0.8649)

    def test_kcycle_saturday_market_consensus_promotes_extreme_signal(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        signal = engine._kcycle_rankingpredict_signal(
            {"ymd": "2026.06.27", "meet": "광명", "race_no": "14"},
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal["tier"], "kcycle_market3_day2_extreme")
        self.assertEqual(signal["leader"], 7)
        self.assertAlmostEqual(signal["expected_top1"], 0.9111)

    def test_kcycle_rankingpredict_support_signal_handles_non_high_confidence(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        signal = engine._kcycle_rankingpredict_signal(
            {"ymd": "2026.06.28", "meet": "광명", "race_no": "7"},
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal["tier"], "kcycle_market3_support")
        self.assertEqual(signal["leader"], 3)
        self.assertAlmostEqual(signal["expected_top1"], 0.6656)

    def test_kcycle_rankingpredict_overlay_promotes_official_leader(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        rows = [
            {"bno": 1, "name": "모델선두", "pwin": 0.62, "pplc": 0.90},
            {"bno": 7, "name": "공식합의", "pwin": 0.31, "pplc": 0.75},
        ]
        out = {
            "rows": rows,
            "picks": [],
            "top": rows[0],
            "top_conf": engine._top_confidence(rows[0], rows),
            "selective_conf": {"tier": "normal"},
        }

        boosted = engine._apply_kcycle_rankingpredict_overlay(
            out,
            rows,
            {"ymd": "2024.12.08", "meet": "광명", "race_no": "16"},
        )

        self.assertEqual(boosted["top"]["bno"], 7)
        self.assertEqual(boosted["top_conf"]["label"], "KCYCLE 공식합의 픽")
        self.assertEqual(boosted["selective_conf"]["tier"], "kcycle_all_first_agree")

    def test_kcycle_extreme_overlay_uses_high_confidence_label(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        rows = [
            {"bno": 1, "name": "모델선두", "pwin": 0.62, "pplc": 0.90},
            {"bno": 7, "name": "극고확신", "pwin": 0.31, "pplc": 0.75},
        ]
        out = {
            "rows": rows,
            "picks": [],
            "top": rows[0],
            "top_conf": engine._top_confidence(rows[0], rows),
            "selective_conf": {"tier": "normal"},
        }

        boosted = engine._apply_kcycle_rankingpredict_overlay(
            out,
            rows,
            {"ymd": "2026.06.27", "meet": "광명", "race_no": "14"},
        )

        self.assertEqual(boosted["top"]["bno"], 7)
        self.assertEqual(boosted["top_conf"]["label"], "KCYCLE 극고확신 픽")
        self.assertEqual(boosted["top_conf"]["race_confidence"], "고확신")
        self.assertEqual(boosted["selective_conf"]["tier"], "kcycle_market3_day2_extreme")

    def test_kcycle_order_signal_rewrites_ordered_exotic_picks(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        rows = [
            {"bno": 1, "name": "기본모델1", "pwin": 0.50, "pplc": 0.78},
            {"bno": 5, "name": "공식1착", "pwin": 0.31, "pplc": 0.72},
            {"bno": 7, "name": "공식3착", "pwin": 0.19, "pplc": 0.61},
        ]
        out = {
            "rows": rows,
            "picks": engine.build_picks(rows),
            "top": rows[0],
            "top_conf": engine._top_confidence(rows[0], rows),
            "selective_conf": {"tier": "normal"},
        }

        boosted = engine._apply_kcycle_rankingpredict_overlay(
            out,
            rows,
            {"ymd": "2026.06.27", "meet": "광명", "race_no": "15"},
        )
        picks = {pick["code"]: pick for pick in boosted["picks"]}

        self.assertEqual(picks["단승"]["pick"], ["5번 공식1착"])
        self.assertEqual(picks["쌍승"]["pick"], ["5번 공식1착 → 1번 기본모델1"])
        self.assertEqual(picks["삼쌍"]["pick"], ["5번 공식1착 → 1번 기본모델1 → 7번 공식3착"])
        self.assertIn("KCYCLE 순서신호", picks["삼쌍"]["prob"])
        self.assertIn("exact 27.2%", picks["삼쌍"]["prob"])

    def test_kcycle_support_overlay_uses_middle_confidence_label(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        rows = [
            {"bno": 4, "name": "모델선두", "pwin": 0.42, "pplc": 0.76},
            {"bno": 3, "name": "보조합의", "pwin": 0.31, "pplc": 0.69},
        ]
        out = {
            "rows": rows,
            "picks": [],
            "top": rows[0],
            "top_conf": engine._top_confidence(rows[0], rows),
            "selective_conf": {"tier": "normal"},
        }

        boosted = engine._apply_kcycle_rankingpredict_overlay(
            out,
            rows,
            {"ymd": "2026.06.28", "meet": "광명", "race_no": "7"},
        )

        self.assertEqual(boosted["top"]["bno"], 3)
        self.assertEqual(boosted["top_conf"]["label"], "KCYCLE 보조합의 픽")
        self.assertEqual(boosted["top_conf"]["race_confidence"], "보통")
        self.assertEqual(boosted["selective_conf"]["tier"], "kcycle_market3_support")

    def test_predict_uses_official_fallback_instead_of_demo_on_card_failure(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0
        app_module._PREDICT_CACHE.clear()
        app_module.app.config["TESTING"] = True
        client = app_module.app.test_client()

        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy"}, clear=False), \
             patch.object(app_module.engine, "fetch_race_card", return_value=(None, "API 호출 실패: timeout")):
            response = client.get("/predict?sport=keirin&date=2026-06-28&meet=광명&race_no=7")

        html = response.data.decode("utf-8")
        self.assertEqual(response.status_code, 200)
        self.assertIn("KCYCLE 공식예상 폴백", html)
        self.assertIn("KCYCLE 시장3합의 보조픽", html)
        self.assertNotIn("데모 캐시 경주", html)

    def test_fetch_race_card_reuses_keirin_card_page_cache(self):
        engine.clear_keirin_card_page_cache()
        page_items = [
            {"race_ymd": "20260628", "meet_nm": "광명", "race_no": "5", "racer_no": "1"},
            {"race_ymd": "20260628", "meet_nm": "광명", "race_no": "7", "racer_no": "3"},
        ]

        def fake_api_page(stnd_yr, page, rows, key, timeout=8):
            if page == 1 and rows == 1:
                return 2000, []
            if page == 2 and rows == 1000:
                return 2000, page_items
            return 2000, []

        with patch.object(engine, "_api_page", side_effect=fake_api_page) as api_page:
            first, first_err = engine.fetch_race_card("2026", "2026-06-28", "광명", "5", "dummy")
            second, second_err = engine.fetch_race_card("2026", "2026-06-28", "광명", "7", "dummy")

        self.assertIsNone(first_err)
        self.assertIsNone(second_err)
        self.assertEqual(first[0]["race_no"], "5")
        self.assertEqual(second[0]["race_no"], "7")
        self.assertEqual(api_page.call_count, 2)

    def test_healthz_reports_rankingpredict_cache_status(self):
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        app_module.app.config["TESTING"] = True
        client = app_module.app.test_client()

        response = client.get("/healthz")
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(data["rankingpredict_cache"]["rows"], 10000)
        self.assertEqual(data["rankingpredict_cache"]["latest_date"], "20260628")
        self.assertIn("pages", data["keirin_card_page_cache"])
        self.assertGreaterEqual(data["keirin_card_page_cache"]["ttl"], 60)

    def test_kcycle_rankingpredict_live_failure_uses_cooldown(self):
        engine._KCYCLE_RANKINGPREDICT = {}
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0

        with patch.dict(os.environ, {"KCYCLE_RANKINGPREDICT_ENABLED": "1"}, clear=False), \
             patch.object(engine.urllib.request, "urlopen", side_effect=TimeoutError("blocked")) as urlopen:
            first = engine._kcycle_rankingpredict_signal(
                {"stnd_yr": "2026", "ymd": "2026.07.03", "meet": "광명", "race_no": "1"},
            )
            second = engine._kcycle_rankingpredict_signal(
                {"stnd_yr": "2026", "ymd": "2026.07.03", "meet": "광명", "race_no": "1"},
            )

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertGreater(urlopen.call_count, 0)
        self.assertLessEqual(urlopen.call_count, 3)
        engine._KCYCLE_RANKINGPREDICT = None
        engine._KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0


if __name__ == "__main__":
    unittest.main()
