import os
import sys
import unittest

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
            meta={"ymd": demo.get("race_ymd"), "meet": "광명", "race_no": "5"},
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


if __name__ == "__main__":
    unittest.main()
