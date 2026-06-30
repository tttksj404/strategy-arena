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

    def test_keirin_predict_without_meta_does_not_crash(self):
        demo = engine.load_demo_race()

        output = engine.predict(demo["items"])

        self.assertNotIn("error", output)
        self.assertEqual(len(output["rows"]), 7)

    def test_keirin_selective_confidence_extreme_tier(self):
        tier = engine._keirin_selective_confidence({"pwin": 0.55, "pplc": 0.91})

        self.assertEqual(tier["tier"], "extreme")
        self.assertAlmostEqual(tier["expected_top1"], 0.8175)

    def test_keirin_selective_confidence_broad_tier(self):
        tier = engine._keirin_selective_confidence({"pwin": 0.61, "pplc": 0.80})

        self.assertEqual(tier["tier"], "broad")
        self.assertAlmostEqual(tier["expected_top1"], 0.7287)


if __name__ == "__main__":
    unittest.main()
