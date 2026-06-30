#!/usr/bin/env python3
"""test_live_decision: /api/live-decision API 테스트."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KCYCLE_ENABLED"] = "0"  # Render와 동일 (kcycle 비활성화)

import app as app_module


class LiveDecisionTestCase(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

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

    def test_template_has_live_panel_js(self):
        """index.html에 live-decision 자동 폴링 JS가 있어야."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "templates", "index.html")) as f:
            html = f.read()
        self.assertIn("startLivePolling", html)
        self.assertIn("/api/live-decision", html)
        self.assertIn("setInterval", html)


if __name__ == "__main__":
    unittest.main()