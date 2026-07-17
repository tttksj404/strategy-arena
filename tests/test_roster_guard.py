#!/usr/bin/env python3
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
import roster_guard


FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXPECTED_GM_R1_NAMES = ["황종대", "이흥주", "박진홍", "최건묵", "이승주", "박유찬", "김성진"]


class RosterGuardTestCase(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        app_module._BASE_PREDICTION_CACHE.clear()
        roster_guard.clear_cache()
        self.client = app_module.app.test_client()

    def test_kcycle_options_fixture_maps_date_to_tms_dayord(self):
        html = (FIXTURE_DIR / "kcycle_tmsdayord_options.html").read_text(encoding="utf-8")

        mapping = roster_guard._parse_kcycle_tmsdayord_options(html, 2026)

        self.assertEqual(mapping["2026-07-03"], (27, 1))

    def test_kcycle_html_fetch_uses_subsecond_timeout(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"<html></html>"
        with patch.object(roster_guard.urllib.request, "urlopen", return_value=response) as urlopen:
            roster_guard._fetch_kcycle_html("https://example.test/race")

        self.assertEqual(urlopen.call_args.kwargs["timeout"], 0.75)

    def test_kcycle_card_fixture_extracts_gwangmyeong_race_one_names(self):
        html = (FIXTURE_DIR / "kcycle_card_20260703_gm_r1.html").read_text(encoding="utf-8")

        names = roster_guard._extract_kcycle_names(html, "광명", "1")

        self.assertEqual(names, EXPECTED_GM_R1_NAMES)
        self.assertEqual(set(names), set(EXPECTED_GM_R1_NAMES))

    def test_kcycle_card_extracts_spaced_name_and_one_digit_generation(self):
        html = (
            (FIXTURE_DIR / "kcycle_card_20260703_gm_r1.html")
            .read_text(encoding="utf-8")
            .replace(">황종대<", ">조 택<", 1)
            .replace(">09기/48세<", ">9기/48세<", 1)
        )
        expected_names = ["조택", *EXPECTED_GM_R1_NAMES[1:]]

        names = roster_guard._extract_kcycle_names(html, "광명", "1")

        self.assertEqual(names, expected_names)
        with patch.object(roster_guard, "_provider_names", return_value=names):
            result = roster_guard.verify_roster(
                "keirin",
                "2026-07-03",
                "광명",
                "1",
                [{"racer_nm": name} for name in expected_names],
            )
        self.assertEqual(result["state"], "verified")

    def test_kcycle_official_provider_uses_fixture_fetches_without_network(self):
        options_html = (FIXTURE_DIR / "kcycle_tmsdayord_options.html").read_text(encoding="utf-8")
        card_html = (FIXTURE_DIR / "kcycle_card_20260703_gm_r1.html").read_text(encoding="utf-8")
        fetched_urls: list[str] = []

        def fake_fetch(url: str) -> str:
            fetched_urls.append(url)
            if url.endswith("/race/card/decision"):
                return options_html
            if url.endswith("/race/card/decision/2026/27/1"):
                return card_html
            self.fail(f"unexpected URL: {url}")

        with patch.object(roster_guard, "_fetch_kcycle_html", side_effect=fake_fetch):
            names = roster_guard._kcycle_official_names("2026-07-03", "광명", "1")

        self.assertEqual(names, EXPECTED_GM_R1_NAMES)
        self.assertEqual(fetched_urls, [
            "https://www.kcycle.or.kr/race/card/decision",
            "https://www.kcycle.or.kr/race/card/decision/2026/27/1",
        ])

    def test_verify_roster_marks_same_kcycle_starters_as_verified(self):
        starters = [{"racer_nm": name} for name in EXPECTED_GM_R1_NAMES]

        with patch.object(roster_guard, "_provider_names", return_value=EXPECTED_GM_R1_NAMES):
            result = roster_guard.verify_roster("keirin", "2026-07-03", "광명", "1", starters)

        self.assertEqual(result["state"], "verified")
        self.assertEqual(result["official_names"], EXPECTED_GM_R1_NAMES)

    def test_verify_roster_marks_two_name_difference_as_mismatch(self):
        starters = [{"racer_nm": name} for name in ["방종대", "이흥주", "박진응", "최건묵", "이승주", "박유찬", "김성진"]]

        with patch.object(roster_guard, "_provider_names", return_value=EXPECTED_GM_R1_NAMES):
            result = roster_guard.verify_roster("keirin", "2026-07-03", "광명", "1", starters)

        self.assertEqual(result["state"], "mismatch")
        self.assertEqual(result["official_names"], EXPECTED_GM_R1_NAMES)

    def test_provider_failure_is_unverified_not_verified(self):
        with patch.object(roster_guard, "_provider_names", return_value=None):
            result = roster_guard.verify_roster("keirin", "2026-07-03", "광명", "1", [{"racer_nm": "김성진"}])

        self.assertEqual(result["state"], "unverified")
        self.assertEqual(result["official_names"], [])

    def test_live_decision_blocks_mismatched_roster_without_picks_rows_or_top(self):
        starters = [
            {"back_no": str(i), "racer_nm": name, "racer_grd_cd": "선발"}
            for i, name in enumerate(["최건묵", "황종대", "이흥주", "박진홍", "이승주", "박유찬", "김성진"], start=1)
        ]
        official = ["방종대", "이홍주", "박진응", "최광목", "이승주", "방효찬", "김성진"]

        with patch.dict(os.environ, {"DATAGOKR_SERVICE_KEY": "dummy", "KCYCLE_ENABLED": "0"}, clear=False), \
             patch.object(app_module.engine, "fetch_race_card", return_value=(starters, None)), \
             patch.object(roster_guard, "_provider_names", return_value=official), \
             patch.object(app_module.engine, "predict", side_effect=AssertionError("prediction should be blocked")):
            response = self.client.get(
                "/api/live-decision?sport=keirin&date=2026-07-03&meet=광명&race_no=1",
                headers={"X-RaceLens-Device-Id": "roster-mismatch-device"},
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "roster_mismatch")
        self.assertEqual(payload["message"], "공식 출주표와 일치하지 않아 예측을 중단했습니다")
        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["picks"], [])
        self.assertIsNone(payload["top"])
        self.assertEqual(payload["roster_verification"]["state"], "mismatch")


if __name__ == "__main__":
    unittest.main()
