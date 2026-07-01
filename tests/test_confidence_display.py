import sys
import unittest
from pathlib import Path

from flask import Flask, render_template_string

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine


class ConfidenceDisplayTest(unittest.TestCase):
    def test_top_confidence_mid_grade_uses_mid_label(self):
        rows = [
            {"bno": 1, "name": "A", "pwin": 0.35, "pplc": 0.71},
            {"bno": 2, "name": "B", "pwin": 0.22, "pplc": 0.62},
            {"bno": 3, "name": "C", "pwin": 0.10, "pplc": 0.40},
        ]

        top_conf = engine._top_confidence(rows[0], rows)

        self.assertEqual(top_conf["grade"], "중")
        self.assertEqual(top_conf["label"], "상대 우세 픽")

    def test_template_renders_race_confidence_badge(self):
        app = Flask(__name__)
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        result = {
            "kind": "ok",
            "src": "live",
            "sport_label": "경마(KRA)",
            "is_future": False,
            "n_starters": 3,
            "info": {"stnd_yr": "2026", "ymd": "20260630", "meet": "서울", "race_no": "1"},
            "top": {"bno": 1, "name": "테스트마", "pwin": 0.62, "pplc": 0.88},
            "top_conf": {"grade": "강", "label": "최고확신 픽", "icon": "🏆", "race_confidence": "고확신"},
            "rows": [
                {"bno": 1, "name": "테스트마", "grade": "", "pwin": 0.62, "pplc": 0.88},
                {"bno": 2, "name": "상대마", "grade": "", "pwin": 0.28, "pplc": 0.70},
                {"bno": 3, "name": "복병마", "grade": "", "pwin": 0.10, "pplc": 0.42},
            ],
            "picks": [],
        }

        with app.app_context():
            html = render_template_string(
                template,
                disclaimer="테스트 면책",
                keirin_meets=["광명"],
                kra_meets=["서울", "제주", "부경"],
                today="2026-06-30",
                has_key=True,
                recent_days=[],
                default_date="2026-06-30",
                schedule_hint="",
                result=result,
                sport="horse",
                date="2026-06-30",
                meet="서울",
                race_no="1",
            )

        self.assertIn('class="conf-badge 고확신"', html)
        self.assertIn("확신도 고확신", " ".join(html.split()))

    def test_template_renders_selective_confidence_tier(self):
        app = Flask(__name__)
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        result = {
            "kind": "ok",
            "src": "live",
            "is_future": False,
            "n_starters": 3,
            "info": {"stnd_yr": "2026", "ymd": "20260630", "meet": "광명", "race_no": "1"},
            "top": {"bno": 1, "name": "테스트선수", "pwin": 0.62, "pplc": 0.91},
            "top_conf": {"grade": "강", "label": "최고확신 픽", "icon": "🏆", "race_confidence": "고확신"},
            "selective_conf": {
                "tier": "kcycle_all_first_agree",
                "label": "KCYCLE 공식합의 86%급 고확신",
                "expected_top1": 0.8649,
                "coverage": 0.0664,
                "rule": "AI 예측·인기배당률·적중률5%·환급률5% 1착 모두 일치",
                "validation_split": "2025 select -> 2026 OOS",
                "rolling_weighted_top1": 0.8086,
                "rolling_coverage": 0.3252,
                "rolling_min_year_top1": 0.7871,
            },
            "rows": [
                {"bno": 1, "name": "테스트선수", "grade": "선발", "pwin": 0.62, "pplc": 0.91},
                {"bno": 2, "name": "상대선수", "grade": "선발", "pwin": 0.28, "pplc": 0.70},
                {"bno": 3, "name": "복병선수", "grade": "선발", "pwin": 0.10, "pplc": 0.42},
            ],
            "picks": [],
        }

        with app.app_context():
            html = render_template_string(
                template,
                disclaimer="테스트 면책",
                keirin_meets=["광명"],
                kra_meets=["서울", "제주", "부경"],
                today="2026-06-30",
                has_key=True,
                recent_days=[],
                default_date="2026-06-30",
                schedule_hint="",
                result=result,
                sport="keirin",
                date="2026-06-30",
                meet="광명",
                race_no="1",
            )

        self.assertIn("KCYCLE 공식합의 86%급 고확신", html)
        self.assertIn("2025 select -&gt; 2026 OOS top1 86.5%", html)
        self.assertIn("롤링 재검증 top1 80.9%", html)
        self.assertIn("최저연도 78.7%", html)

    def test_template_renders_official_support_fallback_as_middle_grade(self):
        app = Flask(__name__)
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        result = {
            "kind": "official_fallback",
            "title": "KCYCLE 공식예상 폴백",
            "info": {"stnd_yr": "2026", "ymd": "20260628", "meet": "광명", "race_no": "7"},
            "signal": {
                "tier": "kcycle_market3_support",
                "label": "KCYCLE 시장3합의 보조픽",
                "expected_top1": 0.6656,
                "coverage": 0.7195,
                "rule": "인기배당률·적중률5%·환급률5% 1착 일치",
                "validation_split": "2025 select -> 2026 OOS 광명, 고확신 제외",
            },
            "order": ["3", "6", "7"],
            "top_bno": "3",
            "fetch_err": "timeout",
        }

        with app.app_context():
            html = render_template_string(
                template,
                disclaimer="테스트 면책",
                keirin_meets=["광명"],
                kra_meets=["서울", "제주", "부경"],
                today="2026-06-30",
                has_key=True,
                recent_days=[],
                default_date="2026-06-30",
                schedule_hint="",
                result=result,
                sport="keirin",
                date="2026-06-28",
                meet="광명",
                race_no="7",
            )

        self.assertIn("KCYCLE 시장3합의 보조픽", html)
        self.assertIn("OOS 광명, 고확신 제외 top1 66.6%", html)
        self.assertIn('class="code">단승<span class="gr 중">중</span>', html)
        self.assertIn('class="code">쌍승<span class="gr 약">약</span>', html)
        self.assertIn("순서권 리스크", html)
        self.assertIn("KCYCLE 보조합의 1순위", html)


if __name__ == "__main__":
    unittest.main()
