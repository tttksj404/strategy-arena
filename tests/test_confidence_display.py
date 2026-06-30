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


if __name__ == "__main__":
    unittest.main()
