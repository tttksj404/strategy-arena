import unittest

import pandas as pd  # noqa: PANDAS_OK — compact dataframe fixtures match production contract

from kra_health_features import add_health_features, parse_health_page


PAGE = (
    "<table><tr><td>1</td><td><a>불타는심장</a></td><td>"
    '<a title="상세보기">2026/06/07 경기후 피로 <span>열기</span></a>'
    '<a title="상세보기">2026/06/05 좌후지교돌상 <span>열기</span></a>'
    '<a title="상세보기">2026/05/22 운동기인성 피로회복 <span>열기</span></a>'
    "</td><td></td></tr></table>"
)


class KraHealthFeaturesTestCase(unittest.TestCase):
    def test_parser_excludes_same_day_treatments(self):
        rows = parse_health_page(PAGE, "1", "20260607", "9")

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["health_treatments"]), 2)
        self.assertEqual(rows[0]["health_treatments"][0]["date"], "20260605")

    def test_feature_builder_counts_recent_health_categories(self):
        health = pd.DataFrame(parse_health_page(PAGE, "1", "20260607", "9"))
        frame = pd.DataFrame([{
            "meet": "서울", "rcDate": "20260607", "rcNo": "9", "chulNo": "1", "rk": "race",
        }])

        result, columns = add_health_features(frame, health)

        self.assertEqual(result.loc[0, "health_treatments_30d"], 2)
        self.assertEqual(result.loc[0, "health_locomotor_90d"], 1)
        self.assertEqual(result.loc[0, "health_fatigue_90d"], 1)
        self.assertEqual(result.loc[0, "health_days_since_treatment"], 2)
        self.assertIn("health_locomotor_90d_rel", columns)


if __name__ == "__main__":
    unittest.main()
