import unittest

import pandas as pd  # noqa: PANDAS_OK — compact dataframe fixtures match production contract

from kra_conditioning_features import add_conditioning_features, parse_conditioning_page


PAGE = (
    "<table><tbody><tr><td>1</td><td><a>불타는심장</a></td>"
    "<td><a>관9</a></td><td><a>관10</a></td><td><a></a></td>"
    "<td><a>기원17</a></td><td><a>관12</a></td><td><a></a></td>"
    "<td><a>기원11</a></td><td><a>기원13</a></td><td><a>관12</a></td>"
    "<td><a>기원19</a></td><td><a></a></td><td><a></a></td></tr></tbody></table>"
)


class KraConditioningFeaturesTestCase(unittest.TestCase):
    def test_parser_extracts_twelve_pre_race_training_slots(self):
        rows = parse_conditioning_page(PAGE, "1", "20260607", "9")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["condition_minutes"], [9, 10, 0, 17, 12, 0, 11, 13, 12, 19, 0, 0])

    def test_feature_builder_calculates_weekly_conditioning(self):
        conditioning = pd.DataFrame(parse_conditioning_page(PAGE, "1", "20260607", "9"))
        frame = pd.DataFrame([{
            "meet": "서울", "rcDate": "20260607", "rcNo": "9", "chulNo": "1", "rk": "race",
        }])

        result, columns = add_conditioning_features(frame, conditioning)

        self.assertEqual(result.loc[0, "condition_sessions_previous"], 4)
        self.assertEqual(result.loc[0, "condition_minutes_current"], 55)
        self.assertAlmostEqual(result.loc[0, "condition_caretaker_share"], 4 / 8)
        self.assertIn("condition_minutes_2w_rel", columns)


if __name__ == "__main__":
    unittest.main()
