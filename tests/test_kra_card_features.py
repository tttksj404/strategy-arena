import unittest

import pandas as pd  # noqa: PANDAS_OK — compact dataframe fixtures match production contract

from kra_card_features import add_card_features


class KraCardFeaturesTestCase(unittest.TestCase):
    def test_card_features_parse_weight_allowance_tools_and_prior_owner_form(self):
        frame = pd.DataFrame([
            {"owNo": "O1", "rcDate": "20250101", "win": 1, "wgJk": "-2", "hrTool": "망사눈,혀끈"},
            {"owNo": "O1", "rcDate": "20250108", "win": 0, "wgJk": "0", "hrTool": "-"},
            {"owNo": "NEW", "rcDate": "20250108", "win": 0, "wgJk": "-4", "hrTool": "눈가면+"},
        ])

        result = add_card_features(frame)

        self.assertEqual(result["jk_weight_allowance"].tolist(), [-2.0, 0.0, -4.0])
        self.assertEqual(result["tool_mesh_eye"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(result["tool_tongue_tie"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(result["tool_eye_mask"].tolist(), [0.0, 0.0, 1.0])
        self.assertAlmostEqual(result.loc[0, "owner_win_prior"], 0.1)
        self.assertAlmostEqual(result.loc[1, "owner_win_prior"], 1.0)
        self.assertAlmostEqual(result.loc[2, "owner_win_prior"], 0.1)

    def test_owner_prior_excludes_other_rows_from_the_same_day(self):
        frame = pd.DataFrame([
            {"owNo": "O1", "rcDate": "20250101", "win": 1, "wgJk": "0", "hrTool": "-"},
            {"owNo": "O1", "rcDate": "20250101", "win": 0, "wgJk": "0", "hrTool": "-"},
        ])

        result = add_card_features(frame)

        self.assertEqual(result["owner_win_prior"].tolist(), [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
