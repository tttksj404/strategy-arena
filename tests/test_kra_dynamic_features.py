import unittest

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — dataframe fixtures match the production contract

from kra_dynamic_features import add_dynamic_history_features, build_dynamic_features
from kra_contextual_history import build_contextual_features
from kra_dynamic_ratings import add_dynamic_ratings
from kra_pedigree_features import add_pedigree_priors
from kra_sectional_features import add_sectional_history_features, build_sectional_features
from kra_training_features import build_features


class KraDynamicFeaturesTestCase(unittest.TestCase):
    def test_sectional_history_uses_venue_specific_times_and_only_prior_dates(self):
        frame = pd.DataFrame({
            "rk": ["s1", "s1", "b1", "b1", "j1", "j1", "future"],
            "meet": ["서울", "서울", "부경", "부경", "제주", "제주", "서울"],
            "hrNo": ["h1", "h2", "h1", "h3", "h1", "h4", "h1"],
            "rcDate": [
                "20260101", "20260101", "20260201", "20260201",
                "20260301", "20260301", "20260401",
            ],
            "rcDist": [1000.0, 1000.0, 1200.0, 1200.0, 900.0, 900.0, 1400.0],
            "rcTime": [60.0, 62.0, 74.0, 76.0, 72.0, 75.0, 200.0],
            "seS1fAccTime": [13.0, 14.0, 0.0, 0.0, 0.0, 0.0, 99.0],
            "seG3fAccTime": [24.0, 25.0, 0.0, 0.0, 0.0, 0.0, 150.0],
            "buS1fTime": [0.0, 0.0, 13.5, 14.5, 0.0, 0.0, 0.0],
            "buG3fAccTime": [0.0, 0.0, 34.0, 35.0, 0.0, 0.0, 0.0],
            "jeS1fTime": [0.0, 0.0, 0.0, 0.0, 17.0, 18.0, 0.0],
            "jeG3fTime": [0.0, 0.0, 0.0, 0.0, 48.0, 51.0, 0.0],
        })

        result = add_sectional_history_features(frame)

        self.assertTrue(np.isnan(result.iloc[0]["hr_sectional_late_adv_mean_4"]))
        self.assertGreater(result.iloc[2]["hr_sectional_late_adv_mean_4"], 0.0)
        self.assertGreater(result.iloc[4]["hr_sectional_late_adv_mean_4"], 0.0)
        self.assertGreater(result.iloc[6]["hr_sectional_late_adv_mean_4"], 0.0)
        self.assertLess(result.iloc[6]["hr_sectional_first200_mean_4"], 20.0)

    def test_sectional_training_contract_adds_history_and_relative_columns(self):
        frame = pd.DataFrame({
            "rk": ["r1", "r2"],
            "meet": ["서울", "서울"],
            "hrNo": ["h1", "h1"],
            "rcDate": ["20260101", "20260201"],
            "rcDist": [1000.0, 1200.0],
            "rcTime": [60.0, 72.0],
            "seS1fAccTime": [13.0, 14.0],
            "seG3fAccTime": [24.0, 34.0],
            "buS1fTime": [0.0, 0.0],
            "buG3fAccTime": [0.0, 0.0],
            "jeS1fTime": [0.0, 0.0],
            "jeG3fTime": [0.0, 0.0],
        })

        result, columns = build_sectional_features(frame, ["rcDist"])

        self.assertIn("hr_sectional_finish_speed_pct_mean_4", columns)
        self.assertIn("hr_sectional_finish_speed_pct_mean_4_rel", columns)
        self.assertAlmostEqual(result.iloc[1]["hr_sectional_last600_last"], 36.0)

    def test_history_features_use_only_earlier_races(self):
        frame = pd.DataFrame({
            "rk": ["r1", "r2", "r3"],
            "hrNo": ["h1", "h1", "h1"],
            "rcDate": ["20260101", "20260201", "20260301"],
            "rcDist": [1000.0, 1200.0, 1400.0],
            "rcTime": [60.0, 70.0, 200.0],
            "field_size": [10.0, 10.0, 10.0],
            "ord": [3.0, 2.0, 1.0],
            "wgHr_base": [480.0, 490.0, 900.0],
            "wgBudam": [55.0, 56.0, 80.0],
            "buS1fOrd": [0.0, 0.0, 0.0],
            "sjS1fOrd": [5.0, 4.0, 1.0],
        })

        result = add_dynamic_history_features(frame)

        self.assertTrue(np.isnan(result.iloc[0]["hr_speed_last"]))
        self.assertAlmostEqual(result.iloc[1]["hr_speed_last"], 1000.0 / 60.0)
        self.assertAlmostEqual(
            result.iloc[2]["hr_speed_mean_3"],
            np.mean([1000.0 / 60.0, 1200.0 / 70.0]),
        )
        self.assertEqual(result.iloc[2]["hr_body_weight_delta"], 410.0)

    def test_same_day_rows_are_not_used_as_history(self):
        frame = pd.DataFrame({
            "rk": ["r1", "r2"],
            "hrNo": ["h1", "h1"],
            "rcDate": ["20260101", "20260101"],
            "rcDist": [1000.0, 1200.0],
            "rcTime": [60.0, 70.0],
            "field_size": [10.0, 10.0],
            "ord": [1.0, 2.0],
            "wgHr_base": [480.0, 490.0],
            "wgBudam": [55.0, 56.0],
            "buS1fOrd": [0.0, 0.0],
            "sjS1fOrd": [1.0, 2.0],
        })

        result = add_dynamic_history_features(frame)

        self.assertTrue(result["hr_speed_last"].isna().all())

    def test_training_contract_includes_dynamic_history_columns(self):
        frame = pd.DataFrame([
            {
                "rk": "r1", "hrNo": "h1", "rcDate": "20260101", "rcDist": 1000.0,
                "rcTime": 60.0, "field_size": 2, "ord": 1, "win": 1, "place": 1,
                "wgHr_base": 480.0, "wgBudam": 55.0, "buS1fOrd": 0.0,
                "sjS1fOrd": 1.0, "rating": 50.0, "age": 3.0, "chulNo": 1.0,
                "jkNo": "j1", "trNo": "t1", "sex": "수", "budam": "별정",
            },
            {
                "rk": "r2", "hrNo": "h1", "rcDate": "20260201", "rcDist": 1200.0,
                "rcTime": 72.0, "field_size": 2, "ord": 2, "win": 0, "place": 1,
                "wgHr_base": 485.0, "wgBudam": 56.0, "buS1fOrd": 0.0,
                "sjS1fOrd": 2.0, "rating": 50.0, "age": 3.0, "chulNo": 1.0,
                "jkNo": "j1", "trNo": "t1", "sex": "수", "budam": "별정",
            },
        ])

        baseline, baseline_columns = build_features(frame)
        result, columns = build_dynamic_features(baseline, baseline_columns)

        self.assertIn("hr_speed_last", columns)
        self.assertAlmostEqual(result.iloc[1]["hr_speed_last"], 1000.0 / 60.0)

    def test_contextual_strength_excludes_current_race_results(self):
        frame = pd.DataFrame([
            {
                "rk": "r1", "hrNo": "h1", "rcDate": "20260101", "rcDist": 1000.0,
                "rcTime": 60.0, "field_size": 2, "ord": 1, "win": 1, "place": 1,
                "wgHr_base": 480.0, "wgBudam": 55.0, "buS1fOrd": 0.0,
                "sjS1fOrd": 1.0, "rating": 50.0, "age": 3.0, "chulNo": 1.0,
                "jkNo": "j1", "trNo": "t1", "sex": "수", "budam": "별정",
                "meet": "서울", "rank": "국5등급", "track": "양호 (5%)", "weather": "맑음",
            },
            {
                "rk": "r1", "hrNo": "h2", "rcDate": "20260101", "rcDist": 1000.0,
                "rcTime": 62.0, "field_size": 2, "ord": 2, "win": 0, "place": 1,
                "wgHr_base": 470.0, "wgBudam": 54.0, "buS1fOrd": 0.0,
                "sjS1fOrd": 2.0, "rating": 45.0, "age": 3.0, "chulNo": 2.0,
                "jkNo": "j2", "trNo": "t2", "sex": "암", "budam": "별정",
                "meet": "서울", "rank": "국5등급", "track": "양호 (5%)", "weather": "맑음",
            },
            {
                "rk": "r2", "hrNo": "h1", "rcDate": "20260201", "rcDist": 1000.0,
                "rcTime": 200.0, "field_size": 2, "ord": 2, "win": 0, "place": 1,
                "wgHr_base": 485.0, "wgBudam": 56.0, "buS1fOrd": 0.0,
                "sjS1fOrd": 2.0, "rating": 50.0, "age": 3.0, "chulNo": 1.0,
                "jkNo": "j1", "trNo": "t1", "sex": "수", "budam": "별정",
                "meet": "서울", "rank": "국5등급", "track": "다습 (12%)", "weather": "흐림",
            },
        ])
        baseline, baseline_columns = build_features(frame)

        result, columns = build_contextual_features(baseline, baseline_columns)

        expected = (1000.0 / 60.0) - np.median([1000.0 / 60.0, 1000.0 / 62.0])
        self.assertAlmostEqual(result.iloc[2]["hr_race_speed_rel_mean_3"], expected)
        self.assertAlmostEqual(result.iloc[2]["hr_same_distance_finish_prior"], 0.0)
        self.assertIn("jk_recent_win_50_rel", columns)

    def test_dynamic_ratings_update_only_after_the_full_race_date(self):
        frame = pd.DataFrame({
            "rk": ["r1", "r1", "r2", "r2", "r3", "r3"],
            "rcDate": ["20260101", "20260101", "20260101", "20260101", "20260201", "20260201"],
            "hrNo": ["h1", "h2", "h1", "h3", "h1", "h2"],
            "jkNo": ["j1", "j2", "j1", "j3", "j1", "j2"],
            "trNo": ["t1", "t2", "t1", "t3", "t1", "t2"],
            "ord": [1, 2, 1, 2, 2, 1],
            "field_size": [2, 2, 2, 2, 2, 2],
        })

        result = add_dynamic_ratings(frame)

        self.assertEqual(result.loc[0, "hr_elo"], result.loc[2, "hr_elo"])
        self.assertEqual(result.loc[0, "jk_elo"], result.loc[2, "jk_elo"])
        self.assertGreater(result.loc[4, "hr_elo"], result.loc[5, "hr_elo"])

    def test_pedigree_priors_exclude_same_day_offspring_results(self):
        frame = pd.DataFrame({
            "rcDate": ["20260101", "20260101", "20260201"],
            "faHrNo": ["s1", "s1", "s1"],
            "moHrNo": ["d1", "d2", "d3"],
            "win": [1, 0, 0],
            "place": [1, 0, 0],
            "finish_pct": [0.0, 1.0, 0.5],
        })

        result = add_pedigree_priors(frame)

        self.assertEqual(result.loc[0, "sire_win_prior"], result.loc[1, "sire_win_prior"])
        self.assertAlmostEqual(result.loc[2, "sire_win_prior"], 1.5 / 7.0)
