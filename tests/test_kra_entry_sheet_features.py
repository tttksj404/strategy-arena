import tempfile
import unittest
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK — compact dataframe fixtures match production contract

from kra_entry_sheet_features import add_entry_sheet_features, load_entry_sheets, parse_entry_sheet


SAMPLE = (
    "제목 : 26.06.06 제 44 일 토 요일 2경주;\n"
    "                                  --총전적--      --1년전적--\n"
    " 마번  마  명  수득상금  1년수득상금  6개월수득상금  1위/2위/3위/계  1위/2위/3위/계\n"
    " 1 델타최강 4,200,000 4,200,000 4,200,000 1 2 3 10 1 1 1 5\n"
    " 2 댄 싱 위 즈 12,000,000 6,000,000 3,000,000 2 1 0 20 1 0 0 4\n"
)


class KraEntrySheetFeaturesTestCase(unittest.TestCase):
    def test_parser_extracts_as_of_race_counts_and_normalizes_names(self):
        rows = parse_entry_sheet(SAMPLE, "1")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rcDate"], "20260606")
        self.assertEqual(rows[0]["entry_career_starts"], 10)
        self.assertEqual(rows[1]["hrName_entry"], "댄싱위즈")

    def test_loader_and_feature_builder_join_by_race_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seoul"
            path.mkdir()
            (path / "sample.rpt").write_text(SAMPLE, encoding="euc-kr")
            entries = load_entry_sheets(Path(directory))
        frame = pd.DataFrame([
            {"meet": "서울", "rcDate": "20260606", "rcNo": "2", "chulNo": "1", "rk": "r"},
            {"meet": "서울", "rcDate": "20260606", "rcNo": "2", "chulNo": "2", "rk": "r"},
        ])

        result, columns = add_entry_sheet_features(frame, entries)

        self.assertAlmostEqual(result.loc[0, "entry_career_win_rate"], 0.1)
        self.assertAlmostEqual(result.loc[0, "entry_year_place_rate"], 0.6)
        self.assertAlmostEqual(result.loc[1, "entry_year_earn_per_start"], 1_500_000)
        self.assertIn("entry_year_win_rate_rel", columns)


if __name__ == "__main__":
    unittest.main()
