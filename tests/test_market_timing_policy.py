import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_kcycle_market_timing_policy as experiment


def complete_board() -> dict[str, float]:
    board = {}
    odds = 10.0
    for first in range(1, 8):
        for second in range(1, 8):
            for third in range(1, 8):
                if len({first, second, third}) == 3:
                    board[f"{first}-{second}-{third}"] = odds
                    odds += 1.0
    board["1-2-3"] = 1.5
    return board


class MarketTimingPolicyTest(unittest.TestCase):
    def test_run_joins_only_pre_start_live_snapshots_with_separate_outcomes(self):
        board = complete_board()
        snapshot_rows = [
            {
                "date": "20260711",
                "meet": "광명",
                "race_no": "1",
                "source": "collector",
                "snapshot_phase": "pre_result_market_snapshot",
                "fetched_at": "2026-07-11T12:00:00",
                "board": board,
            },
            {
                "date": "20260711",
                "meet": "광명",
                "race_no": "2",
                "source": "collector",
                "snapshot_phase": "pre_result_market_snapshot",
                "fetched_at": "2026-07-11T12:01:00",
                "board": board,
            },
            {
                "date": "20260711",
                "meet": "광명",
                "race_no": "3",
                "source": "archive_import",
                "actual_order": "1-2-3",
                "board": board,
            },
        ]
        outcome_rows = [
            {
                "date": "20260711",
                "meet": "광명",
                "race_no": "1",
                "actual_order": [1, 2, 3],
            },
            {
                "date": "20260711",
                "meet": "광명",
                "race_no": "2",
                "actual_order": [1, 2, 3],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            snapshots = Path(directory) / "snapshots.jsonl"
            outcomes = Path(directory) / "outcomes.jsonl"
            snapshots.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in snapshot_rows) + "\n", encoding="utf-8")
            outcomes.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in outcome_rows) + "\n", encoding="utf-8")
            with patch.object(
                experiment.engine,
                "_kcycle_market_timing_policy",
                side_effect=[{"phase": "early"}, {"phase": "post_start"}],
            ):
                result = experiment.run(snapshots, outcomes)

        self.assertEqual(result["timed_with_outcome"], 1)
        self.assertEqual(result["eligible_completed_races"], 1)
        self.assertEqual(result["post_start_with_outcome"], 1)
        self.assertEqual(result["archive_final_proxy"]["n"], 1)
        self.assertEqual(result["status"], "waiting_for_2000_timed_completed_races")


if __name__ == "__main__":
    unittest.main()
