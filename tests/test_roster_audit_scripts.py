#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import audit_roster_consistency as audit
from scripts import rebuild_learning_priors as rebuild


class RosterAuditScriptsTestCase(unittest.TestCase):
    def test_audit_records_mismatch_and_unchecked_truthfully(self):
        starters = [{"racer_nm": "오염A"}, {"racer_nm": "오염B"}]
        official = ["공식A", "공식B"]

        def fake_fetch(stnd_yr, ymd, meet, race_no, key):
            if str(race_no) == "1":
                return starters, None
            return None, "provider failed"

        with patch.object(audit.engine, "fetch_race_card", side_effect=fake_fetch), \
             patch.object(audit.roster_guard, "verify_roster", return_value={
                 "state": "mismatch",
                 "official_names": official,
                 "checked_at": "2026-07-06T00:00:00+00:00",
             }):
            report = audit.build_report(key="dummy", race_keys=[
                ("20260703", "광명", "1"),
                ("20260703", "광명", "2"),
            ])

        self.assertEqual(report["summary"]["mismatch"], 1)
        self.assertEqual(report["summary"]["unchecked"], 1)
        self.assertEqual(report["rows"][0]["official_names"], official)
        self.assertEqual(report["rows"][1]["reason"], "provider failed")

    def test_rebuild_excludes_mismatch_keys_and_backs_up_existing_priors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "roster_audit_report.json"
            priors = root / "participant_learning_priors.json"
            summary = root / "prediction_feedback_summary.json"
            report.write_text(json.dumps({
                "rows": [
                    {"sport": "keirin", "date": "20260703", "meet": "광명", "race_no": "1", "status": "mismatch"},
                    {"sport": "keirin", "date": "20260703", "meet": "광명", "race_no": "2", "status": "matched"},
                ]
            }), encoding="utf-8")
            priors.write_text('{"old": true}\n', encoding="utf-8")
            built = {
                "summary": {
                    "matched_races": 3,
                    "excluded_roster_mismatch_races": 1,
                },
                "generated_at": "2026-07-06T00:00:00+00:00",
                "alpha": 8.0,
                "min_starts_for_live_adjustment": 5,
            }

            with patch.object(rebuild.feedback, "build_feedback", return_value=built) as build_mock, \
                 patch.object(rebuild.feedback, "write_feedback") as write_mock:
                result = rebuild.rebuild(report, priors, summary)

            excluded = build_mock.call_args.kwargs["exclude_keys"]
            self.assertIn(("keirin", "20260703", "광명", "1"), excluded)
            self.assertEqual(result["excluded"], 1)
            self.assertTrue(Path(result["backup_path"]).exists())
            write_mock.assert_called_once_with(built, priors, summary)

    def test_rebuild_stops_when_report_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                rebuild.mismatch_keys_from_report(missing)


if __name__ == "__main__":
    unittest.main()
