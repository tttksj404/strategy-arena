import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import collect_kcycle_result_outcomes as collector


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"stale-popup"


class KcycleResultOutcomeCollectorTest(unittest.TestCase):
    def test_rejects_a_complete_popup_from_a_different_tms_round(self):
        with patch.object(collector.engine, "_resolve_kcycle_tms", return_value=(2026, 29, 1)), patch.object(
            collector.urllib.request,
            "urlopen",
            side_effect=[OSError("current round unavailable"), _Response()],
        ), patch.object(collector, "parse_result_popup", return_value=([1, 2, 3], [], {})):
            record, url = collector.fetch_result_popup("2026", "20260717", "광명", 8)

        self.assertIsNone(record)
        self.assertIsNone(url)


if __name__ == "__main__":
    unittest.main()
