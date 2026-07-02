#!/usr/bin/env python3
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import audit_kcycle_trifecta_snapshot_corpus as audit_script
import engine
from test_live_decision import make_trifecta_candidate_board


class TrifectaSnapshotAuditTestCase(unittest.TestCase):
    def test_audit_recomputes_signal_and_exact_hit(self):
        board = make_trifecta_candidate_board()
        signal = engine._market_trifecta_signal(board)
        key = ("20260628", "광명", "07", engine._trifecta_board_hash(board))
        record = {
            "date": "20260628",
            "meet": "광명",
            "race_no": "7",
            "stnd_yr": "2026",
            "actual_order": "5-1-7",
            "board_count": len(board),
            "board_hash": key[3],
            "signal": engine._live_signal_payload(signal),
            "board": board,
        }

        result = audit_script.audit([record], {engine._snapshot_key_token(key)})

        self.assertTrue(result["ok"])
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["signal_count"], 1)
        self.assertEqual(result["rule_metrics_by_year"]["2026"]["selected_n"], 1)
        self.assertEqual(result["rule_metrics_by_year"]["2026"]["hits"], 1)
        self.assertEqual(result["critical_failures"]["stored_signal_mismatch"], 0)

    def test_audit_fails_on_index_mismatch(self):
        board = make_trifecta_candidate_board()
        record = {
            "date": "20260628",
            "meet": "광명",
            "race_no": "7",
            "stnd_yr": "2026",
            "actual_order": "5-1-7",
            "board_count": len(board),
            "board_hash": engine._trifecta_board_hash(board),
            "signal": engine._live_signal_payload(engine._market_trifecta_signal(board)),
            "board": board,
        }

        result = audit_script.audit([record], set())

        self.assertFalse(result["ok"])
        self.assertEqual(result["critical_failures"]["missing_index_tokens"], 1)

    def test_audit_allows_legacy_50_candidate_tier_downgrade(self):
        board = make_trifecta_candidate_board()
        key = ("20260628", "광명", "07", engine._trifecta_board_hash(board))
        signal = engine._live_signal_payload(engine._market_trifecta_signal(board))
        signal["tier"] = "market_trifecta_50_candidate"
        record = {
            "date": "20260628",
            "meet": "광명",
            "race_no": "7",
            "stnd_yr": "2026",
            "actual_order": "5-1-7",
            "board_count": len(board),
            "board_hash": key[3],
            "signal": signal,
            "board": board,
        }

        result = audit_script.audit([record], {engine._snapshot_key_token(key)})

        self.assertTrue(result["ok"])
        self.assertEqual(result["stored_signal_compatible_downgrades"], 1)
        self.assertEqual(result["critical_failures"]["stored_signal_mismatch"], 0)


if __name__ == "__main__":
    unittest.main()
