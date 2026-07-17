#!/usr/bin/env python3
import os
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import import_kcycle_full_trifecta_archive as importer


def make_board():
    board = {}
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                if len({a, b, c}) == 3:
                    board[f"{a}-{b}-{c}"] = 1000.0
    board["4-2-6"] = 4.7
    return board


class TrifectaArchiveImportTestCase(unittest.TestCase):
    def test_snapshot_from_item_uses_result_date_and_actual_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "keirin.db")
            with closing(sqlite3.connect(db_path)) as con:
                con.execute(
                    "CREATE TABLE race_result ("
                    "stnd_yr TEXT, race_ymd TEXT, meet_nm TEXT, week_tcnt TEXT, day_tcnt TEXT, "
                    "race_no TEXT, rank1 TEXT, rank2 TEXT, rank3 TEXT)"
                )
                con.execute(
                    "INSERT INTO race_result VALUES (?,?,?,?,?,?,?,?,?)",
                    ("2026", "0628", "광명", "26", "3", "05", "④", "②", "⑥"),
                )
                con.commit()

            lookup = importer.load_result_lookup(db_path)
            item = {
                "year": "2026",
                "tms": "26",
                "day": "3",
                "meet": "001",
                "rno": "05",
                "board": make_board(),
                "source_url": "https://www.kcycle.or.kr/race/dividendrate/final/2026/26/3/001/05",
            }
            record = importer.snapshot_from_item(item, lookup)

        self.assertEqual(record["date"], "20260628")
        self.assertEqual(record["meet"], "광명")
        self.assertEqual(record["race_no"], "5")
        self.assertEqual(record["source"], "archive_import")
        self.assertEqual(record["snapshot_phase"], "post_result_archive_join")
        self.assertEqual(record["actual_order"], "4-2-6")
        self.assertEqual(record["board_count"], 210)
        self.assertEqual(record["best20"][0], ("4-2-6", 4.7))

    def test_write_key_index_matches_snapshot_dedupe_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshots.jsonl")
            key = ("20260628", "광명", "05", "abc123")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"date": "20260628", "meet": "광명", "race_no": "5", "board_hash": "abc123"}) + "\n")
            importer.write_key_index(path, {key})

            lines = open(path + ".keys", encoding="utf-8").read().splitlines()

        self.assertEqual(lines, ["20260628\t광명\t05\tabc123"])


if __name__ == "__main__":
    unittest.main()
