#!/usr/bin/env python3
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine

MEET_CODE = {"001": "광명", "002": "창원", "003": "부산", "광명": "광명", "창원": "창원", "부산": "부산"}
MEET_REV = {"광명": "001", "창원": "002", "부산": "003"}
CIRCLE = {chr(0x2460 + i): i + 1 for i in range(9)}


def mach(value):
    text = str(value or "")
    for ch in text:
        if ch in CIRCLE:
            return CIRCLE[ch]
    found = re.match(r"\s*\(?(\d+)", text)
    return int(found.group(1)) if found else None


def load_result_lookup(db_path):
    if not db_path or not Path(db_path).exists():
        return {}
    lookup = {}
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT stnd_yr, race_ymd, meet_nm, week_tcnt, day_tcnt, race_no, rank1, rank2, rank3 "
            "FROM race_result"
        ).fetchall()
    for yr, race_ymd, meet_nm, week_tcnt, day_tcnt, race_no, rank1, rank2, rank3 in rows:
        meet = str(meet_nm or "").strip()
        meet_code = MEET_REV.get(meet, meet)
        date = f"{yr}{re.sub(r'\\D', '', str(race_ymd))[-4:].zfill(4)}"
        target = [mach(rank1), mach(rank2), mach(rank3)]
        actual = "-".join(map(str, target)) if all(x is not None for x in target) else None
        key = (str(yr), meet_code, str(week_tcnt), str(day_tcnt), str(race_no).zfill(2))
        lookup[key] = {"date": date, "meet": meet, "actual_order": actual}
    return lookup


def existing_keys(path):
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((
                str(row.get("date", "")),
                str(row.get("meet", "")),
                str(row.get("race_no", "")).zfill(2),
                str(row.get("board_hash", "")),
            ))
    return keys


def write_key_index(path, keys):
    index_path = Path(f"{path}.keys")
    with index_path.open("w", encoding="utf-8") as f:
        for key in sorted(keys):
            f.write(engine._snapshot_key_token(key) + "\n")


def snapshot_from_item(item, lookup):
    board = {
        str(combo): float(odds)
        for combo, odds in (item.get("board") or {}).items()
        if re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)) and float(odds) > 0
    }
    if len(board) < 150:
        return None
    year = str(item.get("year", "")).strip()
    meet_code = str(item.get("meet", "")).strip()
    tms = str(item.get("tms", "")).strip()
    day = str(item.get("day", "")).strip()
    rno = str(item.get("rno", "")).strip().zfill(2)
    found = lookup.get((year, meet_code, tms, day, rno), {})
    meet = found.get("meet") or MEET_CODE.get(meet_code, meet_code)
    date = found.get("date") or ""
    signal = engine._market_trifecta_signal(board)
    best20 = sorted(board.items(), key=lambda kv: (kv[1], kv[0]))[:20]
    return {
        "schema": "kcycle_trifecta_snapshot_v1",
        "captured_at": "",
        "fetched_at": "",
        "source": "archive_import",
        "stnd_yr": year,
        "date": date,
        "meet": meet,
        "race_no": str(int(rno)),
        "kcycle": {"year": year, "tms": tms, "day": day, "meet": meet_code, "rno": rno},
        "source_url": item.get("source_url"),
        "actual_order": found.get("actual_order"),
        "board_count": len(board),
        "board_hash": engine._trifecta_board_hash(board),
        "best20": best20,
        "signal": engine._live_signal_payload(signal) if signal else None,
        "board": dict(sorted(board.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/Users/tttksj/keirin/data/kcycle_full_trifecta_odds_2018_2026.json")
    parser.add_argument("--db", default="/Users/tttksj/keirin/data/keirin.db")
    parser.add_argument("--out", default=str(ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    out = Path(args.out).expanduser()
    lookup = load_result_lookup(args.db)
    seen = set() if args.replace else existing_keys(out)
    imported = skipped = missing_date = signals = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.replace else "a"
    data = json.loads(source.read_text(encoding="utf-8"))
    with out.open(mode, encoding="utf-8") as f:
        for item in data:
            record = snapshot_from_item(item, lookup)
            if record is None:
                skipped += 1
                continue
            key = (
                str(record.get("date", "")),
                str(record.get("meet", "")),
                str(record.get("race_no", "")).zfill(2),
                str(record.get("board_hash", "")),
            )
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            if not record["date"]:
                missing_date += 1
            if record.get("signal"):
                signals += 1
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            imported += 1
    write_key_index(out, seen)
    print(json.dumps({
        "source": str(source),
        "out": str(out),
        "source_rows": len(data),
        "imported": imported,
        "skipped": skipped,
        "missing_date": missing_date,
        "signals": signals,
        "total_keys": len(seen),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
