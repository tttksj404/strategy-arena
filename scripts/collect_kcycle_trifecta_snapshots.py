#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine


def parse_races(raw):
    if raw == "all":
        return list(range(1, 17))
    races = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            races.extend(range(int(a), int(b) + 1))
        else:
            races.append(int(part))
    return sorted(set(races))


def collect_once(args):
    if args.out:
        os.environ["KCYCLE_TRIFECTA_SNAPSHOT_PATH"] = str(Path(args.out).expanduser())
    stnd_yr = args.date[:4]
    rows = []
    for race_no in parse_races(args.races):
        board, fetched_at = engine.fetch_kcycle_trifecta_board_with_ts(stnd_yr, args.date, race_no)
        signal = engine._market_trifecta_signal(board)
        saved = engine.save_kcycle_trifecta_snapshot(
            stnd_yr,
            args.date,
            args.meet,
            race_no,
            board,
            fetched_at=fetched_at,
            signal=signal,
            source="collector",
        )
        rows.append({
            "race_no": race_no,
            "fetched": bool(board),
            "board_count": len(board or {}),
            "saved": saved,
            "signal_tier": signal.get("tier") if signal else None,
            "fetched_at": fetched_at,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=dt.datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--meet", default="광명")
    parser.add_argument("--races", default="all")
    parser.add_argument("--out", default="")
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    args.date = "".join(ch for ch in args.date if ch.isdigit())[:8]
    if len(args.date) != 8:
        raise SystemExit("--date must be YYYY-MM-DD or YYYYMMDD")
    iteration = 0
    while True:
        iteration += 1
        rows = collect_once(args)
        print(json.dumps({
            "iteration": iteration,
            "date": args.date,
            "meet": args.meet,
            "saved": sum(1 for row in rows if row["saved"]),
            "fetched": sum(1 for row in rows if row["fetched"]),
            "rows": rows,
        }, ensure_ascii=False), flush=True)
        if args.interval <= 0 or iteration >= args.iterations:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
