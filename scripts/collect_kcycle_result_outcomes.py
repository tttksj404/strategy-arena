#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine

DEFAULT_OUT = ROOT / "data" / "kcycle_outcomes.jsonl"
MEET_CODES = {"광명": "001", "창원": "002", "부산": "003"}


def parse_races(raw):
    if raw == "all":
        return list(range(1, 17))
    races = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            races.extend(range(int(start), int(end) + 1))
        else:
            races.append(int(part))
    return sorted(set(races))


def _actual_from_rows(rows):
    finish_by_rank = {}
    racers = []
    in_result_table = False
    for cells in rows:
        if cells[:2] == ["선수명", "순위"]:
            in_result_table = True
            continue
        if in_result_table and cells[:3] == ["승식", "승자", "평균확정배당률"]:
            break
        if not in_result_table or len(cells) < 2:
            continue
        m = re.match(r"\s*([1-7])\s+(.+)", str(cells[0] or ""))
        if not m:
            continue
        try:
            bno = int(m.group(1))
            rank = int(str(cells[1]).strip())
        except ValueError:
            continue
        racers.append({"bno": bno, "name": m.group(2).strip(), "rank": rank})
        if rank in {1, 2, 3}:
            finish_by_rank[rank] = bno
    if not {1, 2, 3} <= set(finish_by_rank):
        return [], racers
    return [finish_by_rank[1], finish_by_rank[2], finish_by_rank[3]], racers


def _payouts_from_rows(rows):
    payouts = {}
    in_payout_table = False
    for cells in rows:
        if cells[:3] == ["승식", "승자", "평균확정배당률"]:
            in_payout_table = True
            continue
        if not in_payout_table or len(cells) < 3:
            continue
        bet_type = str(cells[0] or "").strip()
        winner = re.sub(r"\s+", " ", str(cells[1] or "")).strip()
        odds = engine._kcycle_float(cells[2])
        if bet_type and winner and odds:
            payouts[bet_type] = {"winner": winner, "odds": odds}
    return payouts


def parse_result_popup(html):
    parser = engine._KcycleTableTextParser()
    parser.feed(html or "")
    actual_order, racers = _actual_from_rows(parser.rows)
    payouts = _payouts_from_rows(parser.rows)
    return actual_order, racers, payouts


def fetch_result_popup(stnd_yr, ymd, meet, race_no):
    tms_day = engine._resolve_kcycle_tms(stnd_yr, ymd)
    if not tms_day:
        return None, None
    year, tms, day = tms_day
    if day == 0:
        return None, None
    meet_code = MEET_CODES.get(str(meet).strip())
    if not meet_code:
        return None, None
    rno = (str(race_no).strip().lstrip("0") or "0").zfill(2)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for delta in [0, -1, 1, -2, 2]:
        url = f"https://{engine.KCYCLE_IP}/race/result/general/popup/{year}/{tms + delta}/{day}/{meet_code}/{rno}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Host": "www.kcycle.or.kr",
                    "User-Agent": "Mozilla/5.0",
                    "Referer": f"https://www.kcycle.or.kr/race/result/general/{year}/{tms + delta}/{day}",
                },
            )
            with urllib.request.urlopen(req, timeout=8, context=ctx) as response:
                html = response.read().decode("utf-8", "replace")
            actual_order, racers, payouts = parse_result_popup(html)
            if actual_order:
                return {
                    "schema": "kcycle_result_outcome_v1",
                    "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "source": "kcycle_result_popup",
                    "source_url": url.replace(str(engine.KCYCLE_IP), "www.kcycle.or.kr"),
                    "stnd_yr": str(year),
                    "date": re.sub(r"\D", "", str(ymd or ""))[:8],
                    "meet": str(meet or ""),
                    "race_no": str(race_no or ""),
                    "actual_order": actual_order,
                    "racers": racers,
                    "payouts": payouts,
                }, url
        except Exception:
            continue
    return None, None


def _load_existing_tokens(path):
    tokens = set()
    if not path.exists():
        return tokens
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            date = re.sub(r"\D", "", str(row.get("date") or ""))[:8]
            race_no = str(row.get("race_no") or "").lstrip("0") or str(row.get("race_no") or "")
            actual = "-".join(str(x) for x in (row.get("actual_order") or []))
            tokens.add((date, str(row.get("meet") or ""), race_no, actual))
    return tokens


def collect_once(args):
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens = _load_existing_tokens(out)
    rows = []
    with out.open("a", encoding="utf-8") as handle:
        for race_no in parse_races(args.races):
            record, _url = fetch_result_popup(args.date[:4], args.date, args.meet, race_no)
            saved = False
            if record:
                token = (
                    record["date"],
                    record["meet"],
                    str(record["race_no"]).lstrip("0") or str(record["race_no"]),
                    "-".join(str(x) for x in record["actual_order"]),
                )
                if token not in tokens:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    tokens.add(token)
                    saved = True
            rows.append({
                "race_no": race_no,
                "fetched": bool(record),
                "saved": saved,
                "actual_order": record.get("actual_order") if record else None,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=dt.datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--meet", default="광명")
    parser.add_argument("--races", default="all")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    args.date = re.sub(r"\D", "", args.date)[:8]
    if len(args.date) != 8:
        raise SystemExit("--date must be YYYY-MM-DD or YYYYMMDD")
    rows = collect_once(args)
    print(json.dumps({
        "date": args.date,
        "meet": args.meet,
        "fetched": sum(1 for row in rows if row["fetched"]),
        "saved": sum(1 for row in rows if row["saved"]),
        "rows": rows,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
