#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine
import roster_guard

DEFAULT_JSON = ROOT / "data" / "roster_audit_report.json"
DEFAULT_MD = ROOT / "data" / "roster_audit_report.md"


def _data_key() -> str | None:
    for name in ("DATAGOKR_SERVICE_KEY", "datagokr", "DATAGOKR", "DATA_GO_KR_KEY", "SERVICE_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _name_from_card(starter: dict) -> str:
    return str(starter.get("racer_nm") or "").strip()


def _row(date: str, meet: str, race_no: str, status: str, card_names=None, official_names=None, reason: str = "") -> dict:
    return {
        "sport": "keirin",
        "date": re.sub(r"\D", "", str(date or ""))[:8],
        "meet": meet,
        "race_no": str(int(str(race_no).strip().lstrip("0") or "0")),
        "status": status,
        "card_names": list(card_names or []),
        "official_names": list(official_names or []),
        "reason": reason,
    }


def audit_race(date: str, meet: str, race_no: str, key: str | None) -> dict:
    if not key:
        return _row(date, meet, race_no, "unchecked", reason="DATAGOKR_SERVICE_KEY missing")
    starters, err = engine.fetch_race_card(str(date)[:4], date, meet, race_no, key)
    if err or not starters:
        return _row(date, meet, race_no, "unchecked", reason=str(err or "card roster missing"))
    verification = roster_guard.verify_roster("keirin", date, meet, race_no, starters)
    status = "unchecked" if verification["state"] == "unverified" else verification["state"]
    if status == "verified":
        status = "matched"
    return _row(
        date,
        meet,
        race_no,
        status,
        card_names=[_name_from_card(starter) for starter in starters if _name_from_card(starter)],
        official_names=verification["official_names"],
    )


def recent_keirin_races(days: int, key: str | None) -> list[tuple[str, str, str]]:
    meet = "광명"
    if not key:
        return []
    recent_days = engine.recent_race_days("keirin", meet, key, n=max(1, days))
    races: list[tuple[str, str, str]] = []
    for date in recent_days[:days]:
        for race_no in range(1, 17):
            races.append((date, meet, str(race_no)))
    return races


def build_report(days: int = 14, key: str | None = None, race_keys: list[tuple[str, str, str]] | None = None) -> dict:
    selected_key = key if key is not None else _data_key()
    keys = race_keys if race_keys is not None else recent_keirin_races(days, selected_key)
    rows = [audit_race(date, meet, race_no, selected_key) for date, meet, race_no in keys]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "days": days,
        "summary": counts,
        "rows": rows,
    }


def write_report(report: dict, json_path: Path = DEFAULT_JSON, md_path: Path = DEFAULT_MD) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = report.get("summary", {})
    lines = [
        "# Roster Audit Report",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- matched: {summary.get('matched', 0)}",
        f"- mismatch: {summary.get('mismatch', 0)}",
        f"- unchecked: {summary.get('unchecked', 0)}",
        "",
        "| date | meet | race | status | reason |",
        "|---|---|---:|---|---|",
    ]
    for row in report.get("rows", []):
        lines.append(f"| {row['date']} | {row['meet']} | {row['race_no']} | {row['status']} | {row.get('reason', '')} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out-json", default=str(DEFAULT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_MD))
    args = parser.parse_args()
    report = build_report(days=args.days)
    write_report(report, Path(args.out_json), Path(args.out_md))
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
