#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402

SNAPSHOTS = ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"
OUT_JSONL = ROOT / "data" / "kcycle_entries.jsonl"
OUT_REPORT_JSON = ROOT / "data" / "kcycle_entries_coverage.json"
OUT_REPORT_MD = ROOT / "data" / "kcycle_entries_coverage.md"
PROGRESS = ROOT / "runs" / "prediction_uplift_progress.md"
LOCAL_DB = Path("/Users/tttksj/keirin/data/keirin.db")
MEET_CODE = {"광명": "001", "창원": "002", "부산": "003"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_progress(text: str) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_now()} Phase 1: {text}\n")


def clean_date(value: str) -> str:
    return re.sub(r"\D", "", str(value or ""))[:8]


def db_date(value: str) -> str:
    d = clean_date(value)
    return f"{d[:4]}.{d[4:6]}.{d[6:8]}" if len(d) == 8 else str(value or "")


def race_token(date: str, meet: str, race_no: str) -> tuple[str, str, str]:
    return clean_date(date), str(meet or "").strip(), str(race_no or "").strip().lstrip("0") or "0"


def load_snapshot_races(path: Path) -> dict[tuple[str, str, str], dict]:
    races: dict[tuple[str, str, str], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("actual_order"):
                continue
            token = race_token(str(row.get("date") or ""), str(row.get("meet") or ""), str(row.get("race_no") or ""))
            if token[0] and token not in races:
                races[token] = {
                    "date": token[0],
                    "meet": token[1],
                    "race_no": token[2],
                    "stnd_yr": str(row.get("stnd_yr") or token[0][:4]),
                    "kcycle": row.get("kcycle") or {},
                }
    return races


def existing_keys(path: Path) -> set[str]:
    key_path = Path(f"{path}.keys")
    if not key_path.exists():
        return set()
    return {line.strip() for line in key_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def parse_float(value: object) -> float | None:
    text = str(value or "").replace(",", "").replace('"', ".").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def parse_card_decision_html(html: str, meta: dict) -> list[dict]:
    parser = engine._KcycleTableTextParser()
    parser.feed(html or "")
    entrants = []
    for cells in parser.rows:
        if len(cells) < 15:
            continue
        first = str(cells[0])
        number = re.search(r"\b([1-7])\b", first)
        if not number:
            continue
        name_match = re.search(r"[1-7]\s*([가-힣]{2,5})", first)
        grade_match = re.search(r"\(현재\)\s*([A-Z가-힣0-9]+)", " ".join(cells))
        age_match = re.search(r"/(\d{2})세", first)
        entrants.append(
            {
                "back_no": number.group(1),
                "racer_nm": name_match.group(1) if name_match else "",
                "racer_age": parse_float(age_match.group(1) if age_match else None),
                "racer_grd_cd": grade_match.group(1) if grade_match else "",
                "racer_grd_cur_cd": grade_match.group(1) if grade_match else "",
                "gear_rate": parse_float(cells[1]),
                "rec_200m_scr": parse_float(cells[2]),
                "trng_plc_nm": str(cells[3]).strip(),
                "win_rate": parse_float(cells[4]),
                "high_rate": parse_float(cells[5]),
                "high_3_rate": parse_float(cells[6]),
                "pre_win_cnt": parse_float(cells[8]),
                "brk_win_cnt": parse_float(cells[9]),
                "pas_win_cnt": parse_float(cells[10]),
                "mrk_win_cnt": parse_float(cells[11]),
                "tot_tms_avg_scr": parse_float(cells[13]),
                "recent_rank_text": str(cells[14]).strip(),
            }
        )
    return entrants if len(entrants) >= 5 else []


def fetch_public_entry(meta: dict) -> tuple[list[dict], str | None, str | None]:
    kcycle = meta.get("kcycle") if isinstance(meta.get("kcycle"), dict) else {}
    year = str(kcycle.get("year") or meta.get("stnd_yr") or "")[:4]
    tms = str(kcycle.get("tms") or "").strip()
    day = str(kcycle.get("day") or "").strip()
    if not year or not tms or not day:
        return [], None, "missing_kcycle_tms_day"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = f"https://{engine.KCYCLE_IP}/race/card/decision/{year}/{tms}/{day}"
    request = urllib.request.Request(url, headers={"Host": "www.kcycle.or.kr", "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=8, context=ctx) as response:
            html = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return [], url.replace(engine.KCYCLE_IP, "www.kcycle.or.kr"), f"{type(exc).__name__}: {exc}"
    return parse_card_decision_html(html, meta), url.replace(engine.KCYCLE_IP, "www.kcycle.or.kr"), None


def local_db_entries(tokens: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], list[dict]]:
    if not LOCAL_DB.exists():
        return {}
    wanted_dates = {db_date(token[0]) for token in tokens}
    out: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    query = (
        "select * from race_card where stnd_yr >= '2018' "
        "and race_ymd in ({})"
    ).format(",".join("?" for _ in wanted_dates))
    with sqlite3.connect(LOCAL_DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, sorted(wanted_dates)).fetchall()
    for row in rows:
        item = dict(row)
        token = race_token(str(item.get("race_ymd") or ""), str(item.get("meet_nm") or ""), str(item.get("race_no") or ""))
        if token not in tokens:
            continue
        out[token].append(
            {
                key: item.get(key)
                for key in [
                    "back_no",
                    "racer_nm",
                    "racer_grd_cd",
                    "racer_grd_cur_cd",
                    "racer_grd_bef_cd",
                    "gear_rate",
                    "rec_200m_scr",
                    "trng_plc_nm",
                    "racer_age",
                    "win_rate",
                    "high_rate",
                    "high_3_rate",
                    "tot_tms_avg_scr",
                    "area_tms3_avg_scr",
                    "win_tot_tcnt",
                    "period_no",
                    "race_len",
                    "mrk_win_cnt",
                    "pre_win_cnt",
                    "pas_win_cnt",
                    "brk_win_cnt",
                    "bf1_day1_rank",
                    "bf1_day2_rank",
                    "bf1_day3_rank",
                    "bf2_day1_rank",
                    "bf2_day2_rank",
                    "bf2_day3_rank",
                ]
            }
        )
    return {token: sorted(rows, key=lambda item: int(str(item.get("back_no") or "0"))) for token, rows in out.items() if len(rows) >= 5}


def write_report(payload: dict) -> None:
    lines = [
        "# KCYCLE entry sheet coverage",
        "",
        f"generated_at: {payload['generated_at']}",
        f"snapshot_races: {payload['snapshot_races']}",
        f"written_races: {payload['written_races']}",
        f"coverage: {payload['coverage']:.4f}",
        f"source_used: {payload['source_used']}",
        "",
        "## Public page feasibility probes",
        "| year | date | fetched | rows | error |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in payload["probes"]:
        lines.append(f"| {row['year']} | {row['date']} | {str(row['fetched']).lower()} | {row['entrant_rows']} | {row.get('error') or ''} |")
    lines.extend(["", "## Year coverage", "| year | races | matched | coverage |", "|---:|---:|---:|---:|"])
    for year, row in sorted(payload["year_coverage"].items()):
        lines.append(f"| {year} | {row['races']} | {row['matched']} | {row['coverage']:.4f} |")
    OUT_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=0.6)
    args = parser.parse_args()

    races = load_snapshot_races(SNAPSHOTS)
    sample_years = {}
    for token, meta in races.items():
        year = token[0][:4]
        if year in {"2018", "2022", "2025"} and year not in sample_years:
            sample_years[year] = meta
    probes = []
    public_success = False
    for year in ("2018", "2022", "2025"):
        meta = sample_years.get(year)
        if not meta:
            continue
        entrants, url, error = fetch_public_entry(meta)
        public_success = public_success or bool(entrants)
        probes.append({"year": year, "date": meta["date"], "url": url, "fetched": bool(entrants), "entrant_rows": len(entrants), "error": error})
        time.sleep(max(args.sleep_sec, 0.5))

    token_set = set(races)
    joined = local_db_entries(token_set)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    seen = set() if args.replace else existing_keys(OUT_JSONL)
    mode = "w" if args.replace else "a"
    written = 0
    with OUT_JSONL.open(mode, encoding="utf-8") as handle:
        for token, meta in sorted(races.items()):
            key = "\t".join(token)
            if key in seen or token not in joined:
                continue
            payload = {
                "schema": "kcycle_entry_sheet_v1",
                "source": "local_keirin_db_fallback_after_public_fetch_probe" if not public_success else "kcycle_public_page_or_local_fallback",
                "date": token[0],
                "meet": token[1],
                "race_no": token[2],
                "stnd_yr": meta["stnd_yr"],
                "entrants": joined[token],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            seen.add(key)
            written += 1
    Path(f"{OUT_JSONL}.keys").write_text("\n".join(sorted(seen)) + ("\n" if seen else ""), encoding="utf-8")

    by_year = defaultdict(lambda: {"races": 0, "matched": 0})
    for token in races:
        by_year[token[0][:4]]["races"] += 1
        if token in joined:
            by_year[token[0][:4]]["matched"] += 1
    year_coverage = {
        year: {**row, "coverage": row["matched"] / row["races"] if row["races"] else 0.0}
        for year, row in by_year.items()
    }
    payload = {
        "generated_at": utc_now(),
        "snapshot_races": len(races),
        "written_races": sum(1 for token in races if token in joined),
        "new_rows_written": written,
        "coverage": sum(1 for token in races if token in joined) / len(races) if races else 0.0,
        "source_used": "local_keirin_db_fallback_after_public_fetch_probe",
        "public_probe_success": public_success,
        "probes": probes,
        "year_coverage": year_coverage,
        "fields": [
            "racer_grd_cd",
            "racer_grd_cur_cd",
            "gear_rate",
            "rec_200m_scr",
            "win_rate",
            "high_rate",
            "high_3_rate",
            "trng_plc_nm",
            "racer_age",
        ],
    }
    OUT_REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(payload)
    append_progress(f"entry coverage={payload['coverage']:.4f}, public_probe_success={str(public_success).lower()}, written={payload['written_races']}")
    print(json.dumps({"entries": str(OUT_JSONL), "report": str(OUT_REPORT_JSON), "coverage": payload["coverage"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
