#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Final
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine  # noqa: E402

DB_DEFAULT: Final = Path("/Users/tttksj/keirin/data/keirin.db")
SOURCE_DEFAULT: Final = Path("/Users/tttksj/keirin/data/kcycle_full_trifecta_odds_2018_2026.json")
BASE_DEFAULT: Final = ROOT / "data" / "kcycle_trifecta_snapshots.jsonl"
OUT_DEFAULT: Final = ROOT / "data" / "kcycle_trifecta_snapshots_expansion.jsonl"
PROGRESS: Final = ROOT / "runs" / "prediction_uplift_progress.md"
MEET_REV: Final = {"광명": "001", "창원": "002", "부산": "003"}
MEET_NAME: Final = {"001": "광명", "002": "창원", "003": "부산"}
CIRCLE: Final = {chr(0x2460 + i): i + 1 for i in range(9)}
TARGET_YEARS: Final = {str(year) for year in range(2018, 2027)}


@dataclass(frozen=True, slots=True)
class RaceMeta:
    year: str
    meet_code: str
    tms: str
    day: str
    rno: str
    date: str
    meet_name: str
    actual_order: str | None

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (self.year, self.meet_code, self.tms, self.day, self.rno)

    @property
    def url(self) -> str:
        return (
            "https://www.kcycle.or.kr/race/dividendrate/final/"
            f"{self.year}/{self.tms}/{self.day}/{self.meet_code}/{self.rno}"
        )


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_progress(text: str) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_now()} Round5A expansion: {text}\n")


def mach(value: object) -> int | None:
    text = str(value or "")
    for ch in text:
        if ch in CIRCLE:
            return CIRCLE[ch]
    found = re.match(r"\s*\(?(\d+)", text)
    return int(found.group(1)) if found else None


def clean_date(year: object, race_ymd: object) -> str:
    digits = re.sub(r"\D", "", str(race_ymd or ""))
    if len(digits) >= 8:
        return digits[:8]
    return f"{year}{digits[-4:].zfill(4)}"


def load_db_races(path: Path) -> dict[tuple[str, str, str, str, str], RaceMeta]:
    with closing(sqlite3.connect(path)) as con:
        rows = con.execute(
            "SELECT stnd_yr, race_ymd, meet_nm, week_tcnt, day_tcnt, race_no, rank1, rank2, rank3 "
            "FROM race_result WHERE stnd_yr BETWEEN '2018' AND '2026'"
        ).fetchall()
    out: dict[tuple[str, str, str, str, str], RaceMeta] = {}
    for year, race_ymd, meet_nm, tms, day, race_no, rank1, rank2, rank3 in rows:
        year_text = str(year)
        meet_name = str(meet_nm or "").strip()
        meet_code = MEET_REV.get(meet_name, meet_name)
        rno = str(race_no).strip().zfill(2)
        # 창원/부산 등 일부 행은 rank2에 "③이수원②양승용"처럼 2·3착이 패킹되고
        # rank3이 비어 있다 — 세 컬럼을 이어붙여 원형숫자를 순서대로 뽑는다.
        joined = f"{rank1 or ''}{rank2 or ''}{rank3 or ''}"
        circles = [CIRCLE[ch] for ch in joined if ch in CIRCLE]
        if len(circles) == 3 and len(set(circles)) == 3:
            actual = "-".join(str(item) for item in circles)
        else:
            order = [mach(rank1), mach(rank2), mach(rank3)]
            actual = "-".join(str(item) for item in order) if all(item is not None for item in order) else None
        meta = RaceMeta(
            year=year_text,
            meet_code=meet_code,
            tms=str(tms).strip(),
            day=str(day).strip(),
            rno=rno,
            date=clean_date(year_text, race_ymd),
            meet_name=MEET_NAME.get(meet_code, meet_name),
            actual_order=actual,
        )
        out[meta.key] = meta
    return out


def snapshot_key(row: dict) -> tuple[str, str, str, str, str] | None:
    kcycle = row.get("kcycle") if isinstance(row.get("kcycle"), dict) else {}
    year = str(row.get("stnd_yr") or kcycle.get("year") or str(row.get("date") or "")[:4])
    meet = str(kcycle.get("meet") or MEET_REV.get(str(row.get("meet") or "").strip(), "")).strip()
    tms = str(kcycle.get("tms") or "").strip()
    day = str(kcycle.get("day") or "").strip()
    rno = str(kcycle.get("rno") or row.get("race_no") or "").strip().zfill(2)
    if not year or not meet or not tms or not day or not rno:
        return None
    return (year, meet, tms, day, rno)


def load_existing_keys(paths: list[Path]) -> set[tuple[str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = snapshot_key(row)
                if key is not None:
                    keys.add(key)
    return keys


def is_target_gap(meta: RaceMeta) -> bool:
    if meta.year not in TARGET_YEARS:
        return False
    if meta.meet_code in {"002", "003"}:
        return True
    return meta.meet_code == "001" and meta.year in {"2024", "2025"}


def valid_board(raw_board: object) -> dict[str, float] | None:
    if not isinstance(raw_board, dict):
        return None
    board: dict[str, float] = {}
    for combo, odds in raw_board.items():
        combo_text = str(combo)
        if not re.fullmatch(r"[1-7]-[1-7]-[1-7]", combo_text):
            continue
        if len(set(combo_text.split("-"))) != 3:
            continue
        odds_value = float(str(odds).replace(",", ""))
        if odds_value > 0.0 and odds_value < 2999.0:
            board[combo_text] = odds_value
    return board if len(board) == 210 else None


def record_from_board(meta: RaceMeta, board: dict[str, float], source: str) -> dict:
    signal = engine._market_trifecta_signal(board)
    best20 = sorted(board.items(), key=lambda item: (item[1], item[0]))[:20]
    return {
        "schema": "kcycle_trifecta_snapshot_v1",
        "captured_at": "",
        "fetched_at": utc_now() if source == "round5a_http_scrape" else "",
        "source": source,
        "snapshot_phase": "post_result_archive_join",
        "stnd_yr": meta.year,
        "date": meta.date,
        "meet": meta.meet_name,
        "race_no": str(int(meta.rno)),
        "kcycle": {"year": meta.year, "tms": meta.tms, "day": meta.day, "meet": meta.meet_code, "rno": meta.rno},
        "source_url": meta.url,
        "actual_order": meta.actual_order,
        "board_count": len(board),
        "board_hash": engine._trifecta_board_hash(board),
        "best20": best20,
        "signal": engine._live_signal_payload(signal) if signal else None,
        "board": dict(sorted(board.items())),
    }


def load_archive_records(path: Path, races: dict[tuple[str, str, str, str, str], RaceMeta]) -> dict[tuple[str, str, str, str, str], dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str, str, str], dict] = {}
    for item in rows:
        key = (
            str(item.get("year") or ""),
            str(item.get("meet") or ""),
            str(item.get("tms") or ""),
            str(item.get("day") or ""),
            str(item.get("rno") or "").zfill(2),
        )
        meta = races.get(key)
        if meta is None:
            continue
        board = valid_board(item.get("board"))
        if board is None:
            continue
        out[key] = record_from_board(meta, board, "round5a_local_archive")
    return out


def parse_board_from_html(html: str) -> dict[str, float] | None:
    parser = TextExtractor()
    parser.feed(html)
    text = " ".join(parser.parts)
    board: dict[str, float] = {}
    for match in re.finditer(r"([1-7])\s*[-:]\s*([1-7])\s*[-:]\s*([1-7])\s+([0-9][0-9,]*(?:\.[0-9]+)?)", text):
        combo = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        if len(set(combo.split("-"))) != 3:
            continue
        odds = float(match.group(4).replace(",", ""))
        if odds > 0.0 and odds < 2999.0:
            board[combo] = odds
    return board if len(board) == 210 else None


def fetch_board(meta: RaceMeta, timeout: float) -> dict[str, float] | None:
    request = Request(meta.url, headers={"User-Agent": "Mozilla/5.0 strategy-arena-round5a"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        html = response.read().decode("utf-8", errors="replace")
    board = engine.parse_kcycle_trifecta_board(html)
    return board if board else parse_board_from_html(html)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--skip-http", action="store_true")
    args = parser.parse_args()

    races = load_db_races(args.db)
    existing = load_existing_keys([args.base, args.out])
    target = [meta for meta in races.values() if is_target_gap(meta) and meta.key not in existing]
    archive = load_archive_records(args.source, races)
    local_rows = [archive[meta.key] for meta in target if meta.key in archive]
    if local_rows:
        write_jsonl(args.out, local_rows)
        existing.update(snapshot_key(row) for row in local_rows if snapshot_key(row) is not None)
    remaining = [meta for meta in target if meta.key not in existing]
    append_progress(f"target_gap={len(target)} local_archive_added={len(local_rows)} remaining_http={len(remaining)}")

    fetched = failed = skipped_invalid = 0
    first_error = ""
    if not args.skip_http:
        for idx, meta in enumerate(remaining[: args.limit or None], start=1):
            try:
                board = fetch_board(meta, args.timeout)
                # DB week_tcnt와 사이트 tms 번호가 어긋난 회차가 있어(부산·구연도)
                # engine의 ±2 재시도 휴리스틱을 동일 적용한다.
                if board is None and meta.tms.isdigit():
                    for dt in (-1, 1, -2, 2):
                        alt = int(meta.tms) + dt
                        if alt < 1:
                            continue
                        alt_meta = RaceMeta(
                            year=meta.year, meet_code=meta.meet_code, tms=str(alt),
                            day=meta.day, rno=meta.rno, date=meta.date,
                            meet_name=meta.meet_name, actual_order=meta.actual_order,
                        )
                        time.sleep(max(args.sleep, 0.0))
                        board = fetch_board(alt_meta, args.timeout)
                        if board is not None:
                            break
            except (TimeoutError, OSError, URLError) as exc:
                failed += 1
                first_error = first_error or f"{type(exc).__name__}: {exc}"
                break
            if board is None:
                skipped_invalid += 1
            else:
                write_jsonl(args.out, [record_from_board(meta, board, "round5a_http_scrape")])
                fetched += 1
            if idx % 100 == 0:
                append_progress(f"http_progress={idx}/{len(remaining)} fetched={fetched} invalid={skipped_invalid} failed={failed}")
            time.sleep(max(args.sleep, 0.0))

    print(json.dumps({
        "status": "ok" if not first_error else "partial_network_blocked",
        "target_gap": len(target),
        "local_archive_added": len(local_rows),
        "remaining_http": len(remaining),
        "http_fetched": fetched,
        "http_invalid": skipped_invalid,
        "http_failed": failed,
        "first_error": first_error,
        "out": str(args.out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
