from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_health_features import parse_health_page  # noqa: E402
from tools.kra_conditioning_collect import MEET_DIRS, race_keys  # noqa: E402
from tools.kra_dual_phase_experiment import DEFAULT_DB  # noqa: E402


ENDPOINT: Final = "https://race.kra.co.kr/chulmainfo/chulmaDetailInfoAccessoryState.do"


def collect(archive: Path, key: tuple[str, str, str]) -> tuple[bool, int]:
    meet, race_date, race_number = key
    target = archive / MEET_DIRS[meet] / f"{race_date}_{int(race_number):02d}.json"
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        return False, len(payload["rows"])
    data = urllib.parse.urlencode({
        "Act": "02", "Sub": "1", "meet": meet,
        "rcDate": race_date, "rcNo": str(int(race_number)),
    }).encode()
    request = urllib.request.Request(ENDPOINT, data=data, headers={"User-Agent": "RaceLens-research/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        page = response.read().decode("euc-kr", errors="replace")
    rows = parse_health_page(page, meet, race_date, race_number)
    if not rows:
        raise ValueError(f"no health rows for {meet}/{race_date}/{race_number}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"rows": rows}, ensure_ascii=False) + "\n", encoding="utf-8")
    return True, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--archive", type=Path, default=Path("/Users/tttksj/kra/data/health_archive"))
    parser.add_argument("--from-date", default="20240101")
    parser.add_argument("--until-date", default="20260711")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    keys = race_keys(args.db, args.from_date, args.until_date)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda key: collect(args.archive, key), keys))
    print(f"races={len(keys)} downloaded={sum(created for created, _ in results)} rows={sum(rows for _, rows in results)} archive={args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
