from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final


LIST_URL: Final = "https://race.kra.co.kr/dbdata/textDataList.do"
DOWNLOAD_URL: Final = "https://race.kra.co.kr/dbdata/fileDownLoad.do"
MEETS: Final = {"1": "seoul", "2": "jeju", "3": "busan"}
LINK = re.compile(
    rb'href="/dbdata/fileDownLoad\.do\?fn=([^&"]+/\d{8}dacom01\.rpt)&(?:amp;)?meet=(\d)"'
)


def _request(url: str, data: bytes | None = None) -> bytes:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "RaceLens-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def listing_links(meet: str, year: int, page: int) -> set[tuple[str, str]]:
    body = urllib.parse.urlencode({
        "Act": "12",
        "Sub": "1",
        "meet": meet,
        "fileType": "dacom01",
        "codeName": "출전표",
        "loginAuth": "true",
        "dbFlag": "true",
        "dbFlag3": "true",
        "fileSearchName": str(year),
        "pageIndex": str(page),
    }).encode()
    return {(path.decode(), found_meet.decode()) for path, found_meet in LINK.findall(_request(LIST_URL, body))}


def discover(meet: str, year: int) -> set[tuple[str, str]]:
    links: set[tuple[str, str]] = set()
    for page in range(1, 8):
        found = listing_links(meet, year, page)
        if not found:
            break
        links.update(found)
        if len(found) < 10:
            break
    return links


def download(target_root: Path, link: tuple[str, str]) -> tuple[str, bool]:
    path, meet = link
    target = target_root / MEETS[meet] / Path(path).name
    if target.exists() and target.stat().st_size > 100:
        return str(target), False
    target.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"fn": path, "meet": meet})
    payload = _request(f"{DOWNLOAD_URL}?{query}")
    if len(payload) < 100:
        raise ValueError(f"empty KRA archive response: {path}")
    target.write_bytes(payload)
    return str(target), True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("/Users/tttksj/kra/data/entry_sheet_archive"))
    parser.add_argument("--from-year", type=int, default=2024)
    parser.add_argument("--until-year", type=int, default=2026)
    args = parser.parse_args()
    links = set()
    for meet in MEETS:
        for year in range(args.from_year, args.until_year + 1):
            links.update(discover(meet, year))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda item: download(args.archive, item), sorted(links)))
    print(f"discovered={len(links)} downloaded={sum(created for _, created in results)} archive={args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
