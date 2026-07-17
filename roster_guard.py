from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Final, TypedDict


DEFAULT_KRA_URL = "https://apis.data.go.kr/B551015/API214_1/RaceDetailResult_1"
KCYCLE_BASE_URL: Final = "https://www.kcycle.or.kr"
KCYCLE_DECISION_PATH: Final = "/race/card/decision"
KRA_MEET_CODE = {"서울": "1", "제주": "2", "부경": "3"}
VERIFY_TTL_SEC = 1800
NEGATIVE_TTL_SEC = 60
KCYCLE_TMS_CACHE_TTL_SEC = 12 * 60 * 60
KCYCLE_FETCH_TIMEOUT_SEC = 0.75


class RosterVerification(TypedDict):
    state: str
    official_names: list[str]
    checked_at: str


class _CellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._in_row = False
        self._in_cell = False
        self._row: list[str] = []
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._in_row = True
            self._row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            self._row.append(re.sub(r"\s+", " ", "".join(self._parts)).strip())
            self._parts = []
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if any(self._row):
                self.rows.append(self._row)
            self._row = []
            self._in_row = False


_CACHE: dict[tuple[str, str, str, str], tuple[RosterVerification, float, int]] = {}
_KCYCLE_TMS_CACHE: dict[int, tuple[dict[str, tuple[int, int]], float]] = {}


def _checked_at() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _starter_name(starter: dict, sport: str) -> str:
    key = "hrName" if sport == "horse" else "racer_nm"
    return str(starter.get(key) or "").strip()


def _candidate_names_from_starters(starters: list[dict], sport: str) -> list[str]:
    names: list[str] = []
    for starter in starters:
        name = _starter_name(starter, sport)
        if name:
            names.append(name)
    return names


def _fetch_kcycle_html(url: str, timeout: float = KCYCLE_FETCH_TIMEOUT_SEC) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "strategy-arena"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _select_block(html: str, select_id: str) -> str:
    pattern = re.compile(
        rf"<select\b[^>]*(?:id|name)=[\"']{re.escape(select_id)}[\"'][^>]*>.*?</select>",
        re.DOTALL,
    )
    match = pattern.search(html or "")
    return match.group(0) if match else ""


def _strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def _kcycle_year_options(html: str) -> set[int]:
    block = _select_block(html, "stndYear")
    return {int(value) for value in re.findall(r"<option\b[^>]*value=[\"'](\d{4})[\"']", block)}


def _selected_kcycle_year(html: str) -> int | None:
    block = _select_block(html, "stndYear")
    if not block:
        return None
    selected = re.search(r"<option\b[^>]*value=[\"'](\d{4})[\"'][^>]*selected", block)
    first = selected or re.search(r"<option\b[^>]*value=[\"'](\d{4})[\"']", block)
    return int(first.group(1)) if first else None


def _parse_kcycle_tmsdayord_options(html: str, year: int) -> dict[str, tuple[int, int]]:
    year_options = _kcycle_year_options(html)
    if year_options and year not in year_options:
        return {}
    block = _select_block(html, "tmsDayOrd") or html
    mapping: dict[str, tuple[int, int]] = {}
    option_pattern = re.compile(
        r"<option\b[^>]*value=[\"'](\d+)-(\d+)[\"'][^>]*>(.*?)</option>",
        re.DOTALL,
    )
    for match in option_pattern.finditer(block):
        month_day = re.search(r"(\d{2})월\s*(\d{2})일", _strip_tags(match.group(3)))
        if not month_day:
            continue
        month = int(month_day.group(1))
        day = int(month_day.group(2))
        try:
            race_day = dt.date(year, month, day)
        except ValueError:
            continue
        mapping[race_day.isoformat()] = (int(match.group(1)), int(match.group(2)))
    return mapping


def _kcycle_tmsdayord_mapping(year: int) -> dict[str, tuple[int, int]]:
    cached = _KCYCLE_TMS_CACHE.get(year)
    now = time.time()
    if cached and now - cached[1] < KCYCLE_TMS_CACHE_TTL_SEC:
        return cached[0]
    base_html = _fetch_kcycle_html(f"{KCYCLE_BASE_URL}{KCYCLE_DECISION_PATH}")
    selected_year = _selected_kcycle_year(base_html)
    if selected_year is None or selected_year == year:
        mapping = _parse_kcycle_tmsdayord_options(base_html, year)
    elif year in _kcycle_year_options(base_html):
        try:
            mapping = _parse_kcycle_tmsdayord_options(
                _fetch_kcycle_html(f"{KCYCLE_BASE_URL}{KCYCLE_DECISION_PATH}/tmsDayOrd/{year}"),
                year,
            )
        except (OSError, TimeoutError, urllib.error.URLError):
            mapping = {}
    else:
        mapping = {}
    _KCYCLE_TMS_CACHE[year] = (mapping, now)
    return mapping


def _kcycle_official_names(ymd: str, meet: str, race_no: str) -> list[str] | None:
    digits = re.sub(r"\D", "", str(ymd or ""))
    if len(digits) != 8:
        return None
    year = int(digits[:4])
    race_day = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    try:
        tms_day = _kcycle_tmsdayord_mapping(year).get(race_day)
        if not tms_day:
            return None
        html = _fetch_kcycle_html(f"{KCYCLE_BASE_URL}{KCYCLE_DECISION_PATH}/{year}/{tms_day[0]}/{tms_day[1]}")
    except (OSError, TimeoutError, urllib.error.URLError):
        return None
    names = _extract_kcycle_names(html, meet, race_no)
    if names:
        return names
    return None


def _extract_kcycle_names(html: str, meet: str, race_no: str) -> list[str]:
    try:
        rno = int(str(race_no).strip())
    except ValueError:
        return []
    header = re.compile(rf"{re.escape(str(meet).strip())}\s*{rno:02d}경주")
    match = header.search(html or "")
    if not match:
        return []
    section_end = (html or "").find('class="cptLinkSection"', match.end())
    section = (html or "")[match.start(): section_end if section_end > -1 else None]
    parser = _CellParser()
    parser.feed(section)
    ordered: dict[int, str] = {}
    racer_pattern = re.compile(r"(?<!\d)([1-9])\s*([가-힣](?:\s?[가-힣]){1,3})\s+\d{1,2}기")
    for row in parser.rows:
        row_text = " ".join(row)
        racer = racer_pattern.search(row_text)
        if racer:
            ordered[int(racer.group(1))] = _normalize_name(racer.group(2))
    return [ordered[bno] for bno in sorted(ordered)]


def _data_key() -> str | None:
    for name in ("DATAGOKR_SERVICE_KEY", "datagokr", "DATAGOKR", "DATA_GO_KR_KEY", "SERVICE_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _kra_official_names(ymd: str, meet: str, race_no: str) -> list[str] | None:
    key = _data_key()
    if not key:
        return None
    rc_date = re.sub(r"\D", "", str(ymd or ""))
    if len(rc_date) != 8:
        return None
    rno = str(race_no).strip().lstrip("0") or "0"
    url = os.environ.get("KRA_CARD_URL", DEFAULT_KRA_URL)
    candidates = [meet, KRA_MEET_CODE.get(meet, meet)]
    for meet_value in candidates:
        query = urllib.parse.urlencode({
            "serviceKey": key,
            "_type": "json",
            "numOfRows": 50,
            "pageNo": 1,
            "meet": str(meet_value),
            "rc_date": rc_date,
            "rc_no": rno,
        })
        request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "strategy-arena"})
        try:
            with urllib.request.urlopen(request, timeout=6) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            continue
        body = (payload.get("response") or {}).get("body") or {}
        total = int(body.get("totalCount") or 0)
        items_container = body.get("items") if isinstance(body, dict) else {}
        items = items_container.get("item", []) if isinstance(items_container, dict) else []
        if isinstance(items, dict):
            items = [items]
        if total <= 0 or not items:
            continue
        names = [str(item.get("hrName") or "").strip() for item in items if str(item.get("hrName") or "").strip()]
        if names:
            return names
    return None


def _provider_names(sport: str, ymd: str, meet: str, race_no: str) -> list[str] | None:
    match sport:
        case "keirin":
            return _kcycle_official_names(ymd, meet, race_no)
        case "horse":
            return _kra_official_names(ymd, meet, race_no)
        case _:
            return None


def verify_roster(
    sport: str,
    ymd: str,
    meet: str,
    race_no: str,
    starters: list[dict],
) -> RosterVerification:
    key = (str(sport), re.sub(r"\D", "", str(ymd or ""))[:8], str(meet), str(race_no).strip())
    now = time.time()
    cached = _CACHE.get(key)
    if cached:
        result, cached_at, ttl = cached
        if now - cached_at < ttl:
            return result

    official_names = _provider_names(sport, ymd, meet, race_no) or []
    if not official_names:
        result: RosterVerification = {"state": "unverified", "official_names": [], "checked_at": _checked_at()}
        _CACHE[key] = (result, now, NEGATIVE_TTL_SEC)
        return result

    official_set = {_normalize_name(name) for name in official_names if _normalize_name(name)}
    starter_set = {_normalize_name(name) for name in _candidate_names_from_starters(starters, sport)}
    mismatch_count = len(official_set.symmetric_difference(starter_set))
    state = "mismatch" if mismatch_count >= 2 else "verified"
    result = {"state": state, "official_names": official_names, "checked_at": _checked_at()}
    _CACHE[key] = (result, now, VERIFY_TTL_SEC)
    return result


def clear_cache() -> None:
    _CACHE.clear()
    _KCYCLE_TMS_CACHE.clear()
