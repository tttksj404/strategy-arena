from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — shared chronological model-frame contract


HEALTH_FEATURES: Final = (
    "health_treatments_30d",
    "health_treatments_90d",
    "health_locomotor_90d",
    "health_respiratory_90d",
    "health_fatigue_90d",
    "health_vaccine_90d",
    "health_days_since_treatment",
)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_TREATMENT = re.compile(r'title="상세보기">\s*(\d{4}/\d{2}/\d{2})\s+([^<]+)')
_MEET_CODES: Final = {"서울": "1", "제주": "2", "부산경남": "3", "부경": "3"}


def _text(fragment: str) -> str:
    return re.sub(r"\s+", "", html.unescape(_TAG.sub("", fragment)))


def parse_health_page(
    page: str,
    meet: str,
    race_date: str,
    race_number: str,
) -> list[dict[str, object]]:
    rows = []
    for fragment in _ROW.findall(page):
        cells = _CELL.findall(fragment)
        if len(cells) != 4 or not _text(cells[0]).isdigit():
            continue
        treatments = [
            {"date": date.replace("/", ""), "name": html.unescape(name).strip()}
            for date, name in _TREATMENT.findall(cells[2])
            if date.replace("/", "") < race_date
        ]
        rows.append({
            "meet": str(meet),
            "rcDate": str(race_date),
            "rcNo": str(int(race_number)),
            "chulNo": str(int(_text(cells[0]))),
            "hrName_health": _text(cells[1]),
            "health_treatments": treatments,
        })
    return rows


def load_health(archive: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(archive.rglob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    return pd.DataFrame(rows)


def _days_between(race_date: str, treatment_date: str) -> int:
    race = dt.datetime.strptime(race_date, "%Y%m%d").date()
    treatment = dt.datetime.strptime(treatment_date, "%Y%m%d").date()
    return (race - treatment).days


def add_health_features(
    frame: pd.DataFrame,
    health: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    keys = ["meet", "rcDate", "rcNo", "chulNo"]
    result = frame.copy()
    result["meet"] = result["meet"].astype(str).replace(_MEET_CODES)
    for key in keys:
        result[key] = result[key].astype(str).str.lstrip("0").replace("", "0")
    if health.empty:
        for feature in HEALTH_FEATURES:
            result[feature] = np.nan
    else:
        source = health.copy()
        source["meet"] = source["meet"].astype(str).replace(_MEET_CODES)
        for key in keys:
            source[key] = source[key].astype(str).str.lstrip("0").replace("", "0")
        result = result.merge(source.drop_duplicates(keys, keep="last"), on=keys, how="left", validate="many_to_one")
        treatments = result["health_treatments"].map(lambda value: value if isinstance(value, list) else [])
        dated = [
            [(_days_between(race_date, item["date"]), item["name"]) for item in items]
            for race_date, items in zip(result["rcDate"], treatments)
        ]
        locomotor = ("파행", "근육", "염좌", "교돌상", "건염", "골절", "절음", "관절")
        respiratory = ("감기", "기관지", "비출혈", "폐출혈", "호흡")
        result["health_treatments_30d"] = [sum(days <= 30 for days, _ in items) for items in dated]
        result["health_treatments_90d"] = [sum(days <= 90 for days, _ in items) for items in dated]
        result["health_locomotor_90d"] = [sum(days <= 90 and any(word in name for word in locomotor) for days, name in items) for items in dated]
        result["health_respiratory_90d"] = [sum(days <= 90 and any(word in name for word in respiratory) for days, name in items) for items in dated]
        result["health_fatigue_90d"] = [sum(days <= 90 and "피로" in name for days, name in items) for items in dated]
        result["health_vaccine_90d"] = [sum(days <= 90 and "예방접종" in name for days, name in items) for items in dated]
        result["health_days_since_treatment"] = [min((days for days, _ in items), default=np.nan) for items in dated]
    for feature in HEALTH_FEATURES:
        result[f"{feature}_rel"] = result[feature] - result.groupby("rk")[feature].transform("mean")
    columns = [*HEALTH_FEATURES, *(f"{feature}_rel" for feature in HEALTH_FEATURES)]
    return result, columns
