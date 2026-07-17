from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — shared chronological model-frame contract


CONDITIONING_FEATURES: Final = (
    "condition_sessions_2w",
    "condition_minutes_2w",
    "condition_sessions_current",
    "condition_minutes_current",
    "condition_sessions_previous",
    "condition_minutes_previous",
    "condition_average_minutes",
    "condition_max_minutes",
    "condition_caretaker_share",
    "condition_current_change",
)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_MINUTES = re.compile(r"(\d+)\s*$")
_MEET_CODES: Final = {"서울": "1", "제주": "2", "부산경남": "3", "부경": "3"}


def _text(fragment: str) -> str:
    return re.sub(r"\s+", "", html.unescape(_TAG.sub("", fragment)))


def parse_conditioning_page(
    page: str,
    meet: str,
    race_date: str,
    race_number: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fragment in _ROW.findall(page):
        cells = [_text(cell) for cell in _CELL.findall(fragment)]
        if len(cells) != 14 or not cells[0].isdigit():
            continue
        sessions = cells[2:]
        minutes = [int(match.group(1)) if (match := _MINUTES.search(value)) else 0 for value in sessions]
        rows.append({
            "meet": str(meet),
            "rcDate": str(race_date),
            "rcNo": str(int(race_number)),
            "chulNo": str(int(cells[0])),
            "hrName_condition": cells[1],
            "condition_cells": sessions,
            "condition_minutes": minutes,
        })
    return rows


def load_conditioning(archive: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(archive.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return pd.DataFrame(rows)


def add_conditioning_features(
    frame: pd.DataFrame,
    conditioning: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    keys = ["meet", "rcDate", "rcNo", "chulNo"]
    result = frame.copy()
    result["meet"] = result["meet"].astype(str).replace(_MEET_CODES)
    for key in keys:
        result[key] = result[key].astype(str).str.lstrip("0").replace("", "0")
    if conditioning.empty:
        for feature in CONDITIONING_FEATURES:
            result[feature] = np.nan
    else:
        source = conditioning.copy()
        source["meet"] = source["meet"].astype(str).replace(_MEET_CODES)
        for key in keys:
            source[key] = source[key].astype(str).str.lstrip("0").replace("", "0")
        source = source.drop_duplicates(keys, keep="last")
        result = result.merge(source, on=keys, how="left", validate="many_to_one")
        all_minutes = result["condition_minutes"].map(lambda value: value if isinstance(value, list) else [])
        all_cells = result["condition_cells"].map(lambda value: value if isinstance(value, list) else [])
        result["condition_sessions_2w"] = all_minutes.map(lambda values: sum(item > 0 for item in values))
        result["condition_minutes_2w"] = all_minutes.map(sum)
        result["condition_sessions_previous"] = all_minutes.map(lambda values: sum(item > 0 for item in values[:6]))
        result["condition_minutes_previous"] = all_minutes.map(lambda values: sum(values[:6]))
        result["condition_sessions_current"] = all_minutes.map(lambda values: sum(item > 0 for item in values[6:]))
        result["condition_minutes_current"] = all_minutes.map(lambda values: sum(values[6:]))
        result["condition_average_minutes"] = result["condition_minutes_2w"] / result["condition_sessions_2w"].where(result["condition_sessions_2w"] > 0)
        result["condition_max_minutes"] = all_minutes.map(lambda values: max(values, default=0))
        caretaker = all_cells.map(lambda values: sum(value.startswith("관") for value in values if value))
        result["condition_caretaker_share"] = caretaker / result["condition_sessions_2w"].where(result["condition_sessions_2w"] > 0)
        result["condition_current_change"] = result["condition_minutes_current"] - result["condition_minutes_previous"]
    for feature in CONDITIONING_FEATURES:
        result[f"{feature}_rel"] = result[feature] - result.groupby("rk")[feature].transform("mean")
    columns = [*CONDITIONING_FEATURES, *(f"{feature}_rel" for feature in CONDITIONING_FEATURES)]
    return result, columns
