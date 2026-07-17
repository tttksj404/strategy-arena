from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — shared chronological model-frame contract


MEET_DIRS: Final = {"1": "seoul", "2": "jeju", "3": "busan"}
ENTRY_FEATURES: Final = (
    "entry_career_win_rate",
    "entry_career_place_rate",
    "entry_year_win_rate",
    "entry_year_place_rate",
    "entry_career_earn_per_start",
    "entry_year_earn_per_start",
    "entry_six_month_earn",
    "entry_form_acceleration",
)
_TITLE = re.compile(r"제목\s*:\s*(\d{2})\.(\d{2})\.(\d{2}).*?(\d+)경주")
_STATS = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)"
    r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)
_MEET_CODES: Final = {"서울": "1", "제주": "2", "부산경남": "3", "부경": "3"}


def _normalize_meet(series: pd.Series) -> pd.Series:
    return series.astype(str).replace(_MEET_CODES)


def parse_entry_sheet(text: str, meet: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    race_date = ""
    race_number = ""
    in_stats = False
    for line in text.splitlines():
        if title := _TITLE.search(line):
            race_date = f"20{title.group(1)}{title.group(2)}{title.group(3)}"
            race_number = title.group(4)
            in_stats = False
            continue
        if "--총전적--" in line:
            in_stats = True
            continue
        if not in_stats or not race_date:
            continue
        match = _STATS.match(line)
        if not match:
            continue
        values = [int(value.replace(",", "")) for value in match.groups()[2:]]
        rows.append({
            "meet": str(meet),
            "rcDate": race_date,
            "rcNo": str(int(race_number)),
            "chulNo": str(int(match.group(1))),
            "hrName_entry": re.sub(r"\s+", "", match.group(2)),
            "entry_career_earn": values[0],
            "entry_year_earn": values[1],
            "entry_six_month_earn": values[2],
            "entry_career_wins": values[3],
            "entry_career_seconds": values[4],
            "entry_career_thirds": values[5],
            "entry_career_starts": values[6],
            "entry_year_wins": values[7],
            "entry_year_seconds": values[8],
            "entry_year_thirds": values[9],
            "entry_year_starts": values[10],
        })
    return rows


def load_entry_sheets(archive: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for meet, directory in MEET_DIRS.items():
        for path in sorted((archive / directory).glob("*.rpt")):
            rows.extend(parse_entry_sheet(path.read_text(encoding="euc-kr"), meet))
    return pd.DataFrame(rows)


def add_entry_sheet_features(
    frame: pd.DataFrame,
    entries: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    keys = ["meet", "rcDate", "rcNo", "chulNo"]
    result["meet"] = _normalize_meet(result["meet"])
    for key in keys:
        result[key] = result[key].astype(str).str.lstrip("0").replace("", "0")
    if entries.empty:
        for feature in ENTRY_FEATURES:
            result[feature] = np.nan
    else:
        source = entries.copy()
        source["meet"] = _normalize_meet(source["meet"])
        for key in keys:
            source[key] = source[key].astype(str).str.lstrip("0").replace("", "0")
        source = source.drop_duplicates(keys, keep="last")
        result = result.merge(source, on=keys, how="left", validate="many_to_one")
        career_starts = pd.to_numeric(result["entry_career_starts"], errors="coerce")
        year_starts = pd.to_numeric(result["entry_year_starts"], errors="coerce")
        result["entry_career_win_rate"] = result["entry_career_wins"] / career_starts.where(career_starts > 0)
        result["entry_career_place_rate"] = (
            result["entry_career_wins"] + result["entry_career_seconds"] + result["entry_career_thirds"]
        ) / career_starts.where(career_starts > 0)
        result["entry_year_win_rate"] = result["entry_year_wins"] / year_starts.where(year_starts > 0)
        result["entry_year_place_rate"] = (
            result["entry_year_wins"] + result["entry_year_seconds"] + result["entry_year_thirds"]
        ) / year_starts.where(year_starts > 0)
        result["entry_career_earn_per_start"] = result["entry_career_earn"] / career_starts.where(career_starts > 0)
        result["entry_year_earn_per_start"] = result["entry_year_earn"] / year_starts.where(year_starts > 0)
        result["entry_form_acceleration"] = result["entry_year_win_rate"] - result["entry_career_win_rate"]
    for feature in ENTRY_FEATURES:
        result[f"{feature}_rel"] = result[feature] - result.groupby("rk")[feature].transform("mean")
    columns = [*ENTRY_FEATURES, *(f"{feature}_rel" for feature in ENTRY_FEATURES)]
    return result, columns
