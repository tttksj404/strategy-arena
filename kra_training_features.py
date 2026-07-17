from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing SQLite-to-scikit training contract

from kra_history_features import HORSE_HISTORY_FEATURES, add_horse_history_features
from kra_sectional_features import SECTIONAL_RAW_COLUMNS


REL = (
    "wgBudam", "wgHr_base", "rating", "age", "jk_wr_prior", "tr_wr_prior",
    "hr_win_prior", "hr_place_prior", "hr_finish_prior",
)
NUM = (
    "wgBudam", "wgHr_base", "age", "rating", "rcDist", "chulNo",
    "field_size", "jk_wr_prior", "tr_wr_prior",
)


def _base_weight(value: str) -> float:
    match = re.match(r"\s*(\d+)", str(value or ""))
    return float(match.group(1)) if match else float("nan")


def load_rows(db_path: Path) -> pd.DataFrame:
    columns = (
        "meet,rcDate,rcNo,chulNo,hrNo,ord,winOdds,plcOdds,budam,wgBudam,"
        "wgHr,age,sex,rating,rcDist,jkNo,trNo,owNo,wgJk,hrTool"
        ",rcTime,buS1fOrd,sjS1fOrd,rank,weather,track"
        "," + ",".join(SECTIONAL_RAW_COLUMNS)
    )
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(f"SELECT {columns} FROM race_result", connection)
    numeric = (
        "ord", "winOdds", "plcOdds", "wgBudam", "age", "rating", "rcDist",
        "chulNo", "rcTime", "buS1fOrd", "sjS1fOrd",
        *SECTIONAL_RAW_COLUMNS,
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["ord"].between(1, 20)].copy()
    frame = frame[(frame["winOdds"] > 0) & (frame["plcOdds"] > 0)].copy()
    frame["wgHr_base"] = frame["wgHr"].map(_base_weight)
    frame["rk"] = frame["meet"].astype(str) + "|" + frame["rcDate"].astype(str) + "|" + frame["rcNo"].astype(str)
    frame["field_size"] = frame.groupby("rk")["rk"].transform("size")
    frame = frame[frame["field_size"] >= 2].copy()
    frame["win"] = (frame["ord"] == 1).astype(int)
    frame["place"] = np.where(frame["field_size"] < 8, frame["ord"] <= 2, frame["ord"] <= 3).astype(int)
    return frame


def _expanding_prior(frame: pd.DataFrame, id_column: str) -> pd.Series:
    daily = (
        frame.groupby([id_column, "rcDate"], dropna=False)["win"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values([id_column, "rcDate"])
    )
    grouped = daily.groupby(id_column, dropna=False)
    prior_wins = grouped["sum"].cumsum() - daily["sum"]
    prior_runs = grouped["count"].cumsum() - daily["count"]
    daily["prior"] = (prior_wins + 0.1) / (prior_runs + 1.0)
    lookup = daily.set_index([id_column, "rcDate"])["prior"]
    values = [lookup.get((identifier, date), 0.1) for identifier, date in zip(frame[id_column], frame["rcDate"])]
    return pd.Series(values, index=frame.index, dtype=float)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = add_horse_history_features(frame)
    result["jk_wr_prior"] = _expanding_prior(result, "jkNo")
    result["tr_wr_prior"] = _expanding_prior(result, "trNo")
    for column in REL:
        result[f"{column}_rel"] = result[column] - result.groupby("rk")[column].transform("mean")
    for category, prefix in (("sex", "sex"), ("budam", "bd")):
        result = pd.concat(
            [result, pd.get_dummies(result[category].fillna("NA"), prefix=prefix)],
            axis=1,
        )
    feature_columns = list(NUM) + [f"{column}_rel" for column in REL]
    feature_columns.extend(column for column in HORSE_HISTORY_FEATURES if column not in feature_columns)
    feature_columns.extend(
        column for column in result.columns if column.startswith("sex_") or column.startswith("bd_")
    )
    return result, feature_columns
