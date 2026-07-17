from __future__ import annotations

import re
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing chronological feature-frame contract

from kra_dynamic_features import build_dynamic_features


CONTEXT_HISTORY_FEATURES: Final = (
    "hr_race_speed_rel_mean_3",
    "hr_same_distance_finish_prior",
    "jk_recent_win_50",
    "tr_recent_win_50",
)
CONTEXT_CURRENT_FEATURES: Final = (
    "race_class_level",
    "track_moisture",
    "race_month",
)


def _map_values(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    keys: list[str],
    value: str,
) -> pd.Series:
    lookup = daily.set_index(keys)[value]
    row_keys = list(zip(*(frame[key] for key in keys)))
    return pd.Series([lookup.get(key, np.nan) for key in row_keys], index=frame.index)


def _recent_participant_win_rate(
    frame: pd.DataFrame,
    participant: str,
) -> pd.Series:
    daily = (
        frame.groupby([participant, "rcDate"], dropna=False)["win"]
        .agg([("wins", "sum"), ("runs", "count")])
        .reset_index()
        .sort_values([participant, "rcDate"])
    )
    grouped = daily.groupby(participant, dropna=False)
    prior_wins = grouped["wins"].transform(
        lambda values: values.shift(1).rolling(50, min_periods=1).sum()
    )
    prior_runs = grouped["runs"].transform(
        lambda values: values.shift(1).rolling(50, min_periods=1).sum()
    )
    daily["prior"] = (prior_wins + 0.1) / (prior_runs + 1.0)
    return _map_values(frame, daily, [participant, "rcDate"], "prior").fillna(0.1)


def _context_dummies(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    rank_group = result["rank"].fillna("NA").astype(str).str.extract(r"^([^0-9]+)")[0]
    encoded = pd.get_dummies(
        pd.DataFrame({
            "meet": result["meet"].fillna("NA"),
            "weather": result["weather"].fillna("NA"),
            "rank_group": rank_group.fillna("NA"),
        }),
        prefix=("meet", "weather", "rank_group"),
    )
    result = pd.concat([result, encoded], axis=1)
    return result, list(encoded.columns)


def build_contextual_features(
    frame: pd.DataFrame,
    baseline_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    result, dynamic_columns = build_dynamic_features(frame, baseline_columns)
    distance = pd.to_numeric(result["rcDist"], errors="coerce")
    race_time = pd.to_numeric(result["rcTime"], errors="coerce")
    speed = (distance / race_time).where(race_time > 0)
    result["_race_speed_rel"] = speed - speed.groupby(result["rk"]).transform("median")
    speed_daily = (
        result.groupby(["hrNo", "rcDate"], dropna=False)["_race_speed_rel"]
        .mean()
        .rename("speed_rel")
        .reset_index()
        .sort_values(["hrNo", "rcDate"])
    )
    speed_daily["prior"] = speed_daily.groupby("hrNo", dropna=False)["speed_rel"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    result["hr_race_speed_rel_mean_3"] = _map_values(
        result, speed_daily, ["hrNo", "rcDate"], "prior"
    )
    distance_daily = (
        result.groupby(["hrNo", "rcDist", "rcDate"], dropna=False)["finish_pct"]
        .mean()
        .rename("finish_pct")
        .reset_index()
        .sort_values(["hrNo", "rcDist", "rcDate"])
    )
    distance_daily["prior"] = distance_daily.groupby(
        ["hrNo", "rcDist"], dropna=False
    )["finish_pct"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    result["hr_same_distance_finish_prior"] = _map_values(
        result, distance_daily, ["hrNo", "rcDist", "rcDate"], "prior"
    )
    result["jk_recent_win_50"] = _recent_participant_win_rate(result, "jkNo")
    result["tr_recent_win_50"] = _recent_participant_win_rate(result, "trNo")
    result["race_class_level"] = pd.to_numeric(
        result["rank"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    result["track_moisture"] = pd.to_numeric(
        result["track"].astype(str).str.extract(r"(\d+(?:\.\d+)?)%")[0], errors="coerce"
    )
    result["race_month"] = pd.to_numeric(result["rcDate"].astype(str).str[4:6], errors="coerce")
    for column in CONTEXT_HISTORY_FEATURES:
        result[f"{column}_rel"] = result[column] - result.groupby("rk")[column].transform("mean")
    result, dummy_columns = _context_dummies(result)
    columns = [
        *dynamic_columns,
        *CONTEXT_HISTORY_FEATURES,
        *(f"{column}_rel" for column in CONTEXT_HISTORY_FEATURES),
        *CONTEXT_CURRENT_FEATURES,
        *dummy_columns,
    ]
    return result.drop(columns=["_race_speed_rel"]), list(dict.fromkeys(columns))
