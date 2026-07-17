from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing chronological feature-frame contract


DYNAMIC_HISTORY_FEATURES: Final = (
    "hr_speed_last",
    "hr_speed_mean_3",
    "hr_speed_mean_5",
    "hr_speed_best_5",
    "hr_early_position_mean_3",
    "hr_finish_gain_mean_3",
    "hr_recent_finish_mean_3",
    "hr_body_weight_delta",
    "hr_burden_delta",
    "hr_distance_delta",
)
DYNAMIC_RELATIVE_FEATURES: Final = tuple(
    f"{column}_rel" for column in DYNAMIC_HISTORY_FEATURES
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def add_dynamic_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    distance = _numeric(result, "rcDist")
    race_time = _numeric(result, "rcTime")
    field_size = _numeric(result, "field_size")
    finish = _numeric(result, "ord")
    busan_early = _numeric(result, "buS1fOrd")
    other_early = _numeric(result, "sjS1fOrd")
    early = busan_early.where(busan_early > 0, other_early)
    result["_speed"] = (distance / race_time).where(race_time > 0)
    result["_finish_pct"] = ((finish - 1.0) / (field_size - 1.0)).where(field_size > 1)
    result["_early_pct"] = ((early - 1.0) / (field_size - 1.0)).where(
        (early > 0) & (field_size > 1)
    )
    result["_finish_gain"] = result["_early_pct"] - result["_finish_pct"]
    daily = (
        result.groupby(["hrNo", "rcDate"], dropna=False)
        .agg(
            speed=("_speed", "mean"),
            early_pct=("_early_pct", "mean"),
            finish_gain=("_finish_gain", "mean"),
            finish_pct=("_finish_pct", "mean"),
            body_weight=("wgHr_base", "mean"),
            burden=("wgBudam", "mean"),
            distance=("rcDist", "mean"),
        )
        .reset_index()
        .sort_values(["hrNo", "rcDate"])
    )
    grouped = daily.groupby("hrNo", dropna=False)
    daily["hr_speed_last"] = grouped["speed"].shift(1)
    daily["hr_speed_mean_3"] = grouped["speed"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    daily["hr_speed_mean_5"] = grouped["speed"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    daily["hr_speed_best_5"] = grouped["speed"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).max()
    )
    daily["hr_early_position_mean_3"] = grouped["early_pct"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    daily["hr_finish_gain_mean_3"] = grouped["finish_gain"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    daily["hr_recent_finish_mean_3"] = grouped["finish_pct"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    daily["hr_body_weight_delta"] = daily["body_weight"] - grouped["body_weight"].shift(1)
    daily["hr_burden_delta"] = daily["burden"] - grouped["burden"].shift(1)
    daily["hr_distance_delta"] = daily["distance"] - grouped["distance"].shift(1)
    keys = list(zip(result["hrNo"], result["rcDate"]))
    indexed = daily.set_index(["hrNo", "rcDate"])
    for column in DYNAMIC_HISTORY_FEATURES:
        lookup = indexed[column]
        result[column] = [lookup.get(key, np.nan) for key in keys]
    return result.drop(columns=["_speed", "_finish_pct", "_early_pct", "_finish_gain"])


def build_dynamic_features(
    frame: pd.DataFrame,
    baseline_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    result = add_dynamic_history_features(frame)
    for column in DYNAMIC_HISTORY_FEATURES:
        result[f"{column}_rel"] = result[column] - result.groupby("rk")[column].transform("mean")
    columns = [*baseline_columns, *DYNAMIC_HISTORY_FEATURES, *DYNAMIC_RELATIVE_FEATURES]
    return result, list(dict.fromkeys(columns))
