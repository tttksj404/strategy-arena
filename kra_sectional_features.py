from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing chronological feature-frame contract


SECTIONAL_RAW_COLUMNS: Final = (
    "buS1fTime",
    "buG3fAccTime",
    "seS1fAccTime",
    "seG3fAccTime",
    "jeS1fTime",
    "jeG3fTime",
)
SECTIONAL_HISTORY_FEATURES: Final = (
    "hr_sectional_first200_last",
    "hr_sectional_first200_mean_4",
    "hr_sectional_last600_last",
    "hr_sectional_last600_mean_4",
    "hr_sectional_late_adv_mean_4",
    "hr_sectional_late_adv_best_4",
    "hr_sectional_late_adv_std_4",
    "hr_sectional_early_adv_mean_4",
    "hr_sectional_finish_speed_pct_mean_4",
    "hr_sectional_pace_balance_mean_4",
    "hr_sectional_history_count",
)
SECTIONAL_RELATIVE_FEATURES: Final = tuple(
    f"{column}_rel" for column in SECTIONAL_HISTORY_FEATURES
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _venue_sectionals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    meet = frame["meet"].astype(str)
    total = _numeric(frame, "rcTime")
    first200 = _numeric(frame, "seS1fAccTime").where(meet == "서울")
    first200 = first200.fillna(_numeric(frame, "buS1fTime").where(meet == "부경"))
    first200 = first200.fillna(_numeric(frame, "jeS1fTime").where(meet == "제주"))

    last600 = (total - _numeric(frame, "seG3fAccTime")).where(meet == "서울")
    last600 = last600.fillna(
        (total - _numeric(frame, "buG3fAccTime")).where(meet == "부경")
    )
    last600 = last600.fillna(_numeric(frame, "jeG3fTime").where(meet == "제주"))
    return first200.where(first200.between(8.0, 30.0)), last600.where(
        last600.between(20.0, 80.0)
    )


def _history_lookup(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    column: str,
) -> pd.Series:
    lookup = daily.set_index(["hrNo", "rcDate"])[column]
    keys = list(zip(frame["hrNo"], frame["rcDate"]))
    return pd.Series([lookup.get(key, np.nan) for key in keys], index=frame.index)


def add_sectional_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    first200, last600 = _venue_sectionals(result)
    distance = _numeric(result, "rcDist")
    total = _numeric(result, "rcTime")
    first200_median = first200.groupby(result["rk"]).transform("median")
    last600_median = last600.groupby(result["rk"]).transform("median")
    result["_sectional_first200"] = first200
    result["_sectional_last600"] = last600
    result["_sectional_early_adv"] = (
        (first200_median - first200) / first200_median * 100.0
    )
    result["_sectional_late_adv"] = (
        (last600_median - last600) / last600_median * 100.0
    )
    result["_sectional_finish_speed_pct"] = (
        (600.0 / last600) / (distance / total) * 100.0
    ).where((distance >= 600.0) & (total > 0.0))
    result["_sectional_pace_balance"] = (
        result["_sectional_late_adv"] - result["_sectional_early_adv"]
    )

    daily = (
        result.groupby(["hrNo", "rcDate"], dropna=False)
        .agg(
            first200=("_sectional_first200", "mean"),
            last600=("_sectional_last600", "mean"),
            early_adv=("_sectional_early_adv", "mean"),
            late_adv=("_sectional_late_adv", "mean"),
            finish_speed_pct=("_sectional_finish_speed_pct", "mean"),
            pace_balance=("_sectional_pace_balance", "mean"),
        )
        .reset_index()
        .sort_values(["hrNo", "rcDate"])
    )
    grouped = daily.groupby("hrNo", dropna=False)
    daily["hr_sectional_first200_last"] = grouped["first200"].shift(1)
    daily["hr_sectional_first200_mean_4"] = grouped["first200"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    )
    daily["hr_sectional_last600_last"] = grouped["last600"].shift(1)
    daily["hr_sectional_last600_mean_4"] = grouped["last600"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    )
    daily["hr_sectional_late_adv_mean_4"] = grouped["late_adv"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    )
    daily["hr_sectional_late_adv_best_4"] = grouped["late_adv"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).max()
    )
    daily["hr_sectional_late_adv_std_4"] = grouped["late_adv"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=2).std()
    )
    daily["hr_sectional_early_adv_mean_4"] = grouped["early_adv"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    )
    daily["hr_sectional_finish_speed_pct_mean_4"] = grouped[
        "finish_speed_pct"
    ].transform(lambda values: values.shift(1).rolling(4, min_periods=1).mean())
    daily["hr_sectional_pace_balance_mean_4"] = grouped["pace_balance"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=1).mean()
    )
    daily["hr_sectional_history_count"] = grouped.cumcount().astype(float)
    for column in SECTIONAL_HISTORY_FEATURES:
        result[column] = _history_lookup(result, daily, column)
    return result.drop(
        columns=[
            "_sectional_first200",
            "_sectional_last600",
            "_sectional_early_adv",
            "_sectional_late_adv",
            "_sectional_finish_speed_pct",
            "_sectional_pace_balance",
        ]
    )


def build_sectional_features(
    frame: pd.DataFrame,
    baseline_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    result = add_sectional_history_features(frame)
    for column in SECTIONAL_HISTORY_FEATURES:
        result[f"{column}_rel"] = result[column] - result.groupby("rk")[column].transform(
            "mean"
        )
    columns = [
        *baseline_columns,
        *SECTIONAL_HISTORY_FEATURES,
        *SECTIONAL_RELATIVE_FEATURES,
    ]
    return result, list(dict.fromkeys(columns))
