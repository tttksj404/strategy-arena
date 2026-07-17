from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, TypedDict

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — shared scikit-learn feature-frame contract


WIN_BASELINE: Final = 0.10
PLACE_BASELINE: Final = 0.25
FINISH_BASELINE: Final = 0.50
RATE_STRENGTH: Final = 5.0
FINISH_STRENGTH: Final = 3.0
HORSE_HISTORY_FEATURES: Final = (
    "hr_win_prior",
    "hr_place_prior",
    "hr_finish_prior",
    "hr_starts_log",
    "hr_days_since",
    "hr_win_prior_rel",
    "hr_place_prior_rel",
    "hr_finish_prior_rel",
)


class HorseHistoryRecord(TypedDict):
    win_prior: float
    place_prior: float
    finish_prior: float
    starts_log: float
    last_date: str


class HorseHistorySnapshot(TypedDict):
    records: dict[str, HorseHistoryRecord]


@dataclass(frozen=True, slots=True)
class PriorSpec:
    target: str
    output: str
    baseline: float
    strength: float


def _normalized_id(value: str) -> str:
    return str(value or "").strip()


def _daily_prior(frame: pd.DataFrame, spec: PriorSpec) -> pd.Series:
    daily = (
        frame.groupby(["hrNo", "rcDate"], dropna=False)[spec.target]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["hrNo", "rcDate"])
    )
    grouped = daily.groupby("hrNo", dropna=False)
    prior_sum = grouped["sum"].cumsum() - daily["sum"]
    prior_count = grouped["count"].cumsum() - daily["count"]
    daily[spec.output] = (
        prior_sum + spec.strength * spec.baseline
    ) / (prior_count + spec.strength)
    lookup = daily.set_index(["hrNo", "rcDate"])[spec.output]
    values = [lookup.get((horse, date), spec.baseline) for horse, date in zip(frame["hrNo"], frame["rcDate"])]
    return pd.Series(values, index=frame.index, dtype=float)


def _starts_before_date(frame: pd.DataFrame) -> pd.Series:
    daily = (
        frame.groupby(["hrNo", "rcDate"], dropna=False)
        .size()
        .rename("starts")
        .reset_index()
        .sort_values(["hrNo", "rcDate"])
    )
    daily["prior_starts"] = daily.groupby("hrNo", dropna=False)["starts"].cumsum() - daily["starts"]
    lookup = daily.set_index(["hrNo", "rcDate"])["prior_starts"]
    values = [lookup.get((horse, date), 0) for horse, date in zip(frame["hrNo"], frame["rcDate"])]
    return np.log1p(pd.Series(values, index=frame.index, dtype=float))


def _days_since_previous_start(frame: pd.DataFrame) -> pd.Series:
    daily = frame[["hrNo", "rcDate"]].drop_duplicates().sort_values(["hrNo", "rcDate"])
    daily["date"] = pd.to_datetime(daily["rcDate"], format="%Y%m%d")
    daily["previous_date"] = daily.groupby("hrNo", dropna=False)["date"].shift()
    daily["days_since"] = (daily["date"] - daily["previous_date"]).dt.days
    lookup = daily.set_index(["hrNo", "rcDate"])["days_since"]
    values = [lookup.get((horse, date), np.nan) for horse, date in zip(frame["hrNo"], frame["rcDate"])]
    return pd.Series(values, index=frame.index, dtype=float)


def add_horse_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["finish_pct"] = (result["ord"] - 1) / (result["field_size"] - 1)
    specs = (
        PriorSpec("win", "hr_win_prior", WIN_BASELINE, RATE_STRENGTH),
        PriorSpec("place", "hr_place_prior", PLACE_BASELINE, RATE_STRENGTH),
        PriorSpec("finish_pct", "hr_finish_prior", FINISH_BASELINE, FINISH_STRENGTH),
    )
    for spec in specs:
        result[spec.output] = _daily_prior(result, spec)
    result["hr_starts_log"] = _starts_before_date(result)
    result["hr_days_since"] = _days_since_previous_start(result)
    return result


def build_horse_history_snapshot(frame: pd.DataFrame) -> HorseHistorySnapshot:
    source = frame.copy()
    source["finish_pct"] = (source["ord"] - 1) / (source["field_size"] - 1)
    grouped = source.groupby("hrNo", dropna=False).agg(
        starts=("win", "count"),
        wins=("win", "sum"),
        places=("place", "sum"),
        finish_sum=("finish_pct", "sum"),
        last_date=("rcDate", "max"),
    )
    records: dict[str, HorseHistoryRecord] = {}
    for horse, row in grouped.iterrows():
        starts = float(row["starts"])
        records[_normalized_id(horse)] = {
            "win_prior": float((row["wins"] + RATE_STRENGTH * WIN_BASELINE) / (starts + RATE_STRENGTH)),
            "place_prior": float((row["places"] + RATE_STRENGTH * PLACE_BASELINE) / (starts + RATE_STRENGTH)),
            "finish_prior": float((row["finish_sum"] + FINISH_STRENGTH * FINISH_BASELINE) / (starts + FINISH_STRENGTH)),
            "starts_log": float(np.log1p(starts)),
            "last_date": str(row["last_date"]),
        }
    return {"records": records}


def _elapsed_days(race_date: str | None, last_date: str) -> float:
    if not race_date or not last_date:
        return float("nan")
    try:
        current = datetime.strptime(race_date, "%Y%m%d")
        previous = datetime.strptime(last_date, "%Y%m%d")
    except ValueError:
        return float("nan")
    return float((current - previous).days)


def apply_horse_history_snapshot(
    frame: pd.DataFrame,
    snapshot: HorseHistorySnapshot,
    race_date: str | None,
) -> pd.DataFrame:
    result = frame.copy()
    records = snapshot.get("records", {})
    selected = [records.get(_normalized_id(horse)) for horse in result.get("hrNo", pd.Series(index=result.index, dtype=str))]
    result["hr_win_prior"] = [record["win_prior"] if record else WIN_BASELINE for record in selected]
    result["hr_place_prior"] = [record["place_prior"] if record else PLACE_BASELINE for record in selected]
    result["hr_finish_prior"] = [record["finish_prior"] if record else FINISH_BASELINE for record in selected]
    result["hr_starts_log"] = [record["starts_log"] if record else 0.0 for record in selected]
    result["hr_days_since"] = [
        _elapsed_days(race_date, record["last_date"]) if record else float("nan")
        for record in selected
    ]
    return result
