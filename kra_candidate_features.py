from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — shared scikit-learn feature-frame contract


CANDIDATE_GROUPS: Final = {
    "recent_form": (
        "hr_recent_win", "hr_recent_place", "hr_recent_finish", "hr_recent_starts",
    ),
    "distance_form": ("hr_distance_win", "hr_distance_finish"),
    "meet_form": ("hr_meet_win", "hr_meet_finish"),
    "jockey_pair": ("hr_jk_win", "hr_jk_finish"),
    "state_change": ("hr_rating_change", "hr_burden_change", "hr_distance_change"),
}


@dataclass(frozen=True, slots=True)
class PriorDefinition:
    target: str
    output: str
    baseline: float


def _daily_prior(
    frame: pd.DataFrame,
    group_columns: tuple[str, ...],
    definition: PriorDefinition,
    window: int | None,
) -> tuple[pd.Series, pd.Series]:
    keys = [*group_columns, "rcDate"]
    daily = (
        frame.groupby(keys, dropna=False)[definition.target]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(keys)
    )
    grouped = daily.groupby(list(group_columns), dropna=False, group_keys=False)
    if window is None:
        prior_sum = grouped["sum"].cumsum() - daily["sum"]
        prior_count = grouped["count"].cumsum() - daily["count"]
    else:
        prior_sum = grouped["sum"].transform(
            lambda values: values.shift().rolling(window, min_periods=1).sum()
        )
        prior_count = grouped["count"].transform(
            lambda values: values.shift().rolling(window, min_periods=1).sum()
        )
    daily["prior"] = (prior_sum / prior_count).fillna(definition.baseline)
    daily["prior_count"] = prior_count.fillna(0.0)
    lookup = daily.set_index(keys)
    row_keys = zip(*(frame[column] for column in keys))
    values = [lookup["prior"].get(key, definition.baseline) for key in row_keys]
    count_keys = zip(*(frame[column] for column in keys))
    counts = [lookup["prior_count"].get(key, 0.0) for key in count_keys]
    return (
        pd.Series(values, index=frame.index, dtype=float),
        pd.Series(counts, index=frame.index, dtype=float),
    )


def _previous_value(frame: pd.DataFrame, column: str) -> pd.Series:
    daily = (
        frame.groupby(["hrNo", "rcDate"], dropna=False)[column]
        .mean()
        .reset_index()
        .sort_values(["hrNo", "rcDate"])
    )
    daily["previous"] = daily.groupby("hrNo", dropna=False)[column].shift()
    lookup = daily.set_index(["hrNo", "rcDate"])["previous"]
    values = [lookup.get((horse, date), np.nan) for horse, date in zip(frame["hrNo"], frame["rcDate"])]
    return pd.Series(values, index=frame.index, dtype=float)


def add_candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["finish_pct"] = (result["ord"] - 1) / (result["field_size"] - 1)
    recent = (
        PriorDefinition("win", "hr_recent_win", 0.1),
        PriorDefinition("place", "hr_recent_place", 0.25),
        PriorDefinition("finish_pct", "hr_recent_finish", 0.5),
    )
    for definition in recent:
        values, counts = _daily_prior(result, ("hrNo",), definition, window=5)
        result[definition.output] = values
        if definition.output == "hr_recent_win":
            result["hr_recent_starts"] = counts
    conditioned = (
        (("hrNo", "rcDist"), PriorDefinition("win", "hr_distance_win", 0.1)),
        (("hrNo", "rcDist"), PriorDefinition("finish_pct", "hr_distance_finish", 0.5)),
        (("hrNo", "meet"), PriorDefinition("win", "hr_meet_win", 0.1)),
        (("hrNo", "meet"), PriorDefinition("finish_pct", "hr_meet_finish", 0.5)),
        (("hrNo", "jkNo"), PriorDefinition("win", "hr_jk_win", 0.1)),
        (("hrNo", "jkNo"), PriorDefinition("finish_pct", "hr_jk_finish", 0.5)),
    )
    for group_columns, definition in conditioned:
        result[definition.output], _ = _daily_prior(
            result, group_columns, definition, window=None
        )
    result["hr_rating_change"] = result["rating"] - _previous_value(result, "rating")
    result["hr_burden_change"] = result["wgBudam"] - _previous_value(result, "wgBudam")
    result["hr_distance_change"] = result["rcDist"] - _previous_value(result, "rcDist")
    return result
