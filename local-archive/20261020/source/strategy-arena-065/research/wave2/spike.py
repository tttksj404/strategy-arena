from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK


def hysteresis_position(score: pd.Series, entry_threshold_apr: float, exit_threshold_apr: float) -> pd.Series:
    values: list[float] = []
    active = 0.0
    for value in score:
        if pd.notna(value) and value > entry_threshold_apr:
            active = 1.0
        elif pd.notna(value) and value < exit_threshold_apr:
            active = 0.0
        values.append(active)
    return pd.Series(values, index=score.index, dtype=float).shift(1).fillna(0.0)


def spike_position(
    funding: pd.Series,
    score: pd.Series,
    entry_rate: float,
    exit_threshold_apr: float,
) -> pd.Series:
    daily_spike = funding.resample("1D").max()
    daily_score = score.groupby(pd.DatetimeIndex(score.index).normalize()).last()
    values: list[float] = []
    active = 0.0
    daily_values: list[float] = []
    for day, apr in daily_score.items():
        spike = daily_spike.get(day, 0.0)
        if spike > entry_rate:
            active = 1.0
        elif pd.notna(apr) and apr < exit_threshold_apr:
            active = 0.0
        daily_values.append(active)
    daily_position = pd.Series(daily_values, index=daily_score.index, dtype=float).shift(1).fillna(0.0)
    aligned_position = daily_position.reindex(pd.DatetimeIndex(score.index).normalize()).fillna(0.0)
    values.extend(float(value) for value in aligned_position)
    return pd.Series(values, index=score.index, dtype=float)

__all__ = ["spike_position"]
