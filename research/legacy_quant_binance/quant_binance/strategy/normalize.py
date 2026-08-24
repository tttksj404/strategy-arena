from __future__ import annotations

from typing import Iterable


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def midpoint_percentile_rank(value: float, history: Iterable[float]) -> float:
    ordered = sorted(history)
    if not ordered:
        raise ValueError("history must not be empty")
    lower_count = sum(1 for item in ordered if item < value)
    equal_count = sum(1 for item in ordered if item == value)
    midpoint_rank = lower_count + 0.5 * equal_count
    return midpoint_rank / len(ordered)


def zscore_to_unit(value: float, mean: float, stddev: float) -> float:
    if stddev <= 0:
        return 0.5
    zscore = (value - mean) / stddev
    return clamp((zscore + 2.0) / 4.0)
