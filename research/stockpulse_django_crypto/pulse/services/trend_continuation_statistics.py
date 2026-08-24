from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt

from .trend_continuation_evaluation_types import (
    MIN_WILSON_LOWER_95_PCT,
    Z_95,
    EvaluationResult,
    EvaluationRow,
)


def wilson_lower_95(successes: int, total: int) -> float:
    if total <= 0 or successes < 0 or successes > total:
        return 0.0
    proportion = successes / total
    z_squared = Z_95**2
    numerator = proportion + z_squared / (2 * total) - Z_95 * sqrt(
        (proportion * (1 - proportion) + z_squared / (4 * total)) / total
    )
    result = numerator / (1 + z_squared / total) * 100
    return result if isfinite(result) else 0.0


def mean_return_lower_95(values: Sequence[float]) -> float | None:
    if len(values) < 2 or any(not isfinite(value) for value in values):
        return None
    mean_value = sum(values) / len(values)
    sample_sd = sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))
    result = mean_value - Z_95 * sample_sd / sqrt(len(values))
    return result if isfinite(result) else None


def evaluate_rows(rows: Sequence[EvaluationRow]) -> EvaluationResult:
    resolved = tuple(row for row in rows if row.resolved)
    returns = tuple(row.net_return_pct for row in resolved if row.net_return_pct is not None)
    success_count = sum(row.outcome_success is True for row in resolved)
    lower_mean = mean_return_lower_95(returns)
    statistics_valid = len(resolved) >= 2 and lower_mean is not None and len(returns) == len(resolved)
    return EvaluationResult(
        resolved_count=len(resolved),
        success_count=success_count,
        wilson_lower_95=wilson_lower_95(success_count, len(resolved)),
        mean_return_lower_95=lower_mean if lower_mean is not None else 0.0,
        mean_return_pct=sum(returns) / len(returns) if returns else 0.0,
        statistics_valid=statistics_valid,
    )


def forward_gate(result: EvaluationResult, *, exact_count: int) -> bool:
    return (
        result.statistics_valid
        and result.resolved_count == exact_count
        and result.wilson_lower_95 >= MIN_WILSON_LOWER_95_PCT
        and result.mean_return_lower_95 > 0
        and all(isfinite(value) for value in (result.wilson_lower_95, result.mean_return_lower_95, result.mean_return_pct))
    )
