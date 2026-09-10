"""Selection statistics and source-specific contract checks."""

from itertools import combinations
from math import log

import numpy as np

from selection_bias_stats import PboResult, SelectionBiasError
from x_discovery_types import BoolVector, FloatMatrix, FloatVector


class FeeHurdleInputError(ValueError):
    """Illustrative ticket inputs cannot produce a meaningful hurdle."""


def growth_pbo(returns: FloatMatrix, bins: int = 8) -> PboResult:
    """CSCV log-growth selection with deterministic IS selection and OOS midranks."""
    if (returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < 2 * bins
            or bins < 2 or bins % 2 or not np.isfinite(returns).all() or np.any(returns < -1)):
        raise SelectionBiasError("invalid growth CSCV input")
    # An absorbing-zero account receives a finite log penalty, never later recovery.
    logs = np.log(np.maximum(1 + returns, 1e-12))
    scores = np.column_stack([part.sum(axis=1) for part in np.array_split(logs, bins, axis=1)])
    logits: list[float] = []
    for train in combinations(range(bins), bins // 2):
        test = tuple(index for index in range(bins) if index not in train)
        chosen = int(np.argmax(scores[:, train].sum(axis=1)))
        unseen: FloatVector = scores[:, test].sum(axis=1)
        chosen_score = unseen.item(chosen)
        equal: BoolVector = np.isclose(unseen, chosen_score, rtol=1e-12, atol=1e-12)
        lower: BoolVector = np.logical_and(np.less(unseen, chosen_score), np.logical_not(equal))
        lower_count = np.count_nonzero(lower, axis=None, keepdims=False).item()
        equal_count = np.count_nonzero(equal, axis=None, keepdims=False).item()
        rank = lower_count + (equal_count + 1) / 2
        quantile = rank / (len(unseen) + 1)
        logits.append(log(quantile / (1 - quantile)))
    return PboResult(float(np.mean(np.asarray(logits) <= 0)), len(logits), len(returns),
                     float(np.median(logits)))


def flat_ticket_fee_hurdle(dollars: float, rate: float, floor: float) -> float:
    """Illustrative equal-notional round trip; quoted source fees are not verified."""
    if dollars <= 0 or rate < 0 or floor < 0:
        raise FeeHurdleInputError("invalid illustrative ticket")
    return 2 * max(rate * dollars, floor) / dollars
