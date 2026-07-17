from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — race-group boundary contract


@dataclass(frozen=True, slots=True)
class RerankResult:
    scores: np.ndarray
    switches: int


@dataclass(frozen=True, slots=True)
class RaceScores:
    baseline: np.ndarray
    pairwise: np.ndarray


@dataclass(frozen=True, slots=True)
class RerankPolicy:
    weight: float
    top_k: int


def restricted_rerank(
    frame: pd.DataFrame,
    inputs: RaceScores,
    policy: RerankPolicy,
) -> RerankResult:
    scores = inputs.baseline.copy()
    switches = 0
    for positions in frame.groupby("rk", sort=False).indices.values():
        baseline_order = positions[np.argsort(-inputs.baseline[positions])]
        blended = (
            (1.0 - policy.weight) * inputs.baseline[positions]
            + policy.weight * inputs.pairwise[positions]
        )
        candidate = positions[int(np.argmax(blended))]
        baseline_leader = baseline_order[0]
        if candidate != baseline_leader and candidate in baseline_order[:policy.top_k]:
            scores[candidate] = np.nextafter(scores[baseline_leader], np.inf)
            switches += 1
    return RerankResult(scores, switches)
