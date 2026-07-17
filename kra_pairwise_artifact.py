from __future__ import annotations

from typing import Final, TypedDict

import pandas as pd  # noqa: PANDAS_OK — scikit-learn artifact contract

from kra_model_candidates import ModelSpec, ProbabilityEstimator, fit_candidate
from kra_pairwise_ranker import build_winner_pairs


PAIRWISE_SPEC: Final = ModelSpec(
    "pair_hgb_d3",
    "hgb",
    3,
    300,
    160,
    1.0,
    0.05,
    2.0,
)


class PairwiseArtifact(TypedDict):
    estimator: ProbabilityEstimator
    median: dict[str, float]
    weight: float
    top_k: int
    enabled: bool


def build_pairwise_artifact(
    frame: pd.DataFrame,
    columns: list[str],
) -> PairwiseArtifact:
    pairs = build_winner_pairs(frame, columns)
    training = pairs.values.copy()
    training["target"] = pairs.targets
    estimator, median = fit_candidate(training, columns, "target", PAIRWISE_SPEC)
    return {
        "estimator": estimator,
        "median": {
            key: float(value)
            for key, value in median.items()
            if pd.notna(value)
        },
        "weight": 0.5,
        "top_k": 3,
        "enabled": False,
    }
