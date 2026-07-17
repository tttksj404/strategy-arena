from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, assert_never

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — scikit-learn estimator input contract
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ProbabilityEstimator(Protocol):
    def predict_proba(self, values: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    family: Literal["hgb", "extra_trees", "random_forest", "logistic"]
    max_depth: int | None
    iterations: int
    min_leaf: int
    max_features: float
    learning_rate: float = 0.05
    l2: float = 1.0


def fit_candidate(
    frame: pd.DataFrame,
    columns: list[str],
    target: str,
    spec: ModelSpec,
) -> tuple[ProbabilityEstimator, pd.Series]:
    median = frame[columns].median(numeric_only=True)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(median)
    match spec.family:
        case "hgb":
            model = HistGradientBoostingClassifier(
                max_depth=spec.max_depth,
                learning_rate=spec.learning_rate,
                max_iter=spec.iterations,
                min_samples_leaf=spec.min_leaf,
                l2_regularization=spec.l2,
                max_features=spec.max_features,
                random_state=42,
            )
        case "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=spec.iterations,
                max_depth=spec.max_depth,
                min_samples_leaf=spec.min_leaf,
                max_features=spec.max_features,
                n_jobs=-1,
                random_state=42,
            )
        case "random_forest":
            model = RandomForestClassifier(
                n_estimators=spec.iterations,
                max_depth=spec.max_depth,
                min_samples_leaf=spec.min_leaf,
                max_features=spec.max_features,
                n_jobs=-1,
                random_state=42,
            )
        case "logistic":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0 / spec.l2,
                    max_iter=spec.iterations,
                    random_state=42,
                ),
            )
        case unreachable:
            assert_never(unreachable)
    model.fit(values, frame[target].to_numpy())
    return model, median
