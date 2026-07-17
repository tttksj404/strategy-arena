from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing chronological scikit-learn contract
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from kra_model_evaluation import race_normalize
from kra_pairwise_ranker import PairwiseModel, build_winner_pairs, pairwise_scores


class ModelFamily(StrEnum):
    HGB_CLASSIFIER = "hgb_classifier"
    HGB_REGRESSOR = "hgb_regressor"
    EXTRA_TREES = "extra_trees"
    PAIRWISE_HGB = "pairwise_hgb"


@dataclass(frozen=True, slots=True)
class SearchSpec:
    name: str
    family: ModelFamily
    target: str
    depth: int
    min_leaf: int


SPECS: Final = (
    SearchSpec("dynamic_win_d2", ModelFamily.HGB_CLASSIFIER, "win", 2, 80),
    SearchSpec("dynamic_win_d3", ModelFamily.HGB_CLASSIFIER, "win", 3, 80),
    SearchSpec("dynamic_win_d4", ModelFamily.HGB_CLASSIFIER, "win", 4, 60),
    SearchSpec("dynamic_place_d3", ModelFamily.HGB_CLASSIFIER, "place", 3, 80),
    SearchSpec("dynamic_finish_d3", ModelFamily.HGB_REGRESSOR, "finish_pct", 3, 80),
    SearchSpec("dynamic_extra_d12", ModelFamily.EXTRA_TREES, "win", 12, 20),
    SearchSpec("dynamic_pairwise_d3", ModelFamily.PAIRWISE_HGB, "win", 3, 120),
)


Estimator = HistGradientBoostingClassifier | HistGradientBoostingRegressor | ExtraTreesClassifier


def fit_candidate(
    train: pd.DataFrame,
    columns: list[str],
    spec: SearchSpec,
) -> tuple[Estimator, pd.Series]:
    median = train[columns].median(numeric_only=True)
    values = train[columns].apply(pd.to_numeric, errors="coerce").fillna(median)
    match spec.family:
        case ModelFamily.HGB_CLASSIFIER:
            estimator = HistGradientBoostingClassifier(
                max_depth=spec.depth,
                learning_rate=0.05,
                max_iter=260,
                min_samples_leaf=spec.min_leaf,
                l2_regularization=2.0,
                random_state=42,
            ).fit(values, train[spec.target].to_numpy())
        case ModelFamily.HGB_REGRESSOR:
            estimator = HistGradientBoostingRegressor(
                max_depth=spec.depth,
                learning_rate=0.05,
                max_iter=260,
                min_samples_leaf=spec.min_leaf,
                l2_regularization=2.0,
                random_state=42,
            ).fit(values, train[spec.target].to_numpy())
        case ModelFamily.EXTRA_TREES:
            estimator = ExtraTreesClassifier(
                n_estimators=250,
                max_depth=spec.depth,
                min_samples_leaf=spec.min_leaf,
                max_features=0.8,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            ).fit(values, train[spec.target].to_numpy())
        case ModelFamily.PAIRWISE_HGB:
            pairs = build_winner_pairs(train, columns)
            median = pairs.values[columns].median(numeric_only=True)
            pair_values = pairs.values[columns].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(median)
            estimator = HistGradientBoostingClassifier(
                max_depth=spec.depth,
                learning_rate=0.05,
                max_iter=260,
                min_samples_leaf=spec.min_leaf,
                l2_regularization=2.0,
                random_state=42,
            ).fit(pair_values, pairs.targets)
        case unreachable:
            assert_never(unreachable)
    return estimator, median


def predict_candidate(
    estimator: Estimator,
    median: pd.Series,
    frame: pd.DataFrame,
    columns: list[str],
    family: ModelFamily,
) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(median)
    match family:
        case ModelFamily.HGB_CLASSIFIER | ModelFamily.EXTRA_TREES:
            raw = estimator.predict_proba(values)[:, 1]
        case ModelFamily.HGB_REGRESSOR:
            raw = np.exp(-estimator.predict(values) / 0.25)
        case ModelFamily.PAIRWISE_HGB:
            raw = pairwise_scores(PairwiseModel(estimator, median), frame, columns)
        case unreachable:
            assert_never(unreachable)
    return race_normalize(frame, raw)
