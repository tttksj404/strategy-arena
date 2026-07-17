from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing grouped-ranking contract
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

from kra_model_evaluation import race_normalize
from kra_pairwise_ranker import PairwiseModel, PairwiseTrainingSet, pairwise_scores


class FullOrderFamily(StrEnum):
    PLACKETT_LUCE = "plackett_luce"
    PAIRWISE_HGB = "pairwise_hgb"


@dataclass(frozen=True, slots=True)
class FullOrderSpec:
    name: str
    family: FullOrderFamily
    strength: float
    maximum_rank: int

    @classmethod
    def plackett_luce(
        cls, name: str, ridge: float, maximum_rank: int
    ) -> FullOrderSpec:
        return cls(name, FullOrderFamily.PLACKETT_LUCE, ridge, maximum_rank)

    @classmethod
    def pairwise(
        cls, name: str, depth: int, minimum_leaf: int
    ) -> FullOrderSpec:
        return cls(name, FullOrderFamily.PAIRWISE_HGB, float(depth), minimum_leaf)


SPECS: Final = (
    FullOrderSpec.plackett_luce("plackett_luce_top3_r2", 2.0, 3),
    FullOrderSpec.plackett_luce("plackett_luce_top5_r8", 8.0, 5),
    FullOrderSpec.pairwise("full_order_pairwise_d2", 2, 80),
    FullOrderSpec.pairwise("full_order_pairwise_d3", 3, 120),
)


@dataclass(frozen=True, slots=True)
class PlackettLuceModel:
    median: pd.Series
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    columns: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = _values(frame, self.columns, self.median)
        standardized = np.clip((values - self.mean) / self.scale, -10.0, 10.0)
        score = np.exp(np.clip(standardized @ self.coefficient, -20.0, 20.0))
        return race_normalize(frame, score)


@dataclass(frozen=True, slots=True)
class FullOrderPairwiseModel:
    model: PairwiseModel
    columns: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return race_normalize(frame, pairwise_scores(self.model, frame, self.columns))


FullOrderModel = PlackettLuceModel | FullOrderPairwiseModel


def _values(
    frame: pd.DataFrame, columns: list[str], median: pd.Series
) -> np.ndarray:
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(median)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def build_full_order_pairs(
    frame: pd.DataFrame, columns: list[str]
) -> PairwiseTrainingSet:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    differences: list[pd.Series] = []
    targets: list[int] = []
    for _, race in frame.groupby("rk", sort=False):
        ordered = race.sort_values("ord", kind="stable").index.tolist()
        for better_offset, better_index in enumerate(ordered):
            for worse_index in ordered[better_offset + 1 :]:
                difference = values.loc[better_index] - values.loc[worse_index]
                differences.extend((difference, -difference))
                targets.extend((1, 0))
    return PairwiseTrainingSet(
        pd.DataFrame(differences, columns=columns),
        np.asarray(targets, dtype=np.int8),
    )


def _fit_plackett_luce(
    frame: pd.DataFrame, columns: list[str], spec: FullOrderSpec
) -> PlackettLuceModel:
    median = frame[columns].median(numeric_only=True)
    values = _values(frame, columns, median)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = np.clip((values - mean) / scale, -10.0, 10.0)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    orders = tuple(
        positions.loc[race.sort_values("ord", kind="stable").index].to_numpy(dtype=int)
        for _, race in frame.groupby("rk", sort=False)
    )

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        scores = standardized @ coefficient
        loss = 0.5 * spec.strength * float(coefficient @ coefficient)
        gradient = spec.strength * coefficient
        for order in orders:
            step_count = min(spec.maximum_rank, len(order) - 1)
            weight = 1.0 / max(step_count, 1)
            for step in range(step_count):
                available = order[step:]
                available_scores = scores[available]
                maximum = float(available_scores.max())
                exponent = np.exp(available_scores - maximum)
                probability = exponent / exponent.sum()
                loss += weight * (
                    maximum + float(np.log(exponent.sum())) - scores[order[step]]
                )
                gradient += weight * (
                    probability @ standardized[available]
                    - standardized[order[step]]
                )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(standardized.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 120, "ftol": 1e-8, "maxls": 30},
    )
    return PlackettLuceModel(median, mean, scale, result.x, columns)


def fit_full_order_model(
    frame: pd.DataFrame, columns: list[str], spec: FullOrderSpec
) -> FullOrderModel:
    match spec.family:
        case FullOrderFamily.PLACKETT_LUCE:
            return _fit_plackett_luce(frame, columns, spec)
        case FullOrderFamily.PAIRWISE_HGB:
            pairs = build_full_order_pairs(frame, columns)
            median = pairs.values[columns].median(numeric_only=True)
            values = pairs.values[columns].fillna(median).fillna(0.0)
            estimator = HistGradientBoostingClassifier(
                max_depth=int(spec.strength),
                learning_rate=0.05,
                max_iter=250,
                min_samples_leaf=spec.maximum_rank,
                l2_regularization=3.0,
                random_state=20260712,
            ).fit(values, pairs.targets)
            return FullOrderPairwiseModel(PairwiseModel(estimator, median), columns)
        case unreachable:
            assert_never(unreachable)
