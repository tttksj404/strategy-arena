from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from kra_diversified_rankers import ConditionalLogitModel, fit_conditional_logit
from kra_model_evaluation import race_normalize


class GlobalFamily(StrEnum):
    WINNER_SVM = "winner_svm"
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    CONDITIONAL_LOGIT = "conditional_logit"


@dataclass(frozen=True, slots=True)
class GlobalSpec:
    name: str
    family: GlobalFamily
    strength: float


SPECS: Final = (
    GlobalSpec("winner_svm_c003", GlobalFamily.WINNER_SVM, 0.03),
    GlobalSpec("winner_svm_c010", GlobalFamily.WINNER_SVM, 0.10),
    GlobalSpec("neural_network_a001", GlobalFamily.NEURAL_NETWORK, 0.01),
    GlobalSpec("neural_network_a010", GlobalFamily.NEURAL_NETWORK, 0.10),
    GlobalSpec("random_forest_leaf15", GlobalFamily.RANDOM_FOREST, 15.0),
    GlobalSpec("random_forest_leaf30", GlobalFamily.RANDOM_FOREST, 30.0),
    GlobalSpec("conditional_logit_r2", GlobalFamily.CONDITIONAL_LOGIT, 2.0),
    GlobalSpec("conditional_logit_r8", GlobalFamily.CONDITIONAL_LOGIT, 8.0),
)


Estimator = (
    LinearSVC
    | MLPClassifier
    | RandomForestClassifier
    | ConditionalLogitModel
)


@dataclass(slots=True)
class GlobalModel:
    family: GlobalFamily
    estimator: Estimator
    median: pd.Series
    scaler: StandardScaler | None
    columns: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        match self.family:
            case GlobalFamily.CONDITIONAL_LOGIT:
                raw = self.estimator.predict(frame)
            case GlobalFamily.WINNER_SVM:
                values = _values(frame, self.columns, self.median)
                scaled = self.scaler.transform(values)
                raw = expit(self.estimator.decision_function(scaled))
            case GlobalFamily.NEURAL_NETWORK:
                values = _values(frame, self.columns, self.median)
                scaled = self.scaler.transform(values)
                raw = self.estimator.predict_proba(scaled)[:, 1]
            case GlobalFamily.RANDOM_FOREST:
                values = _values(frame, self.columns, self.median)
                raw = self.estimator.predict_proba(values)[:, 1]
            case unreachable:
                assert_never(unreachable)
        return race_normalize(frame, np.asarray(raw, dtype=float))


def _values(
    frame: pd.DataFrame,
    columns: list[str],
    median: pd.Series,
) -> np.ndarray:
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(median)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def fit_global_model(
    train: pd.DataFrame,
    columns: list[str],
    spec: GlobalSpec,
) -> GlobalModel:
    median = train[columns].median(numeric_only=True)
    values = _values(train, columns, median)
    target = train["win"].to_numpy(dtype=int)
    match spec.family:
        case GlobalFamily.WINNER_SVM:
            scaler = StandardScaler().fit(values)
            estimator = LinearSVC(
                C=spec.strength,
                class_weight="balanced",
                dual="auto",
                max_iter=5000,
                random_state=20260711,
            ).fit(scaler.transform(values), target)
        case GlobalFamily.NEURAL_NETWORK:
            scaler = StandardScaler().fit(values)
            estimator = MLPClassifier(
                hidden_layer_sizes=(48, 24),
                activation="relu",
                solver="adam",
                alpha=spec.strength,
                batch_size=256,
                learning_rate_init=0.001,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=20260711,
            ).fit(
                scaler.transform(values),
                target,
                sample_weight=compute_sample_weight("balanced", target),
            )
        case GlobalFamily.RANDOM_FOREST:
            scaler = None
            estimator = RandomForestClassifier(
                n_estimators=500,
                max_depth=14,
                min_samples_leaf=int(spec.strength),
                max_features=0.7,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=20260711,
            ).fit(values, target)
        case GlobalFamily.CONDITIONAL_LOGIT:
            scaler = None
            estimator = fit_conditional_logit(
                train,
                columns,
                ridge=spec.strength,
            )
        case unreachable:
            assert_never(unreachable)
    return GlobalModel(spec.family, estimator, median, scaler, columns)
