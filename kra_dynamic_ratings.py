from __future__ import annotations

from collections import defaultdict
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing chronological feature-frame contract

from kra_contextual_history import build_contextual_features


INITIAL_ELO: Final = 1500.0
ELO_SCALE: Final = 400.0
RATING_FEATURES: Final = ("hr_elo", "jk_elo", "tr_elo")
ENTITY_SPECS: Final = (
    ("hrNo", "hr_elo", 40.0),
    ("jkNo", "jk_elo", 24.0),
    ("trNo", "tr_elo", 20.0),
)


def _normalized_id(value: str) -> str:
    return str(value or "").strip()


def _expected_score(rating: np.ndarray) -> np.ndarray:
    opponent_mean = (rating.sum() - rating) / np.maximum(len(rating) - 1, 1)
    return 1.0 / (1.0 + np.power(10.0, (opponent_mean - rating) / ELO_SCALE))


def add_dynamic_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for _, feature, _ in ENTITY_SPECS:
        result[feature] = INITIAL_ELO
    ratings = {entity: {} for entity, _, _ in ENTITY_SPECS}
    ordered = result.sort_values(["rcDate", "rk", "ord"])
    for _, day in ordered.groupby("rcDate", sort=True):
        for entity, feature, _ in ENTITY_SPECS:
            current = ratings[entity]
            result.loc[day.index, feature] = [
                current.get(_normalized_id(value), INITIAL_ELO) for value in day[entity]
            ]
        updates = {
            entity: defaultdict(list) for entity, _, _ in ENTITY_SPECS
        }
        for _, race in day.groupby("rk", sort=False):
            field_size = pd.to_numeric(race["field_size"], errors="coerce").to_numpy(float)
            finish = pd.to_numeric(race["ord"], errors="coerce").to_numpy(float)
            actual = 1.0 - (finish - 1.0) / np.maximum(field_size - 1.0, 1.0)
            for entity, feature, _ in ENTITY_SPECS:
                race_ratings = result.loc[race.index, feature].to_numpy(float)
                residual = actual - _expected_score(race_ratings)
                for identifier, value in zip(race[entity], residual):
                    updates[entity][_normalized_id(identifier)].append(float(value))
        for entity, _, step in ENTITY_SPECS:
            current = ratings[entity]
            for identifier, residuals in updates[entity].items():
                current[identifier] = current.get(identifier, INITIAL_ELO) + step * float(
                    np.mean(residuals)
                )
    return result


def build_rating_features(
    frame: pd.DataFrame,
    baseline_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    contextual, contextual_columns = build_contextual_features(frame, baseline_columns)
    result = add_dynamic_ratings(contextual)
    for column in RATING_FEATURES:
        result[f"{column}_rel"] = result[column] - result.groupby("rk")[column].transform("mean")
    columns = [
        *contextual_columns,
        *RATING_FEATURES,
        *(f"{column}_rel" for column in RATING_FEATURES),
    ]
    return result, list(dict.fromkeys(columns))
