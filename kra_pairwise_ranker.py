from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — pair construction uses indexed feature frames


class PairwiseEstimator(Protocol):
    def predict_proba(self, values: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class PairwiseTrainingSet:
    values: pd.DataFrame
    targets: np.ndarray


@dataclass(frozen=True, slots=True)
class PairwiseModel:
    estimator: PairwiseEstimator
    median: pd.Series


def build_winner_pairs(frame: pd.DataFrame, columns: list[str]) -> PairwiseTrainingSet:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    differences = []
    targets = []
    for _, race in frame.groupby("rk", sort=False):
        winner_indices = race.index[race["win"] == 1].tolist()
        if len(winner_indices) != 1:
            continue
        winner_index = winner_indices[0]
        winner = values.loc[winner_index]
        for loser_index in race.index[race["win"] == 0]:
            difference = winner - values.loc[loser_index]
            differences.extend((difference, -difference))
            targets.extend((1, 0))
    return PairwiseTrainingSet(
        pd.DataFrame(differences, columns=columns),
        np.asarray(targets, dtype=np.int8),
    )


def pairwise_scores(
    model: PairwiseModel,
    frame: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    values = (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .fillna(model.median)
    )
    pair_values = []
    pair_positions = []
    for positions in frame.groupby("rk", sort=False).indices.values():
        for left_offset, left in enumerate(positions):
            for right in positions[left_offset + 1:]:
                pair_values.append(values.iloc[left] - values.iloc[right])
                pair_positions.append((left, right))
    if not pair_values:
        return np.zeros(len(frame), dtype=float)
    pairs = pd.DataFrame(pair_values, columns=columns).fillna(model.median)
    probability = model.estimator.predict_proba(pairs)[:, 1]
    scores = np.zeros(len(frame), dtype=float)
    for (left, right), left_probability in zip(pair_positions, probability):
        scores[left] += left_probability
        scores[right] += 1.0 - left_probability
    return scores
