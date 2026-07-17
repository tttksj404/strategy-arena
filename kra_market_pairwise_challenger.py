from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — existing race-grouped model contract
from sklearn.ensemble import HistGradientBoostingClassifier

from kra_model_evaluation import market_probability, race_normalize


@dataclass(frozen=True, slots=True)
class ChallengerSpec:
    top_k: int
    balance_power: float
    threshold: float


@dataclass(frozen=True, slots=True)
class MarketPairwiseChallengerModel:
    estimator: HistGradientBoostingClassifier
    median: pd.Series
    runner_columns: list[str]
    feature_columns: list[str]
    top_k: int

    def predict(self, frame: pd.DataFrame, threshold: float) -> np.ndarray:
        return self.predict_many(frame, (threshold,))[threshold]

    def predict_many(
        self, frame: pd.DataFrame, thresholds: tuple[float, ...]
    ) -> dict[float, np.ndarray]:
        rows, race_keys, challenger_positions = build_challenger_rows(
            frame, self.runner_columns, self.top_k
        )
        values = rows[self.feature_columns].fillna(self.median).fillna(0.0)
        challenger_probability = self.estimator.predict_proba(values)[:, 1]
        best_by_race: dict[str, tuple[int, float]] = {}
        for race_key, position, probability in zip(
            race_keys, challenger_positions, challenger_probability
        ):
            previous = best_by_race.get(race_key)
            if previous is None or probability > previous[1]:
                best_by_race[race_key] = (position, float(probability))

        market = market_probability(frame)
        positions = pd.Series(np.arange(len(frame)), index=frame.index)
        favorites: dict[str, int] = {}
        for race_key, race in frame.groupby("rk", sort=False):
            race_positions = positions.loc[race.index].to_numpy(dtype=int)
            favorites[str(race_key)] = race_positions[
                int(np.argmax(market[race_positions]))
            ]
        outputs: dict[float, np.ndarray] = {}
        for threshold in thresholds:
            output = market.copy()
            for race_key, (challenger, probability) in best_by_race.items():
                if probability >= threshold:
                    favorite = favorites[race_key]
                    output[favorite], output[challenger] = (
                        output[challenger],
                        output[favorite],
                    )
            outputs[threshold] = race_normalize(frame, output)
        return outputs


def build_challenger_rows(
    frame: pd.DataFrame, runner_columns: list[str], top_k: int
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[int, ...]]:
    market = market_probability(frame)
    numeric = frame[runner_columns].apply(pd.to_numeric, errors="coerce")
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    rows: list[dict[str, float | int]] = []
    race_keys: list[str] = []
    challenger_positions: list[int] = []
    has_target = "win" in frame
    for race_key, race in frame.groupby("rk", sort=False):
        race_positions = positions.loc[race.index].to_numpy(dtype=int)
        order = race_positions[np.argsort(-market[race_positions], kind="stable")]
        selected = order[: min(top_k, len(order))]
        favorite = selected[0]
        favorite_values = numeric.iloc[favorite]
        for market_rank, challenger in enumerate(selected[1:], start=2):
            challenger_values = numeric.iloc[challenger]
            row: dict[str, float | int] = {
                "market_rank": market_rank,
                "favorite_probability": float(market[favorite]),
                "challenger_probability": float(market[challenger]),
                "market_gap": float(market[favorite] - market[challenger]),
            }
            for column in runner_columns:
                favorite_value = favorite_values[column]
                challenger_value = challenger_values[column]
                row[f"favorite_{column}"] = float(favorite_value)
                row[f"challenger_{column}"] = float(challenger_value)
                row[f"d_{column}"] = float(challenger_value - favorite_value)
            if has_target:
                row["target"] = int(frame.iloc[challenger]["win"] == 1)
            rows.append(row)
            race_keys.append(str(race_key))
            challenger_positions.append(int(challenger))
    return pd.DataFrame(rows), tuple(race_keys), tuple(challenger_positions)


def fit_challenger_model(
    frame: pd.DataFrame, runner_columns: list[str], spec: ChallengerSpec
) -> MarketPairwiseChallengerModel:
    rows, _, _ = build_challenger_rows(frame, runner_columns, spec.top_k)
    feature_columns = [column for column in rows if column != "target"]
    median = rows[feature_columns].median(numeric_only=True)
    values = rows[feature_columns].fillna(median).fillna(0.0)
    target = rows["target"].to_numpy(dtype=int)
    counts = pd.Series(target).value_counts()
    weights = np.asarray(
        [(len(target) / counts[label]) ** spec.balance_power for label in target],
        dtype=float,
    )
    estimator = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.04,
        max_iter=450,
        min_samples_leaf=60,
        l2_regularization=5.0,
        random_state=20260712,
    ).fit(values, target, sample_weight=weights)
    return MarketPairwiseChallengerModel(
        estimator, median, runner_columns, feature_columns, spec.top_k
    )
