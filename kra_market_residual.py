from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — race-grouped market feature contract
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from kra_diversified_rankers import _numeric_values
from kra_model_evaluation import market_probability, race_normalize


MARKET_FEATURES: Final = (
    "market_win_probability",
    "market_place_probability",
    "market_log_probability",
    "market_rank_percentile",
    "market_is_favorite",
    "market_favorite_probability",
    "market_favorite_gap",
    "market_entropy",
    "market_overround",
    "market_win_place_disagreement",
    "market_consensus_interaction",
    "market_disagreement_interaction",
)


def add_market_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    win_odds = pd.to_numeric(result["winOdds"], errors="coerce").clip(lower=1.001)
    place_odds = pd.to_numeric(result["plcOdds"], errors="coerce").clip(lower=1.001)
    implied_win = 1.0 / win_odds
    implied_place = 1.0 / place_odds
    win_total = implied_win.groupby(result["rk"]).transform("sum")
    place_total = implied_place.groupby(result["rk"]).transform("sum")
    win_probability = implied_win / win_total
    place_probability = implied_place / place_total

    result["market_win_probability"] = win_probability
    result["market_place_probability"] = place_probability
    result["market_log_probability"] = np.log(win_probability.clip(lower=1e-9))
    result["market_rank_percentile"] = win_probability.groupby(result["rk"]).rank(
        pct=True, ascending=True, method="average"
    )
    result["market_is_favorite"] = (
        win_probability.groupby(result["rk"]).rank(ascending=False, method="min") == 1
    ).astype(float)
    result["market_favorite_probability"] = win_probability.groupby(result["rk"]).transform("max")

    sorted_probability = result.assign(_market_p=win_probability).groupby("rk")["_market_p"].nlargest(2)
    gaps = sorted_probability.groupby(level=0).apply(
        lambda values: float(values.iloc[0] - values.iloc[1]) if len(values) > 1 else 0.0,
        include_groups=False,
    )
    result["market_favorite_gap"] = result["rk"].map(gaps).fillna(0.0)

    entropy_term = -(win_probability * np.log(win_probability.clip(lower=1e-9)))
    result["market_entropy"] = entropy_term.groupby(result["rk"]).transform("sum")
    result["market_overround"] = win_total
    win_rank = win_probability.groupby(result["rk"]).rank(pct=True)
    place_rank = place_probability.groupby(result["rk"]).rank(pct=True)
    result["market_win_place_disagreement"] = win_rank - place_rank
    consensus = pd.to_numeric(result.get("consensus_strength", 0.0), errors="coerce").fillna(0.0)
    disagreement = pd.to_numeric(result.get("consensus_disagreement", 0.0), errors="coerce").fillna(0.0)
    result["market_consensus_interaction"] = win_probability * consensus
    result["market_disagreement_interaction"] = win_probability * disagreement
    return result, list(MARKET_FEATURES)


@dataclass(slots=True)
class MarketResidualModel:
    estimator: HistGradientBoostingRegressor
    median: pd.Series
    columns: list[str]

    def predict_adjustment(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric_values(frame, self.columns, self.median)
        return np.clip(self.estimator.predict(values), -1.0, 1.0)

    def predict(self, frame: pd.DataFrame, scale: float = 1.0) -> np.ndarray:
        market = market_probability(frame)
        adjusted = market * np.exp(np.clip(scale * self.predict_adjustment(frame), -4.0, 4.0))
        return race_normalize(frame, adjusted)


def fit_market_residual(train: pd.DataFrame, columns: list[str]) -> MarketResidualModel:
    median = train[columns].median(numeric_only=True)
    values = _numeric_values(train, columns, median)
    market = market_probability(train)
    target = train["win"].to_numpy(dtype=float) - market
    weights = 1.0 / pd.to_numeric(train["field_size"], errors="coerce").clip(lower=2)
    estimator = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=4,
        learning_rate=0.035,
        max_iter=450,
        min_samples_leaf=80,
        l2_regularization=5.0,
        random_state=20260711,
    ).fit(values, target, sample_weight=weights.to_numpy())
    return MarketResidualModel(estimator, median, columns)


def restricted_market_rerank(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    top_k: int,
) -> np.ndarray:
    market = market_probability(frame)
    output = market.copy()
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for _, race in frame.groupby("rk", sort=False):
        race_positions = positions.loc[race.index].to_numpy(dtype=int)
        market_order = race_positions[np.argsort(-market[race_positions], kind="stable")]
        top_positions = market_order[: min(top_k, len(market_order))]
        candidate_order = top_positions[
            np.argsort(-np.asarray(candidate)[top_positions], kind="stable")
        ]
        output[candidate_order] = np.sort(market[top_positions])[::-1]
    return race_normalize(frame, output)


def uncertainty_gate(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    maximum_favorite_probability: float,
    maximum_gap: float,
) -> np.ndarray:
    market = market_probability(frame)
    output = market.copy()
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for _, race in frame.groupby("rk", sort=False):
        race_positions = positions.loc[race.index].to_numpy(dtype=int)
        ordered = np.sort(market[race_positions])[::-1]
        favorite = float(ordered[0])
        gap = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        if favorite <= maximum_favorite_probability or gap <= maximum_gap:
            output[race_positions] = np.asarray(candidate)[race_positions]
    return race_normalize(frame, output)


@dataclass(slots=True)
class FavoriteChallengerModel:
    estimator: HistGradientBoostingClassifier
    median: pd.Series
    feature_columns: list[str]
    runner_columns: list[str]
    top_k: int

    def predict(self, frame: pd.DataFrame, margin: float = 0.0) -> np.ndarray:
        return self.predict_many(frame, [margin])[margin]

    def predict_many(
        self, frame: pd.DataFrame, margins: list[float] | tuple[float, ...]
    ) -> dict[float, np.ndarray]:
        race_frame, race_keys = _challenger_race_frame(
            frame, self.runner_columns, self.top_k, include_target=False
        )
        values = _numeric_values(race_frame, self.feature_columns, self.median)
        probabilities = self.estimator.predict_proba(values)
        classes = self.estimator.classes_.astype(int)
        market = market_probability(frame)
        positions = pd.Series(np.arange(len(frame)), index=frame.index)
        grouped_positions = {
            str(race_key): positions.loc[race.index].to_numpy(dtype=int)
            for race_key, race in frame.groupby("rk", sort=False)
        }
        outputs = {}
        for margin in margins:
            output = market.copy()
            for row_index, race_key in enumerate(race_keys):
                race_positions = grouped_positions[race_key]
                market_order = race_positions[np.argsort(-market[race_positions], kind="stable")]
                class_probability = dict(zip(classes, probabilities[row_index]))
                chosen = max(class_probability, key=class_probability.get)
                favorite_probability = class_probability.get(0, 0.0)
                if chosen >= min(self.top_k, len(market_order)):
                    chosen = 0
                if chosen != 0 and class_probability[chosen] < favorite_probability + margin:
                    chosen = 0
                if chosen != 0:
                    chosen_position = market_order[chosen]
                    favorite_position = market_order[0]
                    output[chosen_position], output[favorite_position] = (
                        output[favorite_position],
                        output[chosen_position],
                    )
            outputs[float(margin)] = race_normalize(frame, output)
        return outputs


def _challenger_race_frame(
    frame: pd.DataFrame,
    runner_columns: list[str],
    top_k: int,
    include_target: bool,
) -> tuple[pd.DataFrame, list[str]]:
    market = market_probability(frame)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    rows = []
    race_keys = []
    numeric = frame[runner_columns].apply(pd.to_numeric, errors="coerce")
    for race_key, race in frame.groupby("rk", sort=False):
        race_positions = positions.loc[race.index].to_numpy(dtype=int)
        order = race_positions[np.argsort(-market[race_positions], kind="stable")]
        selected = order[: min(top_k, len(order))]
        favorite_values = numeric.iloc[selected[0]]
        row: dict[str, float | int] = {
            "field_size": float(len(order)),
            "favorite_probability": float(market[selected[0]]),
            "favorite_gap": float(market[selected[0]] - market[selected[1]]) if len(selected) > 1 else 1.0,
        }
        for rank in range(top_k):
            if rank < len(selected):
                position = selected[rank]
                row[f"market_p_{rank}"] = float(market[position])
                values = numeric.iloc[position]
                for column in runner_columns:
                    row[f"r{rank}_{column}"] = float(values[column]) if pd.notna(values[column]) else np.nan
                    if rank > 0:
                        favorite_value = favorite_values[column]
                        row[f"d{rank}_{column}"] = (
                            float(values[column] - favorite_value)
                            if pd.notna(values[column]) and pd.notna(favorite_value)
                            else np.nan
                        )
            else:
                row[f"market_p_{rank}"] = 0.0
                for column in runner_columns:
                    row[f"r{rank}_{column}"] = np.nan
                    if rank > 0:
                        row[f"d{rank}_{column}"] = np.nan
        if include_target:
            winner_positions = race_positions[race["win"].to_numpy(dtype=int) == 1]
            winner = int(winner_positions[0])
            winner_rank = np.flatnonzero(order == winner)
            row["target"] = int(winner_rank[0]) if len(winner_rank) and winner_rank[0] < top_k else top_k
        rows.append(row)
        race_keys.append(str(race_key))
    return pd.DataFrame(rows), race_keys


def fit_favorite_challenger(
    train: pd.DataFrame,
    runner_columns: list[str],
    top_k: int,
    class_balance_power: float = 0.0,
) -> FavoriteChallengerModel:
    race_frame, _ = _challenger_race_frame(train, runner_columns, top_k, include_target=True)
    feature_columns = [column for column in race_frame if column != "target"]
    median = race_frame[feature_columns].median(numeric_only=True)
    values = _numeric_values(race_frame, feature_columns, median)
    target = race_frame["target"].to_numpy(dtype=int)
    counts = pd.Series(target).value_counts()
    weights = np.asarray(
        [(len(target) / counts[label]) ** class_balance_power for label in target],
        dtype=float,
    )
    estimator = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.035,
        max_iter=450,
        min_samples_leaf=60,
        l2_regularization=6.0,
        random_state=20260711,
    ).fit(values, target, sample_weight=weights)
    return FavoriteChallengerModel(
        estimator, median, feature_columns, runner_columns, top_k
    )
