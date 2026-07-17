from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — chronological grouped-ranking contract
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from kra_model_evaluation import market_probability, race_normalize


PACE_FEATURES: Final = (
    "pace_front_tendency",
    "pace_pressure",
    "pace_front_pressure",
    "pace_closer_opportunity",
    "consensus_strength",
    "consensus_disagreement",
    "rating_percentile",
    "elo_percentile",
    "speed_percentile",
    "recent_finish_percentile",
)


def add_pace_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    early = pd.to_numeric(result["hr_early_position_mean_3"], errors="coerce")
    gain = pd.to_numeric(result["hr_finish_gain_mean_3"], errors="coerce")
    result["pace_front_tendency"] = 1.0 - early
    front = (early <= 0.35).astype(float).where(early.notna(), 0.0)
    result["pace_pressure"] = front.groupby(result["rk"]).transform("sum")
    result["pace_front_pressure"] = result["pace_front_tendency"] * result["pace_pressure"]
    result["pace_closer_opportunity"] = gain * result["pace_pressure"]

    percentile_specs = (
        ("rating", "rating_percentile", False),
        ("hr_elo", "elo_percentile", False),
        ("hr_speed_mean_3", "speed_percentile", False),
        ("hr_recent_finish_mean_3", "recent_finish_percentile", True),
    )
    percentiles = []
    for source, output, ascending in percentile_specs:
        values = pd.to_numeric(result[source], errors="coerce")
        result[output] = values.groupby(result["rk"]).rank(
            pct=True, ascending=ascending, na_option="bottom"
        )
        percentiles.append(output)
    result["consensus_strength"] = result[percentiles].mean(axis=1)
    result["consensus_disagreement"] = result[percentiles].std(axis=1)
    return result, list(PACE_FEATURES)


def _numeric_values(
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


@dataclass(slots=True)
class RaceBalancedModel:
    estimator: HistGradientBoostingClassifier
    median: pd.Series
    columns: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric_values(frame, self.columns, self.median)
        return race_normalize(frame, self.estimator.predict_proba(values)[:, 1])


def fit_race_balanced(
    train: pd.DataFrame,
    columns: list[str],
) -> RaceBalancedModel:
    median = train[columns].median(numeric_only=True)
    values = _numeric_values(train, columns, median)
    weights = 1.0 / pd.to_numeric(train["field_size"], errors="coerce").clip(lower=2)
    estimator = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.04,
        max_iter=500,
        min_samples_leaf=80,
        l2_regularization=3.0,
        random_state=20260711,
    ).fit(values, train["win"].to_numpy(), sample_weight=weights.to_numpy())
    return RaceBalancedModel(estimator, median, columns)


@dataclass(slots=True)
class MarketDistillationModel:
    estimator: HistGradientBoostingRegressor
    median: pd.Series
    columns: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric_values(frame, self.columns, self.median)
        score = np.exp(np.clip(self.estimator.predict(values), -12.0, 2.0))
        return race_normalize(frame, score)


def fit_market_distillation(
    train: pd.DataFrame,
    columns: list[str],
) -> MarketDistillationModel:
    median = train[columns].median(numeric_only=True)
    values = _numeric_values(train, columns, median)
    target = np.log(np.clip(market_probability(train), 1e-8, 1.0))
    weights = 1.0 / pd.to_numeric(train["field_size"], errors="coerce").clip(lower=2)
    estimator = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=5,
        learning_rate=0.04,
        max_iter=500,
        min_samples_leaf=60,
        l2_regularization=4.0,
        random_state=20260711,
    ).fit(values, target, sample_weight=weights.to_numpy())
    return MarketDistillationModel(estimator, median, columns)


def _race_favorite_target(frame: pd.DataFrame, odds_column: str) -> np.ndarray:
    odds = pd.to_numeric(frame[odds_column], errors="coerce")
    favorite_index = odds.groupby(frame["rk"]).idxmin()
    target = pd.Series(0, index=frame.index, dtype=int)
    target.loc[favorite_index] = 1
    return target.to_numpy()


def fit_market_favorite(
    train: pd.DataFrame,
    columns: list[str],
) -> RaceBalancedModel:
    median = train[columns].median(numeric_only=True)
    values = _numeric_values(train, columns, median)
    target = _race_favorite_target(train, "winOdds")
    weights = 1.0 / pd.to_numeric(train["field_size"], errors="coerce").clip(lower=2)
    estimator = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.04,
        max_iter=550,
        min_samples_leaf=60,
        l2_regularization=4.0,
        random_state=20260711,
    ).fit(values, target, sample_weight=weights.to_numpy())
    return RaceBalancedModel(estimator, median, columns)


def fit_place_distillation(
    train: pd.DataFrame,
    columns: list[str],
) -> MarketDistillationModel:
    median = train[columns].median(numeric_only=True)
    values = _numeric_values(train, columns, median)
    implied = race_normalize(
        train, 1.0 / pd.to_numeric(train["plcOdds"], errors="coerce").to_numpy()
    )
    target = np.log(np.clip(implied, 1e-8, 1.0))
    weights = 1.0 / pd.to_numeric(train["field_size"], errors="coerce").clip(lower=2)
    estimator = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=5,
        learning_rate=0.04,
        max_iter=500,
        min_samples_leaf=60,
        l2_regularization=4.0,
        random_state=20260711,
    ).fit(values, target, sample_weight=weights.to_numpy())
    return MarketDistillationModel(estimator, median, columns)


@dataclass(slots=True)
class ConditionalLogitModel:
    median: pd.Series
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    columns: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric_values(frame, self.columns, self.median)
        standardized = (values - self.mean) / self.scale
        score = np.exp(np.clip(standardized @ self.coefficient, -20.0, 20.0))
        return race_normalize(frame, score)


def fit_conditional_logit(
    train: pd.DataFrame,
    columns: list[str],
    ridge: float = 4.0,
    target: np.ndarray | None = None,
) -> ConditionalLogitModel:
    median = train[columns].median(numeric_only=True)
    values = _numeric_values(train, columns, median)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = np.clip((values - mean) / scale, -10.0, 10.0)
    group_codes, _ = pd.factorize(train["rk"], sort=False)
    group_count = int(group_codes.max()) + 1
    target_values = (
        train["win"].to_numpy(dtype=float)
        if target is None
        else np.asarray(target, dtype=float)
    )

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        scores = standardized @ coefficient
        maxima = np.full(group_count, -np.inf)
        np.maximum.at(maxima, group_codes, scores)
        exponent = np.exp(np.clip(scores - maxima[group_codes], -50.0, 0.0))
        totals = np.zeros(group_count)
        np.add.at(totals, group_codes, exponent)
        probability = exponent / totals[group_codes]
        winner_scores = scores[target_values == 1.0]
        log_denominator = maxima + np.log(totals)
        loss = float(log_denominator.sum() - winner_scores.sum())
        loss += 0.5 * ridge * float(coefficient @ coefficient)
        gradient = standardized.T @ (probability - target_values) + ridge * coefficient
        return loss, gradient

    initial = np.zeros(standardized.shape[1], dtype=float)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 180, "ftol": 1e-8, "maxls": 30},
    )
    return ConditionalLogitModel(median, mean, scale, result.x, columns)


def fit_market_favorite_conditional_logit(
    train: pd.DataFrame,
    columns: list[str],
) -> ConditionalLogitModel:
    return fit_conditional_logit(
        train,
        columns,
        target=_race_favorite_target(train, "winOdds"),
    )


def _segment(frame: pd.DataFrame) -> pd.Series:
    distance = pd.to_numeric(frame["rcDist"], errors="coerce")
    band = pd.cut(
        distance,
        bins=(-np.inf, 1300, 1800, np.inf),
        labels=("sprint", "middle", "route"),
    ).astype(str)
    return frame["meet"].astype(str) + "|" + band


@dataclass(slots=True)
class SegmentedExpertModel:
    global_model: RaceBalancedModel | MarketDistillationModel
    experts: dict[str, RaceBalancedModel | MarketDistillationModel]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.global_model.predict(frame)
        segments = _segment(frame)
        for name, model in self.experts.items():
            mask = segments == name
            if mask.any():
                raw[mask.to_numpy()] = model.predict(frame.loc[mask])
        return race_normalize(frame, raw)


def fit_segmented_experts(
    train: pd.DataFrame,
    columns: list[str],
    target: str,
    minimum_rows: int = 1800,
) -> SegmentedExpertModel:
    fitters = {
        "win": fit_race_balanced,
        "market": fit_market_distillation,
        "favorite": fit_market_favorite,
        "place_market": fit_place_distillation,
    }
    fit = fitters[target]
    global_model = fit(train, columns)
    segments = _segment(train)
    experts = {}
    for name, count in segments.value_counts().items():
        if count >= minimum_rows:
            experts[str(name)] = fit(train.loc[segments == name], columns)
    return SegmentedExpertModel(global_model, experts)
