from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — grouped race evaluation contract


@dataclass(frozen=True, slots=True)
class Metrics:
    races: int
    top1: float
    top3: float
    race_logloss: float


def race_normalize(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    raw = pd.Series(np.clip(probability, 1e-9, None), index=frame.index)
    return (raw / raw.groupby(frame["rk"]).transform("sum")).to_numpy()


def market_probability(frame: pd.DataFrame) -> np.ndarray:
    return race_normalize(frame, (1.0 / frame["winOdds"]).to_numpy())


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> Metrics:
    scored = frame[["rk", "win"]].copy()
    scored["p"] = race_normalize(frame, probability)
    ranked = scored.sort_values(["rk", "p"], ascending=[True, False])
    winners = ranked[ranked["win"] == 1]
    winner_probability = np.clip(winners["p"].to_numpy(), 1e-12, 1.0)
    rank = ranked.groupby("rk").cumcount() + 1
    return Metrics(
        races=int(scored["rk"].nunique()),
        top1=float(ranked[rank == 1]["win"].mean()),
        top3=float(ranked[rank <= 3].groupby("rk")["win"].max().mean()),
        race_logloss=float(-np.log(winner_probability).mean()),
    )


def paired_bootstrap_top1(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    baseline: np.ndarray,
    samples: int = 4000,
) -> dict[str, float]:
    scored = frame[["rk", "win"]].copy()
    scored["candidate"] = candidate
    scored["baseline"] = baseline
    differences = []
    for _, race in scored.groupby("rk", sort=False):
        candidate_hit = int(race.loc[race["candidate"].idxmax(), "win"])
        baseline_hit = int(race.loc[race["baseline"].idxmax(), "win"])
        differences.append(candidate_hit - baseline_hit)
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(20260711)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    distribution = values[indices].mean(axis=1)
    return {
        "mean_pp": float(values.mean() * 100.0),
        "ci95_low_pp": float(np.quantile(distribution, 0.025) * 100.0),
        "ci95_high_pp": float(np.quantile(distribution, 0.975) * 100.0),
        "probability_positive": float((distribution > 0).mean()),
    }


def selective_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    scored = frame[["rk", "win"]].copy()
    scored["p"] = race_normalize(frame, probability)
    leaders = scored.loc[scored.groupby("rk")["p"].idxmax()]
    selected = leaders[leaders["p"] >= threshold]
    return {
        "races": int(len(selected)),
        "coverage": float(len(selected) / len(leaders)),
        "top1": float(selected["win"].mean()),
        "threshold": float(threshold),
    }


def leader_threshold(frame: pd.DataFrame, probability: np.ndarray, coverage: float = 0.25) -> float:
    scored = frame[["rk"]].copy()
    scored["p"] = race_normalize(frame, probability)
    leaders = scored.loc[scored.groupby("rk")["p"].idxmax(), "p"]
    return float(leaders.quantile(1.0 - coverage))


def as_dict(value: Metrics) -> dict[str, float | int]:
    return {
        "races": value.races,
        "top1": value.top1,
        "top3": value.top3,
        "race_logloss": value.race_logloss,
    }
