from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd  # noqa: PANDAS_OK — existing SQLite feature-frame contract

from kra_dynamic_ratings import build_rating_features


PEDIGREE_FEATURES: Final = (
    "sire_win_prior",
    "sire_place_prior",
    "sire_finish_prior",
    "dam_win_prior",
    "dam_place_prior",
    "dam_finish_prior",
)


@dataclass(frozen=True, slots=True)
class PriorSpec:
    parent: str
    target: str
    output: str
    baseline: float
    strength: float


SPECS: Final = (
    PriorSpec("faHrNo", "win", "sire_win_prior", 0.10, 5.0),
    PriorSpec("faHrNo", "place", "sire_place_prior", 0.25, 5.0),
    PriorSpec("faHrNo", "finish_pct", "sire_finish_prior", 0.50, 3.0),
    PriorSpec("moHrNo", "win", "dam_win_prior", 0.10, 5.0),
    PriorSpec("moHrNo", "place", "dam_place_prior", 0.25, 5.0),
    PriorSpec("moHrNo", "finish_pct", "dam_finish_prior", 0.50, 3.0),
)


def _normalized_id(value: str) -> str:
    text = str(value or "").strip()
    return "" if text == "-" else text


def load_pedigree(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as connection:
        pedigree = pd.read_sql_query(
            "SELECT hrNo, faHrNo, moHrNo FROM horse",
            connection,
        )
    for column in ("hrNo", "faHrNo", "moHrNo"):
        pedigree[column] = pedigree[column].map(_normalized_id)
    return pedigree.drop_duplicates("hrNo", keep="last")


def merge_pedigree(frame: pd.DataFrame, pedigree: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hrNo"] = result["hrNo"].map(_normalized_id)
    return result.merge(pedigree, on="hrNo", how="left", validate="many_to_one")


def _prior(frame: pd.DataFrame, spec: PriorSpec) -> pd.Series:
    parent = frame[spec.parent].map(_normalized_id)
    source = frame.assign(_parent=parent)
    valid = source[source["_parent"] != ""]
    daily = (
        valid.groupby(["_parent", "rcDate"], dropna=False)[spec.target]
        .agg([("total", "sum"), ("runs", "count")])
        .reset_index()
        .sort_values(["_parent", "rcDate"])
    )
    grouped = daily.groupby("_parent", dropna=False)
    prior_total = grouped["total"].cumsum() - daily["total"]
    prior_runs = grouped["runs"].cumsum() - daily["runs"]
    daily["prior"] = (
        prior_total + spec.strength * spec.baseline
    ) / (prior_runs + spec.strength)
    lookup = daily.set_index(["_parent", "rcDate"])["prior"]
    values = [lookup.get((identifier, date), spec.baseline) for identifier, date in zip(parent, frame["rcDate"])]
    return pd.Series(values, index=frame.index, dtype=float)


def add_pedigree_priors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for spec in SPECS:
        result[spec.output] = _prior(result, spec)
    return result


def build_pedigree_features(
    frame: pd.DataFrame,
    baseline_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    rated, rated_columns = build_rating_features(frame, baseline_columns)
    result = add_pedigree_priors(rated)
    for column in PEDIGREE_FEATURES:
        result[f"{column}_rel"] = result[column] - result.groupby("rk")[column].transform("mean")
    columns = [
        *rated_columns,
        *PEDIGREE_FEATURES,
        *(f"{column}_rel" for column in PEDIGREE_FEATURES),
    ]
    return result, list(dict.fromkeys(columns))
