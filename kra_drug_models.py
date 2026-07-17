from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from kra_full_order_rankers import SPECS as FULL_ORDER_SPECS
from kra_full_order_rankers import fit_full_order_model
from kra_global_rankers import SPECS as GLOBAL_SPECS
from kra_global_rankers import fit_global_model
from kra_max_winrate_models import SPECS as MAX_SPECS
from kra_max_winrate_models import fit_candidate, predict_candidate


BASE_MODEL_NAMES: Final = (
    "v4",
    *(f"sectional_{spec.name}" for spec in MAX_SPECS),
    *(f"global_{spec.name}" for spec in GLOBAL_SPECS),
    *(f"full_order_{spec.name}" for spec in FULL_ORDER_SPECS),
)


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    names: tuple[str, ...]
    fold_predictions: tuple[np.ndarray, ...]
    fingerprint: str
    cache_hit: bool


def _fingerprint(folds: list, columns: list[str]) -> str:
    payload = {
        "version": 3,
        "models": BASE_MODEL_NAMES,
        "columns": columns,
        "folds": [
            {
                "name": fold.name,
                "train_rows": len(fold.train),
                "test_rows": len(fold.test),
                "train_min": str(fold.train["rcDate"].min()),
                "train_max": str(fold.train["rcDate"].max()),
                "test_min": str(fold.test["rcDate"].min()),
                "test_max": str(fold.test["rcDate"].max()),
            }
            for fold in folds
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(
    path: Path,
    fingerprint: str,
    fold_count: int,
) -> PredictionBundle | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            cached_fingerprint = str(cache["fingerprint"].item())
            names = tuple(str(value) for value in cache["names"].tolist())
            predictions = tuple(
                np.asarray(cache[f"fold_{index}"], dtype=float)
                for index in range(fold_count)
            )
    except (OSError, ValueError, KeyError):
        return None
    if cached_fingerprint != fingerprint or names != BASE_MODEL_NAMES:
        return None
    return PredictionBundle(names, predictions, fingerprint, True)


def _fit_fold(fold, columns: list[str]) -> np.ndarray:
    predictions = [np.asarray(fold.baseline_probability, dtype=float)]
    for spec in MAX_SPECS:
        estimator, median = fit_candidate(fold.train, columns, spec)
        predictions.append(
            predict_candidate(estimator, median, fold.test, columns, spec.family)
        )
    for spec in GLOBAL_SPECS:
        predictions.append(
            fit_global_model(fold.train, columns, spec).predict(fold.test)
        )
    for spec in FULL_ORDER_SPECS:
        predictions.append(
            fit_full_order_model(fold.train, columns, spec).predict(fold.test)
        )
    return np.vstack(predictions)


def load_or_fit_predictions(
    folds: list,
    columns: list[str],
    cache_path: Path,
    refresh: bool,
) -> PredictionBundle:
    fingerprint = _fingerprint(folds, columns)
    cached = None if refresh else _load_cache(cache_path, fingerprint, len(folds))
    if cached is not None:
        return cached
    fold_predictions = tuple(_fit_fold(fold, columns) for fold in folds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "fingerprint": np.asarray(fingerprint),
        "names": np.asarray(BASE_MODEL_NAMES),
        **{
            f"fold_{index}": predictions
            for index, predictions in enumerate(fold_predictions)
        },
    }
    np.savez_compressed(cache_path, **arrays)
    return PredictionBundle(BASE_MODEL_NAMES, fold_predictions, fingerprint, False)
