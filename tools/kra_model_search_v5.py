from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — scikit-learn walk-forward contract


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_model_candidates import ModelSpec, fit_candidate  # noqa: E402
from kra_model_evaluation import as_dict, metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_promotion_policy import clears_absolute_top1_lift  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402


DEFAULT_REPORT = ROOT / "runs" / "kra_model_search_v5_results.json"
FOLDS = (
    ("2024H2", "20240101", "20240701", "20250101"),
    ("2025H1", "20240101", "20250101", "20250701"),
    ("2025H2", "20240101", "20250701", "20260101"),
    ("2026H1", "20240101", "20260101", "20260701"),
)
SPECS = (
    ModelSpec("hgb_d2_leaf80", "hgb", 2, 400, 80, 1.0, 0.05, 0.5),
    ModelSpec("hgb_d2_leaf160", "hgb", 2, 400, 160, 1.0, 0.05, 2.0),
    ModelSpec("hgb_d3_leaf60", "hgb", 3, 400, 60, 1.0, 0.04, 1.0),
    ModelSpec("hgb_d3_leaf180", "hgb", 3, 400, 180, 1.0, 0.04, 3.0),
    ModelSpec("hgb_d4_leaf120", "hgb", 4, 350, 120, 1.0, 0.04, 2.0),
    ModelSpec("hgb_d3_subsample", "hgb", 3, 450, 100, 0.7, 0.04, 2.0),
    ModelSpec("extra_d8_leaf10", "extra_trees", 8, 500, 10, 0.8),
    ModelSpec("extra_d12_leaf20", "extra_trees", 12, 500, 20, 0.8),
    ModelSpec("extra_full_leaf40", "extra_trees", None, 500, 40, 0.8),
    ModelSpec("forest_d10_leaf10", "random_forest", 10, 500, 10, 0.8),
    ModelSpec("forest_d14_leaf25", "random_forest", 14, 500, 25, 0.8),
    ModelSpec("forest_full_leaf50", "random_forest", None, 500, 50, 0.8),
)
WEIGHTS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True, slots=True)
class FoldData:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    baseline_probability: np.ndarray


class ModelSearchReport(TypedDict):
    method: str
    baseline: str
    candidates: dict[str, dict]
    selected: str | None
    selected_result: dict | None
    promotion_pass: bool


def _fold_data(frame: pd.DataFrame, columns: list[str]) -> list[FoldData]:
    result = []
    for name, train_from, test_from, test_until in FOLDS:
        train = frame[(frame["rcDate"] >= train_from) & (frame["rcDate"] < test_from)]
        test = frame[(frame["rcDate"] >= test_from) & (frame["rcDate"] < test_until)].copy()
        model, median = _fit(train, columns, "win", CONFIGS[0])
        probability = race_normalize(test, _predict(model, median, test, columns))
        result.append(FoldData(name, train, test, probability))
    return result


def _weight_result(folds: list[FoldData], candidates: list[np.ndarray], weight: float) -> dict:  # noqa: DICT_OK — JSON experiment row
    fold_rows = []
    pooled_frames = []
    pooled_candidate = []
    pooled_baseline = []
    for fold, candidate in zip(folds, candidates):
        blended = (1.0 - weight) * fold.baseline_probability + weight * candidate
        baseline_metrics = metrics(fold.test, fold.baseline_probability)
        candidate_metrics = metrics(fold.test, blended)
        fold_rows.append({
            "fold": fold.name,
            "top1_lift_pp": (candidate_metrics.top1 - baseline_metrics.top1) * 100.0,
            "top3_lift_pp": (candidate_metrics.top3 - baseline_metrics.top3) * 100.0,
            "logloss_delta": candidate_metrics.race_logloss - baseline_metrics.race_logloss,
            "candidate": as_dict(candidate_metrics),
        })
        pooled_frames.append(fold.test)
        pooled_candidate.append(blended)
        pooled_baseline.append(fold.baseline_probability)
    pooled_frame = pd.concat(pooled_frames, ignore_index=True)
    candidate_values = np.concatenate(pooled_candidate)
    baseline_values = np.concatenate(pooled_baseline)
    bootstrap = paired_bootstrap_top1(pooled_frame, candidate_values, baseline_values, 10000)
    logloss_delta = (
        metrics(pooled_frame, candidate_values).race_logloss
        - metrics(pooled_frame, baseline_values).race_logloss
    )
    selection = fold_rows[:3]
    promotion_pass = (
        clears_absolute_top1_lift(bootstrap["mean_pp"])
        and all(row["top1_lift_pp"] > 0 for row in fold_rows)
        and all(row["top3_lift_pp"] >= 0 for row in fold_rows)
        and bootstrap["ci95_low_pp"] > 0
        and logloss_delta <= 0
    )
    return {
        "weight": weight,
        "folds": fold_rows,
        "selection_mean_top1_lift_pp": float(np.mean([row["top1_lift_pp"] for row in selection])),
        "selection_min_top1_lift_pp": min(row["top1_lift_pp"] for row in selection),
        "selection_min_top3_lift_pp": min(row["top3_lift_pp"] for row in selection),
        "pooled_bootstrap": bootstrap,
        "pooled_logloss_delta": logloss_delta,
        "promotion_pass": promotion_pass,
    }


def run_search(db_path: Path) -> ModelSearchReport:
    frame, columns = build_features(load_rows(db_path))
    folds = _fold_data(frame, columns)
    results = {}
    for spec in SPECS:
        probabilities = []
        for fold in folds:
            model, median = fit_candidate(fold.train, columns, "win", spec)
            values = fold.test[columns].apply(pd.to_numeric, errors="coerce").fillna(median)
            probabilities.append(race_normalize(fold.test, model.predict_proba(values)[:, 1]))
        weight_results = [_weight_result(folds, probabilities, weight) for weight in WEIGHTS]
        results[spec.name] = max(
            weight_results,
            key=lambda result: (
                result["selection_min_top1_lift_pp"],
                result["selection_mean_top1_lift_pp"],
                result["selection_min_top3_lift_pp"],
                -result["pooled_logloss_delta"],
            ),
        )
    selected, selected_result = max(
        results.items(),
        key=lambda item: (
            item[1]["selection_min_top1_lift_pp"],
            item[1]["selection_mean_top1_lift_pp"],
            item[1]["selection_min_top3_lift_pp"],
        ),
        default=(None, None),
    )
    return {
        "method": "model_and_ensemble_selection_first_three_folds_confirmation_fourth",
        "baseline": "kra_dual_phase_v4_history_hgb_d3",
        "candidates": results,
        "selected": selected,
        "selected_result": selected_result,
        "promotion_pass": bool(selected_result and selected_result["promotion_pass"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_search(args.db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
