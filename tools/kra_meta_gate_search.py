from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — nested chronological experiment contract
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_diversified_rankers import (  # noqa: E402
    fit_market_distillation,
    fit_place_distillation,
    fit_race_balanced,
    fit_segmented_experts,
)
from kra_model_evaluation import as_dict, metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_promotion_policy import MIN_ABSOLUTE_TOP1_LIFT_PP, clears_absolute_top1_lift  # noqa: E402
from tools.kra_diversified_search import (  # noqa: E402
    CONDITIONING_ARCHIVE,
    ENTRY_ARCHIVE,
    HEALTH_ARCHIVE,
    build_diversified_frame,
)
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402
from tools.kra_max_winrate_search import (  # noqa: E402
    FRESH_FROM,
    WEIGHTS,
    _candidate_evidence,
    _fold_data,
    select_candidate,
)


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h11_results.json"
MODEL_NAMES: Final = (
    "v4",
    "race_balanced",
    "market_teacher",
    "place_teacher",
    "segmented_market_favorite",
)


def _base_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_columns: list[str],
    candidate_columns: list[str],
) -> dict[str, np.ndarray]:
    baseline_model, baseline_median = _fit(train, baseline_columns, "win", CONFIGS[0])
    return {
        "v4": race_normalize(
            test, _predict(baseline_model, baseline_median, test, baseline_columns)
        ),
        "race_balanced": fit_race_balanced(train, candidate_columns).predict(test),
        "market_teacher": fit_market_distillation(train, candidate_columns).predict(test),
        "place_teacher": fit_place_distillation(train, candidate_columns).predict(test),
        "segmented_market_favorite": fit_segmented_experts(
            train, candidate_columns, "favorite"
        ).predict(test),
    }


def _gate_rows(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    include_target: bool,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    scored = frame[["rk", "win", "field_size", "rcDist", "rating", "meet"]].copy()
    for name, values in predictions.items():
        scored[name] = race_normalize(frame, values)
    rows = []
    keys = []
    for race_key, race in scored.groupby("rk", sort=False):
        agreement = {
            name: int(race[name].idxmax()) for name in MODEL_NAMES
        }
        agreement_count = pd.Series(list(agreement.values())).value_counts()
        rating_spread = pd.to_numeric(race["rating"], errors="coerce").max() - pd.to_numeric(
            race["rating"], errors="coerce"
        ).min()
        for candidate_index, name in enumerate(MODEL_NAMES):
            values = race[name].sort_values(ascending=False)
            pick_index = int(values.index[0])
            probability = float(values.iloc[0])
            gap = probability - float(values.iloc[1]) if len(values) > 1 else probability
            entropy = -float(np.sum(np.clip(race[name], 1e-12, 1.0) * np.log(np.clip(race[name], 1e-12, 1.0))))
            row = {
                "candidate_id": float(candidate_index),
                "top_probability": probability,
                "top_gap": gap,
                "entropy": entropy,
                "agreement": float(agreement_count.get(pick_index, 0) / len(MODEL_NAMES)),
                "field_size": float(race["field_size"].iloc[0]),
                "distance": float(race["rcDist"].iloc[0]),
                "rating_spread": float(rating_spread) if pd.notna(rating_spread) else 0.0,
                "meet_code": float({"서울": 1, "1": 1, "제주": 2, "2": 2, "부경": 3, "부산경남": 3, "3": 3}.get(str(race["meet"].iloc[0]), 0)),
            }
            for other_index, other in enumerate(MODEL_NAMES):
                row[f"pick_probability_{other_index}"] = float(race.loc[pick_index, other])
            if include_target:
                row["target"] = int(race.loc[pick_index, "win"])
            rows.append(row)
            keys.append((str(race_key), name))
    return pd.DataFrame(rows), keys


def _gated_predictions(
    train_frame: pd.DataFrame,
    train_predictions: dict[str, np.ndarray],
    test_frame: pd.DataFrame,
    test_predictions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)
    gate_train, _ = _gate_rows(train_frame, train_predictions, include_target=True)
    features = [column for column in gate_train.columns if column != "target"]
    gate = HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.04,
        max_iter=350,
        min_samples_leaf=80,
        l2_regularization=6.0,
        random_state=20260711,
    ).fit(gate_train[features], gate_train["target"])
    gate_test, keys = _gate_rows(test_frame, test_predictions, include_target=False)
    correctness = gate.predict_proba(gate_test[features])[:, 1]
    gate_test = gate_test.assign(correctness=correctness, key=keys)

    hard = np.zeros(len(test_frame), dtype=float)
    soft = np.zeros(len(test_frame), dtype=float)
    conservative = np.asarray(test_predictions["v4"], dtype=float).copy()
    for race_key, race in test_frame.groupby("rk", sort=False):
        rows = gate_test[[key[0] == str(race_key) for key in gate_test["key"]]]
        best = rows.loc[rows["correctness"].idxmax()]
        chosen_name = str(best["key"][1])
        hard[race.index] = test_predictions[chosen_name][race.index]
        raw_weights = rows["correctness"].to_numpy(dtype=float)
        raw_weights = np.clip(raw_weights - 0.15, 1e-6, None)
        raw_weights /= raw_weights.sum()
        soft[race.index] = sum(
            weight * test_predictions[name][race.index]
            for weight, name in zip(raw_weights, MODEL_NAMES)
        )
        baseline_row = rows[rows["key"].map(lambda key: key[1] == "v4")].iloc[0]
        if float(best["correctness"]) >= float(baseline_row["correctness"]) + 0.04:
            conservative[race.index] = test_predictions[chosen_name][race.index]
    return {
        "hard_meta_gate": race_normalize(test_frame, hard),
        "soft_meta_gate": race_normalize(test_frame, soft),
        "conservative_meta_gate": race_normalize(test_frame, conservative),
    }


def _nested_fold_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_columns: list[str],
    candidate_columns: list[str],
) -> dict[str, np.ndarray]:
    dates = np.asarray(sorted(train["rcDate"].unique()))
    cutoff = dates[max(1, int(len(dates) * 0.65))]
    inner_train = train[train["rcDate"] < cutoff]
    gate_train = train[train["rcDate"] >= cutoff].copy()
    inner_predictions = _base_predictions(
        inner_train, gate_train, baseline_columns, candidate_columns
    )
    outer_predictions = _base_predictions(
        train, test, baseline_columns, candidate_columns
    )
    return _gated_predictions(
        gate_train,
        inner_predictions,
        test,
        outer_predictions,
    )


def run_search(
    db_path: Path,
    entry_archive: Path,
    conditioning_archive: Path,
    health_archive: Path,
) -> dict:  # noqa: DICT_OK — JSON research report
    frame, baseline_columns, candidate_columns = build_diversified_frame(
        db_path, entry_archive, conditioning_archive, health_archive
    )
    folds = _fold_data(frame, baseline_columns)
    fold_predictions = [
        _nested_fold_predictions(fold.train, fold.test, baseline_columns, candidate_columns)
        for fold in folds
    ]
    candidates = []
    score_cache: dict[tuple[str, float], list[np.ndarray]] = {}
    for name in ("hard_meta_gate", "soft_meta_gate", "conservative_meta_gate"):
        probabilities = [prediction[name] for prediction in fold_predictions]
        for weight in WEIGHTS:
            evidence, scores = _candidate_evidence(folds, probabilities, name, weight)
            candidates.append(evidence)
            score_cache[(name, weight)] = scores
    selected = select_candidate(tuple(candidates))
    selected_scores = score_cache[(selected.name, selected.weight)]
    pooled_frame = pd.concat([fold.test for fold in folds], ignore_index=True)
    pooled_baseline = np.concatenate([fold.baseline_probability for fold in folds])
    pooled_selected = np.concatenate(selected_scores)
    pooled_bootstrap = paired_bootstrap_top1(
        pooled_frame, pooled_selected, pooled_baseline, samples=10000
    )
    pooled_logloss_delta = metrics(pooled_frame, pooled_selected).race_logloss - metrics(
        pooled_frame, pooled_baseline
    ).race_logloss

    fresh_train = frame[frame["rcDate"] < FRESH_FROM]
    fresh = frame[frame["rcDate"] >= FRESH_FROM].copy()
    baseline_model, baseline_median = _fit(
        fresh_train, baseline_columns, "win", CONFIGS[0]
    )
    fresh_baseline = race_normalize(
        fresh, _predict(baseline_model, baseline_median, fresh, baseline_columns)
    )
    fresh_gate = _nested_fold_predictions(
        fresh_train, fresh, baseline_columns, candidate_columns
    )
    fresh_selected = (
        (1.0 - selected.weight) * fresh_baseline
        + selected.weight * fresh_gate[selected.name]
    )
    fresh_baseline_metrics = metrics(fresh, fresh_baseline)
    fresh_candidate_metrics = metrics(fresh, fresh_selected)
    fresh_bootstrap = paired_bootstrap_top1(
        fresh, fresh_selected, fresh_baseline, samples=10000
    )
    metric_gate = (
        clears_absolute_top1_lift(pooled_bootstrap["mean_pp"])
        and clears_absolute_top1_lift(fresh_bootstrap["mean_pp"])
        and pooled_bootstrap["ci95_low_pp"] > 0
        and fresh_bootstrap["ci95_low_pp"] > 0
        and all(lift > 0 for lift in selected.top1_lifts_pp)
        and all(lift >= 0 for lift in selected.top3_lifts_pp)
        and pooled_logloss_delta <= 0
        and fresh_candidate_metrics.top3 >= fresh_baseline_metrics.top3
        and fresh_candidate_metrics.race_logloss <= fresh_baseline_metrics.race_logloss
    )
    return {
        "method": "nested_chronological_race_level_model_meta_gate",
        "baseline": "kra_dual_phase_v4_history",
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "selection_folds": [fold.name for fold in folds[:3]],
        "confirmation_fold": folds[3].name,
        "fresh_holdout_is_pristine": False,
        "fresh_holdout_note": "20260622-20260711 was already inspected; promotion requires results after 20260711",
        "selected": f"{selected.name}@{selected.weight:.2f}",
        "selected_result": {
            **asdict(selected),
            "pooled_bootstrap": pooled_bootstrap,
            "pooled_logloss_delta": pooled_logloss_delta,
            "fresh_holdout": {
                "from": FRESH_FROM,
                "baseline": as_dict(fresh_baseline_metrics),
                "candidate": as_dict(fresh_candidate_metrics),
                "bootstrap": fresh_bootstrap,
            },
            "metric_gate_without_pristine_requirement": metric_gate,
            "promotion_pass": False,
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "promotion_pass": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--entry-archive", type=Path, default=ENTRY_ARCHIVE)
    parser.add_argument("--conditioning-archive", type=Path, default=CONDITIONING_ARCHIVE)
    parser.add_argument("--health-archive", type=Path, default=HEALTH_ARCHIVE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_search(
        args.db,
        args.entry_archive,
        args.conditioning_archive,
        args.health_archive,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
