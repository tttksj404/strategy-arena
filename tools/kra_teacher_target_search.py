from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — chronological experiment-frame contract


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_diversified_rankers import (  # noqa: E402
    fit_market_distillation,
    fit_market_favorite,
    fit_market_favorite_conditional_logit,
    fit_place_distillation,
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


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h10_results.json"
BASE_CANDIDATES: Final = (
    "market_probability_teacher",
    "market_favorite_classifier",
    "place_probability_teacher",
    "market_favorite_listwise",
    "segmented_market_favorite",
)
ENSEMBLES: Final = {
    "favorite_probability_consensus": (
        "market_probability_teacher",
        "market_favorite_classifier",
    ),
    "win_place_market_consensus": (
        "market_probability_teacher",
        "place_probability_teacher",
    ),
    "favorite_architecture_consensus": (
        "market_favorite_classifier",
        "market_favorite_listwise",
        "segmented_market_favorite",
    ),
    "all_market_teachers": BASE_CANDIDATES,
}


def _fit_base_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> dict[str, np.ndarray]:
    return {
        "market_probability_teacher": fit_market_distillation(train, columns).predict(test),
        "market_favorite_classifier": fit_market_favorite(train, columns).predict(test),
        "place_probability_teacher": fit_place_distillation(train, columns).predict(test),
        "market_favorite_listwise": fit_market_favorite_conditional_logit(
            train, columns
        ).predict(test),
        "segmented_market_favorite": fit_segmented_experts(
            train, columns, "favorite"
        ).predict(test),
    }


def _candidate_predictions(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(base)
    for name, members in ENSEMBLES.items():
        result[name] = np.mean([base[member] for member in members], axis=0)
    return result


def _fresh_diagnostics(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    baseline_metrics = metrics(frame, baseline)
    rows = []
    for name in (*BASE_CANDIDATES, *ENSEMBLES):
        for weight in WEIGHTS:
            score = (1.0 - weight) * baseline + weight * predictions[name]
            candidate_metrics = metrics(frame, score)
            rows.append({
                "name": name,
                "weight": weight,
                "top1_lift_pp": (candidate_metrics.top1 - baseline_metrics.top1) * 100.0,
                "top3_lift_pp": (candidate_metrics.top3 - baseline_metrics.top3) * 100.0,
                "logloss_delta": candidate_metrics.race_logloss - baseline_metrics.race_logloss,
                "bootstrap": paired_bootstrap_top1(frame, score, baseline, samples=10000),
            })
    return sorted(
        rows,
        key=lambda row: (
            row["top1_lift_pp"],
            row["top3_lift_pp"],
            -row["logloss_delta"],
        ),
        reverse=True,
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
        _candidate_predictions(_fit_base_predictions(fold.train, fold.test, candidate_columns))
        for fold in folds
    ]
    candidates = []
    score_cache: dict[tuple[str, float], list[np.ndarray]] = {}
    for name in (*BASE_CANDIDATES, *ENSEMBLES):
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
    baseline_estimator, baseline_median = _fit(
        fresh_train, baseline_columns, "win", CONFIGS[0]
    )
    fresh_baseline = race_normalize(
        fresh, _predict(baseline_estimator, baseline_median, fresh, baseline_columns)
    )
    fresh_predictions = _candidate_predictions(
        _fit_base_predictions(fresh_train, fresh, candidate_columns)
    )
    diagnostics = _fresh_diagnostics(fresh, fresh_baseline, fresh_predictions)
    selected_fresh = next(
        row for row in diagnostics
        if row["name"] == selected.name and row["weight"] == selected.weight
    )
    fresh_selected = (
        (1.0 - selected.weight) * fresh_baseline
        + selected.weight * fresh_predictions[selected.name]
    )
    fresh_baseline_metrics = metrics(fresh, fresh_baseline)
    fresh_candidate_metrics = metrics(fresh, fresh_selected)
    metric_gate = (
        clears_absolute_top1_lift(pooled_bootstrap["mean_pp"])
        and clears_absolute_top1_lift(selected_fresh["bootstrap"]["mean_pp"])
        and pooled_bootstrap["ci95_low_pp"] > 0
        and selected_fresh["bootstrap"]["ci95_low_pp"] > 0
        and all(lift > 0 for lift in selected.top1_lifts_pp)
        and all(lift >= 0 for lift in selected.top3_lifts_pp)
        and pooled_logloss_delta <= 0
        and fresh_candidate_metrics.top3 >= fresh_baseline_metrics.top3
        and fresh_candidate_metrics.race_logloss <= fresh_baseline_metrics.race_logloss
    )
    return {
        "method": "historical_market_target_diversification_without_current_race_odds",
        "baseline": "kra_dual_phase_v4_history",
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "feature_count": len(candidate_columns),
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
                "bootstrap": selected_fresh["bootstrap"],
            },
            "metric_gate_without_pristine_requirement": metric_gate,
            "promotion_pass": False,
        },
        "fresh_diagnostics": diagnostics,
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
