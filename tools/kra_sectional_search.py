#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — chronological scikit-learn contract


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_max_winrate_models import SPECS, fit_candidate, predict_candidate  # noqa: E402
from kra_model_evaluation import (  # noqa: E402
    as_dict,
    market_probability,
    metrics,
    paired_bootstrap_top1,
    race_normalize,
)
from kra_promotion_policy import MIN_ABSOLUTE_TOP1_LIFT_PP, clears_absolute_top1_lift  # noqa: E402
from kra_sectional_features import build_sectional_features  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402
from tools.kra_max_winrate_search import (  # noqa: E402
    FRESH_FROM,
    WEIGHTS,
    _candidate_evidence,
    _fold_data,
    select_candidate,
)


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h16_results.json"


def _market_diagnostics(
    folds: list,
    selected_scores: list[np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows = []
    for fold, selected_score in zip(folds, selected_scores):
        market = market_probability(fold.test)
        candidate_metrics = metrics(fold.test, selected_score)
        market_metrics = metrics(fold.test, market)
        rows.append({
            "fold": fold.name,
            "races": market_metrics.races,
            "candidate_top1": candidate_metrics.top1,
            "market_top1": market_metrics.top1,
            "candidate_vs_market_top1_pp": (
                candidate_metrics.top1 - market_metrics.top1
            ) * 100.0,
            "market_uses_retrospective_closing_odds": True,
        })
    return rows


def run_search(db_path: Path) -> dict:  # noqa: DICT_OK — JSON research report
    source = load_rows(db_path)
    baseline_frame, baseline_columns = build_features(source)
    frame, sectional_columns = build_sectional_features(
        baseline_frame, baseline_columns
    )
    folds = _fold_data(frame, baseline_columns)
    candidates = []
    score_cache: dict[tuple[str, float], list[np.ndarray]] = {}
    for spec in SPECS:
        probabilities = []
        for fold in folds:
            estimator, median = fit_candidate(fold.train, sectional_columns, spec)
            probabilities.append(
                predict_candidate(
                    estimator,
                    median,
                    fold.test,
                    sectional_columns,
                    spec.family,
                )
            )
        for weight in WEIGHTS:
            evidence, scores = _candidate_evidence(
                folds, probabilities, spec.name, weight
            )
            candidates.append(evidence)
            score_cache[(spec.name, weight)] = scores

    selected = select_candidate(tuple(candidates))
    selected_scores = score_cache[(selected.name, selected.weight)]
    pooled_frame = pd.concat([fold.test for fold in folds], ignore_index=True)
    pooled_baseline = np.concatenate([fold.baseline_probability for fold in folds])
    pooled_selected = np.concatenate(selected_scores)
    pooled_bootstrap = paired_bootstrap_top1(
        pooled_frame, pooled_selected, pooled_baseline, samples=10000
    )
    pooled_logloss_delta = (
        metrics(pooled_frame, pooled_selected).race_logloss
        - metrics(pooled_frame, pooled_baseline).race_logloss
    )

    spec = next(item for item in SPECS if item.name == selected.name)
    fresh_train = frame[frame["rcDate"] < FRESH_FROM]
    fresh = frame[frame["rcDate"] >= FRESH_FROM].copy()
    baseline_estimator, baseline_median = _fit(
        fresh_train, baseline_columns, "win", CONFIGS[0]
    )
    fresh_baseline = race_normalize(
        fresh,
        _predict(baseline_estimator, baseline_median, fresh, baseline_columns),
    )
    candidate_estimator, candidate_median = fit_candidate(
        fresh_train, sectional_columns, spec
    )
    fresh_raw = predict_candidate(
        candidate_estimator,
        candidate_median,
        fresh,
        sectional_columns,
        spec.family,
    )
    fresh_selected = (
        (1.0 - selected.weight) * fresh_baseline
        + selected.weight * fresh_raw
    )
    fresh_baseline_metrics = metrics(fresh, fresh_baseline)
    fresh_candidate_metrics = metrics(fresh, fresh_selected)
    fresh_market = market_probability(fresh)
    fresh_market_metrics = metrics(fresh, fresh_market)
    fresh_bootstrap = paired_bootstrap_top1(
        fresh, fresh_selected, fresh_baseline, samples=10000
    )
    metric_gate = (
        clears_absolute_top1_lift(pooled_bootstrap["mean_pp"])
        and clears_absolute_top1_lift(fresh_bootstrap["mean_pp"])
        and pooled_bootstrap["ci95_low_pp"] > 0.0
        and fresh_bootstrap["ci95_low_pp"] > 0.0
        and all(lift > 0.0 for lift in selected.top1_lifts_pp)
        and all(lift >= 0.0 for lift in selected.top3_lifts_pp)
        and pooled_logloss_delta <= 0.0
        and fresh_candidate_metrics.top3 >= fresh_baseline_metrics.top3
        and fresh_candidate_metrics.race_logloss <= fresh_baseline_metrics.race_logloss
    )
    return {
        "method": "venue_aware_sectional_efficiency_chronological_search",
        "baseline": "kra_dual_phase_v4_history",
        "odds_in_candidate": False,
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "feature_count": len(sectional_columns),
        "selection_folds": [fold.name for fold in folds[:3]],
        "confirmation_fold": folds[3].name,
        "fresh_holdout_is_pristine": False,
        "fresh_holdout_note": "20260622-20260711 was already inspected; promotion requires later races",
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
                "retrospective_market": as_dict(fresh_market_metrics),
                "candidate_vs_market_top1_pp": (
                    fresh_candidate_metrics.top1 - fresh_market_metrics.top1
                ) * 100.0,
            },
            "metric_gate_without_pristine_requirement": metric_gate,
            "promotion_pass": False,
        },
        "retrospective_market_diagnostics": _market_diagnostics(
            folds, selected_scores
        ),
        "candidates": [asdict(candidate) for candidate in candidates],
        "promotion_pass": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_search(args.db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
