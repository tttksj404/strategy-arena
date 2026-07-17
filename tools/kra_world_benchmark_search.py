#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_global_rankers import SPECS, fit_global_model  # noqa: E402
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


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h17_results.json"


def run_search(db_path: Path) -> dict:  # noqa: DICT_OK — JSON research report
    baseline_frame, baseline_columns = build_features(load_rows(db_path))
    frame, candidate_columns = build_sectional_features(
        baseline_frame, baseline_columns
    )
    folds = _fold_data(frame, baseline_columns)
    candidates = []
    score_cache: dict[tuple[str, float], list[np.ndarray]] = {}
    for spec in SPECS:
        raw_scores = [
            fit_global_model(fold.train, candidate_columns, spec).predict(fold.test)
            for fold in folds
        ]
        for weight in WEIGHTS:
            evidence, scores = _candidate_evidence(
                folds, raw_scores, spec.name, weight
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

    selected_spec = next(spec for spec in SPECS if spec.name == selected.name)
    fresh_train = frame[frame["rcDate"] < FRESH_FROM]
    fresh = frame[frame["rcDate"] >= FRESH_FROM].copy()
    baseline_estimator, baseline_median = _fit(
        fresh_train, baseline_columns, "win", CONFIGS[0]
    )
    fresh_baseline = race_normalize(
        fresh,
        _predict(baseline_estimator, baseline_median, fresh, baseline_columns),
    )
    fresh_raw = fit_global_model(
        fresh_train, candidate_columns, selected_spec
    ).predict(fresh)
    fresh_selected = (
        (1.0 - selected.weight) * fresh_baseline
        + selected.weight * fresh_raw
    )
    fresh_baseline_metrics = metrics(fresh, fresh_baseline)
    fresh_candidate_metrics = metrics(fresh, fresh_selected)
    fresh_market_metrics = metrics(fresh, market_probability(fresh))
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
    market_diagnostics = []
    for fold, score in zip(folds, selected_scores):
        candidate_metrics = metrics(fold.test, score)
        market_metrics = metrics(fold.test, market_probability(fold.test))
        market_diagnostics.append({
            "fold": fold.name,
            "candidate_top1": candidate_metrics.top1,
            "market_top1": market_metrics.top1,
            "candidate_vs_market_top1_pp": (
                candidate_metrics.top1 - market_metrics.top1
            ) * 100.0,
        })
    return {
        "method": "global_winner_svm_neural_rf_conditional_logit_benchmark",
        "baseline": "kra_dual_phase_v4_history",
        "odds_in_candidate": False,
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "selection_folds": [fold.name for fold in folds[:3]],
        "confirmation_fold": folds[3].name,
        "fresh_holdout_is_pristine": False,
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
        "retrospective_market_diagnostics": market_diagnostics,
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
