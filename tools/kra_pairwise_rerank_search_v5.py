from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — scikit-learn walk-forward contract


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_model_candidates import fit_candidate  # noqa: E402
from kra_model_evaluation import metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_pairwise_ranker import PairwiseModel, build_winner_pairs, pairwise_scores  # noqa: E402
from kra_pairwise_reranker import RaceScores, RerankPolicy, restricted_rerank  # noqa: E402
from kra_promotion_policy import clears_absolute_top1_lift  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import DEFAULT_DB  # noqa: E402
from tools.kra_model_search_v5 import WEIGHTS, _fold_data  # noqa: E402
from tools.kra_pairwise_search_v5 import SPECS  # noqa: E402


DEFAULT_REPORT = ROOT / "runs" / "kra_pairwise_rerank_v5_results.json"
TOP_K_VALUES = (2, 3)


def _policy_result(folds, probabilities, policy: RerankPolicy) -> dict:  # noqa: DICT_OK — JSON experiment row
    fold_rows = []
    pooled_frames = []
    pooled_candidate = []
    pooled_baseline = []
    for fold, probability in zip(folds, probabilities):
        reranked = restricted_rerank(
            fold.test,
            RaceScores(fold.baseline_probability, probability),
            policy,
        )
        baseline_metrics = metrics(fold.test, fold.baseline_probability)
        candidate_metrics = metrics(fold.test, reranked.scores)
        fold_rows.append({
            "fold": fold.name,
            "top1_lift_pp": (candidate_metrics.top1 - baseline_metrics.top1) * 100.0,
            "top3_lift_pp": (candidate_metrics.top3 - baseline_metrics.top3) * 100.0,
            "logloss_delta": 0.0,
            "switches": reranked.switches,
            "switch_rate": reranked.switches / baseline_metrics.races,
        })
        pooled_frames.append(fold.test)
        pooled_candidate.append(reranked.scores)
        pooled_baseline.append(fold.baseline_probability)
    pooled_frame = pd.concat(pooled_frames, ignore_index=True)
    candidate_values = np.concatenate(pooled_candidate)
    baseline_values = np.concatenate(pooled_baseline)
    bootstrap = paired_bootstrap_top1(pooled_frame, candidate_values, baseline_values, 10000)
    selection = fold_rows[:3]
    promotion_pass = (
        clears_absolute_top1_lift(bootstrap["mean_pp"])
        and all(row["top1_lift_pp"] > 0 for row in fold_rows)
        and all(row["top3_lift_pp"] >= 0 for row in fold_rows)
        and bootstrap["ci95_low_pp"] > 0
    )
    return {
        "weight": policy.weight,
        "top_k": policy.top_k,
        "folds": fold_rows,
        "selection_mean_top1_lift_pp": float(np.mean([row["top1_lift_pp"] for row in selection])),
        "selection_min_top1_lift_pp": min(row["top1_lift_pp"] for row in selection),
        "selection_min_top3_lift_pp": min(row["top3_lift_pp"] for row in selection),
        "pooled_bootstrap": bootstrap,
        "pooled_logloss_delta": 0.0,
        "promotion_pass": promotion_pass,
    }


def run_search(db_path: Path) -> dict:  # noqa: DICT_OK — JSON experiment report
    frame, columns = build_features(load_rows(db_path))
    folds = _fold_data(frame, columns)
    results = {}
    for spec in SPECS:
        probabilities = []
        for fold in folds:
            pairs = build_winner_pairs(fold.train, columns)
            training = pairs.values.copy()
            training["target"] = pairs.targets
            estimator, median = fit_candidate(training, columns, "target", spec)
            model = PairwiseModel(estimator, median)
            probabilities.append(
                race_normalize(fold.test, pairwise_scores(model, fold.test, columns))
            )
        policies = [
            _policy_result(folds, probabilities, RerankPolicy(weight, top_k))
            for weight in WEIGHTS
            for top_k in TOP_K_VALUES
        ]
        results[spec.name] = max(
            policies,
            key=lambda result: (
                result["selection_min_top1_lift_pp"],
                result["selection_mean_top1_lift_pp"],
                result["selection_min_top3_lift_pp"],
            ),
        )
    selected, selected_result = max(
        results.items(),
        key=lambda item: (
            item[1]["selection_min_top1_lift_pp"],
            item[1]["selection_mean_top1_lift_pp"],
        ),
        default=(None, None),
    )
    return {
        "method": "restricted_pairwise_top_pick_reranker",
        "baseline": "kra_dual_phase_v4_history_hgb_d3",
        "probabilities": "unchanged_v4",
        "top_three_membership": "unchanged_v4",
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
