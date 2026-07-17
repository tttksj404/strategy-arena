from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_model_candidates import ModelSpec, fit_candidate  # noqa: E402
from kra_model_evaluation import race_normalize  # noqa: E402
from kra_pairwise_ranker import PairwiseModel, build_winner_pairs, pairwise_scores  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import DEFAULT_DB  # noqa: E402
from tools.kra_model_search_v5 import WEIGHTS, _fold_data, _weight_result  # noqa: E402


DEFAULT_REPORT = ROOT / "runs" / "kra_pairwise_search_v5_results.json"
SPECS = (
    ModelSpec("pair_logistic_c10", "logistic", None, 600, 2, 1.0, l2=0.1),
    ModelSpec("pair_logistic_c1", "logistic", None, 600, 2, 1.0, l2=1.0),
    ModelSpec("pair_logistic_c01", "logistic", None, 600, 2, 1.0, l2=10.0),
    ModelSpec("pair_hgb_d2", "hgb", 2, 300, 160, 1.0, 0.05, 2.0),
    ModelSpec("pair_hgb_d3", "hgb", 3, 300, 160, 1.0, 0.05, 2.0),
    ModelSpec("pair_hgb_d4", "hgb", 4, 300, 200, 1.0, 0.04, 3.0),
)


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
        "method": "winner_pairwise_ranker_first_three_folds_confirmation_fourth",
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
