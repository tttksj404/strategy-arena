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

from kra_card_features import CARD_FEATURES, add_card_features  # noqa: E402
from kra_conditioning_features import add_conditioning_features, load_conditioning  # noqa: E402
from kra_diversified_rankers import (  # noqa: E402
    add_pace_interactions,
    fit_conditional_logit,
    fit_market_distillation,
    fit_race_balanced,
    fit_segmented_experts,
)
from kra_dynamic_ratings import build_rating_features  # noqa: E402
from kra_entry_sheet_features import add_entry_sheet_features, load_entry_sheets  # noqa: E402
from kra_health_features import add_health_features, load_health  # noqa: E402
from kra_model_evaluation import as_dict, market_probability, metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_pedigree_features import (  # noqa: E402
    PEDIGREE_FEATURES,
    add_pedigree_priors,
    load_pedigree,
    merge_pedigree,
)
from kra_promotion_policy import MIN_ABSOLUTE_TOP1_LIFT_PP, clears_absolute_top1_lift  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402
from tools.kra_max_winrate_search import (  # noqa: E402
    FRESH_FROM,
    WEIGHTS,
    _candidate_evidence,
    _fold_data,
    select_candidate,
)


ENTRY_ARCHIVE: Final = Path("/Users/tttksj/kra/data/entry_sheet_archive")
CONDITIONING_ARCHIVE: Final = Path("/Users/tttksj/kra/data/conditioning_archive")
HEALTH_ARCHIVE: Final = Path("/Users/tttksj/kra/data/health_archive")
DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h9_results.json"
BASE_CANDIDATES: Final = (
    "race_balanced",
    "market_teacher",
    "listwise_conditional_logit",
    "segmented_win",
    "segmented_market_teacher",
)
ENSEMBLES: Final = {
    "balanced_market_teacher": ("race_balanced", "market_teacher"),
    "listwise_market_teacher": ("listwise_conditional_logit", "market_teacher"),
    "segmented_consensus": ("segmented_win", "segmented_market_teacher"),
    "all_architectures": BASE_CANDIDATES,
}


def build_diversified_frame(
    db_path: Path,
    entry_archive: Path,
    conditioning_archive: Path,
    health_archive: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    source = merge_pedigree(load_rows(db_path), load_pedigree(db_path))
    baseline_frame, baseline_columns = build_features(source)
    rated, rated_columns = build_rating_features(baseline_frame, baseline_columns)
    result = add_pedigree_priors(rated)
    pedigree_columns = [
        *PEDIGREE_FEATURES,
        *(f"{column}_rel" for column in PEDIGREE_FEATURES),
    ]
    for column in PEDIGREE_FEATURES:
        result[f"{column}_rel"] = result[column] - result.groupby("rk")[column].transform("mean")

    result = add_card_features(result)
    card_columns = list(CARD_FEATURES)
    for column in ("jk_weight_allowance", "owner_win_prior"):
        relative = f"{column}_rel"
        result[relative] = result[column] - result.groupby("rk")[column].transform("mean")
        card_columns.append(relative)

    result, entry_columns = add_entry_sheet_features(
        result, load_entry_sheets(entry_archive)
    )
    result, conditioning_columns = add_conditioning_features(
        result, load_conditioning(conditioning_archive)
    )
    result, health_columns = add_health_features(result, load_health(health_archive))
    result, pace_columns = add_pace_interactions(result)
    candidate_columns = list(dict.fromkeys([
        *rated_columns,
        *pedigree_columns,
        *card_columns,
        *entry_columns,
        *conditioning_columns,
        *health_columns,
        *pace_columns,
    ]))
    return result, baseline_columns, candidate_columns


def _fit_base_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> dict[str, np.ndarray]:
    return {
        "race_balanced": fit_race_balanced(train, columns).predict(test),
        "market_teacher": fit_market_distillation(train, columns).predict(test),
        "listwise_conditional_logit": fit_conditional_logit(train, columns).predict(test),
        "segmented_win": fit_segmented_experts(train, columns, "win").predict(test),
        "segmented_market_teacher": fit_segmented_experts(
            train, columns, "market"
        ).predict(test),
    }


def _candidate_predictions(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(base)
    for name, members in ENSEMBLES.items():
        result[name] = np.mean([base[member] for member in members], axis=0)
    return result


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
    fresh_selected = (
        (1.0 - selected.weight) * fresh_baseline
        + selected.weight * fresh_predictions[selected.name]
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
    live_market = market_probability(fresh)
    return {
        "method": "diversified_listwise_segmented_market_distillation_ensemble",
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
                "bootstrap": fresh_bootstrap,
            },
            "metric_gate_without_pristine_requirement": metric_gate,
            "promotion_pass": False,
        },
        "diagnostic_live_market_ceiling": {
            "uses_current_race_closing_odds": True,
            "metrics": as_dict(metrics(fresh, live_market)),
            "bootstrap_vs_intrinsic_v4": paired_bootstrap_top1(
                fresh, live_market, fresh_baseline, samples=10000
            ),
            "eligible_for_intrinsic_promotion": False,
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
