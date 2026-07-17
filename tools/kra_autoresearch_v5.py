from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — shared KRA research-frame contract


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_candidate_features import CANDIDATE_GROUPS, add_candidate_features  # noqa: E402
from kra_model_evaluation import as_dict, metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_promotion_policy import clears_absolute_top1_lift  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402


DEFAULT_REPORT = ROOT / "runs" / "kra_autoresearch_v5_results.json"
FOLDS = (
    ("2024H2", "20240101", "20240701", "20250101"),
    ("2025H1", "20240101", "20250101", "20250701"),
    ("2025H2", "20240101", "20250701", "20260101"),
    ("2026H1", "20240101", "20260101", "20260701"),
)
CANDIDATES = {
    **{name: (name,) for name in CANDIDATE_GROUPS},
    "recent_distance": ("recent_form", "distance_form"),
    "recent_meet": ("recent_form", "meet_form"),
    "recent_state": ("recent_form", "state_change"),
    "recent_distance_meet": ("recent_form", "distance_form", "meet_form"),
    "recent_distance_jockey": ("recent_form", "distance_form", "jockey_pair"),
    "all_groups": tuple(CANDIDATE_GROUPS),
}


class FoldResult(TypedDict):
    fold: str
    baseline: dict[str, float | int]
    candidate: dict[str, float | int]
    top1_lift_pp: float
    top3_lift_pp: float
    logloss_delta: float


class CandidateResult(TypedDict):
    groups: list[str]
    folds: list[FoldResult]
    selection_mean_top1_lift_pp: float
    selection_min_top1_lift_pp: float
    selection_min_top3_lift_pp: float
    pooled_bootstrap: dict[str, float]
    pooled_logloss_delta: float
    promotion_pass: bool


class ResearchReport(TypedDict):
    method: str
    baseline: str
    candidates: dict[str, CandidateResult]
    selected: str | None
    selected_result: CandidateResult | None
    promotion_pass: bool


def _candidate_columns(
    baseline_columns: list[str],
    groups: tuple[str, ...],
) -> list[str]:
    columns = list(baseline_columns)
    for group in groups:
        for column in CANDIDATE_GROUPS[group]:
            columns.extend((column, f"{column}_rel"))
    return list(dict.fromkeys(columns))


def _prepare_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for columns in CANDIDATE_GROUPS.values():
        for column in columns:
            if column in result.columns:
                result[f"{column}_rel"] = (
                    result[column] - result.groupby("rk")[column].transform("mean")
                )
    return result


def run_research(db_path: Path) -> ResearchReport:
    baseline_frame, baseline_columns = build_features(load_rows(db_path))
    frame = _prepare_candidate_frame(add_candidate_features(baseline_frame))
    config = CONFIGS[0]
    fold_cache = []
    for name, train_from, test_from, test_until in FOLDS:
        train = frame[(frame["rcDate"] >= train_from) & (frame["rcDate"] < test_from)]
        test = frame[(frame["rcDate"] >= test_from) & (frame["rcDate"] < test_until)].copy()
        baseline_model, baseline_median = _fit(train, baseline_columns, "win", config)
        baseline_probability = race_normalize(
            test, _predict(baseline_model, baseline_median, test, baseline_columns)
        )
        fold_cache.append((name, train, test, baseline_probability))

    results: dict[str, CandidateResult] = {}
    for candidate_name, groups in CANDIDATES.items():
        columns = _candidate_columns(baseline_columns, groups)
        folds: list[FoldResult] = []
        pooled_frames = []
        pooled_candidate = []
        pooled_baseline = []
        for fold_name, train, test, baseline_probability in fold_cache:
            model, median = _fit(train, columns, "win", config)
            probability = race_normalize(test, _predict(model, median, test, columns))
            baseline_metrics = metrics(test, baseline_probability)
            candidate_metrics = metrics(test, probability)
            folds.append({
                "fold": fold_name,
                "baseline": as_dict(baseline_metrics),
                "candidate": as_dict(candidate_metrics),
                "top1_lift_pp": (candidate_metrics.top1 - baseline_metrics.top1) * 100.0,
                "top3_lift_pp": (candidate_metrics.top3 - baseline_metrics.top3) * 100.0,
                "logloss_delta": candidate_metrics.race_logloss - baseline_metrics.race_logloss,
            })
            pooled_frames.append(test)
            pooled_candidate.append(probability)
            pooled_baseline.append(baseline_probability)
        selection = folds[:3]
        pooled_frame = pd.concat(pooled_frames, ignore_index=True)
        candidate_values = np.concatenate(pooled_candidate)
        baseline_values = np.concatenate(pooled_baseline)
        bootstrap = paired_bootstrap_top1(
            pooled_frame, candidate_values, baseline_values, samples=10000
        )
        pooled_candidate_metrics = metrics(pooled_frame, candidate_values)
        pooled_baseline_metrics = metrics(pooled_frame, baseline_values)
        logloss_delta = pooled_candidate_metrics.race_logloss - pooled_baseline_metrics.race_logloss
        promotion_pass = (
            clears_absolute_top1_lift(bootstrap["mean_pp"])
            and all(fold["top1_lift_pp"] > 0 for fold in folds)
            and all(fold["top3_lift_pp"] >= 0 for fold in folds)
            and bootstrap["ci95_low_pp"] > 0
            and logloss_delta <= 0
        )
        results[candidate_name] = {
            "groups": list(groups),
            "folds": folds,
            "selection_mean_top1_lift_pp": float(np.mean([fold["top1_lift_pp"] for fold in selection])),
            "selection_min_top1_lift_pp": min(fold["top1_lift_pp"] for fold in selection),
            "selection_min_top3_lift_pp": min(fold["top3_lift_pp"] for fold in selection),
            "pooled_bootstrap": bootstrap,
            "pooled_logloss_delta": logloss_delta,
            "promotion_pass": promotion_pass,
        }

    eligible = [
        (name, result) for name, result in results.items()
        if result["selection_min_top1_lift_pp"] > 0
        and result["selection_min_top3_lift_pp"] >= 0
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item[1]["selection_mean_top1_lift_pp"],
            item[1]["selection_min_top1_lift_pp"],
            -item[1]["pooled_logloss_delta"],
        ),
        default=(None, None),
    )[0]
    selected_result = results.get(selected) if selected is not None else None
    return {
        "method": "candidate_selection_first_three_folds_confirmation_fourth_fold",
        "baseline": "kra_dual_phase_v4_history",
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
    report = run_research(args.db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
