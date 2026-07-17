#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — shared KRA research-frame contract

from kra_dual_phase_experiment import (
    CONFIGS,
    DEFAULT_DB,
    HORSE_HISTORY_FEATURES,
    _as_dict,
    _fit,
    _predict,
    _race_normalize,
    build_features,
    load_rows,
    metrics,
    paired_bootstrap_top1,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "runs" / "kra_reverse_candidate_audit.json"
FOLDS = (
    ("2024H2", "20240101", "20240701", "20250101"),
    ("2025H1", "20240101", "20250101", "20250701"),
    ("2025H2", "20240101", "20250701", "20260101"),
    ("2026H1", "20240101", "20260101", "20260701"),
)


def run_audit(db_path: Path) -> dict:  # noqa: DICT_OK — JSON audit report
    frame, candidate_columns = build_features(load_rows(db_path))
    history_columns = set(HORSE_HISTORY_FEATURES)
    baseline_columns = [column for column in candidate_columns if column not in history_columns]
    config = CONFIGS[0]
    fold_rows = []
    pooled_frames = []
    pooled_candidate = []
    pooled_baseline = []
    pooled_place = []

    for name, train_from, test_from, test_until in FOLDS:
        train = frame[(frame["rcDate"] >= train_from) & (frame["rcDate"] < test_from)]
        test = frame[(frame["rcDate"] >= test_from) & (frame["rcDate"] < test_until)].copy()
        base_model, base_median = _fit(train, baseline_columns, "win", config)
        candidate_model, candidate_median = _fit(train, candidate_columns, "win", config)
        place_model, place_median = _fit(train, candidate_columns, "place", config)
        base_probability = _race_normalize(test, _predict(base_model, base_median, test, baseline_columns))
        candidate_probability = _race_normalize(
            test, _predict(candidate_model, candidate_median, test, candidate_columns)
        )
        place_probability = _predict(place_model, place_median, test, candidate_columns)
        base_metrics = metrics(test, base_probability)
        candidate_metrics = metrics(test, candidate_probability)
        place_metrics = metrics(test, place_probability)
        fold_rows.append({
            "fold": name,
            "baseline": _as_dict(base_metrics),
            "horse_history": _as_dict(candidate_metrics),
            "horse_history_top1_lift_pp": (candidate_metrics.top1 - base_metrics.top1) * 100.0,
            "horse_history_top3_lift_pp": (candidate_metrics.top3 - base_metrics.top3) * 100.0,
            "reverse_place_leader": _as_dict(place_metrics),
            "reverse_place_top1_lift_pp": (place_metrics.top1 - candidate_metrics.top1) * 100.0,
        })
        pooled_frames.append(test)
        pooled_candidate.append(candidate_probability)
        pooled_baseline.append(base_probability)
        pooled_place.append(place_probability)

    pooled_frame = pd.concat(pooled_frames, ignore_index=True)
    candidate_values = np.concatenate(pooled_candidate)
    baseline_values = np.concatenate(pooled_baseline)
    place_values = np.concatenate(pooled_place)
    candidate_bootstrap = paired_bootstrap_top1(
        pooled_frame, candidate_values, baseline_values, samples=10000
    )
    place_bootstrap = paired_bootstrap_top1(
        pooled_frame, place_values, candidate_values, samples=10000
    )
    return {
        "method": "strict_date_walk_forward_same_day_excluded",
        "config": config[0],
        "folds": fold_rows,
        "pooled": {
            "horse_history_vs_baseline": candidate_bootstrap,
            "reverse_place_vs_win": place_bootstrap,
        },
        "promotion_gate": {
            "horse_history_all_fold_top1_positive": all(
                row["horse_history_top1_lift_pp"] > 0 for row in fold_rows
            ),
            "horse_history_all_fold_top3_nonnegative": all(
                row["horse_history_top3_lift_pp"] >= 0 for row in fold_rows
            ),
            "horse_history_pooled_ci_low_positive": candidate_bootstrap["ci95_low_pp"] > 0,
            "horse_history_promoted": all(
                row["horse_history_top1_lift_pp"] > 0
                and row["horse_history_top3_lift_pp"] >= 0
                for row in fold_rows
            ) and candidate_bootstrap["ci95_low_pp"] > 0,
            "reverse_place_promoted": place_bootstrap["ci95_low_pp"] > 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_audit(args.db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
