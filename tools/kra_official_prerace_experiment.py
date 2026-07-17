"""Evaluate official KRA pre-race signals with locked chronological folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — pooled chronological fold contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_official_prerace_features import add_official_prerace_features, load_official_prerace_sources
from kra_promotion_policy import clears_absolute_top1_lift
from kra_training_features import build_features, load_rows
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict
from kra_model_evaluation import as_dict, metrics, paired_bootstrap_top1, race_normalize


DEFAULT_ARCHIVE: Final = Path("/Users/tttksj/kra/data/pre_race_api.sqlite")
DEFAULT_REPORT: Final = ROOT / "runs" / "kra_official_prerace_historical_results.json"
FOLDS: Final = (
    ("2024H2", "20240115", "20240701", "20250101"),
    ("2025H1", "20240115", "20250101", "20250701"),
    ("2025H2", "20240115", "20250701", "20260101"),
    ("2026H1", "20240115", "20260101", "20260701"),
)


def historical_promotion_allowed(forward_snapshot_pass: bool, top1_lift_pp: float) -> bool:
    """Require both the user's lift gate and real forward availability evidence."""
    return forward_snapshot_pass and clears_absolute_top1_lift(top1_lift_pp)


def _evaluate_config(frame, baseline_columns: list[str], official_columns: list[str], config) -> dict:  # noqa: DICT_OK — JSON experiment artifact
    rows = []
    pooled_frames = []
    pooled_baseline = []
    pooled_candidate = []
    for name, train_from, test_from, test_until in FOLDS:
        train = frame[(frame["rcDate"] >= train_from) & (frame["rcDate"] < test_from)]
        test = frame[(frame["rcDate"] >= test_from) & (frame["rcDate"] < test_until)]
        baseline_model, baseline_median = _fit(train, baseline_columns, "win", config)
        candidate_model, candidate_median = _fit(train, [*baseline_columns, *official_columns], "win", config)
        baseline = race_normalize(test, _predict(baseline_model, baseline_median, test, baseline_columns))
        candidate = race_normalize(test, _predict(candidate_model, candidate_median, test, [*baseline_columns, *official_columns]))
        baseline_metrics = metrics(test, baseline)
        candidate_metrics = metrics(test, candidate)
        rows.append({
            "fold": name,
            "races": int(test["rk"].nunique()),
            "baseline": as_dict(baseline_metrics),
            "candidate": as_dict(candidate_metrics),
            "top1_lift_pp": (candidate_metrics.top1 - baseline_metrics.top1) * 100.0,
            "top3_lift_pp": (candidate_metrics.top3 - baseline_metrics.top3) * 100.0,
            "logloss_delta": candidate_metrics.race_logloss - baseline_metrics.race_logloss,
        })
        pooled_frames.append(test)
        pooled_baseline.append(baseline)
        pooled_candidate.append(candidate)
    selection = rows[:3]
    return {
        "config": config[0],
        "folds": rows,
        "selection_mean_top1_lift_pp": float(np.mean([row["top1_lift_pp"] for row in selection])),
        "selection_min_top1_lift_pp": min(row["top1_lift_pp"] for row in selection),
        "holdout_top1_lift_pp": rows[-1]["top1_lift_pp"],
        "pooled_bootstrap": paired_bootstrap_top1(
            pd.concat(pooled_frames, ignore_index=True),
            np.concatenate(pooled_candidate),
            np.concatenate(pooled_baseline),
            samples=10_000,
        ),
    }


def run_experiment(db_path: Path, archive: Path) -> dict:  # noqa: DICT_OK — JSON report boundary
    base_frame, baseline_columns = build_features(load_rows(db_path))
    sources = load_official_prerace_sources(archive)
    frame, official_columns = add_official_prerace_features(base_frame, sources)
    results = [_evaluate_config(frame, baseline_columns, official_columns, config) for config in CONFIGS]
    selected = max(
        results,
        key=lambda row: (row["selection_min_top1_lift_pp"], row["selection_mean_top1_lift_pp"]),
    )
    bootstrap = selected["pooled_bootstrap"]
    lift = float(bootstrap["mean_pp"])
    return {
        "status": "historical_research_only",
        "data_availability": "historical_availability_unverified",
        "feature_count": len(official_columns),
        "source_rows": {
            "daily_training": int(len(sources.daily_training)),
            "starting_training": int(len(sources.starting_training)),
            "entry_weight": int(len(sources.entry_weight)),
        },
        "candidates": results,
        "selected": selected,
        "absolute_lift_pp": lift,
        "absolute_lift_gate": clears_absolute_top1_lift(lift),
        "forward_snapshot_gate": False,
        "promotion_pass": historical_promotion_allowed(False, lift),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_experiment(args.db, args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": report["selected"]["config"],
        "absolute_lift_pp": report["absolute_lift_pp"],
        "promotion_pass": report["promotion_pass"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
