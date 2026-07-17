#!/usr/bin/env python3
"""Evaluate and train a leakage-safe KRA pre-race/live-odds model.

Usage:
    python tools/kra_dual_phase_experiment.py
    python tools/kra_dual_phase_experiment.py --save-model static/models/kra_model.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd  # noqa: PANDAS_OK — scikit-learn experiment contract
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_history_features import (  # noqa: E402
    HORSE_HISTORY_FEATURES,
    build_horse_history_snapshot,
)
from kra_model_evaluation import (  # noqa: E402
    as_dict as _as_dict,
    leader_threshold,
    market_probability,
    metrics,
    paired_bootstrap_top1,
    race_normalize as _race_normalize,
    selective_metrics,
)
from kra_pairwise_artifact import build_pairwise_artifact  # noqa: E402
from kra_training_features import NUM, REL, build_features, load_rows  # noqa: E402
DEFAULT_DB = Path("/Users/tttksj/kra/data/kra.db")
DEFAULT_BASELINE_MODEL = Path("/Users/tttksj/kra/models/kra_model.joblib")
DEFAULT_REPORT = ROOT / "runs" / "kra_dual_phase_results.json"
VALID_FROM = "20250101"
HOLDOUT_FROM = "20260101"
CONFIGS = (
    ("hgb_d3", 3, 0.06, 300, 120, 1.0),
    ("hgb_d4", 4, 0.05, 350, 180, 2.0),
    ("hgb_d5", 5, 0.04, 400, 220, 4.0),
)


def _fit(frame: pd.DataFrame, columns: list[str], target: str, config: tuple[str, int, float, int, int, float]):
    _, depth, rate, iterations, min_leaf, l2 = config
    median = frame[columns].median(numeric_only=True)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(median)
    model = HistGradientBoostingClassifier(
        max_depth=depth,
        learning_rate=rate,
        max_iter=iterations,
        min_samples_leaf=min_leaf,
        l2_regularization=l2,
        random_state=42,
    ).fit(values, frame[target].to_numpy())
    return model, median


def _predict(model, median: pd.Series, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(median)
    return model.predict_proba(values)[:, 1]


def baseline_zero_odds_metrics(frame: pd.DataFrame, model_path: Path) -> dict[str, dict[str, float | int]] | None:
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    zero_odds = frame.copy()
    for column in (
        "winOdds", "plcOdds", "imp_win", "imp_plc", "imp_win_norm",
        "imp_plc_norm", "imp_win_rel", "imp_plc_rel",
    ):
        zero_odds[column] = 0.0
    values = zero_odds.reindex(columns=model["cols"], fill_value=0)
    for column in model["cols"]:
        median = model["med"].get(column)
        if median is not None:
            values[column] = values[column].fillna(median)
    values = values.apply(pd.to_numeric, errors="coerce").fillna(0)
    win_probability = model["win"].predict_proba(values)[:, 1]
    place_probability = model["plc"].predict_proba(values)[:, 1]
    return {
        "win_pick": _as_dict(metrics(frame, win_probability)),
        "displayed_place_leader": _as_dict(metrics(frame, place_probability)),
    }


def run_experiment(db_path: Path, baseline_model_path: Path = DEFAULT_BASELINE_MODEL) -> tuple[dict, tuple]:
    frame, columns = build_features(load_rows(db_path))
    development = frame[frame["rcDate"] < VALID_FROM]
    validation = frame[(frame["rcDate"] >= VALID_FROM) & (frame["rcDate"] < HOLDOUT_FROM)]
    train = frame[frame["rcDate"] < HOLDOUT_FROM]
    holdout = frame[frame["rcDate"] >= HOLDOUT_FROM]

    validation_rows = []
    for config in CONFIGS:
        model, median = _fit(development, columns, "win", config)
        probability = _race_normalize(validation, _predict(model, median, validation, columns))
        validation_rows.append((config, metrics(validation, probability)))
    winner_config, winner_validation = max(
        validation_rows,
        key=lambda item: (item[1].top1, item[1].top3, -item[1].race_logloss),
    )

    win_model, win_median = _fit(train, columns, "win", winner_config)
    place_model, place_median = _fit(train, columns, "place", winner_config)
    intrinsic_holdout = _race_normalize(holdout, _predict(win_model, win_median, holdout, columns))
    place_holdout = _predict(place_model, place_median, holdout, columns)
    market_holdout = market_probability(holdout)

    intrinsic_validation_model, intrinsic_validation_median = _fit(development, columns, "win", winner_config)
    intrinsic_validation = _race_normalize(
        validation,
        _predict(intrinsic_validation_model, intrinsic_validation_median, validation, columns),
    )
    market_validation = market_probability(validation)
    blend_candidates = []
    for market_weight in np.linspace(0.0, 1.0, 21):
        blend = (1.0 - market_weight) * intrinsic_validation + market_weight * market_validation
        blend_candidates.append((float(market_weight), metrics(validation, blend)))
    market_weight, blend_validation = max(
        blend_candidates,
        key=lambda item: (item[1].top1, item[1].top3, -item[1].race_logloss),
    )
    blend_holdout = (1.0 - market_weight) * intrinsic_holdout + market_weight * market_holdout

    pre_threshold = leader_threshold(validation, intrinsic_validation)
    live_threshold = leader_threshold(validation, market_validation)

    place_as_top = metrics(holdout, place_holdout)
    deployed_baseline = baseline_zero_odds_metrics(holdout, baseline_model_path)
    intrinsic_metrics = metrics(holdout, intrinsic_holdout)
    market_metrics = metrics(holdout, market_holdout)
    live_metrics = metrics(holdout, blend_holdout)
    selective_pre = selective_metrics(holdout, intrinsic_holdout, pre_threshold)
    selective_live = selective_metrics(holdout, blend_holdout, live_threshold)
    baseline_win_top1 = (
        float(deployed_baseline["win_pick"]["top1"])
        if deployed_baseline is not None
        else float("nan")
    )
    report = {
        "split": {"development_before": VALID_FROM, "validation": f"{VALID_FROM}..{HOLDOUT_FROM}", "locked_holdout_from": HOLDOUT_FROM},
        "selected_config": winner_config[0],
        "selected_live_market_weight": market_weight,
        "validation": {
            "selected_intrinsic": _as_dict(winner_validation),
            "selected_live_blend": _as_dict(blend_validation),
        },
        "holdout": {
            "deployed_odds_model_without_odds": deployed_baseline,
            "intrinsic_place_leader": _as_dict(place_as_top),
            "intrinsic_win_leader": _as_dict(intrinsic_metrics),
            "market": _as_dict(market_metrics),
            "live_blend": _as_dict(live_metrics),
            "intrinsic_win_vs_intrinsic_place_bootstrap": paired_bootstrap_top1(holdout, intrinsic_holdout, place_holdout),
            "live_blend_vs_market_bootstrap": paired_bootstrap_top1(holdout, blend_holdout, market_holdout),
            "selective_pre_race": selective_pre,
            "selective_live": selective_live,
        },
        "promotion_gate": {
            "pre_race_pass": intrinsic_metrics.top1 > baseline_win_top1,
            "pre_race_gain_pp": (intrinsic_metrics.top1 - baseline_win_top1) * 100.0,
            "live_pass": live_metrics.top1 >= market_metrics.top1,
        },
    }
    artifact_parts = (
        frame, columns, winner_config, market_weight, pre_threshold, live_threshold,
        selective_pre, selective_live,
    )
    return report, artifact_parts


def save_artifact(path: Path, artifact_parts: tuple) -> None:
    (
        frame, columns, config, market_weight, pre_threshold, live_threshold,
        selective_pre, selective_live,
    ) = artifact_parts
    win_model, win_median = _fit(frame, columns, "win", config)
    place_model, _ = _fit(frame, columns, "place", config)
    global_win_rate = float(frame["win"].mean())
    def posterior_map(id_column: str) -> dict:  # noqa: DICT_OK — serialized model lookup payload
        grouped = frame.groupby(id_column, dropna=False)["win"].agg(["sum", "count"])
        return ((grouped["sum"] + 0.1) / (grouped["count"] + 1.0)).dropna().to_dict()

    jockey_prior = posterior_map("jkNo")
    trainer_prior = posterior_map("trNo")
    artifact = {
        "win": win_model,
        "plc": place_model,
        "cols": columns,
        "med": {key: float(value) for key, value in win_median.items() if pd.notna(value)},
        "num": list(NUM),
        "rel": list(REL),
        "feats": columns,
        "kind": "kra_dual_phase_v4_history_fresh_holdout_guard",
        "global_win_rate": global_win_rate,
        "jk_prior": {str(key): float(value) for key, value in jockey_prior.items()},
        "tr_prior": {str(key): float(value) for key, value in trainer_prior.items()},
        "horse_history": build_horse_history_snapshot(frame),
        "pairwise": build_pairwise_artifact(frame, columns),
        "live_market_weight": market_weight,
        "confidence_policy": {
            "pre_race_top_probability_min": pre_threshold,
            "live_top_probability_min": live_threshold,
            "pre_race_expected_top1": selective_pre["top1"],
            "pre_race_coverage": selective_pre["coverage"],
            "live_expected_top1": selective_live["top1"],
            "live_coverage": selective_live["coverage"],
        },
        "meta": {
            "trained_rows": int(len(frame)),
            "trained_races": int(frame["rk"].nunique()),
            "date_min": str(frame["rcDate"].min()),
            "date_max": str(frame["rcDate"].max()),
            "meets": sorted(frame["meet"].dropna().unique().tolist()),
            "odds_injected": False,
            "phase_policy": "intrinsic_pre_race_market_blend_live",
            "participant_learning": False,
            "selected_config": config[0],
            "history_features": list(HORSE_HISTORY_FEATURES),
            "pairwise_policy": "candidate_disabled_after_20260622_fresh_holdout",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--save-model", type=Path)
    args = parser.parse_args()
    report, artifact_parts = run_experiment(args.db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.save_model:
        save_artifact(args.save_model, artifact_parts)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
