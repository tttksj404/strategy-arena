from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — chronological race-level experiment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_market_residual import fit_favorite_challenger  # noqa: E402
from kra_model_evaluation import as_dict, market_probability, metrics, paired_bootstrap_top1  # noqa: E402
from kra_promotion_policy import MIN_ABSOLUTE_TOP1_LIFT_PP, clears_absolute_top1_lift  # noqa: E402
from tools.kra_diversified_search import (  # noqa: E402
    CONDITIONING_ARCHIVE,
    ENTRY_ARCHIVE,
    HEALTH_ARCHIVE,
    build_diversified_frame,
)
from tools.kra_dual_phase_experiment import DEFAULT_DB  # noqa: E402
from tools.kra_max_winrate_search import FRESH_FROM, _fold_data  # noqa: E402


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h14_results.json"
TOP_KS: Final = (2, 3)
BALANCE_POWERS: Final = (0.0, 0.25, 0.5)
MARGINS: Final = (0.0, 0.02, 0.05, 0.10, 0.15)
RUNNER_COLUMN_HINTS: Final = (
    "rating",
    "prior",
    "speed",
    "elo",
    "recent",
    "finish",
    "entry_",
    "condition_",
    "health_",
    "pace_",
    "consensus_",
    "percentile",
    "weight",
    "days_since",
)


@dataclass(frozen=True, slots=True)
class ChallengerEvidence:
    top_k: int
    class_balance_power: float
    margin: float
    top1_lifts_pp: tuple[float, ...]
    top3_lifts_pp: tuple[float, ...]
    logloss_deltas: tuple[float, ...]

    @property
    def key(self) -> tuple[int, float, float]:
        return self.top_k, self.class_balance_power, self.margin


def _runner_columns(columns: list[str]) -> list[str]:
    selected = [
        column
        for column in columns
        if any(hint in column for hint in RUNNER_COLUMN_HINTS)
        and not column.startswith(("sex_", "bd_", "weather_", "rank_group_"))
    ]
    return list(dict.fromkeys(selected))


def _select(candidates: list[ChallengerEvidence]) -> ChallengerEvidence:
    return max(
        candidates,
        key=lambda item: (
            min(item.top1_lifts_pp[:3]),
            float(np.mean(item.top1_lifts_pp[:3])),
            min(item.top3_lifts_pp[:3]),
            -float(np.mean(item.logloss_deltas[:3])),
            item.margin,
        ),
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
    runner_columns = _runner_columns(candidate_columns)
    folds = _fold_data(frame, baseline_columns)
    fold_predictions: list[dict[tuple[int, float, float], np.ndarray]] = []
    for fold in folds:
        predictions = {}
        for top_k in TOP_KS:
            for balance in BALANCE_POWERS:
                model = fit_favorite_challenger(
                    fold.train, runner_columns, top_k, balance
                )
                for margin, score in model.predict_many(fold.test, MARGINS).items():
                    predictions[(top_k, balance, margin)] = score
        fold_predictions.append(predictions)

    candidates = []
    for key in fold_predictions[0]:
        top1_lifts = []
        top3_lifts = []
        logloss_deltas = []
        for fold, predictions in zip(folds, fold_predictions):
            baseline = market_probability(fold.test)
            baseline_metrics = metrics(fold.test, baseline)
            candidate_metrics = metrics(fold.test, predictions[key])
            top1_lifts.append((candidate_metrics.top1 - baseline_metrics.top1) * 100.0)
            top3_lifts.append((candidate_metrics.top3 - baseline_metrics.top3) * 100.0)
            logloss_deltas.append(candidate_metrics.race_logloss - baseline_metrics.race_logloss)
        candidates.append(
            ChallengerEvidence(
                top_k=key[0],
                class_balance_power=key[1],
                margin=key[2],
                top1_lifts_pp=tuple(top1_lifts),
                top3_lifts_pp=tuple(top3_lifts),
                logloss_deltas=tuple(logloss_deltas),
            )
        )
    selected = _select(candidates)
    selected_scores = [prediction[selected.key] for prediction in fold_predictions]
    pooled_frame = pd.concat([fold.test for fold in folds], ignore_index=True)
    pooled_market = np.concatenate([market_probability(fold.test) for fold in folds])
    pooled_intrinsic = np.concatenate([fold.baseline_probability for fold in folds])
    pooled_candidate = np.concatenate(selected_scores)
    pooled_market_metrics = metrics(pooled_frame, pooled_market)
    pooled_candidate_metrics = metrics(pooled_frame, pooled_candidate)
    pooled_intrinsic_metrics = metrics(pooled_frame, pooled_intrinsic)
    pooled_bootstrap = paired_bootstrap_top1(
        pooled_frame, pooled_candidate, pooled_market, samples=10000
    )

    fresh_train = frame[frame["rcDate"] < FRESH_FROM]
    fresh = frame[frame["rcDate"] >= FRESH_FROM].copy()
    fresh_model = fit_favorite_challenger(
        fresh_train,
        runner_columns,
        selected.top_k,
        selected.class_balance_power,
    )
    fresh_candidate = fresh_model.predict(fresh, selected.margin)
    fresh_market = market_probability(fresh)
    fresh_bootstrap = paired_bootstrap_top1(
        fresh, fresh_candidate, fresh_market, samples=10000
    )
    retrospective_metric_gate = (
        clears_absolute_top1_lift(pooled_bootstrap["mean_pp"])
        and pooled_bootstrap["ci95_low_pp"] > 0
        and all(lift > 0 for lift in selected.top1_lifts_pp)
        and all(lift >= 0 for lift in selected.top3_lifts_pp)
        and all(delta <= 0 for delta in selected.logloss_deltas)
        and clears_absolute_top1_lift(fresh_bootstrap["mean_pp"])
        and fresh_bootstrap["ci95_low_pp"] > 0
    )
    return {
        "method": "race_level_multiclass_favorite_challenger_switch",
        "baseline": "pure_normalized_win_market",
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "historical_odds_are_retrospective_closing_values": True,
        "runner_feature_count": len(runner_columns),
        "candidate_count": len(candidates),
        "selection_folds": [fold.name for fold in folds[:3]],
        "confirmation_fold": folds[3].name,
        "selected": asdict(selected),
        "selected_result": {
            "pooled": {
                "intrinsic_v4": as_dict(pooled_intrinsic_metrics),
                "market": as_dict(pooled_market_metrics),
                "candidate": as_dict(pooled_candidate_metrics),
                "top1_lift_vs_market_pp": (pooled_candidate_metrics.top1 - pooled_market_metrics.top1) * 100.0,
                "top1_lift_vs_intrinsic_v4_pp": (pooled_candidate_metrics.top1 - pooled_intrinsic_metrics.top1) * 100.0,
                "bootstrap_vs_market": pooled_bootstrap,
            },
            "observed_holdout": {
                "from": FRESH_FROM,
                "market": as_dict(metrics(fresh, fresh_market)),
                "candidate": as_dict(metrics(fresh, fresh_candidate)),
                "bootstrap_vs_market": fresh_bootstrap,
                "is_pristine": False,
            },
            "retrospective_metric_gate": retrospective_metric_gate,
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "fresh_timestamped_forward_validation_required": True,
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
    report = run_search(args.db, args.entry_archive, args.conditioning_archive, args.health_archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
