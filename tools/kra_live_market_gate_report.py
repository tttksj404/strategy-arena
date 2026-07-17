from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — chronological validation contract


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_model_evaluation import as_dict, market_probability, metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_promotion_policy import MIN_ABSOLUTE_TOP1_LIFT_PP  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402
from tools.kra_max_winrate_search import FOLDS, FRESH_FROM  # noqa: E402


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h12_live_results.json"


def _evaluate(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> dict:
    estimator, median = _fit(train, columns, "win", CONFIGS[0])
    baseline = race_normalize(test, _predict(estimator, median, test, columns))
    market = market_probability(test)
    baseline_metrics = metrics(test, baseline)
    market_metrics = metrics(test, market)
    return {
        "baseline": as_dict(baseline_metrics),
        "market": as_dict(market_metrics),
        "top1_lift_pp": (market_metrics.top1 - baseline_metrics.top1) * 100.0,
        "top3_lift_pp": (market_metrics.top3 - baseline_metrics.top3) * 100.0,
        "logloss_delta": market_metrics.race_logloss - baseline_metrics.race_logloss,
        "bootstrap": paired_bootstrap_top1(test, market, baseline, samples=10000),
    }


def run(db_path: Path) -> dict:  # noqa: DICT_OK — JSON research report
    frame, columns = build_features(load_rows(db_path))
    folds = []
    pooled_frames = []
    pooled_baseline = []
    pooled_market = []
    for name, train_from, test_from, test_until in FOLDS:
        train = frame[(frame["rcDate"] >= train_from) & (frame["rcDate"] < test_from)]
        test = frame[(frame["rcDate"] >= test_from) & (frame["rcDate"] < test_until)].copy()
        result = _evaluate(train, test, columns)
        folds.append({"name": name, **result})
        estimator, median = _fit(train, columns, "win", CONFIGS[0])
        pooled_frames.append(test)
        pooled_baseline.append(
            race_normalize(test, _predict(estimator, median, test, columns))
        )
        pooled_market.append(market_probability(test))
    pooled_frame = pd.concat(pooled_frames, ignore_index=True)
    pooled_baseline_values = np.concatenate(pooled_baseline)
    pooled_market_values = np.concatenate(pooled_market)
    fresh_train = frame[frame["rcDate"] < FRESH_FROM]
    fresh = frame[frame["rcDate"] >= FRESH_FROM].copy()
    fresh_result = _evaluate(fresh_train, fresh, columns)
    pooled_baseline_metrics = metrics(pooled_frame, pooled_baseline_values)
    pooled_market_metrics = metrics(pooled_frame, pooled_market_values)
    return {
        "method": "official_pre_start_complete_market_board_runtime_gate",
        "baseline": "kra_dual_phase_v4_history",
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "folds": folds,
        "pooled": {
            "baseline": as_dict(pooled_baseline_metrics),
            "market": as_dict(pooled_market_metrics),
            "top1_lift_pp": (pooled_market_metrics.top1 - pooled_baseline_metrics.top1) * 100.0,
            "top3_lift_pp": (pooled_market_metrics.top3 - pooled_baseline_metrics.top3) * 100.0,
            "logloss_delta": pooled_market_metrics.race_logloss - pooled_baseline_metrics.race_logloss,
            "bootstrap": paired_bootstrap_top1(
                pooled_frame, pooled_market_values, pooled_baseline_values, samples=10000
            ),
        },
        "observed_holdout": {"from": FRESH_FROM, **fresh_result},
        "runtime_safety_contract": {
            "official_start_time_required": True,
            "complete_positive_odds_board_required": True,
            "race_result_must_be_absent": True,
            "snapshot_must_precede_start": True,
        },
        "historical_odds_are_retrospective_closing_values": True,
        "fresh_timestamped_forward_validation_required": True,
        "promotion_pass": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run(args.db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
