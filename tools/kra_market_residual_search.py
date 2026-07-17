from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — chronological grouped-ranking experiment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_diversified_rankers import fit_conditional_logit, fit_race_balanced  # noqa: E402
from kra_market_residual import (  # noqa: E402
    add_market_features,
    fit_market_residual,
    restricted_market_rerank,
    uncertainty_gate,
)
from kra_model_candidates import ModelSpec, fit_candidate as fit_generic_candidate  # noqa: E402
from kra_model_evaluation import as_dict, market_probability, metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_promotion_policy import MIN_ABSOLUTE_TOP1_LIFT_PP, clears_absolute_top1_lift  # noqa: E402
from tools.kra_diversified_search import (  # noqa: E402
    CONDITIONING_ARCHIVE,
    ENTRY_ARCHIVE,
    HEALTH_ARCHIVE,
    build_diversified_frame,
)
from tools.kra_dual_phase_experiment import DEFAULT_DB  # noqa: E402
from tools.kra_max_winrate_search import FRESH_FROM, _fold_data  # noqa: E402


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_max_winrate_h13_results.json"
BLEND_WEIGHTS: Final = (0.25, 0.5, 0.75, 1.0)
FAVORITE_THRESHOLDS: Final = (0.30, 0.35, 0.40)
GAP_THRESHOLDS: Final = (0.03, 0.06, 0.10)


@dataclass(frozen=True, slots=True)
class Policy:
    model: str
    mode: str
    value: float
    secondary: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.model}|{self.mode}|{self.value:.2f}|{self.secondary:.2f}"


@dataclass(frozen=True, slots=True)
class Evidence:
    policy: str
    top1_lifts_pp: tuple[float, ...]
    top3_lifts_pp: tuple[float, ...]
    logloss_deltas: tuple[float, ...]


def _base_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> dict[str, np.ndarray]:
    winner = fit_race_balanced(train, columns).predict(test)
    residual_model = fit_market_residual(train, columns)
    residual_soft = residual_model.predict(test, scale=0.5)
    residual = residual_model.predict(test, scale=1.0)
    residual_strong = residual_model.predict(test, scale=2.0)
    compact_columns = [
        column
        for column in columns
        if column.startswith("market_")
        or column
        in {
            "rating_percentile",
            "elo_percentile",
            "speed_percentile",
            "recent_finish_percentile",
            "consensus_strength",
            "consensus_disagreement",
            "field_size",
            "rcDist",
            "rating",
            "hr_elo",
            "hr_speed_mean_3",
            "hr_recent_finish_mean_3",
        }
    ]
    conditional = fit_conditional_logit(train, compact_columns, ridge=8.0).predict(test)
    generic_specs = (
        ModelSpec("market_extra_trees", "extra_trees", 12, 450, 18, 0.75),
        ModelSpec("market_random_forest", "random_forest", 12, 450, 18, 0.75),
        ModelSpec("market_logistic", "logistic", None, 1200, 1, 1.0, l2=8.0),
    )
    generic_predictions = {}
    for spec in generic_specs:
        estimator, median = fit_generic_candidate(train, columns, "win", spec)
        values = test[columns].apply(pd.to_numeric, errors="coerce").fillna(median).fillna(0.0)
        generic_predictions[spec.name] = race_normalize(
            test, estimator.predict_proba(values)[:, 1]
        )
    return {
        "market_hgb": winner,
        "residual_soft": residual_soft,
        "residual": residual,
        "residual_strong": residual_strong,
        "conditional_compact": conditional,
        "hgb_residual_consensus": race_normalize(test, 0.5 * winner + 0.5 * residual),
        "hgb_conditional_consensus": race_normalize(test, 0.5 * winner + 0.5 * conditional),
        **generic_predictions,
    }


def _policies(model_names: list[str]) -> list[Policy]:
    result = []
    for model in model_names:
        result.extend(Policy(model, "blend", weight) for weight in BLEND_WEIGHTS)
        result.extend((Policy(model, "topk", 2), Policy(model, "topk", 3)))
        for favorite in FAVORITE_THRESHOLDS:
            for gap in GAP_THRESHOLDS:
                result.append(Policy(model, "uncertainty", favorite, gap))
                result.append(Policy(model, "top2_uncertainty", favorite, gap))
    return result


def _apply_policy(frame: pd.DataFrame, raw: np.ndarray, policy: Policy) -> np.ndarray:
    market = market_probability(frame)
    if policy.mode == "blend":
        return race_normalize(frame, (1.0 - policy.value) * market + policy.value * raw)
    if policy.mode == "topk":
        return restricted_market_rerank(frame, raw, int(policy.value))
    if policy.mode == "uncertainty":
        return uncertainty_gate(frame, raw, policy.value, policy.secondary)
    if policy.mode == "top2_uncertainty":
        reranked = restricted_market_rerank(frame, raw, 2)
        return uncertainty_gate(frame, reranked, policy.value, policy.secondary)
    raise ValueError(f"unknown policy mode: {policy.mode}")


def _evaluate_policy(folds, raw_by_fold, policy: Policy) -> tuple[Evidence, list[np.ndarray]]:
    top1_lifts = []
    top3_lifts = []
    logloss_deltas = []
    scores = []
    for fold, raw_predictions in zip(folds, raw_by_fold):
        baseline = market_probability(fold.test)
        score = _apply_policy(fold.test, raw_predictions[policy.model], policy)
        baseline_metrics = metrics(fold.test, baseline)
        candidate_metrics = metrics(fold.test, score)
        scores.append(score)
        top1_lifts.append((candidate_metrics.top1 - baseline_metrics.top1) * 100.0)
        top3_lifts.append((candidate_metrics.top3 - baseline_metrics.top3) * 100.0)
        logloss_deltas.append(candidate_metrics.race_logloss - baseline_metrics.race_logloss)
    return Evidence(
        policy=policy.name,
        top1_lifts_pp=tuple(top1_lifts),
        top3_lifts_pp=tuple(top3_lifts),
        logloss_deltas=tuple(logloss_deltas),
    ), scores


def _select(evidence: list[Evidence]) -> Evidence:
    return max(
        evidence,
        key=lambda item: (
            min(item.top1_lifts_pp[:3]),
            float(np.mean(item.top1_lifts_pp[:3])),
            min(item.top3_lifts_pp[:3]),
            -float(np.mean(item.logloss_deltas[:3])),
        ),
    )


def run_search(
    db_path: Path,
    entry_archive: Path,
    conditioning_archive: Path,
    health_archive: Path,
) -> dict:  # noqa: DICT_OK — JSON research report
    intrinsic_frame, baseline_columns, intrinsic_columns = build_diversified_frame(
        db_path, entry_archive, conditioning_archive, health_archive
    )
    frame, market_columns = add_market_features(intrinsic_frame)
    columns = list(dict.fromkeys([*intrinsic_columns, *market_columns]))
    folds = _fold_data(frame, baseline_columns)
    raw_by_fold = [_base_predictions(fold.train, fold.test, columns) for fold in folds]
    policies = _policies(list(raw_by_fold[0]))
    evidence = []
    score_cache = {}
    for policy in policies:
        result, scores = _evaluate_policy(folds, raw_by_fold, policy)
        evidence.append(result)
        score_cache[policy.name] = scores

    selected = _select(evidence)
    selected_policy = next(policy for policy in policies if policy.name == selected.policy)
    selected_scores = score_cache[selected.policy]
    pooled_frame = pd.concat([fold.test for fold in folds], ignore_index=True)
    pooled_market = np.concatenate([market_probability(fold.test) for fold in folds])
    pooled_intrinsic = np.concatenate([fold.baseline_probability for fold in folds])
    pooled_selected = np.concatenate(selected_scores)
    pooled_metrics = metrics(pooled_frame, pooled_selected)
    pooled_market_metrics = metrics(pooled_frame, pooled_market)
    pooled_intrinsic_metrics = metrics(pooled_frame, pooled_intrinsic)
    pooled_bootstrap = paired_bootstrap_top1(
        pooled_frame, pooled_selected, pooled_market, samples=10000
    )

    fresh_train = frame[frame["rcDate"] < FRESH_FROM]
    fresh = frame[frame["rcDate"] >= FRESH_FROM].copy()
    fresh_raw = _base_predictions(fresh_train, fresh, columns)[selected_policy.model]
    fresh_market = market_probability(fresh)
    fresh_selected = _apply_policy(fresh, fresh_raw, selected_policy)
    fresh_bootstrap = paired_bootstrap_top1(
        fresh, fresh_selected, fresh_market, samples=10000
    )
    confirmation_positive = selected.top1_lifts_pp[3] > 0
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
        "method": "market_residual_hgb_conditional_restricted_rerank_uncertainty_gate",
        "baseline": "pure_normalized_win_market",
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "historical_odds_are_retrospective_closing_values": True,
        "feature_count": len(columns),
        "market_feature_count": len(market_columns),
        "candidate_policy_count": len(policies),
        "selection_folds": [fold.name for fold in folds[:3]],
        "confirmation_fold": folds[3].name,
        "selected": selected.policy,
        "selected_result": {
            **asdict(selected),
            "confirmation_positive": confirmation_positive,
            "pooled": {
                "intrinsic_v4": as_dict(pooled_intrinsic_metrics),
                "market": as_dict(pooled_market_metrics),
                "candidate": as_dict(pooled_metrics),
                "top1_lift_vs_market_pp": (pooled_metrics.top1 - pooled_market_metrics.top1) * 100.0,
                "top1_lift_vs_intrinsic_v4_pp": (pooled_metrics.top1 - pooled_intrinsic_metrics.top1) * 100.0,
                "bootstrap_vs_market": pooled_bootstrap,
            },
            "observed_holdout": {
                "from": FRESH_FROM,
                "market": as_dict(metrics(fresh, fresh_market)),
                "candidate": as_dict(metrics(fresh, fresh_selected)),
                "bootstrap_vs_market": fresh_bootstrap,
                "is_pristine": False,
            },
            "retrospective_metric_gate": retrospective_metric_gate,
        },
        "top_candidates": [
            asdict(item)
            for item in sorted(
                evidence,
                key=lambda item: (
                    min(item.top1_lifts_pp[:3]),
                    float(np.mean(item.top1_lifts_pp[:3])),
                ),
                reverse=True,
            )[:30]
        ],
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
