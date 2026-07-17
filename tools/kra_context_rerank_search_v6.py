from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — shared scikit-learn walk-forward contract


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_candidate_features import add_candidate_features  # noqa: E402
from kra_model_candidates import fit_candidate  # noqa: E402
from kra_model_evaluation import metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_promotion_policy import MIN_ABSOLUTE_TOP1_LIFT_PP, clears_absolute_top1_lift  # noqa: E402
from kra_pairwise_ranker import PairwiseModel, build_winner_pairs, pairwise_scores  # noqa: E402
from kra_pairwise_reranker import RaceScores, RerankPolicy, restricted_rerank  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_autoresearch_v5 import _candidate_columns, _prepare_candidate_frame  # noqa: E402
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402
from tools.kra_model_search_v5 import FoldData, _fold_data  # noqa: E402
from tools.kra_pairwise_search_v5 import SPECS  # noqa: E402


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_context_rerank_v6_results.json"
CONTEXT_GROUPS: Final = ("recent_form", "distance_form", "meet_form")
CONTEXT_WEIGHTS: Final = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
RERANK_WEIGHTS: Final = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    context_weight: float
    rerank_weight: float
    fold_lifts_pp: tuple[float, ...]
    fold_net_wins: tuple[int, ...]


def select_candidate(candidates: tuple[CandidateScore, ...]) -> CandidateScore:
    return max(
        candidates,
        key=lambda candidate: (
            min(candidate.fold_net_wins[:3]),
            float(np.mean(candidate.fold_lifts_pp[:3])),
            -candidate.context_weight,
        ),
    )


def _net_wins(frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray) -> int:
    scored = frame[["rk", "win"]].copy()
    scored["candidate"] = candidate
    scored["baseline"] = baseline
    net = 0
    for _, race in scored.groupby("rk", sort=False):
        candidate_hit = int(race.loc[race["candidate"].idxmax(), "win"])
        baseline_hit = int(race.loc[race["baseline"].idxmax(), "win"])
        net += candidate_hit - baseline_hit
    return net


def _candidate_scores(
    folds: list[FoldData],
    pairwise: list[np.ndarray],
    context: list[np.ndarray],
    context_weight: float,
    rerank_weight: float,
) -> tuple[list[np.ndarray], CandidateScore]:
    scores = []
    lifts = []
    net_wins = []
    for fold, pairwise_probability, context_probability in zip(folds, pairwise, context):
        ensemble = (
            (1.0 - context_weight) * pairwise_probability
            + context_weight * context_probability
        )
        reranked = restricted_rerank(
            fold.test,
            RaceScores(fold.baseline_probability, ensemble),
            RerankPolicy(rerank_weight, 3),
        )
        baseline_metrics = metrics(fold.test, fold.baseline_probability)
        candidate_metrics = metrics(fold.test, reranked.scores)
        scores.append(reranked.scores)
        lifts.append((candidate_metrics.top1 - baseline_metrics.top1) * 100.0)
        net_wins.append(_net_wins(fold.test, reranked.scores, fold.baseline_probability))
    return scores, CandidateScore(
        context_weight=context_weight,
        rerank_weight=rerank_weight,
        fold_lifts_pp=tuple(lifts),
        fold_net_wins=tuple(net_wins),
    )


def run_search(db_path: Path) -> dict:  # noqa: DICT_OK — JSON experiment report
    baseline_frame, baseline_columns = build_features(load_rows(db_path))
    frame = _prepare_candidate_frame(add_candidate_features(baseline_frame))
    context_columns = _candidate_columns(baseline_columns, CONTEXT_GROUPS)
    folds = _fold_data(frame, baseline_columns)
    pairwise_spec = next(spec for spec in SPECS if spec.name == "pair_hgb_d3")
    pairwise_probabilities = []
    context_probabilities = []
    for fold in folds:
        pairs = build_winner_pairs(fold.train, baseline_columns)
        pair_training = pairs.values.copy()
        pair_training["target"] = pairs.targets
        pair_estimator, pair_median = fit_candidate(
            pair_training, baseline_columns, "target", pairwise_spec
        )
        pairwise_probabilities.append(race_normalize(
            fold.test,
            pairwise_scores(
                PairwiseModel(pair_estimator, pair_median),
                fold.test,
                baseline_columns,
            ),
        ))
        context_estimator, context_median = _fit(
            fold.train, context_columns, "win", CONFIGS[0]
        )
        context_probabilities.append(race_normalize(
            fold.test,
            _predict(context_estimator, context_median, fold.test, context_columns),
        ))

    score_cache: dict[tuple[float, float], list[np.ndarray]] = {}
    candidates = []
    for context_weight in CONTEXT_WEIGHTS:
        for rerank_weight in RERANK_WEIGHTS:
            scores, candidate = _candidate_scores(
                folds,
                pairwise_probabilities,
                context_probabilities,
                context_weight,
                rerank_weight,
            )
            score_cache[(context_weight, rerank_weight)] = scores
            candidates.append(candidate)
    selected = select_candidate(tuple(candidates))
    selected_scores = score_cache[(selected.context_weight, selected.rerank_weight)]
    _, v5 = _candidate_scores(
        folds, pairwise_probabilities, context_probabilities, 0.0, 0.5
    )
    v5_scores = score_cache[(0.0, 0.5)]
    pooled_frame = pd.concat([fold.test for fold in folds], ignore_index=True)
    pooled_baseline = np.concatenate([fold.baseline_probability for fold in folds])
    pooled_selected = np.concatenate(selected_scores)
    pooled_v5 = np.concatenate(v5_scores)
    versus_v4 = paired_bootstrap_top1(
        pooled_frame, pooled_selected, pooled_baseline, samples=10000
    )
    versus_v5 = paired_bootstrap_top1(
        pooled_frame, pooled_selected, pooled_v5, samples=10000
    )
    promotion_pass = (
        clears_absolute_top1_lift(versus_v4["mean_pp"])
        and all(lift > 0 for lift in selected.fold_lifts_pp)
        and versus_v4["ci95_low_pp"] > 0
        and versus_v5["ci95_low_pp"] > 0
    )
    return {
        "method": "v5_pairwise_plus_recent_distance_meet_top3_restricted",
        "selection_policy": "first_three_folds_max_min_net_then_mean_lift",
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "selected": asdict(selected),
        "v5_reference": asdict(v5),
        "versus_v4_bootstrap": versus_v4,
        "versus_v5_bootstrap": versus_v5,
        "top_three_membership": "unchanged_v4",
        "probabilities": "unchanged_v4",
        "promotion_pass": promotion_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_search(args.db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
