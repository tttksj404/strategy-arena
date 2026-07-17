from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_model_candidates import fit_candidate  # noqa: E402
from kra_model_evaluation import as_dict, metrics, paired_bootstrap_top1, race_normalize  # noqa: E402
from kra_pairwise_ranker import PairwiseModel, build_winner_pairs, pairwise_scores  # noqa: E402
from kra_pairwise_reranker import RaceScores, RerankPolicy, restricted_rerank  # noqa: E402
from kra_promotion_policy import (  # noqa: E402
    MIN_ABSOLUTE_TOP1_LIFT_PP,
    clears_absolute_top1_lift,
)
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict  # noqa: E402
from tools.kra_pairwise_search_v5 import SPECS  # noqa: E402


DEFAULT_FROM: Final = "20260622"
DEFAULT_REPORT: Final = ROOT / "runs" / "kra_fresh_holdout_20260622.json"


@dataclass(frozen=True, slots=True)
class HoldoutEvidence:
    top1_lift_pp: float
    net_wins: int
    ci95_low_pp: float


def promotion_passes(evidence: HoldoutEvidence) -> bool:
    return (
        clears_absolute_top1_lift(evidence.top1_lift_pp)
        and evidence.net_wins > 0
        and evidence.ci95_low_pp > 0
    )


def _net_wins(frame, candidate: np.ndarray, baseline: np.ndarray) -> int:
    net = 0
    for positions in frame.groupby("rk", sort=False).indices.values():
        candidate_hit = int(frame.iloc[positions[int(np.argmax(candidate[positions]))]]["win"])
        baseline_hit = int(frame.iloc[positions[int(np.argmax(baseline[positions]))]]["win"])
        net += candidate_hit - baseline_hit
    return net


def run_guard(db_path: Path, holdout_from: str) -> dict:  # noqa: DICT_OK — JSON validation report
    frame, columns = build_features(load_rows(db_path))
    train = frame[frame["rcDate"] < holdout_from]
    holdout = frame[frame["rcDate"] >= holdout_from].copy()
    baseline_estimator, baseline_median = _fit(train, columns, "win", CONFIGS[0])
    baseline_probability = race_normalize(
        holdout,
        _predict(baseline_estimator, baseline_median, holdout, columns),
    )
    pairwise_spec = next(spec for spec in SPECS if spec.name == "pair_hgb_d3")
    pairs = build_winner_pairs(train, columns)
    pair_training = pairs.values.copy()
    pair_training["target"] = pairs.targets
    pair_estimator, pair_median = fit_candidate(
        pair_training, columns, "target", pairwise_spec
    )
    pairwise_probability = race_normalize(
        holdout,
        pairwise_scores(
            PairwiseModel(pair_estimator, pair_median),
            holdout,
            columns,
        ),
    )
    candidate_probability = restricted_rerank(
        holdout,
        RaceScores(baseline_probability, pairwise_probability),
        RerankPolicy(0.5, 3),
    ).scores
    bootstrap = paired_bootstrap_top1(
        holdout,
        candidate_probability,
        baseline_probability,
        samples=10000,
    )
    baseline_metrics = metrics(holdout, baseline_probability)
    candidate_metrics = metrics(holdout, candidate_probability)
    evidence = HoldoutEvidence(
        top1_lift_pp=(candidate_metrics.top1 - baseline_metrics.top1) * 100.0,
        net_wins=_net_wins(holdout, candidate_probability, baseline_probability),
        ci95_low_pp=bootstrap["ci95_low_pp"],
    )
    return {
        "holdout_from": holdout_from,
        "minimum_absolute_top1_lift_pp": MIN_ABSOLUTE_TOP1_LIFT_PP,
        "baseline": as_dict(baseline_metrics),
        "pairwise_candidate": as_dict(candidate_metrics),
        "bootstrap": bootstrap,
        "evidence": asdict(evidence),
        "promotion_pass": promotion_passes(evidence),
        "decision": "enable_pairwise" if promotion_passes(evidence) else "keep_v4_baseline",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--from", dest="holdout_from", default=DEFAULT_FROM)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_guard(args.db, args.holdout_from)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
