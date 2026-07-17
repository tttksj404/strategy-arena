#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_drug_assay import build_fold_tensor, screen_genomes  # noqa: E402
from kra_drug_discovery import (  # noqa: E402
    AssayScore,
    Genome,
    MINIMUM_MARKET_ADVANTAGE_PP,
    MARKET_PARITY_TOLERANCE_PP,
    generate_hybrid_library,
    market_parity_pass,
    select_frontier,
)
from kra_drug_models import load_or_fit_predictions  # noqa: E402
from kra_model_evaluation import as_dict, market_probability, metrics, paired_bootstrap_top1  # noqa: E402
from kra_sectional_features import build_sectional_features  # noqa: E402
from kra_hierarchical_search import hierarchical_screen  # noqa: E402
from kra_candidate_features import CANDIDATE_GROUPS, add_candidate_features  # noqa: E402
from kra_card_features import CARD_FEATURES, add_card_features  # noqa: E402
from kra_pedigree_features import build_pedigree_features, load_pedigree, merge_pedigree  # noqa: E402
from kra_training_features import build_features, load_rows  # noqa: E402
from tools.kra_dual_phase_experiment import DEFAULT_DB  # noqa: E402
from tools.kra_max_winrate_search import _fold_data  # noqa: E402


DEFAULT_REPORT: Final = ROOT / "runs" / "kra_drug_discovery_results.json"
DEFAULT_STATE: Final = ROOT / "runs" / "kra_drug_discovery_state.json"
DEFAULT_LEDGER: Final = ROOT / "runs" / "kra_drug_discovery_ledger.jsonl"
DEFAULT_CACHE: Final = ROOT / "runs" / "kra_drug_discovery_predictions.npz"


def _load_state(path: Path) -> dict:  # noqa: DICT_OK — durable JSON state boundary
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parents(state: dict, base_count: int) -> tuple[Genome, ...]:
    parents = []
    for row in state.get("frontier", []):
        weights = tuple(float(value) for value in row.get("weights", []))
        if len(weights) == base_count:
            parents.append(Genome(weights, float(row.get("market_weight", 0.0))))
    return tuple(parents)


def _probability(
    genome: Genome,
    base_predictions: np.ndarray,
    market: np.ndarray,
) -> np.ndarray:
    return np.asarray(genome.weights) @ base_predictions + genome.market_weight * market


def _fold_report(score: AssayScore, folds: list, predictions: tuple[np.ndarray, ...]) -> list[dict]:
    rows = []
    for fold, base_predictions in zip(folds, predictions):
        market = market_probability(fold.test)
        candidate = _probability(score.genome, base_predictions, market)
        candidate_metrics = metrics(fold.test, candidate)
        market_metrics = metrics(fold.test, market)
        v4_metrics = metrics(fold.test, fold.baseline_probability)
        rows.append({
            "fold": fold.name,
            "candidate": as_dict(candidate_metrics),
            "market": as_dict(market_metrics),
            "v4": as_dict(v4_metrics),
            "candidate_vs_market_top1_pp": (
                candidate_metrics.top1 - market_metrics.top1
            ) * 100.0,
            "candidate_vs_v4_top1_pp": (
                candidate_metrics.top1 - v4_metrics.top1
            ) * 100.0,
            "bootstrap_vs_market": paired_bootstrap_top1(
                fold.test, candidate, market, samples=4000
            ),
            "bootstrap_vs_v4": paired_bootstrap_top1(
                fold.test, candidate, fold.baseline_probability, samples=4000
            ),
        })
    return rows


def _score_payload(score: AssayScore, names: tuple[str, ...]) -> dict:
    return {
        "weights": {
            name: float(weight)
            for name, weight in zip(names, score.genome.weights)
            if weight >= 0.001
        },
        "market_weight": score.genome.market_weight,
        "discovery_market_gaps_pp": score.discovery_market_gaps_pp,
        "discovery_v4_lifts_pp": score.discovery_v4_lifts_pp,
        "confirmation_market_gap_pp": score.confirmation_market_gap_pp,
        "confirmation_v4_lift_pp": score.confirmation_v4_lift_pp,
        "historical_market_parity_pass": market_parity_pass(score),
    }


def run_search(args: argparse.Namespace) -> dict:  # noqa: DICT_OK — JSON research report
    started = time.perf_counter()
    source = merge_pedigree(load_rows(args.db), load_pedigree(args.db))
    baseline_frame, baseline_columns = build_features(source)
    enriched = add_candidate_features(baseline_frame)
    candidate_columns = [
        *baseline_columns,
        *[column for group in CANDIDATE_GROUPS.values() for column in group],
    ]
    enriched = add_card_features(enriched)
    candidate_columns.extend(CARD_FEATURES)
    enriched, pedigree_columns = build_pedigree_features(enriched, candidate_columns)
    frame, candidate_columns = build_sectional_features(enriched, pedigree_columns)
    folds = _fold_data(frame, baseline_columns)
    bundle = load_or_fit_predictions(
        folds, candidate_columns, args.cache, args.refresh_cache
    )
    tensors = tuple(
        build_fold_tensor(
            fold.name,
            fold.test,
            predictions,
            market_probability(fold.test),
            v4_model_index=0,
        )
        for fold, predictions in zip(folds, bundle.fold_predictions)
    )
    state = _load_state(args.state)
    seed = int(state.get("next_seed", args.seed))
    parents = _parents(state, len(bundle.names))
    tested = 0
    frontier: tuple[AssayScore, ...] = ()
    stage_reports = []
    assay_evaluations = 0
    for generation in range(args.generations):
        library = generate_hybrid_library(
            base_count=len(bundle.names),
            requested=args.candidates,
            seed=seed + generation,
            maximum_market_weight=args.maximum_market_weight,
            parents=parents,
        )
        scores, stages = hierarchical_screen(
            library,
            tensors,
            args.batch_size,
            stage_beams=(256, 64, max(args.beam_width, 64)),
        )
        stage_reports.extend({
            "generation": generation,
            "fold_count": stage.fold_count,
            "input_count": stage.input_count,
            "output_count": stage.output_count,
            "assay_evaluations": stage.assay_evaluations,
            "best_market_gap_pp": stage.best_market_gap_pp,
        } for stage in stages)
        assay_evaluations += sum(stage.assay_evaluations for stage in stages)
        frontier = select_frontier(scores, args.beam_width)
        parents = tuple(score.genome for score in frontier)
        tested += len(library)
    selected = frontier[0]
    historical_parity = market_parity_pass(selected)
    cycle = int(state.get("cycle", 0)) + 1
    tested_total = int(state.get("tested_total", 0)) + tested
    status = "awaiting_pristine_forward" if historical_parity else "searching"
    selected_payload = _score_payload(selected, bundle.names)
    selected_payload["folds"] = _fold_report(
        selected, folds, bundle.fold_predictions
    )
    report = {
        "status": status,
        "method": "hybrid_structured_pair_boundary_plus_multifidelity_pareto_diversity_beam",
        "cycle": cycle,
        "seed": seed,
        "generations": args.generations,
        "candidates_per_generation": args.candidates,
        "tested_this_cycle": tested,
        "tested_total": tested_total,
        "assay_evaluations_this_cycle": assay_evaluations,
        "stage_beams": [256, 64, max(args.beam_width, 64)],
        "hierarchical_stages": stage_reports,
        "model_library": bundle.names,
        "prediction_cache_hit": bundle.cache_hit,
        "prediction_fingerprint": bundle.fingerprint,
        "market_weight_cap": args.maximum_market_weight,
        "minimum_market_advantage_pp": MINIMUM_MARKET_ADVANTAGE_PP,
        "market_parity_tolerance_pp": MARKET_PARITY_TOLERANCE_PP,
        "selection_uses_confirmation": False,
        "selected": selected_payload,
        "frontier": [_score_payload(score, bundle.names) for score in frontier],
        "historical_market_parity_pass": historical_parity,
        "pristine_forward_pass": False,
        "promotion_pass": False,
        "elapsed_sec": time.perf_counter() - started,
    }
    state_payload = {
        "cycle": cycle,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "next_seed": seed + args.generations,
        "tested_total": tested_total,
        "status": status,
        "best": selected_payload,
        "frontier": [
            {
                "weights": list(score.genome.weights),
                "market_weight": score.genome.market_weight,
            }
            for score in frontier
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.state.write_text(
        json.dumps(state_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.ledger.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps({
            "cycle": cycle,
            "seed": seed,
            "tested": tested,
            "status": status,
            "selected": selected_payload,
        }, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--candidates", type=int, default=20000)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--maximum-market-weight", type=float, default=0.15)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    report = run_search(args)
    print(json.dumps({
        "status": report["status"],
        "cycle": report["cycle"],
        "tested_this_cycle": report["tested_this_cycle"],
        "tested_total": report["tested_total"],
        "cache_hit": report["prediction_cache_hit"],
        "selected": report["selected"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
