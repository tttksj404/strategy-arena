#!/usr/bin/env python3
"""Assay native categorical and listwise models on official KRA pre-race data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — chronological fold evaluation contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_model_evaluation import as_dict, market_probability, metrics, paired_bootstrap_top1, race_normalize
from kra_official_model_search import SPECS, fit_candidate, passes_strict_historical_gate, selection_key
from kra_official_prerace_enrichment import add_official_prerace_enrichment
from kra_official_prerace_features import add_official_prerace_features, load_official_prerace_sources
from kra_training_features import build_features, load_rows
from tools.kra_dual_phase_experiment import CONFIGS, DEFAULT_DB, _fit, _predict


DEFAULT_ARCHIVE: Final = Path("/Users/tttksj/kra/data/pre_race_api.sqlite")
DEFAULT_REPORT: Final = ROOT / "runs" / "kra_official_model_search_results.json"
FOLDS: Final = (
    ("2024H2", "20240115", "20240701", "20250101"),
    ("2025H1", "20240115", "20250101", "20250701"),
    ("2025H2", "20240115", "20250701", "20260101"),
    ("2026H1", "20240115", "20260101", "20260701"),
)
BASELINE_CONFIG: Final = CONFIGS[1]


@dataclass(frozen=True, slots=True)
class PreparedFold:
    """One immutable chronological test boundary and its fixed reference scores."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    baseline: np.ndarray
    market: np.ndarray


def _prepare_folds(frame: pd.DataFrame, baseline_columns: list[str]) -> tuple[PreparedFold, ...]:
    prepared: list[PreparedFold] = []
    for name, train_from, test_from, test_until in FOLDS:
        train = frame[(frame["rcDate"] >= train_from) & (frame["rcDate"] < test_from)]
        test = frame[(frame["rcDate"] >= test_from) & (frame["rcDate"] < test_until)]
        model, median = _fit(train, baseline_columns, "win", BASELINE_CONFIG)
        baseline = race_normalize(test, _predict(model, median, test, baseline_columns))
        prepared.append(PreparedFold(name, train, test, baseline, market_probability(test)))
    return tuple(prepared)


def _evaluate_spec(  # noqa: DICT_OK — serialized experiment artifact
    prepared: tuple[PreparedFold, ...], numeric_columns: list[str], spec
) -> dict:
    rows: list[dict] = []  # noqa: DICT_OK — persisted experiment artifact
    tests: list[pd.DataFrame] = []
    candidates: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    markets: list[np.ndarray] = []
    for fold in prepared:
        candidate = fit_candidate(fold.train, numeric_columns, spec).predict(fold.test)
        candidate_metrics = metrics(fold.test, candidate)
        baseline_metrics = metrics(fold.test, fold.baseline)
        market_metrics = metrics(fold.test, fold.market)
        rows.append({
            "fold": fold.name,
            "races": int(fold.test["rk"].nunique()),
            "candidate": as_dict(candidate_metrics),
            "baseline": as_dict(baseline_metrics),
            "market": as_dict(market_metrics),
            "top1_lift_vs_baseline_pp": (candidate_metrics.top1 - baseline_metrics.top1) * 100.0,
            "top1_lift_vs_market_pp": (candidate_metrics.top1 - market_metrics.top1) * 100.0,
            "top3_lift_vs_market_pp": (candidate_metrics.top3 - market_metrics.top3) * 100.0,
            "logloss_delta_vs_market": candidate_metrics.race_logloss - market_metrics.race_logloss,
        })
        tests.append(fold.test)
        candidates.append(candidate)
        baselines.append(fold.baseline)
        markets.append(fold.market)
    frame = pd.concat(tests, ignore_index=True)
    candidate_values = np.concatenate(candidates)
    baseline_values = np.concatenate(baselines)
    market_values = np.concatenate(markets)
    baseline_lifts = tuple(float(row["top1_lift_vs_baseline_pp"]) for row in rows)
    market_lifts = tuple(float(row["top1_lift_vs_market_pp"]) for row in rows)
    return {
        "name": spec.name,
        "family": spec.family.value,
        "uses_market_features": False,
        "folds": rows,
        "selection_key": selection_key(baseline_lifts, market_lifts),
        "strict_historical_gate": passes_strict_historical_gate(baseline_lifts, market_lifts),
        "bootstrap_vs_baseline": paired_bootstrap_top1(frame, candidate_values, baseline_values, samples=10_000),
        "bootstrap_vs_market": paired_bootstrap_top1(frame, candidate_values, market_values, samples=10_000),
    }


def run_experiment(  # noqa: DICT_OK — persisted report boundary
    db_path: Path, archive: Path, enriched: bool = False
) -> dict:
    """Run the fixed candidate library; selection never reads the locked fourth fold."""
    base_frame, baseline_columns = build_features(load_rows(db_path))
    sources = load_official_prerace_sources(archive)
    frame, official_columns = add_official_prerace_features(base_frame, sources)
    enrichment_columns: list[str] = []
    if enriched:
        frame, enrichment_columns = add_official_prerace_enrichment(frame, sources)
    prepared = _prepare_folds(frame, baseline_columns)
    numeric_columns = [*baseline_columns, *official_columns, *enrichment_columns]
    candidates = [_evaluate_spec(prepared, numeric_columns, spec) for spec in SPECS]
    selected = max(candidates, key=lambda result: tuple(result["selection_key"]))
    return {
        "status": "historical_research_only",
        "data_availability": "historical_availability_unverified",
        "references": {
            "intrinsic": "hgb_d4_base_features",
            "market": "retrospective_closing_odds_only",
        },
        "candidate_feature_count": len(numeric_columns),
        "feature_variant": "enriched" if enriched else "basic",
        "candidates": candidates,
        "selected": selected,
        "promotion_pass": False,
        "promotion_reason": "historical sources and retrospective market rows do not prove pre-start availability",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--enriched", action="store_true")
    args = parser.parse_args()
    report = run_experiment(args.db, args.archive, args.enriched)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": report["selected"]["name"],
        "strict_historical_gate": report["selected"]["strict_historical_gate"],
        "promotion_pass": report["promotion_pass"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
