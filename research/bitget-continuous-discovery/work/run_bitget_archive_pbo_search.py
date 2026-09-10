#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Run a public-data-only broad alt search with CSCV PBO and DSR gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from alt_search_engine import evaluate_path, segment_metrics, test_passes, validation_score
from alt_search_models import Candidate, PathResult, RiskOverlay, SegmentMetrics
from alt_search_signals import build_positions
from bitget_archive_daily import ArchiveDailyError, load_verified_daily_archive
from run_alt_strategy_search import SYMBOLS, ValidationRecord, enumerate_candidates, enumerate_risk_overlays
from selection_bias_stats import SelectionBiasError, cscv_pbo_from_moments, deflated_sharpe


DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_daily_20260824_v2")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_archive_pbo_search_report.json")
PBO_BINS: Final = 8
PBO_MAX: Final = 0.05
DSR_MIN: Final = 0.95


def development_moments(returns: np.ndarray, *, development_end: int, bins: int = PBO_BINS) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Store only development-period sufficient statistics; never read the locked tail."""
    if development_end <= 0 or development_end >= len(returns):
        raise SelectionBiasError("development split must exclude a non-empty locked test tail")
    values = np.asarray(returns[:development_end], dtype=float)
    if values.ndim != 1 or bins < 2 or bins % 2 or values.size < bins * 2 or not np.isfinite(values).all():
        raise SelectionBiasError("development returns require an even bin count and at least two finite observations per bin")
    segments = tuple(np.asarray(segment, dtype=float) for segment in np.array_split(values, bins))
    return (
        np.asarray([segment.sum() for segment in segments], dtype=float),
        np.asarray([(segment * segment).sum() for segment in segments], dtype=float),
        tuple(segment.size for segment in segments),
    )


def _validation_record(
    path: PathResult,
    candidate: Candidate,
    overlay: RiskOverlay,
    train_end: int,
    validation_end: int,
) -> ValidationRecord:
    train, train_stress, train_liquidated = segment_metrics(path, 0, train_end)
    validation, validation_stress, validation_liquidated = segment_metrics(path, train_end, validation_end)
    score = validation_score(validation, validation_stress, validation_liquidated)
    if not isfinite(validation_score(train, train_stress, train_liquidated)):
        score = float("-inf")
    return ValidationRecord(candidate, overlay, train, train_stress, validation, validation_stress, train_liquidated, validation_liquidated, score)


def _winner_key(record: ValidationRecord) -> tuple[str, float, float]:
    return record.candidate.family.value, record.overlay.max_leverage, record.candidate.minimum_daily_quote_volume


def _metric_payload(metrics: SegmentMetrics) -> dict[str, float | int]:
    return {
        "net_return": metrics.net_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "positive_blocks": metrics.positive_blocks,
        "observations": metrics.observations,
    }


def _locked_result(record: ValidationRecord, *, data, locked_start: int, raw_trials: int, pbo: float) -> dict[str, object]:
    path = evaluate_path(data, build_positions(data, record.candidate), record.overlay)
    test, stress, liquidated = segment_metrics(path, locked_start, len(data.dates))
    try:
        dsr = asdict(deflated_sharpe(path.stress_returns[locked_start:], trials=raw_trials))
    except SelectionBiasError:
        dsr = None
    paper_pass = test_passes(test, stress, liquidated)
    dsr_pass = dsr is not None and float(dsr["probability"]) >= DSR_MIN
    return {
        "id": record.identifier,
        "candidate": record.candidate.identifier,
        "family": record.candidate.family.value,
        "risk": record.overlay.identifier,
        "test": _metric_payload(test),
        "test_stress": _metric_payload(stress),
        "test_liquidated": liquidated,
        "deflated_sharpe_stress": dsr,
        "paper_pass": paper_pass,
        "pbo_pass": pbo <= PBO_MAX,
        "dsr_pass": dsr_pass,
        "eligible_for_further_paper_validation": paper_pass and pbo <= PBO_MAX and dsr_pass,
        "live_promotable": False,
    }


def run(archive_root: Path) -> dict[str, object]:
    data, metadata = load_verified_daily_archive(archive_root, SYMBOLS)
    candidates, overlays = enumerate_candidates(), enumerate_risk_overlays()
    train_end, development_end = int(len(data.dates) * 0.5), int(len(data.dates) * 0.75)
    winners: dict[tuple[str, float, float], ValidationRecord] = {}
    all_sums: list[np.ndarray] = []
    all_squares: list[np.ndarray] = []
    bin_counts: tuple[int, ...] | None = None
    for index, candidate in enumerate(candidates, start=1):
        positions = build_positions(data, candidate)
        for overlay in overlays:
            path = evaluate_path(data, positions, overlay)
            record = _validation_record(path, candidate, overlay, train_end, development_end)
            if isfinite(record.score):
                key = _winner_key(record)
                current = winners.get(key)
                if current is None or record.score > current.score:
                    winners[key] = record
            sums, squares, counts = development_moments(path.stress_returns, development_end=development_end)
            all_sums.append(sums)
            all_squares.append(squares)
            if bin_counts is None:
                bin_counts = counts
            elif bin_counts != counts:
                raise SelectionBiasError("candidate paths produced incompatible CSCV bin counts")
        if index % 100 == 0 or index == len(candidates):
            print(f"SEARCH_PROGRESS signals={index}/{len(candidates)} configurations={len(all_sums)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("broad search generated no configurations")
    pbo = cscv_pbo_from_moments(np.asarray(all_sums), np.asarray(all_squares), bin_counts)
    selected = tuple(winners[key] for key in sorted(winners))
    locked = tuple(_locked_result(record, data=data, locked_start=development_end, raw_trials=len(all_sums), pbo=pbo.pbo) for record in selected)
    return {
        "mode": "public-api-paper-only-no-live-orders",
        "selection_independent": False,
        "data": {
            "archive": str(archive_root),
            "snapshot_ms": metadata.snapshot_ms,
            "common_rows": metadata.common_rows,
            "integrity_verified": metadata.integrity_verified,
            "source_hashes": list(metadata.source_hashes),
            "train_rows": train_end,
            "development_rows": development_end,
            "locked_test_rows": len(data.dates) - development_end,
        },
        "selection_bias": {
            "method": "CSCV PBO on cost-adjusted stress development returns; DSR on locked-test stress returns",
            "raw_configurations": len(all_sums),
            "pbo": asdict(pbo),
            "pbo_max": PBO_MAX,
            "dsr_min_probability": DSR_MIN,
        },
        "winners": [
            {
                "id": record.identifier,
                "candidate": record.candidate.identifier,
                "family": record.candidate.family.value,
                "risk": record.overlay.identifier,
                "validation_score": record.score,
            }
            for record in selected
        ],
        "locked_test_results": list(locked),
        "eligible_for_further_paper_validation": [result["id"] for result in locked if result["eligible_for_further_paper_validation"]],
        "live_promotable": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    try:
        report = run(arguments.archive_root)
    except (ArchiveDailyError, SelectionBiasError) as error:
        print(f"SEARCH_ERROR={error}")
        return 1
    arguments.report.write_text(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SEARCH_DONE report={arguments.report}")
    print(f"PBO={report['selection_bias']['pbo']['pbo']:.6f} WINNERS={len(report['winners'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
