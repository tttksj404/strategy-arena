#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pydantic>=2",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      powershell -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
# 2. Run directly (paper-only; no key required):
#      uv run work/run_alt_strategy_search.py
# ──────────────────

"""Run the preregistered broad, paper-only altcoin strategy search."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Final

from alt_search_data import CacheMetadata, load_frozen_daily_cache
from alt_search_engine import evaluate_path, segment_metrics, test_passes, validation_score
from alt_search_models import Candidate, Family, RiskOverlay, SegmentMetrics
from alt_search_signals import build_positions


CACHE_DIR: Final = Path(r"C:\Users\SSAFY\Desktop\2학기\strategy-arena-crypto\research\wave3\cache")
REPORT_PATH: Final = Path(__file__).with_name("bitget_alt_strategy_liquidity_search_report.json")
SYMBOLS: Final = ("ETH", "BNB", "ADA", "SOL", "DOGE", "XRP", "DOT", "LTC", "LINK", "AVAX", "UNI", "ATOM", "NEAR")
LOOKBACKS: Final = (3, 5, 10, 20, 40, 60, 90, 120)
REBALANCE_DAYS: Final = (1, 2, 3)
PAIR_SET: Final = (("ETH", "SOL"), ("ETH", "LINK"), ("SOL", "AVAX"), ("ADA", "DOT"), ("ATOM", "NEAR"), ("UNI", "LINK"))
LIQUIDITY_FLOORS: Final = (0.0, 1_000_000.0, 5_000_000.0, 20_000_000.0)


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """Validation-only evidence for one signal and risk configuration."""

    candidate: Candidate
    overlay: RiskOverlay
    train: SegmentMetrics
    train_stress: SegmentMetrics
    validation: SegmentMetrics
    validation_stress: SegmentMetrics
    train_liquidated: bool
    validation_liquidated: bool
    score: float

    @property
    def identifier(self) -> str:
        """Return an unambiguous record key."""

        return f"{self.candidate.identifier}|{self.overlay.identifier}"


def _candidate(family: Family, lookback: int, secondary: int = 0, top_k: int = 0, direction: int = 0, pair: tuple[str, str] | None = None, entry_z: float = 0.0) -> Candidate:
    """Construct a base signal prior to the fixed rebalance expansion."""

    return Candidate(family, lookback, secondary, top_k, direction, pair, entry_z)


def enumerate_candidates() -> tuple[Candidate, ...]:
    """Enumerate the fixed signal/rebalance grid across every volume-floor lane."""

    base: list[Candidate] = []
    base.extend(_candidate(Family.CROSS_SECTIONAL, lookback, top_k=top_k, direction=direction) for lookback in LOOKBACKS for top_k in (1, 2, 3, 4, 5) for direction in (1, -1))
    base.extend(_candidate(Family.TIME_SERIES, lookback, direction=direction) for lookback in (5, 10, 20, 40, 60, 90, 120) for direction in (1, -1))
    base.extend(_candidate(Family.DONCHIAN, lookback, direction=direction) for lookback in (10, 20, 40, 60, 90, 120) for direction in (1, -1))
    base.extend(_candidate(Family.EMA, fast, secondary=slow, direction=direction) for fast in (5, 10, 20) for slow in (20, 40, 60, 100) if fast < slow for direction in (1, -1))
    base.extend(_candidate(Family.BOLLINGER, lookback, top_k=top_k, direction=-1, entry_z=entry_z) for lookback in (10, 20, 40, 60, 90) for entry_z in (1.0, 1.5, 2.0) for top_k in (1, 3, 5))
    base.extend(_candidate(Family.RANGE, lookback, top_k=top_k, direction=direction) for lookback in (5, 10, 20, 40, 60) for top_k in (1, 3, 5) for direction in (1, -1))
    base.extend(_candidate(Family.PRICE_VOLUME, lookback, top_k=top_k, direction=direction) for lookback in (5, 10, 20, 40, 60) for top_k in (1, 3, 5) for direction in (1, -1))
    base.extend(_candidate(Family.PAIR, lookback, pair=pair, entry_z=entry_z) for pair in PAIR_SET for lookback in (20, 40, 60, 90, 120) for entry_z in (1.0, 1.5, 2.0))
    return tuple(
        replace(candidate, rebalance_days=days, minimum_daily_quote_volume=floor)
        for candidate in base
        for days in REBALANCE_DAYS
        for floor in LIQUIDITY_FLOORS
    )


def enumerate_risk_overlays() -> tuple[RiskOverlay, ...]:
    """Enumerate the 36 preregistered volatility and leverage overlays."""

    return tuple(RiskOverlay(maximum, target, window) for maximum in (1.0, 2.0, 3.0, 5.0) for target in (0.005, 0.01, 0.02) for window in (10, 20, 40))


def _metric_payload(metrics: SegmentMetrics) -> dict[str, float | int]:
    """Serialize a metric value object without NumPy scalar leakage."""

    return {"net_return": metrics.net_return, "max_drawdown": metrics.max_drawdown, "sharpe": metrics.sharpe, "positive_blocks": metrics.positive_blocks, "observations": metrics.observations}


def _validation_payload(record: ValidationRecord) -> dict[str, str | float | bool | dict[str, float | int]]:
    """Serialize validation evidence while intentionally omitting locked-test metrics."""

    return {
        "id": record.identifier,
        "candidate": record.candidate.identifier,
        "family": record.candidate.family.value,
        "risk": record.overlay.identifier,
        "max_leverage": record.overlay.max_leverage,
        "train": _metric_payload(record.train),
        "train_stress": _metric_payload(record.train_stress),
        "validation": _metric_payload(record.validation),
        "validation_stress": _metric_payload(record.validation_stress),
        "train_liquidated": record.train_liquidated,
        "validation_liquidated": record.validation_liquidated,
        "validation_score": record.score if isfinite(record.score) else None,
        "validation_eligible": isfinite(record.score),
    }


def _evaluate_validation(data, positions, candidate: Candidate, overlay: RiskOverlay, train_end: int, validation_end: int) -> ValidationRecord:
    """Evaluate train/validation only; this function never requests test metrics."""

    path = evaluate_path(data, positions, overlay)
    train, train_stress, train_liquidated = segment_metrics(path, 0, train_end)
    validation, validation_stress, validation_liquidated = segment_metrics(path, train_end, validation_end)
    score = validation_score(validation, validation_stress, validation_liquidated)
    if not isfinite(validation_score(train, train_stress, train_liquidated)):
        score = float("-inf")
    return ValidationRecord(candidate, overlay, train, train_stress, validation, validation_stress, train_liquidated, validation_liquidated, score)


def _select_winners(records: tuple[ValidationRecord, ...]) -> tuple[ValidationRecord, ...]:
    """Choose one validation-only winner for each family, leverage, and liquidity lane."""

    groups: dict[tuple[str, float, float], list[ValidationRecord]] = {}
    for record in records:
        groups.setdefault((record.candidate.family.value, record.overlay.max_leverage, record.candidate.minimum_daily_quote_volume), []).append(record)
    winners: list[ValidationRecord] = []
    for key in sorted(groups):
        eligible = [record for record in groups[key] if isfinite(record.score)]
        if eligible:
            winners.append(max(eligible, key=lambda record: record.score))
    return tuple(winners)


def _locked_test(data, winner: ValidationRecord, start: int, end: int) -> dict[str, str | float | bool | dict[str, float | int]]:
    """Evaluate a winner's test once after validation selection has completed."""

    path = evaluate_path(data, build_positions(data, winner.candidate), winner.overlay)
    test, stress, liquidated = segment_metrics(path, start, end)
    passed = test_passes(test, stress, liquidated)
    return {
        "id": winner.identifier,
        "candidate": winner.candidate.identifier,
        "family": winner.candidate.family.value,
        "risk": winner.overlay.identifier,
        "max_leverage": winner.overlay.max_leverage,
        "test": _metric_payload(test),
        "test_stress": _metric_payload(stress),
        "test_liquidated": liquidated,
        "paper_candidate": passed,
        "live_promotable": False,
    }


def _report_metadata(metadata: CacheMetadata, rows: int) -> dict[str, str | int | bool | list[str]]:
    """Build cache provenance for the report."""

    return {"cache": str(CACHE_DIR), "frozen_end": metadata.frozen_end, "common_rows": metadata.common_rows, "integrity_verified": metadata.integrity_verified, "source_hashes": list(metadata.source_hashes), "train_rows": int(rows * 0.5), "validation_rows": int(rows * 0.25), "locked_test_rows": rows - int(rows * 0.75)}


def main() -> int:
    """Execute the broad search and write its single reproducible evidence report."""

    data, metadata = load_frozen_daily_cache(CACHE_DIR, SYMBOLS)
    candidates, overlays = enumerate_candidates(), enumerate_risk_overlays()
    train_end, validation_end = int(len(data.dates) * 0.5), int(len(data.dates) * 0.75)
    print(f"SEARCH_PLAN candidates={len(candidates) * len(overlays)} signals={len(candidates)} overlays={len(overlays)} liquidity_floors={len(LIQUIDITY_FLOORS)} rows={len(data.dates)}", flush=True)
    collected: list[ValidationRecord] = []
    for index, candidate in enumerate(candidates, start=1):
        positions = build_positions(data, candidate)
        collected.extend(_evaluate_validation(data, positions, candidate, overlay, train_end, validation_end) for overlay in overlays)
        if index % 100 == 0 or index == len(candidates):
            print(f"SEARCH_PROGRESS signals={index}/{len(candidates)} configurations={len(collected)}", flush=True)
    records = tuple(collected)
    winners = _select_winners(records)
    locked = tuple(_locked_test(data, winner, validation_end, len(data.dates)) for winner in winners)
    paper_candidates = [result["id"] for result in locked if result["paper_candidate"]]
    report = {
        "mode": "paper-only-no-live-orders",
        "selection_independent": False,
        "data": _report_metadata(metadata, len(data.dates)),
        "contract": {"base_fee_per_side": 0.0006, "base_slippage_per_turnover": 0.0003, "stress_slippage_per_turnover": 0.0006, "liquidation_threshold": "1/leverage - 0.02 using next daily high/low", "test_evaluated_after_validation_selection": True, "minimum_promotable_test_days": 90},
        "grid": {"signal_rebalance_liquidity_configurations": len(candidates), "liquidity_floors": list(LIQUIDITY_FLOORS), "risk_configurations": len(overlays), "total_configurations": len(records), "families": [family.value for family in Family]},
        "validation_records": [_validation_payload(record) for record in records],
        "validation_winners": [_validation_payload(winner) for winner in winners],
        "locked_test_results": list(locked),
        "paper_candidates": paper_candidates,
        "live_promotable": [],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SEARCH_DONE report={REPORT_PATH} validation_winners={len(winners)} paper_candidates={len(paper_candidates)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
