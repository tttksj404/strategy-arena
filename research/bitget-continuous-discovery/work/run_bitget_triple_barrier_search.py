#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly:
#      uv run --script run_bitget_triple_barrier_search.py --interval 4H
# ──────────────────

"""Search matured triple-barrier meta-label portfolios on public Bitget candles."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Final

import numpy as np

from alt_search_engine import evaluate_path, test_passes, validation_score
from alt_search_models import MarketData, PathResult, RiskOverlay, SegmentMetrics
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_search_metrics import development_moments, metric_payload, segment_metrics
from bitget_triple_barrier import (
    TripleBarrierConfig,
    walkforward_triple_barrier_positions,
)
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import (
    DSR_MIN,
    PBO_BINS,
    PBO_MAX,
    SearchSpec,
    btc_regime_mask,
    spec_for_interval,
)
from selection_bias_stats import (
    DeflatedSharpeResult,
    SelectionBiasError,
    cscv_pbo_from_moments,
    deflated_sharpe,
)

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_triple_barrier_search_report.json")
DEFAULT_INTERVAL: Final = "4H"
BTC_TREND_LIMIT: Final = 1.5


@dataclass(frozen=True, slots=True)
class SearchConfig:
    model: TripleBarrierConfig
    btc_volatility_cap: float
    target_bar_volatility: float
    risk_window: int

    @property
    def identifier(self) -> str:
        return (
            f"{self.model.identifier}|btc_trend_z<={BTC_TREND_LIMIT:g}|btc_vol<={self.btc_volatility_cap:g}"
            f"|target_bar_vol={self.target_bar_volatility:g}|risk_window={self.risk_window}|max_leverage=5"
        )

    @property
    def overlay(self) -> RiskOverlay:
        return RiskOverlay(5.0, self.target_bar_volatility, self.risk_window)


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    config: SearchConfig
    score: float


@dataclass(frozen=True, slots=True)
class LockedResult:
    identifier: str
    validation_score: float
    test: SegmentMetrics
    stress: SegmentMetrics
    liquidated: bool
    dsr: DeflatedSharpeResult | None
    paper_pass: bool
    pbo_pass: bool
    dsr_pass: bool


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    serialized_report: str
    pbo: float
    validation_winner_count: int


def _scaled_bars(spec: SearchSpec, base_bars: tuple[int, ...]) -> tuple[int, ...]:
    scale = spec.spread_lookbacks[0] / 30.0
    return tuple(dict.fromkeys(max(2, round(value * scale)) for value in base_bars))


def _search_configs(spec: SearchSpec) -> tuple[SearchConfig, ...]:
    training_samples = max(20, round(100 * spec.spread_lookbacks[0] / 30.0))
    return tuple(
        SearchConfig(model, volatility_cap, target_volatility, spec.risk_windows[0])
        for model in (
            TripleBarrierConfig(
                feature_lookback=lookback,
                barrier_horizon=horizon,
                profit_take=barrier,
                stop_loss=barrier,
                ridge=ridge,
                rebalance_bars=_scaled_bars(spec, (6,))[0],
                minimum_training_samples=training_samples,
                minimum_quote_volume=spec.minimum_quote_volume,
                prediction_threshold=0.0,
            )
            for lookback in _scaled_bars(spec, (12, 30))
            for horizon in _scaled_bars(spec, (6, 12))
            for barrier in (0.01, 0.02)
            for ridge in (0.1, 1.0)
        )
        for volatility_cap in spec.volatility_caps
        for target_volatility in spec.target_bar_volatilities
    )


def _validation_record(path: PathResult, config: SearchConfig, *, train_end: int, development_end: int, bars_per_year: float) -> ValidationRecord:
    train, train_stress, train_liquidated = segment_metrics(path, 0, train_end, bars_per_year=bars_per_year)
    validation, validation_stress, validation_liquidated = segment_metrics(path, train_end, development_end, bars_per_year=bars_per_year)
    score = validation_score(validation, validation_stress, validation_liquidated)
    if not isfinite(validation_score(train, train_stress, train_liquidated)):
        score = float("-inf")
    return ValidationRecord(config, score)


def _locked_result(
    record: ValidationRecord,
    *,
    positions: np.ndarray,
    data: MarketData,
    locked_start: int,
    bars_per_year: float,
    raw_trials: int,
    pbo: float,
) -> LockedResult:
    path = evaluate_path(data, positions, record.config.overlay)
    test, stress, liquidated = segment_metrics(path, locked_start, len(data.dates), bars_per_year=bars_per_year)
    try:
        dsr = deflated_sharpe(path.stress_returns[locked_start:], trials=raw_trials)
    except SelectionBiasError:
        dsr = None
    paper_pass = test_passes(test, stress, liquidated)
    dsr_pass = dsr is not None and dsr.probability >= DSR_MIN
    return LockedResult(record.config.identifier, record.score, test, stress, liquidated, dsr, paper_pass, pbo <= PBO_MAX, dsr_pass)


def run(archive_root: Path, *, interval: str = DEFAULT_INTERVAL) -> SearchOutcome:
    spec = spec_for_interval(interval)
    data, metadata = load_verified_interval_archive(archive_root, ("BTC", *ALT_SYMBOLS), interval=spec.interval)
    configs = _search_configs(spec)
    train_end, development_end = int(len(data.dates) * 0.5), int(len(data.dates) * 0.75)
    btc_index = data.symbols.index("BTC")
    masks = {
        cap: btc_regime_mask(
            data.close_price[:, btc_index],
            trend_bars=spec.trend_bars,
            volatility_bars=spec.volatility_bars,
            trend_limit=BTC_TREND_LIMIT,
            volatility_cap=cap,
        )
        for cap in spec.volatility_caps
    }
    all_sums: list[np.ndarray] = []
    all_squares: list[np.ndarray] = []
    bin_counts: tuple[int, ...] | None = None
    winner: ValidationRecord | None = None
    winner_positions: np.ndarray | None = None
    for completed, config in enumerate(configs, start=1):
        positions = walkforward_triple_barrier_positions(data, config.model, alt_symbols=ALT_SYMBOLS)
        positions[~masks[config.btc_volatility_cap]] = 0.0
        path = evaluate_path(data, positions, config.overlay)
        record = _validation_record(path, config, train_end=train_end, development_end=development_end, bars_per_year=spec.bars_per_year)
        if isfinite(record.score) and (winner is None or record.score > winner.score):
            winner, winner_positions = record, positions
        sums, squares, counts = development_moments(path.stress_returns, development_end=development_end, bins=PBO_BINS)
        all_sums.append(sums)
        all_squares.append(squares)
        if bin_counts is None:
            bin_counts = counts
        elif bin_counts != counts:
            raise SelectionBiasError("triple-barrier configurations produced incompatible CSCV bin counts")
        if completed % 25 == 0 or completed == len(configs):
            print(f"SEARCH_PROGRESS configurations={completed}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("triple-barrier search generated no configurations")
    pbo = cscv_pbo_from_moments(np.asarray(all_sums), np.asarray(all_squares), bin_counts)
    locked: list[LockedResult] = []
    if winner is not None and winner_positions is not None:
        locked.append(
            _locked_result(
                winner,
                positions=winner_positions,
                data=data,
                locked_start=development_end,
                bars_per_year=spec.bars_per_year,
                raw_trials=len(configs),
                pbo=pbo.pbo,
            )
        )
    locked_payload = [
        {
            "id": result.identifier,
            "algorithm": "strictly_lagged_matured_triple_barrier_ridge_meta_label",
            "validation_score": result.validation_score,
            "test": metric_payload(result.test),
            "test_stress": metric_payload(result.stress),
            "test_liquidated": result.liquidated,
            "deflated_sharpe_stress": None if result.dsr is None else asdict(result.dsr),
            "paper_pass": result.paper_pass,
            "pbo_pass": result.pbo_pass,
            "dsr_pass": result.dsr_pass,
            "eligible_for_mark_price_paper_validation": result.paper_pass and result.pbo_pass and result.dsr_pass,
            "live_promotable": False,
        }
        for result in locked
    ]
    eligible = [result.identifier for result in locked if result.paper_pass and result.pbo_pass and result.dsr_pass]
    report = {
        "mode": "public-api-paper-only-no-live-orders",
        "selection_independent": False,
        "strategy": "matured triple-barrier labels with online ridge meta-model; labels are only consumed after their full horizon closes",
        "liquidation_evidence": "market-candle proxy only; full mark-price history remains required before any promotion",
        "data": {
            "interval": metadata.interval,
            "snapshot_ms": metadata.snapshot_ms,
            "common_rows": metadata.common_rows,
            "integrity_verified": metadata.integrity_verified,
            "source_hashes": list(metadata.source_hashes),
            "train_rows": train_end,
            "development_rows": development_end,
            "locked_test_rows": len(data.dates) - development_end,
        },
        "predeclared_search": {
            "algorithm_family": "matured_triple_barrier_online_ridge_meta_label",
            "raw_configurations": len(configs),
            "max_leverage": 5.0,
            "btc_trend_limit": BTC_TREND_LIMIT,
            "minimum_quote_volume_per_bar": spec.minimum_quote_volume,
        },
        "selection_bias": {
            "method": "CSCV PBO on stress development returns; DSR on locked stress returns",
            "pbo": asdict(pbo),
            "pbo_max": PBO_MAX,
            "dsr_min_probability": DSR_MIN,
        },
        "selected_validation_winners": [] if winner is None else [winner.config.identifier],
        "locked_test_results": locked_payload,
        "eligible_for_mark_price_paper_validation": eligible,
        "live_promotable": [],
    }
    return SearchOutcome(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2), pbo.pbo, 0 if winner is None else 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    try:
        outcome = run(arguments.archive_root, interval=arguments.interval)
    except (ArchivePanelError, SelectionBiasError) as error:
        print(f"SEARCH_ERROR={error}")
        return 1
    arguments.report.write_text(outcome.serialized_report, encoding="utf-8")
    print(f"SEARCH_DONE report={arguments.report}")
    print(f"PBO={outcome.pbo:.6f} WINNERS={outcome.validation_winner_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
