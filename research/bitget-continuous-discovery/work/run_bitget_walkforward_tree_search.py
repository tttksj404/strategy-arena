#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2", "scikit-learn>=1.7"]
# ///

"""Evaluate strictly lagged ExtraTrees models on verified Bitget panels."""

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
from alt_search_models import MarketData, PathResult, RiskOverlay
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_search_metrics import development_moments, metric_payload, segment_metrics
from bitget_walkforward_tree import TreeConfig, walkforward_tree_positions
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
    SelectionBiasError,
    cscv_pbo_from_moments,
    deflated_sharpe,
)

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_walkforward_tree_search_report.json")
DEFAULT_INTERVAL: Final = "4H"
BTC_TREND_LIMIT: Final = 1.5
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    tree: TreeConfig
    btc_volatility_cap: float
    target_bar_volatility: float
    risk_window: int

    @property
    def identifier(self) -> str:
        return (
            f"{self.tree.identifier}|btc_trend_z<={BTC_TREND_LIMIT:g}|btc_vol<={self.btc_volatility_cap:g}"
            f"|target_bar_vol={self.target_bar_volatility:g}|risk_window={self.risk_window}|max_leverage=5"
        )

    @property
    def overlay(self) -> RiskOverlay:
        return RiskOverlay(5.0, self.target_bar_volatility, self.risk_window)


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    config: SearchConfig
    score: float


def _scaled_bars(spec: SearchSpec, base_bars: tuple[int, ...]) -> tuple[int, ...]:
    scale = spec.spread_lookbacks[0] / 30.0
    return tuple(dict.fromkeys(max(2, round(value * scale)) for value in base_bars))


def _tree_configs(spec: SearchSpec) -> tuple[TreeConfig, ...]:
    pairs = (
        (_scaled_bars(spec, (6,))[0], _scaled_bars(spec, (30,))[0]),
        (_scaled_bars(spec, (30,))[0], _scaled_bars(spec, (90,))[0]),
    )
    return tuple(
        TreeConfig(
            short,
            long,
            _scaled_bars(spec, (30,))[0],
            training,
            _scaled_bars(spec, (24,))[0],
            16,
            depth,
            8,
            1,
            _scaled_bars(spec, (6,))[0],
            spec.minimum_quote_volume,
        )
        for short, long in pairs
        for training in _scaled_bars(spec, (120, 240))
        for depth in (2, 4)
    )


def _search_configs(spec: SearchSpec) -> tuple[SearchConfig, ...]:
    return tuple(
        SearchConfig(tree, volatility_cap, target_volatility, spec.risk_windows[0])
        for tree in _tree_configs(spec)
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
) -> dict[str, JsonValue]:
    path = evaluate_path(data, positions, record.config.overlay)
    test, stress, liquidated = segment_metrics(path, locked_start, len(data.dates), bars_per_year=bars_per_year)
    try:
        dsr = asdict(deflated_sharpe(path.stress_returns[locked_start:], trials=raw_trials))
    except SelectionBiasError:
        dsr = None
    paper_pass = test_passes(test, stress, liquidated)
    dsr_pass = dsr is not None and float(dsr["probability"]) >= DSR_MIN
    return {
        "id": record.config.identifier,
        "algorithm": "strictly_lagged_extra_trees_return_forecast",
        "validation_score": record.score,
        "test": metric_payload(test),
        "test_stress": metric_payload(stress),
        "test_liquidated": liquidated,
        "deflated_sharpe_stress": dsr,
        "paper_pass": paper_pass,
        "pbo_pass": pbo <= PBO_MAX,
        "dsr_pass": dsr_pass,
        "eligible_for_mark_price_paper_validation": paper_pass and pbo <= PBO_MAX and dsr_pass,
        "live_promotable": False,
    }


def run(archive_root: Path, *, interval: str = DEFAULT_INTERVAL) -> dict[str, JsonValue]:
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
    models = {config.tree.identifier: config.tree for config in configs}
    position_cache = {identifier: walkforward_tree_positions(data, model, alt_symbols=ALT_SYMBOLS) for identifier, model in models.items()}
    all_sums: list[np.ndarray] = []
    all_squares: list[np.ndarray] = []
    bin_counts: tuple[int, ...] | None = None
    winner: ValidationRecord | None = None
    for index, config in enumerate(configs, start=1):
        positions = position_cache[config.tree.identifier].copy()
        positions[~masks[config.btc_volatility_cap]] = 0.0
        path = evaluate_path(data, positions, config.overlay)
        record = _validation_record(path, config, train_end=train_end, development_end=development_end, bars_per_year=spec.bars_per_year)
        if isfinite(record.score) and (winner is None or record.score > winner.score):
            winner = record
        sums, squares, counts = development_moments(path.stress_returns, development_end=development_end, bins=PBO_BINS)
        all_sums.append(sums)
        all_squares.append(squares)
        if bin_counts is None:
            bin_counts = counts
        elif bin_counts != counts:
            raise SelectionBiasError("tree configurations produced incompatible CSCV bin counts")
        if index % 25 == 0 or index == len(configs):
            print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("tree search generated no configurations")
    pbo = cscv_pbo_from_moments(np.asarray(all_sums), np.asarray(all_squares), bin_counts)
    locked: list[dict[str, JsonValue]] = []
    if winner is not None:
        positions = position_cache[winner.config.tree.identifier].copy()
        positions[~masks[winner.config.btc_volatility_cap]] = 0.0
        locked.append(
            _locked_result(
                winner,
                positions=positions,
                data=data,
                locked_start=development_end,
                bars_per_year=spec.bars_per_year,
                raw_trials=len(configs),
                pbo=pbo.pbo,
            )
        )
    return {
        "mode": "public-api-paper-only-no-live-orders",
        "selection_independent": False,
        "strategy": "strictly lagged ExtraTrees forecasts from momentum, volatility, volume surprise, and BTC-relative momentum",
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
            "algorithm_family": "extra_trees_nonlinear_forecast",
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
        "locked_test_results": locked,
        "eligible_for_mark_price_paper_validation": [result["id"] for result in locked if result["eligible_for_mark_price_paper_validation"]],
        "live_promotable": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    try:
        report = run(arguments.archive_root, interval=arguments.interval)
    except (ArchivePanelError, SelectionBiasError) as error:
        print(f"SEARCH_ERROR={error}")
        return 1
    arguments.report.write_text(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SEARCH_DONE report={arguments.report}")
    print(f"PBO={report['selection_bias']['pbo']['pbo']:.6f} WINNERS={len(report['selected_validation_winners'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
