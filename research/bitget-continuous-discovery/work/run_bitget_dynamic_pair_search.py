#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Run a rolling-hedge, half-life-filtered alt-pair paper search on Bitget candles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from alt_search_engine import evaluate_path, test_passes, validation_score
from alt_search_models import MarketData, PathResult, RiskOverlay
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_dynamic_pair_signals import DynamicPairConfig, dynamic_pair_positions
from bitget_search_metrics import development_moments, metric_payload, segment_metrics
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import DSR_MIN, PBO_BINS, PBO_MAX, SearchSpec, btc_regime_mask, spec_for_interval
from selection_bias_stats import SelectionBiasError, cscv_pbo_from_moments, deflated_sharpe


DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_dynamic_pair_search_report.json")
DEFAULT_INTERVAL: Final = "4H"
BTC_TREND_LIMIT: Final = 1.5
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class PairSearchConfig:
    pair_config: DynamicPairConfig
    btc_volatility_cap: float
    target_bar_volatility: float
    risk_window: int

    @property
    def identifier(self) -> str:
        return (
            f"{self.pair_config.identifier}|btc_trend_z<={BTC_TREND_LIMIT:g}|btc_vol<={self.btc_volatility_cap:g}"
            f"|target_bar_vol={self.target_bar_volatility:g}|risk_window={self.risk_window}|max_leverage=5"
        )

    @property
    def overlay(self) -> RiskOverlay:
        return RiskOverlay(5.0, self.target_bar_volatility, self.risk_window)


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    config: PairSearchConfig
    score: float


def _scaled_bars(spec: SearchSpec, base: tuple[int, ...]) -> tuple[int, ...]:
    scale = spec.spread_lookbacks[0] / 30.0
    return tuple(dict.fromkeys(max(3, round(value * scale)) for value in base))


def _configs(spec: SearchSpec) -> tuple[PairSearchConfig, ...]:
    beta_windows = _scaled_bars(spec, (60, 120))
    return tuple(
        PairSearchConfig(
            DynamicPairConfig(pair, beta_window, max(3, beta_window // 2), entry_z, beta_window / 4.0, spec.minimum_quote_volume),
            volatility_cap,
            spec.target_bar_volatilities[0],
            spec.risk_windows[0],
        )
        for pair in combinations(ALT_SYMBOLS, 2)
        for beta_window in beta_windows
        for entry_z in (1.5, 2.0)
        for volatility_cap in spec.volatility_caps
    )


def _validation_record(path: PathResult, config: PairSearchConfig, *, train_end: int, development_end: int, bars_per_year: float) -> ValidationRecord:
    train, train_stress, train_liquidated = segment_metrics(path, 0, train_end, bars_per_year=bars_per_year)
    validation, validation_stress, validation_liquidated = segment_metrics(path, train_end, development_end, bars_per_year=bars_per_year)
    score = validation_score(validation, validation_stress, validation_liquidated)
    if not isfinite(validation_score(train, train_stress, train_liquidated)):
        score = float("-inf")
    return ValidationRecord(config, score)


def _locked_result(record: ValidationRecord, *, positions: np.ndarray, data: MarketData, locked_start: int, bars_per_year: float, raw_trials: int, pbo: float) -> dict[str, JsonValue]:
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
    configs = _configs(spec)
    train_end, development_end = int(len(data.dates) * 0.5), int(len(data.dates) * 0.75)
    btc = data.symbols.index("BTC")
    masks = {
        cap: btc_regime_mask(data.close_price[:, btc], trend_bars=spec.trend_bars, volatility_bars=spec.volatility_bars, trend_limit=BTC_TREND_LIMIT, volatility_cap=cap)
        for cap in spec.volatility_caps
    }
    signals = {config.pair_config.identifier: config.pair_config for config in configs}
    position_cache: dict[str, np.ndarray] = {}
    for index, (identifier, config) in enumerate(signals.items(), start=1):
        position_cache[identifier] = dynamic_pair_positions(data, config)
        if index % 100 == 0 or index == len(signals):
            print(f"SIGNAL_PROGRESS configurations={index}/{len(signals)}", flush=True)
    all_sums: list[np.ndarray] = []
    all_squares: list[np.ndarray] = []
    bin_counts: tuple[int, ...] | None = None
    winner: ValidationRecord | None = None
    for index, config in enumerate(configs, start=1):
        positions = position_cache[config.pair_config.identifier].copy()
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
            raise SelectionBiasError("dynamic-pair configurations produced incompatible CSCV bin counts")
        if index % 100 == 0 or index == len(configs):
            print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("dynamic-pair search generated no configurations")
    pbo = cscv_pbo_from_moments(np.asarray(all_sums), np.asarray(all_squares), bin_counts)
    locked: dict[str, JsonValue] | None = None
    if winner is not None:
        positions = position_cache[winner.config.pair_config.identifier].copy()
        positions[~masks[winner.config.btc_volatility_cap]] = 0.0
        locked = _locked_result(winner, positions=positions, data=data, locked_start=development_end, bars_per_year=spec.bars_per_year, raw_trials=len(configs), pbo=pbo.pbo)
    return {
        "mode": "public-api-paper-only-no-live-orders",
        "selection_independent": False,
        "strategy": "rolling OLS hedge ratio, residual z-score, and half-life-filtered alt-pair mean reversion",
        "liquidation_evidence": "market-candle proxy only; full mark-price history remains required before any promotion",
        "data": {"interval": metadata.interval, "snapshot_ms": metadata.snapshot_ms, "common_rows": metadata.common_rows, "integrity_verified": metadata.integrity_verified, "source_hashes": list(metadata.source_hashes)},
        "predeclared_search": {"raw_configurations": len(configs), "pair_pool": len(set(config.pair_config.pair for config in configs)), "max_leverage": 5.0, "btc_trend_limit": BTC_TREND_LIMIT},
        "selection_bias": {"method": "CSCV PBO on stress development returns; DSR on locked stress returns", "pbo": asdict(pbo), "pbo_max": PBO_MAX, "dsr_min_probability": DSR_MIN},
        "selected_validation_winner": None if winner is None else winner.config.identifier,
        "locked_test_result": locked,
        "eligible_for_mark_price_paper_validation": bool(locked and locked["eligible_for_mark_price_paper_validation"]),
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
    print(f"PBO={report['selection_bias']['pbo']['pbo']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
