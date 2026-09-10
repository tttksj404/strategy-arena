#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Evaluate long-only altcoin trend strategies under a positive BTC risk regime."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from alt_search_engine import evaluate_path, test_passes, validation_score
from alt_search_models import PathResult, RiskOverlay
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_long_only_signals import LongOnlyConfig, LongOnlyStyle, btc_positive_regime_mask, long_only_positions
from bitget_search_metrics import development_moments, metric_payload, segment_metrics
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import DSR_MIN, PBO_BINS, PBO_MAX, SearchSpec, spec_for_interval
from selection_bias_stats import SelectionBiasError, cscv_pbo_from_moments, deflated_sharpe


DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_long_only_search_report.json")
DEFAULT_INTERVAL: Final = "4H"
BTC_TREND_FLOORS: Final = (0.0, 0.5)
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    signal: LongOnlyConfig
    minimum_btc_trend_z: float
    btc_volatility_cap: float
    target_bar_volatility: float
    risk_window: int

    @property
    def identifier(self) -> str:
        return (
            f"{self.signal.identifier}|btc_trend_z>={self.minimum_btc_trend_z:g}|btc_vol<={self.btc_volatility_cap:g}"
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


def _long_only_configs(spec: SearchSpec) -> tuple[LongOnlyConfig, ...]:
    rebalance = _scaled_bars(spec, (3,))[0]
    configs = tuple(
        LongOnlyConfig(style, _scaled_bars(spec, (short,))[0], _scaled_bars(spec, (long,))[0], top_k, rebalance, spec.minimum_quote_volume)
        for short, long in ((6, 30), (30, 90))
        for style in LongOnlyStyle
        for top_k in (1, 2, 3)
    )
    return tuple(dict.fromkeys(configs))


def _search_configs(spec: SearchSpec) -> tuple[SearchConfig, ...]:
    return tuple(
        SearchConfig(signal, minimum_trend_z, volatility_cap, target_volatility, spec.risk_windows[0])
        for signal in _long_only_configs(spec)
        for minimum_trend_z in BTC_TREND_FLOORS
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
    data,
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
        "style": record.config.signal.style.value,
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
        (minimum_trend_z, volatility_cap): btc_positive_regime_mask(
            data.close_price[:, btc_index],
            trend_bars=spec.trend_bars,
            volatility_bars=spec.volatility_bars,
            minimum_trend_z=minimum_trend_z,
            volatility_cap=volatility_cap,
        )
        for minimum_trend_z in BTC_TREND_FLOORS
        for volatility_cap in spec.volatility_caps
    }
    signals = {config.signal.identifier: config.signal for config in configs}
    position_cache = {identifier: long_only_positions(data, signal, alt_symbols=ALT_SYMBOLS) for identifier, signal in signals.items()}
    all_sums: list[np.ndarray] = []
    all_squares: list[np.ndarray] = []
    bin_counts: tuple[int, ...] | None = None
    winners: dict[LongOnlyStyle, ValidationRecord] = {}
    for index, config in enumerate(configs, start=1):
        positions = position_cache[config.signal.identifier].copy()
        positions[~masks[(config.minimum_btc_trend_z, config.btc_volatility_cap)]] = 0.0
        path = evaluate_path(data, positions, config.overlay)
        record = _validation_record(path, config, train_end=train_end, development_end=development_end, bars_per_year=spec.bars_per_year)
        current = winners.get(config.signal.style)
        if isfinite(record.score) and (current is None or record.score > current.score):
            winners[config.signal.style] = record
        sums, squares, counts = development_moments(path.stress_returns, development_end=development_end, bins=PBO_BINS)
        all_sums.append(sums)
        all_squares.append(squares)
        if bin_counts is None:
            bin_counts = counts
        elif bin_counts != counts:
            raise SelectionBiasError("long-only configurations produced incompatible CSCV bin counts")
        if index % 50 == 0 or index == len(configs):
            print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("long-only search generated no configurations")
    pbo = cscv_pbo_from_moments(np.asarray(all_sums), np.asarray(all_squares), bin_counts)
    locked: list[dict[str, JsonValue]] = []
    for style in sorted(winners, key=str):
        record = winners[style]
        positions = position_cache[record.config.signal.identifier].copy()
        positions[~masks[(record.config.minimum_btc_trend_z, record.config.btc_volatility_cap)]] = 0.0
        locked.append(
            _locked_result(
                record,
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
        "strategy": "long-only momentum, multi-horizon trend, and breakout during a lagged positive BTC regime",
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
            "styles": [style.value for style in LongOnlyStyle],
            "raw_configurations": len(configs),
            "max_leverage": 5.0,
            "btc_trend_floor_z": list(BTC_TREND_FLOORS),
            "minimum_quote_volume_per_bar": spec.minimum_quote_volume,
        },
        "selection_bias": {
            "method": "CSCV PBO on stress development returns; DSR on locked stress returns",
            "pbo": asdict(pbo),
            "pbo_max": PBO_MAX,
            "dsr_min_probability": DSR_MIN,
        },
        "selected_validation_winners": [record.config.identifier for _, record in sorted(winners.items(), key=lambda item: str(item[0]))],
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
