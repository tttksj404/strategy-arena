#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Test slower long-only trend holdings to separate raw alpha from high-turnover costs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from alt_search_engine import evaluate_path
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_long_only_signals import LongOnlyConfig, LongOnlyStyle, btc_positive_regime_mask, long_only_positions
from bitget_search_metrics import development_moments
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import PBO_BINS, SearchSpec, spec_for_interval
from run_bitget_long_only_search import BTC_TREND_FLOORS, SearchConfig, ValidationRecord, _locked_result, _validation_record
from selection_bias_stats import SelectionBiasError, cscv_pbo_from_moments


DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_low_turnover_long_only_search_report.json")
DEFAULT_INTERVAL: Final = "4H"
BASE_REBALANCE_BARS: Final = (3, 6, 18, 42)
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def _scaled_bars(spec: SearchSpec, base_bars: tuple[int, ...]) -> tuple[int, ...]:
    scale = spec.spread_lookbacks[0] / 30.0
    return tuple(dict.fromkeys(max(2, round(value * scale)) for value in base_bars))


def _long_only_configs(spec: SearchSpec) -> tuple[LongOnlyConfig, ...]:
    configs = tuple(
        LongOnlyConfig(style, _scaled_bars(spec, (short,))[0], _scaled_bars(spec, (long,))[0], top_k, rebalance, spec.minimum_quote_volume)
        for short, long in ((6, 30), (30, 90))
        for style in LongOnlyStyle
        for top_k in (1, 2, 3)
        for rebalance in _scaled_bars(spec, BASE_REBALANCE_BARS)
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
            raise SelectionBiasError("low-turnover long-only configurations produced incompatible CSCV bin counts")
        if index % 100 == 0 or index == len(configs):
            print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("low-turnover long-only search generated no configurations")
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
        "strategy": "long-only momentum, multi-horizon trend, and breakout with 12-hour to 7-day holds during a lagged positive BTC regime",
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
            "base_rebalance_bars": list(BASE_REBALANCE_BARS),
            "max_leverage": 5.0,
            "btc_trend_floor_z": list(BTC_TREND_FLOORS),
        },
        "selection_bias": {
            "method": "CSCV PBO on stress development returns; DSR on locked stress returns",
            "pbo": asdict(pbo),
            "pbo_max": 0.05,
            "dsr_min_probability": 0.95,
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
