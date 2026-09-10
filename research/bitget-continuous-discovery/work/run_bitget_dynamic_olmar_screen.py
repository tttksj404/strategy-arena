#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Run an explicitly non-promotable dynamic-listing OLMAR breadth screen."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Final

import numpy as np

from alt_search_engine import evaluate_path
from bitget_dynamic_archive_panel import (
    ArchivePanelError,
    load_verified_dynamic_interval_archive,
)
from bitget_olmar import walkforward_olmar_positions
from bitget_search_metrics import development_moments
from run_bitget_4h_pair_regime_search import (
    PBO_BINS,
    btc_regime_mask,
    spec_for_interval,
)
from run_bitget_olmar_search import (
    BTC_TREND_LIMIT,
    _locked_result,
    _search_configs,
    _validation_record,
)
from selection_bias_stats import SelectionBiasError, cscv_pbo_from_moments

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_liquid_alt_daily_20260825_v1")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_dynamic_daily_olmar_screen_report.json")
DYNAMIC_ASSETS: Final = (
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "HYPE",
    "ZEC",
    "DOGE",
    "TRUMP",
    "SUI",
    "ENA",
    "TUT",
    "VELVET",
    "ADA",
    "PEPE",
    "LINK",
    "ONDO",
    "NEAR",
    "PUMP",
    "AAVE",
    "TAO",
    "WLD",
    "PENGU",
    "BCH",
    "VIRTUAL",
    "UNI",
    "BNB",
    "LIT",
    "LTC",
    "XLM",
)


def run(archive_root: Path, *, minimum_listing_bars: int = 180) -> dict[str, object]:
    if minimum_listing_bars < 1:
        raise SelectionBiasError("dynamic listing screen requires a positive listing delay")
    spec = spec_for_interval("1D")
    data, metadata = load_verified_dynamic_interval_archive(
        archive_root,
        DYNAMIC_ASSETS,
        interval=spec.interval,
        minimum_listing_bars=minimum_listing_bars,
    )
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
    winner = None
    winner_positions: np.ndarray | None = None
    for completed, config in enumerate(configs, start=1):
        positions = walkforward_olmar_positions(data, config.model, alt_symbols=DYNAMIC_ASSETS[1:])
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
            raise SelectionBiasError("dynamic OLMAR configurations produced incompatible CSCV bin counts")
        if completed % 25 == 0 or completed == len(configs):
            print(f"SEARCH_PROGRESS configurations={completed}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("dynamic OLMAR screen generated no configurations")
    pbo = cscv_pbo_from_moments(np.asarray(all_sums), np.asarray(all_squares), bin_counts)
    locked: list[dict[str, object]] = []
    if winner is not None and winner_positions is not None:
        result = _locked_result(
            winner,
            positions=winner_positions,
            data=data,
            locked_start=development_end,
            bars_per_year=spec.bars_per_year,
            raw_trials=len(configs),
            pbo=pbo.pbo,
        )
        result["eligible_for_mark_price_paper_validation"] = False
        result["current_universe_survivorship_gate"] = False
        locked.append(result)
    return {
        "mode": "public-api-paper-only-no-live-orders",
        "selection_independent": False,
        "exploratory_only": True,
        "promotion_blocker": "the asset pool is selected from a current liquid-universe snapshot and excludes delisted historical contracts",
        "strategy": "strictly lagged market-neutral online moving-average reversion with causal post-listing eligibility",
        "data": {
            "interval": metadata.interval,
            "snapshot_ms": metadata.snapshot_ms,
            "reference_rows": metadata.reference_rows,
            "integrity_verified": metadata.integrity_verified,
            "minimum_listing_bars": metadata.minimum_listing_bars,
            "asset_count": len(DYNAMIC_ASSETS),
            "source_hashes": list(metadata.source_hashes),
            "train_rows": train_end,
            "development_rows": development_end,
            "locked_test_rows": len(data.dates) - development_end,
        },
        "predeclared_search": {"algorithm_family": "dynamic_market_neutral_olmar", "raw_configurations": len(configs), "max_leverage": 5.0},
        "selection_bias": {"method": "CSCV PBO on stress development returns; DSR on locked stress returns", "pbo": asdict(pbo)},
        "selected_validation_winners": [] if winner is None else [winner.config.identifier],
        "locked_test_results": locked,
        "eligible_for_mark_price_paper_validation": [],
        "live_promotable": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--minimum-listing-bars", type=int, default=180)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    try:
        report = run(arguments.archive_root, minimum_listing_bars=arguments.minimum_listing_bars)
    except (ArchivePanelError, SelectionBiasError) as error:
        print(f"SEARCH_ERROR={error}")
        return 1
    arguments.report.write_text(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SEARCH_DONE report={arguments.report}")
    print(f"PBO={report['selection_bias']['pbo']['pbo']:.6f} WINNERS={len(report['selected_validation_winners'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
