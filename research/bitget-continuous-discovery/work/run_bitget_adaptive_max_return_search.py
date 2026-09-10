#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///
# pyright: reportImplicitRelativeImport=false

"""Search interval-aware adaptive altcoin portfolios on public Bitget candles."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Final, cast

import numpy as np

from alt_search_engine import evaluate_counterfactual_path
from alt_search_models import BASE_FEE, BASE_SLIPPAGE, STRESS_SLIPPAGE
from bitget_adaptive_counterfactual import (
    adaptive_configs,
    family,
    identifier,
    overlay,
    positions_for,
)
from bitget_adaptive_report import (
    BASE_ROUND_TRIP_COST,
    STRESS_ROUND_TRIP_COST,
    JsonValue,
    _candidate_payload,
    _liquidation_payload,
    _metric_net_return,
    _metrics,
    _survival_rows,
    candidate_manifest,
    split_payload,
)
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_directional_change_payloads import dsr_payload, number_payload, pbo_payload
from bitget_history_data import interval_duration_ms
from bitget_search_metrics import (
    development_moments,
    fixed_notional_metrics,
    metric_payload,
)
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import (
    DSR_MIN,
    PBO_BINS,
    PBO_MAX,
    spec_for_interval,
)
from selection_bias_stats import (
    SelectionBiasError,
    cscv_pbo_from_moments,
    deflated_sharpe,
)

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_adaptive_max_return_search_report.json")
DEFAULT_INTERVAL: Final = "5m"


class AdaptiveSearchError(RuntimeError):
    """Raised when adaptive-search inputs violate the runner contract."""


def _split_bounds(rows: int) -> tuple[int, int]:
    if rows < 4:
        raise AdaptiveSearchError("at least four rows are required for a 50/25/25 split")
    train_end = rows // 2
    development_end = train_end + (rows - train_end) // 2
    if train_end < 1 or development_end <= train_end or development_end >= rows:
        raise AdaptiveSearchError("chronological split cannot produce three non-empty segments")
    return train_end, development_end


def select_validation_winner(scores: Mapping[str, float], eligible: Mapping[str, bool]) -> str | None:
    """Select exactly one highest stress fixed-notional validation-PnL candidate."""
    if set(scores) != set(eligible):
        raise AdaptiveSearchError("scores and eligible mappings require matching identifiers")
    candidates = [key for key, value in scores.items() if eligible[key] and isfinite(value)]
    return max(candidates, key=lambda key: scores[key]) if candidates else None


def run(archive_root: Path = DEFAULT_ARCHIVE, *, interval: str = DEFAULT_INTERVAL) -> dict[str, JsonValue]:
    spec = spec_for_interval(interval)
    bar_minutes = interval_duration_ms(spec.interval) // 60_000
    data, metadata = load_verified_interval_archive(archive_root, ("BTC", *ALT_SYMBOLS), interval=spec.interval)
    train_end, development_end = _split_bounds(len(data.dates))
    configs = adaptive_configs(spec)
    if len(configs) != 48 or len({identifier(config) for config in configs}) != 48:
        raise SelectionBiasError("adaptive search must contain 48 unique no-BTC-gate configurations")
    positions_cache: dict[str, np.ndarray] = {}
    model_keys_seen: set[str] = set()
    sums: list[np.ndarray] = []
    squares: list[np.ndarray] = []
    counts: tuple[int, ...] | None = None
    scores: dict[str, float] = {}
    eligible: dict[str, bool] = {}
    records: list[dict[str, JsonValue]] = []
    configs_by_id = {identifier(config): config for config in configs}
    for index, config in enumerate(configs, start=1):
        config_id = identifier(config)
        model_key = f"{family(config).value}|{config.model.identifier}"
        model_keys_seen.add(model_key)
        if model_key not in positions_cache:
            positions_cache[model_key] = positions_for(data, config, alt_symbols=ALT_SYMBOLS)
        path = evaluate_counterfactual_path(data, positions_cache[model_key], overlay(config))
        record = _candidate_payload(
            config,
            data,
            positions_cache[model_key],
            path,
            train_end,
            development_end,
            spec.bars_per_year,
        )
        records.append(record)
        score = fixed_notional_metrics(path.stress_returns[train_end:development_end], bars_per_year=spec.bars_per_year).net_return
        scores[config_id] = score
        eligible[config_id] = isfinite(score)
        candidate_sums, candidate_squares, candidate_counts = development_moments(
            path.stress_returns, development_end=development_end, bins=PBO_BINS
        )
        sums.append(candidate_sums)
        squares.append(candidate_squares)
        if counts is None:
            counts = candidate_counts
        elif counts != candidate_counts:
            raise SelectionBiasError("adaptive candidates produced incompatible CSCV bins")
        if index % 8 == 0 or index == len(configs):
            print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
        next_key = None
        if index < len(configs):
            next_config = configs[index]
            next_key = f"{family(next_config).value}|{next_config.model.identifier}"
        if next_key != model_key:
            del positions_cache[model_key]
    winner_id = select_validation_winner(scores, eligible)
    if winner_id is None:
        raise SelectionBiasError("adaptive search produced no finite validation winner")
    if counts is None:
        raise SelectionBiasError("adaptive search generated no candidates")
    pbo = cscv_pbo_from_moments(np.asarray(sums), np.asarray(squares), counts)
    winner_config = configs_by_id[winner_id]
    winner_positions = positions_for(data, winner_config, alt_symbols=ALT_SYMBOLS)
    winner_path = evaluate_counterfactual_path(data, winner_positions, overlay(winner_config))
    test_base = _metrics(winner_path.returns, development_end, len(data.dates), bars_per_year=spec.bars_per_year)
    test_stress = _metrics(winner_path.stress_returns, development_end, len(data.dates), bars_per_year=spec.bars_per_year)
    dsr_probability: float | None = None
    try:
        dsr_result = deflated_sharpe(winner_path.stress_returns[development_end:], trials=len(configs))
        dsr = dsr_payload(dsr_result)
        dsr_probability = dsr_result.probability
    except SelectionBiasError:
        dsr = None
    leaders: dict[str, JsonValue] = {
        group: max(
            (row for row in records if row["family"] == group),
            key=lambda row: _metric_net_return(row, "validation_stress_fixed_notional"),
        )
        for group in sorted({family(config).value for config in configs})
    }
    return {
        "schema_version": "adaptive-max-return-v2",
        "mode": "public-api-paper-only-no-live-orders",
        "research_only": True,
        "live_promotable": False,
        "strategy": f"{spec.interval} adaptive OLMAR plus directional-change fixed-notional maximum-return search",
        "contract": {
            "report_schema_version": "adaptive-max-return-v2",
            "selection": "single highest stress fixed-notional validation PnL; no locked-test input",
            "selection_objective": "highest_stress_fixed_notional_validation_pnl",
            "fixed_notional_is_not_account_return": True,
            "liquidation_evidence": "market-candle liquidation proxy only",
            "mark_price_status": "pending full mark-price archive and margin specification",
            "btc_regime_gate": "disabled",
            "max_leverage_cap": 5.0,
            "fee_slippage": {
                "base_per_turnover": BASE_FEE + BASE_SLIPPAGE,
                "stress_per_turnover": BASE_FEE + STRESS_SLIPPAGE,
                "survival_base_round_trip": BASE_ROUND_TRIP_COST,
                "survival_stress_round_trip": STRESS_ROUND_TRIP_COST,
            },
            "survival_execution": "next-open entry and next-open signal exit; adverse path includes OHLC and exit-open gap",
            "liquidation_stop_disabled": True,
            "account_equity_path": "not_executable_counterfactual",
        },
        "data": {
            "interval": metadata.interval,
            "snapshot_ms": metadata.snapshot_ms,
            "common_rows": metadata.common_rows,
            "integrity_verified": metadata.integrity_verified,
            "source_hashes": list(metadata.source_hashes),
            "train_rows": train_end,
            "validation_rows": development_end - train_end,
            "locked_test_rows": len(data.dates) - development_end,
            "split": split_payload(data, train_end, development_end),
        },
        "predeclared_search": {
            "raw_configurations": len(configs),
            "unique_models": len(model_keys_seen),
            "max_leverage": 5.0,
            "candidate_manifest": candidate_manifest(records, scores, eligible),
        },
        "selection_bias": {
            "method": "CSCV PBO on development stress returns; DSR on locked test stress returns",
            "multiplicity_scope": "adaptive_run_local_48",
            "global_search_scope": {
                "prior_search_configurations": 139536,
                "global_dsr_computed": False,
                "note": "This PBO/DSR is not a global DSR for the prior 139,536+ configuration search.",
            },
            "pbo": pbo_payload(pbo),
            "pbo_max": PBO_MAX,
            "dsr_min_probability": DSR_MIN,
        },
        "selected_validation_winner": winner_id,
        "family_validation_leaders": leaders,
        "selected_validation_winner_record": _candidate_payload(
            winner_config,
            data,
            winner_positions,
            winner_path,
            train_end,
            development_end,
            spec.bars_per_year,
        ),
        "locked_test": {
            "id": winner_id,
            "fixed_notional": {
                "base": number_payload(metric_payload(test_base)),
                "stress": number_payload(metric_payload(test_stress)),
            },
            "liquidation_proxy": _liquidation_payload(winner_path, development_end, len(data.dates)),
            "deflated_sharpe_stress": dsr,
            "pbo_pass": pbo.pbo <= PBO_MAX,
            "dsr_pass": dsr_probability is not None and dsr_probability >= DSR_MIN,
            "survival": {
                "validation": _survival_rows(data, winner_positions, train_end, development_end, bar_minutes=bar_minutes),
                "test": _survival_rows(data, winner_positions, development_end, len(data.dates), bar_minutes=bar_minutes),
            },
        },
        "live_promotable_ids": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    _ = parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    _ = parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    archive_root = cast(Path, arguments.archive_root)
    interval = cast(str, arguments.interval)
    report_path = cast(Path, arguments.report)
    try:
        report = run(archive_root, interval=interval)
    except (ArchivePanelError, SelectionBiasError, AdaptiveSearchError) as error:
        print(f"SEARCH_ERROR={error}")
        return 1
    report_path.write_text(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SEARCH_DONE report={arguments.report} WINNER={report['selected_validation_winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
