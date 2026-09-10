#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

# ─── How to run ───
# 1. Run: uv run run_bitget_short_horizon_max_return_search.py --interval 5m
# 2. This produces a research-only report and never submits an exchange order.
# ──────────────────

"""Search short-horizon altcoin factor variants for counterfactual return only."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Final

import numpy as np

from alt_search_engine import evaluate_counterfactual_path
from alt_search_models import (
    BASE_FEE,
    BASE_SLIPPAGE,
    STRESS_SLIPPAGE,
    PathResult,
    RiskOverlay,
    SegmentMetrics,
)
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_factor_signals import (
    FactorConfig,
    FactorExposure,
    FactorFamily,
    build_factor_position_variants,
)
from bitget_history_data import interval_duration_ms
from bitget_search_metrics import (
    development_moments,
    fixed_notional_metrics,
)
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import PBO_BINS, SearchSpec, spec_for_interval
from selection_bias_stats import (
    SelectionBiasError,
    cscv_pbo_from_moments,
    deflated_sharpe,
)

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_INTERVAL: Final = "5m"
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_5m_short_horizon_max_return_fixed_notional_report.json")
MAX_LEVERAGE: Final = 5.0
TARGET_BAR_VOLATILITY: Final = 1.0
VOLATILITY_WINDOW: Final = 6


@dataclass(frozen=True, slots=True)
class MaxReturnConfig:
    factor: FactorConfig
    exposure: FactorExposure
    overlay: RiskOverlay

    @property
    def identifier(self) -> str:
        return f"{self.factor.identifier}|exposure={self.exposure.value}|counterfactual_max_leverage={self.overlay.max_leverage:g}"


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    config: MaxReturnConfig
    validation: SegmentMetrics
    validation_stress: SegmentMetrics


def _bars(interval: str, minutes: tuple[int, ...]) -> tuple[int, ...]:
    duration = interval_duration_ms(interval)
    return tuple(dict.fromkeys(max(1, ceil(value * 60_000 / duration)) for value in minutes))


def _factor_configs(spec: SearchSpec) -> tuple[FactorConfig, ...]:
    lookbacks = tuple(dict.fromkeys(max(2, bars) for bars in _bars(spec.interval, (5, 15, 30))))
    rebalances = _bars(spec.interval, (5, 15))
    return tuple(
        FactorConfig(
            family=family,
            lookback=lookback,
            top_k=top_k,
            rebalance_bars=rebalance_bars,
            minimum_quote_volume=spec.minimum_quote_volume,
            beta_window=max(2, lookback * 2),
            volume_multiplier=1.25 if family is FactorFamily.VOLUME_TREND else 1.0,
        )
        for family in FactorFamily
        for lookback in lookbacks
        for top_k in (1, 2)
        for rebalance_bars in rebalances
    )


def _all_configs(spec: SearchSpec) -> tuple[MaxReturnConfig, ...]:
    return tuple(
        MaxReturnConfig(
            factor=factor,
            exposure=exposure,
            overlay=RiskOverlay(MAX_LEVERAGE, TARGET_BAR_VOLATILITY, VOLATILITY_WINDOW),
        )
        for factor in _factor_configs(spec)
        for exposure in FactorExposure
    )


def select_validation_winner(records: tuple[ValidationRecord, ...]) -> ValidationRecord:
    """Select highest validation fixed-notional PnL without test or liquidation filtering."""

    if not records:
        raise SelectionBiasError("short-horizon return search generated no validation records")
    return max(records, key=lambda record: record.validation.net_return)


def _development_statistics(path: PathResult, development_end: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    return development_moments(path.stress_returns, development_end=development_end, bins=PBO_BINS)


def _fixed_notional_payload(metrics: SegmentMetrics) -> dict[str, float | int]:
    return {
        "fixed_notional_net_pnl": metrics.net_return,
        "max_additive_drawdown": metrics.max_drawdown,
        "sharpe_annualized": metrics.sharpe,
        "positive_blocks": metrics.positive_blocks,
        "observations": metrics.observations,
    }


def _leader_payload(record: ValidationRecord) -> dict[str, object]:
    return {
        "id": record.config.identifier,
        "validation_fixed_notional": _fixed_notional_payload(record.validation),
        "validation_fixed_notional_stress": _fixed_notional_payload(record.validation_stress),
    }


def _test_payload(path: PathResult, *, development_end: int, bars_per_year: float, trials: int) -> dict[str, object]:
    test = fixed_notional_metrics(path.returns[development_end:], bars_per_year=bars_per_year)
    stress = fixed_notional_metrics(path.stress_returns[development_end:], bars_per_year=bars_per_year)
    try:
        dsr = asdict(deflated_sharpe(path.stress_returns[development_end:], trials=trials))
    except SelectionBiasError:
        dsr = None
    return {
        "test_fixed_notional": _fixed_notional_payload(test),
        "test_fixed_notional_stress": _fixed_notional_payload(stress),
        "all_path_liquidation_proxy_hits": path.liquidations,
        "first_liquidation_proxy_index": path.first_liquidation_index,
        "deflated_sharpe_stress_diagnostic": dsr,
        "live_promotable": False,
    }


def run(archive_root: Path, *, interval: str = DEFAULT_INTERVAL) -> dict[str, object]:
    spec = spec_for_interval(interval)
    data, metadata = load_verified_interval_archive(archive_root, ("BTC", *ALT_SYMBOLS), interval=spec.interval)
    configs = _all_configs(spec)
    factor_configs = _factor_configs(spec)
    train_end = int(len(data.dates) * 0.5)
    development_end = int(len(data.dates) * 0.75)
    records: list[ValidationRecord] = []
    sums: list[np.ndarray] = []
    squares: list[np.ndarray] = []
    counts: tuple[int, ...] | None = None
    family_leaders: dict[str, ValidationRecord] = {}
    index = 0
    for factor in factor_configs:
        variants = build_factor_position_variants(data, factor, alt_symbols=ALT_SYMBOLS)
        for exposure in FactorExposure:
            index += 1
            config = MaxReturnConfig(
                factor=factor,
                exposure=exposure,
                overlay=RiskOverlay(MAX_LEVERAGE, TARGET_BAR_VOLATILITY, VOLATILITY_WINDOW),
            )
            path = evaluate_counterfactual_path(data, variants.positions_for(exposure), config.overlay)
            record = ValidationRecord(
                config=config,
                validation=fixed_notional_metrics(
                    path.returns[train_end:development_end], bars_per_year=spec.bars_per_year
                ),
                validation_stress=fixed_notional_metrics(
                    path.stress_returns[train_end:development_end], bars_per_year=spec.bars_per_year
                ),
            )
            records.append(record)
            label = f"{config.factor.family.value}:{config.exposure.value}"
            previous = family_leaders.get(label)
            if previous is None or record.validation.net_return > previous.validation.net_return:
                family_leaders[label] = record
            current_sums, current_squares, current_counts = _development_statistics(path, development_end)
            sums.append(current_sums)
            squares.append(current_squares)
            if counts is None:
                counts = current_counts
            elif counts != current_counts:
                raise SelectionBiasError("short-horizon configurations produced incompatible CSCV bin counts")
            print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
    if counts is None:
        raise SelectionBiasError("short-horizon return search generated no CSCV statistics")
    winner = select_validation_winner(tuple(records))
    winner_positions = build_factor_position_variants(
        data, winner.config.factor, alt_symbols=ALT_SYMBOLS
    ).positions_for(winner.config.exposure)
    winner_path = evaluate_counterfactual_path(data, winner_positions, winner.config.overlay)
    pbo = cscv_pbo_from_moments(np.asarray(sums), np.asarray(squares), counts)
    locked = _test_payload(winner_path, development_end=development_end, bars_per_year=spec.bars_per_year, trials=len(configs))
    return {
        "mode": "public-data-paper-only-fixed-notional-counterfactual-no-liquidation-stop",
        "research_only": True,
        "live_promotable": False,
        "selection_independent": False,
        "strategy": f"{spec.interval} short-horizon altcoin factor fixed-notional PnL search without BTC regime or liquidation-stop selection gates",
        "data": {
            "interval": metadata.interval,
            "snapshot_ms": metadata.snapshot_ms,
            "common_rows": metadata.common_rows,
            "integrity_verified": metadata.integrity_verified,
            "source_hashes": list(metadata.source_hashes),
            "train_rows": train_end,
            "validation_rows": development_end - train_end,
            "locked_test_rows": len(data.dates) - development_end,
        },
        "predeclared_search": {
            "raw_configurations": len(configs),
            "factor_families": [family.value for family in FactorFamily],
            "exposures": [exposure.value for exposure in FactorExposure],
            "short_horizon_minutes": [5, 15, 30],
            "rebalance_minutes": [5, 15],
            "max_leverage": MAX_LEVERAGE,
        },
        "contract": {
            "objective": "highest_cost_adjusted_fixed_notional_validation_pnl",
            "test_evaluated_after_validation_selection": True,
            "account_equity_compounding": False,
            "fixed_notional_per_bar": True,
            "pnl_unit": "one initial-capital notional redeployed each bar; not an account return",
            "liquidation_stop_disabled": True,
            "liquidation_proxy_still_recorded": True,
            "not_an_executable_account_equity_path": True,
            "base_fee_per_side": BASE_FEE,
            "base_slippage_per_turnover": BASE_SLIPPAGE,
            "stress_slippage_per_turnover": STRESS_SLIPPAGE,
        },
        "selection_bias_diagnostic": {
            "method": "CSCV PBO on cost-adjusted stress development returns; DSR on selected locked stress return",
            "pbo": asdict(pbo),
        },
        "family_validation_leaders": {label: _leader_payload(record) for label, record in sorted(family_leaders.items())},
        "selected_validation_winner": _leader_payload(winner),
        "locked_test_result": {"id": winner.config.identifier, **locked},
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
