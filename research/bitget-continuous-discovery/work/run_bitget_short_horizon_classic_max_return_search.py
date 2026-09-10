#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Search diverse short-horizon altcoin classic signals by counterfactual fixed-notional PnL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Final

import numpy as np

from alt_search_engine import evaluate_counterfactual_path
from alt_search_models import (
    BASE_FEE,
    BASE_SLIPPAGE,
    STRESS_SLIPPAGE,
    Candidate,
    Family,
    MarketData,
    PathResult,
    RiskOverlay,
    SegmentMetrics,
)
from alt_search_signals import build_positions
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_history_data import interval_duration_ms
from bitget_search_metrics import development_moments, fixed_notional_metrics
from run_alt_strategy_search import PAIR_SET
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import PBO_BINS, SearchSpec, spec_for_interval
from selection_bias_stats import (
    SelectionBiasError,
    cscv_pbo_from_moments,
    deflated_sharpe,
)

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_INTERVAL: Final = "5m"
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_5m_short_horizon_classic_max_return_fixed_notional_report.json")
MAX_LEVERAGE: Final = 5.0
TARGET_BAR_VOLATILITY: Final = 1.0
VOLATILITY_WINDOW: Final = 6
CLASSIC_FAMILIES: Final = (
    Family.TIME_SERIES,
    Family.DONCHIAN,
    Family.EMA,
    Family.BOLLINGER,
    Family.RANGE,
    Family.PRICE_VOLUME,
    Family.PAIR,
)


@dataclass(frozen=True, slots=True)
class ClassicConfig:
    candidate: Candidate
    overlay: RiskOverlay

    @property
    def identifier(self) -> str:
        return f"{self.candidate.identifier}|counterfactual_max_leverage={self.overlay.max_leverage:g}"


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    config: ClassicConfig
    validation: SegmentMetrics
    validation_stress: SegmentMetrics


def _bars(interval: str, minutes: tuple[int, ...]) -> tuple[int, ...]:
    duration = interval_duration_ms(interval)
    return tuple(dict.fromkeys(max(1, ceil(value * 60_000 / duration)) for value in minutes))


def _short_candidates(spec: SearchSpec) -> tuple[Candidate, ...]:
    lookbacks = tuple(dict.fromkeys(max(2, bars) for bars in _bars(spec.interval, (5, 15, 30))))
    rebalances = _bars(spec.interval, (5, 15))
    ema_pairs = tuple((fast, slow) for fast in lookbacks for slow in lookbacks if fast < slow) or ((lookbacks[0], lookbacks[0] * 2),)
    base: list[Candidate] = []
    base.extend(Candidate(Family.TIME_SERIES, lookback, 0, 0, direction, None, 0.0) for lookback in lookbacks for direction in (1, -1))
    base.extend(Candidate(Family.DONCHIAN, lookback, 0, 0, direction, None, 0.0) for lookback in lookbacks for direction in (1, -1))
    base.extend(Candidate(Family.BOLLINGER, lookback, 0, top_k, -1, None, entry_z) for lookback in lookbacks for top_k in (1, 2) for entry_z in (1.0, 1.5) if entry_z <= sqrt(lookback - 1))
    for family in (Family.RANGE, Family.PRICE_VOLUME):
        base.extend(Candidate(family, lookback, 0, top_k, direction, None, 0.0) for lookback in lookbacks for top_k in (1, 2) for direction in (1, -1))
    base.extend(Candidate(Family.EMA, fast, slow, 0, direction, None, 0.0) for fast, slow in ema_pairs for direction in (1, -1))
    base.extend(Candidate(Family.PAIR, lookback, 0, 0, 0, pair, entry_z) for pair in PAIR_SET for lookback in lookbacks for entry_z in (1.0, 1.5) if entry_z <= sqrt(lookback - 1))
    return tuple(
        Candidate(
            candidate.family,
            candidate.lookback,
            candidate.secondary,
            candidate.top_k,
            candidate.direction,
            candidate.pair,
            candidate.entry_z,
            rebalance_bars,
            spec.minimum_quote_volume,
        )
        for candidate in base
        for rebalance_bars in rebalances
    )


def _all_configs(spec: SearchSpec) -> tuple[ClassicConfig, ...]:
    overlay = RiskOverlay(MAX_LEVERAGE, TARGET_BAR_VOLATILITY, VOLATILITY_WINDOW)
    return tuple(ClassicConfig(candidate, overlay) for candidate in _short_candidates(spec))


def _reinflated_positions(data: MarketData, candidate: Candidate) -> np.ndarray:
    try:
        indices = np.asarray([data.symbols.index(symbol) for symbol in ALT_SYMBOLS], dtype=int)
    except ValueError as error:
        raise SelectionBiasError("classic short-horizon search requires every requested alt in the panel") from error
    alt_data = MarketData(
        data.dates,
        ALT_SYMBOLS,
        data.open_price[:, indices],
        data.high_price[:, indices],
        data.low_price[:, indices],
        data.close_price[:, indices],
        data.quote_volume[:, indices],
    )
    positions = np.zeros_like(data.close_price)
    positions[:, indices] = build_positions(alt_data, candidate)
    return positions


def select_validation_winner(records: tuple[ValidationRecord, ...]) -> ValidationRecord:
    if not records:
        raise SelectionBiasError("classic short-horizon return search generated no validation records")
    return max(records, key=lambda record: record.validation.net_return)


def _fixed_payload(metrics: SegmentMetrics) -> dict[str, float | int]:
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
        "validation_fixed_notional": _fixed_payload(record.validation),
        "validation_fixed_notional_stress": _fixed_payload(record.validation_stress),
    }


def _locked_payload(path: PathResult, *, start: int, bars_per_year: float, trials: int) -> dict[str, object]:
    test = fixed_notional_metrics(path.returns[start:], bars_per_year=bars_per_year)
    stress = fixed_notional_metrics(path.stress_returns[start:], bars_per_year=bars_per_year)
    try:
        dsr = asdict(deflated_sharpe(path.stress_returns[start:], trials=trials))
    except SelectionBiasError:
        dsr = None
    return {
        "test_fixed_notional": _fixed_payload(test),
        "test_fixed_notional_stress": _fixed_payload(stress),
        "all_path_liquidation_proxy_hits": path.liquidations,
        "first_liquidation_proxy_index": path.first_liquidation_index,
        "deflated_sharpe_stress_diagnostic": dsr,
        "live_promotable": False,
    }


def run(archive_root: Path, *, interval: str = DEFAULT_INTERVAL) -> dict[str, object]:
    spec = spec_for_interval(interval)
    data, metadata = load_verified_interval_archive(archive_root, ("BTC", *ALT_SYMBOLS), interval=spec.interval)
    configs = _all_configs(spec)
    train_end, development_end = int(len(data.dates) * 0.5), int(len(data.dates) * 0.75)
    records: list[ValidationRecord] = []
    sums: list[np.ndarray] = []
    squares: list[np.ndarray] = []
    counts: tuple[int, ...] | None = None
    family_leaders: dict[str, ValidationRecord] = {}
    for index, config in enumerate(configs, start=1):
        path = evaluate_counterfactual_path(data, _reinflated_positions(data, config.candidate), config.overlay)
        record = ValidationRecord(
            config,
            fixed_notional_metrics(path.returns[train_end:development_end], bars_per_year=spec.bars_per_year),
            fixed_notional_metrics(path.stress_returns[train_end:development_end], bars_per_year=spec.bars_per_year),
        )
        records.append(record)
        leader = family_leaders.get(config.candidate.family.value)
        if leader is None or record.validation.net_return > leader.validation.net_return:
            family_leaders[config.candidate.family.value] = record
        current_sums, current_squares, current_counts = development_moments(path.stress_returns, development_end=development_end, bins=PBO_BINS)
        sums.append(current_sums)
        squares.append(current_squares)
        if counts is None:
            counts = current_counts
        elif counts != current_counts:
            raise SelectionBiasError("classic short-horizon configurations produced incompatible CSCV bin counts")
        print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
    if counts is None:
        raise SelectionBiasError("classic short-horizon return search generated no CSCV statistics")
    winner = select_validation_winner(tuple(records))
    winner_path = evaluate_counterfactual_path(data, _reinflated_positions(data, winner.config.candidate), winner.config.overlay)
    pbo = cscv_pbo_from_moments(np.asarray(sums), np.asarray(squares), counts)
    return {
        "mode": "public-data-paper-only-classic-fixed-notional-counterfactual-no-liquidation-stop",
        "research_only": True,
        "live_promotable": False,
        "selection_independent": False,
        "strategy": f"{spec.interval} short-horizon classic altcoin PnL search without BTC regime or liquidation-stop selection gates",
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
            "families": [family.value for family in CLASSIC_FAMILIES],
            "short_horizon_minutes": [5, 15, 30],
            "rebalance_minutes": [5, 15],
            "alt_pair_pool": len(PAIR_SET),
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
            "method": "CSCV PBO on cost-adjusted stress development returns; DSR on selected locked stress PnL",
            "pbo": asdict(pbo),
        },
        "family_validation_leaders": {family: _leader_payload(record) for family, record in sorted(family_leaders.items())},
        "selected_validation_winner": _leader_payload(winner),
        "locked_test_result": {"id": winner.config.identifier, **_locked_payload(winner_path, start=development_end, bars_per_year=spec.bars_per_year, trials=len(configs))},
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
