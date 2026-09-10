#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Run a public-data-only cross-sectional altcoin reversal experiment by interval."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from alt_search_engine import evaluate_path, test_passes, validation_score
from alt_search_models import FloatMatrix, MarketData, PathResult, RiskOverlay, SegmentMetrics
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import DSR_MIN, PBO_BINS, PBO_MAX, SearchSpec, btc_regime_mask, spec_for_interval
from selection_bias_stats import SelectionBiasError, cscv_pbo_from_moments, deflated_sharpe


DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_cross_sectional_reversal_report.json")
DEFAULT_INTERVAL: Final = "1H"


@dataclass(frozen=True, slots=True)
class CrossReversalConfig:
    lookback: int
    top_k: int
    rebalance_bars: int
    trend_bars: int
    volatility_bars: int
    btc_trend_limit: float
    btc_volatility_cap: float
    target_bar_volatility: float
    volatility_window: int
    minimum_quote_volume: float

    @property
    def identifier(self) -> str:
        return (
            f"reversal_lb={self.lookback}|top_k={self.top_k}|rebalance={self.rebalance_bars}"
            f"|btc_window={self.trend_bars}/{self.volatility_bars}|btc_trend_z<={self.btc_trend_limit:g}"
            f"|btc_vol<={self.btc_volatility_cap:g}|target_bar_vol={self.target_bar_volatility:g}"
            f"|risk_window={self.volatility_window}|minimum_quote_volume={self.minimum_quote_volume:g}|max_leverage=5"
        )

    @property
    def overlay(self) -> RiskOverlay:
        # The shared legacy field name represents target volatility per current bar here.
        return RiskOverlay(5.0, self.target_bar_volatility, self.volatility_window)


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    config: CrossReversalConfig
    validation: SegmentMetrics
    validation_stress: SegmentMetrics
    train_liquidated: bool
    validation_liquidated: bool
    score: float


def _unique_scaled_bars(spec: SearchSpec, base_bars: tuple[int, ...]) -> tuple[int, ...]:
    scale = spec.spread_lookbacks[0] / 30.0
    return tuple(dict.fromkeys(max(1, round(value * scale)) for value in base_bars))


def _all_configs(spec: SearchSpec) -> tuple[CrossReversalConfig, ...]:
    lookbacks = _unique_scaled_bars(spec, (1, 2, 3))
    rebalances = _unique_scaled_bars(spec, (1, 3))
    return tuple(
        CrossReversalConfig(
            lookback,
            top_k,
            rebalance_bars,
            spec.trend_bars,
            spec.volatility_bars,
            trend_limit,
            volatility_cap,
            target_volatility,
            risk_window,
            spec.minimum_quote_volume,
        )
        for lookback in lookbacks
        for top_k in (1, 2, 3)
        for rebalance_bars in rebalances
        for trend_limit in (1.0, 1.5)
        for volatility_cap in spec.volatility_caps
        for target_volatility in spec.target_bar_volatilities
        for risk_window in spec.risk_windows
    )


def _cross_reversal_signal(
    data: MarketData,
    *,
    lookback: int,
    top_k: int,
    rebalance_bars: int,
    minimum_quote_volume: float,
) -> FloatMatrix:
    """Long prior losers and short prior winners using only the close and volume before each return."""
    positions = np.zeros_like(data.close_price)
    alt_indices = np.asarray([data.symbols.index(symbol) for symbol in ALT_SYMBOLS], dtype=int)
    if top_k * 2 > alt_indices.size:
        raise SelectionBiasError("cross-sectional top-k needs distinct long and short assets")
    for index in range(lookback + 1, len(positions)):
        if index % rebalance_bars:
            positions[index] = positions[index - 1]
            active = np.abs(positions[index]) > 0.0
            if np.any(active & (data.quote_volume[index - 1] < minimum_quote_volume)):
                positions[index] = 0.0
            continue
        prior_returns = data.close_price[index - 1, alt_indices] / data.close_price[index - 1 - lookback, alt_indices] - 1.0
        liquid = data.quote_volume[index - 1, alt_indices] >= minimum_quote_volume
        valid = np.flatnonzero(liquid & np.isfinite(prior_returns))
        if valid.size < top_k * 2:
            continue
        ordered = valid[np.argsort(prior_returns[valid], kind="stable")]
        long_assets = alt_indices[ordered[:top_k]]
        short_assets = alt_indices[ordered[-top_k:]]
        positions[index, long_assets] = 0.5 / top_k
        positions[index, short_assets] = -0.5 / top_k
    return positions


def _metrics(returns: np.ndarray, *, bars_per_year: float) -> SegmentMetrics:
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        return SegmentMetrics(0.0, 0.0, 0.0, 0, 0)
    equity = np.cumprod(1.0 + values)
    drawdown = 1.0 - equity / np.maximum.accumulate(equity)
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
    sharpe = float(values.mean() / standard_deviation * sqrt(bars_per_year)) if standard_deviation > 0.0 else 0.0
    blocks = np.array_split(values, 4)
    positive_blocks = sum(bool(np.prod(1.0 + block) > 1.0) for block in blocks if block.size)
    return SegmentMetrics(float(equity[-1] - 1.0), float(drawdown.max()), sharpe, positive_blocks, values.size)


def _segment(path: PathResult, start: int, end: int, *, bars_per_year: float) -> tuple[SegmentMetrics, SegmentMetrics, bool]:
    liquidated = start <= path.first_liquidation_index < end
    return _metrics(path.returns[start:end], bars_per_year=bars_per_year), _metrics(path.stress_returns[start:end], bars_per_year=bars_per_year), liquidated


def _development_moments(returns: np.ndarray, *, development_end: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    values = np.asarray(returns[:development_end], dtype=float)
    if development_end <= 0 or development_end >= len(returns) or values.size < PBO_BINS * 2 or not np.isfinite(values).all():
        raise SelectionBiasError("development period cannot support CSCV moments")
    segments = tuple(np.asarray(segment, dtype=float) for segment in np.array_split(values, PBO_BINS))
    return (
        np.asarray([segment.sum() for segment in segments], dtype=float),
        np.asarray([(segment * segment).sum() for segment in segments], dtype=float),
        tuple(segment.size for segment in segments),
    )


def _validation_record(
    path: PathResult,
    config: CrossReversalConfig,
    *,
    train_end: int,
    development_end: int,
    bars_per_year: float,
) -> ValidationRecord:
    train, train_stress, train_liquidated = _segment(path, 0, train_end, bars_per_year=bars_per_year)
    validation, validation_stress, validation_liquidated = _segment(path, train_end, development_end, bars_per_year=bars_per_year)
    score = validation_score(validation, validation_stress, validation_liquidated)
    if not isfinite(validation_score(train, train_stress, train_liquidated)):
        score = float("-inf")
    return ValidationRecord(config, validation, validation_stress, train_liquidated, validation_liquidated, score)


def _metric_payload(metrics: SegmentMetrics) -> dict[str, float | int]:
    return {
        "net_return": metrics.net_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe_annualized": metrics.sharpe,
        "positive_blocks": metrics.positive_blocks,
        "observations": metrics.observations,
    }


def run(archive_root: Path, *, interval: str = DEFAULT_INTERVAL) -> dict[str, object]:
    spec = spec_for_interval(interval)
    data, metadata = load_verified_interval_archive(archive_root, ("BTC", *ALT_SYMBOLS), interval=spec.interval)
    configs = _all_configs(spec)
    train_end, development_end = int(len(data.dates) * 0.5), int(len(data.dates) * 0.75)
    btc = data.symbols.index("BTC")
    regime_masks = {
        (trend_limit, volatility_cap): btc_regime_mask(
            data.close_price[:, btc],
            trend_bars=spec.trend_bars,
            volatility_bars=spec.volatility_bars,
            trend_limit=trend_limit,
            volatility_cap=volatility_cap,
        )
        for trend_limit in (1.0, 1.5)
        for volatility_cap in spec.volatility_caps
    }
    all_sums: list[np.ndarray] = []
    all_squares: list[np.ndarray] = []
    bin_counts: tuple[int, ...] | None = None
    winner: ValidationRecord | None = None
    active_signal_key: tuple[int, int, int] | None = None
    active_signal: FloatMatrix | None = None
    for index, config in enumerate(configs, start=1):
        signal_key = config.lookback, config.top_k, config.rebalance_bars
        if signal_key != active_signal_key:
            active_signal = _cross_reversal_signal(
                data,
                lookback=config.lookback,
                top_k=config.top_k,
                rebalance_bars=config.rebalance_bars,
                minimum_quote_volume=spec.minimum_quote_volume,
            )
            active_signal_key = signal_key
        if active_signal is None:
            raise AssertionError("cross-sectional signal must be populated before evaluation")
        positions = active_signal.copy()
        positions[~regime_masks[(config.btc_trend_limit, config.btc_volatility_cap)]] = 0.0
        path = evaluate_path(data, positions, config.overlay)
        record = _validation_record(path, config, train_end=train_end, development_end=development_end, bars_per_year=spec.bars_per_year)
        if isfinite(record.score) and (winner is None or record.score > winner.score):
            winner = record
        sums, squares, counts = _development_moments(path.stress_returns, development_end=development_end)
        all_sums.append(sums)
        all_squares.append(squares)
        if bin_counts is None:
            bin_counts = counts
        elif bin_counts != counts:
            raise SelectionBiasError("reversal configurations produced incompatible CSCV bin counts")
        if index % 100 == 0 or index == len(configs):
            print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("reversal search generated no configurations")
    pbo = cscv_pbo_from_moments(np.asarray(all_sums), np.asarray(all_squares), bin_counts)
    locked_result: dict[str, object] | None = None
    if winner is not None:
        signal = _cross_reversal_signal(
            data,
            lookback=winner.config.lookback,
            top_k=winner.config.top_k,
            rebalance_bars=winner.config.rebalance_bars,
            minimum_quote_volume=spec.minimum_quote_volume,
        )
        positions = signal.copy()
        positions[~regime_masks[(winner.config.btc_trend_limit, winner.config.btc_volatility_cap)]] = 0.0
        path = evaluate_path(data, positions, winner.config.overlay)
        test, stress, liquidated = _segment(path, development_end, len(data.dates), bars_per_year=spec.bars_per_year)
        try:
            dsr = asdict(deflated_sharpe(path.stress_returns[development_end:], trials=len(configs)))
        except SelectionBiasError:
            dsr = None
        paper_pass = test_passes(test, stress, liquidated)
        dsr_pass = dsr is not None and float(dsr["probability"]) >= DSR_MIN
        locked_result = {
            "id": winner.config.identifier,
            "validation_score": winner.score,
            "test": _metric_payload(test),
            "test_stress": _metric_payload(stress),
            "test_liquidated": liquidated,
            "deflated_sharpe_stress": dsr,
            "paper_pass": paper_pass,
            "pbo_pass": pbo.pbo <= PBO_MAX,
            "dsr_pass": dsr_pass,
            "eligible_for_further_paper_validation": paper_pass and pbo.pbo <= PBO_MAX and dsr_pass,
            "live_promotable": False,
        }
    return {
        "mode": "public-api-paper-only-no-live-orders",
        "selection_independent": False,
        "strategy": f"{spec.interval} altcoin cross-sectional short-horizon reversal with lagged BTC volatility and trend gate",
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
            "raw_configurations": len(configs),
            "max_leverage": 5.0,
            "minimum_quote_volume_per_bar": spec.minimum_quote_volume,
        },
        "selection_bias": {
            "method": "CSCV PBO on stress returns from the train-plus-validation development period; DSR on locked stress returns",
            "pbo": asdict(pbo),
            "pbo_max": PBO_MAX,
            "dsr_min_probability": DSR_MIN,
        },
        "selected_validation_winner": None if winner is None else winner.config.identifier,
        "locked_test_result": locked_result,
        "eligible_for_further_paper_validation": bool(locked_result and locked_result["eligible_for_further_paper_validation"]),
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
