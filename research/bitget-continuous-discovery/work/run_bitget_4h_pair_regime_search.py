#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

"""Run a separate, public-data-only 4H alt-pair mean-reversion experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from math import isfinite, sqrt
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from alt_search_engine import evaluate_path, test_passes, validation_score
from alt_search_models import FloatMatrix, MarketData, PathResult, RiskOverlay, SegmentMetrics
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_history_data import interval_duration_ms
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from selection_bias_stats import SelectionBiasError, cscv_pbo_from_moments, deflated_sharpe


DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_4h_pair_regime_search_report.json")
INTERVAL: Final = "4H"
BASE_INTERVAL_MS: Final = interval_duration_ms(INTERVAL)
PBO_BINS: Final = 8
PBO_MAX: Final = 0.05
DSR_MIN: Final = 0.95
BASE_MINIMUM_QUOTE_VOLUME: Final = 1_000_000.0


@dataclass(frozen=True, slots=True)
class SearchSpec:
    interval: str
    bars_per_year: float
    spread_lookbacks: tuple[int, ...]
    trend_bars: int
    volatility_bars: int
    volatility_caps: tuple[float, ...]
    target_bar_volatilities: tuple[float, ...]
    risk_windows: tuple[int, ...]
    minimum_quote_volume: float


def spec_for_interval(interval: str) -> SearchSpec:
    """Scale the same 4H economic horizons and per-bar risk caps to a native interval."""
    duration = interval_duration_ms(interval)
    horizon_scale = BASE_INTERVAL_MS / duration
    volatility_scale = sqrt(duration / BASE_INTERVAL_MS)
    bars_per_year = 365.25 * 24.0 * 60.0 * 60.0 * 1000.0 / duration
    scale_bars = lambda bars: max(2, round(bars * horizon_scale))
    return SearchSpec(
        interval=interval,
        bars_per_year=bars_per_year,
        spread_lookbacks=tuple(scale_bars(bars) for bars in (30, 60, 120)),
        trend_bars=scale_bars(30),
        volatility_bars=scale_bars(30),
        volatility_caps=tuple(value * volatility_scale for value in (0.015, 0.025)),
        target_bar_volatilities=tuple(value * volatility_scale for value in (0.0025, 0.005)),
        risk_windows=tuple(scale_bars(bars) for bars in (30, 60)),
        minimum_quote_volume=BASE_MINIMUM_QUOTE_VOLUME * duration / BASE_INTERVAL_MS,
    )


@dataclass(frozen=True, slots=True)
class PairRegimeConfig:
    pair: tuple[str, str]
    spread_lookback: int
    entry_z: float
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
            f"pair={self.pair[0]}-{self.pair[1]}|spread_lb={self.spread_lookback}|entry_z={self.entry_z:g}"
            f"|btc_window={self.trend_bars}/{self.volatility_bars}"
            f"|btc_trend_z<={self.btc_trend_limit:g}|btc_vol<={self.btc_volatility_cap:g}"
            f"|target_bar_vol={self.target_bar_volatility:g}|risk_window={self.volatility_window}"
            f"|minimum_quote_volume={self.minimum_quote_volume:g}|max_leverage=5"
        )

    @property
    def overlay(self) -> RiskOverlay:
        # The shared engine's field name is historical; this is deliberately a per-4H-bar target.
        return RiskOverlay(5.0, self.target_bar_volatility, self.volatility_window)


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    config: PairRegimeConfig
    validation: SegmentMetrics
    validation_stress: SegmentMetrics
    train_liquidated: bool
    validation_liquidated: bool
    score: float


def _all_configs(spec: SearchSpec) -> tuple[PairRegimeConfig, ...]:
    return tuple(
        PairRegimeConfig(
            pair,
            spread_lookback,
            entry_z,
            spec.trend_bars,
            spec.volatility_bars,
            trend_limit,
            volatility_cap,
            target_volatility,
            risk_window,
            spec.minimum_quote_volume,
        )
        for pair in combinations(ALT_SYMBOLS, 2)
        for spread_lookback in spec.spread_lookbacks
        for entry_z in (1.5, 2.0)
        for trend_limit in (1.0, 1.5)
        for volatility_cap in spec.volatility_caps
        for target_volatility in spec.target_bar_volatilities
        for risk_window in spec.risk_windows
    )


def btc_regime_mask(
    close_price: np.ndarray,
    *,
    trend_bars: int,
    volatility_bars: int,
    trend_limit: float,
    volatility_cap: float,
) -> np.ndarray:
    """Use only closed BTC observations strictly prior to each candidate 4H return."""
    prices = np.asarray(close_price, dtype=float)
    if prices.ndim != 1 or prices.size < max(trend_bars, volatility_bars) + 2 or not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise SelectionBiasError("BTC regime input is not a finite positive price path")
    log_returns = np.zeros_like(prices)
    log_returns[1:] = np.log(prices[1:] / prices[:-1])
    mask = np.zeros(prices.size, dtype=bool)
    first = max(trend_bars, volatility_bars) + 1
    indices = np.arange(first, prices.size)
    prefix = np.concatenate((np.asarray([0.0]), np.cumsum(log_returns)))
    square_prefix = np.concatenate((np.asarray([0.0]), np.cumsum(log_returns * log_returns)))
    totals = prefix[indices] - prefix[indices - volatility_bars]
    square_totals = square_prefix[indices] - square_prefix[indices - volatility_bars]
    variance = np.maximum((square_totals - totals * totals / volatility_bars) / (volatility_bars - 1), 0.0)
    volatility = np.sqrt(variance)
    trend = np.log(prices[indices - 1] / prices[indices - 1 - trend_bars])
    trend_z = np.full_like(volatility, np.inf)
    positive_volatility = volatility > 0.0
    trend_z[positive_volatility] = np.abs(trend[positive_volatility] / (volatility[positive_volatility] * sqrt(trend_bars)))
    mask[indices] = (volatility <= volatility_cap) & (trend_z <= trend_limit)
    return mask


def _pair_signal(
    data: MarketData,
    *,
    pair: tuple[str, str],
    spread_lookback: int,
    entry_z: float,
    minimum_quote_volume: float,
) -> FloatMatrix:
    """Build lagged dollar-neutral pair signals before applying a regime gate."""
    positions = np.zeros_like(data.close_price)
    base, quote = (data.symbols.index(symbol) for symbol in pair)
    spread = np.log(data.close_price[:, base]) - np.log(data.close_price[:, quote])
    first = spread_lookback + 1
    indices = np.arange(first, len(positions))
    prefix = np.concatenate((np.asarray([0.0]), np.cumsum(spread)))
    square_prefix = np.concatenate((np.asarray([0.0]), np.cumsum(spread * spread)))
    totals = prefix[indices] - prefix[indices - spread_lookback]
    square_totals = square_prefix[indices] - square_prefix[indices - spread_lookback]
    mean = totals / spread_lookback
    variance = np.maximum((square_totals - totals * totals / spread_lookback) / (spread_lookback - 1), 0.0)
    standard_deviation = np.sqrt(variance)
    zscore = np.zeros_like(standard_deviation)
    nonzero = standard_deviation > 0.0
    zscore[nonzero] = (spread[indices[nonzero] - 1] - mean[nonzero]) / standard_deviation[nonzero]
    liquid = np.all(data.quote_volume[indices - 1][:, (base, quote)] >= minimum_quote_volume, axis=1)
    short = liquid & nonzero & (zscore >= entry_z)
    long = liquid & nonzero & (zscore <= -entry_z)
    positions[indices[short], base], positions[indices[short], quote] = -0.5, 0.5
    positions[indices[long], base], positions[indices[long], quote] = 0.5, -0.5
    return positions


def pair_positions(data: MarketData, config: PairRegimeConfig) -> FloatMatrix:
    """Build lagged pair positions and apply the BTC risk-off gate without future data."""
    btc = data.symbols.index("BTC")
    regime = btc_regime_mask(
        data.close_price[:, btc],
        trend_bars=config.trend_bars,
        volatility_bars=config.volatility_bars,
        trend_limit=config.btc_trend_limit,
        volatility_cap=config.btc_volatility_cap,
    )
    positions = _pair_signal(
        data,
        pair=config.pair,
        spread_lookback=config.spread_lookback,
        entry_z=config.entry_z,
        minimum_quote_volume=config.minimum_quote_volume,
    )
    positions[~regime] = 0.0
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
    config: PairRegimeConfig,
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
        "sharpe_4h_annualized": metrics.sharpe,
        "positive_blocks": metrics.positive_blocks,
        "observations": metrics.observations,
    }


def _training_pair_correlations(data: MarketData, train_end: int) -> list[dict[str, float | str]]:
    alt_indices = tuple(data.symbols.index(symbol) for symbol in ALT_SYMBOLS)
    returns = np.diff(np.log(data.close_price[:train_end, alt_indices]), axis=0)
    correlation = np.corrcoef(returns, rowvar=False)
    rows = [
        {"pair": f"{left}-{right}", "correlation": float(correlation[left_index, right_index])}
        for left_index, left in enumerate(ALT_SYMBOLS)
        for right_index, right in enumerate(ALT_SYMBOLS)
        if right_index > left_index
    ]
    return sorted(rows, key=lambda row: abs(float(row["correlation"])), reverse=True)[:10]


def run(archive_root: Path, *, interval: str = INTERVAL) -> dict[str, object]:
    spec = spec_for_interval(interval)
    data, metadata = load_verified_interval_archive(archive_root, ("BTC", *ALT_SYMBOLS), interval=spec.interval)
    configs = _all_configs(spec)
    train_end, development_end = int(len(data.dates) * 0.5), int(len(data.dates) * 0.75)
    all_sums: list[np.ndarray] = []
    all_squares: list[np.ndarray] = []
    bin_counts: tuple[int, ...] | None = None
    winner: ValidationRecord | None = None
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
    active_signal_key: tuple[tuple[str, str], int, float] | None = None
    active_signal: FloatMatrix | None = None
    for index, config in enumerate(configs, start=1):
        signal_key = config.pair, config.spread_lookback, config.entry_z
        if signal_key != active_signal_key:
            active_signal = _pair_signal(
                data,
                pair=config.pair,
                spread_lookback=config.spread_lookback,
                entry_z=config.entry_z,
                minimum_quote_volume=spec.minimum_quote_volume,
            )
            active_signal_key = signal_key
        if active_signal is None:
            raise AssertionError("pair-signal cache must be populated before evaluation")
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
            raise SelectionBiasError("pair configurations produced incompatible CSCV bin counts")
        if index % 1_000 == 0 or index == len(configs):
            print(f"SEARCH_PROGRESS configurations={index}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise SelectionBiasError("pair search generated no configurations")
    pbo = cscv_pbo_from_moments(np.asarray(all_sums), np.asarray(all_squares), bin_counts)
    locked_result: dict[str, object] | None = None
    if winner is not None:
        positions = _pair_signal(
            data,
            pair=winner.config.pair,
            spread_lookback=winner.config.spread_lookback,
            entry_z=winner.config.entry_z,
            minimum_quote_volume=spec.minimum_quote_volume,
        )
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
        "strategy": f"{spec.interval} altcoin pair mean reversion with lagged BTC volatility and trend gate",
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
            "training_top_absolute_correlations": _training_pair_correlations(data, train_end),
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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--interval", default=INTERVAL)
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
