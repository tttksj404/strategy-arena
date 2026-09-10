# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "statsmodels", "pydantic>=2", "typer", "rich"]
# ///
# How to run: uv run work/run_x_volume_control.py
"""Original active-normalized daily signal and volume-exit controlled ablations."""

from __future__ import annotations

import logging
from hashlib import sha256
from pathlib import Path

import numpy as np
import typer
from pydantic import BaseModel
from rich.logging import RichHandler

from alt_search_models import Candidate, Family, MarketData
from alt_search_signals import build_positions
from bitget_archive_daily import load_verified_daily_archive
from bitget_archive_panel import load_verified_interval_archive
from run_x_discovery import END, SYMBOLS, TEST, VALIDATION, WORK
from x_discovery_account import Execution, daily_returns, simulate
from x_discovery_results import Metrics, metrics
from x_discovery_signals import volume_ratio
from x_discovery_types import FloatMatrix


class ControlRow(BaseModel):
    interval: str
    leverage: float
    volume_exit: float
    validation_base: Metrics
    validation_stress: Metrics
    historical_base: Metrics
    historical_stress: Metrics
    validation_filtered_asset_bars: int
    historical_filtered_asset_bars: int


class ControlReport(BaseModel):
    mode: str = "post-v2-preregistered-control"
    selection_independent: bool = False
    live_promotable: bool = False
    preregistration_sha256: str
    rows: tuple[ControlRow, ...]
    source_partition_hashes: tuple[str, ...]
    original_signal_source_sha256: str
    limitations: tuple[str, ...]


def original_signal(data: MarketData) -> FloatMatrix:
    """Call the original engine on the original BTC-excluded universe."""
    alt = MarketData(data.dates, data.symbols[1:], data.open_price[:, 1:], data.high_price[:, 1:],
                     data.low_price[:, 1:], data.close_price[:, 1:], data.quote_volume[:, 1:])
    candidate = Candidate(Family.DONCHIAN, 90, 0, 1, 1, None, 0.,
                          minimum_daily_quote_volume=1e6)
    positions = build_positions(alt, candidate)
    return np.column_stack((np.zeros(len(data.dates)), positions))


def align_signal(daily: MarketData, positions: FloatMatrix, target: MarketData) -> FloatMatrix:
    """Use only decisions already known at each target open, never the next daily bar."""
    stamps = np.asarray([item.timestamp() for item in daily.dates])
    indices = np.searchsorted(stamps, [item.timestamp() for item in target.dates], side="right") - 1
    output = np.zeros_like(target.close_price)
    valid = indices >= 0
    output[valid] = positions[indices[valid]]
    return output


def main(daily_archive: Path = WORK / "bitget_history_full_daily_20260824_v2",
         intraday_archive: Path = WORK / "bitget_history_full_intraday_20260824_v5",
         output: Path = WORK / "x_volume_control_20260910") -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)])
    logger = logging.getLogger(__name__)
    output.mkdir(parents=True, exist_ok=True)
    daily, daily_metadata = load_verified_daily_archive(daily_archive, SYMBOLS)
    original = original_signal(daily)
    rows: list[ControlRow] = []
    hashes: list[str] = list(daily_metadata.source_hashes)
    for interval, bars in (("1D", 1), ("1H", 24), ("5m", 288)):
        logger.info("CONTROL loading %s", interval)
        if interval == "1D":
            data, signal = daily, original
        else:
            data, metadata = load_verified_interval_archive(intraday_archive, SYMBOLS, interval=interval)
            hashes.extend(metadata.source_hashes)
            signal = align_signal(daily, original, data)
        start, split, end = data.dates.index(VALIDATION), data.dates.index(TEST), data.dates.index(END)
        scenarios = ((1., 0.), (2., 0.), (5., 0.)) if interval == "1D" else ((2., 0.), (2., .2), (2., .5))
        ratio = np.ones_like(signal) if interval == "1D" else volume_ratio(data.quote_volume, bars)
        for leverage, threshold in scenarios:
            logger.info("CONTROL %s leverage=%s volume_exit=%s", interval, leverage, threshold)
            weights = signal * leverage * (ratio >= threshold)
            results: list[Metrics] = []
            for label, left, right in (("validation", start, split), ("historical", split, end)):
                for stress in (False, True):
                    contract = Execution(left, right, bars, .0018 if stress else .0009, .0003 if stress else .0001)
                    path = simulate(data, weights, contract)
                    results.append(metrics(path, weights[left:right], bars))
                    if stress:
                        np.savez_compressed(output / f"{interval}_{leverage:g}_{threshold:g}_{label}.npz",
                                            daily_stress_returns=daily_returns(path, bars))
            rows.append(ControlRow(interval=interval, leverage=leverage, volume_exit=threshold,
                                   validation_base=results[0], validation_stress=results[1],
                                   historical_base=results[2], historical_stress=results[3],
                                   validation_filtered_asset_bars=np.count_nonzero(np.logical_and(np.not_equal(signal[start:split], 0), ratio[start:split] < threshold), axis=None, keepdims=False).item(),
                                   historical_filtered_asset_bars=np.count_nonzero(np.logical_and(np.not_equal(signal[split:end], 0), ratio[split:end] < threshold), axis=None, keepdims=False).item()))
    report = ControlReport(preregistration_sha256=sha256((WORK / "X_CONTROL_PREREG_20260910.md").read_bytes()).hexdigest(),
                           rows=tuple(rows), source_partition_hashes=tuple(hashes),
                           original_signal_source_sha256=sha256((WORK / "alt_search_signals.py").read_bytes()).hexdigest(),
                           limitations=("Post-v2 auxiliary controls, no new strategy selection", "Reused history and surviving universe",
                                        "Within-interval comparisons only: target-equity rebalancing changes with interval",
                                        "Conservative market-bar account barrier, no exact mark/MMR liquidation",
                                        "No full memecoin or institutional flow reproduction", "5-minute sampling does not guarantee 60-second reaction"))
    _ = (output / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("X_CONTROL_COMPLETE scenarios=%s", len(rows))


if __name__ == "__main__":
    typer.run(main)
