# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "statsmodels", "pydantic>=2", "typer", "rich"]
# ///
# How to run: uv run work/run_x_discovery.py
"""Execute the predeclared 245-candidate retrospective paper-only experiment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import typer
from rich.logging import RichHandler

from alt_search_models import FloatMatrix, FloatVector, MarketData
from bitget_archive_daily import load_verified_daily_archive
from bitget_archive_panel import load_verified_interval_archive
from selection_bias_stats import SelectionBiasError, deflated_sharpe
from x_discovery_account import Execution, daily_returns, shadow_returns, simulate
from x_discovery_pairs import PairChoice
from x_discovery_results import (
    Configuration,
    DataEvidence,
    DiscoveryReport,
    Sizing,
    TestEvaluation,
    Trial,
    metrics,
    select_winner,
    size_schedule,
)
from x_discovery_signals import (
    SignalFamily,
    SignalSpec,
    build_signal,
    specifications,
    volume_ratio,
)
from x_discovery_statistics import growth_pbo

WORK = Path(__file__).resolve().parent
SYMBOLS = ("BTC", "ETH", "BNB", "ADA", "SOL", "DOGE", "XRP", "DOT", "LTC", "LINK", "AVAX", "UNI", "ATOM", "NEAR")
TRAIN = datetime(2024, 4, 22, 16, tzinfo=UTC)
VALIDATION = datetime(2024, 6, 21, 16, tzinfo=UTC)
TEST = datetime(2025, 6, 22, 16, tzinfo=UTC)
END = datetime(2026, 8, 22, 16, tzinfo=UTC)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Context:
    data: MarketData
    bars_per_day: int
    train: int
    validation: int
    test: int
    end: int


def prefix(data: MarketData, end: int) -> MarketData:
    return MarketData(data.dates[:end], data.symbols, data.open_price[:end], data.high_price[:end],
                      data.low_price[:end], data.close_price[:end], data.quote_volume[:end])


def archive_for(interval: str, intraday: Path, daily: Path) -> Path:
    return daily if interval == "1D" else intraday


def configs(interval: str, signal: SignalSpec) -> tuple[Configuration, ...]:
    values = [Configuration(interval, signal, sizing) for sizing in Sizing]
    if interval == "1H" and signal.family in (SignalFamily.DONCHIAN, SignalFamily.PRESSURE):
        values += [Configuration(interval, signal, sizing, threshold)
                   for threshold in (.2, .5) for sizing in (Sizing.FIXED1, Sizing.FIXED2, Sizing.FIXED5)]
    return tuple(values)


def exposure(data: MarketData, signal: FloatMatrix, config: Configuration, bars_per_day: int) -> FloatMatrix:
    filtered = signal.copy()
    if config.volume_exit:
        filtered *= volume_ratio(data.quote_volume, bars_per_day) >= config.volume_exit
    shadow = shadow_returns(data, filtered, bars_per_day)
    return filtered * size_schedule(config.sizing, shadow, bars_per_day)[:, None]


def evaluate(context: Context, weights: FloatMatrix, config: Configuration,
             pool: tuple[PairChoice, ...], *, test: bool) -> tuple[Trial, FloatVector]:
    start, end = (context.test, context.end) if test else (context.validation, context.test)
    base = simulate(context.data, weights, Execution(start, end, context.bars_per_day, .0009, .0001))
    stress = simulate(context.data, weights, Execution(start, end, context.bars_per_day, .0018, .0003))
    base_metrics = metrics(base, weights[start:end], context.bars_per_day)
    stress_metrics = metrics(stress, weights[start:end], context.bars_per_day)
    failures: list[str] = []
    if stress_metrics.active_days < 30:
        failures.append("active_days_below_30")
    if stress.liquidation_index >= 0:
        failures.append("market_bar_liquidation_proxy_hit")
    if stress_metrics.compounded_return <= 0:
        failures.append("nonpositive_cost_stressed_return")
    eligible = stress_metrics.active_days >= 30 and stress.liquidation_index < 0
    trial = Trial(config=config, identifier=config.identifier, parent_signal=config.signal.identifier,
                  base=base_metrics, stress=stress_metrics, eligible=eligible,
                  failure_reasons=tuple(failures), pair_pool=pool)
    return trial, daily_returns(stress, context.bars_per_day)


def fixed_test_ids(trials: tuple[Trial, ...]) -> tuple[str, ...]:
    winner = select_winner(trials)
    chosen: set[str] = set()
    if winner:
        chosen.add(winner.identifier)
        chosen.update(item.identifier for item in trials
                      if item.config.interval == winner.config.interval
                      and item.parent_signal == winner.parent_signal and item.config.volume_exit == 0)
    for family in SignalFamily:
        best = select_winner(tuple(item for item in trials if item.config.signal.family == family))
        if best:
            chosen.add(best.identifier)
    chosen.update(item.identifier for item in trials if item.config.signal.family == SignalFamily.DONCHIAN
                  and item.config.sizing == Sizing.FIXED2 and item.config.interval in ("1H", "1D"))
    return tuple(sorted(chosen))


def run(archive: Path, output: Path, daily_archive: Path) -> DiscoveryReport:
    output.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, Context] = {}
    evidence: list[DataEvidence] = []
    validation: list[Trial] = []
    paths: list[FloatVector] = []
    for interval, bars in (("1H", 24), ("4H", 6), ("1D", 1)):
        LOGGER.info("Loading and hashing %s", interval)
        root = archive_for(interval, archive, daily_archive)
        if interval == "1D":
            data, metadata = load_verified_daily_archive(root, SYMBOLS)
        else:
            data, metadata = load_verified_interval_archive(root, SYMBOLS, interval=interval)
        context = Context(data, bars, data.dates.index(TRAIN), data.dates.index(VALIDATION),
                          data.dates.index(TEST), data.dates.index(END))
        contexts[interval] = context
        evidence.append(DataEvidence(interval=interval, rows=len(data.dates), start=data.dates[0].isoformat(),
                                     end=data.dates[-1].isoformat(), snapshot_ms=metadata.snapshot_ms,
                                     partition_hashes=metadata.source_hashes))
        development = prefix(data, context.test + 1)
        dev_context = Context(development, bars, context.train, context.validation, context.test, context.test)
        for spec in specifications(interval):
            LOGGER.info("Validation %s %s", interval, spec.identifier)
            signal, pool = build_signal(development, spec, bars_per_day=bars, train_end=context.train)
            for config in configs(interval, spec):
                trial, daily = evaluate(dev_context, exposure(development, signal, config, bars), config, pool, test=False)
                validation.append(trial)
                paths.append(daily)
    trials = tuple(validation)
    ids = fixed_test_ids(trials)
    locked_sha256 = lock_test_ids(output / "test_ids_locked.txt", ids)
    np.savez_compressed(output / "validation_paths.npz", identifiers=np.asarray([item.identifier for item in trials]),
                        daily_stress_returns=np.stack(paths))
    _ = (output / "validation_registry.json").write_text(
        "[\n" + ",\n".join(item.model_dump_json() for item in trials) + "\n]", encoding="utf-8")
    LOGGER.info("Validation complete: %s candidates; test list locked: %s", len(trials), len(ids))
    test_evaluations: list[TestEvaluation] = []
    for identifier in ids:
        selected = next(item for item in trials if item.identifier == identifier)
        config, pool = selected.config, selected.pair_pool
        context = contexts[config.interval]
        LOGGER.info("Historical test %s", config.label)
        signal, _ = build_signal(context.data, config.signal, bars_per_day=context.bars_per_day,
                                 train_end=context.train, pool=pool)
        weights = exposure(context.data, signal, config, context.bars_per_day)
        result, daily = evaluate(context, weights, config, pool, test=True)
        try:
            local = deflated_sharpe(daily, trials=len(trials))
            cumulative = deflated_sharpe(daily, trials=139536 + len(trials))
        except SelectionBiasError:
            local, cumulative = None, None
        test_evaluations.append(TestEvaluation(trial=result, dsr_local_approximation=local,
                                              dsr_cumulative_approximation=cumulative))
        np.savez_compressed(output / f"historical_{identifier}.npz", daily_stress_returns=daily)
    active = [index for index, item in enumerate(trials) if item.stress.active_days >= 30]
    matrix = np.stack([paths[index] for index in active]) if active else np.zeros((0, 0))
    pbo = growth_pbo(matrix) if len(active) >= 2 else None
    winner = select_winner(trials)
    report = DiscoveryReport(candidate_count=len(trials),
                             preregistration_sha256=sha256((WORK / "X_DISCOVERY_PREREG_20260910.md").read_bytes()).hexdigest(),
                             train_end=TRAIN.isoformat(), validation_start=VALIDATION.isoformat(),
                             historical_test_start=TEST.isoformat(), evaluation_end=END.isoformat(),
                             winner_id=winner.identifier if winner else None, test_ids_sha256=locked_sha256,
                             validation=trials, historical_test=tuple(test_evaluations), pbo=pbo,
                             pbo_candidate_count=len(active), pbo_unique_paths=len(np.unique(matrix, axis=0)), data=tuple(evidence),
                             limitations=("Reused historical evaluation, not fresh OOS", "Fixed surviving universe",
                                          "Market-price bars and conservative simultaneous adverse extrema, not mark-price margin rules",
                                          "Assumed fees and financing, no historical orderbook/funding", "No institutional-options or memecoin strategy replication",
                                          "Candle-pressure proxy uses OHLCV only; volume rule tested hourly, not sub-minute execution",
                                          "Local and cumulative trial counts produce heuristic DSR, not a global independence proof"))
    _ = (output / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    LOGGER.info("X_DISCOVERY_COMPLETE candidate_count=%s winner=%s", len(trials), report.winner_id)
    return report


def lock_test_ids(path: Path, identifiers: tuple[str, ...]) -> str:
    _ = path.write_bytes(("\n".join(identifiers) + "\n").encode("utf-8"))
    return sha256(path.read_bytes()).hexdigest()


def main(archive: Path = WORK / "bitget_history_full_intraday_20260824_v5",
         output: Path = WORK / "x_discovery_20260910",
         daily_archive: Path = WORK / "bitget_history_full_daily_20260824_v2") -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)])
    report = run(archive, output, daily_archive)
    typer.echo(f"Saved {output / 'report.json'}; {report.candidate_count} candidates; live_promotable=false")


if __name__ == "__main__":
    typer.run(main)
