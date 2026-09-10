"""Core contract and report assembly for resumable OLMAR searches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Final, override

import numpy as np
from pydantic import TypeAdapter, ValidationError

from alt_search_engine import evaluate_path, test_passes
from alt_search_models import FloatMatrix, MarketData
from bitget_archive_panel import ArchivePanelMetadata, load_verified_interval_archive
from bitget_directional_change_payloads import dsr_payload, number_payload
from bitget_olmar import walkforward_olmar_positions
from bitget_olmar_chunk_contract import (
    ChunkArtifact,
    ChunkContractError,
    ChunkRecord,
    JsonValue,
    complete_records,
)
from bitget_search_metrics import metric_payload, segment_metrics
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import (
    DSR_MIN,
    PBO_MAX,
    btc_regime_mask,
    spec_for_interval,
)
from run_bitget_olmar_search import BTC_TREND_LIMIT, SearchConfig, search_configs
from selection_bias_stats import (
    SelectionBiasError,
    cscv_pbo_from_moments,
    deflated_sharpe,
)

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_INTERVAL: Final = "5m"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class MergeError(RuntimeError):
    """Raised when durable chunks cannot reproduce one full OLMAR search."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class LockedOutcome:
    """One locked-test report and its independently evaluated eligibility bit."""

    payload: dict[str, JsonValue]
    eligible: bool


def _load_chunk(path: Path) -> ChunkArtifact:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MergeError(f"cannot read chunk artifact: {path}") from error
    try:
        payload = JSON_OBJECT.validate_json(raw)
    except ValidationError as error:
        raise MergeError(f"chunk artifact is not a valid JSON object: {path}") from error
    try:
        return ChunkArtifact.from_payload(payload)
    except ChunkContractError as error:
        raise MergeError(f"chunk artifact violates its contract: {path}") from error


def _moment_arrays(records: Sequence[ChunkRecord]) -> tuple[FloatMatrix, FloatMatrix]:
    """Return CSCV sufficient statistics in deterministic configuration order."""

    return (
        np.asarray(tuple(record.sums for record in records), dtype=float),
        np.asarray(tuple(record.squares for record in records), dtype=float),
    )


def _winner_record(records: Sequence[ChunkRecord]) -> ChunkRecord | None:
    """Select the highest finite development score without looking at locked returns."""

    winner: ChunkRecord | None = None
    for record in records:
        if record.validation_score is not None and isfinite(record.validation_score) and (
            winner is None or winner.validation_score is None or record.validation_score > winner.validation_score
        ):
            winner = record
    return winner


def _locked_outcome(
    record: ChunkRecord,
    config: SearchConfig,
    *,
    positions: FloatMatrix,
    data: MarketData,
    locked_start: int,
    bars_per_year: float,
    raw_trials: int,
    pbo: float,
) -> LockedOutcome:
    path = evaluate_path(data, positions, config.overlay)
    test, stress, liquidated = segment_metrics(path, locked_start, len(data.dates), bars_per_year=bars_per_year)
    try:
        dsr = deflated_sharpe(path.stress_returns[locked_start:], trials=raw_trials)
    except SelectionBiasError:
        dsr = None
    paper_pass = test_passes(test, stress, liquidated)
    dsr_pass = dsr is not None and dsr.probability >= DSR_MIN
    eligible = paper_pass and pbo <= PBO_MAX and dsr_pass
    return LockedOutcome(
        payload={
            "id": config.identifier,
            "algorithm": "strictly_lagged_market_neutral_online_moving_average_reversion",
            "validation_score": record.validation_score,
            "test": number_payload(metric_payload(test)),
            "test_stress": number_payload(metric_payload(stress)),
            "test_liquidated": liquidated,
            "deflated_sharpe_stress": None if dsr is None else dsr_payload(dsr),
            "paper_pass": paper_pass,
            "pbo_pass": pbo <= PBO_MAX,
            "dsr_pass": dsr_pass,
            "eligible_for_mark_price_paper_validation": eligible,
            "live_promotable": False,
        },
        eligible=eligible,
    )


def _validate_contract(
    records: Sequence[ChunkRecord],
    chunks: Sequence[ChunkArtifact],
    configs: Sequence[SearchConfig],
    metadata: ArchivePanelMetadata,
    data: MarketData,
    *,
    interval: str,
) -> ChunkArtifact:
    master = chunks[0]
    if master.interval != interval or len(records) != len(configs):
        raise MergeError("chunk interval or configuration count does not match the predeclared grid")
    if any(record.identifier != configs[record.index].identifier for record in records):
        raise MergeError("chunk identifiers do not match the predeclared OLMAR grid")
    if (
        metadata.snapshot_ms != master.snapshot_ms
        or metadata.interval != master.interval
        or metadata.common_rows != master.common_rows
        or metadata.source_hashes != master.source_hashes
        or not metadata.integrity_verified
        or master.train_end != int(len(data.dates) * 0.5)
        or master.development_end != int(len(data.dates) * 0.75)
        or master.locked_test_rows != len(data.dates) - master.development_end
    ):
        raise MergeError("chunk provenance does not match the current verified archive")
    return master


def run(archive_root: Path, *, chunk_paths: tuple[Path, ...], interval: str = DEFAULT_INTERVAL) -> dict[str, JsonValue]:
    """Merge immutable chunks and evaluate exactly one validation-selected locked test."""

    chunks = tuple(_load_chunk(path) for path in chunk_paths)
    records = complete_records(chunks)
    spec = spec_for_interval(interval)
    configs = search_configs(spec)
    data, metadata = load_verified_interval_archive(archive_root, ("BTC", *ALT_SYMBOLS), interval=spec.interval)
    master = _validate_contract(records, chunks, configs, metadata, data, interval=spec.interval)
    sums, squares = _moment_arrays(records)
    pbo = cscv_pbo_from_moments(sums, squares, master.bin_counts)
    winner = _winner_record(records)
    locked: list[dict[str, JsonValue]] = []
    eligible: list[str] = []
    winners: list[str] = []
    if winner is not None:
        config = configs[winner.index]
        btc_index = data.symbols.index("BTC")
        mask = btc_regime_mask(
            data.close_price[:, btc_index],
            trend_bars=spec.trend_bars,
            volatility_bars=spec.volatility_bars,
            trend_limit=BTC_TREND_LIMIT,
            volatility_cap=config.btc_volatility_cap,
        )
        positions = walkforward_olmar_positions(data, config.model, alt_symbols=ALT_SYMBOLS)
        positions[~mask] = 0.0
        outcome = _locked_outcome(
            winner,
            config,
            positions=positions,
            data=data,
            locked_start=master.development_end,
            bars_per_year=spec.bars_per_year,
            raw_trials=len(configs),
            pbo=pbo.pbo,
        )
        winners.append(config.identifier)
        locked.append(outcome.payload)
        if outcome.eligible:
            eligible.append(config.identifier)
    source_hashes: list[JsonValue] = [*metadata.source_hashes]
    locked_payloads: list[JsonValue] = [*locked]
    winner_ids: list[JsonValue] = [*winners]
    eligible_ids: list[JsonValue] = [*eligible]
    data_payload: dict[str, JsonValue] = {
        "interval": metadata.interval,
        "snapshot_ms": metadata.snapshot_ms,
        "common_rows": metadata.common_rows,
        "integrity_verified": metadata.integrity_verified,
        "source_hashes": source_hashes,
        "train_rows": master.train_end,
        "development_rows": master.development_end,
        "locked_test_rows": master.locked_test_rows,
    }
    predeclared_payload: dict[str, JsonValue] = {
        "algorithm_family": "market_neutral_online_moving_average_reversion",
        "raw_configurations": len(configs),
        "max_leverage": 5.0,
        "btc_trend_limit": BTC_TREND_LIMIT,
        "minimum_quote_volume_per_bar": spec.minimum_quote_volume,
    }
    pbo_payload: dict[str, JsonValue] = {
        "pbo": pbo.pbo,
        "splits": pbo.splits,
        "candidates": pbo.candidates,
        "median_logit": pbo.median_logit,
    }
    selection_bias_payload: dict[str, JsonValue] = {
        "method": "CSCV PBO on stress development returns; DSR on locked stress returns",
        "pbo": pbo_payload,
        "pbo_max": PBO_MAX,
        "dsr_min_probability": DSR_MIN,
    }
    return {
        "mode": "public-api-paper-only-no-live-orders",
        "selection_independent": False,
        "strategy": "strictly lagged market-neutral adaptation of online moving-average reversion",
        "liquidation_evidence": "market-candle proxy only; full mark-price history remains required before any promotion",
        "data": data_payload,
        "predeclared_search": predeclared_payload,
        "selection_bias": selection_bias_payload,
        "selected_validation_winners": winner_ids,
        "locked_test_results": locked_payloads,
        "eligible_for_mark_price_paper_validation": eligible_ids,
        "live_promotable": [],
    }
