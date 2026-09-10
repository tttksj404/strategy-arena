#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

# ─── How to run ───
# uv run --script run_bitget_olmar_chunk.py --interval 5m --start-index 0 --end-index 4 --report bitget_5m_olmar_chunk_000_004.json

"""Run one durable, complete-model-group chunk of the predeclared OLMAR search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import Final

import numpy as np

from alt_search_engine import evaluate_path, validation_score
from alt_search_models import PathResult
from bitget_archive_panel import ArchivePanelError, load_verified_interval_archive
from bitget_olmar import walkforward_olmar_positions
from bitget_olmar_chunk_contract import ChunkArtifact, ChunkContractError, ChunkRecord
from bitget_search_metrics import development_moments, segment_metrics
from run_alt_strategy_search import SYMBOLS as ALT_SYMBOLS
from run_bitget_4h_pair_regime_search import (
    PBO_BINS,
    btc_regime_mask,
    spec_for_interval,
)
from run_bitget_olmar_search import BTC_TREND_LIMIT, SearchConfig, search_configs
from selection_bias_stats import SelectionBiasError

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_intraday_20260824_v5")
DEFAULT_INTERVAL: Final = "5m"
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_olmar_chunk.json")


class ChunkRunError(ValueError):
    """Raised when a requested OLMAR chunk would not be safely mergeable."""


def _model_group_size(configs: Sequence[SearchConfig]) -> int:
    if not configs:
        raise ChunkRunError("no OLMAR configurations are available")
    identifiers = [config.model.identifier for config in configs]
    first = identifiers[0]
    group_size = sum(identifier == first for identifier in identifiers)
    if group_size <= 0 or len(configs) % group_size:
        raise ChunkRunError("OLMAR configurations do not have uniform model groups")
    for start in range(0, len(configs), group_size):
        group = identifiers[start : start + group_size]
        if len(set(group)) != 1:
            raise ChunkRunError("OLMAR configurations are not grouped by model")
    return group_size


def _chunk_bounds(configs: Sequence[SearchConfig], *, start_index: int, end_index: int) -> tuple[int, int]:
    group_size = _model_group_size(configs)
    if (
        start_index < 0
        or end_index > len(configs)
        or start_index >= end_index
        or start_index % group_size
        or end_index % group_size
    ):
        raise ChunkRunError("chunk bounds must cover complete OLMAR model-groups")
    return start_index, end_index


def _validation_score(path: PathResult, *, train_end: int, development_end: int, bars_per_year: float) -> float:
    train, train_stress, train_liquidated = segment_metrics(path, 0, train_end, bars_per_year=bars_per_year)
    validation, validation_stress, validation_liquidated = segment_metrics(
        path,
        train_end,
        development_end,
        bars_per_year=bars_per_year,
    )
    score = validation_score(validation, validation_stress, validation_liquidated)
    if not isfinite(validation_score(train, train_stress, train_liquidated)):
        return float("-inf")
    return score


def _stored_validation_score(score: float) -> float | None:
    """Convert a noncandidate non-finite score into a JSON-safe nullable value."""

    return score if isfinite(score) else None


def run(
    archive_root: Path,
    *,
    interval: str = DEFAULT_INTERVAL,
    start_index: int,
    end_index: int,
) -> ChunkArtifact:
    spec = spec_for_interval(interval)
    configs = search_configs(spec)
    start_index, end_index = _chunk_bounds(configs, start_index=start_index, end_index=end_index)
    data, metadata = load_verified_interval_archive(archive_root, ("BTC", *ALT_SYMBOLS), interval=spec.interval)
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
    bin_counts: tuple[int, ...] | None = None
    records: list[ChunkRecord] = []
    base_positions: dict[str, np.ndarray] = {}
    for index in range(start_index, end_index):
        config = configs[index]
        model_key = config.model.identifier
        if model_key not in base_positions:
            base_positions[model_key] = walkforward_olmar_positions(data, config.model, alt_symbols=ALT_SYMBOLS)
        positions = base_positions[model_key].copy()
        positions[~masks[config.btc_volatility_cap]] = 0.0
        path = evaluate_path(data, positions, config.overlay)
        sums, squares, counts = development_moments(path.stress_returns, development_end=development_end, bins=PBO_BINS)
        if bin_counts is None:
            bin_counts = counts
        elif bin_counts != counts:
            raise ChunkRunError("OLMAR chunk produced incompatible CSCV bin counts")
        records.append(
            ChunkRecord(
                index=index,
                identifier=config.identifier,
                validation_score=_stored_validation_score(
                    _validation_score(
                        path,
                        train_end=train_end,
                        development_end=development_end,
                        bars_per_year=spec.bars_per_year,
                    )
                ),
                sums=tuple(float(value) for value in sums),
                squares=tuple(float(value) for value in squares),
            )
        )
        print(f"CHUNK_PROGRESS configurations={index + 1}/{len(configs)}", flush=True)
    if bin_counts is None:
        raise ChunkRunError("OLMAR chunk generated no configurations")
    return ChunkArtifact(
        interval=metadata.interval,
        snapshot_ms=metadata.snapshot_ms,
        common_rows=metadata.common_rows,
        integrity_verified=metadata.integrity_verified,
        source_hashes=tuple(metadata.source_hashes),
        train_end=train_end,
        development_end=development_end,
        locked_test_rows=len(data.dates) - development_end,
        full_configurations=len(configs),
        start_index=start_index,
        end_index=end_index,
        bin_counts=bin_counts,
        records=tuple(records),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args(argv)
    try:
        artifact = run(
            arguments.archive_root,
            interval=arguments.interval,
            start_index=arguments.start_index,
            end_index=arguments.end_index,
        )
    except (ArchivePanelError, ChunkContractError, ChunkRunError, SelectionBiasError) as error:
        print(f"CHUNK_ERROR={error}")
        return 1
    arguments.report.write_text(json.dumps(artifact.payload(), allow_nan=False, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CHUNK_DONE report={arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
