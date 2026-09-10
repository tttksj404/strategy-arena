#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pydantic>=2"]
# ///

# Usage: python work/run_bitget_top_strategy_diagnostic.py --report PATH
# Public archive only; this script never sends orders or requires credentials.
# pyright: reportImplicitRelativeImport=false

"""Reproduce the two named daily-bar candidates and expose their path evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Final, NotRequired, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict

from alt_search_engine import evaluate_path, segment_metrics
from alt_search_models import (
    BASE_FEE,
    BASE_SLIPPAGE,
    LIQUIDATION_RESERVE,
    STRESS_SLIPPAGE,
    Candidate,
    Family,
    FloatMatrix,
    FloatVector,
    MarketData,
    PathResult,
    RiskOverlay,
)
from alt_search_signals import build_positions
from bitget_archive_daily import ArchiveDailyError, load_verified_daily_archive

DEFAULT_ARCHIVE: Final = Path(__file__).with_name("bitget_history_full_daily_20260824_v2")
DEFAULT_REPORT: Final = Path(__file__).with_name("bitget_top_strategy_diagnostic_report.json")
SYMBOLS: Final = ("ETH", "BNB", "ADA", "SOL", "DOGE", "XRP", "DOT", "LTC", "LINK", "AVAX", "UNI", "ATOM", "NEAR")
class SelectionRow(BaseModel):
    """Typed subset of one locked-test row from the source PBO report."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    id: str
    paper_pass: bool
    pbo_pass: bool
    dsr_pass: bool
    eligible_for_further_paper_validation: bool
    live_promotable: bool


class SelectionSource(BaseModel):
    """Typed source report boundary for selection-gate provenance."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    selection_independent: bool
    locked_test_results: tuple[SelectionRow, ...]


class MetricPayload(TypedDict):
    net_return: float
    max_drawdown: float
    sharpe: float
    positive_blocks: int
    observations: int


class LeveragePayload(TypedDict):
    active_observations: int
    min: float
    median: float
    p95: float
    max: float
    mean: float


class SymbolContribution(TypedDict):
    base_arithmetic_bar_return_contribution: float
    stress_arithmetic_bar_return_contribution: float
    gross_arithmetic_bar_return_contribution: float
    active_bars: int


class DatePayload(TypedDict):
    start: str
    end: str


class SegmentPayload(TypedDict):
    rows: int
    dates: DatePayload
    base: MetricPayload
    stress: MetricPayload
    liquidation_hits: int
    first_liquidation_index: int | None
    active_bar_count: int
    leverage: LeveragePayload
    symbol_contribution: dict[str, SymbolContribution]
    liquidated: bool


class DiagnosticPayload(TypedDict):
    event_count: int
    left_censored_event_count: int
    right_censored_event_count: int
    completed_event_terminal_direction_winners: int
    completed_event_winner_liquidation_proxy: int
    liquidation_proxy_events: int
    mae_median: float | None
    mae_p95: float | None
    mfe_median: NotRequired[float | None]
    completed_event_terminal_direction_win_rate: NotRequired[float]
    completed_event_winner_liquidation_rate: NotRequired[float]
    mfe_p95: NotRequired[float]


class CounterfactualPayload(TypedDict):
    scenario: str
    base: MetricPayload
    stress: MetricPayload
    market_ohlc_proxy_liquidation_hit_count: int
    first_market_ohlc_proxy_liquidation_index: int | None
    counterfactual_continues_after_first_liquidation: bool
    post_first_liquidation_base_arithmetic_return: float
    post_first_liquidation_stress_arithmetic_return: float


class SelectionGatePayload(TypedDict):
    candidate_id: str
    overlay_id: str
    eligible_for_further_paper_validation: bool
    paper_pass: bool
    pbo_pass: bool
    dsr_pass: bool
    live_promotable: bool


class SelectionSourcePayload(TypedDict):
    path: str
    sha256: str


class StrategyPayload(TypedDict):
    candidate: str
    overlay: str
    segments: dict[str, SegmentPayload]
    locked_test_daily_bar_diagnostic: DiagnosticPayload
    fixed_5x_counterfactual: CounterfactualPayload


class DataPayload(TypedDict):
    archive_root: str
    snapshot_ms: int
    common_rows: int
    integrity_verified: bool
    source_hashes: list[str]
    split: dict[str, int]


class ContractPayload(TypedDict):
    execution: str
    donchian_direction_1: str
    tp_sl: str
    liquidation: str
    funding: str
    mark_price: str
    selection_independent: bool
    selection: str


class ReportPayload(TypedDict):
    mode: str
    observed_at: str
    live_promotable: list[str]
    data: DataPayload
    selection_audit_source: SelectionSourcePayload
    selection_audit: dict[str, SelectionGatePayload]
    contract: ContractPayload
    strategies: dict[str, StrategyPayload]


EXPECTED_SELECTION_SOURCE_SHA256: Final = "dfcd96aead7a059ba32704d4d1c3cd6bdc97377f2db919b511458185fecf8871"
DEFAULT_SELECTION_REPORT: Final = Path(__file__).with_name("bitget_archive_pbo_search_report.json")


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """One previously named candidate and its fixed risk overlay."""

    name: str
    candidate: Candidate
    overlay: RiskOverlay


def strategy_specs() -> tuple[StrategySpec, ...]:
    """Return exactly the raw-top and paper-gate candidates from the search."""

    return (
        StrategySpec("raw_top_donchian", Candidate(Family.DONCHIAN, 90, 0, 0, 1, None, 0.0, 1, 1_000_000.0), RiskOverlay(5.0, 0.005, 40)),
        StrategySpec("paper_gate_top_price_volume", Candidate(Family.PRICE_VOLUME, 60, 0, 1, 1, None, 0.0, 2, 20_000_000.0), RiskOverlay(5.0, 0.01, 40)),
    )


def split_bounds(rows: int) -> tuple[int, int, int, int]:
    """Return chronological 50/25/25 train, validation, and locked-test bounds."""

    if rows < 4:
        raise ValueError("at least four daily rows are required")
    train_end = rows // 2
    validation_end = train_end + (rows - train_end) // 2
    return 0, train_end, validation_end, rows


def _leverage_payload(values: FloatVector) -> LeveragePayload:
    active = values[values > 0.0]
    if not len(active):
        return {"active_observations": 0, "min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {"active_observations": len(active), "min": float(active.min()), "median": float(np.median(active)), "p95": float(np.quantile(active, 0.95)), "max": float(active.max()), "mean": float(active.mean())}


def _matrix_value(values: FloatMatrix, row: int, column: int) -> float:
    """Read one float without relying on numpy's dynamically typed scalar index."""

    return float(values.item(row, column))


def _position_sign(value: float) -> float:
    """Return the sign used by the position event state machine."""

    return 1.0 if value > 0.0 else -1.0 if value < 0.0 else 0.0


def _symbol_contributions(data: MarketData, positions: FloatMatrix, path: PathResult, start: int, end: int) -> dict[str, SymbolContribution]:
    """Attribute arithmetic close-to-close return contributions by symbol."""

    asset_returns = np.zeros_like(data.close_price)
    asset_returns[1:] = data.close_price[1:] / data.close_price[:-1] - 1.0
    scaled = positions * path.leverage[:, None]
    turnover = np.abs(scaled)
    turnover[1:] = np.abs(scaled[1:] - scaled[:-1])
    base = scaled * asset_returns - turnover * (BASE_FEE + BASE_SLIPPAGE)
    stress = scaled * asset_returns - turnover * (BASE_FEE + STRESS_SLIPPAGE)
    return {
        symbol: {"base_arithmetic_bar_return_contribution": float(base[start:end, column].sum()), "stress_arithmetic_bar_return_contribution": float(stress[start:end, column].sum()), "gross_arithmetic_bar_return_contribution": float((scaled[start:end, column] * asset_returns[start:end, column]).sum()), "active_bars": int(np.count_nonzero(np.abs(positions[start:end, column]) > 0.0))}
        for column, symbol in enumerate(data.symbols)
    }


def _fixed_5x_counterfactual(data: MarketData, positions: FloatMatrix) -> tuple[PathResult, tuple[int, ...]]:
    """Simulate active bars at exactly 5x without stopping after proxy hits."""

    asset_returns = np.zeros_like(data.close_price)
    asset_returns[1:] = data.close_price[1:] / data.close_price[:-1] - 1.0
    active = np.abs(positions).sum(axis=1) > 0.0
    leverage = np.where(active, 5.0, 0.0)
    scaled = positions * leverage[:, None]
    turnover = np.abs(scaled).sum(axis=1)
    turnover[1:] = np.abs(scaled[1:] - scaled[:-1]).sum(axis=1)
    raw_unscaled = (positions * asset_returns).sum(axis=1)
    net = raw_unscaled * leverage - turnover * (BASE_FEE + BASE_SLIPPAGE)
    stress = raw_unscaled * leverage - turnover * (BASE_FEE + STRESS_SLIPPAGE)
    long_adverse = np.zeros_like(data.close_price)
    short_adverse = np.zeros_like(data.close_price)
    long_adverse[1:] = (data.close_price[:-1] - data.low_price[1:]) / data.close_price[:-1]
    short_adverse[1:] = (data.high_price[1:] - data.close_price[:-1]) / data.close_price[:-1]
    threshold = np.full_like(leverage, np.inf)
    threshold[active] = 1.0 / leverage[active] - LIQUIDATION_RESERVE
    touched = ((scaled > 0.0) & (long_adverse >= threshold[:, None])) | ((scaled < 0.0) & (short_adverse >= threshold[:, None]))
    hits = tuple(int(index) for index in np.flatnonzero(touched.any(axis=1)))
    return PathResult(net, stress, leverage, len(hits), hits[0] if hits else -1), hits


def _fixed_5x_payload(path: PathResult, hits: tuple[int, ...], start: int, end: int) -> CounterfactualPayload:
    """Report fixed-5x locked-test metrics without mixing them with dynamic sizing."""

    base, stress, _ = segment_metrics(path, start, end)
    segment_hits = tuple(index for index in hits if start <= index < end)
    first = segment_hits[0] if segment_hits else None
    post_hit = path.returns[first + 1 : end] if first is not None else np.array([], dtype=float)
    post_stress = path.stress_returns[first + 1 : end] if first is not None else np.array([], dtype=float)
    return {
        "scenario": "fixed 5x on every active bar; independent counterfactual, not the dynamic-volatility result",
        "base": {"net_return": base.net_return, "max_drawdown": base.max_drawdown, "sharpe": base.sharpe, "positive_blocks": base.positive_blocks, "observations": base.observations},
        "stress": {"net_return": stress.net_return, "max_drawdown": stress.max_drawdown, "sharpe": stress.sharpe, "positive_blocks": stress.positive_blocks, "observations": stress.observations},
        "market_ohlc_proxy_liquidation_hit_count": len(segment_hits),
        "first_market_ohlc_proxy_liquidation_index": first,
        "counterfactual_continues_after_first_liquidation": bool(first is not None and len(post_hit) > 0),
        "post_first_liquidation_base_arithmetic_return": float(post_hit.sum()),
        "post_first_liquidation_stress_arithmetic_return": float(post_stress.sum()),
    }


def _segment_payload(data: MarketData, positions: FloatMatrix, path: PathResult, start: int, end: int) -> SegmentPayload:
    base, stress, liquidated = segment_metrics(path, start, end)
    active = np.abs(positions[start:end]).sum(axis=1) > 0.0
    first = path.first_liquidation_index
    return {
        "rows": end - start,
        "dates": {"start": data.dates[start].isoformat(), "end": data.dates[end - 1].isoformat()},
        "base": {"net_return": base.net_return, "max_drawdown": base.max_drawdown, "sharpe": base.sharpe, "positive_blocks": base.positive_blocks, "observations": base.observations},
        "stress": {"net_return": stress.net_return, "max_drawdown": stress.max_drawdown, "sharpe": stress.sharpe, "positive_blocks": stress.positive_blocks, "observations": stress.observations},
        "liquidation_hits": int(path.liquidations if first >= start and first < end else 0),
        "first_liquidation_index": first if first >= start and first < end else None,
        "active_bar_count": int(active.sum()),
        "leverage": _leverage_payload(path.leverage[start:end]),
        "symbol_contribution": _symbol_contributions(data, positions, path, start, end),
        "liquidated": liquidated,
    }


def daily_bar_diagnostics(data: MarketData, positions: FloatMatrix, leverage: float | FloatVector, start: int, end: int) -> DiagnosticPayload:
    """Measure path excursions and engine-matched daily liquidation proxies.

    An entry at row ``i`` starts at ``close[i-1]`` because the engine applies row i's
    position to ``close[i] / close[i-1]``. Each event ends on the first sign exit or
    at the locked-test boundary. Liquidation is checked independently on every bar
    with that bar's prior close, high/low, and dynamic leverage, exactly as in the
    paper engine; this remains a market-candle proxy rather than Bitget mark price.
    """

    if start < 1 or end > len(data.dates) or start >= end:
        raise ValueError("invalid diagnostic bounds or leverage")
    if isinstance(leverage, np.ndarray):
        if leverage.shape != (len(data.dates),) or np.any(~np.isfinite(leverage)) or np.any(leverage < 0.0):
            raise ValueError("leverage vector must align with daily rows and be non-negative")
    elif not np.isfinite(leverage) or leverage <= 0.0:
        raise ValueError("invalid diagnostic bounds or leverage")
    maes: list[float] = []
    mfes: list[float] = []
    asset_count = len(data.symbols)
    left_censored = sum(
        int(
            _position_sign(_matrix_value(positions, start, asset)) != 0.0
            and _position_sign(_matrix_value(positions, start, asset)) == _position_sign(_matrix_value(positions, start - 1, asset))
        )
        for asset in range(asset_count)
    )
    right_censored = 0
    winners = winner_liquidations = liquidation_events = 0
    for index in range(start, end):
        for asset in range(asset_count):
            direction = _position_sign(_matrix_value(positions, index, asset))
            previous_direction = _position_sign(_matrix_value(positions, index - 1, asset))
            if direction == 0.0 or direction == previous_direction:
                continue
            exit_index = index + 1
            while exit_index < end and _position_sign(_matrix_value(positions, exit_index, asset)) == direction:
                exit_index += 1
            right_censored += int(exit_index >= end)
            if exit_index >= end:
                continue
            entry = _matrix_value(data.close_price, index - 1, asset)
            lowest = min(_matrix_value(data.low_price, bar, asset) for bar in range(index, exit_index))
            highest = max(_matrix_value(data.high_price, bar, asset) for bar in range(index, exit_index))
            adverse = max(0.0, (entry - lowest) / entry) if direction > 0 else max(0.0, (highest - entry) / entry)
            favorable = max(0.0, (highest - entry) / entry) if direction > 0 else max(0.0, (entry - lowest) / entry)
            eventual = direction * (_matrix_value(data.close_price, exit_index - 1, asset) / entry - 1.0) > 0.0
            liquidation = False
            for bar in range(index, exit_index):
                bar_entry = _matrix_value(data.close_price, bar - 1, asset)
                bar_low = _matrix_value(data.low_price, bar, asset)
                bar_high = _matrix_value(data.high_price, bar, asset)
                bar_adverse = ((bar_entry - bar_low) / bar_entry if direction > 0 else (bar_high - bar_entry) / bar_entry)
                bar_leverage = float(leverage.item(bar)) if isinstance(leverage, np.ndarray) else leverage
                liquidation = liquidation or (
                    bar_leverage > 0.0
                    and bar_adverse >= 1.0 / bar_leverage - LIQUIDATION_RESERVE
                )
            maes.append(adverse)
            mfes.append(favorable)
            winners += int(eventual)
            liquidation_events += int(liquidation)
            winner_liquidations += int(eventual and liquidation)
    if not maes:
        return {"event_count": 0, "left_censored_event_count": left_censored, "right_censored_event_count": right_censored, "completed_event_terminal_direction_winners": 0, "completed_event_winner_liquidation_proxy": 0, "liquidation_proxy_events": 0, "mae_median": None, "mae_p95": None, "mfe_median": None}
    mae, mfe = np.asarray(maes), np.asarray(mfes)
    return {"event_count": len(maes), "left_censored_event_count": left_censored, "right_censored_event_count": right_censored, "completed_event_terminal_direction_winners": winners, "completed_event_terminal_direction_win_rate": winners / len(maes), "completed_event_winner_liquidation_proxy": winner_liquidations, "completed_event_winner_liquidation_rate": winner_liquidations / winners if winners else 0.0, "liquidation_proxy_events": liquidation_events, "mae_median": float(np.median(mae)), "mae_p95": float(np.quantile(mae, 0.95)), "mfe_median": float(np.median(mfe)), "mfe_p95": float(np.quantile(mfe, 0.95))}


def selection_audit(selection_report: Path) -> tuple[SelectionSourcePayload, dict[str, SelectionGatePayload]]:
    """Import only hash-verified, exact gate rows from the source PBO report."""

    try:
        source_bytes = selection_report.read_bytes()
    except OSError as error:
        raise ArchiveDailyError(f"cannot read selection source {selection_report}") from error
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != EXPECTED_SELECTION_SOURCE_SHA256:
        raise ArchiveDailyError(f"selection source hash mismatch for {selection_report}")
    try:
        source = SelectionSource.model_validate_json(source_bytes)
    except ValueError as error:
        raise ArchiveDailyError(f"invalid selection source {selection_report}") from error
    if source.selection_independent:
        raise ArchiveDailyError("selection source unexpectedly claims independent selection")
    expected_gates: dict[str, tuple[bool, bool, bool, bool, bool]] = {
        "raw_top_donchian": (False, False, False, False, False),
        "paper_gate_top_price_volume": (False, True, False, False, False),
    }
    audit: dict[str, SelectionGatePayload] = {}
    for spec in strategy_specs():
        source_id = f"{spec.candidate.identifier}|{spec.overlay.identifier}"
        matching = tuple(row for row in source.locked_test_results if row.id == source_id)
        if len(matching) != 1:
            raise ArchiveDailyError(f"selection source missing unique row {source_id}")
        row = matching[0]
        actual = (row.eligible_for_further_paper_validation, row.paper_pass, row.pbo_pass, row.dsr_pass, row.live_promotable)
        if actual != expected_gates[spec.name]:
            raise ArchiveDailyError(f"selection gate changed for {source_id}")
        audit[spec.name] = {"candidate_id": spec.candidate.identifier, "overlay_id": spec.overlay.identifier, "eligible_for_further_paper_validation": row.eligible_for_further_paper_validation, "paper_pass": row.paper_pass, "pbo_pass": row.pbo_pass, "dsr_pass": row.dsr_pass, "live_promotable": row.live_promotable}
    return {"path": str(selection_report), "sha256": source_hash}, audit


def run(archive_root: Path = DEFAULT_ARCHIVE, selection_report: Path = DEFAULT_SELECTION_REPORT) -> ReportPayload:
    """Load the verified daily archive and return the two-candidate evidence payload."""

    data, metadata = load_verified_daily_archive(archive_root, SYMBOLS)
    bounds = split_bounds(len(data.dates))
    selection_source, selection_gates = selection_audit(selection_report)
    results: dict[str, StrategyPayload] = {}
    for spec in strategy_specs():
        positions = build_positions(data, spec.candidate)
        path = evaluate_path(data, positions, spec.overlay)
        fixed_path, fixed_hits = _fixed_5x_counterfactual(data, positions)
        results[spec.name] = {"candidate": spec.candidate.identifier, "overlay": spec.overlay.identifier, "segments": {"train": _segment_payload(data, positions, path, bounds[0], bounds[1]), "validation": _segment_payload(data, positions, path, bounds[1], bounds[2]), "locked_test": _segment_payload(data, positions, path, bounds[2], bounds[3])}, "locked_test_daily_bar_diagnostic": daily_bar_diagnostics(data, positions, path.leverage, bounds[2], bounds[3]), "fixed_5x_counterfactual": _fixed_5x_payload(fixed_path, fixed_hits, bounds[2], bounds[3])}
    return {"mode": "public-archive-paper-only", "observed_at": datetime.now(UTC).isoformat(), "live_promotable": [], "data": {"archive_root": str(archive_root), "snapshot_ms": metadata.snapshot_ms, "common_rows": metadata.common_rows, "integrity_verified": metadata.integrity_verified, "source_hashes": list(metadata.source_hashes), "split": {"train_rows": bounds[1], "validation_rows": bounds[2] - bounds[1], "locked_test_rows": bounds[3] - bounds[2]}}, "selection_audit_source": selection_source, "selection_audit": selection_gates, "contract": {"execution": "close-to-close: row i position applies to close[i]/close[i-1]", "donchian_direction_1": "bidirectional breakout: long above prior high and short below prior low", "tp_sl": "no explicit TP or SL; exits are signal changes or end of segment", "liquidation": "next daily high/low proxy using each bar's prior close and that bar's dynamic leverage; path stops at first proxy hit", "funding": "not reflected", "mark_price": "not reflected", "selection_independent": False, "selection": "two candidates were selected from the prior validation search; each was evaluated once on the locked test afterward, without locked-test feedback; neither is promotable"}, "strategies": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    _ = parser.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION_REPORT)
    _ = parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parsed: object = parser.parse_args(argv)
    archive_root = getattr(parsed, "archive_root", None)
    selection_report = getattr(parsed, "selection_report", None)
    report_path = getattr(parsed, "report", None)
    if not isinstance(archive_root, Path) or not isinstance(selection_report, Path) or not isinstance(report_path, Path):
        raise TypeError("argparse returned invalid path arguments")
    try:
        report = run(archive_root, selection_report)
    except ArchiveDailyError as error:
        print(f"DIAGNOSTIC_ERROR={error}")
        return 1
    _ = report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DIAGNOSTIC_DONE report={report_path} strategies={len(report['strategies'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
