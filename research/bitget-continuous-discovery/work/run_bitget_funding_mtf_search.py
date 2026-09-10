#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx2[http2,brotli,zstd]",
#     "numpy",
#     "pydantic>=2",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      powershell -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
# 2. Run the public-data paper study:
#      uv run work/run_bitget_funding_mtf_search.py
# ──────────────────

"""Run the fixed, short-horizon funding-price exploration on public Bitget data."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from alt_search_models import RiskOverlay
from bitget_funding_mtf_data import (
    FundingFetchRequest,
    FundingStudyError,
    funding_history,
    funding_matrix,
    timeframe_market,
    trim_to_funding_coverage,
)
from bitget_funding_mtf_core import (
    LEVERAGE_CAPS,
    FundingCandidate,
    FundingPath,
    FundingPathInputs,
    PaperMetrics,
    SignalInputs,
    ValidationRecord,
    build_positions,
    candidate_grid,
    chronological_split,
    evaluate_funding_path,
    metrics,
    select_validation_winners,
    short_horizon_pass,
    validation_score,
)
from bitget_mtf_core import MarketSnapshot
from bitget_mtf_paper_experiment import ALT_UNIVERSE, BitgetRequestError, create_client, fetch_candles, fetch_markets


FUNDING_INTERVALS: Final = ("1H", "2H", "4H", "6H", "12H", "1D")
INTERVAL_HOURS: Final = {"1H": 1.0, "2H": 2.0, "4H": 4.0, "6H": 6.0, "12H": 12.0, "1D": 24.0}
MIN_QUOTE_VOLUME: Final = 1_000_000.0
MAX_SPREAD: Final = 0.001
REPORT_PATH: Final = Path(__file__).with_name("bitget_alt_funding_mtf_search_report.json")


class MetricsPayload(BaseModel):
    """JSON-safe report representation of chronological paper metrics."""

    model_config = ConfigDict(frozen=True)
    net_return: float
    max_drawdown: float
    sharpe: float
    positive_blocks: int
    observations: int


class CandidateReport(BaseModel):
    """One validation-selected candidate with its now-unlocked test result."""

    model_config = ConfigDict(frozen=True)
    candidate: str
    max_leverage: float
    max_realized_leverage: float
    validation: MetricsPayload
    locked_test: MetricsPayload
    locked_test_stress: MetricsPayload
    simulated_liquidation: bool
    short_horizon_paper_pass: bool
    active_symbols: tuple[str, ...]
    market_gate: bool
    promotion: str = "paper-only"


class IntervalReport(BaseModel):
    """Evidence from one closed-bar timeframe."""

    model_config = ConfigDict(frozen=True)
    interval: str
    common_bars: int
    observation_days: float
    funding_event_counts: dict[str, int]
    configurations: int
    validation_eligible: int
    validation_winners: tuple[CandidateReport, ...]


class FundingStudyReport(BaseModel):
    """Top-level public-only artifact with promotion explicitly disabled."""

    model_config = ConfigDict(frozen=True)
    observed_at: datetime
    mode: str = "public-api-paper-only"
    source_endpoints: tuple[str, ...]
    short_horizon_nonpromotable: bool = True
    intervals: tuple[IntervalReport, ...]
    live_promotable: tuple[str, ...] = ()


def _metrics_payload(value: PaperMetrics) -> MetricsPayload:
    """Convert an internal immutable metric value to the report boundary type."""

    return MetricsPayload(
        net_return=value.net_return,
        max_drawdown=value.max_drawdown,
        sharpe=value.sharpe,
        positive_blocks=value.positive_blocks,
        observations=value.observations,
    )


def risk_overlay(max_leverage: float, interval_hours: float) -> RiskOverlay:
    """Scale the fixed daily volatility target to the evaluated closed-bar duration."""

    return RiskOverlay(max_leverage, 0.005 * math.sqrt(interval_hours / 24.0), max(10, round(40.0 * 24.0 / interval_hours)))


def market_gate(active_symbols: tuple[str, ...], markets: dict[str, MarketSnapshot]) -> bool:
    """Require an actual current signal plus a liquid, tight public ticker for every leg."""

    return bool(active_symbols) and all(
        snapshot.quote_volume_usdt >= MIN_QUOTE_VOLUME and snapshot.spread_fraction <= MAX_SPREAD
        for symbol in active_symbols
        if (snapshot := markets.get(symbol)) is not None
    ) and all(symbol in markets for symbol in active_symbols)


def _slice_metrics(path: FundingPath, start: int, end: int, bars_per_day: int) -> tuple[PaperMetrics, PaperMetrics, bool]:
    """Measure a closed chronological slice and its independent stress counterpart."""

    liquidated = start <= path.first_liquidation_index < end
    return metrics(path.returns[start:end], bars_per_day), metrics(path.stress_returns[start:end], bars_per_day), liquidated


def run() -> FundingStudyReport:
    """Collect public data and execute the fixed short-horizon search without trading."""

    observed_at = datetime.now(UTC)
    candidates = candidate_grid()
    with create_client() as client:
        markets = fetch_markets(client)
        histories = {symbol: funding_history(client, FundingFetchRequest(symbol)) for symbol in ALT_UNIVERSE}
        reports: list[IntervalReport] = []
        for interval in FUNDING_INTERVALS:
            print(f"[Bitget public funding] {interval}: collecting {len(ALT_UNIVERSE)} contracts", flush=True)
            candle_map = {symbol: fetch_candles(client, symbol, interval, int(observed_at.timestamp() * 1000)) for symbol in ALT_UNIVERSE}
            timeframe = trim_to_funding_coverage(timeframe_market(candle_map), histories)
            interval_hours = INTERVAL_HOURS[interval]
            bars_per_day = max(1, round(24.0 / interval_hours))
            funding = funding_matrix(timeframe, histories)
            split = chronological_split(len(timeframe.timestamps_ms))
            paths: dict[tuple[str, float], FundingPath] = {}
            validation: list[ValidationRecord] = []
            for candidate in candidates:
                positions = build_positions(SignalInputs(timeframe.market.close_price, funding, interval_hours), candidate)
                for leverage in LEVERAGE_CAPS:
                    path = evaluate_funding_path(FundingPathInputs(timeframe.market, funding, positions, risk_overlay(leverage, interval_hours)))
                    paths[(candidate.identifier, leverage)] = path
                    base, stress, liquidated = _slice_metrics(path, split.train_end, split.validation_end, bars_per_day)
                    score, passes = validation_score(base, stress, liquidated)
                    validation.append(ValidationRecord(candidate, leverage, score, passes))
            winners: list[CandidateReport] = []
            for selected in select_validation_winners(tuple(validation)):
                path = paths[(selected.candidate.identifier, selected.max_leverage)]
                base, stress, liquidated = _slice_metrics(path, split.validation_end, len(timeframe.timestamps_ms), bars_per_day)
                final_positions = build_positions(SignalInputs(timeframe.market.close_price, funding, interval_hours), selected.candidate)[-1]
                active = tuple(symbol for column, symbol in enumerate(timeframe.market.symbols) if abs(final_positions[column]) > 0.0)
                winners.append(CandidateReport(
                    candidate=selected.candidate.identifier, max_leverage=selected.max_leverage,
                    max_realized_leverage=path.max_realized_leverage,
                    validation=_metrics_payload(_slice_metrics(path, split.train_end, split.validation_end, bars_per_day)[0]),
                    locked_test=_metrics_payload(base), locked_test_stress=_metrics_payload(stress),
                    simulated_liquidation=liquidated, short_horizon_paper_pass=short_horizon_pass(base, stress, liquidated),
                    active_symbols=active, market_gate=market_gate(active, markets),
                ))
            reports.append(IntervalReport(
                interval=interval, common_bars=len(timeframe.timestamps_ms), observation_days=len(timeframe.timestamps_ms) / bars_per_day,
                funding_event_counts={symbol: len(events) for symbol, events in histories.items()},
                configurations=len(candidates) * len(LEVERAGE_CAPS), validation_eligible=sum(record.passes for record in validation),
                validation_winners=tuple(winners),
            ))
            print(f"  {interval}: validation winners={len(winners)}; short-paper passes={sum(winner.short_horizon_paper_pass for winner in winners)}", flush=True)
    return FundingStudyReport(
        observed_at=observed_at,
        source_endpoints=("/api/v2/mix/market/candles", "/api/v2/mix/market/history-fund-rate", "/api/v2/mix/market/tickers"),
        intervals=tuple(reports),
    )


def save(report: FundingStudyReport) -> Path:
    """Persist only derived public-data evidence, never credentials or order instructions."""

    REPORT_PATH.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    """Execute the nonpromotable public Bitget funding-price research wave."""

    try:
        report = run()
    except (BitgetRequestError, FundingStudyError) as error:
        print(str(error))
        return 1
    path = save(report)
    print(f"EVIDENCE={path}")
    print("LIVE_PROMOTABLE=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
