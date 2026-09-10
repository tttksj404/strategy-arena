#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx2[http2,brotli,zstd]",
#     "numpy",
#     "pydantic>=2",
# ]
# ///

"""Recheck realized exposure for every current 5x public-paper screen pass."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from bitget_mtf_paper_experiment import ALT_UNIVERSE, create_client, fetch_candles, fetch_markets
from run_bitget_winner_screen import _screen_one, _to_market_data, parse_candidate_identifier, parse_overlay_identifier


SCREEN_PATH = Path(__file__).with_name("bitget_alt_liquidity_winner_screen_report.json")
AUDIT_PATH = Path(__file__).with_name("bitget_5x_leverage_audit_report.json")


class AuditInputError(Exception):
    """The prerequisite public screen is not suitable for a leverage audit."""


class ScreenRow(BaseModel):
    """The only current-screen fields needed for a narrow exposure audit."""

    model_config = ConfigDict(frozen=True)

    candidate: str
    risk: str
    max_leverage: float
    exchange_screen_pass: bool


class ScreenInterval(BaseModel):
    """One public-screen interval and its already selected candidates."""

    model_config = ConfigDict(frozen=True)

    interval: str
    results: tuple[ScreenRow, ...]


class ScreenReport(BaseModel):
    """Boundary for the all-timeframe screen artifact."""

    model_config = ConfigDict(frozen=True)

    mode: str
    results: tuple[ScreenInterval, ...]


class CandidateAudit(BaseModel):
    """One fresh, public-only leverage diagnostic for a 5x-cap candidate."""

    model_config = ConfigDict(frozen=True)

    interval: str
    candidate: str
    risk: str
    max_leverage: float
    max_realized_leverage: float
    base_return: float
    stressed_return: float
    max_drawdown: float
    positive_blocks: int
    simulated_liquidation: bool
    market_gate: bool
    paper_pass: bool
    active_symbols: tuple[str, ...]


class AuditReport(BaseModel):
    """Serializable public-data snapshot with no live-trading authorization."""

    model_config = ConfigDict(frozen=True)

    observed_at: datetime
    mode: str
    source_screen: str
    candidates: tuple[CandidateAudit, ...]
    live_promotable: tuple[str, ...] = ()


def select_cap_five_screen_passes(report: ScreenReport) -> tuple[tuple[str, ScreenRow], ...]:
    """Retain every current public-screen pass capped at exactly five times exposure."""

    return tuple(
        (interval.interval, row)
        for interval in report.results
        for row in interval.results
        if row.exchange_screen_pass and row.max_leverage == 5.0
    )


def _candidate_audit(interval: str, row: ScreenRow, data, markets) -> CandidateAudit:
    """Evaluate one fixed screen winner without changing its signal or risk settings."""

    result = _screen_one(data, interval, parse_candidate_identifier(row.candidate), parse_overlay_identifier(row.risk), markets)
    return CandidateAudit(
        interval=interval,
        candidate=row.candidate,
        risk=row.risk,
        max_leverage=result["max_leverage"],
        max_realized_leverage=result["max_realized_leverage"],
        base_return=result["test"]["net_return"],
        stressed_return=result["stress_return"],
        max_drawdown=result["test"]["max_drawdown"],
        positive_blocks=result["test"]["positive_blocks"],
        simulated_liquidation=result["simulated_liquidation"],
        market_gate=result["market_gate"],
        paper_pass=result["exchange_screen_pass"],
        active_symbols=tuple(result["active_symbols"]),
    )


def main() -> int:
    """Fetch closed public candles once per selected timeframe and persist the audit."""

    screen = ScreenReport.model_validate_json(SCREEN_PATH.read_text(encoding="utf-8"))
    if screen.mode != "public-api-paper-only":
        raise AuditInputError("screen report is not public-api-paper-only")
    selected = select_cap_five_screen_passes(screen)
    if not selected:
        raise AuditInputError("screen report has no current 5x paper passes to audit")
    by_interval = {interval: tuple(row for found, row in selected if found == interval) for interval in sorted({item[0] for item in selected})}
    audited: list[CandidateAudit] = []
    now_ms = int(time.time() * 1000)
    with create_client() as client:
        markets = fetch_markets(client)
        for interval, rows in by_interval.items():
            candles = {symbol: fetch_candles(client, symbol, interval, now_ms) for symbol in ALT_UNIVERSE}
            data = _to_market_data(candles)
            audited.extend(_candidate_audit(interval, row, data, markets) for row in rows)
    report = AuditReport(
        observed_at=datetime.now(UTC),
        mode="public-api-paper-only",
        source_screen=str(SCREEN_PATH),
        candidates=tuple(audited),
    )
    AUDIT_PATH.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"BITGET_5X_AUDIT_DONE report={AUDIT_PATH} candidates={len(audited)} live_promotable=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
