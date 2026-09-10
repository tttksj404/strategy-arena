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
# 2. Run after the broad search report exists:
#      uv run work/run_bitget_winner_screen.py
# ──────────────────

"""Screen validation winners on public Bitget candles across every supported interval."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from alt_search_engine import evaluate_path, segment_metrics
from alt_search_models import Candidate, Family, MarketData, RiskOverlay
from alt_search_signals import build_positions
from bitget_mtf_core import Candle, MarketSnapshot
from bitget_mtf_paper_experiment import ALT_UNIVERSE, INTERVALS, BitgetRequestError, create_client, fetch_candles, fetch_markets


REPORT_PATH: Final = Path(__file__).with_name("bitget_alt_strategy_liquidity_search_report.json")
SCREEN_PATH: Final = Path(__file__).with_name("bitget_alt_liquidity_winner_screen_report.json")
INTERVAL_HOURS: Final = {"1m": 1.0 / 60.0, "3m": 3.0 / 60.0, "5m": 5.0 / 60.0, "15m": 0.25, "30m": 0.5, "1H": 1.0, "2H": 2.0, "4H": 4.0, "6H": 6.0, "12H": 12.0, "1D": 24.0, "3D": 72.0, "1W": 168.0, "1M": 720.0}
MIN_QUOTE_VOLUME: Final = 5_000_000.0
MAX_SPREAD: Final = 0.001


class ScreenInputError(Exception):
    """A completed broad-search report is missing a required typed value."""


class WinnerRow(BaseModel):
    """The minimal selection row used at the report boundary."""

    model_config = ConfigDict(frozen=True)
    id: str
    candidate: str
    risk: str


class BroadSearchReport(BaseModel):
    """The completed paper-search report boundary."""

    model_config = ConfigDict(frozen=True)
    validation_winners: tuple[WinnerRow, ...]


class ScreenMetrics(TypedDict):
    net_return: float
    max_drawdown: float
    sharpe: float
    positive_blocks: int


class ScreenResult(TypedDict):
    candidate: str
    risk: str
    max_leverage: float
    max_realized_leverage: float
    active_symbols: list[str]
    oos_days: float
    observations: int
    market_gate: bool
    test: ScreenMetrics
    stress_return: float
    simulated_liquidation: bool
    exchange_screen_pass: bool
    promotion: str


class IntervalScreenResult(TypedDict, total=False):
    interval: str
    common_bars: int
    observation_days: float
    results: list[ScreenResult]
    error: str


def parse_candidate_identifier(identifier: str) -> Candidate:
    """Parse the deterministic candidate identifier emitted by the broad search."""

    fragments = identifier.split("|")
    if len(fragments) < 8:
        raise ScreenInputError(f"malformed candidate identifier: {identifier}")
    labels = {fragment.split("=", 1)[0]: fragment.split("=", 1)[1] for fragment in fragments[1:] if "=" in fragment}
    try:
        pair = None if labels["pair"] == "none" else tuple(labels["pair"].split("-", 1))
        if pair is not None and len(pair) != 2:
            raise ScreenInputError(f"malformed pair in candidate identifier: {identifier}")
        return Candidate(Family(fragments[0]), int(labels["lb"]), int(labels["sec"]), int(labels["k"]), int(labels["dir"]), pair, float(labels["z"]), int(labels["reb"]), float(labels.get("liq", "0")))
    except (KeyError, ValueError) as error:
        raise ScreenInputError(f"malformed candidate identifier: {identifier}") from error


def parse_overlay_identifier(identifier: str) -> RiskOverlay:
    """Parse the deterministic risk-overlay identifier emitted by the broad search."""

    labels = {fragment.split("=", 1)[0]: fragment.split("=", 1)[1] for fragment in identifier.split("|") if "=" in fragment}
    try:
        return RiskOverlay(float(labels["L"]), float(labels["tv"]), int(labels["vw"]))
    except (KeyError, ValueError) as error:
        raise ScreenInputError(f"malformed overlay identifier: {identifier}") from error


def has_promotable_oos_duration(days: float) -> bool:
    """Require 90 calendar days even when a chart interval has few bars."""

    return days >= 90.0


def _to_market_data(candles: dict[str, tuple[Candle, ...]]) -> MarketData:
    """Align Bitget closed candles without filling missing observations."""

    by_symbol = {symbol[:-4]: {candle.timestamp_ms: candle for candle in rows} for symbol, rows in candles.items()}
    symbols = tuple(sorted(by_symbol))
    common = set(by_symbol[symbols[0]])
    for rows in tuple(by_symbol.values())[1:]:
        common.intersection_update(rows)
    stamps = tuple(sorted(common))
    if len(stamps) < 20:
        raise ScreenInputError("fewer than 20 common closed Bitget candles")
    dates = tuple(datetime.fromtimestamp(stamp / 1000.0, tz=UTC) for stamp in stamps)
    def matrix(field: str) -> np.ndarray:
        return np.asarray([[getattr(by_symbol[symbol][stamp], field) for symbol in symbols] for stamp in stamps], dtype=float)
    return MarketData(dates, symbols, matrix("open_price"), matrix("high_price"), matrix("low_price"), matrix("close_price"), matrix("quote_volume_usdt"))


def _rescaled_overlay(overlay: RiskOverlay, interval: str) -> RiskOverlay:
    """Scale daily risk target by square root of interval duration, not leverage cap."""

    target = overlay.target_daily_vol * math.sqrt(INTERVAL_HOURS[interval] / 24.0)
    return RiskOverlay(overlay.max_leverage, target, overlay.volatility_window)


def _rescaled_candidate(candidate: Candidate, interval: str) -> Candidate:
    """Convert a daily liquidity floor to the quote turnover of one closed screen bar."""

    per_bar_floor = candidate.minimum_daily_quote_volume * INTERVAL_HOURS[interval] / 24.0
    return replace(candidate, minimum_daily_quote_volume=per_bar_floor)


def market_gate_for_active_symbols(active_symbols: tuple[str, ...], candidate: Candidate, markets: Mapping[str, MarketSnapshot]) -> bool:
    """Require current liquidity only for the assets a candidate is actually signaling."""

    required_volume = max(MIN_QUOTE_VOLUME, candidate.minimum_daily_quote_volume)
    return all(
        (snapshot := markets.get(f"{symbol}USDT")) is not None
        and snapshot.quote_volume_usdt >= required_volume
        and snapshot.spread_fraction <= MAX_SPREAD
        for symbol in active_symbols
    )


def _screen_one(data: MarketData, interval: str, candidate: Candidate, overlay: RiskOverlay, markets: Mapping[str, MarketSnapshot]) -> ScreenResult:
    """Evaluate an independently selected candidate over the last 40% of one interval."""

    split = int(len(data.dates) * 0.6)
    positions = build_positions(data, _rescaled_candidate(candidate, interval))
    active_symbols = tuple(symbol for symbol, weight in zip(data.symbols, positions[-1], strict=True) if abs(float(weight)) > 0.0)
    market_gate = market_gate_for_active_symbols(active_symbols, candidate, markets)
    path = evaluate_path(data, positions, _rescaled_overlay(overlay, interval))
    test, stress, liquidated = segment_metrics(path, split, len(data.dates))
    oos_days = (data.dates[-1] - data.dates[split]).total_seconds() / 86_400.0
    min_observations = max(30, int(math.ceil(90.0 / (INTERVAL_HOURS[interval] / 24.0))))
    passed = market_gate and has_promotable_oos_duration(oos_days) and test.observations >= min_observations and not liquidated and test.net_return > 0.0 and stress.net_return > 0.0 and test.max_drawdown <= 0.25 and test.positive_blocks >= 3
    return {"candidate": candidate.identifier, "risk": overlay.identifier, "max_leverage": overlay.max_leverage, "max_realized_leverage": float(path.leverage.max(initial=0.0)), "active_symbols": list(active_symbols), "oos_days": oos_days, "observations": test.observations, "market_gate": market_gate, "test": {"net_return": test.net_return, "max_drawdown": test.max_drawdown, "sharpe": test.sharpe, "positive_blocks": test.positive_blocks}, "stress_return": stress.net_return, "simulated_liquidation": liquidated, "exchange_screen_pass": passed, "promotion": "paper-only" if passed else "screen-only-or-rejected"}


def _load_winners() -> tuple[WinnerRow, ...]:
    """Parse the validation-only winner list from the research report."""

    try:
        report = BroadSearchReport.model_validate_json(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ScreenInputError(f"cannot load broad-search report: {error}") from error
    if not report.validation_winners:
        raise ScreenInputError("broad search produced no validation winners to screen")
    return report.validation_winners


def main() -> int:
    """Fetch public candles and write an all-timeframe exchange-screen report."""

    winners = _load_winners()
    parsed = tuple((parse_candidate_identifier(row.candidate), parse_overlay_identifier(row.risk)) for row in winners)
    now_ms = int(time.time() * 1000)
    interval_results: list[IntervalScreenResult] = []
    with create_client() as client:
        markets = fetch_markets(client)
        for interval in INTERVALS:
            print(f"BITGET_SCREEN interval={interval} candidates={len(parsed)}", flush=True)
            try:
                candles = {symbol: fetch_candles(client, symbol, interval, now_ms) for symbol in ALT_UNIVERSE}
                data = _to_market_data(candles)
                results = [_screen_one(data, interval, candidate, overlay, markets) for candidate, overlay in parsed]
                interval_results.append({"interval": interval, "common_bars": len(data.dates), "observation_days": (data.dates[-1] - data.dates[0]).total_seconds() / 86_400.0, "results": results})
            except (BitgetRequestError, ScreenInputError) as error:
                interval_results.append({"interval": interval, "error": str(error), "results": []})
    report = {"mode": "public-api-paper-only", "observed_at": datetime.now(UTC).isoformat(), "selection_source": str(REPORT_PATH), "all_intervals_considered": list(INTERVALS), "results": interval_results, "live_promotable": []}
    SCREEN_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    passes = sum(int(item["exchange_screen_pass"]) for interval in interval_results for item in interval.get("results", []))
    print(f"BITGET_SCREEN_DONE report={SCREEN_PATH} paper_screen_passes={passes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
