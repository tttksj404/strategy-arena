# Command-line orchestration for Wave-1 fetch, run, gate, and report stages.

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
import sys
from typing import assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import CACHE_DIR, REPORT_DIR, RESULTS_DIR, JsonValue, PipelineError, StrategyResult, close_correlation, ensure_output_dirs, integrity_report, load_frame, load_json, report_payload, save_json, strategy_payload, validate_symbol
from research.wave1.fam_funding import F1_CANDIDATES, FundingCandidate, neighbor_candidates as funding_neighbors, run_cached as run_funding
from research.wave1.fam_session import F3_CANDIDATES, SessionCandidate, SessionKind, run_cached as run_session
from research.wave1.fam_tsmom import F2_CANDIDATES, TsmomCandidate, neighbor_candidates as tsmom_neighbors, run_cached as run_tsmom
from research.wave1.fetch_binance import BinanceFundingRequest, BinanceKlineRequest, cached_frame, exchange_symbols, fetch_exchange_info, fetch_funding, fetch_klines, fetch_quote_volumes, fetch_spot_exchange_info, quote_volumes
from research.wave1.fetch_bitget import BitgetCandleRequest, YahooDailyRequest, contract_symbols, fetch_candles, fetch_contracts, fetch_funding as fetch_bitget_funding, fetch_yahoo_daily
from research.wave1.gate_reporting import ReportPaths, evaluate_result_file, write_reports, write_summary
from research.wave1.gates import MetricInput, calculate_metrics


Candidate = FundingCandidate | TsmomCandidate | SessionCandidate
ALL_CANDIDATES = (*F1_CANDIDATES, *F2_CANDIDATES, *F3_CANDIDATES)
ALL_IDS = tuple(candidate.candidate_id for candidate in ALL_CANDIDATES)
END_MS = int(pd.Timestamp("2026-07-15T00:00:00Z").timestamp() * 1000)
START_MS = int(pd.Timestamp("2019-09-01T00:00:00Z").timestamp() * 1000)


class Stage(StrEnum):
    FETCH = "fetch"
    RUN = "run"
    GATES = "gates"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class FetchContext:
    session: requests.Session
    force: bool
    start_ms: int = START_MS
    end_ms: int = END_MS


def _cache_binance_symbol(symbol: str, context: FetchContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol = validate_symbol(symbol)
    funding = cached_frame(
        CACHE_DIR / f"binance_funding_{symbol}.csv.gz",
        context.force,
        lambda: fetch_funding(BinanceFundingRequest(symbol, context.start_ms, context.end_ms), context.session),
    )
    perp = cached_frame(
        CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz",
        context.force,
        lambda: fetch_klines(BinanceKlineRequest(symbol, "1d", context.start_ms, context.end_ms), context.session),
    )
    perp_4h = cached_frame(
        CACHE_DIR / f"binance_fapi_{symbol}_4h.csv.gz",
        context.force,
        lambda: fetch_klines(BinanceKlineRequest(symbol, "4h", context.start_ms, context.end_ms), context.session),
    )
    spot = cached_frame(
        CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz",
        context.force,
        lambda: fetch_klines(BinanceKlineRequest(symbol, "1d", context.start_ms, context.end_ms, "spot"), context.session),
    )
    return funding, perp, perp_4h, spot


def _cache_bitget_candles(symbol: str, granularity: str, context: FetchContext) -> pd.DataFrame:
    symbol = validate_symbol(symbol)
    path = CACHE_DIR / f"bitget_{symbol}_{granularity}.csv.gz"
    return cached_frame(
        path,
        context.force,
        lambda: fetch_candles(BitgetCandleRequest(symbol, granularity, context.start_ms, context.end_ms), context.session),
    )


def _stage_fetch(force: bool) -> None:
    ensure_output_dirs()
    universe_path = CACHE_DIR / "universe.json"
    with requests.Session() as session:
        context = FetchContext(session, force)
        futures = exchange_symbols(fetch_exchange_info(session))
        spot = exchange_symbols(fetch_spot_exchange_info(session))
        bitget = contract_symbols(fetch_contracts(session))
        volumes = quote_volumes(fetch_quote_volumes(session))
        ranked = sorted(futures & spot & bitget, key=lambda symbol: volumes.get(symbol, 0.0), reverse=True)
        valid_symbols: list[str] = []
        integrity: dict[str, bool] = {}
        source_integrity: dict[str, bool] = {}
        integrity_reports: dict[str, JsonValue] = {}
        for count, symbol in enumerate(ranked, start=1):
            funding, perp, perp_4h, spot_frame = _cache_binance_symbol(symbol, context)
            has_history = not funding.empty and funding.index.min() <= pd.Timestamp("2024-07-15T00:00:00Z")
            bitget_daily = _cache_bitget_candles(symbol, "1D", context)
            reports = {
                f"binance_funding_{symbol}": integrity_report(funding, timedelta(hours=8)),
                f"binance_fapi_{symbol}_1d": integrity_report(perp, timedelta(days=1)),
                f"binance_fapi_{symbol}_4h": integrity_report(perp_4h, timedelta(hours=4)),
                f"binance_spot_{symbol}_1d": integrity_report(spot_frame, timedelta(days=1)),
                f"bitget_{symbol}_1D": integrity_report(bitget_daily, timedelta(days=1)),
            }
            # Bitget daily has exchange-side single-day holes at quarter boundaries; it is a cross-validation
            # series, so eligibility gates on Binance research sources only (Bitget must exist + correlate).
            binance_names = [name for name in reports if name.startswith("binance_")]
            reports_pass = all(reports[name].valid for name in binance_names) and not any(frame.empty for frame in (funding, perp, perp_4h, spot_frame, bitget_daily))
            integrity_reports.update({name: report_payload(report) for name, report in reports.items()})
            correlation = close_correlation(perp, bitget_daily)
            passes = has_history and reports_pass and pd.notna(correlation) and correlation > 0.99
            integrity[symbol] = bool(passes)
            if passes:
                valid_symbols.append(symbol)
            if count % 10 == 0:
                print(f"fetch: checked {count} symbols")
            if len(valid_symbols) == 40:
                break
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            funding, perp, perp_4h, _ = _cache_binance_symbol(symbol, context)
            bitget_daily = _cache_bitget_candles(symbol, "1D", context)
            checks = {
                f"binance_fapi_{symbol}_1d": (perp, timedelta(days=1)),
                f"binance_fapi_{symbol}_4h": (perp_4h, timedelta(hours=4)),
                f"binance_funding_{symbol}": (funding, timedelta(hours=8)),
                f"bitget_{symbol}_1D": (bitget_daily, timedelta(days=1)),
            }
            for name, (frame, interval) in checks.items():
                report = integrity_report(frame, interval)
                source_integrity[name] = not frame.empty and report.valid
                integrity_reports[name] = report_payload(report)
            source_integrity[f"correlation_{symbol}"] = bool(pd.notna(close_correlation(perp, bitget_daily)) and close_correlation(perp, bitget_daily) > 0.99)
        session_symbols = ("SPYUSDT", "QQQUSDT", "TSLAUSDT", "NVDAUSDT", "MSTRUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT")
        for count, symbol in enumerate(session_symbols, start=1):
            candles = _cache_bitget_candles(symbol, "1H", context)
            candle_report = integrity_report(candles, timedelta(hours=1))
            source_integrity[f"bitget_{symbol}_1H"] = not candles.empty and candle_report.valid
            integrity_reports[f"bitget_{symbol}_1H"] = report_payload(candle_report)
            funding_path = CACHE_DIR / f"bitget_funding_{symbol}.csv.gz"
            funding = cached_frame(funding_path, force, lambda symbol=symbol: fetch_bitget_funding(symbol, session))
            funding_report = integrity_report(funding, timedelta(hours=8))
            source_integrity[f"bitget_funding_{symbol}"] = not funding.empty and funding_report.valid
            integrity_reports[f"bitget_funding_{symbol}"] = report_payload(funding_report)
            if count % 10 == 0:
                print(f"fetch: cached {count} session symbols")
        yahoo_start = int(pd.Timestamp("2005-01-01T00:00:00Z").timestamp())
        yahoo_end = int(pd.Timestamp("2026-07-15T00:00:00Z").timestamp())
        for symbol in ("SPY", "QQQ"):
            yahoo = cached_frame(
                CACHE_DIR / f"yahoo_{symbol}_1d.csv.gz",
                force,
                lambda symbol=symbol: fetch_yahoo_daily(YahooDailyRequest(symbol, yahoo_start, yahoo_end), session),
            )
            yahoo_report = integrity_report(yahoo, timedelta(days=1))
            source_integrity[f"yahoo_{symbol}_1d"] = not yahoo.empty and yahoo_report.valid
            integrity_reports[f"yahoo_{symbol}_1d"] = report_payload(yahoo_report)
    save_json(universe_path, {"symbols": valid_symbols, "integrity": integrity, "source_integrity": source_integrity, "integrity_reports": integrity_reports, "frozen_end": "2026-07-14"})


def _universe() -> tuple[tuple[str, ...], dict[str, bool], dict[str, bool]]:
    payload = load_json(CACHE_DIR / "universe.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise PipelineError("cache/universe.json is invalid")
    symbols = tuple(validate_symbol(str(symbol)) for symbol in payload["symbols"])
    integrity = payload.get("integrity")
    source_integrity = payload.get("source_integrity")
    universe_checks = {symbol: integrity.get(symbol) is True for symbol in symbols} if isinstance(integrity, dict) else {}
    source_checks = {str(name): value is True for name, value in source_integrity.items()} if isinstance(source_integrity, dict) else {}
    return symbols, universe_checks, source_checks


def _candidate_data_valid(candidate: Candidate, universe_checks: dict[str, bool], source_checks: dict[str, bool]) -> bool:
    match candidate:
        case FundingCandidate():
            return bool(universe_checks) and all(universe_checks.values())
        case TsmomCandidate():
            return all(
                source_checks.get(name) is True
                for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
                for name in (f"binance_fapi_{symbol}_1d", f"binance_fapi_{symbol}_4h", f"binance_funding_{symbol}", f"bitget_{symbol}_1D", f"bitget_funding_{symbol}", f"correlation_{symbol}")
            )
        case SessionCandidate():
            required: dict[SessionKind, tuple[str, ...]] = {
                F3_CANDIDATES[0].hypothesis: ("yahoo_SPY_1d", "yahoo_QQQ_1d", "bitget_SPYUSDT_1H", "bitget_QQQUSDT_1H"),
                F3_CANDIDATES[1].hypothesis: ("bitget_SPYUSDT_1H", "bitget_BTCUSDT_1H"),
                F3_CANDIDATES[2].hypothesis: tuple(f"bitget_funding_{symbol}USDT" for symbol in ("SPY", "QQQ", "TSLA", "NVDA", "MSTR")),
            }
            return all(source_checks.get(name) is True for name in required[candidate.hypothesis])
        case unreachable:
            assert_never(unreachable)


def _run_candidate(candidate: Candidate, symbols: tuple[str, ...]) -> StrategyResult:
    match candidate:
        case FundingCandidate():
            return run_funding(CACHE_DIR, symbols, candidate)
        case TsmomCandidate():
            return run_tsmom(CACHE_DIR, candidate)
        case SessionCandidate():
            return run_session(CACHE_DIR, candidate)
        case unreachable:
            assert_never(unreachable)


def _neighbor_sharpes(candidate: Candidate, symbols: tuple[str, ...]) -> list[float]:
    match candidate:
        case FundingCandidate():
            neighbors: tuple[Candidate, ...] = funding_neighbors(candidate)
        case TsmomCandidate():
            neighbors = tsmom_neighbors(candidate)
        case SessionCandidate():
            return []
        case unreachable:
            assert_never(unreachable)
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    sharpes: list[float] = []
    for neighbor in neighbors:
        result = _run_candidate(neighbor, symbols)
        is_equity = result.equity[result.equity.index <= split]
        is_trades = tuple(float(value) for value in result.trade_returns[result.trade_returns.index <= split])
        sharpes.append(calculate_metrics(MetricInput(is_equity, is_trades)).sharpe)
    return sharpes


def _stage_run(only: str | None) -> None:
    ensure_output_dirs()
    symbols, universe_checks, source_checks = _universe()
    candidates = tuple(candidate for candidate in ALL_CANDIDATES if only is None or candidate.candidate_id == only)
    if not candidates:
        raise PipelineError(f"unknown candidate: {only}")
    for count, candidate in enumerate(candidates, start=1):
        result = _run_candidate(candidate, symbols)
        data_valid = _candidate_data_valid(candidate, universe_checks, source_checks)
        metadata = {**result.metadata, "data_valid": data_valid, "neighbor_is_sharpes": _neighbor_sharpes(candidate, symbols)}
        save_json(RESULTS_DIR / f"{candidate.candidate_id}.json", strategy_payload(replace(result, metadata=metadata)))
        if count % 10 == 0 or count == len(candidates):
            print(f"run: completed {count}/{len(candidates)} candidates")


def _stage_gates(only: str | None) -> None:
    btc_path = CACHE_DIR / "binance_fapi_BTCUSDT_1d.csv.gz"
    btc_returns = load_frame(btc_path)["close"].pct_change() if btc_path.exists() else pd.Series(dtype=float)
    selected = tuple(candidate_id for candidate_id in ALL_IDS if only is None or candidate_id == only)
    summaries: dict[str, tuple] = {}
    for candidate_id in selected:
        path = RESULTS_DIR / f"{candidate_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summaries[candidate_id] = evaluate_result_file(path, btc_returns)
    write_summary(RESULTS_DIR / "gates_summary.md", summaries)


def _stage_report() -> None:
    paths = ReportPaths(RESULTS_DIR, Path(__file__).resolve().parent / "REGISTRY.md", REPORT_DIR / "wave1_report.md")
    write_reports(paths, ALL_IDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-1 preregistered research pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    parser.add_argument("--only", choices=ALL_IDS)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match args.stage:
        case Stage.FETCH:
            _stage_fetch(args.force)
        case Stage.RUN:
            _stage_run(args.only)
        case Stage.GATES:
            _stage_gates(args.only)
        case Stage.REPORT:
            _stage_report()
        case unreachable:
            assert_never(unreachable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
