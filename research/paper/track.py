#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "requests"]
# ///

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TypeVar

import pandas as pd  # noqa: PANDAS_OK
import requests

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.paper.ledger import LedgerEntry, Position, append_entries, latest_entries, read_entries, settle_entry
from research.paper.market_data import LiveSnapshot, collect_live_snapshot, current_funding_rates
from research.paper.status import render_failure_status, render_status
from research.wave1.common import PipelineError
from research.wave1.fam_funding import F1_CANDIDATES, FundingCandidate, carry_position, funding_score
from research.wave2.funding import W2_FUNDING_CANDIDATES
from research.wave3.engine import CandidateConfig, Target, W3_CANDIDATES, current_targets

BASE_DIR = Path(__file__).resolve().parent
LEDGER_PATH = BASE_DIR / "ledger" / "paper_ledger.jsonl"
STATUS_PATH = BASE_DIR / "STATUS.md"
TRACKED_IDS = ("W2c", "F1e", "W3c", "W3d")
CandidateT = TypeVar("CandidateT")


def _candidate(candidates: tuple[CandidateT, ...], candidate_id: str) -> CandidateT:
    return next(candidate for candidate in candidates if getattr(candidate, "candidate_id") == candidate_id)


def _carry_signal(snapshot: LiveSnapshot, candidate: FundingCandidate) -> tuple[tuple[str, float], ...]:
    scored: list[tuple[str, float]] = []
    for symbol, series in snapshot.funding_series.items():
        if candidate.majors_only and symbol not in {"BTCUSDT", "ETHUSDT"}:
            continue
        score = funding_score(series, candidate.window_days)
        active = carry_position(score, candidate)
        if score.empty or active.empty or float(active.iloc[-1]) <= 0.0 or pd.isna(score.iloc[-1]):
            continue
        scored.append((symbol, float(score.iloc[-1])))
    return tuple(sorted(scored, key=lambda item: (-item[1], item[0]))[: candidate.top_k])


def _carry_positions(snapshot: LiveSnapshot, candidate: FundingCandidate, equity: float) -> tuple[Position, ...]:
    selected = _carry_signal(snapshot, candidate)
    if not selected:
        return ()
    notional = equity / len(selected)
    positions: list[Position] = []
    for symbol, _score in selected:
        perp_price = snapshot.perp_prices.get(symbol)
        spot_price = snapshot.spot_prices.get(symbol)
        if perp_price is None or spot_price is None:
            continue
        positions.extend((
            Position(symbol, "spot", "long", 1.0, notional / spot_price, spot_price, spot_price, notional, 0.0),
            Position(symbol, "perp", "short", -1.0, notional / perp_price, perp_price, perp_price, notional, snapshot.funding_rates.get(symbol, 0.0)),
        ))
    return tuple(positions)


def _momentum_targets(snapshot: LiveSnapshot, candidate: CandidateConfig, observed_at: pd.Timestamp) -> tuple[Target, ...]:
    return current_targets(snapshot.wave3_markets, candidate, observed_at)


def _momentum_positions(snapshot: LiveSnapshot, candidate: CandidateConfig, equity: float, observed_at: pd.Timestamp) -> tuple[Position, ...]:
    targets = _momentum_targets(snapshot, candidate, observed_at)
    positions: list[Position] = []
    for target in targets:
        price = snapshot.perp_prices.get(target.signal.symbol)
        if price is None:
            continue
        notional = equity * target.weight
        side = "long" if target.direction > 0.0 else "short"
        positions.append(Position(target.signal.symbol, "perp", side, target.direction, notional / price, price, price, notional, snapshot.funding_rates.get(target.signal.symbol, 0.0)))
    return tuple(positions)


def _signal_and_positions(snapshot: LiveSnapshot, candidate_id: str, equity: float, observed_at: pd.Timestamp) -> tuple[str, tuple[Position, ...]]:
    if candidate_id in {"W2c", "F1e"}:
        candidate = _candidate(W2_FUNDING_CANDIDATES if candidate_id == "W2c" else F1_CANDIDATES, candidate_id)
        positions = _carry_positions(snapshot, candidate, equity)
        symbols = ", ".join(sorted({position.symbol for position in positions})) or "cash"
        return (f"{candidate_id} carry selected: {symbols}", positions)
    candidate = _candidate(W3_CANDIDATES, candidate_id)
    positions = _momentum_positions(snapshot, candidate, equity, observed_at)
    symbols = ", ".join(f"{position.symbol}:{position.side}" for position in positions) or "cash"
    return (f"{candidate_id} weekly momentum targets: {symbols}", positions)


def _update_funding_rates(snapshot: LiveSnapshot, planned: dict[str, tuple[Position, ...]]) -> dict[str, float]:
    symbols = tuple(sorted({position.symbol for positions in planned.values() for position in positions if position.instrument == "perp" and position.symbol not in snapshot.funding_rates}))
    if not symbols:
        return snapshot.funding_rates
    return {**snapshot.funding_rates, **current_funding_rates(symbols, snapshot.wave3_markets)}


def run_once() -> int:
    snapshot = collect_live_snapshot()
    entries = read_entries(LEDGER_PATH)
    latest = latest_entries(entries)
    planned: dict[str, tuple[Position, ...]] = {}
    signals: dict[str, str] = {}
    for candidate_id in TRACKED_IDS:
        previous = latest.get(candidate_id)
        equity = previous.virtual_equity if previous is not None else 300.0
        signal, positions = _signal_and_positions(snapshot, candidate_id, equity, snapshot.observed_at)
        planned[candidate_id] = positions
        signals[candidate_id] = signal
    funding_rates = _update_funding_rates(snapshot, planned)
    planned = {
        candidate_id: tuple(position if position.funding_rate == funding_rates.get(position.symbol, position.funding_rate) else Position(position.symbol, position.instrument, position.side, position.direction, position.quantity, position.entry_price, position.mark_price, position.notional_usdt, funding_rates.get(position.symbol, position.funding_rate)) for position in positions)
        for candidate_id, positions in planned.items()
    }
    next_entries: list[LedgerEntry] = []
    for candidate_id in TRACKED_IDS:
        next_entries.append(settle_entry(latest.get(candidate_id), planned[candidate_id], snapshot.observed_at.isoformat(), funding_rates, candidate_id, signals[candidate_id], snapshot.source_names))
    added = append_entries(LEDGER_PATH, tuple(next_entries))
    final_entries = read_entries(LEDGER_PATH)
    render_status(final_entries, STATUS_PATH)
    print(f"paper: appended {added} daily records; status={STATUS_PATH}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only paper forward-validation tracker")
    parser.add_argument("--run-once", action="store_true", help="fetch public data, mark virtual positions, and append one daily ledger record")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_once:
        build_parser().print_help()
        return 2
    try:
        return run_once()
    except (PipelineError, requests.RequestException) as error:
        render_failure_status(STATUS_PATH, pd.Timestamp.now(tz="UTC").isoformat(), str(error))
        print(f"paper: live data unavailable: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
