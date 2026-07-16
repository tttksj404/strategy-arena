from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Final, Mapping

import pandas as pd

from research.wave1.common import JsonValue, PipelineError

INITIAL_EQUITY: Final = 300.0
MAKER_FEE_RATE: Final = 0.0002
FUNDING_INTERVAL_HOURS: Final = 8.0


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    instrument: str
    side: str
    direction: float
    quantity: float
    entry_price: float
    mark_price: float
    notional_usdt: float
    funding_rate: float


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    candidate_id: str
    run_date: str
    observed_at: str
    virtual_equity: float
    pnl_delta: float
    funding_delta: float
    cumulative_funding: float
    maker_fees: float
    positions: tuple[Position, ...]
    signal: str
    source_names: tuple[str, ...]


def position_payload(position: Position) -> JsonValue:
    return {
        "symbol": position.symbol,
        "instrument": position.instrument,
        "side": position.side,
        "direction": position.direction,
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "mark_price": position.mark_price,
        "notional_usdt": position.notional_usdt,
        "funding_rate": position.funding_rate,
    }


def entry_payload(entry: LedgerEntry) -> JsonValue:
    return {
        "candidate_id": entry.candidate_id,
        "run_date": entry.run_date,
        "observed_at": entry.observed_at,
        "virtual_equity": entry.virtual_equity,
        "pnl_delta": entry.pnl_delta,
        "funding_delta": entry.funding_delta,
        "cumulative_funding": entry.cumulative_funding,
        "maker_fees": entry.maker_fees,
        "positions": [position_payload(position) for position in entry.positions],
        "signal": entry.signal,
        "source_names": list(entry.source_names),
    }


def _number(value: JsonValue, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"paper ledger field {field} is not numeric")
    return float(value)


def _text(value: JsonValue, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PipelineError(f"paper ledger field {field} is not text")
    return value


def _parse_position(value: JsonValue) -> Position:
    if not isinstance(value, dict):
        raise PipelineError("paper ledger position is not an object")
    return Position(
        _text(value.get("symbol"), "symbol"),
        _text(value.get("instrument"), "instrument"),
        _text(value.get("side"), "side"),
        _number(value.get("direction"), "direction"),
        _number(value.get("quantity"), "quantity"),
        _number(value.get("entry_price"), "entry_price"),
        _number(value.get("mark_price"), "mark_price"),
        _number(value.get("notional_usdt"), "notional_usdt"),
        _number(value.get("funding_rate"), "funding_rate"),
    )


def _parse_entry(value: JsonValue) -> LedgerEntry:
    if not isinstance(value, dict) or not isinstance(value.get("positions"), list):
        raise PipelineError("paper ledger entry is invalid")
    raw_sources = value.get("source_names", [])
    if not isinstance(raw_sources, list) or not all(isinstance(item, str) for item in raw_sources):
        raise PipelineError("paper ledger sources are invalid")
    return LedgerEntry(
        _text(value.get("candidate_id"), "candidate_id"),
        _text(value.get("run_date"), "run_date"),
        _text(value.get("observed_at"), "observed_at"),
        _number(value.get("virtual_equity"), "virtual_equity"),
        _number(value.get("pnl_delta"), "pnl_delta"),
        _number(value.get("funding_delta"), "funding_delta"),
        _number(value.get("cumulative_funding"), "cumulative_funding"),
        _number(value.get("maker_fees"), "maker_fees"),
        tuple(_parse_position(item) for item in value["positions"]),
        _text(value.get("signal"), "signal"),
        tuple(raw_sources),
    )


def read_entries(path: Path) -> tuple[LedgerEntry, ...]:
    if not path.exists():
        return ()
    entries: list[LedgerEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(_parse_entry(json.loads(line)))
    return tuple(entries)


def append_entries(path: Path, entries: tuple[LedgerEntry, ...]) -> int:
    existing = read_entries(path)
    keys = {(entry.candidate_id, entry.run_date) for entry in existing}
    pending = tuple(entry for entry in entries if (entry.candidate_id, entry.run_date) not in keys)
    if pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            for entry in pending:
                stream.write(json.dumps(entry_payload(entry), ensure_ascii=False, allow_nan=False, separators=(",", ":")))
                stream.write("\n")
    return len(pending)


def latest_entries(entries: tuple[LedgerEntry, ...]) -> Mapping[str, LedgerEntry]:
    latest: dict[str, LedgerEntry] = {}
    for entry in entries:
        current = latest.get(entry.candidate_id)
        if current is None or entry.observed_at > current.observed_at:
            latest[entry.candidate_id] = entry
    return latest


def _position_key(position: Position) -> tuple[str, str, float]:
    return (position.symbol, position.instrument, position.direction)


def settle_entry(
    previous: LedgerEntry | None,
    current_positions: tuple[Position, ...],
    observed_at: str,
    funding_rates: dict[str, float],
    candidate_id: str,
    signal: str,
    source_names: tuple[str, ...],
    maker_fee_rate: float = MAKER_FEE_RATE,
) -> LedgerEntry:
    previous_positions = { _position_key(item): item for item in previous.positions } if previous is not None else {}
    settled: list[Position] = []
    for current in current_positions:
        old = previous_positions.get(_position_key(current))
        settled.append(replace(current, entry_price=old.entry_price if old is not None else current.mark_price))
    current_map = {_position_key(item): item for item in settled}
    base_equity = previous.virtual_equity if previous is not None else INITIAL_EQUITY
    price_pnl = 0.0
    funding_delta = 0.0
    maker_fees = 0.0
    if previous is not None:
        elapsed = max(0.0, (pd_timestamp(observed_at) - pd_timestamp(previous.observed_at)).total_seconds() / 3600.0)
        for key, old in previous_positions.items():
            current = current_map.get(key)
            if current is not None:
                price_pnl += old.direction * old.quantity * (current.mark_price - old.mark_price)
                if old.instrument == "perp":
                    rate = funding_rates.get(old.symbol, old.funding_rate)
                    funding_delta += -old.direction * old.notional_usdt * rate * elapsed / FUNDING_INTERVAL_HOURS
            else:
                maker_fees += old.notional_usdt * maker_fee_rate
        for key, current in current_map.items():
            if key not in previous_positions:
                maker_fees += current.notional_usdt * maker_fee_rate
    else:
        maker_fees = sum(position.notional_usdt * maker_fee_rate for position in settled)
    if previous is not None:
        for key, current in current_map.items():
            old = previous_positions.get(key)
            if old is not None and abs(current.notional_usdt - old.notional_usdt) > 1e-12:
                maker_fees += abs(current.notional_usdt - old.notional_usdt) * maker_fee_rate
    pnl_delta = price_pnl + funding_delta - maker_fees
    cumulative = (previous.cumulative_funding if previous is not None else 0.0) + funding_delta
    timestamp = pd_timestamp(observed_at)
    return LedgerEntry(
        candidate_id,
        timestamp.date().isoformat(),
        timestamp.isoformat(),
        base_equity + pnl_delta,
        pnl_delta,
        funding_delta,
        cumulative,
        maker_fees,
        tuple(settled),
        signal,
        source_names,
    )


def pd_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


__all__ = ["INITIAL_EQUITY", "LedgerEntry", "MAKER_FEE_RATE", "Position", "append_entries", "entry_payload", "latest_entries", "read_entries", "settle_entry"]
