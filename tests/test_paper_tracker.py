from __future__ import annotations

from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK

from research.paper import track
from research.paper.ledger import LedgerEntry, Position, append_entries, read_entries, settle_entry
from research.paper.market_data import LiveSnapshot
from research.wave3.engine import AssetMarket
from research.wave3.universe import AssetListing


def _position(mark_price: float = 100.0) -> Position:
    return Position("BTCUSDT", "perp", "short", -1.0, 1.0, mark_price, mark_price, 100.0, 0.001)


def _entry(observed_at: str, position: Position) -> LedgerEntry:
    return settle_entry(None, (position,), observed_at, {"BTCUSDT": 0.001}, "F1e", "synthetic", ("synthetic",), maker_fee_rate=0.0)


def test_ledger_append_writes_jsonl_record(tmp_path: Path) -> None:
    entry = _entry("2026-07-15T00:00:00+00:00", _position())
    path = tmp_path / "paper_ledger.jsonl"

    added = append_entries(path, (entry,))

    assert added == 1
    assert len(read_entries(path)) == 1
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_ledger_append_is_idempotent_for_same_candidate_and_day(tmp_path: Path) -> None:
    first = _entry("2026-07-15T00:00:00+00:00", _position())
    replacement = _entry("2026-07-15T23:00:00+00:00", _position(101.0))
    path = tmp_path / "paper_ledger.jsonl"

    assert append_entries(path, (first,)) == 1
    assert append_entries(path, (replacement,)) == 0

    entries = read_entries(path)
    assert len(entries) == 1
    assert entries[0].observed_at == first.observed_at


def test_funding_accrues_to_short_perp_when_rate_is_positive() -> None:
    previous = _entry("2026-07-15T00:00:00+00:00", _position())
    current_position = Position("BTCUSDT", "perp", "short", -1.0, 1.0, 100.0, 100.0, 100.0, 0.001)

    current = settle_entry(previous, (current_position,), "2026-07-15T08:00:00+00:00", {"BTCUSDT": 0.001}, "F1e", "synthetic", ("synthetic",), maker_fee_rate=0.0)

    assert current.funding_delta == 0.1
    assert current.cumulative_funding == 0.1
    assert current.virtual_equity == previous.virtual_equity + 0.1


def test_run_once_updates_all_four_candidates_without_orders(tmp_path: Path, monkeypatch) -> None:
    dates = pd.date_range("2025-12-01", periods=230, freq="D", tz="UTC")
    funding_dates = pd.date_range("2026-07-09", periods=24, freq="8h", tz="UTC")
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT")
    markets = {}
    funding_series = {}
    prices = {}
    for offset, symbol in enumerate(symbols):
        values = 100.0 + (dates.dayofyear.to_numpy() * (offset + 1) * 0.1) + ((dates.dayofyear.to_numpy() % 5) * (offset + 1))
        bars = pd.DataFrame({"close": values, "quote_volume": 1_000_000.0 + offset}, index=dates)
        funding = pd.Series(0.0002 + offset * 0.00001, index=funding_dates)
        listing = AssetListing(symbol, "crypto", pd.Timestamp("2024-01-01", tz="UTC"), True, True)
        markets[symbol] = AssetMarket(listing, bars, None, funding)
        funding_series[symbol] = funding
        prices[symbol] = float(values[-1])
    snapshot = LiveSnapshot(
        pd.Timestamp("2026-07-16T12:00:00Z"),
        funding_series,
        {symbol: float(series.iloc[-1]) for symbol, series in funding_series.items()},
        prices,
        prices,
        markets,
        ("synthetic",),
    )
    monkeypatch.setattr(track, "collect_live_snapshot", lambda: snapshot)
    monkeypatch.setattr(track, "LEDGER_PATH", tmp_path / "ledger" / "paper_ledger.jsonl")
    monkeypatch.setattr(track, "STATUS_PATH", tmp_path / "STATUS.md")

    assert track.run_once() == 0
    assert track.run_once() == 0
    entries = read_entries(track.LEDGER_PATH)
    assert len(entries) == 4
    assert {entry.candidate_id for entry in entries} == {"W2c", "F1e", "W3c", "W3d"}
    assert "실주문" in track.STATUS_PATH.read_text(encoding="utf-8")
