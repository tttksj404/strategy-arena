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


def _synthetic_snapshot(n_symbols: int) -> LiveSnapshot:
    """Builds a LiveSnapshot with `n_symbols` distinct symbols -- large enough (>=200, the
    biggest declared requirement, W2c's) that every research/paper/fidelity.py check can PASS,
    unlike a handful of hand-picked symbols. Exercises the same "realistic-sized universe"
    condition the live Bitget-ranked collector (research/paper/market_data.py) is meant to
    produce after the universe-coverage fix."""
    dates = pd.date_range("2025-12-01", periods=230, freq="D", tz="UTC")
    funding_dates = pd.date_range("2026-07-09", periods=24, freq="8h", tz="UTC")
    symbols = ("BTCUSDT", "ETHUSDT") + tuple(f"SYM{i:03d}USDT" for i in range(n_symbols - 2))
    markets = {}
    funding_series = {}
    prices = {}
    for offset, symbol in enumerate(symbols):
        values = 100.0 + (dates.dayofyear.to_numpy() * ((offset % 7) + 1) * 0.1) + ((dates.dayofyear.to_numpy() % 5) * ((offset % 7) + 1))
        bars = pd.DataFrame({"close": values, "quote_volume": 1_000_000.0 + offset}, index=dates)
        funding = pd.Series(0.0002 + (offset % 11) * 0.00001, index=funding_dates)
        listing = AssetListing(symbol, "crypto", pd.Timestamp("2024-01-01", tz="UTC"), True, True)
        markets[symbol] = AssetMarket(listing, bars, None, funding)
        funding_series[symbol] = funding
        prices[symbol] = float(values[-1])
    return LiveSnapshot(
        pd.Timestamp("2026-07-16T12:00:00Z"),
        funding_series,
        {symbol: float(series.iloc[-1]) for symbol, series in funding_series.items()},
        prices,
        prices,
        markets,
        ("synthetic",),
    )


def test_run_once_updates_all_tracked_candidates_without_orders(tmp_path: Path, monkeypatch) -> None:
    snapshot = _synthetic_snapshot(200)
    monkeypatch.setattr(track, "collect_live_snapshot", lambda: snapshot)
    monkeypatch.setattr(track, "LEDGER_PATH", tmp_path / "ledger" / "paper_ledger.jsonl")
    monkeypatch.setattr(track, "STATUS_PATH", tmp_path / "STATUS.md")

    assert track.run_once() == 0
    assert track.run_once() == 0
    entries = read_entries(track.LEDGER_PATH)
    assert len(entries) == 5
    assert {entry.candidate_id for entry in entries} == {"W2c", "F1e", "W3c", "W3d", "G1"}
    assert all(entry.fidelity_ok for entry in entries)
    assert "실주문" in track.STATUS_PATH.read_text(encoding="utf-8")


def test_run_once_forces_cash_and_flags_fidelity_when_universe_is_too_small(tmp_path: Path, monkeypatch) -> None:
    """The regression this whole task is about: a snapshot that covers far fewer symbols than a
    candidate's declared spec (research/paper/fidelity.py) must NOT be traded on silently. G1
    (needs top100) and W2c (needs top200) must come back as cash, flagged fidelity_ok=False --
    not a quietly-wrong signal computed from a partial universe."""
    snapshot = _synthetic_snapshot(6)
    monkeypatch.setattr(track, "collect_live_snapshot", lambda: snapshot)
    monkeypatch.setattr(track, "LEDGER_PATH", tmp_path / "ledger" / "paper_ledger.jsonl")
    monkeypatch.setattr(track, "STATUS_PATH", tmp_path / "STATUS.md")

    assert track.run_once() == 0
    entries = {entry.candidate_id: entry for entry in read_entries(track.LEDGER_PATH)}

    assert entries["G1"].fidelity_ok is False
    assert entries["G1"].positions == ()
    assert "FIDELITY_FAIL" in entries["G1"].signal
    assert entries["W2c"].fidelity_ok is False
    assert entries["W2c"].positions == ()
    # F1e is majors_only (BTC/ETH) -- present even in a 6-symbol snapshot, so it stays valid.
    assert entries["F1e"].fidelity_ok is True

    status_text = track.STATUS_PATH.read_text(encoding="utf-8")
    assert "유니버스 사양 미충족" in status_text
    assert "FIDELITY_FAIL" in status_text
