from __future__ import annotations

from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK
import pytest
import requests

from research.paper import market_data


def test_rank_bitget_universe_orders_by_volume_descending_with_symbol_tiebreak() -> None:
    volumes = {"BUSDT": 5.0, "AUSDT": 5.0, "CUSDT": 10.0, "DUSDT": 1.0}
    ranked = market_data.rank_bitget_universe(volumes, limit=3)
    # CUSDT highest volume first; BUSDT/AUSDT tie on volume, symbol breaks the tie alphabetically.
    assert ranked == ("CUSDT", "AUSDT", "BUSDT")


def test_rank_bitget_universe_truncates_to_limit() -> None:
    volumes = {f"SYM{i}USDT": float(i) for i in range(10)}
    assert len(market_data.rank_bitget_universe(volumes, limit=4)) == 4


def test_bitget_mix_volumes_prefers_usdt_volume_falls_back_to_quote_volume() -> None:
    payload = [
        {"symbol": "AUSDT", "usdtVolume": "123.5", "quoteVolume": "999"},
        {"symbol": "BUSDT", "quoteVolume": "77.0"},
        {"symbol": "CUSDT"},  # neither field present -> defaults to 0.0, never dropped/raised
        {"not_a_symbol_entry": True},
    ]
    volumes = market_data.bitget_mix_volumes(payload)
    assert volumes["AUSDT"] == 123.5
    assert volumes["BUSDT"] == 77.0
    assert volumes["CUSDT"] == 0.0
    assert len(volumes) == 3


def test_bitget_mix_prices_skips_missing_or_nonpositive_lastpr() -> None:
    payload = [
        {"symbol": "AUSDT", "lastPr": "63661.5"},
        {"symbol": "BUSDT", "lastPr": "0"},
        {"symbol": "CUSDT", "lastPr": None},
        {"symbol": "DUSDT"},
    ]
    prices = market_data.bitget_mix_prices(payload)
    assert prices == {"AUSDT": 63661.5}


def test_bitget_spot_prices_parses_bulk_payload() -> None:
    payload = [{"symbol": "MAGMAUSDT", "lastPr": "0.0123"}, {"symbol": "TMUSDT", "lastPr": "-1"}]
    prices = market_data.bitget_spot_prices(payload)
    assert prices == {"MAGMAUSDT": 0.0123}


def test_bitget_payload_helpers_reject_non_list_shapes() -> None:
    from research.wave1.common import PipelineError

    with pytest.raises(PipelineError):
        market_data.bitget_mix_volumes({"not": "a list"})
    with pytest.raises(PipelineError):
        market_data.bitget_mix_prices({"not": "a list"})
    with pytest.raises(PipelineError):
        market_data.bitget_spot_prices({"not": "a list"})


def test_funding_cache_round_trip_and_same_day_skip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(market_data, "FUNDING_CACHE_DIR", tmp_path / "bitget_funding")
    series = pd.Series([0.0001, 0.0002], index=pd.DatetimeIndex([pd.Timestamp("2026-07-27T00:00:00Z"), pd.Timestamp("2026-07-27T08:00:00Z")]))

    assert market_data.load_cached_funding("BTCUSDT", "2026-07-28") is None  # nothing cached yet

    market_data.save_cached_funding("BTCUSDT", "2026-07-28", series)
    same_day = market_data.load_cached_funding("BTCUSDT", "2026-07-28")
    assert same_day is not None
    assert list(same_day.to_numpy()) == list(series.to_numpy())

    # A later calendar day must NOT reuse yesterday's cache -- forces a fresh fetch.
    assert market_data.load_cached_funding("BTCUSDT", "2026-07-29") is None


def test_extend_with_bitget_carry_universe_skips_symbols_already_covered(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(market_data, "FUNDING_CACHE_DIR", tmp_path / "bitget_funding")
    monkeypatch.setattr(market_data, "_fetch_bitget_mix_tickers", lambda session: [
        {"symbol": "BTCUSDT", "usdtVolume": "100", "lastPr": "60000"},
        {"symbol": "NEWUSDT", "usdtVolume": "50", "lastPr": "1.5"},
    ])
    monkeypatch.setattr(market_data, "_fetch_bitget_spot_tickers", lambda session: [
        {"symbol": "NEWUSDT", "lastPr": "1.4"},
    ])
    calls: list[str] = []

    def fake_fetch(symbol: str, session, page_size: int = 100) -> pd.Series:
        calls.append(symbol)
        return pd.Series([0.0003], index=pd.DatetimeIndex([pd.Timestamp("2026-07-28T00:00:00Z")]))

    monkeypatch.setattr(market_data, "_fetch_bitget_funding_recent", fake_fetch)

    funding_series = {"BTCUSDT": pd.Series([0.0001], index=pd.DatetimeIndex([pd.Timestamp("2026-07-01", tz="UTC")]))}
    perp_prices: dict[str, float] = {"BTCUSDT": 59999.0}
    spot_prices: dict[str, float] = {}

    failed = market_data.extend_with_bitget_carry_universe(
        requests.Session(), funding_series, perp_prices, spot_prices, "2026-07-28", loop_deadline=float("inf")
    )

    assert calls == ["NEWUSDT"]  # BTCUSDT already had funding data -- never refetched/overwritten
    assert failed == ()
    assert perp_prices["BTCUSDT"] == 59999.0  # untouched, existing Binance-sourced price kept
    assert "NEWUSDT" in funding_series
    assert perp_prices["NEWUSDT"] == 1.5
    assert spot_prices["NEWUSDT"] == 1.4


def test_extend_with_bitget_carry_universe_excludes_failed_symbols_without_guessing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(market_data, "FUNDING_CACHE_DIR", tmp_path / "bitget_funding")
    monkeypatch.setattr(market_data, "_fetch_bitget_mix_tickers", lambda session: [
        {"symbol": "BADUSDT", "usdtVolume": "10", "lastPr": "1.0"},
        {"symbol": "EMPTYUSDT", "usdtVolume": "5", "lastPr": "1.0"},
    ])
    monkeypatch.setattr(market_data, "_fetch_bitget_spot_tickers", lambda session: [])

    def fake_fetch(symbol: str, session, page_size: int = 100) -> pd.Series:
        if symbol == "BADUSDT":
            from research.wave1.common import PipelineError

            raise PipelineError("boom")
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))  # EMPTYUSDT: fetched fine, no history yet

    monkeypatch.setattr(market_data, "_fetch_bitget_funding_recent", fake_fetch)

    funding_series: dict[str, pd.Series] = {}
    failed = market_data.extend_with_bitget_carry_universe(
        requests.Session(), funding_series, {}, {}, "2026-07-28", loop_deadline=float("inf")
    )

    assert set(failed) == {"BADUSDT", "EMPTYUSDT"}
    assert funding_series == {}  # neither symbol assumed/filled with a guess


def test_extend_with_bitget_carry_universe_respects_time_budget(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(market_data, "FUNDING_CACHE_DIR", tmp_path / "bitget_funding")
    monkeypatch.setattr(market_data, "_fetch_bitget_mix_tickers", lambda session: [
        {"symbol": "AUSDT", "usdtVolume": "10", "lastPr": "1.0"},
    ])
    monkeypatch.setattr(market_data, "_fetch_bitget_spot_tickers", lambda session: [])

    def fail_if_called(symbol: str, session, page_size: int = 100) -> pd.Series:
        raise AssertionError("must not fetch once the time budget is already exhausted")

    monkeypatch.setattr(market_data, "_fetch_bitget_funding_recent", fail_if_called)

    funding_series: dict[str, pd.Series] = {}
    # loop_deadline already in the past -> every ranked symbol should be skipped, not fetched.
    failed = market_data.extend_with_bitget_carry_universe(
        requests.Session(), funding_series, {}, {}, "2026-07-28", loop_deadline=0.0
    )

    assert failed == ("AUSDT",)
    assert funding_series == {}
