from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK

from research.wave3.engine import AssetMarket, CandidateConfig, btc_above_ma200, run_candidate
from research.wave3.strategy import RankedSignal, select_max_z_candidates, update_hysteresis
from research.wave3.universe import AssetListing, eligible_symbols_at


def test_listing_aware_crypto_enters_exactly_after_sixty_days() -> None:
    # Given
    listing = AssetListing("NEWUSDT", "crypto", pd.Timestamp("2026-01-01", tz="UTC"), True, True)
    dates = pd.date_range("2026-01-31", periods=30, freq="D", tz="UTC")
    volumes = pd.DataFrame({"NEWUSDT": 100.0}, index=dates)

    # When
    before = eligible_symbols_at((listing,), volumes, pd.Timestamp("2026-03-01", tz="UTC"))
    exact = eligible_symbols_at((listing,), volumes, pd.Timestamp("2026-03-02", tz="UTC"))

    # Then
    assert before == ()
    assert exact == ("NEWUSDT",)


def test_volume_filter_uses_only_the_previous_thirty_days() -> None:
    # Given
    dates = pd.date_range("2026-01-01", periods=31, freq="D", tz="UTC")
    listings = tuple(
        AssetListing(f"S{index:03d}USDT", "crypto", pd.Timestamp("2025-01-01", tz="UTC"), True, True)
        for index in range(151)
    )
    values = {listing.symbol: 1.0 for listing in listings}
    values["S150USDT"] = 0.0
    volumes = pd.DataFrame(values, index=dates)
    volumes.loc[dates[-1], "S150USDT"] = 1_000_000.0

    # When
    result = eligible_symbols_at(listings, volumes, dates[-1])

    # Then
    assert "S150USDT" not in result
    assert len(result) == 150


def test_w3e_selects_the_largest_factor_z_scores() -> None:
    # Given
    carry_z = pd.Series({"A": 2.0, "B": 0.5, "C": 0.0})
    momentum_z = pd.Series({"A": 0.1, "B": 3.0, "C": 1.5, "D": 2.5})

    # When
    selected = select_max_z_candidates(carry_z, momentum_z, top_k=3)

    # Then
    assert [(item.symbol, item.factor, item.rank) for item in selected] == [
        ("B", "momentum", 1),
        ("D", "momentum", 2),
        ("A", "carry", 3),
    ]


def test_w3e_keeps_a_position_until_it_falls_outside_rank_ten() -> None:
    # Given
    previous = (RankedSignal("HELD", "carry", 9.0, 5), RankedSignal("GONE", "momentum", 8.0, 11))
    today = (
        RankedSignal("NEW1", "momentum", 10.0, 1),
        RankedSignal("NEW2", "momentum", 9.5, 2),
        RankedSignal("NEW3", "carry", 9.0, 3),
        RankedSignal("HELD", "carry", 4.0, 5),
        RankedSignal("GONE", "momentum", 3.0, 11),
    )

    # When
    result = update_hysteresis(previous, today, entry_count=3, exit_rank=10)

    # Then
    assert tuple(item.symbol for item in result) == ("NEW1", "NEW2", "NEW3", "HELD")


def test_w3e_runs_a_synthetic_cross_asset_market() -> None:
    # Given
    dates = pd.date_range("2026-01-01", periods=70, freq="D", tz="UTC")
    funding_dates = pd.date_range("2026-01-01", periods=210, freq="8h", tz="UTC")
    markets = {}
    for offset, symbol in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
        close = pd.Series(100.0 + offset + dates.dayofyear.to_numpy() * (offset + 1) * 0.1, index=dates)
        bars = pd.DataFrame({"close": close, "quote_volume": 100.0 + offset}, index=dates)
        listing = AssetListing(symbol, "crypto", pd.Timestamp("2025-01-01", tz="UTC"), True, True)
        markets[symbol] = AssetMarket(listing, bars, bars, pd.Series(0.0001, index=funding_dates))

    # When
    result = run_candidate(markets, CandidateConfig("W3e", "max_z", 3, 1, hysteresis=True))

    # Then
    assert result.candidate_id == "W3e"
    assert len(result.equity) == 70
    assert result.equity.notna().all()


def test_w3d_regime_is_cash_until_btc_has_a_valid_ma200() -> None:
    # Given
    dates = pd.date_range("2026-01-01", periods=201, freq="D", tz="UTC")
    close = pd.Series(100.0, index=dates)
    close.iloc[-1] = 50.0

    # When
    regime = btc_above_ma200(close)

    # Then
    assert regime.iloc[:199].eq(False).all()
    assert regime.iloc[-1] is False or not bool(regime.iloc[-1])
