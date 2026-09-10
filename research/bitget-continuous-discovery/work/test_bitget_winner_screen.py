from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from alt_search_models import Candidate, Family, MarketData, RiskOverlay
from bitget_mtf_core import MarketSnapshot
from run_bitget_winner_screen import _rescaled_candidate, _screen_one, has_promotable_oos_duration, market_gate_for_active_symbols, parse_candidate_identifier, parse_overlay_identifier


def test_candidate_identifier_round_trips_fixed_pair_configuration() -> None:
    candidate = parse_candidate_identifier("pair|lb=40|sec=0|k=0|dir=0|pair=ETH-SOL|z=1.5|reb=2|liq=5000000")

    assert candidate.family is Family.PAIR
    assert candidate.lookback == 40
    assert candidate.pair == ("ETH", "SOL")
    assert candidate.entry_z == 1.5
    assert candidate.rebalance_days == 2
    assert candidate.minimum_daily_quote_volume == 5_000_000.0


def test_overlay_identifier_round_trips_and_short_oos_cannot_promote() -> None:
    overlay = parse_overlay_identifier("L=5|tv=0.01|vw=20")

    assert overlay.max_leverage == 5.0
    assert overlay.target_daily_vol == 0.01
    assert overlay.volatility_window == 20
    assert not has_promotable_oos_duration(89.9)
    assert has_promotable_oos_duration(90.0)


def test_market_gate_checks_only_assets_the_candidate_is_currently_signaling() -> None:
    candidate = Candidate(Family.TIME_SERIES, 20, 0, 0, 1, None, 0.0, minimum_daily_quote_volume=0.0)
    markets = {
        "ETHUSDT": MarketSnapshot(quote_volume_usdt=6_000_000.0, spread_fraction=0.0001),
        "ATOMUSDT": MarketSnapshot(quote_volume_usdt=700_000.0, spread_fraction=0.0001),
    }

    assert market_gate_for_active_symbols(("ETH",), candidate, markets)
    assert not market_gate_for_active_symbols(("ETH", "ATOM"), candidate, markets)


def test_daily_liquidity_floor_scales_to_the_screen_interval() -> None:
    candidate = Candidate(Family.TIME_SERIES, 20, 0, 0, 1, None, 0.0, minimum_daily_quote_volume=24_000_000.0)

    scaled = _rescaled_candidate(candidate, "4H")

    assert scaled.minimum_daily_quote_volume == 4_000_000.0
    assert scaled.lookback == candidate.lookback
    assert scaled.rebalance_days == candidate.rebalance_days


def test_screen_one_records_realized_leverage_within_its_declared_cap() -> None:
    dates = tuple(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=4 * index) for index in range(200))
    symbols = ("ETH", "SOL")
    close = np.asarray([[100.0 * (1.001 ** index), 50.0 * (1.0012 ** index)] for index in range(200)])
    data = MarketData(dates, symbols, close, close * 1.001, close * 0.999, close, np.full_like(close, 30_000_000.0))
    candidate = Candidate(Family.TIME_SERIES, 20, 0, 0, 1, None, 0.0, minimum_daily_quote_volume=0.0)
    overlay = RiskOverlay(max_leverage=5.0, target_daily_vol=0.005, volatility_window=20)
    markets = {f"{symbol}USDT": MarketSnapshot(quote_volume_usdt=30_000_000.0, spread_fraction=0.0001) for symbol in symbols}

    result = _screen_one(data, "4H", candidate, overlay, markets)

    assert 0.0 < result["max_realized_leverage"] <= result["max_leverage"]
