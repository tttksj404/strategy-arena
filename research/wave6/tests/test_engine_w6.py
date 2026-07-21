# Unit tests for the pure wave-6 signal/pricing functions (no network, no cached files).

from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.common import PipelineError
from research.wave6.engine_w6 import (
    align_token_underlying,
    deviation_fade_position,
    eligible_listing_symbols,
    funding_window_trades,
    intraday_round_trip_cost,
    listing_short_trade,
    read_symbol_frame,
    spillover_trades,
    weekend_trades,
)


def _bars(pairs: dict[pd.Timestamp, tuple[float, float]]) -> pd.DataFrame:
    index = pd.DatetimeIndex(sorted(pairs))
    return pd.DataFrame({"open": [pairs[ts][0] for ts in index], "close": [pairs[ts][1] for ts in index]}, index=index)


def _next_weekday(anchor: pd.Timestamp, weekday: int) -> pd.Timestamp:
    return anchor + pd.Timedelta(days=(weekday - anchor.weekday()) % 7)


# --------------------------------------------------------------------------------------
# W6a / W6b -- funding window
# --------------------------------------------------------------------------------------


def test_funding_window_uses_previous_settlement_not_current() -> None:
    # Given: settlement[0] itself pays a large negative funding, but there is no *prior*
    # settlement to reference yet; settlement[1] pays flat funding but settlement[0] (the
    # settlement immediately before it) was strongly negative.
    settlements = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="8h")
    funding = pd.Series([-0.0006, 0.0, 0.0004, 0.0], index=settlements)
    price_index = pd.DatetimeIndex([ts + offset for ts in settlements for offset in (pd.Timedelta(hours=-1), pd.Timedelta(hours=1))])
    price_open = pd.Series(100.0, index=price_index)
    price_open.loc[settlements[1] + pd.Timedelta(hours=1)] = 101.0

    # When
    frame = funding_window_trades(funding, price_open, threshold=0.0003, direction=1.0)

    # Then: settlement[0] cannot trigger (no proxy yet); settlement[1] triggers off settlement[0]'s
    # realized funding, not its own (flat) funding.
    assert not bool(frame.loc[settlements[0], "triggered"])
    assert bool(frame.loc[settlements[1], "triggered"])
    assert frame.loc[settlements[1], "raw_return"] == pytest.approx(0.01)
    assert not bool(frame.loc[settlements[2], "triggered"])


def test_funding_window_short_direction_profits_from_decline() -> None:
    # Given
    settlements = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="8h")
    funding = pd.Series([0.0004, 0.0, 0.0], index=settlements)
    price_index = pd.DatetimeIndex([ts + offset for ts in settlements for offset in (pd.Timedelta(hours=-1), pd.Timedelta(hours=1))])
    price_open = pd.Series(100.0, index=price_index)
    price_open.loc[settlements[1] + pd.Timedelta(hours=1)] = 98.0

    # When
    frame = funding_window_trades(funding, price_open, threshold=0.0003, direction=-1.0)

    # Then
    assert bool(frame.loc[settlements[1], "triggered"])
    assert frame.loc[settlements[1], "raw_return"] == pytest.approx(0.02)


def test_funding_window_rejects_invalid_direction() -> None:
    settlements = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="8h")
    funding = pd.Series([0.0, 0.0], index=settlements)
    with pytest.raises(PipelineError):
        funding_window_trades(funding, pd.Series(dtype=float), threshold=0.0003, direction=0.5)


# --------------------------------------------------------------------------------------
# W6c -- open spillover
# --------------------------------------------------------------------------------------


def test_spillover_follows_positive_pre_open_bar_and_skips_weekends() -> None:
    # Given: a Monday with a +1% [12:00,13:00) bar, and a Saturday with an even larger move that
    # must be excluded entirely.
    monday = _next_weekday(pd.Timestamp("2026-01-01T00:00:00Z"), 0)
    saturday = _next_weekday(pd.Timestamp("2026-01-01T00:00:00Z"), 5)
    price = _bars(
        {
            monday + pd.Timedelta(hours=12): (100.0, 101.0),
            monday + pd.Timedelta(hours=13): (101.0, 101.5),
            monday + pd.Timedelta(hours=16): (103.0, 103.0),
            saturday + pd.Timedelta(hours=12): (100.0, 110.0),
            saturday + pd.Timedelta(hours=13): (110.0, 110.0),
            saturday + pd.Timedelta(hours=16): (120.0, 120.0),
        }
    )

    # When
    trades = spillover_trades(price)

    # Then
    assert list(trades.index) == [monday + pd.Timedelta(hours=13)]
    assert trades.iloc[0]["signal"] == 1.0
    assert trades.iloc[0]["raw_return"] == pytest.approx(103.0 / 101.0 - 1.0)


def test_spillover_shorts_when_pre_open_bar_is_negative() -> None:
    # Given
    monday = _next_weekday(pd.Timestamp("2026-01-01T00:00:00Z"), 0)
    price = _bars(
        {
            monday + pd.Timedelta(hours=12): (100.0, 99.0),
            monday + pd.Timedelta(hours=13): (99.0, 99.0),
            monday + pd.Timedelta(hours=16): (95.0, 95.0),
        }
    )

    # When
    trades = spillover_trades(price)

    # Then: short the follow-through, so a further decline is a gain.
    assert trades.iloc[0]["signal"] == -1.0
    assert trades.iloc[0]["raw_return"] == pytest.approx(-1.0 * (95.0 / 99.0 - 1.0))


# --------------------------------------------------------------------------------------
# W6d -- weekend drift
# --------------------------------------------------------------------------------------


def test_weekend_trades_saturday_to_monday_always_long() -> None:
    # Given
    saturday = _next_weekday(pd.Timestamp("2026-01-01T00:00:00Z"), 5)
    monday = saturday + pd.Timedelta(days=2)
    price = pd.DataFrame({"open": [100.0, 108.0]}, index=pd.DatetimeIndex([saturday, monday]))

    # When
    trades = weekend_trades(price)

    # Then
    assert list(trades.index) == [saturday]
    assert trades.iloc[0]["raw_return"] == pytest.approx(0.08)


# --------------------------------------------------------------------------------------
# W6e -- alignment and deviation fade
# --------------------------------------------------------------------------------------


def test_align_token_underlying_matches_half_hour_offset_bars() -> None:
    # Given: Bitget-style on-the-hour token bars and Yahoo-style :30-offset underlying bars, as
    # actually observed in the wave-6 cache (NYSE opens 9:30 ET = 13:30 UTC).
    token = pd.Series(
        [100.0, 101.0, 102.0, 103.0],
        index=pd.DatetimeIndex(["2026-01-05T13:00Z", "2026-01-05T14:00Z", "2026-01-05T15:00Z", "2026-01-05T16:00Z"]),
    )
    underlying = pd.Series([100.2, 101.4, 102.1], index=pd.DatetimeIndex(["2026-01-05T13:30Z", "2026-01-05T14:30Z", "2026-01-05T15:30Z"]))

    # When
    aligned = align_token_underlying(token, underlying)

    # Then: each :30 underlying bar pairs with the most recent (backward) on-the-hour token bar.
    assert len(aligned) == 3
    assert aligned.loc[pd.Timestamp("2026-01-05T13:30Z"), "token"] == pytest.approx(100.0)
    assert aligned.loc[pd.Timestamp("2026-01-05T14:30Z"), "token"] == pytest.approx(101.0)
    assert aligned.loc[pd.Timestamp("2026-01-05T15:30Z"), "token"] == pytest.approx(102.0)


def test_align_token_underlying_drops_unmatched_gaps() -> None:
    # Given: an underlying bar with no token data anywhere near it.
    token = pd.Series([100.0], index=pd.DatetimeIndex(["2026-01-05T13:00Z"]))
    underlying = pd.Series([100.0, 200.0], index=pd.DatetimeIndex(["2026-01-05T13:30Z", "2026-01-09T13:30Z"]))

    # When
    aligned = align_token_underlying(token, underlying)

    # Then
    assert len(aligned) == 1
    assert aligned.index[0] == pd.Timestamp("2026-01-05T13:30Z")


# --------------------------------------------------------------------------------------
# W6e -- deviation fade
# --------------------------------------------------------------------------------------


def test_deviation_fade_hysteresis_and_forced_session_close() -> None:
    # Given: bar1 breaches +entry (rich -> fade short); bar2 converges inside +/-exit (flatten);
    # bar3 breaches again but is also the session's last bar, so it must not carry a position.
    index = pd.date_range("2026-01-05T13:30:00Z", periods=4, freq="1h")
    dev = pd.Series([0.0, 0.004, 0.0005, 0.004], index=index)
    session_end = pd.Series([False, False, False, True], index=index)

    # When
    position = deviation_fade_position(dev, entry_dev=0.003, exit_dev=0.001, session_end=session_end)

    # Then
    assert position.tolist() == [0.0, -1.0, 0.0, 0.0]


def test_deviation_fade_rejects_invalid_thresholds() -> None:
    index = pd.date_range("2026-01-05T13:30:00Z", periods=2, freq="1h")
    dev = pd.Series([0.0, 0.0], index=index)
    session_end = pd.Series([False, False], index=index)
    with pytest.raises(PipelineError):
        deviation_fade_position(dev, entry_dev=0.001, exit_dev=0.002, session_end=session_end)


# --------------------------------------------------------------------------------------
# W6f -- new-listing effect
# --------------------------------------------------------------------------------------


def test_eligible_listing_symbols_filters_missing_launch_time() -> None:
    # Given
    contracts = [
        {"symbol": "AUSDT", "launchTime": ""},
        {"symbol": "BUSDT", "launchTime": "0"},
        {"symbol": "CUSDT", "launchTime": "1700000000000"},
        {"symbol": "DUSDT"},
        {"not_a_symbol": True},
    ]

    # When
    eligible = eligible_listing_symbols(contracts)

    # Then
    assert [symbol for symbol, _ in eligible] == ["CUSDT"]


def test_listing_short_trade_profits_from_decline() -> None:
    # Given
    onboard = pd.Timestamp("2026-01-01T00:00:00Z")
    dates = pd.date_range(onboard, periods=10, freq="1D")
    daily = pd.DataFrame({"open": [10.0] * 10, "close": [10.0] * 10}, index=dates)
    daily.loc[onboard + pd.Timedelta(days=7), "close"] = 8.0

    # When
    trade = listing_short_trade(daily, onboard)

    # Then: short at D+2 open (10.0), cover at D+7 close (8.0) -> +20% for the short.
    assert trade == pytest.approx(0.2)


def test_listing_short_trade_returns_none_when_dates_missing() -> None:
    onboard = pd.Timestamp("2026-01-01T00:00:00Z")
    daily = pd.DataFrame({"open": [10.0], "close": [10.0]}, index=[onboard])
    assert listing_short_trade(daily, onboard) is None


# --------------------------------------------------------------------------------------
# Costs and cache I/O
# --------------------------------------------------------------------------------------


def test_intraday_round_trip_cost_matches_spec_formula() -> None:
    # SPEC.md W6a/b: "테이커 0.06%+슬리피지 2배(2bp) 왕복" -> 2 legs x (0.0006 + 0.0002).
    cost = intraday_round_trip_cost("BTCUSDT")
    assert cost == pytest.approx(2.0 * (0.0006 + 0.0002))


def test_read_symbol_frame_rejects_missing_file(tmp_path: object) -> None:
    with pytest.raises(PipelineError):
        read_symbol_frame(tmp_path, "binance_fapi_", "BTCUSDT", "_1h.csv.gz")  # type: ignore[arg-type]
