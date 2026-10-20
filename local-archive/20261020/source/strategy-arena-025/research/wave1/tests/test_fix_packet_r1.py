from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.backtest import BacktestConfig, run_backtest
from research.wave1.fam_funding import FundingCandidate, FundingMarket, run_portfolio


def _market(
    rates_by_day: list[float],
    *,
    spot_close_by_day: list[float] | None = None,
    perp_close_by_day: list[float] | None = None,
) -> FundingMarket:
    index = pd.date_range("2026-01-01", periods=len(rates_by_day) * 3, freq="8h", tz="UTC")
    spot_open: list[float] = []
    spot_close: list[float] = []
    perp_open: list[float] = []
    perp_close: list[float] = []
    funding: list[float] = []
    previous_perp_close = 100.0
    for day, rate in enumerate(rates_by_day):
        spot_close_value = 100.0 if spot_close_by_day is None else spot_close_by_day[day]
        perp_close_value = 100.0 if perp_close_by_day is None else perp_close_by_day[day]
        for slot in range(3):
            spot_open.append(100.0)
            spot_close.append(spot_close_value)
            perp_open.append(previous_perp_close if slot == 0 else perp_close_value)
            perp_close.append(perp_close_value)
            funding.append(rate)
        previous_perp_close = perp_close_value
    spot = pd.DataFrame({"open": spot_open, "close": spot_close}, index=index)
    perp = pd.DataFrame({"open": perp_open, "close": perp_close}, index=index)
    return FundingMarket(spot=spot, perp=perp, funding=pd.Series(funding, index=index, dtype=float))


def test_trailing_stop_uses_direction_only_not_leverage() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 101.0, 100.0],
            "low": [100.0, 100.0, 94.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    )
    signals = pd.Series([0.0, 1.0, 1.0, 0.0], index=index)
    config = BacktestConfig(fee_rate=0.0, slippage_rate=0.0, stop_distance=0.05)

    # When
    one_x = run_backtest(bars, signals, config)
    leveraged = run_backtest(bars, signals * 2.5, config)

    # Then
    for result in (one_x, leveraged):
        stop = next(fill for fill in result.fills if fill.reason == "stop")
        assert stop.timestamp == index[2]
        assert stop.price == pytest.approx(95.0)


def test_funding_portfolio_past_equity_is_unchanged_by_future_bars() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    market = _market([0.001] * 8)
    future_market = _market(
        [0.001] * 6 + [0.02, -0.02],
        spot_close_by_day=[100.0] * 6 + [500.0, 1.0],
        perp_close_by_day=[100.0] * 6 + [2.0, 800.0],
    )
    candidate = FundingCandidate("test", window_days=1, threshold_apr=0.08, top_k=1)

    # When
    baseline = run_portfolio({"BTCUSDT": market}, candidate)
    mutated = run_portfolio({"BTCUSDT": future_market}, candidate)

    # Then
    pd.testing.assert_series_equal(baseline.equity.iloc[:-2], mutated.equity.iloc[:-2], check_exact=True)
    assert baseline.equity.index.equals(index)


def test_partial_rebalance_updates_trade_weight_without_changing_equity() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    candidate = FundingCandidate("test", window_days=1, threshold_apr=0.08, top_k=4)
    markets = {
        "AUSDT": _market([0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0]),
        "BUSDT": _market([0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0]),
    }

    # When
    result = run_portfolio(markets, candidate)

    # Then
    assert result.equity.iloc[-1] == pytest.approx(301.404643, rel=1e-6)
    assert result.trade_returns.loc[index[5]] == pytest.approx(0.00174357, rel=1e-3)
