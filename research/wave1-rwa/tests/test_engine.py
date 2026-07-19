"""Golden cases for execution, costs, funding and liquidation."""

import pandas as pd

from src.engine import Costs, run_backtest


def candles(rows: list[tuple[int, float, float, float, float]]) -> pd.DataFrame:
    """Build a minimal UTC hourly fixture."""
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).assign(
        ts=lambda x: pd.to_datetime(x.ts, unit="h", origin="2026-01-01", utc=True),
        vol_base=1.0, vol_quote=100.0,
    )


def test_costs_apply_taker_spread_and_slippage_per_side() -> None:
    """Given a measured spread, the configured round trip is two-sided."""
    assert Costs(5).round_trip_rate == 2 * (0.0006 + 0.0005 + 0.0001)
    assert Costs(0).round_trip_rate == 2 * (0.0006 + 0.0001 + 0.0001)


def test_execution_uses_next_open_not_same_close() -> None:
    """Given a close spike, a next-open entry cannot harvest that prior bar."""
    frame = candles([(0, 100, 100, 100, 100), (1, 100, 200, 100, 200), (2, 100, 100, 100, 100)])
    result = run_backtest(frame, pd.Series([0.0, 1.0, 0.0]), pd.DataFrame(columns=["ts", "rate"]), 2, Costs(1))
    assert result.net_return < 0
    assert result.equity.iloc[-1] < 300


def test_funding_is_signed_and_covered() -> None:
    """Given a positive long funding event, funding is recorded as a cost."""
    frame = candles([(0, 100, 100, 100, 100), (1, 100, 101, 99, 100), (2, 100, 100, 100, 100)])
    funding = pd.DataFrame({"ts": [frame.ts.iloc[1]], "rate": [0.01]})
    result = run_backtest(frame, pd.Series([1.0, 1.0, 1.0]), funding, 2, Costs(1))
    assert result.funding_paid > 5.9
    assert result.funding_coverage == 1 / 3


def test_liquidation_sets_equity_to_zero() -> None:
    """Given a maintenance-adjusted liquidation touch, equity is conservatively zeroed."""
    frame = candles([(0, 100, 100, 100, 100), (1, 100, 100, 40, 100), (2, 100, 100, 100, 100)])
    result = run_backtest(frame, pd.Series([1.0, 1.0, 1.0]), pd.DataFrame(columns=["ts", "rate"]), 2, Costs(1))
    assert result.liquidated
    assert result.equity.iloc[-1] == 0


def test_leverage_multiplies_price_pnl() -> None:
    """Given L=2 and a +10% move with zero-ish costs, equity gains ~20% (old engine returned ~10%)."""
    frame = candles([(0, 100, 100, 100, 100), (1, 100, 110, 100, 110), (2, 110, 110, 110, 110)])
    result = run_backtest(frame, pd.Series([1.0, 1.0, 0.0]), pd.DataFrame(columns=["ts", "rate"]), 2, Costs(1))
    assert result.net_return > 0.15, f"levered PnL missing: {result.net_return:.4f}"


def test_trade_return_includes_entry_cost() -> None:
    """A flat round trip must lose about the two-sided cost at notional L, captured in the trade return."""
    frame = candles([(0, 100, 100, 100, 100), (1, 100, 100, 100, 100), (2, 100, 100, 100, 100), (3, 100, 100, 100, 100)])
    result = run_backtest(frame, pd.Series([1.0, 1.0, 0.0, 0.0]), pd.DataFrame(columns=["ts", "rate"]), 5, Costs(1))
    expected = -Costs(1).round_trip_rate * 5
    assert abs(result.trade_returns[0] - expected) < 3e-4, (result.trade_returns, expected)
