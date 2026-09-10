# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pytest"]
# ///
# How to run: uv run --with pytest python -m pytest work/test_x_discovery_account.py -q
"""Hand-calculated account and chronology regressions."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import x_discovery_account as account
from alt_search_models import MarketData


def panel(opens: list[float], lows: list[float] | None = None) -> MarketData:
    prices = np.asarray(opens)[:, None]
    low = prices if lows is None else np.asarray(lows)[:, None]
    return MarketData(
        tuple(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(len(opens))),
        ("ALT",), prices.copy(), prices.copy(), low.copy(), prices.copy(), np.full_like(prices, 1e7),
    )


def test_account_charges_open_and_close_fees_on_actual_notional() -> None:
    # Given: $1 equity, 1x position, price doubles, 1% each-way costs.
    data = panel([100.0, 200.0])
    # When: one position is held to the terminal open.
    result = account.simulate(data, np.ones((2, 1)), account.Execution(0, 1, 1, 0.01, 0.0))
    # Then: 1 - .01 + 1 - .02 = 1.97, no re-investment in the same bar.
    assert result.equity[-1] == pytest.approx(1.97)


def test_liquidated_account_cannot_collect_later_recovery() -> None:
    # Given: 5x long crosses a conservative margin barrier before recovering.
    data = panel([100.0, 110.0, 150.0], [70.0, 100.0, 150.0])
    # When: the account follows the complete path.
    result = account.simulate(data, np.full((3, 1), 5.0), account.Execution(0, 2, 1, 0.0, 0.0))
    # Then: the first adverse bar terminates equity permanently.
    assert result.liquidation_index == 0
    np.testing.assert_array_equal(result.equity, [1.0, 0.0, 0.0])


def test_terminal_open_gap_is_part_of_liquidation_check() -> None:
    data = panel([100.0, 70.0], [100.0, 70.0])
    result = account.simulate(data, np.full((2, 1), 5.0), account.Execution(0, 1, 1, 0.0, 0.0))
    assert result.liquidation_index == 0


def test_kelly_uses_only_completed_returns() -> None:
    returns = np.asarray([0.01, -0.005, 0.02, 0.01, -0.01, 0.005, 0.03, -0.02])
    changed = returns.copy()
    changed[5:] = 0.8
    first = account.kelly_schedule(returns, 4, False)
    second = account.kelly_schedule(changed, 4, False)
    np.testing.assert_array_equal(first[:6], second[:6])
    assert np.all((first >= 0) & (first <= 5))


def test_invalid_exposure_fails_closed() -> None:
    data = panel([100.0, 101.0])
    with pytest.raises(account.AccountInputError):
        account.simulate(data, np.full((2, 1), np.nan), account.Execution(0, 1, 1, 0.0, 0.0))
