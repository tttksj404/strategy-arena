# pyright: reportImplicitRelativeImport=false

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from alt_search_models import MarketData
from bitget_path_survival import (
    NoSurvivalEventsError,
    SurvivalMetricError,
    SurvivalSpec,
    evaluate_signal_survival,
)


def _market(
    *,
    lows: tuple[float, ...],
    closes: tuple[float, ...],
    opens: tuple[float, ...] | None = None,
) -> MarketData:
    rows = len(closes)
    dates = tuple(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * index) for index in range(rows))
    close = np.asarray(closes, dtype=float)[:, None]
    open_price = np.asarray(opens or (100.0,) * rows, dtype=float)[:, None]
    low = np.asarray(lows, dtype=float)[:, None]
    high = np.maximum(open_price, close) + 1.0
    return MarketData(dates, ("ALT",), open_price, high, low, close, np.ones_like(close))


def _spec() -> SurvivalSpec:
    return SurvivalSpec(start=1, end=3, horizon_bars=2, leverage=5.0, round_trip_cost=0.0, liquidation_reserve=0.02)


def test_eventual_winner_is_counted_as_liquidated_when_mae_breaks_the_five_x_buffer() -> None:
    # Given: a long entry eventually gains 10%, but first trades 21% below entry.
    data = _market(lows=(99.0, 79.0, 85.0, 99.0), closes=(100.0, 90.0, 110.0, 100.0))
    positions = np.asarray(((0.0,), (1.0,), (1.0,), (1.0,)))

    # When: the two-bar signal path is evaluated at 5x.
    metrics = evaluate_signal_survival(data, positions, _spec())

    # Then: direction is correct, but the path is not survivable.
    assert metrics.event_count == 1
    assert metrics.cost_adjusted_terminal_win_rate == 1.0
    assert metrics.market_proxy_survival_rate == 0.0
    assert metrics.eventual_winner_market_proxy_liquidation_rate == 1.0
    assert metrics.mean_signal_equity_delta == -1.0
    assert metrics.survivor_time_to_profit_bars_median is None
    assert metrics.exact_bitget_liquidation is False


def test_surviving_winner_reports_cost_adjusted_leveraged_equity_delta() -> None:
    # Given: a long entry gains 8% with only 10% maximum adverse excursion.
    data = _market(lows=(99.0, 90.0, 94.0, 99.0), closes=(100.0, 96.0, 108.0, 100.0))
    positions = np.asarray(((0.0,), (1.0,), (1.0,), (1.0,)))
    spec = SurvivalSpec(start=1, end=3, horizon_bars=2, leverage=5.0, round_trip_cost=0.001, liquidation_reserve=0.02)

    # When: the path is evaluated with round-trip costs.
    metrics = evaluate_signal_survival(data, positions, spec)

    # Then: the event survives and earns the leveraged net terminal return.
    assert metrics.profitable_survivor_rate == 1.0
    assert metrics.mean_signal_equity_delta == pytest.approx(0.395)
    assert metrics.survivor_time_to_profit_bars_median == 2.0


def test_signal_episode_exits_at_next_open_when_the_strategy_position_closes() -> None:
    # Given: a one-bar position closes before a later rally inside the fixed horizon.
    data = _market(
        lows=(99.0, 99.0, 99.0, 99.0, 99.0),
        closes=(100.0, 99.0, 120.0, 120.0, 120.0),
    )
    positions = np.asarray(((0.0,), (1.0,), (0.0,), (0.0,), (0.0,)))
    spec = SurvivalSpec(
        start=1,
        end=5,
        horizon_bars=4,
        leverage=5.0,
        round_trip_cost=0.0,
        liquidation_reserve=0.02,
    )

    # When: the strategy-aware episode is evaluated.
    metrics = evaluate_signal_survival(data, positions, spec)

    # Then: it exits at row 2 open (100), so the later close at 120 is ignored.
    assert metrics.cost_adjusted_terminal_win_rate == 0.0
    assert metrics.mean_signal_equity_delta == 0.0
    assert metrics.mfe_median == pytest.approx(0.01)


def test_non_finite_positions_are_rejected() -> None:
    data = _market(lows=(99.0, 99.0, 99.0, 99.0), closes=(100.0, 100.0, 100.0, 100.0))
    positions = np.zeros((4, 1), dtype=float)
    positions[1, 0] = np.nan

    with pytest.raises(SurvivalMetricError, match="finite"):
        evaluate_signal_survival(data, positions, _spec())


@pytest.mark.parametrize(
    ("direction", "gap_open", "gap_low"),
    ((1.0, 70.0, 70.0), (-1.0, 130.0, 99.0)),
)
def test_exit_open_gap_is_included_in_liquidation_proxy(
    direction: float,
    gap_open: float,
    gap_low: float,
) -> None:
    data = _market(
        opens=(100.0, 100.0, gap_open, 100.0),
        lows=(99.0, 99.0, gap_low, 99.0),
        closes=(100.0, 100.0, 100.0, 100.0),
    )
    positions = np.asarray(((0.0,), (direction,), (0.0,), (0.0,)))

    metrics = evaluate_signal_survival(data, positions, _spec())

    assert metrics.market_proxy_survival_rate == 0.0
    assert metrics.mean_signal_equity_delta == -1.0


def test_no_new_position_event_is_rejected() -> None:
    # Given: no non-zero position ever begins.
    data = _market(lows=(99.0, 99.0, 99.0, 99.0), closes=(100.0, 100.0, 100.0, 100.0))
    positions = np.zeros((4, 1), dtype=float)

    # When / Then: the diagnostic refuses an empty sample.
    with pytest.raises(NoSurvivalEventsError, match="no entry events"):
        evaluate_signal_survival(data, positions, _spec())
