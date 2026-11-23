"""Synthetic tests for the deep validation statistics."""

from datetime import datetime, timedelta, timezone

import numpy as np

from research.validation.deep_stats import (
    TimedValue,
    block_bootstrap,
    compare_funding,
    deflated_sharpe,
    leave_one_year_out,
    trade_bootstrap,
)


def _equity(values: list[float], start: datetime) -> tuple[TimedValue, ...]:
    return tuple(TimedValue(start + timedelta(days=index), value) for index, value in enumerate(values))


def test_trade_bootstrap_is_reproducible_and_reports_kelly() -> None:
    result = trade_bootstrap((0.01, 0.02, -0.005, 0.015) * 8, seed=7)
    repeated = trade_bootstrap((0.01, 0.02, -0.005, 0.015) * 8, seed=7)
    assert result == repeated
    assert result.paths == 10_000
    assert result.quarter_kelly_fraction == result.kelly_fraction * 0.25
    assert result.unit.p05 > 0.0


def test_leave_one_year_out_excludes_exact_calendar_year() -> None:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    values = [100.0, 110.0, 99.0, 108.9]
    equity = (TimedValue(start, values[0]), TimedValue(datetime(2020, 12, 31, tzinfo=timezone.utc), values[1]), TimedValue(datetime(2021, 1, 1, tzinfo=timezone.utc), values[2]), TimedValue(datetime(2021, 12, 31, tzinfo=timezone.utc), values[3]))
    rows = leave_one_year_out(equity)
    assert [row.year for row in rows] == [2020, 2021]
    assert np.isclose(rows[0].heldout_return, 0.10)
    assert np.isclose(rows[1].heldout_return, -0.01)
    assert rows[0].remaining_days == 2


def test_dsr_is_finite_for_nonconstant_daily_returns() -> None:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    values = [100.0]
    for index in range(1, 400):
        values.append(values[-1] * (1.001 if index % 3 else 0.998))
    result = deflated_sharpe(_equity(values, start))
    assert result.trials == 28
    assert np.isfinite(result.score)
    assert 0.0 <= result.probability <= 1.0


def test_block_shuffle_preserves_final_capital_and_compares_signals() -> None:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    trades = tuple(TimedValue(start + timedelta(days=index * 30), 0.01 if index % 2 else -0.005) for index in range(12))
    result = block_bootstrap(trades, seed=11)
    assert result.block_count == 4
    assert np.isclose(result.final_p05, result.final_p95)
    funding = {"BTCUSDT": tuple(TimedValue(start + timedelta(hours=index * 8), 0.001) for index in range(30))}
    compared = compare_funding(funding, funding, threshold_apr=0.08)
    assert compared.sign_agreement == 1.0
    assert compared.entry_agreement == 1.0


def test_funding_matching_uses_eight_hour_buckets_and_resets_after_gaps() -> None:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    left = tuple(TimedValue(start + timedelta(hours=index * 8), 0.001) for index in range(42))
    right = tuple(TimedValue(start + timedelta(hours=index * 8, milliseconds=3), 0.001) for index in range(42))
    continuous = compare_funding({"BTCUSDT": left}, {"BTCUSDT": right}, threshold_apr=0.08)
    right_with_gap = tuple(item for index, item in enumerate(right) if index != 21)
    gap = compare_funding({"BTCUSDT": left}, {"BTCUSDT": right_with_gap}, threshold_apr=0.08)
    assert continuous.observations == 22
    assert gap.observations == 1
