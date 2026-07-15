# Verifies that close-derived signals execute only at the next bar open.

from __future__ import annotations

import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.backtest import BacktestConfig, run_backtest
from research.wave1.common import PipelineError
from research.wave1.fam_tsmom import F2_CANDIDATES, TsmomExecution, run_candidate


def test_entry_uses_next_bar_open_when_signal_changes_at_close() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 110.0, 125.0, 130.0],
            "high": [102.0, 112.0, 130.0, 133.0],
            "low": [99.0, 108.0, 124.0, 128.0],
            "close": [101.0, 111.0, 129.0, 131.0],
        },
        index=index,
    )
    signals = pd.Series([0.0, 1.0, 1.0, 0.0], index=index)

    # When
    result = run_backtest(
        bars,
        signals,
        BacktestConfig(fee_rate=0.0, slippage_rate=0.0),
    )

    # Then
    entry = next(fill for fill in result.fills if fill.reason == "entry")
    assert entry.timestamp == index[2]
    assert entry.price == 125.0


def test_empty_bars_fail_with_pipeline_error() -> None:
    # Given
    bars = pd.DataFrame(columns=["open", "high", "low", "close"])

    # When / Then
    with pytest.raises(PipelineError, match="must not be empty"):
        run_backtest(bars, pd.Series(dtype=float))


def test_future_atr_change_does_not_rewrite_past_equity() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=70, freq="D", tz="UTC")
    close = pd.Series([100.0 + offset * 0.5 for offset in range(70)], index=index)
    bars = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close}, index=index)
    changed = bars.copy()
    changed.loc[index[-1], ["high", "low", "close"]] = [1_000.0, 1.0, 500.0]
    execution = TsmomExecution(F2_CANDIDATES[0], "BTCUSDT")

    # When
    baseline = run_candidate(bars, execution)
    mutated = run_candidate(changed, execution)

    # Then
    pd.testing.assert_series_equal(baseline.equity.iloc[:-1], mutated.equity.iloc[:-1])


def test_intraday_funding_applies_after_open_rebalance() -> None:
    # Given
    index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [100.0] * 4,
            "low": [100.0] * 4,
            "close": [100.0] * 4,
            "funding_open": [0.0, 0.0, 0.01, 0.0],
            "funding_rate": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    signals = pd.Series([0.0, 1.0, 1.0, 0.0], index=index)

    # When
    open_only = run_backtest(bars, signals, BacktestConfig(fee_rate=0.0, slippage_rate=0.0))
    bars.loc[index[2], ["funding_open", "funding_rate"]] = [0.0, 0.01]
    intraday = run_backtest(bars, signals, BacktestConfig(fee_rate=0.0, slippage_rate=0.0))

    # Then
    assert open_only.equity.iloc[-1] == pytest.approx(300.0)
    assert intraday.equity.iloc[-1] == pytest.approx(297.0)
