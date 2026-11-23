# F2 trend signals, volatility targeting, and candidate backtests.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, assert_never

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.backtest import BacktestConfig, BacktestResult, run_backtest
from research.wave1.common import StrategyResult, load_frame
from research.wave1.costs import PERP_TAKER_RATE, slippage_rate


class TsmomRule(StrEnum):
    DONCHIAN = "donchian"
    MA_CROSS = "ma_cross"
    MA_SLOPE = "ma_slope"


@dataclass(frozen=True, slots=True)
class TsmomCandidate:
    candidate_id: str
    rule: TsmomRule
    window: int
    long_only: bool


@dataclass(frozen=True, slots=True)
class TsmomExecution:
    candidate: TsmomCandidate
    symbol: str
    stress_multiplier: float = 1.0


F2_CANDIDATES: Final = (
    TsmomCandidate("F2a", TsmomRule.DONCHIAN, 20, False),
    TsmomCandidate("F2b", TsmomRule.DONCHIAN, 55, False),
    TsmomCandidate("F2c", TsmomRule.DONCHIAN, 20, True),
    TsmomCandidate("F2d", TsmomRule.DONCHIAN, 55, True),
    TsmomCandidate("F2e", TsmomRule.MA_CROSS, 200, True),
    TsmomCandidate("F2f", TsmomRule.MA_SLOPE, 100, False),
)


def neighbor_candidates(candidate: TsmomCandidate) -> tuple[TsmomCandidate, ...]:
    windows = {max(2, round(candidate.window * 0.8)), round(candidate.window * 1.2)}
    return tuple(TsmomCandidate(candidate.candidate_id, candidate.rule, window, candidate.long_only) for window in sorted(windows))


def donchian_signal(close: pd.Series, window: int, long_only: bool) -> pd.Series:
    upper = close.rolling(window, min_periods=window).max().shift(1)
    lower = close.rolling(window, min_periods=window).min().shift(1)
    values: list[float] = []
    state = 0.0
    for price, high, low in zip(close, upper, lower, strict=True):
        if pd.notna(high) and price > high:
            state = 1.0
        elif pd.notna(low) and price < low:
            state = 0.0 if long_only else -1.0
        values.append(state)
    return pd.Series(values, index=close.index, dtype=float)


def vol_target_fraction(
    realized_vol: pd.Series,
    target_vol: float = 0.015,
    leverage_cap: float = 3.0,
) -> pd.Series:
    safe = realized_vol.where(realized_vol > 0.0)
    fraction = (target_vol / safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return fraction.clip(lower=0.0, upper=leverage_cap)


def atr(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    previous_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1).rolling(window, min_periods=window).mean()


def candidate_signal(bars: pd.DataFrame, candidate: TsmomCandidate) -> pd.Series:
    close = bars["close"]
    match candidate.rule:
        case TsmomRule.DONCHIAN:
            return donchian_signal(close, candidate.window, candidate.long_only)
        case TsmomRule.MA_CROSS:
            fast_window = max(2, round(candidate.window / 4))
            fast = close.rolling(fast_window, min_periods=fast_window).mean()
            slow = close.rolling(candidate.window, min_periods=candidate.window).mean()
            return (fast > slow).astype(float).fillna(0.0)
        case TsmomRule.MA_SLOPE:
            average = close.rolling(candidate.window, min_periods=candidate.window).mean()
            return np.sign(average.diff()).fillna(0.0)
        case unreachable:
            assert_never(unreachable)


def run_candidate(bars: pd.DataFrame, execution: TsmomExecution) -> BacktestResult:
    returns = bars["close"].pct_change()
    realized = returns.rolling(20, min_periods=20).std(ddof=1)
    positions = candidate_signal(bars, execution.candidate) * vol_target_fraction(realized)
    stop_distance = 3.0 * atr(bars) / bars["close"]
    config = BacktestConfig(
        fee_rate=PERP_TAKER_RATE,
        slippage_rate=slippage_rate(execution.symbol, execution.stress_multiplier),
        stop_distance=stop_distance
        if execution.candidate.rule is TsmomRule.DONCHIAN and not execution.candidate.long_only
        else None,
    )
    return run_backtest(bars, positions, config)


def run_cached(cache_dir: Path, candidate: TsmomCandidate) -> StrategyResult:
    asset_results: list[BacktestResult] = []
    stress_results: list[BacktestResult] = []
    signed_positions: list[pd.Series] = []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        bars = load_frame(cache_dir / f"binance_fapi_{symbol}_1d.csv.gz")
        funding_path = cache_dir / f"binance_funding_{symbol}.csv.gz"
        if funding_path.exists():
            funding = load_frame(funding_path)["funding_rate"]
            midnight = funding[funding.index.hour == 0].resample("1D").sum()
            intraday = funding[funding.index.hour != 0].resample("1D").sum()
            bars["funding_open"] = midnight.reindex(bars.index).fillna(0.0)
            bars["funding_rate"] = intraday.reindex(bars.index).fillna(0.0)
        execution = TsmomExecution(candidate, symbol)
        asset_results.append(run_candidate(bars, execution))
        stress_results.append(run_candidate(bars, TsmomExecution(candidate, symbol, 2.0)))
        signed_positions.append(asset_results[-1].positions / 3.0)
    daily_returns = pd.concat(
        [result.equity.pct_change().rename(str(index)) for index, result in enumerate(asset_results)],
        axis=1,
    ).mean(axis=1).fillna(0.0)
    stress_returns = pd.concat(
        [result.equity.pct_change().rename(str(index)) for index, result in enumerate(stress_results)],
        axis=1,
    ).mean(axis=1).fillna(0.0)
    equity = 300.0 * (1.0 + daily_returns).cumprod()
    stress_equity = 300.0 * (1.0 + stress_returns).cumprod()
    position_frame = pd.concat(signed_positions, axis=1).reindex(equity.index).fillna(0.0)
    positions = position_frame.abs().sum(axis=1)
    turnover = position_frame.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = float(position_frame.iloc[0].abs().sum())
    trades = [trade for result in asset_results for trade in result.trades]
    trade_returns = pd.Series(
        [trade.return_fraction / 3.0 for trade in trades],
        index=pd.DatetimeIndex([trade.exit_time for trade in trades]),
        dtype=float,
    ).sort_index()
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    stress_is = stress_equity[stress_equity.index <= split]
    oos_stress = stress_equity[stress_equity.index > split]
    anchor = float(stress_is.iloc[-1]) if not stress_is.empty else 300.0
    stress_total = float(oos_stress.iloc[-1] / anchor - 1.0) if not oos_stress.empty else 0.0
    return StrategyResult(
        candidate_id=candidate.candidate_id,
        family="F2",
        equity=equity,
        trade_returns=trade_returns,
        positions=positions,
        turnover=turnover,
        stress_total_return=stress_total,
        metadata={
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "exploratory_only": False,
            "cost_model_valid": True,
            "intended_factor": "time_series_momentum",
            "max_concurrent_positions": 3,
            "max_position_weight": 1.0 / 3.0,
            "min_position_weight": 1.0 / 3.0,
            "min_order_usdt": 5.0,
            "candidate_config": {
                "rule": candidate.rule.value,
                "window": candidate.window,
                "long_only": candidate.long_only,
                "target_daily_vol": 0.015,
                "backtest_leverage_cap": 3.0,
            },
        },
    )
