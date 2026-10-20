# Runs no-lookahead event backtests with costs, stops, funding, and compounding.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import PipelineError
from research.wave1.costs import LegCost, funding_cashflow, transaction_cost


REQUIRED_COLUMNS: Final = frozenset({"open", "high", "low", "close"})


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_capital: float = 300.0
    fee_rate: float = 0.0006
    slippage_rate: float = 0.0001
    stop_distance: float | pd.Series | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    timestamp: pd.Timestamp
    price: float
    position: float
    reason: str
    cost: float


@dataclass(frozen=True, slots=True)
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: float
    entry_price: float
    exit_price: float
    pnl: float
    return_fraction: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity: pd.Series
    fills: tuple[Fill, ...]
    trades: tuple[Trade, ...]
    positions: pd.Series


@dataclass(frozen=True, slots=True)
class StopState:
    row: pd.Series
    reference_price: float
    position: float
    distance: float


def _stop_fill(state: StopState) -> float | None:
    open_price = float(state.row["open"])
    direction = 1.0 if state.position > 0.0 else -1.0
    level = state.reference_price * (1.0 - state.distance * direction)
    if state.position > 0.0 and open_price <= level:
        level = open_price
    elif state.position < 0.0 and open_price >= level:
        level = open_price
    if state.position > 0.0 and float(state.row["low"]) <= level:
        return level
    if state.position < 0.0 and float(state.row["high"]) >= level:
        return level
    return None


def run_backtest(
    bars: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    missing = REQUIRED_COLUMNS.difference(bars.columns)
    if missing:
        raise PipelineError(f"missing OHLC columns: {sorted(missing)}")
    ordered = bars.sort_index().copy()
    if ordered.empty:
        raise PipelineError("OHLC bars must not be empty")
    desired = signals.reindex(ordered.index).fillna(0.0).shift(1).fillna(0.0)
    if isinstance(config.stop_distance, pd.Series):
        stop_distances = config.stop_distance.reindex(ordered.index).shift(1)
    else:
        stop_distances = pd.Series(config.stop_distance, index=ordered.index, dtype=float)
    leg = LegCost(config.fee_rate, config.slippage_rate)
    equity = config.initial_capital
    curve = pd.Series(index=ordered.index, dtype=float)
    position_curve = pd.Series(0.0, index=ordered.index, dtype=float)
    fills: list[Fill] = []
    trades: list[Trade] = []
    position = 0.0
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    entry_equity = equity
    trailing_reference = 0.0
    previous_close = float(ordered.iloc[0]["close"])
    curve.iloc[0] = equity

    for offset in range(1, len(ordered)):
        timestamp = pd.Timestamp(ordered.index[offset])
        row = ordered.iloc[offset]
        open_price = float(row["open"])
        equity *= 1.0 + position * (open_price / previous_close - 1.0)
        if "funding_open" in ordered.columns and position != 0.0:
            equity += funding_cashflow(equity, float(row["funding_open"]), position)

        target = float(desired.iloc[offset])
        if target != position:
            if position != 0.0 and entry_time is not None:
                cost = transaction_cost(equity * abs(position), leg)
                equity -= cost
                fills.append(Fill(timestamp, open_price, 0.0, "exit", cost))
                pnl = equity - entry_equity
                trades.append(Trade(entry_time, timestamp, position, entry_price, open_price, pnl, pnl / entry_equity))
            position = target
            if position != 0.0:
                entry_equity = equity
                cost = transaction_cost(equity * abs(position), leg)
                equity -= cost
                fills.append(Fill(timestamp, open_price, position, "entry", cost))
                entry_price = open_price
                entry_time = timestamp
                trailing_reference = open_price
            else:
                entry_time = None

        if "funding_rate" in ordered.columns and position != 0.0:
            equity += funding_cashflow(equity, float(row["funding_rate"]), position)
        exit_price = float(row["close"])
        stop_fill = None
        distance = float(stop_distances.iloc[offset]) if pd.notna(stop_distances.iloc[offset]) else 0.0
        if position != 0.0 and distance > 0.0:
            stop_fill = _stop_fill(StopState(row, trailing_reference, position, distance))
            if stop_fill is not None:
                exit_price = stop_fill
        equity *= 1.0 + position * (exit_price / open_price - 1.0)
        if stop_fill is not None and entry_time is not None:
            cost = transaction_cost(equity * abs(position), leg)
            equity -= cost
            fills.append(Fill(timestamp, stop_fill, 0.0, "stop", cost))
            pnl = equity - entry_equity
            trades.append(Trade(entry_time, timestamp, position, entry_price, stop_fill, pnl, pnl / entry_equity))
            position = 0.0
            entry_time = None
        elif position > 0.0:
            trailing_reference = max(trailing_reference, float(row["high"]))
        elif position < 0.0:
            trailing_reference = min(trailing_reference, float(row["low"]))
        curve.iloc[offset] = equity
        position_curve.iloc[offset] = position
        previous_close = float(row["close"])

    if position != 0.0 and entry_time is not None:
        timestamp = pd.Timestamp(ordered.index[-1])
        close_price = float(ordered.iloc[-1]["close"])
        cost = transaction_cost(equity * abs(position), leg)
        equity -= cost
        fills.append(Fill(timestamp, close_price, 0.0, "final_exit", cost))
        pnl = equity - entry_equity
        trades.append(Trade(entry_time, timestamp, position, entry_price, close_price, pnl, pnl / entry_equity))
        curve.iloc[-1] = equity
        position_curve.iloc[-1] = 0.0
    return BacktestResult(equity=curve, fills=tuple(fills), trades=tuple(trades), positions=position_curve)
