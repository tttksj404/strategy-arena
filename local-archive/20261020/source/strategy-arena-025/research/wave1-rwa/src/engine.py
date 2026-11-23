"""Paper-fidelity leveraged backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEE_RATE = 0.0006
SLIPPAGE = 0.0001
MAINTENANCE = 0.005


@dataclass(frozen=True, slots=True)
class Costs:
    """Execution cost configuration."""

    half_spread_bp: float = 1.0
    taker_rate: float = FEE_RATE
    slippage_rate: float = SLIPPAGE

    @property
    def round_trip_rate(self) -> float:
        """Return the one-entry/one-exit cost rate."""
        return 2 * (self.taker_rate + max(1.0, self.half_spread_bp) / 10_000 + self.slippage_rate)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Backtest output used by the gates and leaderboard."""

    equity: pd.Series
    net_return: float
    cagr: float
    mdd: float
    sharpe: float
    win_rate: float
    trades: int
    fees_paid: float
    funding_paid: float
    liquidated: bool
    turnover: float
    funding_coverage: float
    trade_returns: tuple[float, ...]


def _funding_map(funding: pd.DataFrame, index: pd.DatetimeIndex) -> tuple[pd.Series, float]:
    """Align funding events to bars and return coverage."""
    if funding.empty:
        return pd.Series(0.0, index=index), 0.0
    rates = funding.set_index("ts")["rate"].sort_index()
    matched = rates.reindex(index)
    coverage = float(matched.notna().mean())
    return matched.fillna(0.0), coverage


def run_backtest(candles: pd.DataFrame, signal: pd.Series, funding: pd.DataFrame, leverage: int,
                 costs: Costs, initial_equity: float = 300.0) -> BacktestResult:
    """Run signals with t-close decision and t+1-open execution."""
    if candles.empty or len(candles) != len(signal):
        raise ValueError("candles and signal must have equal non-zero length")
    frame = candles.reset_index(drop=True)
    decision = pd.Series(signal.to_numpy(dtype=float), index=frame.index).fillna(0.0).clip(-1, 1)
    target = decision.shift(1).fillna(0.0).to_numpy()
    fund_rates, coverage = _funding_map(funding, pd.DatetimeIndex(frame["ts"]))
    equity = initial_equity
    position = 0.0
    entry = 0.0
    fees = 0.0
    funding_paid = 0.0
    turnover = 0.0
    liquidated = False
    trade_returns: list[float] = []
    open_trade_equity = equity
    values: list[float] = []
    for i, row in frame.iterrows():
        open_price, close_price = float(row["open"]), float(row["close"])
        if i > 0 and position:
            equity *= 1 + position * leverage * (open_price / float(frame.iloc[i - 1]["close"]) - 1)
            if equity <= 0:
                liquidated = True
                equity = 0.0
                position = 0.0
                values.extend([0.0] * (len(frame) - i))
                break
        next_position = float(target[i])
        if next_position != position:
            side_rate = costs.taker_rate + max(1.0, costs.half_spread_bp) / 10_000 + costs.slippage_rate
            if position:
                exit_notional = equity * leverage * abs(position)
                exit_cost = exit_notional * side_rate
                equity = max(0.0, equity - exit_cost)
                fees += exit_cost
                turnover += exit_notional
                if open_trade_equity:
                    trade_returns.append(equity / open_trade_equity - 1)
            if next_position:
                open_trade_equity = equity
                entry_notional = equity * leverage * abs(next_position)
                entry_cost = entry_notional * side_rate
                equity = max(0.0, equity - entry_cost)
                fees += entry_cost
                turnover += entry_notional
                entry = open_price
            position = next_position
        if position:
            funding_cost = equity * leverage * position * float(fund_rates.iloc[i])
            equity -= funding_cost
            funding_paid += funding_cost
            if (position > 0 and float(row["low"]) <= entry * (1 - 1 / leverage + MAINTENANCE)) or (position < 0 and float(row["high"]) >= entry * (1 + 1 / leverage - MAINTENANCE)):
                liquidated = True
                equity = 0.0
                position = 0.0
                values.extend([0.0] * (len(frame) - i))
                break
            equity *= 1 + position * leverage * (close_price / open_price - 1)
        values.append(max(0.0, equity))
    if position and open_trade_equity:
        trade_returns.append(equity / open_trade_equity - 1)
    series = pd.Series(values, index=frame["ts"].iloc[:len(values)]).replace([np.inf, -np.inf], np.nan).ffill().fillna(initial_equity)
    daily = series.resample("1D").last().dropna()
    daily_returns = daily.pct_change().dropna()
    sharpe = float(np.sqrt(365) * daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 1 and daily_returns.std() else 0.0
    peak = series.cummax()
    mdd = float((1 - series / peak).max()) if not series.empty else 1.0
    years = max(1 / 365, (series.index[-1] - series.index[0]).total_seconds() / 31_557_600) if len(series) > 1 else 1 / 365
    net = float(series.iloc[-1] / initial_equity - 1) if not series.empty else -1.0
    return BacktestResult(series, net, float((series.iloc[-1] / initial_equity) ** (1 / years) - 1) if series.iloc[-1] > 0 else -1.0,
                          mdd, sharpe, float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0,
                          len(trade_returns), fees, funding_paid, liquidated, turnover, coverage, tuple(trade_returns))
