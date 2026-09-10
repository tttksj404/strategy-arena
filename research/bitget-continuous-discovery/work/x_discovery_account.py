"""Next-open, absorbing-ruin account simulation; not an exchange margin engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alt_search_models import MarketData
from x_discovery_types import FloatMatrix, FloatVector


class AccountInputError(ValueError):
    """Inputs cannot support a causally valid account evaluation."""


@dataclass(frozen=True, slots=True)
class Execution:
    start: int
    end: int
    bars_per_day: int
    cost: float
    carry: float


@dataclass(frozen=True, slots=True)
class AccountPath:
    equity: FloatVector
    returns: FloatVector
    liquidation_index: int
    active_bars: int
    turnover: float
    average_gross: float


def simulate(data: MarketData, weights: FloatMatrix, execution: Execution) -> AccountPath:
    """Trade weights known at each open; the end index is a closing fill only."""
    start, end = execution.start, execution.end
    if (weights.shape != data.open_price.shape or not np.isfinite(weights).all()
            or not 0 <= start < end < len(data.dates) or execution.bars_per_day < 1
            or execution.cost < 0 or execution.carry < 0):
        raise AccountInputError("invalid execution bounds, costs, or exposure")
    relevant = slice(start, end + 1)
    if any(not np.isfinite(matrix[relevant]).all() or np.any(matrix[relevant] <= 0)
           for matrix in (data.open_price, data.high_price, data.low_price)):
        raise AccountInputError("non-positive or non-finite prices")
    equity: FloatVector = np.zeros(end - start + 1)
    equity[0] = 1.0
    previous_quantity: FloatVector = np.zeros(weights.shape[1])
    active = 0
    turnover = 0.0
    gross_sum = 0.0
    liquidation = -1
    for local, index in enumerate(range(start, end)):
        opening: FloatVector = data.open_price[index:index + 1].reshape(-1)
        next_open: FloatVector = data.open_price[index + 1:index + 2].reshape(-1)
        capital = equity.item(local)
        target: FloatVector = weights[index:index + 1].reshape(-1) * capital
        quantity: FloatVector = target / opening
        traded = np.sum(np.abs(target - previous_quantity * opening), dtype=np.float64).item()
        gross = np.sum(np.abs(target), dtype=np.float64).item()
        fee = execution.cost * traded
        financing = execution.carry * gross / execution.bars_per_day
        net_capital = capital - fee - financing
        low: FloatVector = data.low_price[index:index + 1].reshape(-1)
        high: FloatVector = data.high_price[index:index + 1].reshape(-1)
        worst_price: FloatVector = np.where(quantity >= 0, np.minimum(low, next_open),
                                            np.maximum(high, next_open))
        worst_equity = net_capital + float(quantity @ (worst_price - opening))
        close_equity = net_capital + float(quantity @ (next_open - opening))
        turnover += traded / capital
        gross_sum += gross / capital
        active += int(gross > 0)
        if (gross > 0 and worst_equity <= 0.02 * gross) or close_equity <= 0:
            liquidation = index
            break
        equity[local + 1] = close_equity
        previous_quantity = quantity
    if liquidation < 0:
        final_open: FloatVector = data.open_price[end:end + 1].reshape(-1)
        closing_notional = np.sum(np.abs(previous_quantity * final_open), dtype=np.float64).item()
        turnover += closing_notional / equity.item(-1)
        equity[-1] = max(0.0, equity.item(-1) - execution.cost * closing_notional)
        if equity.item(-1) == 0:
            liquidation = end
    returns = np.divide(equity[1:], equity[:-1], out=np.ones_like(equity[1:]),
                        where=equity[:-1] > 0) - 1
    return AccountPath(equity, returns, liquidation, active, turnover, gross_sum / (end - start))


def shadow_returns(data: MarketData, weights: FloatMatrix, bars_per_day: int) -> FloatVector:
    """Causal one-unit return observations for sizing, not a compound PnL claim."""
    ratios = data.open_price[1:] / data.open_price[:-1]
    held = weights[:-1]
    drifted = np.vstack((np.zeros((1, held.shape[1])), held[:-1] * ratios[:-1]))
    turnover = np.abs(held - drifted).sum(axis=1)
    values = (held * (ratios - 1)).sum(axis=1) - .0009 * turnover
    values -= .0001 * np.abs(held).sum(axis=1) / bars_per_day
    return np.append(values, 0.0)


def kelly_schedule(returns: FloatVector, window: int, conservative: bool) -> FloatVector:
    """Quarter Kelly estimated strictly before index i; no future-return access."""
    if window < 2 or returns.ndim != 1 or not np.isfinite(returns).all():
        raise AccountInputError("invalid Kelly observations")
    output = np.zeros_like(returns)
    if len(returns) <= window:
        return output
    cumulative = np.concatenate(([0.0], np.cumsum(returns)))
    squared = np.concatenate(([0.0], np.cumsum(returns * returns)))
    means = (cumulative[window:-1] - cumulative[:-window-1]) / window
    variance = np.maximum((squared[window:-1] - squared[:-window-1]) / window - means**2, 0)
    if conservative:
        means -= np.sqrt(variance / window)
    estimate = np.divide(.25 * means, variance, out=np.zeros_like(means), where=variance > 1e-14)
    output[window:] = np.clip(estimate, 0, 5)
    return output


def daily_returns(path: AccountPath, bars_per_day: int) -> FloatVector:
    """Compound consecutive bars into days anchored to the evaluation start."""
    return np.asarray([np.prod(1 + path.returns[start:start + bars_per_day]) - 1
                       for start in range(0, len(path.returns), bars_per_day)])
