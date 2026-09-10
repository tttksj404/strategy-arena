"""Causal signal adapters for the predeclared X-source hypotheses."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import numpy as np

from alt_search_models import MarketData
from x_discovery_pairs import PairChoice, pair_positions, select_pairs
from x_discovery_types import BoolVector, FloatMatrix, FloatVector


class SignalFamily(StrEnum):
    OU = "ou_pairs"
    RESIDUAL = "factor_residual"
    DONCHIAN = "donchian"
    PRESSURE = "candle_pressure_proxy"


class SignalInputError(ValueError):
    """Signal inputs cannot satisfy the declared causal contract."""


@dataclass(frozen=True, slots=True)
class SignalSpec:
    family: SignalFamily
    days: int
    horizon: int = 0
    direction: int = 1
    entry: float = 0.0

    @property
    def identifier(self) -> str:
        return f"{self.family}|d={self.days}|h={self.horizon}|dir={self.direction}|e={self.entry:g}"


def specifications(interval: str) -> tuple[SignalSpec, ...]:
    values = [SignalSpec(SignalFamily.OU, days, entry=entry)
              for days in (60, 120) for entry in (1.5, 2.)]
    values += [SignalSpec(SignalFamily.RESIDUAL, days, horizon, direction)
               for days in (30, 60) for horizon in (3, 7) for direction in (-1, 1)]
    values += [SignalSpec(SignalFamily.DONCHIAN, 90)]
    if interval == "1H":
        values += [SignalSpec(SignalFamily.PRESSURE, 1, horizon, entry=entry)
                   for horizon in (6, 24) for entry in (2., 3.)]
    return tuple(values)


def rolling_sum_before(values: FloatMatrix, window: int) -> FloatMatrix:
    result = np.zeros_like(values)
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    result[window:] = cumulative[window:-1] - cumulative[:-window-1]
    return result


def volume_ratio(volume: FloatMatrix, bars_per_day: int) -> FloatMatrix:
    """Article rule: completed six-hour volume / (completed 24-hour volume / 4)."""
    if bars_per_day % 4:
        raise SignalInputError("six-hour buckets require an interval dividing six hours")
    daily = rolling_sum_before(volume, bars_per_day)
    six_hour = rolling_sum_before(volume, bars_per_day // 4)
    result = np.divide(4 * six_hour, daily, out=np.zeros_like(volume), where=daily > 0)
    result[:bars_per_day] = 1
    return result


def _channel(values: FloatMatrix, window: int, *, high: bool) -> FloatMatrix:
    output = np.full_like(values, np.nan)
    for column in range(values.shape[1]):
        queue: deque[int] = deque()
        series: FloatVector = values[:, column] * (1 if high else -1)
        for index, value in enumerate(series):
            while queue and queue[0] <= index - window:
                _ = queue.popleft()
            while queue and series[queue[-1]] <= value:
                _ = queue.pop()
            queue.append(index)
            if index >= window - 1:
                output[index, column] = values[queue[0], column]
    return output


def donchian(data: MarketData, bars_per_day: int, *, days: int = 90, first_asset: int = 1) -> FloatMatrix:
    window = days * bars_per_day
    output = np.zeros_like(data.close_price)
    upper = _channel(data.high_price, window, high=True)
    lower = _channel(data.low_price, window, high=False)
    signal = (data.close_price[1:-1] > upper[:-2]).astype(float)
    signal -= (data.close_price[1:-1] < lower[:-2]).astype(float)
    output[2:] = signal
    output[:, :first_asset] = 0
    output *= rolling_sum_before(data.quote_volume, bars_per_day) >= 1e6
    output /= len(data.symbols) - first_asset
    return output


def factor_positions(data: MarketData, spec: SignalSpec, bars_per_day: int) -> FloatMatrix:
    returns: FloatMatrix = np.vstack((np.zeros((1, len(data.symbols))), np.diff(np.log(data.close_price), axis=0)))
    output = np.zeros_like(returns)
    window, horizon = spec.days * bars_per_day, spec.horizon * bars_per_day
    for index in range(window + 1, len(returns), bars_per_day):
        sample: FloatMatrix = returns[index-window:index]
        design: FloatMatrix = np.column_stack((np.ones(window), sample[:, :2]))
        beta: FloatMatrix = np.linalg.lstsq(design, sample[:, 2:], rcond=None)[0]
        residual: FloatMatrix = sample[:, 2:] - design @ beta
        scale: FloatVector = np.empty(residual.shape[1])
        horizon_root: FloatVector = np.empty(1)
        _ = np.std(residual, axis=0, out=scale)
        _ = np.sqrt(np.asarray([horizon], dtype=np.float64), out=horizon_root)
        _ = np.multiply(scale, horizon_root, out=scale)
        scores = np.divide(residual[-horizon:].sum(axis=0), scale,
                           out=np.zeros_like(scale), where=scale > 1e-9)
        rank = np.argsort(scores)
        weights: FloatVector = np.zeros(len(data.symbols))
        weights[rank[-2:] + 2] = .25 * spec.direction
        weights[rank[:2] + 2] = -.25 * spec.direction
        weights[:2] = -(beta[1:] @ weights[2:])
        normalizer = np.sum(np.abs(weights), dtype=np.float64).item()
        weights /= normalizer
        output[index:min(index + bars_per_day, len(output))] = weights
    return output


def pressure_positions(data: MarketData, spec: SignalSpec, bars_per_day: int) -> FloatMatrix:
    """Volume-spike candle continuation; NOT institutional orders, delta, or GEX."""
    output = np.zeros_like(data.close_price)
    daily = rolling_sum_before(data.quote_volume, bars_per_day)
    average = daily / bars_per_day
    ratio = np.divide(data.quote_volume, average, out=np.zeros_like(average), where=average > 0)
    spread = data.high_price - data.low_price
    location = np.divide(data.close_price - data.low_price, spread,
                         out=np.full_like(spread, .5), where=spread > 0)
    direction = np.where(location >= .8, 1., np.where(location <= .2, -1., 0.))
    candidate = direction * (ratio >= spec.entry) * (daily >= 1e6)
    candidate[:, 0] = 0
    held = np.zeros(len(data.symbols))
    age = np.zeros(len(data.symbols), dtype=int)
    for index in range(bars_per_day + 1, len(output)):
        age += 1
        held[age >= spec.horizon] = 0
        changed: BoolVector = np.not_equal(candidate[index - 1:index].reshape(-1), 0)
        held[changed] = candidate[index-1, changed]
        age[changed] = 0
        output[index] = held / (len(data.symbols) - 1)
    return output


def build_signal(data: MarketData, spec: SignalSpec, *, bars_per_day: int,
                 train_end: int, pool: tuple[PairChoice, ...] | None = None) -> tuple[FloatMatrix, tuple[PairChoice, ...]]:
    match spec.family:
        case SignalFamily.OU:
            choices = select_pairs(data, train_end, spec.days, bars_per_day) if pool is None else pool
            return pair_positions(data, choices, spec.days, spec.entry, bars_per_day, train_end), choices
        case SignalFamily.RESIDUAL:
            return factor_positions(data, spec, bars_per_day), ()
        case SignalFamily.DONCHIAN:
            return donchian(data, bars_per_day), ()
        case SignalFamily.PRESSURE:
            return pressure_positions(data, spec, bars_per_day), ()
        case _:
            assert_never(spec.family)
