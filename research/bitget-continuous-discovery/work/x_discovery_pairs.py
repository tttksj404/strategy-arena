"""Training-selected, past-only OU pair hypotheses for a paper experiment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from statsmodels.tsa.stattools import coint

from alt_search_models import MarketData
from x_discovery_types import FloatMatrix, FloatVector


@dataclass(frozen=True, slots=True)
class PairFit:
    beta: float
    intercept: float
    mean: float
    sigma: float
    half_life: float
    p_value: float


@dataclass(frozen=True, slots=True)
class PairChoice:
    left: int
    right: int
    p_value: float


def fit_pair(left: FloatVector, right: FloatVector, *, test: bool) -> PairFit | None:
    """Estimate spread and stationary AR(1) only from the supplied past window."""
    design = np.column_stack((np.ones(len(right)), right))
    coefficients: FloatVector = np.linalg.lstsq(design, left, rcond=None)[0]
    intercept: float = coefficients.item(0)
    beta: float = coefficients.item(1)
    if not .2 <= beta <= 5:
        return None
    spread = left - design @ np.asarray([intercept, beta])
    ar_design = np.column_stack((np.ones(len(spread) - 1), spread[:-1]))
    ar_coefficients: FloatVector = np.linalg.lstsq(ar_design, spread[1:], rcond=None)[0]
    alpha: float = ar_coefficients.item(0)
    phi: float = ar_coefficients.item(1)
    if not 0 < phi < 1:
        return None
    logarithms: FloatVector = np.empty(2)
    _ = np.log(np.asarray([2.0, phi]), out=logarithms)
    half_life = -logarithms.item(0) / logarithms.item(1)
    if not 2 <= half_life <= len(left) / 4:
        return None
    innovations: FloatVector = spread[1:] - ar_design @ np.asarray([alpha, phi])
    sigma_values: FloatVector = np.empty(1)
    divisor: FloatVector = np.empty(1)
    _ = np.std(innovations, ddof=2, keepdims=True, out=sigma_values)
    _ = np.sqrt(np.asarray([1 - phi * phi]), out=divisor)
    _ = np.divide(sigma_values, divisor, out=sigma_values)
    sigma = sigma_values.item()
    if sigma <= 1e-8:
        return None
    p_value = float(coint(left, right, maxlag=1, autolag=None)[1]) if test else 0.0
    return PairFit(float(beta), float(intercept), float(alpha / (1 - phi)),
                   sigma, half_life, p_value)


def select_pairs(data: MarketData, train_end: int, days: int, bars_per_day: int) -> tuple[PairChoice, ...]:
    """Freeze at most six eligible pairs using training data, never test prices."""
    logs = np.log(data.close_price[max(0, train_end - days * bars_per_day):train_end])
    pairs: list[PairChoice] = []
    for left, right in combinations(range(1, len(data.symbols)), 2):
        fit = fit_pair(logs[:, left], logs[:, right], test=True)
        if fit is not None and fit.p_value <= .05:
            pairs.append(PairChoice(left, right, fit.p_value))
    return tuple(sorted(pairs, key=lambda item: item.p_value)[:6])


def pair_positions(data: MarketData, pool: tuple[PairChoice, ...], days: int,
                   entry: float, bars_per_day: int, train_end: int) -> FloatMatrix:
    """Enter after a completed signal bar, and stop at mean crossing or invalid fit."""
    output = np.zeros_like(data.close_price)
    if not pool:
        return output
    logs: FloatMatrix = np.log(data.close_price)
    window = days * bars_per_day
    for pair in pool:
        fit: PairFit | None = None
        weekly_valid = True
        direction = 0
        opened = 0
        for index in range(max(window, train_end), len(logs)):
            elapsed = index - train_end
            if fit is None or elapsed % bars_per_day == 0:
                check = elapsed % (7 * bars_per_day) == 0
                fit = fit_pair(logs[index-window:index, pair.left],
                               logs[index-window:index, pair.right], test=check)
                if check:
                    weekly_valid = fit is not None and fit.p_value <= .05
            if fit is None or not weekly_valid:
                direction = 0
                continue
            left_value = logs[index - 1:index, pair.left:pair.left + 1].item()
            right_value = logs[index - 1:index, pair.right:pair.right + 1].item()
            spread = left_value - fit.intercept - fit.beta * right_value
            score = (spread - fit.mean) / fit.sigma
            expired = index - opened >= min(4 * fit.half_life, 10 * bars_per_day)
            if direction and (direction * score >= 0 or expired):
                direction = 0
            elif direction == 0 and abs(score) >= entry:
                direction = -1 if score > 0 else 1
                opened = index
            output[index, pair.left] += direction / (1 + fit.beta) / len(pool)
            output[index, pair.right] -= direction * fit.beta / (1 + fit.beta) / len(pool)
    return output
