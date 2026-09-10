"""Path-aware signal diagnostics for paper-only leveraged strategy research."""
# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from alt_search_models import FloatMatrix, MarketData


class SurvivalMetricError(ValueError):
    """Raised when a path-survival diagnostic cannot be evaluated."""


class NoSurvivalEventsError(SurvivalMetricError):
    """Raised when a valid segment contains no signal-entry events."""


@dataclass(frozen=True, slots=True)
class SurvivalSpec:
    start: int
    end: int
    horizon_bars: int
    leverage: float
    round_trip_cost: float
    liquidation_reserve: float


@dataclass(frozen=True, slots=True)
class SurvivalMetrics:
    event_count: int
    cost_adjusted_terminal_win_rate: float
    market_proxy_survival_rate: float
    profitable_survivor_rate: float
    eventual_winner_market_proxy_liquidation_rate: float
    mean_signal_equity_delta: float
    mae_median: float
    mae_p95: float
    mfe_median: float
    survivor_time_to_profit_bars_median: float | None
    p95_market_proxy_leverage_boundary: float
    exact_bitget_liquidation: bool


def _validate(data: MarketData, positions: FloatMatrix, spec: SurvivalSpec) -> None:
    rows, assets = data.close_price.shape
    price_panels = (data.open_price, data.high_price, data.low_price, data.close_price)
    if positions.shape != (rows, assets) or any(panel.shape != (rows, assets) for panel in price_panels):
        raise SurvivalMetricError("price and position panels must have identical two-dimensional shapes")
    if rows != len(data.dates) or not all(np.isfinite(panel).all() for panel in price_panels):
        raise SurvivalMetricError("price panels must be finite and aligned to the timestamp index")
    if not np.isfinite(positions).all():
        raise SurvivalMetricError("positions must be finite")
    if any(
        not isfinite(value)
        for value in (spec.leverage, spec.round_trip_cost, spec.liquidation_reserve)
    ):
        raise SurvivalMetricError("leverage, costs, and liquidation reserve must be finite")
    if any(np.any(panel <= 0.0) for panel in price_panels):
        raise SurvivalMetricError("path-survival prices must be positive")
    if np.any(data.high_price < np.maximum(data.open_price, data.close_price)) or np.any(
        data.low_price > np.minimum(data.open_price, data.close_price)
    ):
        raise SurvivalMetricError("high/low panels must enclose open and close prices")
    if spec.start < 1 or spec.end > rows or spec.start >= spec.end:
        raise SurvivalMetricError("evaluation bounds must be chronological and leave a prior position row")
    if spec.horizon_bars < 1 or spec.horizon_bars > spec.end - spec.start:
        raise SurvivalMetricError("horizon must fit entirely inside the evaluation segment")
    if spec.leverage <= 0.0 or spec.round_trip_cost < 0.0 or spec.liquidation_reserve < 0.0:
        raise SurvivalMetricError("leverage must be positive and cost assumptions must be non-negative")
    if 1.0 / spec.leverage <= spec.liquidation_reserve:
        raise SurvivalMetricError("liquidation reserve leaves no positive adverse-excursion buffer")


def evaluate_signal_survival(data: MarketData, positions: FloatMatrix, spec: SurvivalSpec) -> SurvivalMetrics:
    """Measure sign-change entries using next-open execution and strategy exits.

    Each entry is weighted equally and ends at the first zero/opposite sign or the
    fixed horizon. Same-sign size changes are intentionally ignored. A profitable
    terminal direction can still be recorded as liquidated when its preceding
    market-candle adverse excursion reaches the leverage buffer. The result is a
    signal diagnostic, not an account path or an exact Bitget liquidation model.
    """

    _validate(data, positions, spec)
    adverse_excursions: list[float] = []
    favorable_excursions: list[float] = []
    equity_deltas: list[float] = []
    survivor_time_to_profit: list[int] = []
    direction_wins = 0
    survivors = 0
    profitable_survivors = 0
    winner_liquidations = 0
    liquidation_buffer = 1.0 / spec.leverage - spec.liquidation_reserve
    last_entry = spec.end - spec.horizon_bars

    for index in range(spec.start, last_entry + 1):
        current_signs = np.sign(positions[index])
        prior_signs = np.sign(positions[index - 1])
        event_assets = np.flatnonzero((current_signs != 0.0) & (current_signs != prior_signs))
        for asset in event_assets:
            direction = float(current_signs[asset])
            entry = float(data.open_price[index, asset])
            planned_end = index + spec.horizon_bars
            future_signs = np.sign(positions[index + 1 : planned_end, asset])
            changed = np.flatnonzero(future_signs != direction)
            exit_index = index + 1 + int(changed[0]) if changed.size else planned_end
            path_low = float(data.low_price[index:exit_index, asset].min())
            path_high = float(data.high_price[index:exit_index, asset].max())
            if exit_index < planned_end:
                terminal = float(data.open_price[exit_index, asset])
                path_low = min(path_low, terminal)
                path_high = max(path_high, terminal)
            else:
                terminal = float(data.close_price[planned_end - 1, asset])
            gross_terminal = direction * (terminal / entry - 1.0)
            if direction > 0.0:
                adverse = max(0.0, (entry - path_low) / entry)
                favorable = max(0.0, (path_high - entry) / entry)
            else:
                adverse = max(0.0, (path_high - entry) / entry)
                favorable = max(0.0, (entry - path_low) / entry)
            survives = adverse < liquidation_buffer
            wins = gross_terminal > spec.round_trip_cost
            adverse_excursions.append(adverse)
            favorable_excursions.append(favorable)
            survivors += int(survives)
            direction_wins += int(wins)
            profitable_survivors += int(wins and survives)
            winner_liquidations += int(wins and not survives)
            equity_deltas.append(spec.leverage * (gross_terminal - spec.round_trip_cost) if survives else -1.0)
            if wins and survives:
                signed_path = direction * (data.close_price[index:exit_index, asset] / entry - 1.0)
                first_profitable = np.flatnonzero(signed_path > spec.round_trip_cost)
                if first_profitable.size:
                    survivor_time_to_profit.append(int(first_profitable[0]) + 1)
                elif exit_index < planned_end:
                    survivor_time_to_profit.append(exit_index - index + 1)

    if not adverse_excursions:
        raise NoSurvivalEventsError("no entry events fit the requested segment and horizon")
    event_count = len(adverse_excursions)
    adverse_values = np.asarray(adverse_excursions, dtype=float)
    favorable_values = np.asarray(favorable_excursions, dtype=float)
    mae_p95 = float(np.quantile(adverse_values, 0.95))
    return SurvivalMetrics(
        event_count=event_count,
        cost_adjusted_terminal_win_rate=direction_wins / event_count,
        market_proxy_survival_rate=survivors / event_count,
        profitable_survivor_rate=profitable_survivors / event_count,
        eventual_winner_market_proxy_liquidation_rate=(
            winner_liquidations / direction_wins if direction_wins else 0.0
        ),
        mean_signal_equity_delta=float(np.mean(equity_deltas)),
        mae_median=float(np.median(adverse_values)),
        mae_p95=mae_p95,
        mfe_median=float(np.median(favorable_values)),
        survivor_time_to_profit_bars_median=(
            float(np.median(survivor_time_to_profit)) if survivor_time_to_profit else None
        ),
        p95_market_proxy_leverage_boundary=1.0 / (mae_p95 + spec.liquidation_reserve),
        exact_bitget_liquidation=False,
    )
