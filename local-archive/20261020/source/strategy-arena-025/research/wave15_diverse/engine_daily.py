# Wave-15 generic daily delta-neutral bookkeeping loop, shared by B1 (dual yield), B2
# (directional USDT-collateral short), and C1 (predictive entry). Structurally this is
# research.wave13_liquidity.engine13._run_liquidity_loop's per-timestamp bookkeeping (gap PnL,
# intraday PnL, turnover cost, trade-close accounting, final forced unwind) copied ONE more
# time down the wave10->wave11->wave12->wave13 chain -- continuing that precedent rather than
# importing wave13's private loop function -- with exactly TWO substantive extensions SPEC.md
# actually needs for this wave:
#
#   1. `structure` picks the per-symbol return formula: "spot_perp" (B1, C1 -- identical to
#      every prior wave: spot_ret - perp_ret + funding, delta-neutral by construction) or
#      "perp_only_short" (B2 -- -(perp_ret) + funding, NO spot leg, i.e. NOT delta-neutral;
#      see gates15.gate_s1_structure's `delta_neutral_by_construction` flag, which reads this
#      same string).
#   2. `extra_annual_yield` adds a constant daily accrual (earn_apr/365) on top of the base
#      return while a position is open (B1's Simple Earn overlay / B2's USDT collateral
#      yield -- both ASSUMED, never fetched; see common15.ASSUMED_FLEXIBLE_EARN_APR).
#
# Signal computation (active_frame/score_frame) is deliberately NOT this module's job --
# every wave15 daily candidate uses a different entry/exit rule (B1/B2 reuse the plain
# realized-funding threshold; C1 uses signals15.py's fixed-coefficient predictive z-score),
# so callers build active_frame/score_frame themselves and this loop only consumes them.
#
# Implementation note: the hot loop is numpy-array-based (pre-extract every input frame with
# .to_numpy() once, iterate by integer position), not repeated pandas .loc[timestamp, ...]
# label lookups -- this wave's shared universe is breadth=200 x ~2500 daily bars (B1/B2 each
# run this loop 3x: base/stress/carry-only; C1 2x), and per-call label-lookup overhead at that
# scale was the dominant cost in an earlier .loc-based draft. Every arithmetic step below is
# still exactly research.wave13_liquidity.engine13._run_liquidity_loop's formula, just reading
# from a numpy row instead of a pandas Series each iteration -- tests/test_wave15.py's B1/B2
# tests pin the numeric behavior this refactor must preserve.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final, Literal

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave15_diverse.common15 import Wave10Result

Structure = Literal["spot_perp", "perp_only_short"]

STRUCTURES: Final = ("spot_perp", "perp_only_short")


def run_generic_carry(
    price_frames: dict[str, pd.DataFrame],
    active_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    top_k: int,
    leg_fraction: float,
    cost_rate_frame: pd.DataFrame,
    structure: Structure = "spot_perp",
    extra_annual_yield: float = 0.0,
    capital_start: float | None = None,
) -> tuple[Wave10Result, float]:
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure {structure!r}, expected one of {STRUCTURES}")
    from research.wave15_diverse.common15 import ACTIVE_CAPITAL

    spot_open_frame = price_frames["spot_open"]
    symbols = tuple(spot_open_frame.columns)
    index = spot_open_frame.index
    n, n_symbols = len(index), len(symbols)

    spot_open = spot_open_frame.to_numpy(dtype=float)
    spot_close = price_frames["spot_close"].reindex(columns=symbols).to_numpy(dtype=float)
    perp_open = price_frames["perp_open"].reindex(columns=symbols).to_numpy(dtype=float)
    perp_close = price_frames["perp_close"].reindex(columns=symbols).to_numpy(dtype=float)
    funding = price_frames["funding_daily"].reindex(columns=symbols).to_numpy(dtype=float)
    active = active_frame.reindex(index=index, columns=symbols).to_numpy(dtype=float)
    score = score_frame.reindex(index=index, columns=symbols).to_numpy(dtype=float)
    cost_rate = cost_rate_frame.reindex(index=index, columns=symbols).to_numpy(dtype=float)

    spot_close_prev = np.vstack([np.full(n_symbols, np.nan), spot_close[:-1]])
    perp_close_prev = np.vstack([np.full(n_symbols, np.nan), perp_close[:-1]])
    available = np.isfinite(spot_open) & np.isfinite(spot_close) & np.isfinite(perp_open) & np.isfinite(perp_close)

    extra_daily_yield = extra_annual_yield / 365.0
    capital = ACTIVE_CAPITAL if capital_start is None else capital_start

    equity = np.empty(n, dtype=float)
    turnover_arr = np.empty(n, dtype=float)
    exposure_arr = np.empty(n, dtype=float)
    concurrent_arr = np.empty(n, dtype=int)
    trade_values: list[float] = []
    trade_times: list[pd.Timestamp] = []
    total_cost_usdt = 0.0

    previous_weights = np.zeros(n_symbols, dtype=float)
    trade_growth: dict[int, float] = {}
    trade_weight_at_open: dict[int, float] = {}

    for i in range(n):
        timestamp = index[i]
        eligible_mask = (active[i] > 0.0) & available[i]
        weights = np.zeros(n_symbols, dtype=float)
        if eligible_mask.any():
            eligible_scores = np.where(eligible_mask, score[i], np.nan)
            if np.isfinite(eligible_scores).any():
                ranked = np.argsort(np.where(np.isfinite(eligible_scores), eligible_scores, -np.inf))[::-1][:top_k]
                ranked = [idx for idx in ranked if np.isfinite(eligible_scores[idx])]
                weights[ranked] = leg_fraction

        spot_gap = spot_open[i] / spot_close_prev[i] - 1.0
        perp_gap = perp_open[i] / perp_close_prev[i] - 1.0
        if structure == "spot_perp":
            gap_by_symbol = np.nan_to_num(spot_gap - perp_gap, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            gap_by_symbol = np.nan_to_num(-perp_gap, nan=0.0, posinf=0.0, neginf=0.0)
        capital *= 1.0 + float(np.dot(gap_by_symbol, previous_weights))

        weight_delta = weights - previous_weights
        turnover = float(np.abs(weight_delta).sum())
        cost_return = float(np.dot(np.abs(weight_delta), cost_rate[i]))
        capital_before_cost = capital
        capital *= 1.0 - cost_return
        total_cost_usdt += capital_before_cost - capital

        if structure == "spot_perp":
            base_intraday = spot_close[i] / spot_open[i] - perp_close[i] / perp_open[i]
        else:
            base_intraday = -(perp_close[i] / perp_open[i] - 1.0)
        intraday = np.nan_to_num(base_intraday + funding[i] + extra_daily_yield, nan=0.0, posinf=0.0, neginf=0.0)
        capital *= 1.0 + float(np.dot(intraday, weights))

        for symbol_idx in range(n_symbols):
            previous_weight = float(previous_weights[symbol_idx])
            current_weight = float(weights[symbol_idx])
            leg_rate = float(cost_rate[i, symbol_idx])
            if previous_weight > 0.0 and symbol_idx in trade_growth:
                trade_growth[symbol_idx] *= 1.0 + float(gap_by_symbol[symbol_idx])
            if previous_weight > 0.0 and current_weight == 0.0:
                trade_growth[symbol_idx] *= 1.0 - leg_rate
                trade_values.append((trade_growth.pop(symbol_idx) - 1.0) * trade_weight_at_open.pop(symbol_idx))
                trade_times.append(pd.Timestamp(timestamp))
            elif previous_weight == 0.0 and current_weight > 0.0:
                trade_growth[symbol_idx] = 1.0 - leg_rate
                trade_weight_at_open[symbol_idx] = current_weight
            elif previous_weight > 0.0 and current_weight > 0.0 and previous_weight != current_weight:
                trade_growth[symbol_idx] *= 1.0 - abs(current_weight - previous_weight) * leg_rate / max(current_weight, previous_weight)
                trade_weight_at_open[symbol_idx] = current_weight
            if current_weight > 0.0:
                trade_growth[symbol_idx] *= 1.0 + float(intraday[symbol_idx])

        equity[i] = capital
        turnover_arr[i] = turnover
        exposure_arr[i] = float(np.abs(weights).sum())
        concurrent_arr[i] = int((weights != 0.0).sum())
        previous_weights = weights

    if n > 0 and float(np.abs(previous_weights).sum()) > 0.0:
        final_timestamp = pd.Timestamp(index[-1])
        final_cost = float(np.dot(previous_weights, cost_rate[-1]))
        capital_before_final_cost = capital
        capital *= 1.0 - final_cost
        total_cost_usdt += capital_before_final_cost - capital
        equity[-1] = capital
        turnover_arr[-1] += float(np.abs(previous_weights).sum())
        for symbol_idx, growth in trade_growth.items():
            leg_rate = float(cost_rate[-1, symbol_idx])
            trade_values.append((growth * (1.0 - leg_rate) - 1.0) * trade_weight_at_open[symbol_idx])
            trade_times.append(final_timestamp)

    equity_series = pd.Series(equity, index=index, dtype=float)
    positions_series = pd.Series(exposure_arr, index=index, dtype=float)
    turnover_series = pd.Series(turnover_arr, index=index, dtype=float)
    trades = pd.Series(trade_values, index=pd.DatetimeIndex(trade_times), dtype=float).sort_index()
    result = Wave10Result(
        equity=equity_series,
        positions=positions_series,
        turnover=turnover_series,
        trade_returns=trades,
        max_concurrent_positions=int(concurrent_arr.max()) if n > 0 else 0,
        symbols_used=symbols,
    )
    return result, total_cost_usdt


__all__ = ["Structure", "STRUCTURES", "run_generic_carry"]
