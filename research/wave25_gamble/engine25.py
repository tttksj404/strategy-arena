# Wave-25 convex-gamble engine (B0-B7). See research/wave25_gamble/SPEC.md for the frozen
# pre-registration this module implements, and configs25.py for every numeric threshold.
#
# ---------------------------------------------------------------------------------------
# B0 is NOT reimplemented here
# ---------------------------------------------------------------------------------------
# SPEC.md: "B0 = wave-20 V1 재현 ... 엔진 research/wave20_convex/engine20.py 참고·재사용." run_b0
# below calls research.wave20_convex.engine20.run_v1 directly (backtest path) or replays its
# exact own array-level logic against fresher data (live path, _run_b0_with_hourly) -- neither
# path re-derives V1's math independently, so B0's numbers are guaranteed identical to wave20's
# own V1 results (skew 1.76, $138.48 final, 153 trades) rather than a lookalike that could
# silently drift from the literal baseline this whole tournament is measured against.
#
# ---------------------------------------------------------------------------------------
# Convex position lifecycle (B1-B7) -- SPEC.md "볼록 구조 강제"
# ---------------------------------------------------------------------------------------
# Every B1-B7 position goes through the SAME state machine (simulate_convex_directional,
# driven by run_multi_symbol_convex), independent of which indicator generated its entry
# signal:
#   1. Entry fills at the bar AFTER the signal bar's open (t -> t+1 discipline, identical to
#      engine20.simulate_breakout_reversal's own convention).
#   2. A HARD stop-loss is fixed the instant a position opens: stop_distance =
#      min(HARD_STOP_PCT * entry_price, HARD_STOP_ATR_MULT * ATR_at_entry) -- "진입가 -3% 또는
#      -1×ATR 중 가까운 쪽" (SPEC.md): whichever of the two distances is SMALLER (tighter to
#      entry) is the one that binds. This never moves for the life of the trade.
#   3. NO fixed take-profit exists anywhere in this module. The only way to lock in a
#      favorable move is a TRAILING stop: once the position has moved
#      TRAILING_ACTIVATE_ATR_MULT * ATR_at_entry in its favor, the trailing stop arms at
#      breakeven (entry_price) and from then on only ever tightens toward the running
#      favorable price extreme, trailing TRAILING_ATR_MULT * ATR(now) behind it. Because it
#      seeds at breakeven and only ever improves, a trailing exit can never itself produce a
#      loss worse than roughly the entry cost -- the asymmetry (small fixed downside, running
#      unbounded upside) is structural, not tuned per candidate, which is what SPEC.md's own
#      "대칭(±동일폭) 설정 금지" rules out by construction (there is no second fixed distance
#      anywhere that could accidentally mirror the stop).
#   4. Both the hard stop and the trailing stop are checked INTRABAR (against that bar's own
#      high/low, not just its close) and fill gap-aware: if the bar's own OPEN already gapped
#      through the stop level, the fill is the (worse) open price, not the stale stop level --
#      matches research.wave20_convex.engine20.run_v4's own stop/take-profit convention
#      exactly (`if low <= sl_price: exit at sl_price`), just extended with the gap check.
#   5. A single $25-sleeve, single-symbol-at-a-time, BTC/ETH/SOL priority-scan mechanic (one
#      open position across the whole sleeve at any time) -- the same convention
#      engine20.run_v4 already uses for its own BTC/ETH/SOL scan, extended here with the
#      stop/trailing lifecycle above instead of V4's fixed +3%/-3%.
#   6. Isolated-margin liquidation floor at $0 (never negative) -- identical disclosed
#      simplification to engine20's own module docstring (a leg is force-closed the instant
#      its mark-to-market loss would reach -100% of the capital committed to it). The hard
#      stop above makes this a rare tail event, not the ordinary exit path; gates25.py's P2
#      checks the REALIZED dollar size of any such event explicitly (a stop or liquidation on
#      a since-grown sleeve can still exceed the original $25 allocation in dollar terms even
#      though it can never exceed the CURRENT sleeve's own equity -- P2 is the gate that
#      catches this, not this module).
#
# ---------------------------------------------------------------------------------------
# Cost model -- reused unmodified from wave20
# ---------------------------------------------------------------------------------------
# Single-leg cost (maker fee once + measured-slippage mapping once per entry/exit
# transition), identical convention and identical reused functions
# (research.wave20_convex.engine20.one_leg_cost_rate_series / worst_case_cost, themselves
# thin wrappers around research.wave13_liquidity.costs_measured, unmodified). See
# engine20.py's own module docstring for why single-leg (one outright directional
# instrument), not costs_measured.cost_rate_from_bp's 2x spot+perp carry-pair convention.

from __future__ import annotations

from dataclasses import dataclass
import functools
from pathlib import Path
import sys
from typing import Any, Callable, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave13_liquidity import costs_measured
from research.wave13_liquidity.costs_measured import MeasuredCostMapping
from research.wave20_convex import dataio20, engine20
from research.wave20_convex.engine20 import (
    GambleResult,
    Trade,
    combine_portfolio,
    load_stable_leg,
    one_leg_cost_rate_series,
    pad_to_calendar,
    worst_case_cost,
)
from research.wave25_gamble import indicators25
from research.wave25_gamble.configs25 import (
    B1_CONFIG,
    B2_CONFIG,
    B3_CONFIG,
    B4_CONFIG,
    B5_CONFIG,
    B6_CONFIG,
    B7_CONFIG,
    GAMBLE_CAPITAL,
    HARD_STOP_ATR_MULT,
    HARD_STOP_PCT,
    MAX_HOLD_BARS_1H,
    RISK_ATR_WINDOW,
    STABLE_CAPITAL,
    SYMBOLS,
    TRAILING_ACTIVATE_ATR_MULT,
    TRAILING_ATR_MULT,
    WAVE1_CACHE_DIR,
    WAVE6_CACHE_DIR,
    B1Config,
    B2Config,
    B3Config,
    B4Config,
    B5Config,
    B6Config,
    B7Config,
)


class Wave25Error(Exception):
    pass


def _clip_fraction(value: float) -> float:
    return max(value, -1.0)


# ---------------------------------------------------------------------------
# Data loading (thin, read-only; reuses research.wave20_convex.dataio20's already-hardened
# CSV reader -- see that module's own `_parse_timestamp_column` docstring for the "mixed"
# timestamp format edge case it already fixes -- rather than re-deriving the same parser).
# ---------------------------------------------------------------------------


def load_hourly(symbol: str) -> pd.DataFrame:
    return dataio20.load_hourly(symbol, WAVE6_CACHE_DIR)


def load_daily(symbol: str) -> pd.DataFrame:
    return dataio20.load_daily(symbol, WAVE1_CACHE_DIR)


def _resolve_frames(
    symbols: tuple[str, ...],
    hourly_frames: dict[str, pd.DataFrame] | None,
    daily_frames: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    hourly = hourly_frames if hourly_frames is not None else {s: load_hourly(s) for s in symbols}
    daily = daily_frames if daily_frames is not None else {s: load_daily(s) for s in symbols}
    return hourly, daily


# ---------------------------------------------------------------------------
# Per-indicator entry-signal generators. Each returns a Series aligned to `hourly`'s own
# index: +1.0 / -1.0 on the bar a NEW entry condition fires, 0.0 otherwise (never a
# continuously-held state -- see sticky_state() below for the regime version B7 needs).
# Decided using data through bar t (point-in-time); engine25's own position-lifecycle loop
# is solely responsible for the t -> t+1 fill-timing discipline, not these functions.
# ---------------------------------------------------------------------------


def macd_signal(hourly: pd.DataFrame, config: B1Config = B1_CONFIG) -> pd.Series:
    """B1: MACD histogram sign flip (crosses zero)."""
    frame = indicators25.macd(hourly["close"], config.fast, config.slow, config.signal)
    hist = frame["histogram"]
    prev = hist.shift(1)
    signal = pd.Series(0.0, index=hourly.index)
    signal[(prev <= 0.0) & (hist > 0.0)] = 1.0
    signal[(prev >= 0.0) & (hist < 0.0)] = -1.0
    return signal


def adx_dmi_signal(hourly: pd.DataFrame, config: B2Config = B2_CONFIG) -> pd.Series:
    """B2: ADX(14) > 25 AND a +DI/-DI crossover on the same bar."""
    frame = indicators25.adx_dmi(hourly, config.window)
    plus_di, minus_di, adx = frame["plus_di"], frame["minus_di"], frame["adx"]
    prev_plus, prev_minus = plus_di.shift(1), minus_di.shift(1)
    bullish_cross = (prev_plus <= prev_minus) & (plus_di > minus_di)
    bearish_cross = (prev_minus <= prev_plus) & (minus_di > plus_di)
    strong = adx > config.adx_threshold
    signal = pd.Series(0.0, index=hourly.index)
    signal[bullish_cross & strong] = 1.0
    signal[bearish_cross & strong] = -1.0
    return signal


def adx_dmi_active_mask(hourly: pd.DataFrame, config: B2Config = B2_CONFIG) -> pd.Series:
    adx = indicators25.adx_dmi(hourly, config.window)["adx"]
    return adx > config.adx_threshold


def supertrend_signal(hourly: pd.DataFrame, config: B3Config = B3_CONFIG) -> pd.Series:
    """B3: Supertrend direction flip."""
    frame = indicators25.supertrend(hourly, config.window, config.multiplier)
    direction = frame["direction"]
    prev = direction.shift(1)
    signal = pd.Series(0.0, index=hourly.index)
    signal[(prev <= 0.0) & (direction > 0.0)] = 1.0
    signal[(prev >= 0.0) & (direction < 0.0)] = -1.0
    return signal


def keltner_signal(hourly: pd.DataFrame, config: B4Config = B4_CONFIG) -> pd.Series:
    """B4: close crosses beyond the Keltner outer band."""
    frame = indicators25.keltner_channel(hourly, config.window, config.atr_window, config.multiplier)
    close = hourly["close"]
    prev_close = close.shift(1)
    prev_upper = frame["upper"].shift(1)
    prev_lower = frame["lower"].shift(1)
    signal = pd.Series(0.0, index=hourly.index)
    signal[(prev_close <= prev_upper) & (close > frame["upper"])] = 1.0
    signal[(prev_close >= prev_lower) & (close < frame["lower"])] = -1.0
    return signal


def mtf_confluence_signal(hourly: pd.DataFrame, daily: pd.DataFrame, config: B5Config = B5_CONFIG) -> pd.Series:
    """B5: 1D MA50-slope trend direction gates a 1H fixed-lookback ATR-momentum breakout.
    Deliberately NOT a Donchian N-bar high/low channel (that family is already dead per
    SPEC.md's own "테스트 완료" list) -- the trigger here is "price MOVED
    breakout_atr_multiplier*ATR over the trailing breakout_lookback_bars hours", a rate-of-
    change/momentum read, not an N-bar-extreme read. Only fires in the direction the daily
    trend currently allows (confluence); the daily trend itself is lagged one full day before
    being broadcast onto the hourly grid (indicators25.align_daily_to_intraday) so no
    same-day daily close ever leaks into an hour that occurred before that day closed."""
    daily_slope = indicators25.moving_average_slope(daily["close"], config.daily_ma_window, config.daily_slope_lookback)
    daily_trend = np.sign(daily_slope)
    hourly_trend = indicators25.align_daily_to_intraday(daily_trend, hourly.index)
    hourly_atr = indicators25.atr(hourly, config.breakout_atr_window)
    close = hourly["close"]
    momentum = close - close.shift(config.breakout_lookback_bars)
    threshold = config.breakout_atr_multiplier * hourly_atr
    raw_long = (momentum >= threshold) & threshold.notna()
    raw_short = (momentum <= -threshold) & threshold.notna()
    # NOTE: `.shift(1, fill_value=False)`, NOT `.shift(1).fillna(False)` -- shifting a bool
    # Series introduces a leading NaN, which silently upcasts the WHOLE series to `object`
    # dtype even after `.fillna(False)` (the fill value doesn't restore the dtype). `~` on an
    # object-dtype series of Python bools is bitwise NOT on int (`~False == -1`, `~True ==
    # -2`), both truthy -- so `raw_long & ~raw_long.shift(1).fillna(False)` silently degrades
    # to just `raw_long` (no edge-triggering at all). `shift(1, fill_value=False)` keeps proper
    # bool dtype throughout and was verified against this exact failure mode before being
    # adopted here; tests/test_wave25.py pins the edge-triggered (not level-triggered) behavior.
    new_long = raw_long & ~raw_long.shift(1, fill_value=False)
    new_short = raw_short & ~raw_short.shift(1, fill_value=False)
    signal = pd.Series(0.0, index=hourly.index)
    signal[new_long & (hourly_trend > 0.0)] = 1.0
    signal[new_short & (hourly_trend < 0.0)] = -1.0
    return signal


def stochastic_signal(hourly: pd.DataFrame, config: B6Config = B6_CONFIG) -> pd.Series:
    """B6: %K exits the oversold/overbought band, confirmed by a same-timeframe MA-slope
    trend filter (SPEC.md: "과매도/과매수 이탈 + 추세 필터")."""
    frame = indicators25.stochastic(hourly, config.k_window, config.d_window)
    k = frame["percent_k"]
    prev_k = k.shift(1)
    trend_slope = indicators25.moving_average_slope(hourly["close"], config.trend_ma_window, config.trend_slope_lookback)
    uptrend = trend_slope > 0.0
    downtrend = trend_slope < 0.0
    bullish_exit = (prev_k < config.oversold) & (k >= config.oversold)
    bearish_exit = (prev_k > config.overbought) & (k <= config.overbought)
    signal = pd.Series(0.0, index=hourly.index)
    signal[bullish_exit & uptrend] = 1.0
    signal[bearish_exit & downtrend] = -1.0
    return signal


# ---------------------------------------------------------------------------
# B7 -- ensemble agreement (SPEC.md: "B1~B6 중 동시 3개 이상 동일 방향 발화 시만 진입").
# ---------------------------------------------------------------------------


def sticky_state(signal: pd.Series, active_mask: pd.Series | None = None) -> pd.Series:
    """Turns an edge-triggered `signal` (nonzero only on the bar a member's OWN entry
    condition fires) into a persistent directional regime: holds the last nonzero direction
    until the opposite signal fires. If `active_mask` is given (B2's ADX>25 condition), the
    regime is force-zeroed wherever the mask is False -- so B7's per-member "vote" is never
    looser than that member's own standalone entry definition (B2 still requires ADX>25 to
    "count" as bullish/bearish for the ensemble, not just the DI ordering alone)."""
    nonzero = signal.where(signal != 0.0)
    state = nonzero.ffill().fillna(0.0)
    if active_mask is not None:
        state = state.where(active_mask.fillna(False), 0.0)
    return state


def ensemble_signal(member_signals: dict[str, pd.Series], member_active_masks: dict[str, pd.Series | None], min_agree: int) -> pd.Series:
    """Fires (edge-triggered) the FIRST bar the count of members sticky-agreeing on one
    direction reaches `min_agree` -- not every bar while agreement merely persists (that
    would re-"enter" a position B7 already holds on every subsequent bar)."""
    if not member_signals:
        raise Wave25Error("ensemble_signal: no member signals supplied")
    states = pd.DataFrame({name: sticky_state(sig, member_active_masks.get(name)) for name, sig in member_signals.items()})
    bullish_count = (states > 0.0).sum(axis=1)
    bearish_count = (states < 0.0).sum(axis=1)
    agree_long = bullish_count >= min_agree
    agree_short = bearish_count >= min_agree
    # See mtf_confluence_signal's identical comment: shift(1, fill_value=False), never
    # shift(1).fillna(False), for a bool Series that is about to be negated with `~`.
    new_long = agree_long & ~agree_long.shift(1, fill_value=False)
    new_short = agree_short & ~agree_short.shift(1, fill_value=False)
    signal = pd.Series(0.0, index=states.index)
    signal[new_long] = 1.0
    signal[new_short] = -1.0
    return signal


def _member_signals_and_masks(hourly: pd.DataFrame, daily: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series | None]]:
    signals = {
        "B1": macd_signal(hourly, B1_CONFIG),
        "B2": adx_dmi_signal(hourly, B2_CONFIG),
        "B3": supertrend_signal(hourly, B3_CONFIG),
        "B4": keltner_signal(hourly, B4_CONFIG),
        "B5": mtf_confluence_signal(hourly, daily, B5_CONFIG),
        "B6": stochastic_signal(hourly, B6_CONFIG),
    }
    masks: dict[str, pd.Series | None] = {"B1": None, "B2": adx_dmi_active_mask(hourly, B2_CONFIG), "B3": None, "B4": None, "B5": None, "B6": None}
    return signals, masks


def ensemble_signal_for_symbol(hourly: pd.DataFrame, daily: pd.DataFrame, config: B7Config = B7_CONFIG) -> pd.Series:
    signals, masks = _member_signals_and_masks(hourly, daily)
    return ensemble_signal(signals, masks, config.min_agree)


# ---------------------------------------------------------------------------
# Generic multi-symbol convex position-lifecycle simulator (B1-B7's shared engine core).
# ---------------------------------------------------------------------------


def build_candidate_frames(
    hourly: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    mapping: MeasuredCostMapping,
    stress_multiplier: float,
    symbols: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Assembles the seven aligned (calendar x symbol) frames run_multi_symbol_convex needs,
    from each symbol's own NATIVE (non-reindexed) hourly OHLCV + signal series. ATR and cost
    rate are computed on each symbol's own native bar sequence FIRST (so a rolling/ATR window
    never spans a gap introduced merely by a later-listed symbol's shorter history), then
    every series is reindexed onto the shared union calendar -- exact reindex (introduces NaN
    for a symbol's pre-listing bars, never fabricated) for OHLCV/ATR/signal, forward-fill only
    for the daily-cost-rate broadcast (matches engine20.run_v1's own daily->hourly convention)."""
    calendar = pd.DatetimeIndex(sorted(set().union(*(hourly[s].index for s in symbols))))
    open_frame = pd.DataFrame({s: hourly[s]["open"] for s in symbols}).reindex(calendar)
    high_frame = pd.DataFrame({s: hourly[s]["high"] for s in symbols}).reindex(calendar)
    low_frame = pd.DataFrame({s: hourly[s]["low"] for s in symbols}).reindex(calendar)
    close_frame = pd.DataFrame({s: hourly[s]["close"] for s in symbols}).reindex(calendar)
    risk_atr_frame = pd.DataFrame({s: indicators25.atr(hourly[s], RISK_ATR_WINDOW) for s in symbols}).reindex(calendar)
    signal_frame = pd.DataFrame({s: signals[s] for s in symbols}).reindex(calendar).fillna(0.0)

    cost_columns: dict[str, pd.Series] = {}
    for s in symbols:
        daily_from_hourly = dataio20.resample_hourly_to_daily(hourly[s])
        daily_cost = one_leg_cost_rate_series(daily_from_hourly["quote_volume"], mapping, stress_multiplier)
        cost_columns[s] = daily_cost.reindex(calendar, method="ffill")
    cost_frame = pd.DataFrame(cost_columns)

    return {
        "open": open_frame,
        "high": high_frame,
        "low": low_frame,
        "close": close_frame,
        "risk_atr": risk_atr_frame,
        "cost": cost_frame,
        "signal": signal_frame,
    }


def run_multi_symbol_convex(
    frames: dict[str, pd.DataFrame],
    priority: tuple[str, ...],
    worst_cost: float,
    candidate_id: str,
    hard_stop_pct: float = HARD_STOP_PCT,
    hard_stop_atr_mult: float = HARD_STOP_ATR_MULT,
    trailing_activate_atr_mult: float = TRAILING_ACTIVATE_ATR_MULT,
    trailing_atr_mult: float = TRAILING_ATR_MULT,
    max_bars_in_position: int = MAX_HOLD_BARS_1H,
    starting_equity: float = GAMBLE_CAPITAL,
    extra_metadata: dict[str, Any] | None = None,
) -> GambleResult:
    calendar = frames["open"].index
    n = len(calendar)
    symbols = list(priority)
    o = {s: frames["open"][s].to_numpy(dtype=float) for s in symbols}
    h = {s: frames["high"][s].to_numpy(dtype=float) for s in symbols}
    lo = {s: frames["low"][s].to_numpy(dtype=float) for s in symbols}
    c = {s: frames["close"][s].to_numpy(dtype=float) for s in symbols}
    ra = {s: frames["risk_atr"][s].to_numpy(dtype=float) for s in symbols}
    cf = {s: frames["cost"][s].to_numpy(dtype=float) for s in symbols}
    sig = {s: frames["signal"][s].to_numpy(dtype=float) for s in symbols}

    equity_out = np.empty(n, dtype=float)
    trades: list[Trade] = []
    total_cost = 0.0
    sleeve_equity = starting_equity
    dead = False

    held_symbol: str | None = None
    direction = 0.0
    entry_price = float("nan")
    entry_risk_atr = float("nan")
    stop_price = float("nan")
    trailing_armed = False
    trailing_stop = float("nan")
    extreme = float("nan")
    entry_time: pd.Timestamp | None = None
    entry_equity = starting_equity
    entry_idx = -1
    # pending action decided at bar i-1, executed at bar i's open: (action, symbol, direction, atr_at_decision)
    pending: tuple[str, str, float, float] | None = None

    if n == 0:
        raise Wave25Error(f"{candidate_id}: empty calendar -- no symbol had any cached data")

    for i in range(n):
        ts = calendar[i]

        # --- Step A: execute the action decided at bar i-1, filled at bar i's own open. ---
        if pending is not None and not dead:
            action, symbol, new_direction, atr_at_decision = pending
            pending = None

            if action == "reverse" and held_symbol is not None:
                fill_price = o[held_symbol][i]
                if not np.isnan(fill_price) and fill_price > 0.0:
                    prior_close = c[held_symbol][i - 1] if i > 0 else float("nan")
                    if not np.isnan(prior_close) and prior_close > 0.0:
                        gap_ret = fill_price / prior_close - 1.0
                        sleeve_equity = sleeve_equity * (1.0 + direction * gap_ret)
                    if sleeve_equity <= 0.0:
                        trades.append(Trade(held_symbol, direction, entry_time, ts, entry_price, float(fill_price), entry_equity, -entry_equity, -1.0, "liquidated", entry_equity))
                        sleeve_equity = 0.0
                        dead = True
                    else:
                        rate = cf[held_symbol][i] if not np.isnan(cf[held_symbol][i]) else worst_cost
                        equity_before = sleeve_equity
                        sleeve_equity *= 1.0 - rate
                        exit_cost = equity_before - sleeve_equity
                        total_cost += exit_cost
                        pnl = sleeve_equity - entry_equity
                        trades.append(
                            Trade(held_symbol, direction, entry_time, ts, entry_price, float(fill_price), entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "signal_reversal", exit_cost)
                        )
                    held_symbol = None
                    direction = 0.0

            if action in ("open", "reverse") and new_direction != 0.0 and not dead and sleeve_equity > 0.0:
                fill_price = o[symbol][i]
                if not np.isnan(fill_price) and fill_price > 0.0:
                    rate = cf[symbol][i] if not np.isnan(cf[symbol][i]) else worst_cost
                    equity_before_entry = sleeve_equity
                    sleeve_equity *= 1.0 - rate
                    total_cost += equity_before_entry - sleeve_equity
                    if sleeve_equity <= 0.0:
                        dead = True
                        sleeve_equity = 0.0
                    else:
                        held_symbol = symbol
                        direction = new_direction
                        entry_price = float(fill_price)
                        entry_risk_atr = float(atr_at_decision) if not np.isnan(atr_at_decision) and atr_at_decision > 0.0 else entry_price * hard_stop_pct / hard_stop_atr_mult
                        stop_distance = min(hard_stop_pct * entry_price, hard_stop_atr_mult * entry_risk_atr)
                        stop_price = entry_price - direction * stop_distance
                        extreme = entry_price
                        trailing_armed = False
                        trailing_stop = float("nan")
                        entry_time = ts
                        entry_equity = sleeve_equity
                        entry_idx = i

        # --- Step B: mark-to-market + intrabar exit checks for the held position at bar i. ---
        if held_symbol is not None and not dead:
            ref_price = entry_price if entry_time == ts else c[held_symbol][i - 1]
            bar_open, bar_high, bar_low, bar_close = o[held_symbol][i], h[held_symbol][i], lo[held_symbol][i], c[held_symbol][i]

            exit_price: float | None = None
            exit_reason: str | None = None
            if not np.isnan(bar_low) and not np.isnan(bar_high):
                if direction > 0.0:
                    if bar_low <= stop_price:
                        exit_price = min(stop_price, bar_open) if not np.isnan(bar_open) else stop_price
                        exit_reason = "stop_loss"
                    elif trailing_armed and bar_low <= trailing_stop:
                        exit_price = min(trailing_stop, bar_open) if not np.isnan(bar_open) else trailing_stop
                        exit_reason = "trailing_exit"
                else:
                    if bar_high >= stop_price:
                        exit_price = max(stop_price, bar_open) if not np.isnan(bar_open) else stop_price
                        exit_reason = "stop_loss"
                    elif trailing_armed and bar_high >= trailing_stop:
                        exit_price = max(trailing_stop, bar_open) if not np.isnan(bar_open) else trailing_stop
                        exit_reason = "trailing_exit"
            if exit_price is None and (i - entry_idx) >= max_bars_in_position:
                exit_price = bar_close
                exit_reason = "max_hold"

            if exit_price is not None and not np.isnan(ref_price) and ref_price > 0.0:
                bar_ret = exit_price / ref_price - 1.0
                sleeve_equity = sleeve_equity * (1.0 + direction * bar_ret)
                if sleeve_equity <= 0.0:
                    trades.append(Trade(held_symbol, direction, entry_time, ts, entry_price, float(exit_price), entry_equity, -entry_equity, -1.0, "liquidated", entry_equity))
                    sleeve_equity = 0.0
                    dead = True
                else:
                    rate = cf[held_symbol][i] if not np.isnan(cf[held_symbol][i]) else worst_cost
                    equity_before_cost = sleeve_equity
                    sleeve_equity *= 1.0 - rate
                    exit_cost = equity_before_cost - sleeve_equity
                    total_cost += exit_cost
                    pnl = sleeve_equity - entry_equity
                    trades.append(
                        Trade(held_symbol, direction, entry_time, ts, entry_price, float(exit_price), entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, exit_reason, exit_cost)
                    )
                held_symbol = None
                direction = 0.0
            elif not np.isnan(bar_close) and bar_close > 0.0 and not np.isnan(ref_price) and ref_price > 0.0:
                bar_ret = bar_close / ref_price - 1.0
                sleeve_equity = sleeve_equity * (1.0 + direction * bar_ret)
                if sleeve_equity <= 0.0:
                    trades.append(Trade(held_symbol, direction, entry_time, ts, entry_price, float(bar_close), entry_equity, -entry_equity, -1.0, "liquidated", entry_equity))
                    sleeve_equity = 0.0
                    dead = True
                    held_symbol = None
                    direction = 0.0
                else:
                    extreme = max(extreme, bar_close) if direction > 0.0 else min(extreme, bar_close)
                    profit_distance = direction * (bar_close - entry_price)
                    if not trailing_armed and profit_distance >= trailing_activate_atr_mult * entry_risk_atr:
                        trailing_armed = True
                        trailing_stop = entry_price
                    if trailing_armed:
                        current_atr = ra[held_symbol][i]
                        if not np.isnan(current_atr) and current_atr > 0.0:
                            candidate_level = extreme - direction * trailing_atr_mult * current_atr
                            trailing_stop = max(trailing_stop, candidate_level) if direction > 0.0 else min(trailing_stop, candidate_level)

        equity_out[i] = sleeve_equity

        # --- Step C: decide the action for bar i+1 using data known through bar i. ---
        if not dead and sleeve_equity > 0.0 and i + 1 < n:
            if held_symbol is not None:
                current_signal = sig[held_symbol][i]
                if current_signal != 0.0 and np.sign(current_signal) == -np.sign(direction):
                    pending = ("reverse", held_symbol, float(np.sign(current_signal)), ra[held_symbol][i])
            else:
                for symbol in priority:
                    s_val = sig[symbol][i]
                    if s_val != 0.0:
                        pending = ("open", symbol, float(np.sign(s_val)), ra[symbol][i])
                        break

    if held_symbol is not None and not dead:
        last_idx = n - 1
        last_close = c[held_symbol][last_idx]
        ref_price = entry_price if entry_time == calendar[last_idx] else (c[held_symbol][last_idx - 1] if last_idx > 0 else entry_price)
        if not np.isnan(last_close) and not np.isnan(ref_price) and ref_price > 0.0:
            bar_ret = last_close / ref_price - 1.0
            sleeve_equity = sleeve_equity * (1.0 + direction * bar_ret)
        if sleeve_equity <= 0.0:
            sleeve_equity = 0.0
            trades.append(Trade(held_symbol, direction, entry_time, calendar[last_idx], entry_price, float(last_close), entry_equity, -entry_equity, -1.0, "liquidated", entry_equity))
        else:
            rate = cf[held_symbol][last_idx] if not np.isnan(cf[held_symbol][last_idx]) else worst_cost
            equity_before_cost = sleeve_equity
            sleeve_equity *= 1.0 - rate
            exit_cost = equity_before_cost - sleeve_equity
            total_cost += exit_cost
            pnl = sleeve_equity - entry_equity
            trades.append(
                Trade(held_symbol, direction, entry_time, calendar[last_idx], entry_price, float(last_close), entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "end_of_data", exit_cost)
            )
        equity_out[last_idx] = sleeve_equity

    equity_hourly = pd.Series(equity_out, index=calendar)
    equity_daily = equity_hourly.resample("1D").last().ffill()
    trade_symbol_counts = {symbol: sum(1 for t in trades if t.symbol == symbol) for symbol in symbols}
    exit_reason_counts: dict[str, int] = {}
    for t in trades:
        exit_reason_counts[t.exit_reason] = exit_reason_counts.get(t.exit_reason, 0) + 1
    metadata: dict[str, Any] = {
        "n_trades": len(trades),
        "trade_symbol_counts": trade_symbol_counts,
        "exit_reason_counts": exit_reason_counts,
        "total_cost_usdt": total_cost,
        "final_equity_usdt": float(equity_daily.iloc[-1]) if len(equity_daily) else starting_equity,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return GambleResult(candidate_id=candidate_id, equity=equity_daily, trades=tuple(trades), symbols_used=tuple(symbols), metadata=metadata)


# ---------------------------------------------------------------------------
# Candidate runners.
# ---------------------------------------------------------------------------

SignalFn = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


def _run_signal_candidate(
    candidate_id: str,
    signal_fn: SignalFn,
    config_dict: dict[str, Any],
    symbols: tuple[str, ...] = SYMBOLS,
    mapping: MeasuredCostMapping | None = None,
    stress_multiplier: float = 1.0,
    hourly_frames: dict[str, pd.DataFrame] | None = None,
    daily_frames: dict[str, pd.DataFrame] | None = None,
) -> GambleResult:
    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    hourly, daily = _resolve_frames(symbols, hourly_frames, daily_frames)
    signals = {s: signal_fn(hourly[s], daily[s]) for s in symbols}
    frames = build_candidate_frames(hourly, signals, mapping, stress_multiplier, symbols)
    worst_cost = worst_case_cost(mapping, stress_multiplier)
    return run_multi_symbol_convex(
        frames,
        priority=symbols,
        worst_cost=worst_cost,
        candidate_id=candidate_id,
        extra_metadata={"config": config_dict, "symbols_scanned": list(symbols)},
    )


def run_b0(mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    """B0 = wave20 V1 verbatim reproduction (see module docstring). Not reimplemented."""
    from research.wave20_convex.configs20 import V1_CONFIG

    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    v1_result = engine20.run_v1(V1_CONFIG, mapping=mapping, stress_multiplier=stress_multiplier)
    metadata = dict(v1_result.metadata)
    metadata["source"] = "research.wave20_convex.engine20.run_v1 (verbatim reuse, not reimplemented)"
    return GambleResult(candidate_id="B0", equity=v1_result.equity, trades=v1_result.trades, symbols_used=v1_result.symbols_used, metadata=metadata)


def _run_b0_with_hourly(hourly_btc: pd.DataFrame, mapping: MeasuredCostMapping, stress_multiplier: float = 1.0) -> GambleResult:
    """Live-stage counterpart to run_b0: replays V1's own array-level logic
    (engine20.simulate_breakout_reversal + the exact same regime-filter/ATR construction
    engine20.run_v1 itself uses) against a caller-supplied (possibly network-freshened) BTC
    hourly frame, instead of engine20.run_v1's own cache-bound loader. Every helper function
    called here is imported from engine20 unmodified -- this is V1's own math, not a
    lookalike."""
    from research.wave20_convex.configs20 import V1_CONFIG

    daily = dataio20.resample_hourly_to_daily(hourly_btc)
    vol20 = engine20.realized_vol(daily["close"], V1_CONFIG.vol_window_days)
    vol_pct_rank = engine20.trailing_percentile_rank(vol20, V1_CONFIG.vol_percentile_lookback_days)
    low_vol_regime_daily = vol_pct_rank < V1_CONFIG.vol_percentile_threshold
    armable_daily = low_vol_regime_daily.shift(1).fillna(False)
    cost_daily = one_leg_cost_rate_series(daily["quote_volume"], mapping, stress_multiplier)
    armable_hourly = armable_daily.reindex(hourly_btc.index, method="ffill").fillna(False).astype(bool)
    cost_hourly = cost_daily.reindex(hourly_btc.index, method="ffill")
    atr_daily_lagged = engine20.atr(daily, V1_CONFIG.atr_window_days).shift(1)
    atr_hourly = atr_daily_lagged.reindex(hourly_btc.index, method="ffill")
    worst_cost = worst_case_cost(mapping, stress_multiplier)

    equity_arr, trades, total_cost = engine20.simulate_breakout_reversal(
        index=hourly_btc.index,
        open_arr=hourly_btc["open"].to_numpy(dtype=float),
        close_arr=hourly_btc["close"].to_numpy(dtype=float),
        atr_arr=atr_hourly.to_numpy(dtype=float),
        cost_arr=cost_hourly.to_numpy(dtype=float),
        worst_cost=worst_cost,
        atr_multiplier=V1_CONFIG.atr_multiplier,
        armable_arr=armable_hourly.to_numpy(dtype=bool),
        forced_initial_direction=None,
        max_bars_in_position=None,
        starting_equity=GAMBLE_CAPITAL,
        symbol=V1_CONFIG.symbol,
    )
    equity_hourly = pd.Series(equity_arr, index=hourly_btc.index)
    equity_daily = equity_hourly.resample("1D").last().ffill()
    return GambleResult(
        candidate_id="B0",
        equity=equity_daily,
        trades=tuple(trades),
        symbols_used=(V1_CONFIG.symbol,),
        metadata={"n_trades": len(trades), "total_cost_usdt": total_cost, "final_equity_usdt": float(equity_daily.iloc[-1]) if len(equity_daily) else GAMBLE_CAPITAL},
    )


def run_b1(config: B1Config = B1_CONFIG, symbols: tuple[str, ...] = SYMBOLS, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_signal_candidate(
        "B1", lambda hourly, daily: macd_signal(hourly, config), {"fast": config.fast, "slow": config.slow, "signal": config.signal},
        symbols, mapping, stress_multiplier, hourly_frames, daily_frames,
    )


def run_b2(config: B2Config = B2_CONFIG, symbols: tuple[str, ...] = SYMBOLS, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_signal_candidate(
        "B2", lambda hourly, daily: adx_dmi_signal(hourly, config), {"window": config.window, "adx_threshold": config.adx_threshold},
        symbols, mapping, stress_multiplier, hourly_frames, daily_frames,
    )


def run_b3(config: B3Config = B3_CONFIG, symbols: tuple[str, ...] = SYMBOLS, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_signal_candidate(
        "B3", lambda hourly, daily: supertrend_signal(hourly, config), {"window": config.window, "multiplier": config.multiplier},
        symbols, mapping, stress_multiplier, hourly_frames, daily_frames,
    )


def run_b4(config: B4Config = B4_CONFIG, symbols: tuple[str, ...] = SYMBOLS, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_signal_candidate(
        "B4", lambda hourly, daily: keltner_signal(hourly, config), {"window": config.window, "atr_window": config.atr_window, "multiplier": config.multiplier},
        symbols, mapping, stress_multiplier, hourly_frames, daily_frames,
    )


def run_b5(config: B5Config = B5_CONFIG, symbols: tuple[str, ...] = SYMBOLS, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_signal_candidate(
        "B5", lambda hourly, daily: mtf_confluence_signal(hourly, daily, config),
        {
            "daily_ma_window": config.daily_ma_window, "daily_slope_lookback": config.daily_slope_lookback,
            "breakout_lookback_bars": config.breakout_lookback_bars, "breakout_atr_window": config.breakout_atr_window,
            "breakout_atr_multiplier": config.breakout_atr_multiplier,
        },
        symbols, mapping, stress_multiplier, hourly_frames, daily_frames,
    )


def run_b6(config: B6Config = B6_CONFIG, symbols: tuple[str, ...] = SYMBOLS, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_signal_candidate(
        "B6", lambda hourly, daily: stochastic_signal(hourly, config),
        {
            "k_window": config.k_window, "d_window": config.d_window, "oversold": config.oversold, "overbought": config.overbought,
            "trend_ma_window": config.trend_ma_window, "trend_slope_lookback": config.trend_slope_lookback,
        },
        symbols, mapping, stress_multiplier, hourly_frames, daily_frames,
    )


def run_b7(config: B7Config = B7_CONFIG, symbols: tuple[str, ...] = SYMBOLS, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_signal_candidate(
        "B7", lambda hourly, daily: ensemble_signal_for_symbol(hourly, daily, config), {"min_agree": config.min_agree},
        symbols, mapping, stress_multiplier, hourly_frames, daily_frames,
    )


RUNNERS: Final[dict[str, Callable[..., GambleResult]]] = {
    "B0": run_b0,
    "B1": run_b1,
    "B2": run_b2,
    "B3": run_b3,
    "B4": run_b4,
    "B5": run_b5,
    "B6": run_b6,
    "B7": run_b7,
}


def run_candidate(candidate_id: str, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    if candidate_id not in RUNNERS:
        raise Wave25Error(f"unknown wave25 candidate: {candidate_id}")
    if candidate_id == "B0":
        return run_b0(mapping=mapping, stress_multiplier=stress_multiplier)
    return RUNNERS[candidate_id](mapping=mapping, stress_multiplier=stress_multiplier)


__all__ = [
    "GambleResult",
    "RUNNERS",
    "SignalFn",
    "Trade",
    "Wave25Error",
    "adx_dmi_active_mask",
    "adx_dmi_signal",
    "build_candidate_frames",
    "combine_portfolio",
    "ensemble_signal",
    "ensemble_signal_for_symbol",
    "keltner_signal",
    "load_daily",
    "load_hourly",
    "load_stable_leg",
    "macd_signal",
    "mtf_confluence_signal",
    "pad_to_calendar",
    "run_b0",
    "run_b1",
    "run_b2",
    "run_b3",
    "run_b4",
    "run_b5",
    "run_b6",
    "run_b7",
    "run_candidate",
    "run_multi_symbol_convex",
    "stochastic_signal",
    "sticky_state",
    "supertrend_signal",
]
