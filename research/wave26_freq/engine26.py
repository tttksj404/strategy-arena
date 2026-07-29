# Wave-26 frequency-control overlay (C0-C7). See research/wave26_freq/SPEC.md for the frozen
# pre-registration this module implements, and configs26.py for every numeric threshold.
#
# ---------------------------------------------------------------------------------------
# What this module does NOT do
# ---------------------------------------------------------------------------------------
# SPEC.md: "신호 로직은 research/wave25_gamble의 것을 그대로 임포트 재사용 ... 신호를 재구현하거나
# 변경하지 말 것 -- 이번 wave가 바꾸는 건 진입 허용 조건(빈도 통제)뿐이다." Concretely: every
# indicator formula (indicators25.py), every base entry-signal generator (engine25.macd_signal /
# supertrend_signal / ensemble_signal_for_symbol), the entire convex stop/trailing lifecycle
# (HARD_STOP_PCT/HARD_STOP_ATR_MULT/TRAILING_*), the cost model (wave13_liquidity.costs_measured
# via engine20's single-leg wrappers), the capital contract, and V1's own breakout/chandelier-
# reversal arithmetic (research.wave20_convex.engine20.simulate_breakout_reversal) are all
# imported and called UNMODIFIED below. Nothing in this file re-derives an indicator formula or
# changes a stop/trailing/breakout THRESHOLD.
#
# ---------------------------------------------------------------------------------------
# What this module DOES add: an entry-ADMISSION gate, uniformly, on top of each reused engine
# ---------------------------------------------------------------------------------------
# Three independent knobs (SPEC.md "빈도 통제 3축"), each of which only ever makes an entry
# HARDER to admit, never easier, and never touches how a position -- once opened -- is managed
# or exits:
#   1. Cooldown (N days after ANY exit): DYNAMIC, path-dependent state. It cannot be
#      precomputed as a static filter on the raw signal series, because it depends on WHEN an
#      exit actually happens during the simulation -- and exits are driven by price action
#      (stop-loss / trailing-stop / max-hold), not just by the signal series. This is why the
#      two functions below (run_multi_symbol_convex_controlled /
#      simulate_breakout_reversal_controlled) are FORKS of engine25.run_multi_symbol_convex /
#      engine20.simulate_breakout_reversal rather than a pre-filter wrapped around them: cooldown
#      state (`cooldown_remaining`) has to live inside the same forward pass that decides fills.
#   2. ADX(14) > 20 regime filter, and 3. signal-value 20-day z-score > 1.0: both STATIC,
#      precomputable per (symbol, bar) boolean masks (adx_active / z_active in
#      build_controlled_frames / the adx_active_arr parameter) -- computed once, ANDed into the
#      entry-admission check every bar the forked loop runs.
#
# A subtlety common to BOTH forks: engine25's own "reverse" action (a signal flip against a held
# position exits AND immediately re-opens the opposite side, same bar, same fill) is split into
# two separate actions here: "close" (exit only, recorded as exit_reason="signal_flip_exit") and
# a later, independently-gated "open". This split is not optional -- "no re-entry for N days
# after an exit" and "instantaneously flip to the opposite side on the same bar" are mutually
# exclusive when N>0, so implementing cooldown AT ALL requires this decomposition. Closing a
# position is NEVER gated by cooldown/ADX/z (those gate NEW risk; forcing a position to be held
# past its own invalidation signal just because the sleeve is "on cooldown" would be perverse).
# Only the OPEN half is gated. See each function's own docstring for the exact bar-by-bar
# semantics (cooldown_remaining decrement timing, etc.) and tests/test_wave26.py for the pinned
# boundary behavior.

from __future__ import annotations

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
from research.wave20_convex import dataio20
from research.wave20_convex import engine20 as wave20_engine
from research.wave25_gamble import engine25, indicators25
from research.wave25_gamble.configs25 import B1_CONFIG, B3_CONFIG, B7_CONFIG
from research.wave25_gamble.engine25 import (
    GambleResult,
    Trade,
    combine_portfolio,
    load_stable_leg,
    one_leg_cost_rate_series,
    pad_to_calendar,
    worst_case_cost,
)
from research.wave26_freq.configs26 import (
    ADX_REGIME_THRESHOLD,
    ADX_REGIME_WINDOW,
    C0_SPEC,
    C1_SPEC,
    C2_SPEC,
    C3_SPEC,
    C4_SPEC,
    C5_SPEC,
    C6_SPEC,
    C7_SPEC,
    GAMBLE_CAPITAL,
    HARD_STOP_ATR_MULT,
    HARD_STOP_PCT,
    MAX_HOLD_BARS_1H,
    SYMBOLS,
    TRAILING_ACTIVATE_ATR_MULT,
    TRAILING_ATR_MULT,
    WAVE6_CACHE_DIR,
    ControlSpec,
    Z_SCORE_THRESHOLD,
    Z_SCORE_WINDOW_BARS,
)


class Wave26Error(Exception):
    pass


def _clip_fraction(value: float) -> float:
    return max(value, -1.0)


# ---------------------------------------------------------------------------
# Data loading (thin, read-only -- identical convention to engine25.load_hourly/load_daily).
# ---------------------------------------------------------------------------


def load_hourly(symbol: str) -> pd.DataFrame:
    return engine25.load_hourly(symbol)


def load_daily(symbol: str) -> pd.DataFrame:
    return engine25.load_daily(symbol)


def _resolve_frames(
    symbols: tuple[str, ...],
    hourly_frames: dict[str, pd.DataFrame] | None,
    daily_frames: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    hourly = hourly_frames if hourly_frames is not None else {s: load_hourly(s) for s in symbols}
    daily = daily_frames if daily_frames is not None else {s: load_daily(s) for s in symbols}
    return hourly, daily


# ---------------------------------------------------------------------------
# Base signals (SPEC.md "기저" column) -- thin (hourly, daily) -> Series wrappers around
# engine25's own unmodified signal generators, uniform call signature for _run_controlled_candidate
# below (mirrors engine25._run_signal_candidate's own `SignalFn` convention).
# ---------------------------------------------------------------------------


def _macd_base_signal(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    return engine25.macd_signal(hourly, B1_CONFIG)


def _supertrend_base_signal(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    return engine25.supertrend_signal(hourly, B3_CONFIG)


def _ensemble_base_signal(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    return engine25.ensemble_signal_for_symbol(hourly, daily, B7_CONFIG)


BASE_SIGNAL_FNS: Final[dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.Series]]] = {
    "MACD": _macd_base_signal,
    "SUPERTREND": _supertrend_base_signal,
    "ENSEMBLE": _ensemble_base_signal,
}

CANDIDATE_BASE_CONFIG: Final[dict[str, dict[str, Any]]] = {
    "MACD": {"fast": B1_CONFIG.fast, "slow": B1_CONFIG.slow, "signal": B1_CONFIG.signal},
    "SUPERTREND": {"window": B3_CONFIG.window, "multiplier": B3_CONFIG.multiplier},
    "ENSEMBLE": {"min_agree": B7_CONFIG.min_agree},
}


# ---------------------------------------------------------------------------
# Signal-strength series for the z-score filter (SPEC.md "신호값 20일 z-score > 1.0"). Only C3
# (MACD) and C6 (ensemble) use this axis (SPEC.md's own candidate table) -- Supertrend/V1 never
# need a strength_fn. Both readouts reuse ONLY values engine25 already computes for its own
# signal generation; neither introduces a new formula.
# ---------------------------------------------------------------------------


def macd_histogram_strength(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """'How strong is the MACD signal right now' -- the absolute value of the SAME histogram
    engine25.macd_signal's own zero-cross trigger is built from (B1_CONFIG's (12,26,9),
    unmodified), read continuously bar-by-bar instead of only at its zero-cross bars. `daily` is
    unused (kept only for BASE_SIGNAL_FNS-style call-signature uniformity)."""
    frame = indicators25.macd(hourly["close"], B1_CONFIG.fast, B1_CONFIG.slow, B1_CONFIG.signal)
    return frame["histogram"].abs()


def ensemble_agreement_strength(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """'How many of B1-B6 currently agree on ONE direction' -- reuses engine25's own six member
    signals + sticky_state (the exact building blocks B7's own ensemble_signal_for_symbol uses
    internally) to expose the underlying vote COUNT that function only ever thresholds against
    `min_agree`, at every bar (not just the bars agreement first crosses the threshold)."""
    signals, masks = engine25._member_signals_and_masks(hourly, daily)  # reuses B7's own private wiring helper verbatim -- see module docstring
    states = pd.DataFrame({name: engine25.sticky_state(sig, masks.get(name)) for name, sig in signals.items()})
    bullish_count = (states > 0.0).sum(axis=1).astype(float)
    bearish_count = (states < 0.0).sum(axis=1).astype(float)
    return pd.concat([bullish_count, bearish_count], axis=1).max(axis=1)


STRENGTH_FNS: Final[dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.Series]]] = {
    "MACD": macd_histogram_strength,
    "ENSEMBLE": ensemble_agreement_strength,
}


def zscore_gate_mask(strength: pd.Series, window_bars: int = Z_SCORE_WINDOW_BARS, threshold: float = Z_SCORE_THRESHOLD) -> pd.Series:
    """Boolean mask: True at bar t iff strength[t]'s own trailing `window_bars`-bar z-score
    (mean/std from bars [t-window_bars+1, t], INCLUDING bar t itself -- causal, matches
    indicators25.py's module-wide "uses only data up to and including bar t" convention and
    engine25.adx_dmi_active_mask's own identical same-bar-inclusive convention) exceeds
    `threshold`. NaN (insufficient history, or a zero-variance window) maps to False -- an
    undetermined filter blocks the entry rather than silently admitting it."""
    rolling = strength.rolling(window_bars, min_periods=window_bars)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (strength - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    return (z > threshold).fillna(False)


# ---------------------------------------------------------------------------
# Frame assembly: engine25.build_candidate_frames's own seven frames, PLUS two new static
# per-(symbol,bar) admission masks.
# ---------------------------------------------------------------------------


def build_controlled_frames(
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    mapping: MeasuredCostMapping,
    stress_multiplier: float,
    symbols: tuple[str, ...],
    adx_gate: bool,
    z_gate: bool,
    strength_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series] | None,
) -> dict[str, pd.DataFrame]:
    frames = dict(engine25.build_candidate_frames(hourly, signals, mapping, stress_multiplier, symbols))
    calendar = frames["open"].index

    if adx_gate:
        adx_cols: dict[str, pd.Series] = {}
        for s in symbols:
            adx_native = indicators25.adx_dmi(hourly[s], ADX_REGIME_WINDOW)["adx"]  # computed on the symbol's OWN native bars first -- same convention as risk_atr in engine25.build_candidate_frames
            active = (adx_native > ADX_REGIME_THRESHOLD).reindex(calendar).fillna(False)
            adx_cols[s] = active.astype(bool)
        frames["adx_active"] = pd.DataFrame(adx_cols)
    else:
        frames["adx_active"] = pd.DataFrame(True, index=calendar, columns=list(symbols))

    if z_gate:
        if strength_fn is None:
            raise Wave26Error("build_controlled_frames: z_gate=True requires a strength_fn")
        z_cols: dict[str, pd.Series] = {}
        for s in symbols:
            strength = strength_fn(hourly[s], daily[s])
            mask = zscore_gate_mask(strength)
            z_cols[s] = mask.reindex(calendar).fillna(False).astype(bool)
        frames["z_active"] = pd.DataFrame(z_cols)
    else:
        frames["z_active"] = pd.DataFrame(True, index=calendar, columns=list(symbols))

    return frames


# ---------------------------------------------------------------------------
# B-family controlled engine -- fork of engine25.run_multi_symbol_convex. Every line NOT
# touching the close/open split or the cooldown/adx/z admission check is copied verbatim
# (identical cost application, gap handling, stop/trailing arithmetic, Trade construction).
# ---------------------------------------------------------------------------


def run_multi_symbol_convex_controlled(
    frames: dict[str, pd.DataFrame],  # must include "adx_active" and "z_active" (see build_controlled_frames) alongside engine25.build_candidate_frames's own seven
    priority: tuple[str, ...],
    worst_cost: float,
    candidate_id: str,
    cooldown_bars: int,
    hard_stop_pct: float = HARD_STOP_PCT,
    hard_stop_atr_mult: float = HARD_STOP_ATR_MULT,
    trailing_activate_atr_mult: float = TRAILING_ACTIVATE_ATR_MULT,
    trailing_atr_mult: float = TRAILING_ATR_MULT,
    max_bars_in_position: int = MAX_HOLD_BARS_1H,
    starting_equity: float = GAMBLE_CAPITAL,
    extra_metadata: dict[str, Any] | None = None,
) -> GambleResult:
    """Cooldown semantics (pinned by tests/test_wave26.py): the bar an exit is FILLED sets
    `cooldown_remaining = cooldown_bars`; entries are blocked for exactly the next `cooldown_bars`
    bars (decremented once per bar at the end of Step C) and become admissible again starting
    `cooldown_bars + 1` bars after the exit. ADX/z gates are independent static per-(symbol,bar)
    AND conditions checked at the same bar as the signal itself (no new lookahead: still
    decide-at-bar-i / fill-at-bar-i+1's-open, unchanged from engine25)."""
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
    aa = {s: frames["adx_active"][s].to_numpy(dtype=bool) for s in symbols}
    za = {s: frames["z_active"][s].to_numpy(dtype=bool) for s in symbols}

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
    cooldown_remaining = 0
    entry_opportunities = 0
    entries_admitted = 0
    blocked_by_cooldown = 0
    blocked_by_gate = 0
    # pending action decided at bar i-1, executed at bar i's open: (action, symbol, direction, atr_at_decision)
    pending: tuple[str, str, float, float] | None = None

    if n == 0:
        raise Wave26Error(f"{candidate_id}: empty calendar -- no symbol had any cached data")

    for i in range(n):
        ts = calendar[i]

        # --- Step A: execute the action decided at bar i-1, filled at bar i's own open. ---
        if pending is not None and not dead:
            action, symbol, new_direction, atr_at_decision = pending
            pending = None

            if action == "close" and held_symbol is not None:
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
                            Trade(held_symbol, direction, entry_time, ts, entry_price, float(fill_price), entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "signal_flip_exit", exit_cost)
                        )
                        cooldown_remaining = cooldown_bars
                    held_symbol = None
                    direction = 0.0

            elif action == "open" and new_direction != 0.0 and not dead and sleeve_equity > 0.0:
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
                    cooldown_remaining = cooldown_bars
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
                    # Always honored as a CLOSE, never gated (see module docstring) -- does NOT
                    # immediately reopen the opposite side the way engine25's own "reverse" does.
                    pending = ("close", held_symbol, 0.0, ra[held_symbol][i])
            else:
                any_raw_signal = any(sig[symbol][i] != 0.0 for symbol in priority)
                if any_raw_signal:
                    entry_opportunities += 1
                if cooldown_remaining <= 0:
                    admitted = False
                    for symbol in priority:
                        s_val = sig[symbol][i]
                        if s_val != 0.0 and aa[symbol][i] and za[symbol][i]:
                            pending = ("open", symbol, float(np.sign(s_val)), ra[symbol][i])
                            admitted = True
                            break
                    if admitted:
                        entries_admitted += 1
                    elif any_raw_signal:
                        blocked_by_gate += 1
                elif any_raw_signal:
                    blocked_by_cooldown += 1
            cooldown_remaining = max(0, cooldown_remaining - 1)

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
        "cooldown_bars_setting": cooldown_bars,
        "final_cooldown_remaining_bars": cooldown_remaining,
        "entry_admission": {
            "entry_opportunities": entry_opportunities,
            "entries_admitted": entries_admitted,
            "blocked_by_cooldown": blocked_by_cooldown,
            "blocked_by_gate": blocked_by_gate,
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return GambleResult(candidate_id=candidate_id, equity=equity_daily, trades=tuple(trades), symbols_used=tuple(symbols), metadata=metadata)


# ---------------------------------------------------------------------------
# V1-family controlled engine (C0/C7) -- fork of engine20.simulate_breakout_reversal. C0 itself
# never calls this (SPEC.md: C0 has NO control, so it reuses engine25.run_b0 verbatim -- see
# run_c0 below); only C7 does. V1's own armable/anchor/breakout-threshold arithmetic is
# byte-for-byte unchanged; the only addition is the cooldown/ADX gate on the FINAL decision to
# actually open a position (the anchor itself keeps arming/tracking exactly as V1 always has).
# ---------------------------------------------------------------------------


def simulate_breakout_reversal_controlled(
    index: pd.DatetimeIndex,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
    cost_arr: np.ndarray,
    worst_cost: float,
    atr_multiplier: float,
    armable_arr: np.ndarray | None,
    adx_active_arr: np.ndarray | None,  # bool per bar, or None = ADX gate disabled (always admissible)
    cooldown_bars: int,
    starting_equity: float,
    symbol: str,
) -> tuple[np.ndarray, list[Trade], float, dict[str, Any]]:
    n = len(close_arr)
    equity_out = np.empty(n, dtype=float)
    trades: list[Trade] = []
    total_cost = 0.0
    entry_opportunities = 0
    entries_admitted = 0
    blocked_by_cooldown = 0
    blocked_by_gate = 0
    if n == 0:
        return equity_out, trades, total_cost, {
            "entry_opportunities": 0, "entries_admitted": 0, "blocked_by_cooldown": 0, "blocked_by_gate": 0, "final_cooldown_remaining_bars": 0,
        }

    sleeve_equity = starting_equity
    direction = 0.0
    anchor = float("nan")
    entry_price = float("nan")
    extreme = float("nan")
    entry_time: pd.Timestamp | None = None
    entry_equity = starting_equity
    entry_idx = -1
    pending: tuple[str, float] | None = None
    dead = False
    cooldown_remaining = 0

    for i in range(n):
        ts = index[i]

        # --- Step A: execute a pending action decided at bar i-1, filled at bar i's open. By
        # construction "close" pending only ever occurs while direction!=0, and "open" pending
        # only ever occurs while direction==0.0 (Step C below only ever sets one or the other
        # depending on the CURRENT direction) -- so, unlike engine20's original combined
        # "reverse" (which could carry BOTH a close leg and an open leg in the same fill), each
        # branch here is exclusive. ---
        if pending is not None and not dead:
            action, new_direction = pending
            pending = None
            rate = cost_arr[i] if not np.isnan(cost_arr[i]) else worst_cost
            fill_price = float(open_arr[i])

            if action == "close" and direction != 0.0:
                prior_close = close_arr[i - 1] if i > 0 else float("nan")
                if not np.isnan(prior_close) and prior_close > 0.0:
                    gap_ret = fill_price / prior_close - 1.0
                    sleeve_equity = sleeve_equity * (1.0 + direction * gap_ret)
                if sleeve_equity <= 0.0:
                    trades.append(Trade(symbol, direction, entry_time, ts, entry_price, fill_price, entry_equity, -entry_equity, -1.0, "liquidated", entry_equity))
                    sleeve_equity = 0.0
                    dead = True
                    direction = 0.0
                else:
                    equity_before_exit_cost = sleeve_equity
                    sleeve_equity = sleeve_equity * (1.0 - rate)
                    exit_cost = equity_before_exit_cost - sleeve_equity
                    total_cost += exit_cost
                    pnl = sleeve_equity - entry_equity
                    trades.append(
                        Trade(symbol, direction, entry_time, ts, entry_price, fill_price, entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "signal_flip_exit", exit_cost)
                    )
                    direction = 0.0
                    cooldown_remaining = cooldown_bars

            elif action == "open" and new_direction != 0.0 and not dead and sleeve_equity > 0.0:
                equity_before_entry_cost = sleeve_equity
                sleeve_equity = sleeve_equity * (1.0 - rate)
                total_cost += equity_before_entry_cost - sleeve_equity
                if sleeve_equity <= 0.0:
                    dead = True
                    sleeve_equity = 0.0
                else:
                    direction = new_direction
                    entry_price = fill_price
                    extreme = fill_price
                    entry_time = ts
                    entry_equity = sleeve_equity
                    entry_idx = i
                    anchor = float("nan")

        # --- Step B: mark-to-market bar i's own session, then update the trailing extreme. (Unchanged from engine20.simulate_breakout_reversal.) ---
        if direction != 0.0 and not dead:
            ref_price = entry_price if entry_time == ts else close_arr[i - 1]
            if ref_price is not None and not np.isnan(ref_price) and ref_price > 0.0 and not np.isnan(close_arr[i]):
                bar_ret = close_arr[i] / ref_price - 1.0
                sleeve_equity = sleeve_equity * (1.0 + direction * bar_ret)
            if not np.isnan(close_arr[i]):
                extreme = max(extreme, close_arr[i]) if direction > 0.0 else min(extreme, close_arr[i])
            if sleeve_equity <= 0.0:
                trades.append(Trade(symbol, direction, entry_time, ts, entry_price, float(close_arr[i]), entry_equity, -entry_equity, -1.0, "liquidated", entry_equity))
                sleeve_equity = 0.0
                dead = True
                direction = 0.0

        equity_out[i] = sleeve_equity

        # --- Step C: decide, using bar i's own close + atr[i], what to do at bar i+1's open. ---
        if not dead and sleeve_equity > 0.0 and i + 1 < n:
            atr_i = atr_arr[i]
            if direction == 0.0:
                # V1's OWN arming rule -- byte-for-byte unchanged, never gated: the anchor keeps
                # tracking regardless of cooldown/ADX so a blocked breakout doesn't get a "fresh"
                # anchor for free once the gate clears (see module docstring).
                if armable_arr is not None and bool(armable_arr[i]) and np.isnan(anchor):
                    anchor = close_arr[i]
                if not np.isnan(anchor) and not np.isnan(atr_i) and atr_i > 0.0:
                    threshold = atr_multiplier * atr_i
                    breakout_direction = 0.0
                    if close_arr[i] - anchor >= threshold:
                        breakout_direction = 1.0
                    elif anchor - close_arr[i] >= threshold:
                        breakout_direction = -1.0
                    if breakout_direction != 0.0:
                        entry_opportunities += 1
                        gate_ok = (cooldown_remaining <= 0) and (adx_active_arr is None or bool(adx_active_arr[i]))
                        if gate_ok:
                            pending = ("open", breakout_direction)
                            entries_admitted += 1
                        elif cooldown_remaining > 0:
                            blocked_by_cooldown += 1
                        else:
                            blocked_by_gate += 1
            else:
                # Chandelier-style reversal trigger (V1's OWN math, unchanged) -- always honored
                # as a CLOSE, never gated (see module docstring).
                if not np.isnan(atr_i) and atr_i > 0.0:
                    threshold = atr_multiplier * atr_i
                    if direction > 0.0 and (extreme - close_arr[i]) >= threshold:
                        pending = ("close", 0.0)
                    elif direction < 0.0 and (close_arr[i] - extreme) >= threshold:
                        pending = ("close", 0.0)
            cooldown_remaining = max(0, cooldown_remaining - 1)

    # force-close any position still open at the end of available data (unchanged from engine20;
    # NOT a cooldown-triggering exit -- it's an artifact of the backtest window ending, not a
    # genuine signal/stop event, so cooldown_remaining is left as-is).
    if direction != 0.0 and not dead:
        rate = cost_arr[-1] if not np.isnan(cost_arr[-1]) else worst_cost
        equity_before_exit_cost = sleeve_equity
        sleeve_equity = sleeve_equity * (1.0 - rate)
        exit_cost = equity_before_exit_cost - sleeve_equity
        total_cost += exit_cost
        pnl = sleeve_equity - entry_equity
        trades.append(
            Trade(symbol, direction, entry_time, index[-1], entry_price, float(close_arr[-1]), entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "end_of_data", exit_cost)
        )
        equity_out[-1] = sleeve_equity

    diagnostics = {
        "entry_opportunities": entry_opportunities,
        "entries_admitted": entries_admitted,
        "blocked_by_cooldown": blocked_by_cooldown,
        "blocked_by_gate": blocked_by_gate,
        "cooldown_bars_setting": cooldown_bars,
        "final_cooldown_remaining_bars": cooldown_remaining,
    }
    return equity_out, trades, total_cost, diagnostics


# ---------------------------------------------------------------------------
# Candidate runners.
# ---------------------------------------------------------------------------


def run_c0(mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    """C0 = wave25 B0 verbatim (V1 reproduction), NO frequency control (SPEC.md: "없음 --
    기준선"). Must reproduce wave25 B0's numbers exactly -- see tests/test_wave26.py. Accepts
    (unused) hourly_frames/daily_frames only so every RUNNERS entry shares one call signature
    (the live stage in run_wave26.py calls every candidate uniformly)."""
    del hourly_frames, daily_frames
    result = engine25.run_b0(mapping=mapping, stress_multiplier=stress_multiplier)
    metadata = dict(result.metadata)
    metadata["source"] = "research.wave25_gamble.engine25.run_b0 (verbatim reuse, not reimplemented) -- wave26 C0 baseline, no frequency control"
    metadata["control"] = {"cooldown_days": 0, "adx_gate": False, "z_gate": False}
    metadata["base_family"] = "V1"
    return GambleResult(candidate_id="C0", equity=result.equity, trades=result.trades, symbols_used=result.symbols_used, metadata=metadata)


def run_c0_live(hourly_btc: pd.DataFrame, mapping: MeasuredCostMapping, stress_multiplier: float = 1.0) -> GambleResult:
    """Live-stage counterpart, mirrors engine25._run_b0_with_hourly exactly (replays V1's own
    math against caller-supplied, possibly network-freshened BTC hourly data)."""
    result = engine25._run_b0_with_hourly(hourly_btc, mapping, stress_multiplier=stress_multiplier)
    metadata = dict(result.metadata)
    metadata["control"] = {"cooldown_days": 0, "adx_gate": False, "z_gate": False}
    metadata["base_family"] = "V1"
    return GambleResult(candidate_id="C0", equity=result.equity, trades=result.trades, symbols_used=result.symbols_used, metadata=metadata)


def _run_controlled_candidate(
    spec: ControlSpec,
    symbols: tuple[str, ...] = SYMBOLS,
    mapping: MeasuredCostMapping | None = None,
    stress_multiplier: float = 1.0,
    hourly_frames: dict[str, pd.DataFrame] | None = None,
    daily_frames: dict[str, pd.DataFrame] | None = None,
) -> GambleResult:
    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    hourly, daily = _resolve_frames(symbols, hourly_frames, daily_frames)
    base_signal_fn = BASE_SIGNAL_FNS[spec.base_family]
    signals = {s: base_signal_fn(hourly[s], daily[s]) for s in symbols}
    strength_fn = STRENGTH_FNS.get(spec.base_family) if spec.z_gate else None
    frames = build_controlled_frames(hourly, daily, signals, mapping, stress_multiplier, symbols, spec.adx_gate, spec.z_gate, strength_fn)
    worst_cost = worst_case_cost(mapping, stress_multiplier)
    cooldown_bars = spec.cooldown_days * 24
    return run_multi_symbol_convex_controlled(
        frames,
        priority=symbols,
        worst_cost=worst_cost,
        candidate_id=spec.candidate_id,
        cooldown_bars=cooldown_bars,
        extra_metadata={
            "control": {"cooldown_days": spec.cooldown_days, "adx_gate": spec.adx_gate, "z_gate": spec.z_gate},
            "base_family": spec.base_family,
            "config": CANDIDATE_BASE_CONFIG[spec.base_family],
            "symbols_scanned": list(symbols),
        },
    )


def run_c1(symbols=SYMBOLS, mapping=None, stress_multiplier=1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_controlled_candidate(C1_SPEC, symbols, mapping, stress_multiplier, hourly_frames, daily_frames)


def run_c2(symbols=SYMBOLS, mapping=None, stress_multiplier=1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_controlled_candidate(C2_SPEC, symbols, mapping, stress_multiplier, hourly_frames, daily_frames)


def run_c3(symbols=SYMBOLS, mapping=None, stress_multiplier=1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_controlled_candidate(C3_SPEC, symbols, mapping, stress_multiplier, hourly_frames, daily_frames)


def run_c4(symbols=SYMBOLS, mapping=None, stress_multiplier=1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_controlled_candidate(C4_SPEC, symbols, mapping, stress_multiplier, hourly_frames, daily_frames)


def run_c5(symbols=SYMBOLS, mapping=None, stress_multiplier=1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_controlled_candidate(C5_SPEC, symbols, mapping, stress_multiplier, hourly_frames, daily_frames)


def run_c6(symbols=SYMBOLS, mapping=None, stress_multiplier=1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    return _run_controlled_candidate(C6_SPEC, symbols, mapping, stress_multiplier, hourly_frames, daily_frames)


def run_c7(mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0, hourly_frames=None, daily_frames=None) -> GambleResult:
    """C7 = V1's own breakout/chandelier-reversal engine (BTC only, byte-for-byte unchanged --
    see simulate_breakout_reversal_controlled's own docstring), plus cooldown(5 days)+ADX(14)>20
    gating the entry decision only. SPEC.md: "기준선에도 통제를 걸면 더 좋아지나?"."""
    from research.wave20_convex.configs20 import V1_CONFIG

    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    symbol = V1_CONFIG.symbol
    if hourly_frames is not None and symbol in hourly_frames:
        hourly = hourly_frames[symbol]
        daily = daily_frames[symbol] if daily_frames is not None and symbol in daily_frames else dataio20.resample_hourly_to_daily(hourly)
    else:
        hourly = dataio20.load_hourly(symbol, WAVE6_CACHE_DIR)
        daily = dataio20.resample_hourly_to_daily(hourly)

    vol20 = wave20_engine.realized_vol(daily["close"], V1_CONFIG.vol_window_days)
    vol_pct_rank = wave20_engine.trailing_percentile_rank(vol20, V1_CONFIG.vol_percentile_lookback_days)
    armable_daily = (vol_pct_rank < V1_CONFIG.vol_percentile_threshold).shift(1).fillna(False)
    cost_daily = one_leg_cost_rate_series(daily["quote_volume"], mapping, stress_multiplier)
    armable_hourly = armable_daily.reindex(hourly.index, method="ffill").fillna(False).astype(bool)
    cost_hourly = cost_daily.reindex(hourly.index, method="ffill")
    atr_daily_lagged = wave20_engine.atr(daily, V1_CONFIG.atr_window_days).shift(1)
    atr_hourly = atr_daily_lagged.reindex(hourly.index, method="ffill")
    adx_hourly = indicators25.adx_dmi(hourly, ADX_REGIME_WINDOW)["adx"]
    adx_active_hourly = (adx_hourly > ADX_REGIME_THRESHOLD).reindex(hourly.index).fillna(False).astype(bool)

    worst_cost = worst_case_cost(mapping, stress_multiplier)
    cooldown_bars = C7_SPEC.cooldown_days * 24

    equity_arr, trades, total_cost, diagnostics = simulate_breakout_reversal_controlled(
        index=hourly.index,
        open_arr=hourly["open"].to_numpy(dtype=float),
        close_arr=hourly["close"].to_numpy(dtype=float),
        atr_arr=atr_hourly.to_numpy(dtype=float),
        cost_arr=cost_hourly.to_numpy(dtype=float),
        worst_cost=worst_cost,
        atr_multiplier=V1_CONFIG.atr_multiplier,
        armable_arr=armable_hourly.to_numpy(dtype=bool),
        adx_active_arr=adx_active_hourly.to_numpy(dtype=bool),
        cooldown_bars=cooldown_bars,
        starting_equity=GAMBLE_CAPITAL,
        symbol=symbol,
    )
    equity_hourly = pd.Series(equity_arr, index=hourly.index)
    equity_daily = equity_hourly.resample("1D").last().ffill()
    n_signal_flip_exits = sum(1 for t in trades if t.exit_reason == "signal_flip_exit")
    metadata = {
        "n_trades": len(trades),
        "n_signal_flip_exits": n_signal_flip_exits,
        "n_bars": int(len(hourly)),
        "armable_days": int(armable_daily.sum()),
        "total_cost_usdt": total_cost,
        "final_equity_usdt": float(equity_daily.iloc[-1]) if len(equity_daily) else GAMBLE_CAPITAL,
        "control": {"cooldown_days": C7_SPEC.cooldown_days, "adx_gate": True, "z_gate": False},
        "base_family": "V1",
        "entry_admission": diagnostics,
        "config": {
            "symbol": symbol,
            "atr_window_days": V1_CONFIG.atr_window_days,
            "atr_multiplier": V1_CONFIG.atr_multiplier,
            "vol_window_days": V1_CONFIG.vol_window_days,
            "vol_percentile_lookback_days": V1_CONFIG.vol_percentile_lookback_days,
            "vol_percentile_threshold": V1_CONFIG.vol_percentile_threshold,
        },
    }
    return GambleResult(candidate_id="C7", equity=equity_daily, trades=tuple(trades), symbols_used=(symbol,), metadata=metadata)


RUNNERS: Final[dict[str, Callable[..., GambleResult]]] = {
    "C0": run_c0,
    "C1": run_c1,
    "C2": run_c2,
    "C3": run_c3,
    "C4": run_c4,
    "C5": run_c5,
    "C6": run_c6,
    "C7": run_c7,
}


def run_candidate(candidate_id: str, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    if candidate_id not in RUNNERS:
        raise Wave26Error(f"unknown wave26 candidate: {candidate_id}")
    if candidate_id == "C0":
        return run_c0(mapping=mapping, stress_multiplier=stress_multiplier)
    return RUNNERS[candidate_id](mapping=mapping, stress_multiplier=stress_multiplier)


__all__ = [
    "BASE_SIGNAL_FNS",
    "CANDIDATE_BASE_CONFIG",
    "GambleResult",
    "RUNNERS",
    "STRENGTH_FNS",
    "Trade",
    "Wave26Error",
    "build_controlled_frames",
    "combine_portfolio",
    "ensemble_agreement_strength",
    "load_daily",
    "load_hourly",
    "load_stable_leg",
    "macd_histogram_strength",
    "pad_to_calendar",
    "run_c0",
    "run_c0_live",
    "run_c1",
    "run_c2",
    "run_c3",
    "run_c4",
    "run_c5",
    "run_c6",
    "run_c7",
    "run_candidate",
    "run_multi_symbol_convex_controlled",
    "simulate_breakout_reversal_controlled",
    "zscore_gate_mask",
]
