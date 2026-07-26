# Wave-20 convex-gamble engines (V1-V5). See research/wave20_convex/SPEC.md for the frozen
# pre-registration this module implements, and configs20.py for every numeric threshold
# (nothing here should hardcode a SPEC.md number a second time).
#
# ---------------------------------------------------------------------------------------
# Why single-leg cost, not costs_measured.cost_rate_from_bp (wave13's own 2x formula)
# ---------------------------------------------------------------------------------------
# research.wave13_liquidity.costs_measured.cost_rate_from_bp prices a delta-neutral
# spot+perp CARRY PAIR (two simultaneous instruments per "position" -- see that module's own
# docstring). Every wave20 candidate is an outright DIRECTIONAL position in ONE instrument
# (long or short a single perpetual, no offsetting spot leg), so this module halves that
# convention: one_leg_cost_rate_* below = maker fee ONCE + measured slippage ONCE. A full
# round trip still pays this twice (once at entry, once at exit/reversal), applied exactly
# the same way engine13/engine18 apply their own turnover-based cost -- only the per-leg
# rate differs, not the mechanism. The underlying measured-slippage MAPPING itself
# (costs_measured.fit_mapping / slippage_bp_for_volume, fit from real $45-order Bitget
# book-walks) is reused completely unmodified.
#
# ---------------------------------------------------------------------------------------
# Why every candidate floors sleeve equity at $0 (G1's structural loss cap)
# ---------------------------------------------------------------------------------------
# SPEC.md G1 requires "최대손실이 구조적으로 배분액($25) 이내" -- a structure that CAN lose more
# than its own allocation must be rejected outright. V1 and V3's whipsaw-reversal rule can go
# SHORT, and an unleveraged short has theoretically unbounded loss (price can rise >100%). To
# keep the structure G1-compliant this module models every leg as ISOLATED MARGIN, auto-closed
# the instant its mark-to-market loss would reach -100% of the capital committed to it (a
# realistic model of how isolated-margin perpetuals actually behave; the liquidation FEE itself
# is not separately modeled beyond the maker+slippage cost already charged, a disclosed
# simplification). This is implemented as a hard floor at 0.0 on sleeve equity, checked every
# bar, never allowed to go negative -- gates20.gate_g1_structural_loss_cap re-verifies this
# empirically from the saved equity series, it is not just a comment-level claim.
#
# ---------------------------------------------------------------------------------------
# Shared t -> t+1 no-lookahead discipline
# ---------------------------------------------------------------------------------------
# V1/V3/V4 (event/breakout candidates) decide using bar t's own close (plus indicators
# computed through bar t-1) and FILL at bar t+1's open -- the same discipline
# research.wave9_100usd.engine_w9's own module docstring states explicitly for its
# breakout table. V2 instead reuses research.wave1.fam_funding's carry_position/funding_score
# UNMODIFIED, which bake their own 1-day lag directly into the returned series (see that
# module), and then follows engine13/18/fam_funding.run_portfolio's own established
# gap+intraday decomposition to apply a same-day decision -- a different but equally
# lookahead-free convention, kept identical to the functions it reuses rather than bolting a
# second lag on top of an already-lagged signal.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Callable, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.fam_funding import FundingCandidate, carry_position, funding_score
from research.wave13_liquidity import costs_measured
from research.wave13_liquidity.costs_measured import MeasuredCostMapping
from research.wave20_convex import dataio20
from research.wave20_convex.configs20 import (
    GAMBLE_CAPITAL,
    I5_RESULTS_PATH,
    MAKER_FEE_RATE,
    STABLE_CAPITAL,
    V1Config,
    V1_CONFIG,
    V2Config,
    V2_CONFIG,
    V3Config,
    V3_CONFIG,
    V4Config,
    V4_CONFIG,
    V5Config,
    V5_CONFIG,
    WAVE1_CACHE_DIR,
    WAVE3_CACHE_DIR,
    WAVE6_CACHE_DIR,
)


class Wave20Error(Exception):
    pass


# ---------------------------------------------------------------------------
# Shared math primitives. Reimplemented locally (not imported from
# research.wave9_100usd.engine_w9, which would transitively import
# research.wave4_leverage.sweep -- out of bounds for this wave per the task brief). Formulas
# match the repo-wide convention (true_range = max of the three standard components; ATR =
# simple rolling mean of true_range; matches wave9's own true_range/atr exactly, just kept
# dependency-free here).
# ---------------------------------------------------------------------------


def true_range(ohlc: pd.DataFrame) -> pd.Series:
    prior_close = ohlc["close"].shift(1)
    ranges = pd.concat(
        [ohlc["high"] - ohlc["low"], (ohlc["high"] - prior_close).abs(), (ohlc["low"] - prior_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(ohlc: pd.DataFrame, window: int) -> pd.Series:
    return true_range(ohlc).rolling(window, min_periods=window).mean()


def expanding_atr(ohlc: pd.DataFrame, min_periods: int = 2) -> pd.Series:
    """Used only by V3: a brand-new listing has no PRE-listing bars to compute a fixed-window
    ATR from, so its within-window reversal threshold instead grows from the listing's own
    observed bars (mean true_range over bars seen so far). Disclosed simplification vs a
    "real" 14-bar ATR -- see run_v3's docstring."""
    return true_range(ohlc).expanding(min_periods=min_periods).mean()


def realized_vol(close: pd.Series, window_days: int) -> pd.Series:
    return close.pct_change().rolling(window_days, min_periods=window_days).std()


def trailing_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of series[t] within the trailing `window` observations ending at t
    (inclusive) -- point-in-time by construction (never looks past t). Implemented by hand
    (rather than pandas' `.rolling().rank(pct=True)`, whose exact tie-breaking/edge semantics
    are less auditable) so tests/test_wave20.py can pin the exact definition being used."""

    def _rank_last(window_values: np.ndarray) -> float:
        last = window_values[-1]
        return float(np.mean(window_values <= last))

    return series.rolling(window, min_periods=window).apply(_rank_last, raw=True)


def one_leg_cost_rate_series(
    quote_volume: pd.Series, mapping: MeasuredCostMapping, stress_multiplier: float = 1.0, window: int = costs_measured.ROLLING_WINDOW_DAYS
) -> pd.Series:
    """Single-leg point-in-time cost rate (see module docstring) for ONE symbol's daily
    quote-volume series."""
    known_avg = quote_volume.rolling(window, min_periods=window).mean().shift(1)
    bp_frame = costs_measured.bp_frame_from_known_avg(known_avg.to_frame(name="v"), mapping)
    return MAKER_FEE_RATE + bp_frame["v"] * 0.0001 * stress_multiplier


def one_leg_cost_rate_frame(
    quote_volume_frame: pd.DataFrame, mapping: MeasuredCostMapping, stress_multiplier: float = 1.0, window: int = costs_measured.ROLLING_WINDOW_DAYS
) -> pd.DataFrame:
    known_avg = quote_volume_frame.rolling(window, min_periods=window).mean().shift(1)
    bp_frame = costs_measured.bp_frame_from_known_avg(known_avg, mapping)
    return MAKER_FEE_RATE + bp_frame * 0.0001 * stress_multiplier


def worst_case_cost(mapping: MeasuredCostMapping, stress_multiplier: float = 1.0) -> float:
    return MAKER_FEE_RATE + mapping.worst_bp * 0.0001 * stress_multiplier


# ---------------------------------------------------------------------------
# Shared result types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    direction: float  # +1.0 long, -1.0 short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    entry_equity_usdt: float  # sleeve equity right after this trade's entry cost, its own compounding base
    pnl_usdt: float  # net of entry+exit costs, floor-clipped at -entry_equity_usdt (G1)
    pnl_fraction: float  # pnl_usdt / entry_equity_usdt
    exit_reason: str
    cost_usdt: float  # entry + exit dollar cost attributed to this trade


@dataclass(frozen=True, slots=True)
class GambleResult:
    candidate_id: str
    equity: pd.Series  # $GAMBLE_CAPITAL-basis, DAILY (last-of-day), gap-free over the candidate's own native calendar
    trades: tuple[Trade, ...]
    symbols_used: tuple[str, ...]
    metadata: dict


def _clip_fraction(value: float) -> float:
    return max(value, -1.0)


# ---------------------------------------------------------------------------
# Shared breakout/reversal core (V1 continuous, V3 per-listing-window). Both are the SAME
# state machine ("flat and armed" -> ±multiplier*ATR breakout opens a position; while in a
# position, ±multiplier*ATR against entry reverses it) -- see module docstring's t->t+1 note
# for the execution-timing discipline this implements bar by bar.
# ---------------------------------------------------------------------------


def simulate_breakout_reversal(
    index: pd.DatetimeIndex,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,  # ATR usable AT bar i (caller has already shifted so this never uses bar i's own high/low/close)
    cost_arr: np.ndarray,  # one-leg cost rate usable at bar i; NaN falls back to worst_cost
    worst_cost: float,
    atr_multiplier: float,
    armable_arr: np.ndarray | None,  # bool per bar, consulted only while FLAT; None => no auto-arm (used with forced_initial_direction)
    forced_initial_direction: float | None,  # V3: bar 0 opens a position immediately (the listing print itself is the signal)
    max_bars_in_position: int | None,  # V3: force-close this many bars after entry
    starting_equity: float,
    symbol: str,
) -> tuple[np.ndarray, list[Trade], float]:
    n = len(close_arr)
    equity_out = np.empty(n, dtype=float)
    trades: list[Trade] = []
    total_cost = 0.0
    if n == 0:
        return equity_out, trades, total_cost

    sleeve_equity = starting_equity
    direction = 0.0
    anchor = float("nan")
    entry_price = float("nan")
    extreme = float("nan")  # trailing favorable extreme since entry -- see reversal note below
    entry_time: pd.Timestamp | None = None
    entry_equity = starting_equity
    entry_idx = -1
    pending: tuple[str, float] | None = None
    dead = False

    if forced_initial_direction is not None and n > 0:
        rate0 = cost_arr[0] if not np.isnan(cost_arr[0]) else worst_cost
        fill0 = open_arr[0]
        equity_before = sleeve_equity
        sleeve_equity = sleeve_equity * (1.0 - rate0)
        total_cost += equity_before - sleeve_equity
        direction = forced_initial_direction
        entry_price = float(fill0)
        extreme = entry_price
        entry_time = index[0]
        entry_equity = sleeve_equity
        entry_idx = 0

    for i in range(n):
        ts = index[i]

        # --- Step A: execute a pending action decided at bar i-1, filled at bar i's open. ---
        if pending is not None and not dead:
            action, new_direction = pending
            pending = None
            rate = cost_arr[i] if not np.isnan(cost_arr[i]) else worst_cost
            fill_price = float(open_arr[i])
            if direction != 0.0:
                prior_close = close_arr[i - 1] if i > 0 else float("nan")
                if not np.isnan(prior_close) and prior_close > 0.0:
                    gap_ret = fill_price / prior_close - 1.0
                    sleeve_equity = sleeve_equity * (1.0 + direction * gap_ret)
                if sleeve_equity <= 0.0:
                    trades.append(
                        Trade(symbol, direction, entry_time, ts, entry_price, fill_price, entry_equity, -entry_equity, -1.0, "liquidated", entry_equity)
                    )
                    sleeve_equity = 0.0
                    dead = True
                    direction = 0.0
                if not dead:
                    equity_before_exit_cost = sleeve_equity
                    sleeve_equity = sleeve_equity * (1.0 - rate)
                    exit_cost = equity_before_exit_cost - sleeve_equity
                    total_cost += exit_cost
                    pnl = sleeve_equity - entry_equity
                    trades.append(
                        Trade(symbol, direction, entry_time, ts, entry_price, fill_price, entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "reversal", exit_cost)
                    )
                    direction = 0.0
            if not dead and new_direction != 0.0:
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

        # --- Step B: mark-to-market bar i's own session, then update the trailing extreme. ---
        if direction != 0.0 and not dead:
            ref_price = entry_price if entry_time == ts else close_arr[i - 1]
            if ref_price is not None and not np.isnan(ref_price) and ref_price > 0.0 and not np.isnan(close_arr[i]):
                bar_ret = close_arr[i] / ref_price - 1.0
                sleeve_equity = sleeve_equity * (1.0 + direction * bar_ret)
            if not np.isnan(close_arr[i]):
                extreme = max(extreme, close_arr[i]) if direction > 0.0 else min(extreme, close_arr[i])
            if sleeve_equity <= 0.0:
                trades.append(
                    Trade(symbol, direction, entry_time, ts, entry_price, float(close_arr[i]), entry_equity, -entry_equity, -1.0, "liquidated", entry_equity)
                )
                sleeve_equity = 0.0
                dead = True
                direction = 0.0

        # --- max_bars_in_position: forced close AT this bar's close (no next bar to open into for a windowed candidate). ---
        if direction != 0.0 and not dead and max_bars_in_position is not None and (i - entry_idx) >= max_bars_in_position:
            rate = cost_arr[i] if not np.isnan(cost_arr[i]) else worst_cost
            equity_before_exit_cost = sleeve_equity
            sleeve_equity = sleeve_equity * (1.0 - rate)
            exit_cost = equity_before_exit_cost - sleeve_equity
            total_cost += exit_cost
            pnl = sleeve_equity - entry_equity
            trades.append(
                Trade(symbol, direction, entry_time, ts, entry_price, float(close_arr[i]), entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "window_close", exit_cost)
            )
            direction = 0.0

        equity_out[i] = sleeve_equity

        # --- Step C: decide, using bar i's own close + atr[i] (already point-in-time), what to do at bar i+1's open. ---
        if not dead and sleeve_equity > 0.0 and i + 1 < n:
            atr_i = atr_arr[i]
            if direction == 0.0:
                if armable_arr is not None and bool(armable_arr[i]) and np.isnan(anchor):
                    anchor = close_arr[i]
                if not np.isnan(anchor) and not np.isnan(atr_i) and atr_i > 0.0:
                    threshold = atr_multiplier * atr_i
                    if close_arr[i] - anchor >= threshold:
                        pending = ("open", 1.0)
                    elif anchor - close_arr[i] >= threshold:
                        pending = ("open", -1.0)
            else:
                # Reversal is measured from the TRAILING extreme since entry (chandelier-exit
                # style), not the fixed entry price: anchoring to a fixed entry price would let
                # a strongly-trending position become structurally un-reversible once price has
                # moved far enough away (ATR grows with the price level, so
                # entry_price -/+ multiplier*ATR_now eventually falls outside any range price
                # could plausibly revisit) -- that would manufacture "convexity" as a mechanical
                # artifact of a stale anchor rather than a genuinely repeated whipsaw/breakout
                # test, which is exactly the confound SPEC.md's G3 bootstrap is there to catch.
                # Trailing the extreme keeps the ±multiplier*ATR band live against CURRENT price
                # for as long as the position is held, so every pullback of that size is tested.
                if not np.isnan(atr_i) and atr_i > 0.0:
                    threshold = atr_multiplier * atr_i
                    if direction > 0.0 and (extreme - close_arr[i]) >= threshold:
                        pending = ("reverse", -1.0)
                    elif direction < 0.0 and (close_arr[i] - extreme) >= threshold:
                        pending = ("reverse", 1.0)

    # force-close any position still open at the end of available data
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

    return equity_out, trades, total_cost


# ---------------------------------------------------------------------------
# V1 -- 양방향 돌파 추격 (long-vol straddle approximation), BTC perp.
# ---------------------------------------------------------------------------


def run_v1(config: V1Config = V1_CONFIG, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    hourly = dataio20.load_hourly(config.symbol, WAVE6_CACHE_DIR)
    daily = dataio20.resample_hourly_to_daily(hourly)

    vol20 = realized_vol(daily["close"], config.vol_window_days)
    vol_pct_rank = trailing_percentile_rank(vol20, config.vol_percentile_lookback_days)
    low_vol_regime_daily = vol_pct_rank < config.vol_percentile_threshold
    armable_daily = low_vol_regime_daily.shift(1).fillna(False)
    cost_daily = one_leg_cost_rate_series(daily["quote_volume"], mapping, stress_multiplier)

    armable_hourly = armable_daily.reindex(hourly.index, method="ffill").fillna(False).astype(bool)
    cost_hourly = cost_daily.reindex(hourly.index, method="ffill")
    # Daily ATR (see configs20.V1Config.atr_window_days docstring), lagged by one day the same
    # way armable_daily is, then broadcast onto the hourly execution grid: the value usable for
    # ALL of day D+1's hourly bars is the ATR computed through day D's own close.
    atr_daily_lagged = atr(daily, config.atr_window_days).shift(1)
    atr_hourly = atr_daily_lagged.reindex(hourly.index, method="ffill")

    worst_cost = worst_case_cost(mapping, stress_multiplier)
    equity_arr, trades, total_cost = simulate_breakout_reversal(
        index=hourly.index,
        open_arr=hourly["open"].to_numpy(dtype=float),
        close_arr=hourly["close"].to_numpy(dtype=float),
        atr_arr=atr_hourly.to_numpy(dtype=float),
        cost_arr=cost_hourly.to_numpy(dtype=float),
        worst_cost=worst_cost,
        atr_multiplier=config.atr_multiplier,
        armable_arr=armable_hourly.to_numpy(dtype=bool),
        forced_initial_direction=None,
        max_bars_in_position=None,
        starting_equity=GAMBLE_CAPITAL,
        symbol=config.symbol,
    )
    equity_hourly = pd.Series(equity_arr, index=hourly.index)
    equity_daily = equity_hourly.resample("1D").last().ffill()

    n_reversals = sum(1 for t in trades if t.exit_reason == "reversal")
    return GambleResult(
        candidate_id=config.candidate_id,
        equity=equity_daily,
        trades=tuple(trades),
        symbols_used=(config.symbol,),
        metadata={
            "n_trades": len(trades),
            "n_reversals": n_reversals,
            "n_bars": int(len(hourly)),
            "armable_days": int(armable_daily.sum()),
            "total_cost_usdt": total_cost,
            "final_equity_usdt": float(equity_daily.iloc[-1]) if len(equity_daily) else GAMBLE_CAPITAL,
            "config": {
                "symbol": config.symbol,
                "atr_window_days": config.atr_window_days,
                "atr_multiplier": config.atr_multiplier,
                "vol_window_days": config.vol_window_days,
                "vol_percentile_lookback_days": config.vol_percentile_lookback_days,
                "vol_percentile_threshold": config.vol_percentile_threshold,
            },
        },
    )


# ---------------------------------------------------------------------------
# V2 -- 꼬리 사냥 (funding-extreme directional LONG, deliberately opposite of a carry trade).
# ---------------------------------------------------------------------------


def run_v2(config: V2Config = V2_CONFIG, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    symbols = tuple(s for s in dataio20.wave1_symbols_with_funding() if s not in config.excluded_symbols)
    candidate = FundingCandidate("V2_squeeze", config.funding_window_days, config.entry_threshold_apr, config.top_k)

    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    funding_daily: dict[str, pd.Series] = {}
    raw_scores: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    quote_volumes: dict[str, pd.Series] = {}
    used_symbols: list[str] = []
    for symbol in symbols:
        daily = dataio20.try_load_daily(symbol, WAVE1_CACHE_DIR)
        if daily is None:
            continue
        try:
            funding = dataio20.load_funding_rate(symbol, WAVE1_CACHE_DIR)
        except dataio20.DataError:
            continue
        score = funding_score(funding, config.funding_window_days).resample("1D").last()
        closes[symbol] = daily["close"]
        opens[symbol] = daily["open"]
        quote_volumes[symbol] = daily["quote_volume"]
        funding_daily[symbol] = funding.resample("1D").sum()
        raw_scores[symbol] = score
        active[symbol] = carry_position(score, candidate)
        used_symbols.append(symbol)

    if not used_symbols:
        raise Wave20Error("V2: no symbols with both price and funding cache data")

    close_frame = pd.DataFrame(closes).sort_index()
    open_frame = pd.DataFrame(opens).reindex(close_frame.index)
    funding_frame = pd.DataFrame(funding_daily).reindex(close_frame.index).fillna(0.0)
    active_frame = pd.DataFrame(active).reindex(close_frame.index).fillna(0.0)
    score_frame = pd.DataFrame(raw_scores).reindex(close_frame.index).shift(1)
    quote_volume_frame = pd.DataFrame(quote_volumes).reindex(close_frame.index)
    cost_rate_frame = one_leg_cost_rate_frame(quote_volume_frame, mapping, stress_multiplier)
    worst_cost = worst_case_cost(mapping, stress_multiplier)

    sleeve_equity = GAMBLE_CAPITAL
    held: str | None = None
    entry_price = float("nan")
    entry_time: pd.Timestamp | None = None
    entry_equity = GAMBLE_CAPITAL
    trades: list[Trade] = []
    equity_values: list[float] = []
    dead = False
    total_cost = 0.0

    for ts in close_frame.index:
        if dead:
            equity_values.append(0.0)
            continue

        if held is not None:
            prior_close = close_frame[held].shift(1).loc[ts]
            today_open = open_frame[held].loc[ts]
            if pd.notna(prior_close) and prior_close > 0.0 and pd.notna(today_open):
                gap_ret = float(today_open) / float(prior_close) - 1.0
                sleeve_equity *= 1.0 + gap_ret
            if sleeve_equity <= 0.0:
                exit_price = float(today_open) if pd.notna(today_open) else float(entry_price)
                trades.append(Trade(held, 1.0, entry_time, ts, entry_price, exit_price, entry_equity, -entry_equity, -1.0, "liquidated", entry_equity))
                sleeve_equity = 0.0
                dead = True
                held = None

        if not dead:
            eligible_row = active_frame.loc[ts]
            eligible_symbols = eligible_row[eligible_row > 0.0].index
            available = close_frame.loc[ts].notna() & open_frame.loc[ts].notna()
            eligible_symbols = eligible_symbols.intersection(available[available].index)
            ranked = score_frame.loc[ts, eligible_symbols].dropna().nlargest(config.top_k).index
            new_symbol = str(ranked[0]) if len(ranked) > 0 else None

            if new_symbol != held:
                rate = float(cost_rate_frame[held].loc[ts]) if held is not None and pd.notna(cost_rate_frame[held].loc[ts]) else worst_cost
                if held is not None:
                    fill_price = float(open_frame[held].loc[ts])
                    equity_before = sleeve_equity
                    sleeve_equity *= 1.0 - rate
                    exit_cost = equity_before - sleeve_equity
                    total_cost += exit_cost
                    pnl = sleeve_equity - entry_equity
                    trades.append(Trade(held, 1.0, entry_time, ts, entry_price, fill_price, entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "rotated", exit_cost))
                if new_symbol is not None:
                    rate_new = float(cost_rate_frame[new_symbol].loc[ts]) if pd.notna(cost_rate_frame[new_symbol].loc[ts]) else worst_cost
                    fill_price_new = float(open_frame[new_symbol].loc[ts])
                    equity_before_entry = sleeve_equity
                    sleeve_equity *= 1.0 - rate_new
                    total_cost += equity_before_entry - sleeve_equity
                    entry_price = fill_price_new
                    entry_time = ts
                    entry_equity = sleeve_equity
                held = new_symbol

        if held is not None and not dead:
            o = open_frame[held].loc[ts]
            c = close_frame[held].loc[ts]
            funding_today = funding_frame[held].loc[ts]
            if pd.notna(o) and float(o) > 0.0 and pd.notna(c):
                intraday_ret = float(c) / float(o) - 1.0
                sleeve_equity *= 1.0 + intraday_ret - float(funding_today)
            if sleeve_equity <= 0.0:
                trades.append(Trade(held, 1.0, entry_time, ts, entry_price, float(c) if pd.notna(c) else entry_price, entry_equity, -entry_equity, -1.0, "liquidated", entry_equity))
                sleeve_equity = 0.0
                dead = True
                held = None

        equity_values.append(sleeve_equity)

    if held is not None and not dead:
        final_ts = close_frame.index[-1]
        rate = float(cost_rate_frame[held].loc[final_ts]) if pd.notna(cost_rate_frame[held].loc[final_ts]) else worst_cost
        equity_before = sleeve_equity
        sleeve_equity *= 1.0 - rate
        exit_cost = equity_before - sleeve_equity
        total_cost += exit_cost
        pnl = sleeve_equity - entry_equity
        trades.append(
            Trade(held, 1.0, entry_time, final_ts, entry_price, float(close_frame[held].loc[final_ts]), entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else 0.0, "end_of_data", exit_cost)
        )
        equity_values[-1] = sleeve_equity

    equity_series = pd.Series(equity_values, index=close_frame.index, dtype=float)
    return GambleResult(
        candidate_id=config.candidate_id,
        equity=equity_series,
        trades=tuple(trades),
        symbols_used=tuple(used_symbols),
        metadata={
            "n_trades": len(trades),
            "universe_size": len(used_symbols),
            "total_cost_usdt": total_cost,
            "final_equity_usdt": float(equity_series.iloc[-1]) if len(equity_series) else GAMBLE_CAPITAL,
            "config": {
                "funding_window_days": config.funding_window_days,
                "entry_threshold_apr": config.entry_threshold_apr,
                "exit_threshold_apr": config.exit_threshold_apr,
                "top_k": config.top_k,
            },
        },
    )


# ---------------------------------------------------------------------------
# V3 -- 신규상장 첫 7일 (event-driven, single $25 sleeve, non-overlapping listings only).
# ---------------------------------------------------------------------------


def run_v3(config: V3Config = V3_CONFIG, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    symbols = dataio20.wave3_symbols()
    listings = dataio20.first_candle_dates(symbols, WAVE3_CACHE_DIR, min_rows=config.min_rows_required)
    if not listings:
        raise Wave20Error("V3: no candidate listings found")
    global_floor = min(listings.values())
    guard = pd.Timedelta(days=config.min_listing_gap_days)
    qualifying = sorted(
        ((symbol, first_date) for symbol, first_date in listings.items() if first_date > global_floor + guard),
        key=lambda item: item[1],
    )

    worst_cost = worst_case_cost(mapping, stress_multiplier)
    sleeve_equity = GAMBLE_CAPITAL
    trades: list[Trade] = []
    daily_pieces: list[pd.Series] = []
    busy_until: pd.Timestamp | None = None
    n_detected = len(qualifying)
    n_traded = 0
    n_skipped_overlap = 0
    symbols_traded: list[str] = []
    total_cost = 0.0

    for symbol, first_date in qualifying:
        if sleeve_equity <= 0.0:
            break
        if busy_until is not None and first_date <= busy_until:
            n_skipped_overlap += 1
            continue
        frame = dataio20.try_load_daily(symbol, WAVE3_CACHE_DIR)
        if frame is None or len(frame) < 2:
            continue
        window = frame.iloc[: config.hold_days + 1]
        if len(window) < 2:
            continue
        quote_volume = window["quote_volume"]
        cost_series = one_leg_cost_rate_series(quote_volume, mapping, stress_multiplier, window=costs_measured.ROLLING_WINDOW_DAYS)
        atr_series = expanding_atr(window).shift(1)

        equity_arr, window_trades, window_cost = simulate_breakout_reversal(
            index=window.index,
            open_arr=window["open"].to_numpy(dtype=float),
            close_arr=window["close"].to_numpy(dtype=float),
            atr_arr=atr_series.to_numpy(dtype=float),
            cost_arr=cost_series.to_numpy(dtype=float),
            worst_cost=worst_cost,
            atr_multiplier=config.atr_multiplier,
            armable_arr=None,
            forced_initial_direction=1.0,
            max_bars_in_position=config.hold_days,
            starting_equity=sleeve_equity,
            symbol=symbol,
        )
        daily_pieces.append(pd.Series(equity_arr, index=window.index))
        trades.extend(window_trades)
        total_cost += window_cost
        sleeve_equity = float(equity_arr[-1])
        busy_until = pd.Timestamp(window.index[-1])
        n_traded += 1
        symbols_traded.append(symbol)

    if not daily_pieces:
        full_calendar = pd.date_range(global_floor, global_floor, freq="1D", tz="UTC")
        equity_series = pd.Series([GAMBLE_CAPITAL], index=full_calendar)
    else:
        global_max = max(pd.Timestamp(dataio20.try_load_daily(s, WAVE3_CACHE_DIR).index[-1]) for s in symbols) if symbols else daily_pieces[-1].index[-1]
        full_calendar = pd.date_range(global_floor.normalize(), global_max.normalize(), freq="1D", tz="UTC")
        equity_series = pd.Series(index=full_calendar, dtype=float)
        equity_series.loc[full_calendar < daily_pieces[0].index[0]] = GAMBLE_CAPITAL
        for piece in daily_pieces:
            equity_series.loc[piece.index] = piece.values
        equity_series = equity_series.ffill().fillna(GAMBLE_CAPITAL)

    return GambleResult(
        candidate_id=config.candidate_id,
        equity=equity_series,
        trades=tuple(trades),
        symbols_used=tuple(symbols_traded),
        metadata={
            "n_trades": len(trades),
            "n_listings_detected": n_detected,
            "n_listings_traded": n_traded,
            "n_listings_skipped_overlap": n_skipped_overlap,
            "total_cost_usdt": total_cost,
            "final_equity_usdt": float(equity_series.iloc[-1]) if len(equity_series) else GAMBLE_CAPITAL,
            "sample_size_note": "UNDETERMINED_IF_LOW: G3/G4/G5 need n_trades large enough for a skew claim -- see configs20.G3_MIN_TRADES",
            "config": {
                "hold_days": config.hold_days,
                "atr_multiplier": config.atr_multiplier,
                "min_listing_gap_days": config.min_listing_gap_days,
                "min_rows_required": config.min_rows_required,
            },
        },
    )


# ---------------------------------------------------------------------------
# V4 -- 청산 캐스케이드 반등 (SYMMETRIC control group), BTC/ETH/SOL 1H, single $25 sleeve.
# ---------------------------------------------------------------------------


def run_v4(config: V4Config = V4_CONFIG, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    worst_cost = worst_case_cost(mapping, stress_multiplier)

    hourly_frames: dict[str, pd.DataFrame] = {}
    daily_cost: dict[str, pd.Series] = {}
    for symbol in config.symbols:
        hourly = dataio20.try_load_hourly(symbol, WAVE6_CACHE_DIR)
        if hourly is None:
            continue
        hourly_frames[symbol] = hourly
        daily = dataio20.resample_hourly_to_daily(hourly)
        daily_cost[symbol] = one_leg_cost_rate_series(daily["quote_volume"], mapping, stress_multiplier)
    if not hourly_frames:
        raise Wave20Error("V4: no 1H cache data available for any configured symbol")

    calendar = sorted(set().union(*(frame.index for frame in hourly_frames.values())))
    calendar = pd.DatetimeIndex(calendar)
    opens = pd.DataFrame({s: f["open"] for s, f in hourly_frames.items()}).reindex(calendar)
    closes = pd.DataFrame({s: f["close"] for s, f in hourly_frames.items()}).reindex(calendar)
    highs = pd.DataFrame({s: f["high"] for s, f in hourly_frames.items()}).reindex(calendar)
    lows = pd.DataFrame({s: f["low"] for s, f in hourly_frames.items()}).reindex(calendar)
    returns_1bar = closes / closes.shift(1) - 1.0
    cost_hourly = pd.DataFrame(
        {s: daily_cost[s].reindex(calendar, method="ffill") for s in hourly_frames}
    )

    n = len(calendar)
    sleeve_equity = GAMBLE_CAPITAL
    equity_out = np.empty(n, dtype=float)
    trades: list[Trade] = []
    dead = False
    total_cost = 0.0

    held_symbol: str | None = None
    entry_price = float("nan")
    entry_time: pd.Timestamp | None = None
    entry_equity = GAMBLE_CAPITAL
    entry_idx = -1
    pending_symbol: str | None = None

    priority = list(config.symbols)

    for i in range(n):
        ts = calendar[i]

        if pending_symbol is not None and not dead:
            symbol = pending_symbol
            pending_symbol = None
            fill_price = opens[symbol].iloc[i]
            if pd.notna(fill_price) and float(fill_price) > 0.0:
                rate = cost_hourly[symbol].iloc[i]
                rate = float(rate) if pd.notna(rate) else worst_cost
                equity_before = sleeve_equity
                sleeve_equity *= 1.0 - rate
                total_cost += equity_before - sleeve_equity
                held_symbol = symbol
                entry_price = float(fill_price)
                entry_time = ts
                entry_equity = sleeve_equity
                entry_idx = i

        if held_symbol is not None and not dead:
            high = highs[held_symbol].iloc[i]
            low = lows[held_symbol].iloc[i]
            close = closes[held_symbol].iloc[i]
            tp_price = entry_price * (1.0 + config.take_profit)
            sl_price = entry_price * (1.0 + config.stop_loss)
            bars_held = i - entry_idx
            exit_reason: str | None = None
            exit_price: float | None = None
            if pd.notna(low) and float(low) <= sl_price:
                exit_reason, exit_price = "stop_loss", sl_price
            elif pd.notna(high) and float(high) >= tp_price:
                exit_reason, exit_price = "take_profit", tp_price
            elif bars_held >= config.max_hold_bars:
                exit_reason = "max_hold"
                exit_price = float(close) if pd.notna(close) else entry_price

            if exit_reason is not None:
                rate = cost_hourly[held_symbol].iloc[i]
                rate = float(rate) if pd.notna(rate) else worst_cost
                gross_return = exit_price / entry_price - 1.0
                equity_before = sleeve_equity
                sleeve_equity = sleeve_equity * (1.0 + max(gross_return, -1.0))
                if sleeve_equity <= 0.0:
                    sleeve_equity = 0.0
                    exit_cost = 0.0
                else:
                    equity_before_cost = sleeve_equity
                    sleeve_equity *= 1.0 - rate
                    exit_cost = equity_before_cost - sleeve_equity
                    total_cost += exit_cost
                pnl = sleeve_equity - entry_equity
                trades.append(Trade(held_symbol, 1.0, entry_time, ts, entry_price, exit_price, entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else -1.0, exit_reason, exit_cost))
                del equity_before
                held_symbol = None
                if sleeve_equity <= 0.0:
                    dead = True

        equity_out[i] = sleeve_equity

        if held_symbol is None and not dead and sleeve_equity > 0.0 and i + 1 < n:
            for symbol in priority:
                if symbol not in returns_1bar.columns:
                    continue
                ret = returns_1bar[symbol].iloc[i]
                if pd.notna(ret) and float(ret) <= config.drop_threshold:
                    pending_symbol = symbol
                    break

    if held_symbol is not None and not dead:
        close = closes[held_symbol].iloc[-1]
        exit_price = float(close) if pd.notna(close) else entry_price
        rate = cost_hourly[held_symbol].iloc[-1]
        rate = float(rate) if pd.notna(rate) else worst_cost
        gross_return = exit_price / entry_price - 1.0
        sleeve_equity = sleeve_equity * (1.0 + max(gross_return, -1.0))
        equity_before_cost = sleeve_equity
        sleeve_equity *= 1.0 - rate
        exit_cost = equity_before_cost - sleeve_equity
        total_cost += exit_cost
        pnl = sleeve_equity - entry_equity
        trades.append(Trade(held_symbol, 1.0, entry_time, calendar[-1], entry_price, exit_price, entry_equity, pnl, _clip_fraction(pnl / entry_equity) if entry_equity > 0 else -1.0, "end_of_data", exit_cost))
        equity_out[-1] = sleeve_equity

    equity_hourly = pd.Series(equity_out, index=calendar)
    equity_daily = equity_hourly.resample("1D").last().ffill()
    trade_symbol_counts = {symbol: sum(1 for t in trades if t.symbol == symbol) for symbol in config.symbols}
    return GambleResult(
        candidate_id=config.candidate_id,
        equity=equity_daily,
        trades=tuple(trades),
        symbols_used=config.symbols,
        metadata={
            "n_trades": len(trades),
            "trade_symbol_counts": trade_symbol_counts,
            "total_cost_usdt": total_cost,
            "final_equity_usdt": float(equity_daily.iloc[-1]) if len(equity_daily) else GAMBLE_CAPITAL,
            "config": {
                "symbols": list(config.symbols),
                "drop_threshold": config.drop_threshold,
                "take_profit": config.take_profit,
                "stop_loss": config.stop_loss,
                "max_hold_bars": config.max_hold_bars,
            },
        },
    )


# ---------------------------------------------------------------------------
# V5 -- 복권 바스켓 (5-name equal-weight, point-in-time cheap+volatile selection, 30d hold).
# ---------------------------------------------------------------------------


def run_v5(config: V5Config = V5_CONFIG, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    worst_cost = worst_case_cost(mapping, stress_multiplier)
    symbols = tuple(s for s in dataio20.wave3_symbols() if s not in config.excluded_symbols)

    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    quote_volumes: dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = dataio20.try_load_daily(symbol, WAVE3_CACHE_DIR)
        if frame is None:
            continue
        closes[symbol] = frame["close"]
        opens[symbol] = frame["open"]
        quote_volumes[symbol] = frame["quote_volume"]

    close_frame = pd.DataFrame(closes).sort_index()
    open_frame = pd.DataFrame(opens).reindex(close_frame.index)
    quote_volume_frame = pd.DataFrame(quote_volumes).reindex(close_frame.index)
    close_ffilled = close_frame.ffill()
    vol_frame = close_frame.pct_change().rolling(config.vol_lookback_days, min_periods=config.vol_lookback_days).std()
    history_count = close_frame.notna().cumsum()
    eligible_mask = close_frame.notna() & (history_count >= config.min_history_days) & vol_frame.notna()
    cost_rate_frame = one_leg_cost_rate_frame(quote_volume_frame, mapping, stress_multiplier)

    calendar = close_frame.index
    n = len(calendar)
    equity_values = np.full(n, np.nan, dtype=float)
    trades: list[Trade] = []
    sleeve_equity = GAMBLE_CAPITAL
    n_attempted = 0
    n_baskets = 0
    baskets_seen: list[list[str]] = []

    idx = min(config.min_history_days, max(n - 1, 0))
    while idx < n - 1:
        ts = calendar[idx]
        entry_idx = idx + 1
        entry_ts = calendar[entry_idx]
        exit_idx = min(idx + config.rebalance_days, n - 1)
        n_attempted += 1

        row_mask = eligible_mask.loc[ts]
        candidates = [s for s in row_mask.index[row_mask] if pd.notna(open_frame[s].iloc[entry_idx])]
        basket: list[str] = []
        if len(candidates) >= config.basket_size:
            prices = close_frame.loc[ts, candidates]
            cutoff = prices.quantile(config.cheap_price_percentile)
            cheap = [s for s in candidates if prices[s] <= cutoff]
            pool = cheap if len(cheap) >= config.basket_size else candidates
            vols = vol_frame.loc[ts, pool].dropna().sort_values(ascending=False)
            basket = list(vols.index[: config.basket_size])

        if np.isnan(equity_values[idx]):
            equity_values[idx] = sleeve_equity

        if len(basket) == 0 or sleeve_equity <= 0.0:
            for j in range(idx, exit_idx + 1):
                if np.isnan(equity_values[j]):
                    equity_values[j] = sleeve_equity
            idx = exit_idx
            continue

        n_baskets += 1
        baskets_seen.append(basket)
        equity_before = sleeve_equity
        per_leg_alloc = equity_before / len(basket)
        leg_entry_price: dict[str, float] = {}
        leg_start_equity: dict[str, float] = {}
        for symbol in basket:
            fill_price = float(open_frame[symbol].iloc[entry_idx])
            rate = cost_rate_frame[symbol].iloc[entry_idx]
            rate = float(rate) if pd.notna(rate) else worst_cost
            leg_start_equity[symbol] = per_leg_alloc * (1.0 - rate)
            leg_entry_price[symbol] = fill_price

        for j in range(entry_idx, exit_idx + 1):
            total = 0.0
            for symbol in basket:
                price_now = close_ffilled[symbol].iloc[j]
                if pd.isna(price_now):
                    price_now = leg_entry_price[symbol]
                leg_return = float(price_now) / leg_entry_price[symbol] - 1.0
                total += leg_start_equity[symbol] * max(1.0 + leg_return, 0.0)
            equity_values[j] = total

        exit_total = 0.0
        for symbol in basket:
            price_now = close_ffilled[symbol].iloc[exit_idx]
            if pd.isna(price_now):
                price_now = leg_entry_price[symbol]
            leg_return = float(price_now) / leg_entry_price[symbol] - 1.0
            gross_leg_equity = leg_start_equity[symbol] * max(1.0 + leg_return, 0.0)
            rate = cost_rate_frame[symbol].iloc[exit_idx]
            rate = float(rate) if pd.notna(rate) else worst_cost
            net_leg_equity = gross_leg_equity * (1.0 - rate) if gross_leg_equity > 0.0 else 0.0
            pnl = net_leg_equity - per_leg_alloc
            trades.append(
                Trade(
                    symbol, 1.0, entry_ts, calendar[exit_idx], leg_entry_price[symbol], float(price_now), per_leg_alloc, pnl,
                    _clip_fraction(pnl / per_leg_alloc) if per_leg_alloc > 0 else -1.0, "basket_unwind",
                    (per_leg_alloc - leg_start_equity[symbol]) + (gross_leg_equity - net_leg_equity),
                )
            )
            exit_total += net_leg_equity

        sleeve_equity = exit_total
        equity_values[exit_idx] = sleeve_equity
        idx = exit_idx

    if n > 0 and np.isnan(equity_values[0]):
        equity_values[0] = GAMBLE_CAPITAL
    equity_series = pd.Series(equity_values, index=calendar).ffill().fillna(GAMBLE_CAPITAL)

    total_cost = float(sum(t.cost_usdt for t in trades))
    used_symbols = sorted({symbol for basket in baskets_seen for symbol in basket})
    return GambleResult(
        candidate_id=config.candidate_id,
        equity=equity_series,
        trades=tuple(trades),
        symbols_used=tuple(used_symbols),
        metadata={
            "n_trades": len(trades),
            "n_rebalances_attempted": n_attempted,
            "n_rebalances_with_basket": n_baskets,
            "n_unique_symbols_selected": len(used_symbols),
            "total_cost_usdt": total_cost,
            "final_equity_usdt": float(equity_series.iloc[-1]) if len(equity_series) else GAMBLE_CAPITAL,
            "config": {
                "rebalance_days": config.rebalance_days,
                "basket_size": config.basket_size,
                "vol_lookback_days": config.vol_lookback_days,
                "min_history_days": config.min_history_days,
                "cheap_price_percentile": config.cheap_price_percentile,
            },
        },
    )


# ---------------------------------------------------------------------------
# Stable leg (I5, read verbatim -- not re-simulated) + portfolio combination.
# ---------------------------------------------------------------------------


def _series_from_records(records: list[dict]) -> pd.Series:
    if not records:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records])
    values = [float(item["value"]) for item in records]
    return pd.Series(values, index=index, dtype=float).sort_index()


def load_stable_leg(stable_capital: float = STABLE_CAPITAL, path: Path = I5_RESULTS_PATH) -> tuple[pd.Series, dict]:
    """research/wave18_idle/results/I5.json, read verbatim and rescaled from its own $90
    active-capital basis to `stable_capital` ($75) -- I5 itself is NOT re-simulated here (the
    task brief: "안정 레그 = I5 ... 결과 파일을 읽어" -- read the result, engine18.py is cited
    only as the source of truth for how that file was produced)."""
    if not path.exists():
        raise Wave20Error(f"missing stable-leg source: {path} -- run research/wave18_idle's own pipeline first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    equity = _series_from_records(payload["equity"])
    active_capital = float(payload["capital_contract"]["active_capital_usdt"])
    rescaled = stable_capital * equity / active_capital
    info = {
        "source": str(path),
        "source_candidate_id": payload.get("candidate_id"),
        "source_active_capital_usdt": active_capital,
        "source_full_period_annualized": payload.get("full_period_annualized"),
        "rescaled_to_usdt": stable_capital,
    }
    return rescaled, info


def pad_to_calendar(equity: pd.Series, calendar: pd.DatetimeIndex, initial_value: float) -> pd.Series:
    """Extends a gambling sleeve's own (possibly shorter/sparser) equity series onto the
    full stable-leg calendar: flat at `initial_value` (uninvested cash) before the sleeve's
    own data starts, forward-filled across any interior gaps, flat at the last known value
    past the sleeve's own data end. Never fabricates a return the sleeve did not earn."""
    reindexed = equity.reindex(calendar)
    first_valid = reindexed.first_valid_index()
    if first_valid is not None:
        reindexed.loc[reindexed.index <= first_valid] = reindexed.loc[reindexed.index <= first_valid].fillna(initial_value)
    return reindexed.ffill().fillna(initial_value)


def combine_portfolio(stable_equity: pd.Series, gamble_equity: pd.Series, gamble_initial: float = GAMBLE_CAPITAL) -> pd.DataFrame:
    """Full $100 system: dollar-additive union of the two legs' own equity paths. Returns a
    frame with stable/gamble/total columns on the STABLE leg's own calendar (I5's, the
    longest/most complete series in this wave) so every candidate's combined result shares
    one comparable index."""
    calendar = stable_equity.index
    gamble_on_calendar = pad_to_calendar(gamble_equity, calendar, gamble_initial)
    total = stable_equity + gamble_on_calendar
    return pd.DataFrame({"stable": stable_equity, "gamble": gamble_on_calendar, "total": total})


RUNNERS: Final[dict[str, Callable[[], GambleResult]]] = {
    "V1": run_v1,
    "V2": run_v2,
    "V3": run_v3,
    "V4": run_v4,
    "V5": run_v5,
}


def run_candidate(candidate_id: str, mapping: MeasuredCostMapping | None = None, stress_multiplier: float = 1.0) -> GambleResult:
    mapping = mapping if mapping is not None else costs_measured.fit_mapping()
    if candidate_id == "V1":
        return run_v1(V1_CONFIG, mapping, stress_multiplier)
    if candidate_id == "V2":
        return run_v2(V2_CONFIG, mapping, stress_multiplier)
    if candidate_id == "V3":
        return run_v3(V3_CONFIG, mapping, stress_multiplier)
    if candidate_id == "V4":
        return run_v4(V4_CONFIG, mapping, stress_multiplier)
    if candidate_id == "V5":
        return run_v5(V5_CONFIG, mapping, stress_multiplier)
    raise Wave20Error(f"unknown wave20 candidate: {candidate_id}")


__all__ = [
    "GambleResult",
    "RUNNERS",
    "Trade",
    "Wave20Error",
    "atr",
    "combine_portfolio",
    "expanding_atr",
    "load_stable_leg",
    "one_leg_cost_rate_frame",
    "one_leg_cost_rate_series",
    "pad_to_calendar",
    "realized_vol",
    "run_candidate",
    "run_v1",
    "run_v2",
    "run_v3",
    "run_v4",
    "run_v5",
    "simulate_breakout_reversal",
    "trailing_percentile_rank",
    "true_range",
    "worst_case_cost",
]
