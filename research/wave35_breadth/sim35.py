# Wave-35: V1's breakout/reversal state machine, FROZEN, executed on DAILY bars, run per symbol
# over a point-in-time-eligible universe. Only two things change vs wave20's V1: the universe and
# the capital allocation. No parameter of V1 is touched here (see SPEC.md section 1).
#
# WHY A PER-SYMBOL UNIT SIMULATION IS SOUND
# -----------------------------------------
# Under the fixed-notional model every entry's base is a constant $100/N, so a symbol's own trade
# stream does NOT depend on the account's equity path or on N (the only equity coupling would be
# compounding, which fixed-notional removes by construction, plus account death, which is applied
# afterwards on the combined path). That lets each symbol be simulated ONCE per leverage with a
# unit base of 1.0; a portfolio of N symbols is then the union of those trade streams scaled by
# $100/N. Nothing here approximates the SIGNAL: the state machine is causal per symbol and never
# sees another symbol or the account.
#
# The calibration path (compound=True) instead uses entry base = current sleeve equity, because
# that is what wave30_riskcap's "fixed" L=1 8.49x number is, and the daily-vs-hourly gap has to be
# measured against comparable accounting. One disclosed difference vs sim30: this module charges
# the entry cost ADDITIVELY into the trade's mark (base*(1 - rate*L + L*ret)) instead of
# multiplicatively (base*(1-rate*L)*(1+L*ret)), because the fixed-notional arm needs "dollars per
# entry" to include both legs' cost. The relative gap per trade is rate*L^2*ret ~ 1e-5.

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wave13_liquidity import costs_measured
from research.wave20_convex import dataio20
from research.wave20_convex.configs20 import V1_CONFIG, WAVE3_CACHE_DIR
from research.wave20_convex.engine20 import (
    atr,
    one_leg_cost_rate_series,
    realized_vol,
    trailing_percentile_rank,
    worst_case_cost,
)

MAINT_MARGIN = 0.005  # Bitget USDT-M maintenance margin, same constant sim30 uses

# --- SPEC.md section 3, frozen point-in-time eligibility rules ---
MIN_OWN_BARS = 400             # R1: indicator warm-up (20 + 365 vol-percentile, + ATR14/cost30 slack)
MIN_TRAILING_DOLLAR_VOL = 1e6  # R2: trailing-30d mean quote volume floor, $/day
RESELECT_DAYS = 30             # R3: universe re-selection cadence


@dataclass(frozen=True, slots=True)
class T35:
    symbol: str
    entry_i: int              # index into the symbol's own daily calendar
    exit_i: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: float
    roi: float                # pnl per unit base, net of BOTH costs, floored at -1.0
    reason: str
    mark_path: np.ndarray     # marked pnl per unit base for days entry_i..exit_i (last == roi)


@dataclass(slots=True)
class SymbolRun:
    symbol: str
    index: pd.DatetimeIndex
    trades: tuple[T35, ...]
    eligible: np.ndarray             # bool per day: R1 & R2 both satisfied (point-in-time)
    trailing_dollar_vol: np.ndarray  # R2 statistic, shift(1)-ed
    final_equity: float              # only meaningful when compound=True


def v1_daily_inputs(daily: pd.DataFrame, mapping) -> dict:
    """Exactly the series run_v1 builds. run_v1 broadcasts these same DAILY series onto an hourly
    execution grid; here the daily grid IS the execution grid. Same lags, same cost mapping, same
    regime filter, same ATR window/multiplier -- V1 is frozen."""
    cfg = V1_CONFIG
    vol_rank = trailing_percentile_rank(
        realized_vol(daily["close"], cfg.vol_window_days), cfg.vol_percentile_lookback_days
    )
    armable = (vol_rank < cfg.vol_percentile_threshold).shift(1).fillna(False)
    trail_vol = (daily["quote_volume"]
                 .rolling(costs_measured.ROLLING_WINDOW_DAYS, min_periods=costs_measured.ROLLING_WINDOW_DAYS)
                 .mean().shift(1))
    return {
        "index": daily.index,
        "open": daily["open"].to_numpy(dtype=float),
        "high": daily["high"].to_numpy(dtype=float),
        "low": daily["low"].to_numpy(dtype=float),
        "close": daily["close"].to_numpy(dtype=float),
        "atr": atr(daily, cfg.atr_window_days).shift(1).to_numpy(dtype=float),
        "cost": one_leg_cost_rate_series(daily["quote_volume"], mapping, 1.0).to_numpy(dtype=float),
        "armable": armable.to_numpy(dtype=bool),
        "trail_vol": trail_vol.to_numpy(dtype=float),
        "worst_cost": worst_case_cost(mapping, 1.0),
        "mult": cfg.atr_multiplier,
    }


def simulate_symbol(symbol: str, inp: dict, leverage: float = 1.0,
                    compound: bool = False, starting_equity: float = 1.0) -> SymbolRun:
    index = inp["index"]
    op, hi, lo, cl = inp["open"], inp["high"], inp["low"], inp["close"]
    atr_a, cost_a, arm_a = inp["atr"], inp["cost"], inp["armable"]
    worst, mult = inp["worst_cost"], inp["mult"]
    n = len(index)
    liq_dist = max(0.0, 1.0 / leverage - MAINT_MARGIN)

    trades: list[T35] = []
    sleeve = starting_equity
    direction = 0.0
    anchor = entry_price = extreme = float("nan")
    entry_base = starting_equity
    entry_cost_frac = 0.0
    entry_i = -1
    entry_date: pd.Timestamp | None = None
    mark = 0.0
    marks: list[float] = []
    pending: tuple[str, float] | None = None
    dead = False

    def value_frac(price: float) -> float:
        return max(0.0, 1.0 + leverage * direction * (price / entry_price - 1.0))

    def emit(i: int, roi: float, reason: str) -> None:
        nonlocal direction, sleeve, mark, marks
        marks.append(roi)
        trades.append(T35(symbol, entry_i, i, entry_date, index[i], direction,
                          roi, reason, np.asarray(marks, dtype=float)))
        if compound:
            sleeve = max(0.0, entry_base * (1.0 + roi))
            if sleeve <= 0.0:
                dead_flag[0] = True
        direction, mark, marks = 0.0, 0.0, []

    dead_flag = [False]

    for i in range(n):
        dead = dead or dead_flag[0]
        # --- A: fill the action decided at bar i-1, at bar i's open (engine20's t->t+1 rule). ---
        if pending is not None and not dead:
            _action, new_dir = pending
            pending = None
            fill = float(op[i])
            if direction != 0.0:
                adverse = direction * (fill / entry_price - 1.0)
                if adverse <= -liq_dist:
                    emit(i, -1.0, "liquidated")
                else:
                    rate = cost_a[i] if not np.isnan(cost_a[i]) else worst
                    vf = value_frac(fill)
                    roi = max(-1.0, vf - 1.0 - entry_cost_frac - rate * leverage * (fill / entry_price))
                    emit(i, roi, "reversal")
            dead = dead or dead_flag[0]
            if not dead and new_dir != 0.0 and not np.isnan(fill) and fill > 0.0:
                rate = cost_a[i] if not np.isnan(cost_a[i]) else worst
                entry_base = sleeve if compound else starting_equity
                if entry_base <= 0.0:
                    dead = True
                else:
                    direction, entry_price, extreme = new_dir, fill, fill
                    entry_date, entry_i = index[i], i
                    entry_cost_frac = rate * leverage
                    mark = -entry_cost_frac
                    marks = [mark]
                    anchor = float("nan")

        # --- B: intrabar liquidation (no stop -- SPEC.md 5), then mark to this bar's close. ---
        if direction != 0.0 and not dead:
            adverse_px = float(lo[i]) if direction > 0.0 else float(hi[i])
            adverse = direction * (adverse_px / entry_price - 1.0)
            if adverse <= -liq_dist:
                if index[i] == entry_date:
                    marks = marks[:-1]          # replace the entry-day mark with the wipe
                emit(i, -1.0, "liquidated")
            else:
                mark = value_frac(float(cl[i])) - 1.0 - entry_cost_frac
                if index[i] == entry_date:
                    marks[-1] = mark
                else:
                    marks.append(mark)
                extreme = max(extreme, cl[i]) if direction > 0.0 else min(extreme, cl[i])

        # --- C: decide at bar i's close what to do at bar i+1's open. ---
        if not dead and i + 1 < n:
            a = atr_a[i]
            if direction == 0.0:
                if bool(arm_a[i]) and np.isnan(anchor):
                    anchor = cl[i]
                if not np.isnan(anchor) and not np.isnan(a) and a > 0.0:
                    if cl[i] - anchor >= mult * a:
                        pending = ("open", 1.0)
                    elif anchor - cl[i] >= mult * a:
                        pending = ("open", -1.0)
            elif not np.isnan(a) and a > 0.0:
                if direction > 0.0 and (extreme - cl[i]) >= mult * a:
                    pending = ("reverse", -1.0)
                elif direction < 0.0 and (cl[i] - extreme) >= mult * a:
                    pending = ("reverse", 1.0)

    if direction != 0.0 and not dead:
        i = n - 1
        rate = cost_a[i] if not np.isnan(cost_a[i]) else worst
        price = float(cl[i])
        roi = max(-1.0, value_frac(price) - 1.0 - entry_cost_frac - rate * leverage * (price / entry_price))
        marks = marks[:-1] if marks else marks
        emit(i, roi, "end_of_data")

    trail = inp["trail_vol"]
    own_bars = np.arange(1, n + 1)
    eligible = (own_bars >= MIN_OWN_BARS) & (np.nan_to_num(trail, nan=0.0) >= MIN_TRAILING_DOLLAR_VOL)
    return SymbolRun(symbol, index, tuple(trades), eligible, trail, float(sleeve))


def load_universe(min_bars: int = MIN_OWN_BARS) -> dict[str, pd.DataFrame]:
    """Every research/wave3/cache Binance fapi daily symbol with enough rows to EVER satisfy R1.
    Dropping a symbol that can never be eligible is not survivorship selection -- it removes rows
    that R1 would reject on every single date anyway."""
    out: dict[str, pd.DataFrame] = {}
    for symbol in dataio20.wave3_symbols():
        frame = dataio20.try_load_daily(symbol, WAVE3_CACHE_DIR)
        if frame is None or len(frame) < min_bars + 5:
            continue
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if frame["close"].isna().all():
            continue
        out[symbol] = frame
    return out
