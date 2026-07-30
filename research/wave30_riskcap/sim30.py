# Wave-30: V1's breakout/reversal state machine with the two axes wave-29 never tested --
# LEVERAGE and a HARD STOP. wave-29 raised leverage on a strategy that has no stop at all, so the
# liquidation band walked straight into V1's own MAE distribution and every high-leverage cell died.
# A stop placed INSIDE the liquidation band changes the arithmetic: the worst single trade becomes
# L*s (a number we choose) instead of -100% (a number the market chooses).
#
# TWO POSITION MODELS, because engine20's choice and reality differ and it matters here:
#
#   "rebalanced"  -- equity *= (1 + L*dir*bar_return) every bar. This is what engine20 does.
#                    Exposure is reset to L x CURRENT equity each bar, so a short can never be
#                    liquidated by a slow grind (the position shrinks as it loses). Used ONLY to
#                    prove this harness reproduces run_v1 exactly at L=1 (Gate 0).
#   "fixed"       -- equity = entry_equity * (1 + L*dir*(P/entry - 1)). Coins held are fixed at
#                    entry, which is what an isolated-margin perp actually is, and it is the only
#                    model under which the liquidation price sits exactly 1/L - maintenance from
#                    entry. Every swept cell uses this, because the whole question is wipe risk.
#
# The two coincide exactly for LONGS at any L (prod(1+r) = P_exit/P_entry) and for everything at
# L=1 long-only; they diverge for shorts, which is why the L=1 baselines differ and why the sweep
# reports its own L=1 "fixed" row as the like-for-like baseline.

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
from research.wave20_convex.configs20 import GAMBLE_CAPITAL, V1_CONFIG, WAVE6_CACHE_DIR
from research.wave20_convex.engine20 import (
    atr,
    one_leg_cost_rate_series,
    realized_vol,
    trailing_percentile_rank,
    worst_case_cost,
)

MAINT_MARGIN = 0.005      # Bitget USDT-M maintenance margin
TAKER_PREMIUM = 0.0004    # stop exits are taker orders (0.06%) vs the maker 0.02% baseline


@dataclass(frozen=True, slots=True)
class T30:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: float
    entry_equity: float
    pnl: float
    roi: float            # pnl / entry_equity -- "한 번 들어갈 때마다의 ROI", the user's own metric
    reason: str


@dataclass(frozen=True, slots=True)
class R30:
    leverage: float
    stop: float | None
    model: str
    equity: pd.Series     # daily, sleeve dollars
    trades: tuple[T30, ...]
    final: float


def v1_inputs() -> dict:
    """Exactly the series run_v1 builds -- same lags, same cost mapping, same regime filter."""
    cfg = V1_CONFIG
    mapping = costs_measured.fit_mapping()
    hourly = dataio20.load_hourly(cfg.symbol, WAVE6_CACHE_DIR)
    daily = dataio20.resample_hourly_to_daily(hourly)
    vol_rank = trailing_percentile_rank(
        realized_vol(daily["close"], cfg.vol_window_days), cfg.vol_percentile_lookback_days
    )
    armable_daily = (vol_rank < cfg.vol_percentile_threshold).shift(1).fillna(False)
    cost_daily = one_leg_cost_rate_series(daily["quote_volume"], mapping, 1.0)
    return {
        "index": hourly.index,
        "open": hourly["open"].to_numpy(dtype=float),
        "high": hourly["high"].to_numpy(dtype=float),
        "low": hourly["low"].to_numpy(dtype=float),
        "close": hourly["close"].to_numpy(dtype=float),
        "atr": atr(daily, cfg.atr_window_days).shift(1).reindex(hourly.index, method="ffill").to_numpy(dtype=float),
        "cost": cost_daily.reindex(hourly.index, method="ffill").to_numpy(dtype=float),
        "armable": armable_daily.reindex(hourly.index, method="ffill").fillna(False).astype(bool).to_numpy(dtype=bool),
        "worst_cost": worst_case_cost(mapping, 1.0),
        "atr_multiplier": cfg.atr_multiplier,
    }


def simulate(inp: dict, leverage: float = 1.0, stop: float | None = None, model: str = "fixed",
             start_at: pd.Timestamp | None = None, end_at: pd.Timestamp | None = None,
             starting_equity: float = GAMBLE_CAPITAL, max_hold_bars: int | None = None,
             atr_multiplier: float | None = None) -> R30:
    idx = inp["index"]
    sel = np.ones(len(idx), dtype=bool)
    if start_at is not None:
        sel &= idx > start_at
    if end_at is not None:
        sel &= idx <= end_at
    keep = np.flatnonzero(sel)
    index = idx[keep]
    op, hi, lo, cl = inp["open"][keep], inp["high"][keep], inp["low"][keep], inp["close"][keep]
    atr_a, cost_a, arm_a = inp["atr"][keep], inp["cost"][keep], inp["armable"][keep]
    worst = inp["worst_cost"]
    mult = inp["atr_multiplier"] if atr_multiplier is None else atr_multiplier
    n = len(index)
    linear = model == "fixed"
    entry_idx = -1

    liq_dist = max(0.0, 1.0 / leverage - MAINT_MARGIN)
    # A stop only helps if it fires BEFORE liquidation; otherwise the cell is just wave-29 again.
    stop_dist = None if stop is None else min(stop, liq_dist)

    equity_out = np.empty(n, dtype=float)
    trades: list[T30] = []
    sleeve = starting_equity
    direction = 0.0
    anchor = entry_price = extreme = float("nan")
    entry_equity = starting_equity
    entry_time: pd.Timestamp | None = None
    pending: tuple[str, float] | None = None
    dead = False

    def value_at(price: float, ref: float, sleeve_at_ref: float) -> float:
        """Marked equity at `price`. 'fixed' measures from entry; 'rebalanced' from the last mark."""
        if linear:
            return entry_equity * (1.0 + leverage * direction * (price / entry_price - 1.0))
        return sleeve_at_ref * (1.0 + leverage * direction * (price / ref - 1.0))

    def close_at(i: int, price: float, ref: float, sleeve_at_ref: float, reason: str, taker: bool) -> None:
        nonlocal sleeve, direction, entry_time
        rate = (cost_a[i] if not np.isnan(cost_a[i]) else worst) + (TAKER_PREMIUM if taker else 0.0)
        gross = max(0.0, value_at(price, ref, sleeve_at_ref))
        # Exit cost is charged on the notional actually held at exit. Under 'fixed' that is the
        # coins bought at entry marked at the exit price; under 'rebalanced' it is L x equity,
        # which at L=1 reduces to engine20's own `sleeve *= (1 - rate)`.
        notional = leverage * entry_equity * (price / entry_price) if linear else leverage * gross
        sleeve = max(0.0, gross - rate * notional)
        trades.append(T30(entry_time, index[i], direction, entry_equity,
                          sleeve - entry_equity, (sleeve - entry_equity) / entry_equity, reason))
        direction = 0.0

    def wipe(i: int) -> None:
        nonlocal sleeve, direction, dead
        trades.append(T30(entry_time, index[i], direction, entry_equity, -entry_equity, -1.0, "liquidated"))
        sleeve, direction, dead = 0.0, 0.0, True

    for i in range(n):
        # --- A: fill the action decided at bar i-1, at bar i's open (engine20's t->t+1 rule). ---
        if pending is not None and not dead:
            action, new_dir = pending
            pending = None
            fill = float(op[i])
            if direction != 0.0:
                ref = float(cl[i - 1]) if i > 0 else entry_price
                # A gap straight through the stop/liquidation fills at the open, not at the level.
                adverse = direction * (fill / entry_price - 1.0)
                ref_adverse = direction * (fill / ref - 1.0)
                if stop_dist is not None and adverse <= -stop_dist:
                    close_at(i, fill, ref, sleeve, "stop_gap", taker=True)
                elif (adverse if linear else ref_adverse) <= -liq_dist:
                    wipe(i)
                else:
                    close_at(i, fill, ref, sleeve, action if action == "reverse" else "exit", taker=False)
            if not dead and new_dir != 0.0:
                rate = cost_a[i] if not np.isnan(cost_a[i]) else worst
                sleeve = sleeve * (1.0 - rate * leverage)   # entry cost on notional = L x sleeve
                if sleeve <= 0.0:
                    sleeve, dead = 0.0, True
                else:
                    direction, entry_price, extreme = new_dir, fill, fill
                    entry_time, entry_equity, anchor = index[i], sleeve, float("nan")
                    entry_idx = i

        # --- B: intrabar stop / liquidation, then mark to this bar's close. ---
        if direction != 0.0 and not dead:
            ref = entry_price if entry_time == index[i] else float(cl[i - 1])
            sleeve_at_ref = sleeve
            adverse_px = float(lo[i]) if direction > 0.0 else float(hi[i])
            adverse = direction * (adverse_px / entry_price - 1.0)
            ref_adverse = direction * (adverse_px / ref - 1.0)
            if stop_dist is not None and adverse <= -stop_dist:
                close_at(i, entry_price * (1.0 - direction * stop_dist), ref, sleeve_at_ref, "stopped", taker=True)
            elif (adverse if linear else ref_adverse) <= -liq_dist:
                wipe(i)
            else:
                sleeve = value_at(float(cl[i]), ref, sleeve_at_ref)
                extreme = max(extreme, cl[i]) if direction > 0.0 else min(extreme, cl[i])
                if sleeve <= 0.0:
                    wipe(i)

        # --- max_hold_bars: force the position closed at this bar's close (engine20's V3 rule).
        # Step B already marked to cl[i], so this is exit-cost-only, like the end-of-data close. ---
        if direction != 0.0 and not dead and max_hold_bars is not None and (i - entry_idx) >= max_hold_bars:
            close_at(i, float(cl[i]), float(cl[i]), sleeve, "max_hold", taker=False)

        equity_out[i] = sleeve

        # --- C: decide at bar i's close what to do at bar i+1's open. ---
        if not dead and sleeve > 0.0 and i + 1 < n:
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
        # Step B already marked the sleeve to the last bar's close; passing that same price as the
        # reference makes this an exit-cost-only close instead of re-applying the final bar's move.
        close_at(n - 1, float(cl[-1]), float(cl[-1]), sleeve, "end_of_data", taker=False)
        equity_out[-1] = sleeve

    eq = pd.Series(equity_out, index=index).resample("1D").last().ffill()
    return R30(leverage, stop, model, eq, tuple(trades), float(eq.iloc[-1]) if len(eq) else starting_equity)


def fidelity_check() -> tuple[bool, str]:
    """Gate 0: L=1 / no stop / rebalanced must reproduce engine20.run_v1 exactly."""
    from research.wave20_convex.engine20 import run_v1
    ref = run_v1()
    mine = simulate(v1_inputs(), leverage=1.0, stop=None, model="rebalanced")
    target = float(ref.equity.iloc[-1])
    diff = abs(mine.final - target)
    ok = len(mine.trades) == len(ref.trades) and diff < 1e-6
    return ok, (f"trades {len(mine.trades)} vs {len(ref.trades)} | "
                f"final ${mine.final:.6f} vs ${target:.6f} | diff {diff:.2e}")


if __name__ == "__main__":
    ok, msg = fidelity_check()
    print(("PASS " if ok else "FAIL ") + msg)
    raise SystemExit(0 if ok else 1)
