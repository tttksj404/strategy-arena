# Wave-25 indicator primitives (SPEC.md "미탐색" table): MACD, ADX/DMI, Supertrend, Keltner
# Channel, Stochastic, and multi-timeframe (MTF) alignment helpers. Every formula below is
# implemented by hand from its own textbook definition (no ta-lib/pandas_ta or any other TA
# package) -- only generic pandas/numpy rolling/ewm primitives are used, the same convention
# research/wave20_convex/engine20.py's own true_range/atr/trailing_percentile_rank already
# follow. This module is pure math: it has no knowledge of positions, capital, or costs (that
# lives in engine25.py) and no knowledge of P1-P5 gate thresholds (gates25.py).
#
# ---------------------------------------------------------------------------------------
# Point-in-time discipline
# ---------------------------------------------------------------------------------------
# Every function here is causal (uses only data up to and including bar t to produce its
# value AT bar t) -- none of them peek forward. The one place lookahead could sneak in is
# multi-timeframe alignment (a DAILY value must not leak into an HOURLY bar that occurred
# before that day closed); align_daily_to_intraday enforces this explicitly with a shift(1)
# before reindexing, identical to engine20.run_v1's own
# `armable_daily.shift(1).reindex(hourly.index, method="ffill")` convention. Everything that
# consumes this module's output is still responsible for its OWN entry-vs-fill timing
# (decide using bar t, fill at bar t+1's open) -- that discipline lives in engine25.py, not
# here.

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

REQUIRED_OHLC_COLUMNS = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Shared primitives: true range / ATR / EMA. Reimplemented locally (matches
# engine20.true_range/atr's own formulas exactly -- max of the three standard components;
# ATR = simple rolling mean of true range) so this module has no import-time dependency on
# any other wave's engine, per the task brief's "전부 직접 구현".
# ---------------------------------------------------------------------------


def true_range(ohlc: pd.DataFrame) -> pd.Series:
    prior_close = ohlc["close"].shift(1)
    ranges = pd.concat(
        [ohlc["high"] - ohlc["low"], (ohlc["high"] - prior_close).abs(), (ohlc["low"] - prior_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Simple rolling mean of true_range -- NaN for the first `window`-1 bars (insufficient
    history), matching engine20.atr's own min_periods=window convention exactly."""
    return true_range(ohlc).rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Standard exponential moving average: EMA[0] = series[0], EMA[t] = alpha*series[t] +
    (1-alpha)*EMA[t-1] for t>=1, alpha = 2/(span+1). pandas' `.ewm(span=span,
    adjust=False).mean()` implements exactly this recursion (not the "adjusted"/weighted
    variant) -- tests/test_wave25.py pins this against a hand-computed 4-point series so the
    exact recursive formula being used is auditable, not just asserted here."""
    return series.ewm(span=span, adjust=False).mean()


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing method (the classic running-moving-average used by the original
    ADX/DMI/RSI papers, a.k.a. RMA): seeded with a simple average of the first `window`
    observations (first valid output at index `window`-1, matching every other
    rolling-window primitive in this repo), then recursively smoothed[t] = smoothed[t-1] -
    smoothed[t-1]/window + raw[t]/window (equivalently smoothed[t-1]*(window-1)/window +
    raw[t]/window -- Wilder's own ATR definition: "ATR_today = (ATR_yesterday*13 +
    TR_today)/14" for a 14-period ATR). This differs from a plain `.ewm(alpha=1/window)`
    only in its seed. The `raw[t]/window` term (not bare `raw[t]`) matters: omitting it turns
    this into an unbounded accumulator rather than a running AVERAGE -- caught by
    tests/test_wave25.py's ADX-stays-within-[0,100] regression test, which fails loudly
    (values in the hundreds) under the un-scaled version."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < window or window <= 0:
        return pd.Series(out, index=series.index)
    out[window - 1] = float(np.nanmean(values[:window]))
    for t in range(window, n):
        out[t] = out[t - 1] - out[t - 1] / window + values[t] / window
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# MACD(12,26,9) -- SPEC.md B1.
# ---------------------------------------------------------------------------


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Standard MACD: macd_line = EMA(close,fast) - EMA(close,slow); signal_line =
    EMA(macd_line,signal); histogram = macd_line - signal_line. The first `slow`-1 bars of
    macd_line (and correspondingly the first `slow+signal`-2 bars of signal_line/histogram)
    are masked to NaN -- an EMA has no hard window so pandas would otherwise happily emit a
    numeric (but barely-seeded, unreliable) value from bar 0; masking keeps this module's
    "insufficient history -> NaN" convention consistent with atr()/wilder_smooth() above, and
    with it, downstream signal generation naturally treats warmup bars as "no signal" (NaN
    comparisons are False in pandas)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_line = macd_line.copy()
    macd_line.iloc[: slow - 1] = np.nan
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


# ---------------------------------------------------------------------------
# ADX / DMI(14) -- SPEC.md B2.
# ---------------------------------------------------------------------------


def adx_dmi(ohlc: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """+DI/-DI/ADX via Wilder's original method: directional movement (+DM/-DM) and true
    range are each Wilder-smoothed over `window`, +DI/-DI are the smoothed-DM/smoothed-TR
    ratios (x100), DX is their normalized absolute spread (x100), and ADX is DX itself
    Wilder-smoothed over `window` again. All four raw components (up_move, down_move, TR)
    are point-in-time at bar t (use only high/low/close through bar t)."""
    high = ohlc["high"]
    low = ohlc["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=ohlc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=ohlc.index)
    tr = true_range(ohlc)

    tr_smooth = wilder_smooth(tr, window)
    plus_dm_smooth = wilder_smooth(plus_dm, window)
    minus_dm_smooth = wilder_smooth(minus_dm, window)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_dm_smooth / tr_smooth
        minus_di = 100.0 * minus_dm_smooth / tr_smooth
        di_sum = plus_di + minus_di
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = wilder_smooth(dx, window)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "dx": dx, "adx": adx})


# ---------------------------------------------------------------------------
# Supertrend(10, 3.0) -- SPEC.md B3.
# ---------------------------------------------------------------------------


def supertrend(ohlc: pd.DataFrame, window: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Standard Supertrend: basic bands = hl2 +/- multiplier*ATR(window); final bands only
    ever tighten toward price (a final upper band can fall but never rise while price stays
    below it, and symmetrically for the lower band); the active line is the lower band while
    in an uptrend and the upper band while in a downtrend, flipping the instant close crosses
    to the other side of the currently-active band. `direction` is +1.0 (uptrend) / -1.0
    (downtrend) / 0.0 (undetermined, still warming up -- before ATR is defined)."""
    hl2 = (ohlc["high"] + ohlc["low"]) / 2.0
    atr_val = atr(ohlc, window)
    basic_upper = (hl2 + multiplier * atr_val).to_numpy(dtype=float)
    basic_lower = (hl2 - multiplier * atr_val).to_numpy(dtype=float)
    close = ohlc["close"].to_numpy(dtype=float)
    n = len(close)

    final_upper = np.full(n, np.nan, dtype=float)
    final_lower = np.full(n, np.nan, dtype=float)
    direction = np.zeros(n, dtype=float)
    line = np.full(n, np.nan, dtype=float)

    seeded = False
    for t in range(n):
        if np.isnan(basic_upper[t]) or np.isnan(basic_lower[t]):
            continue
        if not seeded:
            final_upper[t] = basic_upper[t]
            final_lower[t] = basic_lower[t]
            direction[t] = 1.0 if close[t] > hl2.iloc[t] else -1.0
            line[t] = final_lower[t] if direction[t] > 0.0 else final_upper[t]
            seeded = True
            continue
        final_upper[t] = basic_upper[t] if (basic_upper[t] < final_upper[t - 1] or close[t - 1] > final_upper[t - 1]) else final_upper[t - 1]
        final_lower[t] = basic_lower[t] if (basic_lower[t] > final_lower[t - 1] or close[t - 1] < final_lower[t - 1]) else final_lower[t - 1]
        if direction[t - 1] > 0.0:
            direction[t] = -1.0 if close[t] < final_lower[t] else 1.0
        else:
            direction[t] = 1.0 if close[t] > final_upper[t] else -1.0
        line[t] = final_lower[t] if direction[t] > 0.0 else final_upper[t]

    return pd.DataFrame(
        {
            "final_upper": final_upper,
            "final_lower": final_lower,
            "direction": direction,
            "line": line,
        },
        index=ohlc.index,
    )


# ---------------------------------------------------------------------------
# Keltner Channel(20, 2xATR) -- SPEC.md B4.
# ---------------------------------------------------------------------------


def keltner_channel(ohlc: pd.DataFrame, window: int = 20, atr_window: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    """Middle = EMA(close, window); upper/lower = middle +/- multiplier*ATR(atr_window). A
    single `window` governs both the EMA and the ATR by default (SPEC.md's "(20, 2xATR)" pins
    one period), but atr_window is exposed separately for testability."""
    middle = ema(ohlc["close"], window)
    atr_val = atr(ohlc, atr_window)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})


# ---------------------------------------------------------------------------
# Stochastic(14,3) -- SPEC.md B6.
# ---------------------------------------------------------------------------


def stochastic(ohlc: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """Fast stochastic: %K = 100*(close - lowest_low(k_window)) / (highest_high(k_window) -
    lowest_low(k_window)); %D = simple moving average of %K over d_window. A zero-range
    window (highest_high == lowest_low, a perfectly flat market) maps to NaN rather than a
    divide-by-zero -- true flat OHLC windows are vanishingly rare in real crypto data but must
    not raise/produce inf."""
    lowest_low = ohlc["low"].rolling(k_window, min_periods=k_window).min()
    highest_high = ohlc["high"].rolling(k_window, min_periods=k_window).max()
    band = highest_high - lowest_low
    band_safe = band.replace(0.0, np.nan)
    percent_k = 100.0 * (ohlc["close"] - lowest_low) / band_safe
    percent_d = percent_k.rolling(d_window, min_periods=d_window).mean()
    return pd.DataFrame({"percent_k": percent_k, "percent_d": percent_d})


# ---------------------------------------------------------------------------
# Multi-timeframe (MTF) alignment -- SPEC.md B5.
# ---------------------------------------------------------------------------


def moving_average_slope(close: pd.Series, ma_window: int, slope_lookback: int) -> pd.Series:
    """Simple moving average of `close` over `ma_window`, and its own change over the
    trailing `slope_lookback` observations (ma[t] - ma[t-slope_lookback]) -- the raw slope
    value; callers apply np.sign() for a pure up/down/flat trend read (SPEC.md B5: "1D
    추세방향(MA50 기울기)")."""
    moving_average = close.rolling(ma_window, min_periods=ma_window).mean()
    return moving_average - moving_average.shift(slope_lookback)


def align_daily_to_intraday(daily_series: pd.Series, intraday_index: pd.DatetimeIndex) -> pd.Series:
    """Broadcasts a DAILY series onto a finer `intraday_index` for MTF confluence, lagged by
    one full day first: only YESTERDAY's daily close (and everything computed from it) is
    knowable at any hour of TODAY. Identical convention to
    engine20.run_v1's own `armable_daily.shift(1).reindex(hourly.index, method="ffill")` --
    tests/test_wave25.py pins that a daily value change never leaks into intraday bars from
    before the NEXT daily bar closes."""
    lagged = daily_series.shift(1)
    return lagged.reindex(intraday_index, method="ffill")


__all__ = [
    "REQUIRED_OHLC_COLUMNS",
    "adx_dmi",
    "align_daily_to_intraday",
    "atr",
    "ema",
    "keltner_channel",
    "macd",
    "moving_average_slope",
    "stochastic",
    "supertrend",
    "true_range",
    "wilder_smooth",
]
