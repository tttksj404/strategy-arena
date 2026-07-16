"""Fixed-parameter strategy battery for hourly RWA futures."""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """A named strategy and its fixed parameters."""

    name: str
    params: dict[str, int | float | str]


def specs() -> list[StrategySpec]:
    """Return the fixed strategy battery."""
    return [StrategySpec("B0_buy_hold", {"direction": 1}), StrategySpec("B1_tsmom_ma", {"fast": 10, "slow": 50}),
            StrategySpec("B1_tsmom_ma", {"fast": 20, "slow": 100}), StrategySpec("B1_donchian", {"lookback": 20}),
            StrategySpec("B2_mean_reversion", {"lookback": 24, "z_entry": 2, "max_hold": 48}),
            StrategySpec("B3_session", {"window": 3}), StrategySpec("B4_vol_breakout", {"bb": 20, "atr_stop": 2, "max_hold": 20})]


def _daily_signal(candles: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    close = candles.set_index("ts")["close"].resample("1D").last()
    signal = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float) * 2 - 1
    signal = signal.shift(1)
    return signal.reindex(candles["ts"], method="ffill").fillna(0).reset_index(drop=True)


def _donchian(candles: pd.DataFrame, lookback: int) -> pd.Series:
    close = candles["close"]
    upper = candles["high"].rolling(lookback).max().shift(1)
    lower = candles["low"].rolling(lookback).min().shift(1)
    signal = pd.Series(0.0, index=candles.index)
    signal[close > upper] = 1
    signal[close < lower] = -1
    return signal.replace(0, np.nan).ffill().fillna(0)


def _mean_reversion(candles: pd.DataFrame, lookback: int, z_entry: float, max_hold: int) -> pd.Series:
    close = candles["close"].astype(float)
    mean, std = close.rolling(lookback).mean(), close.rolling(lookback).std()
    z = (close - mean) / std.replace(0, np.nan)
    out = np.zeros(len(candles))
    held = 0
    for i, value in enumerate(z.fillna(0)):
        if held:
            held += 1
            if held >= max_hold or (out[i - 1] > 0 and value >= 0) or (out[i - 1] < 0 and value <= 0):
                out[i] = 0
                held = 0
            else:
                out[i] = out[i - 1]
        elif value > z_entry:
            out[i], held = -1, 1
        elif value < -z_entry:
            out[i], held = 1, 1
    return pd.Series(out, index=candles.index)


def session_candidates(candles: pd.DataFrame) -> list[tuple[str, set[int], float]]:
    """Select three contiguous hour-of-week windows using train data only: best long, two worst short."""
    local = candles["ts"].dt.tz_convert(ZoneInfo("America/New_York"))
    returns = candles["close"].pct_change()
    buckets = returns.groupby(local.dt.dayofweek * 24 + local.dt.hour).mean().reindex(range(168)).fillna(0)
    def window(start: int) -> set[int]:
        return {(start + offset) % 168 for offset in range(3)}
    scores = [(float(buckets.iloc[(start + np.arange(3)) % 168].mean()), start) for start in range(168)]
    top = [(f"hour_{start}_long", window(start), 1.0) for _, start in sorted(scores, reverse=True)[:1]]
    bottom = [(f"hour_{start}_short", window(start), -1.0) for _, start in sorted(scores)[:2]]
    return top + bottom


def _session(candles: pd.DataFrame, buckets: set[int], direction: float = 1.0) -> pd.Series:
    local = candles["ts"].dt.tz_convert(ZoneInfo("America/New_York"))
    bucket_ids = local.dt.dayofweek * 24 + local.dt.hour
    return pd.Series(np.where(bucket_ids.isin(buckets), direction, 0.0), index=candles.index)


def _vol_breakout(candles: pd.DataFrame, bb: int, atr_stop: float, max_hold: int) -> pd.Series:
    close, high, low = candles["close"], candles["high"], candles["low"]
    mid = close.rolling(bb).mean(); width = (close.rolling(bb).std() * 4 / mid).replace([np.inf, -np.inf], np.nan)
    squeeze = width < width.rolling(100).quantile(0.2)
    atr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    out = np.zeros(len(candles)); held = 0; stop = 0.0
    for i in range(1, len(candles)):
        if held:
            held += 1
            if (out[i - 1] > 0 and low.iloc[i] <= stop) or (out[i - 1] < 0 and high.iloc[i] >= stop) or held >= max_hold:
                out[i], held = 0, 0
            else: out[i] = out[i - 1]
        elif bool(squeeze.iloc[i - 1]) and close.iloc[i] > high.iloc[i - 1]:
            out[i], held, stop = 1, 1, close.iloc[i] - atr_stop * atr.iloc[i]
        elif bool(squeeze.iloc[i - 1]) and close.iloc[i] < low.iloc[i - 1]:
            out[i], held, stop = -1, 1, close.iloc[i] + atr_stop * atr.iloc[i]
    return pd.Series(out, index=candles.index)


def signal_for(spec: StrategySpec, candles: pd.DataFrame, session_buckets: tuple[str, set[int], float] | None = None) -> pd.Series:
    """Build a close-time signal; the engine applies the next-open shift."""
    match spec.name:
        case "B0_buy_hold": return pd.Series(float(spec.params["direction"]), index=candles.index)
        case "B1_tsmom_ma": return _daily_signal(candles, int(spec.params["fast"]), int(spec.params["slow"]))
        case "B1_donchian": return _donchian(candles, int(spec.params["lookback"]))
        case "B2_mean_reversion": return _mean_reversion(candles, int(spec.params["lookback"]), float(spec.params["z_entry"]), int(spec.params["max_hold"]))
        case "B3_session": return _session(candles, session_buckets[1], session_buckets[2]) if session_buckets else pd.Series(0.0, index=candles.index)
        case "B4_vol_breakout": return _vol_breakout(candles, int(spec.params["bb"]), float(spec.params["atr_stop"]), int(spec.params["max_hold"]))
        case _: raise ValueError(f"unknown strategy {spec.name}")


def cross_section_signals(frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Weekly top-3/bottom-3 20-day return cross-section signals."""
    daily: dict[str, pd.Series] = {symbol: frame.set_index("ts")["close"].resample("1D").last() for symbol, frame in frames.items()}
    panel = pd.DataFrame(daily).sort_index()
    ranks = panel.pct_change(20).shift(1).rank(axis=1, ascending=False)
    result: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        rank = ranks[symbol].where(ranks.index.to_series().dt.dayofweek == 0).ffill()
        signal = np.where(rank <= 3, 1.0, np.where(rank > len(frames) - 3, -1.0, 0.0))
        result[symbol] = pd.Series(signal, index=ranks.index).reindex(frame["ts"], method="ffill").fillna(0).reset_index(drop=True)
    return result
