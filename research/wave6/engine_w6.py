# Wave-6 pure signal/pricing functions: funding-window, spillover, weekend, deviation-fade,
# and new-listing helpers. No network or disk I/O lives here so every function is unit-testable
# with synthetic frames (see research/wave6/tests).

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import PipelineError, load_frame, validate_symbol
from research.wave1.costs import PERP_TAKER_RATE, slippage_rate


BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR / "cache"
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
WAVE1_CACHE_DIR: Final = BASE_DIR.parent / "wave1" / "cache"
WAVE3_CACHE_DIR: Final = BASE_DIR.parent / "wave3" / "cache"

OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")

# Pre-registered thresholds (SPEC.md) -- fixed, no sweep.
FUNDING_THRESHOLD: Final = 0.0003  # 0.03%
SPILLOVER_SIGNAL_HOUR: Final = 12
SPILLOVER_ENTRY_HOUR: Final = 13
SPILLOVER_EXIT_HOUR: Final = 16
DEVIATION_ENTRY: Final = 0.003  # 0.3%
DEVIATION_EXIT: Final = 0.001  # 0.1%
SESSION_START_MINUTES: Final = 13 * 60 + 30  # 13:30 UTC
SESSION_END_MINUTES: Final = 20 * 60  # 20:00 UTC
LISTING_SHORT_OFFSET_DAYS: Final = 2
LISTING_COVER_OFFSET_DAYS: Final = 7
MIN_LISTING_SAMPLE: Final = 40


def intraday_round_trip_cost(symbol: str, stress_multiplier: float = 2.0) -> float:
    """Two-leg taker round trip with the wave-6 intraday convention (2x base slippage).

    SPEC.md preamble: "인트라데이는 스프레드 현실화(슬리피지 2배 적용)"; W6a/b restate this
    explicitly as "테이커 0.06%+슬리피지 2배(2bp) 왕복". stress_multiplier=4.0 doubles it again
    for the gate-7 cost-stress overlay.
    """
    return 2.0 * (PERP_TAKER_RATE + slippage_rate(symbol, stress_multiplier))


def read_symbol_frame(cache_dir: Path, prefix: str, symbol: str, suffix: str) -> pd.DataFrame:
    """Load a validated, path-contained cached frame."""
    safe_symbol = validate_symbol(symbol)
    cache_root = cache_dir.resolve()
    path = (cache_root / f"{prefix}{safe_symbol}{suffix}").resolve()
    if not path.is_file() or path.parent != cache_root:
        raise PipelineError(f"required wave-6 cache file is missing: {path.name}")
    return load_frame(path)


# --------------------------------------------------------------------------------------
# W6a / W6b -- funding window
# --------------------------------------------------------------------------------------


def previous_settlement_funding(funding: pd.Series) -> pd.Series:
    """Funding realized at the prior settlement -- the pre-registered proxy for the next payment.

    History has no forward-looking predicted-funding series, so SPEC.md pins the entry signal to
    the funding rate that was actually paid at the settlement immediately before the decision
    point (documented in the wave-6 report as a known proxy limitation).
    """
    return funding.shift(1)


def funding_window_trades(funding: pd.Series, price_open: pd.Series, threshold: float, direction: float) -> pd.DataFrame:
    """One row per funding settlement with entry/exit open prices and a trigger flag.

    direction=+1.0 is the long-below-threshold hypothesis (W6a); direction=-1.0 is the
    short-above-threshold hypothesis (W6b). Entry executes at the open of the bar starting one
    hour before settlement (the bar whose close, one hour earlier, is already confirmed); exit
    executes at the open of the bar starting one hour after settlement.
    """
    if direction not in (1.0, -1.0):
        raise PipelineError("direction must be +1.0 or -1.0")
    proxy = previous_settlement_funding(funding)
    settlements = pd.DatetimeIndex(funding.index)
    entry_time = settlements - pd.Timedelta(hours=1)
    exit_time = settlements + pd.Timedelta(hours=1)
    entry_open = price_open.reindex(entry_time).to_numpy(dtype=float)
    exit_open = price_open.reindex(exit_time).to_numpy(dtype=float)
    proxy_values = proxy.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        triggered = (proxy_values < -threshold) if direction > 0.0 else (proxy_values > threshold)
    has_prices = ~(np.isnan(entry_open) | np.isnan(exit_open))
    triggered = triggered & has_prices & ~np.isnan(proxy_values)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_return = direction * (exit_open / entry_open - 1.0)
    return pd.DataFrame(
        {"proxy_funding": proxy_values, "entry_open": entry_open, "exit_open": exit_open, "triggered": triggered, "raw_return": raw_return},
        index=settlements,
    )


# --------------------------------------------------------------------------------------
# W6c -- open spillover
# --------------------------------------------------------------------------------------


def spillover_trades(price: pd.DataFrame, exit_hour: int = SPILLOVER_EXIT_HOUR) -> pd.DataFrame:
    """One row per UTC weekday with the pre-open BTC direction and the follow trade.

    Approximation (documented in the wave-6 report): hourly bars only exist on the hour, so the
    pre-registered 12:30->13:30 UTC pre-open window is approximated by the [12:00,13:00) bar's own
    return, and the 13:30 entry is approximated by the 13:00 bar's open (the moment that bar's
    close is confirmed).
    """
    if not {"open", "close"}.issubset(price.columns):
        raise PipelineError("price frame must contain open and close columns")
    dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(price.index).normalize())))
    dates = dates[dates.weekday < 5]
    signal_ts = dates + pd.Timedelta(hours=SPILLOVER_SIGNAL_HOUR)
    entry_ts = dates + pd.Timedelta(hours=SPILLOVER_ENTRY_HOUR)
    exit_ts = dates + pd.Timedelta(hours=exit_hour)
    open_signal = price["open"].reindex(signal_ts).to_numpy(dtype=float)
    close_signal = price["close"].reindex(signal_ts).to_numpy(dtype=float)
    open_entry = price["open"].reindex(entry_ts).to_numpy(dtype=float)
    open_exit = price["open"].reindex(exit_ts).to_numpy(dtype=float)
    valid = ~(np.isnan(open_signal) | np.isnan(close_signal) | np.isnan(open_entry) | np.isnan(open_exit))
    with np.errstate(invalid="ignore"):
        signal = np.sign(close_signal / open_signal - 1.0)
    signal = np.where(np.isnan(signal), 0.0, signal)
    with np.errstate(invalid="ignore"):
        raw_return = signal * (open_exit / open_entry - 1.0)
    frame = pd.DataFrame({"signal": signal, "raw_return": raw_return, "valid": valid}, index=entry_ts)
    return frame[frame["valid"]].drop(columns="valid")


# --------------------------------------------------------------------------------------
# W6d -- weekend drift
# --------------------------------------------------------------------------------------


def weekend_trades(price: pd.DataFrame, hold_days: int = 2) -> pd.DataFrame:
    """One row per Saturday 00:00 UTC -> +hold_days entry/exit open pair (always-long BTC)."""
    if "open" not in price.columns:
        raise PipelineError("price frame must contain an open column")
    dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(price.index).normalize())))
    saturdays = dates[dates.weekday == 5]
    entry_ts = saturdays
    exit_ts = saturdays + pd.Timedelta(days=hold_days)
    entry_open = price["open"].reindex(entry_ts).to_numpy(dtype=float)
    exit_open = price["open"].reindex(exit_ts).to_numpy(dtype=float)
    valid = ~(np.isnan(entry_open) | np.isnan(exit_open))
    with np.errstate(invalid="ignore"):
        raw_return = exit_open / entry_open - 1.0
    frame = pd.DataFrame({"raw_return": raw_return, "valid": valid}, index=entry_ts)
    return frame[frame["valid"]].drop(columns="valid")


# --------------------------------------------------------------------------------------
# W6e -- token vs underlying deviation fade
# --------------------------------------------------------------------------------------


def regular_session_mask(index: pd.DatetimeIndex) -> np.ndarray:
    minutes = index.hour * 60 + index.minute
    return (minutes >= SESSION_START_MINUTES) & (minutes <= SESSION_END_MINUTES)


def align_token_underlying(token: pd.Series, underlying: pd.Series, tolerance: pd.Timedelta = pd.Timedelta(minutes=59)) -> pd.DataFrame:
    """Pair each underlying bar with the most recent token bar at or before it.

    Bitget's SPYUSDT bars are stamped on the hour; Yahoo's regular-session SPY bars are stamped
    30 minutes later (the NYSE opens at 9:30 ET = 13:30 UTC), so an exact-timestamp join finds
    almost no matches. This uses a backward as-of merge on the underlying's own timestamps -- it
    only ever reaches back to a token price already known by that moment, never ahead.
    """
    underlying_sorted = underlying.dropna().sort_index()
    token_sorted = token.dropna().sort_index()
    left = pd.DataFrame({"under": underlying_sorted.to_numpy()}, index=pd.DatetimeIndex(underlying_sorted.index))
    right = pd.DataFrame({"token": token_sorted.to_numpy()}, index=pd.DatetimeIndex(token_sorted.index))
    merged = pd.merge_asof(left, right, left_index=True, right_index=True, direction="backward", tolerance=tolerance)
    return merged.dropna()[["token", "under"]]


def session_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True at the last observed bar of each UTC calendar date in `index`."""
    normalized = pd.DatetimeIndex(index).normalize()
    last_per_day = pd.Series(index, index=normalized).groupby(level=0).max()
    return pd.Series(pd.Index(index).isin(last_per_day.to_numpy()), index=index)


def deviation_fade_position(dev: pd.Series, entry_dev: float, exit_dev: float, session_end: pd.Series) -> pd.Series:
    """Hysteresis fade toward the underlying, forced flat at the last bar of each session.

    dev > entry_dev means the token trades rich vs. the underlying -> fade short (-1). dev <
    -entry_dev means the token trades cheap -> fade long (+1). |dev| < exit_dev flattens. The
    session-close override always wins so the position never carries overnight.
    """
    if entry_dev <= exit_dev or exit_dev < 0.0:
        raise PipelineError("deviation thresholds must satisfy entry > exit >= 0")
    values: list[float] = []
    active = 0.0
    for timestamp, value in dev.items():
        if pd.notna(value):
            if value > entry_dev:
                active = -1.0
            elif value < -entry_dev:
                active = 1.0
            elif abs(value) < exit_dev:
                active = 0.0
        if bool(session_end.loc[timestamp]):
            active = 0.0
        values.append(active)
    return pd.Series(values, index=dev.index, dtype=float)


# --------------------------------------------------------------------------------------
# W6f -- new-listing effect
# --------------------------------------------------------------------------------------


def eligible_listing_symbols(contracts: object) -> tuple[tuple[str, pd.Timestamp], ...]:
    """Symbols with a populated Bitget launchTime, paired with their listing timestamp."""
    if not isinstance(contracts, list):
        raise PipelineError("bitget contracts payload must be a list")
    eligible: list[tuple[str, pd.Timestamp]] = []
    for item in contracts:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol")
        launch = item.get("launchTime")
        if not isinstance(symbol, str) or not isinstance(launch, str) or launch in ("", "0"):
            continue
        try:
            launch_ms = int(launch)
        except ValueError:
            continue
        if launch_ms <= 0:
            continue
        eligible.append((symbol, pd.to_datetime(launch_ms, unit="ms", utc=True)))
    return tuple(eligible)


def listing_short_trade(
    daily: pd.DataFrame,
    onboard_date: pd.Timestamp,
    short_offset_days: int = LISTING_SHORT_OFFSET_DAYS,
    cover_offset_days: int = LISTING_COVER_OFFSET_DAYS,
) -> float | None:
    """Short-at-D+short_offset open, cover-at-D+cover_offset close return (short: gains on decline)."""
    base = pd.Timestamp(onboard_date).normalize()
    entry_date = base + pd.Timedelta(days=short_offset_days)
    exit_date = base + pd.Timedelta(days=cover_offset_days)
    daily_index = pd.DatetimeIndex(daily.index).normalize()
    positioned = daily.set_axis(daily_index)
    if entry_date not in positioned.index or exit_date not in positioned.index:
        return None
    entry_open = float(positioned.loc[entry_date, "open"])
    exit_close = float(positioned.loc[exit_date, "close"])
    if not np.isfinite(entry_open) or entry_open <= 0.0 or not np.isfinite(exit_close):
        return None
    return -(exit_close / entry_open - 1.0)


__all__ = [
    "BASE_DIR",
    "CACHE_DIR",
    "DEVIATION_ENTRY",
    "DEVIATION_EXIT",
    "FUNDING_THRESHOLD",
    "LISTING_COVER_OFFSET_DAYS",
    "LISTING_SHORT_OFFSET_DAYS",
    "MIN_LISTING_SAMPLE",
    "OOS_SPLIT",
    "REPORT_DIR",
    "RESULTS_DIR",
    "SPILLOVER_ENTRY_HOUR",
    "SPILLOVER_EXIT_HOUR",
    "SPILLOVER_SIGNAL_HOUR",
    "WAVE1_CACHE_DIR",
    "WAVE3_CACHE_DIR",
    "align_token_underlying",
    "deviation_fade_position",
    "eligible_listing_symbols",
    "funding_window_trades",
    "intraday_round_trip_cost",
    "listing_short_trade",
    "previous_settlement_funding",
    "read_symbol_frame",
    "regular_session_mask",
    "session_end_mask",
    "spillover_trades",
    "weekend_trades",
]
