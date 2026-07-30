# Wave-30 data layer. Builds ONE immutable, fully-precomputed market cache from the local
# 1h cache so that a single genome evaluation costs no rolling-window recomputation.
#
# Why 1h and why only BTC/ETH/SOL: see SPEC.md "데이터 (동결) 및 그 한계". At 20x the
# liquidation band is 4.5% wide, which a daily bar cannot resolve -- daily MAE understates
# how often price passed through the band intrabar. The local 1h cache
# (research/wave6/cache) covers exactly three symbols and the Binance API is region-blocked
# (HTTP 451) in this sandbox, so three symbols is a DATA CONSTRAINT, not a design choice.
#
# Every rolling statistic is precomputed once per (symbol, lookback) over the 8 frozen
# lookbacks in SPEC.md, because the search evaluates ~76,000 genomes and recomputing a
# 60,000-bar rolling window per genome would dominate runtime.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
HOURLY_CACHE: Final = REPO_ROOT / "research" / "wave6" / "cache"
DAILY_CACHE: Final = REPO_ROOT / "research" / "wave3" / "cache"
I5_RESULTS: Final = REPO_ROOT / "research" / "wave18_idle" / "results" / "I5.json"

SYMBOLS: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
LOOKBACKS: Final = (6, 12, 24, 48, 72, 120, 168, 336)
OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")


class OOSLeakageError(RuntimeError):
    """Raised when the evolutionary loop tries to read post-OOS_SPLIT data. SPEC.md
    contamination block #1 -- enforced in code, pinned by tests/test_wave30.py."""


@dataclass(frozen=True)
class SymbolArrays:
    """All per-bar arrays for one symbol, aligned to the shared global hourly index.
    Bars where the symbol had not listed yet are NaN in prices and False in `tradable`."""

    symbol: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tradable: np.ndarray  # bool: this bar has a usable OHLC print
    funding_at_bar: np.ndarray  # funding rate CHARGED at this bar (0.0 on non-funding bars)
    cost_rate: float  # one-way cost fraction of notional (taker + measured slippage)
    # per-lookback precomputed features, keyed by lookback
    prior_high: dict[int, np.ndarray]  # max(high) over the L bars ENDING AT i-1
    prior_low: dict[int, np.ndarray]  # min(low) over the L bars ENDING AT i-1
    ret: dict[int, np.ndarray]  # close[i]/close[i-L] - 1
    vol: dict[int, np.ndarray]  # stdev of hourly log returns over the L bars ending at i
    zscore: dict[int, np.ndarray]  # (close[i] - ma_L) / sd_L, both over bars ending at i


@dataclass(frozen=True)
class MarketCache:
    index: pd.DatetimeIndex  # shared hourly index
    symbols: tuple[str, ...]
    arrays: dict[str, SymbolArrays]
    is_mask: np.ndarray  # bool: bar is in-sample (<= OOS_SPLIT)
    day_of_bar: np.ndarray  # int index into `daily_index` for each hourly bar
    daily_index: pd.DatetimeIndex
    stable_daily_factor: np.ndarray  # I5 daily growth factor, aligned to daily_index
    stable_per_dollar: np.ndarray  # value of $1 allocated to the stable sleeve, day by day
    n_bars: int

    def signal_universe(self, symbols: tuple[str, ...]) -> tuple[SymbolArrays, ...]:
        return tuple(self.arrays[s] for s in symbols)


def _load_hourly(symbol: str) -> pd.DataFrame:
    path = HOURLY_CACHE / f"binance_fapi_{symbol}_1h.csv.gz"
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp").set_index("timestamp")
    return frame[["open", "high", "low", "close", "quote_volume"]].astype(float)


def _load_funding(symbol: str) -> pd.Series:
    path = DAILY_CACHE / f"binance_funding_{symbol}.csv.gz"
    frame = pd.read_csv(path)
    # Funding stamps carry sub-second jitter (e.g. "...16:00:00.001000+00:00"), so parse as
    # mixed ISO8601 rather than pinning one format.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    series = frame.set_index("timestamp")["funding_rate"].astype(float)
    return series[~series.index.duplicated(keep="first")].sort_index()


def _prior_extreme(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    """Rolling max/min over the `window` bars ENDING AT i-1 (bar i's own print excluded, so a
    breakout test at bar i's close never consults bar i's own high/low)."""
    series = pd.Series(values)
    rolled = series.rolling(window, min_periods=window).max() if kind == "max" else series.rolling(window, min_periods=window).min()
    return rolled.shift(1).to_numpy(dtype=float)


def _one_way_cost_rate(symbol: str, mean_quote_volume: float) -> float:
    """Taker fee + measured Bitget slippage for a SINGLE leg of a SINGLE instrument.

    Taker (not maker) because every exit path in engine30 -- stop, liquidation gap, forced
    max-hold close -- is a market order, and a breakout entry that waits for a maker fill is
    a different (unbacktested) strategy. Slippage reuses wave13's measured volume mapping
    completely unmodified; see research/wave13_liquidity/costs_measured.py.
    """
    from research.wave13_liquidity import costs_measured

    mapping = costs_measured.fit_mapping()
    slippage_bp = costs_measured.slippage_bp_for_volume(mean_quote_volume, mapping)
    taker_rate = 0.0006  # Bitget/Binance USDT-M taker, same constant wave29_lev10 used
    return taker_rate + slippage_bp / 10_000.0


def _stable_daily_factor(daily_index: pd.DatetimeIndex) -> np.ndarray:
    """I5's REALISED daily growth factors (research/wave18_idle/results/I5.json), reindexed
    onto our daily index. Days outside I5's own span get a factor of exactly 1.0 -- the
    stable sleeve is modelled as idle cash there rather than extrapolated."""
    import json

    payload = json.loads(I5_RESULTS.read_text(encoding="utf-8"))
    equity = pd.Series(
        [float(item["value"]) for item in payload["equity"]],
        index=pd.to_datetime([item["timestamp"] for item in payload["equity"]], utc=True),
    ).sort_index()
    factor = (equity / equity.shift(1)).dropna()
    factor = factor.reindex(daily_index.tz_convert("UTC")).fillna(1.0)
    return factor.to_numpy(dtype=float)


@lru_cache(maxsize=1)
def build_market_cache() -> MarketCache:
    frames = {symbol: _load_hourly(symbol) for symbol in SYMBOLS}
    index = frames[SYMBOLS[0]].index
    for symbol in SYMBOLS[1:]:
        index = index.union(frames[symbol].index)
    index = pd.DatetimeIndex(index).sort_values()
    n_bars = len(index)

    arrays: dict[str, SymbolArrays] = {}
    for symbol, frame in frames.items():
        aligned = frame.reindex(index)
        open_a = aligned["open"].to_numpy(dtype=float)
        high_a = aligned["high"].to_numpy(dtype=float)
        low_a = aligned["low"].to_numpy(dtype=float)
        close_a = aligned["close"].to_numpy(dtype=float)
        tradable = np.isfinite(open_a) & np.isfinite(high_a) & np.isfinite(low_a) & np.isfinite(close_a)

        # Funding is charged at discrete 8h stamps; attribute each stamp to the hourly bar
        # that contains it, and 0.0 everywhere else.
        funding = _load_funding(symbol)
        bucket = funding.index.floor("1h")
        funding_at_bar = pd.Series(funding.to_numpy(dtype=float), index=bucket).groupby(level=0).sum()
        funding_at_bar = funding_at_bar.reindex(index).fillna(0.0).to_numpy(dtype=float)

        log_close = np.log(np.where(tradable, close_a, np.nan))
        log_ret = np.concatenate([[np.nan], np.diff(log_close)])
        log_ret_series = pd.Series(log_ret)

        prior_high: dict[int, np.ndarray] = {}
        prior_low: dict[int, np.ndarray] = {}
        ret_map: dict[int, np.ndarray] = {}
        vol_map: dict[int, np.ndarray] = {}
        z_map: dict[int, np.ndarray] = {}
        close_series = pd.Series(np.where(tradable, close_a, np.nan))
        for lookback in LOOKBACKS:
            prior_high[lookback] = _prior_extreme(np.where(tradable, high_a, np.nan), lookback, "max")
            prior_low[lookback] = _prior_extreme(np.where(tradable, low_a, np.nan), lookback, "min")
            shifted = close_series.shift(lookback)
            ret_map[lookback] = (close_series / shifted - 1.0).to_numpy(dtype=float)
            vol_map[lookback] = log_ret_series.rolling(lookback, min_periods=lookback).std(ddof=1).to_numpy(dtype=float)
            mean = close_series.rolling(lookback, min_periods=lookback).mean()
            sd = close_series.rolling(lookback, min_periods=lookback).std(ddof=1)
            z_map[lookback] = ((close_series - mean) / sd.replace(0.0, np.nan)).to_numpy(dtype=float)

        mean_volume = float(np.nanmean(aligned["quote_volume"].to_numpy(dtype=float)) * 24.0)
        arrays[symbol] = SymbolArrays(
            symbol=symbol,
            open=open_a,
            high=high_a,
            low=low_a,
            close=close_a,
            tradable=tradable,
            funding_at_bar=funding_at_bar,
            cost_rate=_one_way_cost_rate(symbol, mean_volume),
            prior_high=prior_high,
            prior_low=prior_low,
            ret=ret_map,
            vol=vol_map,
            zscore=z_map,
        )

    is_mask = np.asarray(index <= OOS_SPLIT)
    daily_index = pd.DatetimeIndex(index.floor("1D").unique()).sort_values()
    day_of_bar = daily_index.get_indexer(index.floor("1D"))
    factor = _stable_daily_factor(daily_index)
    return MarketCache(
        index=index,
        symbols=SYMBOLS,
        arrays=arrays,
        is_mask=is_mask,
        day_of_bar=day_of_bar,
        daily_index=daily_index,
        stable_daily_factor=factor,
        stable_per_dollar=stable_value_per_dollar(factor),
        n_bars=n_bars,
    )


def stable_value_per_dollar(factor: np.ndarray) -> np.ndarray:
    """Value of $1 allocated to the stable sleeve, day by day.

    I5's own capital contract (research/wave18_idle/results/I5.json capital_contract) puts
    RESERVE_FRACTION=0.10 in idle cash and compounds only the remaining 90%, and I5.json's
    equity series is that 90% ACTIVE leg alone. Applying its growth factors to the whole
    stable allocation would therefore quietly hand the stable sleeve a return I5 never earned,
    so the reserve split is reproduced here: 90% compounds at I5's realised factors, 10% sits
    flat. This also makes the sleeve_fraction=0 limit exactly equal to a $100-basis I5 system,
    which is the baseline P2 is measured against.
    """
    reserve_fraction = 0.10
    return (1.0 - reserve_fraction) * np.cumprod(factor) + reserve_fraction


def i5_baseline_total_curve(cache: MarketCache) -> np.ndarray:
    """A $100-basis I5-only system on the same daily calendar. This is the sleeve_fraction->0
    limit of every wave30 candidate, so any comparison against it is apples to apples."""
    return 100.0 * cache.stable_per_dollar


def is_bar_count(cache: MarketCache) -> int:
    return int(cache.is_mask.sum())
