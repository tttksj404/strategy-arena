# Wave-20 cache readers. Deliberately thin and read-only: every function here loads directly
# from research/wave1/cache, research/wave3/cache, or research/wave6/cache (gzipped Binance
# OHLCV/funding CSVs already fetched by those earlier waves -- see SPEC.md "배경"). Nothing in
# this module writes to another wave's directory, and nothing hits the network (network is
# available this session per the task brief, but every symbol/timeframe V1-V5 need is already
# on disk -- see the wave-by-wave cache inventory in configs20.py's module docstring).
#
# This module does NOT go through research.wave13_liquidity.universe_liquidity /
# research.wave12_frontier.universe_frontier's point-in-time-breadth machinery: that pipeline
# resolves a DYNAMIC top-N-by-volume universe for L1-L5's cross-sectional carry ranking, which
# none of V1-V5 need (V1/V4 trade a fixed named symbol set, V2 ranks within a fixed 70-symbol
# funding-covered set, V3/V5 use wave3's full static 332-symbol cache directly). Going through
# that pipeline here would silently re-import wave12's own FROZEN_END/candidate_pool universe
# definition into a wave whose SPEC never asked for it.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave20_convex.configs20 import WAVE1_CACHE_DIR, WAVE3_CACHE_DIR, WAVE6_CACHE_DIR

OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume", "quote_volume")


class DataError(Exception):
    pass


def _parse_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    """read_csv(parse_dates=[...]) silently leaves `timestamp` as plain str under this
    environment's pandas (3.0.3) whenever the column mixes formats -- most rows have no
    fractional seconds but some do (e.g. research/wave1/cache's funding files), which trips
    the fast-path parser into giving up quietly rather than raising. Parsed explicitly and
    robustly here (format='mixed') so every loader in this module gets a real DatetimeIndex,
    not a str index that would silently break every downstream .resample()/.rolling() call."""
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    return frame


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"missing cache file: {path}")
    frame = pd.read_csv(path)
    frame = _parse_timestamp_column(frame)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[list(OHLCV_COLUMNS)]


def load_daily(symbol: str, cache_dir: Path) -> pd.DataFrame:
    """research/{wave1,wave3}/cache's shared binance_fapi_{symbol}_1d.csv.gz layout."""
    return _read_ohlcv_csv(cache_dir / f"binance_fapi_{symbol}_1d.csv.gz")


def load_hourly(symbol: str, cache_dir: Path = WAVE6_CACHE_DIR) -> pd.DataFrame:
    """research/wave6/cache's binance_fapi_{symbol}_1h.csv.gz layout (BTC/ETH/SOL only)."""
    return _read_ohlcv_csv(cache_dir / f"binance_fapi_{symbol}_1h.csv.gz")


def try_load_daily(symbol: str, cache_dir: Path) -> pd.DataFrame | None:
    try:
        return load_daily(symbol, cache_dir)
    except DataError:
        return None


def try_load_hourly(symbol: str, cache_dir: Path = WAVE6_CACHE_DIR) -> pd.DataFrame | None:
    try:
        return load_hourly(symbol, cache_dir)
    except DataError:
        return None


def load_funding_rate(symbol: str, cache_dir: Path = WAVE1_CACHE_DIR) -> pd.Series:
    """research/wave1/cache's binance_funding_{symbol}.csv.gz -- raw per-8h funding_rate,
    NOT annualized (see research.wave1.fam_funding.funding_score for the shared annualization
    convention every wave reuses)."""
    path = cache_dir / f"binance_funding_{symbol}.csv.gz"
    if not path.exists():
        raise DataError(f"missing funding cache file: {path}")
    frame = pd.read_csv(path)
    frame = _parse_timestamp_column(frame)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame["funding_rate"].astype(float)


def wave1_symbols_with_funding() -> tuple[str, ...]:
    """Symbols in research/wave1/cache carrying BOTH a daily-perp OHLCV file and a funding
    file -- V2's candidate universe (SPEC.md "펀딩+가격"). Sorted for determinism (filesystem
    glob order is not guaranteed)."""
    price = {path.name.removeprefix("binance_fapi_").removesuffix("_1d.csv.gz") for path in WAVE1_CACHE_DIR.glob("binance_fapi_*_1d.csv.gz")}
    funding = {path.name.removeprefix("binance_funding_").removesuffix(".csv.gz") for path in WAVE1_CACHE_DIR.glob("binance_funding_*.csv.gz")}
    return tuple(sorted(price & funding))


def wave3_symbols() -> tuple[str, ...]:
    """All 332 symbols in research/wave3/cache (V3/V5's broad universe)."""
    return tuple(sorted(path.name.removeprefix("binance_fapi_").removesuffix("_1d.csv.gz") for path in WAVE3_CACHE_DIR.glob("binance_fapi_*_1d.csv.gz")))


def first_candle_dates(symbols: tuple[str, ...], cache_dir: Path = WAVE3_CACHE_DIR, min_rows: int = 1) -> dict[str, pd.Timestamp]:
    """Point-in-time listing-date proxy for V3 (SPEC.md background: Bitget launchTime is
    blank for every contract, confirmed empirically in wave-6, so each symbol's own first
    cached daily candle stands in for its listing date). Only symbols with >= min_rows daily
    rows on file are returned -- a symbol with too little history to even attempt a
    hold_days-long trade is not a usable "listing" observation."""
    listings: dict[str, pd.Timestamp] = {}
    for symbol in symbols:
        frame = try_load_daily(symbol, cache_dir)
        if frame is None or len(frame) < min_rows:
            continue
        listings[symbol] = pd.Timestamp(frame.index[0])
    return listings


def resample_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """OHLCV daily bars aggregated from 1H bars -- used by V1/V4 for the daily-granularity
    realized-vol regime filter / daily cost-model volume lookup, while the 1H bars themselves
    still drive entry/exit execution timing."""
    daily = hourly.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}
    )
    return daily.dropna(subset=["open", "close"])


__all__ = [
    "OHLCV_COLUMNS",
    "DataError",
    "first_candle_dates",
    "load_daily",
    "load_funding_rate",
    "load_hourly",
    "resample_hourly_to_daily",
    "try_load_daily",
    "try_load_hourly",
    "wave1_symbols_with_funding",
    "wave3_symbols",
]
