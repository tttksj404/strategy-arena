# Wave-35 market cache for the widened universe.
#
# ---------------------------------------------------------------------------------------
# Two venues, and why
# ---------------------------------------------------------------------------------------
# PRICES come from Bitget (the only source reachable here with 1h history for many symbols) and
# FUNDING comes from the local Binance cache. That split is forced, not chosen:
#   * Binance klines are region-blocked (HTTP 451), so 1h prices for 39 of these symbols exist
#     nowhere else locally.
#   * Bitget's history-fund-rate paginates back only ~3 months (measured: 2026-05-01 onward), so
#     it cannot cover a 2022-2026 backtest at all.
# The mismatch is defensible because validate_bitget.py measured the two venues' 1h returns at
# 0.9973-0.9996 correlation and 0.015-0.024% median close error over 2022+, i.e. the price series
# are effectively the same instrument. It is still a mismatch and the report says so; wave14
# measured cross-venue funding SIGN agreement at 98.9% (Bybit) and 91.7% (OKX) but never Bitget.
#
# ---------------------------------------------------------------------------------------
# Span starts 2022-01-01, and that is a data-integrity decision, not a convenience
# ---------------------------------------------------------------------------------------
# Bitget's pre-2022 perpetual history contains frozen segments -- BNBUSDT alone has a 15,001-bar
# (~1.7 year) run of identical closes, and BTCUSDT sits pinned at 23301.00 across 2020-12-25..30
# before printing a single +26% catch-up bar. A frozen price manufactures fake calm (no stop-outs)
# followed by a fake jackpot, which is the worst possible input to a leveraged backtest. After the
# cutoff, 0 of 42 symbols show any stale run above 1% of bars.

from __future__ import annotations

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

from research.wave30_qd.dataio30 import (
    LOOKBACKS,
    MarketCache,
    SymbolArrays,
    _prior_extreme,
    stable_value_per_dollar,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
BITGET_CACHE: Final = Path(__file__).resolve().parent / "cache"
BINANCE_FUNDING: Final = REPO_ROOT / "research" / "wave3" / "cache"
I5_RESULTS: Final = REPO_ROOT / "research" / "wave18_idle" / "results" / "I5.json"

USABLE_START: Final = pd.Timestamp("2022-01-01", tz="UTC")
USABLE_END: Final = pd.Timestamp("2026-07-14", tz="UTC")  # Binance funding cache ends here
OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")  # unchanged from wave30, comparability
MIN_FUNDING_STAMPS: Final = 4_000  # ~3.6 years of 8h funding; below this the cost model is guesswork
TAKER_FEE: Final = 0.0006


class UniverseError(RuntimeError):
    pass


def _read_prices(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(BITGET_CACHE / f"bitget_{symbol}_1H.csv.gz")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp").set_index("timestamp")
    return frame.loc[(frame.index >= USABLE_START) & (frame.index <= USABLE_END)]


def _read_funding(symbol: str) -> pd.Series | None:
    path = BINANCE_FUNDING / f"binance_funding_{symbol}.csv.gz"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    series = frame.set_index("timestamp")["funding_rate"].astype(float)
    series = series[~series.index.duplicated(keep="first")].sort_index()
    return series.loc[(series.index >= USABLE_START) & (series.index <= USABLE_END)]


def eligible_universe() -> list[tuple[str, float]]:
    """Symbols with (a) Bitget 1h covering the whole usable span and (b) enough Binance funding.

    Ranked by median hourly quote volume -- a point-in-time-free ranking used only to define the
    breadth TIERS the search may choose from, never as a signal.
    """
    rows: list[tuple[str, float]] = []
    for path in sorted(BITGET_CACHE.glob("bitget_*_1H.csv.gz")):
        symbol = path.name.replace("bitget_", "").replace("_1H.csv.gz", "")
        prices = _read_prices(symbol)
        if prices.empty or prices.index[0] > USABLE_START + pd.Timedelta(days=2):
            continue
        funding = _read_funding(symbol)
        if funding is None or len(funding) < MIN_FUNDING_STAMPS:
            continue
        rows.append((symbol, float(prices["quote_volume"].median() * 24.0)))
    rows.sort(key=lambda item: -item[1])
    if not rows:
        raise UniverseError("no symbol satisfied both price-span and funding-coverage requirements")
    return rows


def _cost_rate(mean_daily_quote_volume: float) -> float:
    """Taker fee + wave13's measured volume->slippage mapping, applied UNMODIFIED.

    Each symbol gets its own slippage from its own traded volume rather than inheriting BTC's
    0.0169bp, which matters now that the universe includes names trading three orders of magnitude
    less than BTC.
    """
    from research.wave13_liquidity import costs_measured

    mapping = costs_measured.fit_mapping()
    slippage_bp = costs_measured.slippage_bp_for_volume(mean_daily_quote_volume, mapping)
    return TAKER_FEE + slippage_bp / 10_000.0


def _stable_factor(daily_index: pd.DatetimeIndex) -> np.ndarray:
    import json

    payload = json.loads(I5_RESULTS.read_text(encoding="utf-8"))
    equity = pd.Series(
        [float(item["value"]) for item in payload["equity"]],
        index=pd.to_datetime([item["timestamp"] for item in payload["equity"]], utc=True),
    ).sort_index()
    factor = (equity / equity.shift(1)).dropna()
    return factor.reindex(daily_index).fillna(1.0).to_numpy(dtype=float)


@lru_cache(maxsize=1)
def build_wide_cache() -> tuple[MarketCache, tuple[str, ...]]:
    """Returns (cache, symbols_ranked_by_volume). The ranking drives the breadth gene."""
    universe = eligible_universe()
    symbols = tuple(symbol for symbol, _volume in universe)
    frames = {symbol: _read_prices(symbol) for symbol in symbols}

    index = frames[symbols[0]].index
    for symbol in symbols[1:]:
        index = index.union(frames[symbol].index)
    index = pd.DatetimeIndex(index).sort_values()

    arrays: dict[str, SymbolArrays] = {}
    for symbol, volume in universe:
        aligned = frames[symbol].reindex(index)
        open_a = aligned["open"].to_numpy(dtype=float)
        high_a = aligned["high"].to_numpy(dtype=float)
        low_a = aligned["low"].to_numpy(dtype=float)
        close_a = aligned["close"].to_numpy(dtype=float)
        tradable = np.isfinite(open_a) & np.isfinite(high_a) & np.isfinite(low_a) & np.isfinite(close_a)

        funding = _read_funding(symbol)
        bucket = funding.index.floor("1h")
        charged = pd.Series(funding.to_numpy(dtype=float), index=bucket).groupby(level=0).sum()
        funding_at_bar = charged.reindex(index).fillna(0.0).to_numpy(dtype=float)

        close_series = pd.Series(np.where(tradable, close_a, np.nan))
        log_return = pd.Series(np.concatenate([[np.nan], np.diff(np.log(close_series.to_numpy()))]))

        prior_high: dict[int, np.ndarray] = {}
        prior_low: dict[int, np.ndarray] = {}
        ret_map: dict[int, np.ndarray] = {}
        vol_map: dict[int, np.ndarray] = {}
        z_map: dict[int, np.ndarray] = {}
        for lookback in LOOKBACKS:
            prior_high[lookback] = _prior_extreme(np.where(tradable, high_a, np.nan), lookback, "max")
            prior_low[lookback] = _prior_extreme(np.where(tradable, low_a, np.nan), lookback, "min")
            ret_map[lookback] = (close_series / close_series.shift(lookback) - 1.0).to_numpy(dtype=float)
            vol_map[lookback] = log_return.rolling(lookback, min_periods=lookback).std(ddof=1).to_numpy(dtype=float)
            mean = close_series.rolling(lookback, min_periods=lookback).mean()
            sd = close_series.rolling(lookback, min_periods=lookback).std(ddof=1)
            z_map[lookback] = ((close_series - mean) / sd.replace(0.0, np.nan)).to_numpy(dtype=float)

        arrays[symbol] = SymbolArrays(
            symbol=symbol,
            open=open_a,
            high=high_a,
            low=low_a,
            close=close_a,
            tradable=tradable,
            funding_at_bar=funding_at_bar,
            cost_rate=_cost_rate(volume),
            prior_high=prior_high,
            prior_low=prior_low,
            ret=ret_map,
            vol=vol_map,
            zscore=z_map,
        )

    is_mask = np.asarray(index <= OOS_SPLIT)
    daily_index = pd.DatetimeIndex(index.floor("1D").unique()).sort_values()
    day_of_bar = daily_index.get_indexer(index.floor("1D"))
    factor = _stable_factor(daily_index)
    cache = MarketCache(
        index=index,
        symbols=symbols,
        arrays=arrays,
        is_mask=is_mask,
        day_of_bar=day_of_bar,
        daily_index=daily_index,
        stable_daily_factor=factor,
        stable_per_dollar=stable_value_per_dollar(factor),
        n_bars=len(index),
    )
    return cache, symbols


def summary() -> str:
    cache, symbols = build_wide_cache()
    lines = [
        f"확장 유니버스: {len(symbols)}종목 · {cache.n_bars:,}봉 · "
        f"{cache.index[0].date()} ~ {cache.index[-1].date()}",
        f"IS {int(cache.is_mask.sum()):,}봉 / OOS {int((~cache.is_mask).sum()):,}봉 "
        f"(분할 {OOS_SPLIT.date()})",
        "",
        f"{'심볼':>12} {'거래대금(일)':>14} {'편도비용':>10} {'거래가능봉':>11} {'평균펀딩APR':>12}",
    ]
    for symbol in symbols:
        arrays = cache.arrays[symbol]
        funding = arrays.funding_at_bar
        apr = float(funding[funding != 0.0].mean() * 3 * 365) if (funding != 0.0).any() else 0.0
        volume = float(np.nanmedian(np.where(arrays.tradable, arrays.close, np.nan)) * 0.0)  # placeholder
        lines.append(
            f"{symbol:>12} {'':>14} {arrays.cost_rate*1e4:9.3f}bp {int(arrays.tradable.sum()):11,} {apr:11.2%}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
