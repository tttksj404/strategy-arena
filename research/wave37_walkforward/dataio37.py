# Wave-37 data layer: Binance-only daily panel.
#
# The whole point of this module is what it does NOT do: it never reads a second venue. wave36 died
# because its signal came from Binance funding while its fills came from Bitget prices, and the two
# venues' funding rankings agree only about half the time (measured: cross-sectional Spearman 0.43,
# top/bottom-k set overlap 50%). Here signal, funding and price all come from research/wave3/cache,
# which is Binance. Gate Y1 asserts that property against this file.
#
# Daily bars rather than hourly is also deliberate: wave35 measured cost against typical bar range
# and daily is the most forgiving timeframe by a wide margin (1m 2.33x, 1H 0.20x, 1D 0.03x), and the
# Binance hourly cache covers only three symbols anyway.

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
BINANCE_CACHE: Final = REPO_ROOT / "research" / "wave3" / "cache"
I5_RESULTS: Final = REPO_ROOT / "research" / "wave18_idle" / "results" / "I5.json"

START: Final = pd.Timestamp("2022-01-01", tz="UTC")
MIN_FUNDING_STAMPS: Final = 1_000
MIN_DAILY_QUOTE_VOLUME: Final = 5_000_000.0  # point-in-time liquidity filter (30d mean)
VOLUME_WINDOW: Final = 30
TAKER_FEE: Final = 0.0006


@dataclass(frozen=True)
class DailyPanel:
    days: pd.DatetimeIndex
    symbols: tuple[str, ...]
    open: np.ndarray  # (n_days, n_symbols) fill price for decisions made on the previous close
    close: np.ndarray
    funding_daily: np.ndarray  # sum of the day's 8h stamps
    volume_ok: np.ndarray  # bool, point-in-time 30d mean quote volume >= threshold
    tradable: np.ndarray
    cost_rate: np.ndarray  # (n_symbols,) one-way
    stable_per_dollar: np.ndarray  # value of $1 in the stable sleeve, per day


def _read_klines(symbol: str) -> pd.DataFrame | None:
    path = BINANCE_CACHE / f"binance_fapi_{symbol}_1d.csv.gz"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp").set_index("timestamp")
    return frame[["open", "high", "low", "close", "quote_volume"]].astype(float)


def _read_funding(symbol: str) -> pd.Series | None:
    path = BINANCE_CACHE / f"binance_funding_{symbol}.csv.gz"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    series = frame.set_index("timestamp")["funding_rate"].astype(float)
    return series[~series.index.duplicated(keep="first")].sort_index()


def _cost_rate(mean_daily_quote_volume: float) -> float:
    from research.wave13_liquidity import costs_measured

    mapping = costs_measured.fit_mapping()
    return TAKER_FEE + costs_measured.slippage_bp_for_volume(mean_daily_quote_volume, mapping) / 10_000.0


def _stable_per_dollar(days: pd.DatetimeIndex) -> np.ndarray:
    import json

    payload = json.loads(I5_RESULTS.read_text(encoding="utf-8"))
    equity = pd.Series(
        [float(item["value"]) for item in payload["equity"]],
        index=pd.to_datetime([item["timestamp"] for item in payload["equity"]], utc=True),
    ).sort_index()
    factor = (equity / equity.shift(1)).dropna().reindex(days).fillna(1.0)
    return (0.90 * np.cumprod(factor.to_numpy(dtype=float)) + 0.10)


@lru_cache(maxsize=1)
def build_daily_panel() -> DailyPanel:
    candidates: list[str] = []
    for path in sorted(BINANCE_CACHE.glob("binance_fapi_*_1d.csv.gz")):
        symbol = path.name.replace("binance_fapi_", "").replace("_1d.csv.gz", "")
        funding = _read_funding(symbol)
        if funding is None:
            continue
        if int((funding.index >= START).sum()) < MIN_FUNDING_STAMPS:
            continue
        candidates.append(symbol)

    klines = {}
    fundings = {}
    for symbol in candidates:
        frame = _read_klines(symbol)
        if frame is None or frame.empty:
            continue
        frame = frame.loc[frame.index >= START]
        if len(frame) < 400:
            continue
        klines[symbol] = frame
        fundings[symbol] = _read_funding(symbol)

    symbols = tuple(sorted(klines))
    days = None
    for frame in klines.values():
        days = frame.index if days is None else days.union(frame.index)
    days = pd.DatetimeIndex(days).sort_values()

    n_days, n_symbols = len(days), len(symbols)
    open_a = np.full((n_days, n_symbols), np.nan)
    close_a = np.full((n_days, n_symbols), np.nan)
    funding_a = np.zeros((n_days, n_symbols))
    volume_ok = np.zeros((n_days, n_symbols), dtype=bool)
    cost = np.zeros(n_symbols)

    for index, symbol in enumerate(symbols):
        aligned = klines[symbol].reindex(days)
        open_a[:, index] = aligned["open"].to_numpy(dtype=float)
        close_a[:, index] = aligned["close"].to_numpy(dtype=float)
        # Point-in-time liquidity: trailing 30d mean quote volume through YESTERDAY.
        rolling = aligned["quote_volume"].rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean().shift(1)
        volume_ok[:, index] = (rolling >= MIN_DAILY_QUOTE_VOLUME).fillna(False).to_numpy()
        cost[index] = _cost_rate(float(np.nanmedian(aligned["quote_volume"].to_numpy(dtype=float))))

        series = fundings[symbol]
        daily = series.groupby(series.index.floor("1D")).sum()
        funding_a[:, index] = daily.reindex(days).fillna(0.0).to_numpy(dtype=float)

    tradable = np.isfinite(open_a) & np.isfinite(close_a)
    return DailyPanel(
        days=days,
        symbols=symbols,
        open=open_a,
        close=close_a,
        funding_daily=funding_a,
        volume_ok=volume_ok & tradable,
        tradable=tradable,
        cost_rate=cost,
        stable_per_dollar=_stable_per_dollar(days),
    )


def summary() -> str:
    panel = build_daily_panel()
    eligible_per_day = panel.volume_ok.sum(axis=1)
    return "\n".join(
        [
            f"Binance 단일출처 일봉 패널: {len(panel.symbols)}종목 x {len(panel.days)}일 "
            f"({panel.days[0].date()} ~ {panel.days[-1].date()})",
            f"유동성 필터(30일 평균 거래대금 >= ${MIN_DAILY_QUOTE_VOLUME/1e6:.0f}M) 통과 종목수: "
            f"중앙 {int(np.median(eligible_per_day))} · 최소 {int(eligible_per_day.min())} · 최대 {int(eligible_per_day.max())}",
            f"편도비용: 최소 {panel.cost_rate.min()*1e4:.3f}bp · 중앙 {np.median(panel.cost_rate)*1e4:.3f}bp "
            f"· 최대 {panel.cost_rate.max()*1e4:.3f}bp",
            f"일별 펀딩 합계 중앙 APR: {np.median(panel.funding_daily[panel.funding_daily!=0])*365*100:.2f}%",
        ]
    )


if __name__ == "__main__":
    print(summary())
