#!/usr/bin/env python3
# Wave-38 data layer: delta-neutral carry panel from a SINGLE venue.
#
# Every series here (spot OHLC, perp OHLC, funding) is read from research/wave3/cache, i.e. Binance
# only. wave36 was invalidated precisely because it ranked symbols on one venue's funding and filled
# at another's prices, so sharing one venue is a structural requirement of this wave, not a
# convenience.
#
# Why not reuse wave13/wave18's own loader (research.wave13_liquidity.universe_liquidity), which
# already builds exactly this kind of panel for L4/I5: its cache chain
# (wave12_frontier/cache -> wave11_yield/cache -> wave1/cache) is missing files its own universe
# needs -- binance_fapi_AAVEUSDT_1d.csv.gz among them -- so L4/I5 cannot currently be re-run at all.
# Backfilling that chain from wave3/cache would change which symbols L4's top-200 breadth rule
# selects, which would make the published I5 result non-reproducible. The validated artifact is left
# untouched and this wave builds independently, the same choice wave37 made.
#
# What IS reused, deliberately and without modification: fam_funding.funding_score /
# carry_position for the signal (so "15% APR" means here exactly what it means for L4), and
# costs_measured for the cost model (wave13's lesson was that cost models must be measured, never
# assumed).

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
import pandas as pd

from research.wave1.fam_funding import carry_position, funding_score

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
CACHE: Final = REPO_ROOT / "research" / "wave3" / "cache"

WINDOW_DAYS: Final = 7  # L4's own funding window, unchanged
THRESHOLD_APR: Final = 0.15  # L4's own entry bar, unchanged (I3 lowered it to 0.08 and lost)
MIN_LIQUIDITY_USDT: Final = 5_000_000.0  # mean daily quote volume required to be tradable that day
MIN_HISTORY_DAYS: Final = 365
# A symbol is not tradable as a delta-neutral pair until BOTH legs have been listed a while. Spot and perp
# list on different dates -- WIFUSDT's perp began 2024-01-18 and its spot only 2024-03-05 -- and on the
# later leg's first day its price discovery is not a basis a hedged book could have captured. WIFUSDT's
# spot opened at 1.1892 and closed at 1.5470 (+30%) while its perp moved +4.8%, which this engine booked
# as a 25.26% basis gain on a single day. The pattern is systemic, not one symbol: of 464 days with
# |basis| > 2%, 52.6% fall within five days of a symbol's first jointly-valid observation, and the share
# plateaus by day 10. Thirty days requires a month of both legs trading before the pair is eligible, which
# removes the artifact without discarding genuine later dislocations.
MIN_LISTING_AGE_DAYS: Final = 30
# Even after the listing-age guard, nine seasoned symbol-days show |daily basis| above 10% -- SOLUSDT on
# 2022-11-10 (FTX collapse) has a perp open of 11.69 against a spot open of 14.08. Those are real
# dislocations rather than bad rows, but a daily bar's "open" is each product's FIRST TRADE, and on such a
# day the two first trades are seconds apart amid violent movement. Treating a 17% gap between them as a
# spread a hedged book could have captured simultaneously is the same assumption that got I4 rejected in
# wave18: an execution that only exists in calm arithmetic. The pair's daily basis has p99 = 0.463%, so
# this limit sits roughly 4x above the 99th percentile. Excluding these days can only REMOVE profit the
# engine would otherwise book, never add any, so the filter is conservative by construction. Sensitivity
# to it is reported rather than assumed -- see research/wave44_correction.
BASIS_SANITY_LIMIT: Final = 0.02


@dataclass(frozen=True, slots=True)
class CarryPanel:
    """Aligned per-day matrices, all shaped (n_days, n_symbols) unless noted.

    Rows are days, columns are symbols, in the order given by `symbols`. NaN means the symbol had no
    usable data that day and is excluded from selection by `tradable`.
    """

    days: pd.DatetimeIndex
    symbols: tuple[str, ...]
    spot_open: np.ndarray
    spot_close: np.ndarray
    perp_open: np.ndarray
    perp_close: np.ndarray
    perp_high: np.ndarray  # needed to test whether a levered short perp leg would be liquidated intraday
    funding_daily: np.ndarray  # summed 8h funding rates for the day (income to a short perp when > 0)
    ranking_apr: np.ndarray  # shift(1)'d 7d funding APR -- the causal ranking metric
    raw_apr: np.ndarray  # unshifted 7d funding APR, kept so a different entry threshold can be applied
                         # without re-reading 271 symbols from disk (carry_position does its own shift)
    active: np.ndarray  # carry_position hysteresis output, already shift(1)'d internally
    tradable: np.ndarray  # bool: all four prices present AND liquidity floor met
    cost_rate: np.ndarray  # per-symbol round-trip-leg cost rate for the day
    quote_volume: np.ndarray


def _read_ohlc(prefix: str, symbol: str) -> pd.DataFrame | None:
    path = CACHE / f"binance_{prefix}_{symbol}_1d.csv.gz"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp").set_index("timestamp")
    wanted = ["open", "close"]
    for optional in ("high", "low", "quote_volume"):
        if optional in frame.columns:
            wanted.append(optional)
    return frame[wanted].astype(float)


def _read_funding(symbol: str) -> pd.Series | None:
    path = CACHE / f"binance_funding_{symbol}.csv.gz"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    series = frame.set_index("timestamp")["funding_rate"].astype(float)
    return series[~series.index.duplicated(keep="first")].sort_index()


def _eligible_symbols() -> list[str]:
    """Symbols with all three legs on disk and enough history for the signal to exist.

    Carry needs a spot leg to be delta-neutral, so a perp-only symbol is not a carry candidate no
    matter how attractive its funding is.
    """
    symbols = []
    for funding_path in sorted(CACHE.glob("binance_funding_*.csv.gz")):
        symbol = funding_path.name[len("binance_funding_") : -len(".csv.gz")]
        if not (CACHE / f"binance_spot_{symbol}_1d.csv.gz").exists():
            continue
        if not (CACHE / f"binance_fapi_{symbol}_1d.csv.gz").exists():
            continue
        symbols.append(symbol)
    return symbols


@lru_cache(maxsize=1)
def _cost_mapping():
    from research.wave13_liquidity import costs_measured

    return costs_measured.fit_mapping()


def build_cost_and_liquidity(
    quote_volume_frame: pd.DataFrame, symbols: tuple[str, ...], stress_multiplier: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Cost rates and the liquidity mask, both from wave13's validated helpers.

    An earlier version of this module computed the cost rate by hand as
    `taker_fee + slippage_bp/10_000`, which was wrong in two ways that both flattered the strategy:
    the validated convention (costs_measured.cost_rate_from_bp) charges the MAKER fee and, critically,
    multiplies BOTH fee and slippage by 2.0 because a delta-neutral carry trades two legs -- spot and
    perp -- on every rebalance. Hand-rolling the formula understated turnover cost by roughly half,
    which is exactly the mistake wave13 was run to eliminate ("measure the cost model, never assume
    it"), so the validated function is now called directly and no local formula exists.

    The liquidity average likewise comes from costs_measured.point_in_time_known_avg, which shifts by
    one day. The earlier plain rolling mean included the current day's own volume, letting today's
    tradability decision peek at today's data.
    """
    from research.wave13_liquidity import costs_measured

    mapping = _cost_mapping()
    cost_frame = costs_measured.build_cost_rate_frame(
        quote_volume_frame, symbols, mapping, stress_multiplier
    )
    known_avg = costs_measured.point_in_time_known_avg(quote_volume_frame).reindex(columns=list(symbols))
    liquid = (known_avg >= MIN_LIQUIDITY_USDT).to_numpy(dtype=bool)
    return cost_frame.reindex(columns=list(symbols)).to_numpy(dtype=float), liquid


@lru_cache(maxsize=1)
def build_panel() -> CarryPanel:
    spot_open_cols: dict[str, pd.Series] = {}
    spot_close_cols: dict[str, pd.Series] = {}
    perp_open_cols: dict[str, pd.Series] = {}
    perp_close_cols: dict[str, pd.Series] = {}
    perp_high_cols: dict[str, pd.Series] = {}
    funding_cols: dict[str, pd.Series] = {}
    apr_cols: dict[str, pd.Series] = {}
    raw_apr_cols: dict[str, pd.Series] = {}
    active_cols: dict[str, pd.Series] = {}
    volume_cols: dict[str, pd.Series] = {}

    for symbol in _eligible_symbols():
        funding = _read_funding(symbol)
        if funding is None or len(funding) < WINDOW_DAYS * 3 + MIN_HISTORY_DAYS:
            continue
        spot = _read_ohlc("spot", symbol)
        perp = _read_ohlc("fapi", symbol)
        if spot is None or perp is None or len(spot) < MIN_HISTORY_DAYS or len(perp) < MIN_HISTORY_DAYS:
            continue

        raw_apr = funding_score(funding, WINDOW_DAYS).resample("1D").last()
        # carry_position does its own shift(1) internally; feeding it the RAW score is engine13's and
        # engine18's own convention and must not be pre-shifted here or the signal is lagged twice.
        active = carry_position(raw_apr, _ThresholdCandidate(THRESHOLD_APR))

        spot_open_cols[symbol] = spot["open"]
        spot_close_cols[symbol] = spot["close"]
        perp_open_cols[symbol] = perp["open"]
        perp_close_cols[symbol] = perp["close"]
        perp_high_cols[symbol] = perp["high"] if "high" in perp.columns else perp["close"]
        funding_cols[symbol] = funding.resample("1D").sum()
        apr_cols[symbol] = raw_apr.shift(1)  # ranking must use yesterday's known score
        raw_apr_cols[symbol] = raw_apr
        active_cols[symbol] = active
        volume_cols[symbol] = (
            perp["quote_volume"] if "quote_volume" in perp.columns else pd.Series(dtype=float)
        )

    spot_open = pd.DataFrame(spot_open_cols).sort_index()
    days = spot_open.index
    symbols = tuple(spot_open.columns)

    def align(columns: dict[str, pd.Series], fill: float | None = None) -> np.ndarray:
        frame = pd.DataFrame(columns).reindex(index=days, columns=list(symbols))
        if fill is not None:
            frame = frame.fillna(fill)
        return frame.to_numpy(dtype=float)

    spot_open_arr = spot_open.to_numpy(dtype=float)
    spot_close_arr = align(spot_close_cols)
    perp_open_arr = align(perp_open_cols)
    perp_close_arr = align(perp_close_cols)
    perp_high_arr = align(perp_high_cols)
    funding_arr = align(funding_cols, fill=0.0)
    apr_arr = align(apr_cols)
    raw_apr_arr = align(raw_apr_cols)
    active_arr = align(active_cols, fill=0.0)
    volume_frame = pd.DataFrame(volume_cols).reindex(index=days, columns=list(symbols))
    cost_arr, liquid = build_cost_and_liquidity(volume_frame, symbols)
    # Capacity checks size a leg against the volume actually knowable at decision time, so the same
    # shifted trailing average is exported rather than the raw same-day volume.
    from research.wave13_liquidity import costs_measured

    volume_arr = (
        costs_measured.point_in_time_known_avg(volume_frame)
        .reindex(columns=list(symbols))
        .to_numpy(dtype=float)
    )

    prices_present = (
        np.isfinite(spot_open_arr)
        & np.isfinite(spot_close_arr)
        & np.isfinite(perp_open_arr)
        & np.isfinite(perp_close_arr)
    )
    # Age measured from the first day BOTH legs quote, so a late-listing spot leg resets the clock.
    listing_age = np.full(prices_present.shape, -1, dtype=np.int64)
    for column in range(prices_present.shape[1]):
        valid_days = np.flatnonzero(prices_present[:, column])
        if len(valid_days):
            listing_age[valid_days, column] = valid_days - valid_days[0]
    seasoned = listing_age >= MIN_LISTING_AGE_DAYS
    with np.errstate(invalid="ignore", divide="ignore"):
        daily_basis = (spot_close_arr / spot_open_arr - 1.0) - (perp_close_arr / perp_open_arr - 1.0)
    plausible = ~(np.abs(np.nan_to_num(daily_basis, nan=0.0)) > BASIS_SANITY_LIMIT)
    tradable = prices_present & liquid & seasoned & plausible

    return CarryPanel(
        days=days,
        symbols=symbols,
        spot_open=spot_open_arr,
        spot_close=spot_close_arr,
        perp_open=perp_open_arr,
        perp_close=perp_close_arr,
        perp_high=perp_high_arr,
        funding_daily=funding_arr,
        ranking_apr=apr_arr,
        raw_apr=raw_apr_arr,
        active=active_arr,
        tradable=tradable,
        cost_rate=cost_arr,
        quote_volume=volume_arr,
    )


def with_threshold(panel: CarryPanel, threshold_apr: float) -> CarryPanel:
    """Same panel with the carry entry threshold changed, without touching the disk.

    Only `active` depends on the threshold, and it is derived from `raw_apr` by fam_funding.carry_position
    -- the identical function and hysteresis rule L4 uses, so a threshold of 0.15 here reproduces L4's own
    signal exactly. Reading 271 symbols takes ~30s, so a threshold sweep recomputes from the cached panel
    instead of rebuilding it. carry_position applies its own shift(1), which is why the UNSHIFTED raw_apr
    is the correct input; passing the pre-shifted ranking_apr would lag the signal twice.
    """
    import dataclasses

    raw = pd.DataFrame(panel.raw_apr, index=panel.days, columns=list(panel.symbols))
    active = np.column_stack(
        [carry_position(raw[symbol], _ThresholdCandidate(threshold_apr)).to_numpy() for symbol in panel.symbols]
    )
    return dataclasses.replace(panel, active=np.nan_to_num(active, nan=0.0))


@dataclass(frozen=True, slots=True)
class _ThresholdCandidate:
    """Minimal stand-in for fam_funding.FundingCandidate.

    carry_position only reads `.threshold_apr`, and constructing the real FundingCandidate here would
    require inventing a candidate_id/window/top_k that this module has no use for and that could be
    mistaken for a wave38 strategy parameter.
    """

    threshold_apr: float


def main() -> int:
    panel = build_panel()
    print(f"=== wave38 패널 (Binance 단일, research/wave3/cache) ===")
    print(f"종목 {len(panel.symbols)} · 일수 {len(panel.days)}")
    print(f"기간 {panel.days[0].date()} ~ {panel.days[-1].date()}")
    tradable_per_day = panel.tradable.sum(axis=1)
    print(f"거래가능 종목수/일: 중앙 {np.median(tradable_per_day):.0f} · 최대 {tradable_per_day.max()}")
    qualifying = (panel.active > 0.0) & panel.tradable
    counts = qualifying.sum(axis=1)
    print(f"자격(15% APR 통과 + 거래가능) 종목수/일: 중앙 {np.median(counts):.0f} · 평균 {counts.mean():.2f}")
    print(f"  1개 이상 {float((counts >= 1).mean()):.1%} · 2개 이상 {float((counts >= 2).mean()):.1%} "
          f"· 3개 이상 {float((counts >= 3).mean()):.1%} · 5개 이상 {float((counts >= 5).mean()):.1%}")
    print(f"비용률: 중앙 {np.nanmedian(panel.cost_rate):.5f} · 최소 {np.nanmin(panel.cost_rate):.5f} "
          f"· 최대 {np.nanmax(panel.cost_rate):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
