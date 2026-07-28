# Wave-23 GA fitness engine: short-term-profit-maximization backtest for 5 strategy kinds
# sharing ONE position-lifecycle simulator.
#
# ---------------------------------------------------------------------------------------
# Why one shared lifecycle simulator instead of 5 bespoke engines
# ---------------------------------------------------------------------------------------
# SPEC.md's gene table (entry_z, holding_days, position_fraction, stop_loss_pct,
# take_profit_pct, universe_breadth, max_concurrent) applies UNIFORMLY across all 5
# strategy_kind values -- only the SIGNAL that decides when/which-direction to enter differs
# per kind. So this module factors every kind down to a common shape:
#   KindSignal.z            -- how extreme today's reading is vs THIS SYMBOL'S OWN recent
#                               history (a rolling z-score, ROLLING_Z_WINDOW/MIN_PERIODS days),
#                               already shift(1)'d so entry_z comparisons never look ahead.
#   KindSignal.direction     -- +1/-1/0, the position direction entry would take, same shift.
#   KindSignal.gap_return / .intraday_return -- the LONG-side day return (direction-agnostic,
#                               NOT shifted -- this is the day's own realized economics once a
#                               position is already open), used by _simulate_lifecycle with a
#                               +-1 sign applied per the position's OWN carried-forward
#                               direction (set once at entry, independent of what the signal
#                               does on later days -- see _simulate_lifecycle's docstring).
# All 4 directional kinds (momentum/breakout/funding_spike/convex_dual) trade the SAME
# instrument (perp, no offsetting spot leg) so they share ONE gap/intraday formula
# (_directional_gap_intraday); only carry keeps wave21_ga's own spot+perp delta-neutral
# formula (_carry_gap_intraday, copied line-for-line from research.wave21_ga.fitness /
# research.wave13_liquidity.engine13's per-day economics). Every per-genome evaluation then
# reduces to: (a) threshold z against entry_z to get today's entry-eligible symbols, (b) run
# _simulate_lifecycle (the one part that is a genuine day-by-day state machine -- a position's
# stop-loss/take-profit/holding-day exit is path-dependent, so unlike wave21_ga's pure
# hysteresis this cannot be collapsed to a lookahead-free vectorized mask-then-ffill), (c)
# compound the resulting weights into an equity curve exactly like wave21_ga's own
# _compound_factor. Everything genome-INDEPENDENT (z/direction/gap/intraday arrays for all 5
# kinds, liquidity/cost frames) is computed ONCE in build_market_cache and reused across every
# evaluation -- same "expensive precompute once, cheap per-eval" shape wave21_ga's fitness.py
# established, needed for the same reason (this wave's GA+random budget is
# 60*25*5*2 = 15,000 backtests).
#
# ---------------------------------------------------------------------------------------
# Leverage: 1x fixed BY CONSTRUCTION, not by post-hoc gate
# ---------------------------------------------------------------------------------------
# genome23.Genome.normalized_weight divides position_fraction down whenever
# position_fraction * max_concurrent would exceed 1.0 gross -- see that property's own
# docstring for why this wave deliberately does NOT repeat wave21_ga's H4-gate-catches-it-after
# -the-fact design (which let the GA spend its whole budget on an infeasible corner, then
# required a manual post-hoc genome edit whose own DSR was never gated -- wave22_overfit's G1).
# _simulate_lifecycle and run_backtest below use genome.normalized_weight EVERYWHERE a position
# size is needed, so gross exposure never exceeds ACTIVE_CAPITAL for ANY genome this module
# ever backtests.
#
# ---------------------------------------------------------------------------------------
# OOS sealing (SPEC.md 오염 차단 1, same structural pattern as research.wave21_ga.fitness)
# ---------------------------------------------------------------------------------------
# evaluate_genome() (fitness23.py) calls ONLY run_backtest(genome, cache, mode=MODE_IS) --
# hardcoded, no caller-supplied mode -- so ga23.py/random_search23.py have no code path capable
# of requesting OOS data even by mistake. run_backtest(mode=MODE_IS) slices every array to the
# CONTIGUOUS IS PREFIX (cache.is_row_mask, asserted contiguous at build time) before any
# signal/lifecycle/compounding computation touches it, and re-asserts (raising OOSLeakageError)
# that the resulting equity index never exceeds OOS_SPLIT. oos_slice() is the only function in
# this package allowed to read OOS-range data, and only under mode=MODE_OOS_FINAL.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.fam_funding import FundingCandidate, FundingMarket, funding_score
from research.wave10_carry100.engine import ACTIVE_CAPITAL, OOS_SPLIT
from research.wave13_liquidity import costs_measured, engine13
from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.configs13 import Wave13Config
from research.wave20_convex.engine20 import atr as _atr
from research.wave23_ga_short import genome23
from research.wave23_ga_short.genome23 import Genome

MODE_IS: Final = "IS"
MODE_OOS_FINAL: Final = "OOS_FINAL"
VALID_MODES: Final[tuple[str, ...]] = (MODE_IS, MODE_OOS_FINAL)

MAKER_FEE_RATE_ONE_LEG: Final = 0.0002  # matches wave20_convex.configs20.MAKER_FEE_RATE / wave2 W2_MAKER_FEE_RATE (single-leg maker fee)

ROLLING_Z_WINDOW: Final = 180  # trailing window a raw signal is standardized against -- "how extreme vs its own recent history"
ROLLING_Z_MIN_PERIODS: Final = 60  # a symbol needs >=60 trailing days before it can generate any signal at all
MOMENTUM_LOOKBACK_DAYS: Final = 30  # matches research.wave3.engine._momentum (close.pct_change(30))
BREAKOUT_SMA_DAYS: Final = 20
BREAKOUT_ATR_DAYS: Final = 14  # matches wave20_convex.configs20.V1Config.atr_window_days
CARRY_WINDOW_DAYS: Final = 7  # matches wave21_ga/wave18_idle's own L4 window
FUNDING_SPIKE_WINDOW_DAYS: Final = 3  # matches wave20_convex.configs20.V2Config.funding_window_days ("acute squeeze", not a persistent regime)
_MIN_CUTOFF_DAYS: Final = 120

_MAX_BREADTH: Final = max(genome23.UNIVERSE_BREADTH_CHOICES)  # 200
_UNIVERSE_CONFIG: Final = Wave13Config(
    FundingCandidate("W23_UNIVERSE", 7, 0.15, 1),  # placeholder candidate -- universe_liquidity only reads universe_kind/breadth/history_months off this config (see wave21_ga.fitness._GA_UNIVERSE_CONFIG's identical precedent)
    0.50,
    "breadth",
    None,
    _MAX_BREADTH,
    12.0,
    None,
    None,
    "wave23_ga_short superset universe loader (breadth=200, 12mo) -- every genome's own (smaller) universe_breadth is a column SLICE of this.",
)


class OOSLeakageError(Exception):
    """Raised the moment OOS-range (> OOS_SPLIT) data is touched outside mode=MODE_OOS_FINAL."""


@dataclass(frozen=True, slots=True)
class KindSignal:
    z: np.ndarray  # [days, symbols] float, shift(1)'d -- usable for a decision made ON day t
    direction: np.ndarray  # [days, symbols] int8 in {-1, 0, +1}, shift(1)'d in lockstep with z
    gap_return: np.ndarray  # [days, symbols] float, LONG-side, NOT shifted (day t's own realized economics)
    intraday_return: np.ndarray  # [days, symbols] float, LONG-side, NOT shifted


@dataclass(frozen=True, slots=True)
class MarketCache:
    symbols: tuple[str, ...]
    index: pd.DatetimeIndex
    is_row_mask: np.ndarray
    available: np.ndarray  # bool[days, symbols]
    liquidity_ok: np.ndarray  # bool[days, symbols]
    breadth_masks: dict[int, np.ndarray]  # universe_breadth -> bool[symbols]
    cost_rate_pair: np.ndarray  # float[days, symbols] -- 2-leg (spot+perp) convention, carry only
    cost_rate_pair_stress: np.ndarray
    cost_rate_one_leg: np.ndarray  # float[days, symbols] -- 1-leg convention, the 4 directional kinds
    cost_rate_one_leg_stress: np.ndarray
    signals: dict[str, KindSignal]  # strategy_kind -> KindSignal


# ---------------------------------------------------------------------------
# Signal construction (genome-independent -- computed once per cache build).
# ---------------------------------------------------------------------------


def _rolling_zscore(raw: pd.DataFrame, window: int = ROLLING_Z_WINDOW, min_periods: int = ROLLING_Z_MIN_PERIODS) -> pd.DataFrame:
    mean = raw.rolling(window, min_periods=min_periods).mean()
    std = raw.rolling(window, min_periods=min_periods).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (raw - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _z_and_direction(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """raw -> (z, direction), both shift(1)'d so a value at row t is computable from data
    through row t-1 only (matches the repo-wide `.shift(1)` lag-before-use convention, e.g.
    wave21_ga.fitness's `ranking = raw.shift(1)`)."""
    z_lagged = _rolling_zscore(raw).shift(1)
    z_arr = z_lagged.to_numpy(dtype=float)
    finite = np.isfinite(z_arr)
    direction_arr = np.where(finite, np.sign(z_arr), 0.0).astype(np.int8)
    return z_arr, direction_arr


def _directional_gap_intraday(perp_open: pd.DataFrame, perp_close: pd.DataFrame, funding: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """LONG-side day return for a single-leg perp position (shared by momentum/breakout/
    funding_spike/convex_dual -- all 4 trade the identical instrument, only their entry
    SIGNAL differs). A short position's realized return is the negation of this, applied by
    _simulate_lifecycle via the position's own carried-forward direction -- see module
    docstring."""
    perp_close_prev = perp_close.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = perp_open / perp_close_prev - 1.0
        intraday = perp_close / perp_open - 1.0
    gap_arr = np.nan_to_num(gap.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    intraday_arr = np.nan_to_num((intraday - funding).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return gap_arr, intraday_arr


def _carry_gap_intraday(
    spot_open: pd.DataFrame, spot_close: pd.DataFrame, perp_open: pd.DataFrame, perp_close: pd.DataFrame, funding: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Delta-neutral spot-long/perp-short pair economics -- copied formula-for-formula from
    research.wave21_ga.fitness._compound_factor / research.wave13_liquidity.engine13's own
    per-day loop (see that module's docstring for the engine13-equivalence test this formula
    rests on)."""
    spot_close_prev = spot_close.shift(1)
    perp_close_prev = perp_close.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        spot_gap = spot_open / spot_close_prev - 1.0
        perp_gap = perp_open / perp_close_prev - 1.0
        intraday = spot_close / spot_open - perp_close / perp_open
    gap_arr = np.nan_to_num((spot_gap - perp_gap).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    intraday_arr = np.nan_to_num((intraday + funding).to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return gap_arr, intraday_arr


def _build_signals(
    symbols: tuple[str, ...],
    index: pd.DatetimeIndex,
    spot_open: pd.DataFrame,
    spot_close: pd.DataFrame,
    perp_open: pd.DataFrame,
    perp_high: pd.DataFrame,
    perp_low: pd.DataFrame,
    perp_close: pd.DataFrame,
    funding: pd.DataFrame,
) -> dict[str, KindSignal]:
    carry_raw = pd.DataFrame({s: funding_score(_symbol_funding_series(funding, s), CARRY_WINDOW_DAYS) for s in symbols}, index=index)
    spike_raw = pd.DataFrame({s: funding_score(_symbol_funding_series(funding, s), FUNDING_SPIKE_WINDOW_DAYS) for s in symbols}, index=index)
    momentum_raw = perp_close.pct_change(MOMENTUM_LOOKBACK_DAYS).replace([np.inf, -np.inf], np.nan)

    sma20 = perp_close.rolling(BREAKOUT_SMA_DAYS, min_periods=BREAKOUT_SMA_DAYS).mean()
    atr14 = pd.DataFrame({s: _atr(pd.DataFrame({"high": perp_high[s], "low": perp_low[s], "close": perp_close[s]}), BREAKOUT_ATR_DAYS) for s in symbols}, index=index)
    with np.errstate(divide="ignore", invalid="ignore"):
        breakout_raw = (perp_close - sma20) / atr14
    breakout_raw = breakout_raw.replace([np.inf, -np.inf], np.nan)

    carry_z, carry_dir = _z_and_direction(carry_raw)
    spike_z, spike_dir = _z_and_direction(spike_raw)
    momentum_z, momentum_dir = _z_and_direction(momentum_raw)
    breakout_z, breakout_dir = _z_and_direction(breakout_raw)

    agree = np.isfinite(momentum_z) & np.isfinite(breakout_z) & (np.sign(momentum_z) == np.sign(breakout_z)) & (momentum_dir != 0)
    convex_z = np.where(agree, (momentum_z + breakout_z) / 2.0, np.nan)
    convex_dir = np.where(agree, momentum_dir, 0).astype(np.int8)

    directional_gap, directional_intraday = _directional_gap_intraday(perp_open, perp_close, funding)
    carry_gap, carry_intraday = _carry_gap_intraday(spot_open, spot_close, perp_open, perp_close, funding)

    return {
        genome23.STRATEGY_KIND_CARRY: KindSignal(carry_z, carry_dir, carry_gap, carry_intraday),
        genome23.STRATEGY_KIND_MOMENTUM: KindSignal(momentum_z, momentum_dir, directional_gap, directional_intraday),
        genome23.STRATEGY_KIND_BREAKOUT: KindSignal(breakout_z, breakout_dir, directional_gap, directional_intraday),
        genome23.STRATEGY_KIND_FUNDING_SPIKE: KindSignal(spike_z, spike_dir, directional_gap, directional_intraday),
        genome23.STRATEGY_KIND_CONVEX_DUAL: KindSignal(convex_z, convex_dir, directional_gap, directional_intraday),
    }


def _symbol_funding_series(funding_frame: pd.DataFrame, symbol: str) -> pd.Series:
    return funding_frame[symbol]


# ---------------------------------------------------------------------------
# Market cache: real-cache loader + a synthetic (test-only) constructor.
# ---------------------------------------------------------------------------


def _one_leg_cost_frame(quote_volume: pd.DataFrame, mapping: costs_measured.MeasuredCostMapping, stress_multiplier: float) -> pd.DataFrame:
    known_avg = costs_measured.point_in_time_known_avg(quote_volume)
    bp_frame = costs_measured.bp_frame_from_known_avg(known_avg, mapping)
    return MAKER_FEE_RATE_ONE_LEG + bp_frame * 0.0001 * stress_multiplier


def build_market_cache() -> MarketCache:
    pool = ul.load_candidate_pool()
    symbols = ul.verify_cache_and_load_symbols(_UNIVERSE_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    missing = [s for s in symbols if s not in markets]
    if missing:
        raise RuntimeError(f"wave23_ga_short market cache: {len(missing)} symbols missing from load_markets_for_symbols: {missing[:5]}")

    spot_open_cols: dict[str, pd.Series] = {}
    spot_close_cols: dict[str, pd.Series] = {}
    perp_open_cols: dict[str, pd.Series] = {}
    perp_high_cols: dict[str, pd.Series] = {}
    perp_low_cols: dict[str, pd.Series] = {}
    perp_close_cols: dict[str, pd.Series] = {}
    funding_cols: dict[str, pd.Series] = {}
    for symbol in symbols:
        market = markets[symbol]
        funding_daily = market.funding.resample("1D").sum()
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        spot_open_cols[symbol] = spot_daily["open"]
        spot_close_cols[symbol] = spot_daily["close"]
        perp_open_cols[symbol] = perp_daily["open"]
        perp_high_cols[symbol] = perp_daily["high"]
        perp_low_cols[symbol] = perp_daily["low"]
        perp_close_cols[symbol] = perp_daily["close"]
        funding_cols[symbol] = funding_daily

    ordered = list(symbols)
    spot_open_frame = pd.DataFrame(spot_open_cols)[ordered].sort_index()
    index = spot_open_frame.index
    spot_close_frame = pd.DataFrame(spot_close_cols)[ordered].reindex(index)
    perp_open_frame = pd.DataFrame(perp_open_cols)[ordered].reindex(index)
    perp_high_frame = pd.DataFrame(perp_high_cols)[ordered].reindex(index)
    perp_low_frame = pd.DataFrame(perp_low_cols)[ordered].reindex(index)
    perp_close_frame = pd.DataFrame(perp_close_cols)[ordered].reindex(index)
    funding_frame = pd.DataFrame(funding_cols)[ordered].reindex(index).fillna(0.0)

    available = (
        spot_open_frame.notna().to_numpy()
        & spot_close_frame.notna().to_numpy()
        & perp_open_frame.notna().to_numpy()
        & perp_close_frame.notna().to_numpy()
    )

    mapping = costs_measured.fit_mapping()
    cost_rate_pair_frame, liquidity_ok_frame = engine13.build_cost_and_liquidity_frames(_UNIVERSE_CONFIG, symbols, index, mapping, engine13.DEFAULT_STRESS_MULTIPLIER)
    cost_rate_pair_stress_frame, _ = engine13.build_cost_and_liquidity_frames(_UNIVERSE_CONFIG, symbols, index, mapping, engine13.STRESS_MULTIPLIER)

    quote_volume_frame = ul.load_quote_volume_frame(symbols)[ordered].reindex(index)
    cost_rate_one_leg_frame = _one_leg_cost_frame(quote_volume_frame, mapping, engine13.DEFAULT_STRESS_MULTIPLIER)
    cost_rate_one_leg_stress_frame = _one_leg_cost_frame(quote_volume_frame, mapping, engine13.STRESS_MULTIPLIER)

    breadth_masks: dict[int, np.ndarray] = {}
    for breadth in genome23.UNIVERSE_BREADTH_CHOICES:
        mask = np.zeros(len(symbols), dtype=bool)
        mask[: min(breadth, len(symbols))] = True
        breadth_masks[breadth] = mask

    signals = _build_signals(ordered, index, spot_open_frame, spot_close_frame, perp_open_frame, perp_high_frame, perp_low_frame, perp_close_frame, funding_frame)

    is_row_mask = np.asarray(index <= OOS_SPLIT, dtype=bool)
    prefix_length = int(is_row_mask.sum())
    if prefix_length > 0 and not bool(is_row_mask[:prefix_length].all()):
        raise RuntimeError("wave23_ga_short market cache: IS row mask is not a contiguous prefix of `index`")

    _ = pool
    return MarketCache(
        symbols=tuple(ordered),
        index=index,
        is_row_mask=is_row_mask,
        available=available,
        liquidity_ok=liquidity_ok_frame.to_numpy(dtype=bool),
        breadth_masks=breadth_masks,
        cost_rate_pair=cost_rate_pair_frame.to_numpy(dtype=float),
        cost_rate_pair_stress=cost_rate_pair_stress_frame.to_numpy(dtype=float),
        cost_rate_one_leg=cost_rate_one_leg_frame.to_numpy(dtype=float),
        cost_rate_one_leg_stress=cost_rate_one_leg_stress_frame.to_numpy(dtype=float),
        signals=signals,
    )


def market_cache_from_markets(
    markets: dict[str, FundingMarket],
    *,
    flat_cost_rate: float = 0.001,
    always_liquid: bool = True,
) -> MarketCache:
    """Test-only constructor: builds a MarketCache from an in-memory `markets` dict, no disk
    I/O, flat constant one-leg/pair cost rates and an always-liquid mask -- mirrors
    wave21_ga.fitness.market_cache_from_markets's own synthetic-fixture convention."""
    symbols = tuple(markets.keys())
    spot_open_cols: dict[str, pd.Series] = {}
    spot_close_cols: dict[str, pd.Series] = {}
    perp_open_cols: dict[str, pd.Series] = {}
    perp_high_cols: dict[str, pd.Series] = {}
    perp_low_cols: dict[str, pd.Series] = {}
    perp_close_cols: dict[str, pd.Series] = {}
    funding_cols: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        funding_daily = market.funding.resample("1D").sum()
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_source = market.perp.copy()
        if "high" not in perp_source.columns:
            perp_source["high"] = perp_source[["open", "close"]].max(axis=1)
        if "low" not in perp_source.columns:
            perp_source["low"] = perp_source[["open", "close"]].min(axis=1)
        perp_daily = perp_source.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        spot_open_cols[symbol] = spot_daily["open"]
        spot_close_cols[symbol] = spot_daily["close"]
        perp_open_cols[symbol] = perp_daily["open"]
        perp_high_cols[symbol] = perp_daily["high"]
        perp_low_cols[symbol] = perp_daily["low"]
        perp_close_cols[symbol] = perp_daily["close"]
        funding_cols[symbol] = funding_daily

    ordered = list(symbols)
    spot_open_frame = pd.DataFrame(spot_open_cols)[ordered].sort_index()
    index = spot_open_frame.index
    spot_close_frame = pd.DataFrame(spot_close_cols)[ordered].reindex(index)
    perp_open_frame = pd.DataFrame(perp_open_cols)[ordered].reindex(index)
    perp_high_frame = pd.DataFrame(perp_high_cols)[ordered].reindex(index)
    perp_low_frame = pd.DataFrame(perp_low_cols)[ordered].reindex(index)
    perp_close_frame = pd.DataFrame(perp_close_cols)[ordered].reindex(index)
    funding_frame = pd.DataFrame(funding_cols)[ordered].reindex(index).fillna(0.0)

    available = (
        spot_open_frame.notna().to_numpy()
        & spot_close_frame.notna().to_numpy()
        & perp_open_frame.notna().to_numpy()
        & perp_close_frame.notna().to_numpy()
    )
    n_days, n_symbols = available.shape
    liquidity_ok = np.full((n_days, n_symbols), always_liquid, dtype=bool)
    cost_rate_pair = np.full((n_days, n_symbols), flat_cost_rate, dtype=float)
    cost_rate_one_leg = np.full((n_days, n_symbols), flat_cost_rate / 2.0, dtype=float)
    breadth_masks = {breadth: np.ones(n_symbols, dtype=bool) for breadth in genome23.UNIVERSE_BREADTH_CHOICES}

    signals = _build_signals(ordered, index, spot_open_frame, spot_close_frame, perp_open_frame, perp_high_frame, perp_low_frame, perp_close_frame, funding_frame)
    is_row_mask = np.asarray(index <= OOS_SPLIT, dtype=bool)

    return MarketCache(
        symbols=tuple(ordered),
        index=index,
        is_row_mask=is_row_mask,
        available=available,
        liquidity_ok=liquidity_ok,
        breadth_masks=breadth_masks,
        cost_rate_pair=cost_rate_pair,
        cost_rate_pair_stress=cost_rate_pair,
        cost_rate_one_leg=cost_rate_one_leg,
        cost_rate_one_leg_stress=cost_rate_one_leg,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Position lifecycle (the one genuinely-sequential piece -- see module docstring).
# ---------------------------------------------------------------------------


def _simulate_lifecycle(
    entry_ok: np.ndarray,  # bool[days, symbols]
    direction: np.ndarray,  # int8[days, symbols], -1/0/+1
    rank_strength: np.ndarray,  # float[days, symbols], used only to break capacity ties (higher = higher priority)
    long_gap: np.ndarray,  # float[days, symbols]
    long_intraday: np.ndarray,  # float[days, symbols]
    holding_days: int,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    max_concurrent: int,
    normalized_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Day-by-day (vectorized ACROSS symbols each day) position lifecycle: a symbol enters
    when entry_ok fires and it is currently flat (subject to `max_concurrent` capacity, ties
    broken by rank_strength), and exits at the EARLIEST of stop_loss_pct / take_profit_pct
    (checked against the position's OWN cumulative return since ITS entry -- kind-agnostic:
    works identically whether that cumulative return comes from carry's spot+perp spread or a
    directional kind's perp price action) / holding_days. A position's direction is fixed at
    entry from `direction[entry_day, symbol]` and carried forward for the position's whole
    life, independent of what the raw signal does on later days (mirrors how a real position
    does not silently change side just because a new day's signal reading would have picked
    the other direction).

    Ordering within one day t (deliberately: entries decided FIRST, from ONLY the state
    carried in from yesterday, THEN one unified accrual+exit pass over everything open as of
    today, entrants included):
      1. `was_open` = state carried in from yesterday. New entries are drawn ONLY from
         `~was_open` symbols with `entry_ok[t]` -- a symbol that was open yesterday can never
         enter "fresh" today even if it happens to exit later in this same iteration (a slot a
         same-day exit frees becomes available to a NEW entry starting tomorrow, not today --
         a deliberate simplification that also means a signal cannot manufacture an unbroken
         same-day rollover chain in the SAME symbol, which would defeat SPEC.md's "보유기간
         <=14일 강제" in substance while satisfying it in the entry_day bookkeeping alone).
      2. Today's return accrues for EVERY position open today -- carried-over AND
         just-entered alike -- in one pass, and exit (stop-loss / take-profit / holding_days)
         is checked immediately after. This makes the entry day count as day 1 of the hold:
         days_held = t - entry_day is 0 on the entry day itself, so `holding_days=1` exits the
         SAME day it entered (after earning that one day's return) rather than the day after --
         checked explicitly by tests/test_wave23.py::test_holding_days_gives_exact_hold_length.
    Returns (magnitude, signed): magnitude[t,s] is normalized_weight when a position is open
    in s on day t (0 otherwise); signed[t,s] = direction * magnitude (0 when flat) --
    run_backtest turns both into an equity curve exactly like wave21_ga.fitness._compound_factor
    turns its own `weights` into one."""
    n_days, n_symbols = entry_ok.shape
    open_dir = np.zeros(n_symbols, dtype=np.int8)
    entry_day = np.full(n_symbols, -1, dtype=np.int64)
    cum_return = np.zeros(n_symbols, dtype=float)
    magnitude = np.zeros((n_days, n_symbols), dtype=float)
    signed = np.zeros((n_days, n_symbols), dtype=float)

    for t in range(n_days):
        was_open = open_dir != 0  # state carried in from yesterday

        capacity = max_concurrent - int(was_open.sum())
        if capacity > 0:
            candidates = np.where(entry_ok[t] & (~was_open))[0]
            if candidates.size > capacity:
                order = np.argsort(-rank_strength[t, candidates], kind="stable")[:capacity]
                candidates = candidates[order]
            if candidates.size:
                open_dir[candidates] = direction[t, candidates]
                entry_day[candidates] = t
                # cum_return reset to 0 here; the unified accrual pass below (idx_open includes
                # these candidates, since open_dir is now nonzero for them) fills in day t's
                # own return in one place instead of a second, duplicated formula.
                cum_return[candidates] = 0.0

        idx_open = np.where(open_dir != 0)[0]
        if idx_open.size:
            d = open_dir[idx_open].astype(float)
            day_factor = (1.0 + d * long_gap[t, idx_open]) * (1.0 + d * long_intraday[t, idx_open])
            cum_return[idx_open] = (1.0 + cum_return[idx_open]) * day_factor - 1.0
            magnitude[t, idx_open] = normalized_weight
            signed[t, idx_open] = d * normalized_weight

            days_held = t - entry_day[idx_open]  # 0 on the entry day itself
            hit_stop = (cum_return[idx_open] <= -stop_loss_pct) if stop_loss_pct is not None else np.zeros(idx_open.size, dtype=bool)
            hit_take = (cum_return[idx_open] >= take_profit_pct) if take_profit_pct is not None else np.zeros(idx_open.size, dtype=bool)
            hit_hold = days_held >= (holding_days - 1)  # today WAS the holding_days-th day of the hold -> exit now, after accruing it
            idx_exit = idx_open[hit_stop | hit_take | hit_hold]
            if idx_exit.size:
                open_dir[idx_exit] = 0
                entry_day[idx_exit] = -1
                cum_return[idx_exit] = 0.0

    return magnitude, signed


# ---------------------------------------------------------------------------
# Per-genome backtest.
# ---------------------------------------------------------------------------


def _run_backtest_components(genome: Genome, cache: MarketCache, mode: str, stress: bool) -> tuple[pd.Series, np.ndarray]:
    """Shared implementation behind run_backtest / run_backtest_with_weights -- returns
    (equity, signed) so callers that need the realized per-symbol signed weights (gates23's K5
    empirical gross-cap spot-check) do not have to re-run the lifecycle simulation a second
    time."""
    if mode not in VALID_MODES:
        raise ValueError(f"run_backtest: unknown mode {mode!r}, expected one of {VALID_MODES}")

    cutoff = int(cache.is_row_mask.sum()) if mode == MODE_IS else len(cache.index)
    if cutoff < _MIN_CUTOFF_DAYS:
        raise ValueError(f"run_backtest: sliced date range too short ({cutoff} rows) for mode={mode!r}")
    index = cache.index[:cutoff]

    sig = cache.signals[genome.strategy_kind]
    z = sig.z[:cutoff]
    direction = sig.direction[:cutoff]
    long_gap = sig.gap_return[:cutoff]
    long_intraday = sig.intraday_return[:cutoff]

    available = cache.available[:cutoff]
    liquidity_ok = cache.liquidity_ok[:cutoff]
    breadth_mask = cache.breadth_masks[genome.universe_breadth]

    raw_entry = (z >= genome.entry_z) if genome.is_long_only else (np.abs(z) >= genome.entry_z)
    entry_ok = raw_entry & available & liquidity_ok & breadth_mask[None, :] & np.isfinite(z)
    rank_strength = np.abs(z)

    magnitude, signed = _simulate_lifecycle(
        entry_ok, direction, rank_strength, long_gap, long_intraday,
        genome.holding_days, genome.stop_loss_pct, genome.take_profit_pct,
        genome.max_concurrent, genome.normalized_weight,
    )
    del magnitude

    is_carry = genome.strategy_kind == genome23.STRATEGY_KIND_CARRY
    if is_carry:
        cost_rate = (cache.cost_rate_pair_stress if stress else cache.cost_rate_pair)[:cutoff]
    else:
        cost_rate = (cache.cost_rate_one_leg_stress if stress else cache.cost_rate_one_leg)[:cutoff]

    n_symbols = len(cache.symbols)
    if cutoff > 1:
        signed_prev = np.vstack([np.zeros((1, n_symbols)), signed[:-1]])
    else:
        signed_prev = np.zeros_like(signed)

    gap_return = np.sum(signed_prev * long_gap, axis=1)
    intraday_return = np.sum(signed * long_intraday, axis=1)
    cost_return = np.sum(np.abs(signed - signed_prev) * cost_rate, axis=1)

    factor = (1.0 + gap_return) * (1.0 - cost_return) * (1.0 + intraday_return)
    equity_values = ACTIVE_CAPITAL * np.cumprod(factor)
    final_signed = signed[-1]
    if np.sum(np.abs(final_signed)) > 0.0:
        final_cost = float(np.sum(np.abs(final_signed) * cost_rate[-1]))
        equity_values[-1] *= 1.0 - final_cost
    equity = pd.Series(equity_values, index=index, dtype=float)

    if mode == MODE_IS and bool((equity.index > OOS_SPLIT).any()):
        raise OOSLeakageError("run_backtest: IS-mode equity index extends past OOS_SPLIT")
    return equity, signed


def run_backtest(genome: Genome, cache: MarketCache, mode: str, stress: bool = False) -> pd.Series:
    equity, _signed = _run_backtest_components(genome, cache, mode, stress)
    return equity


def run_backtest_with_weights(genome: Genome, cache: MarketCache, mode: str, stress: bool = False) -> tuple[pd.Series, np.ndarray]:
    """Same as run_backtest, but also returns the realized signed weights ([days, symbols],
    direction * normalized_weight, 0 when flat) -- used by gates23.gate_k5_executability as an
    empirical spot-check on top of the constructional gross<=1x guarantee."""
    return _run_backtest_components(genome, cache, mode, stress)


def oos_slice(equity: pd.Series, mode: str) -> pd.Series:
    """The ONLY function in this package allowed to read OOS-range (> OOS_SPLIT) data."""
    if mode != MODE_OOS_FINAL:
        raise OOSLeakageError(f"oos_slice: called with mode={mode!r} -- OOS data is sealed until mode={MODE_OOS_FINAL!r}")
    return equity[equity.index > OOS_SPLIT]


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start_value, end_value = float(equity.iloc[0]), float(equity.iloc[-1])
    if start_value <= 0.0:
        return 0.0
    days = max((pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).total_seconds() / 86_400.0, 1.0)
    growth = end_value / start_value
    return float(growth ** (365.0 / days) - 1.0) if growth > 0.0 else -1.0


def max_drawdown(equity: pd.Series) -> float:
    values = equity.to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.nan_to_num(1.0 - values / peaks, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(drawdown))


__all__ = [
    "BREAKOUT_ATR_DAYS",
    "BREAKOUT_SMA_DAYS",
    "CARRY_WINDOW_DAYS",
    "FUNDING_SPIKE_WINDOW_DAYS",
    "MODE_IS",
    "MODE_OOS_FINAL",
    "MOMENTUM_LOOKBACK_DAYS",
    "ROLLING_Z_MIN_PERIODS",
    "ROLLING_Z_WINDOW",
    "VALID_MODES",
    "KindSignal",
    "MarketCache",
    "OOSLeakageError",
    "build_market_cache",
    "cagr",
    "market_cache_from_markets",
    "max_drawdown",
    "oos_slice",
    "run_backtest",
    "run_backtest_with_weights",
]
