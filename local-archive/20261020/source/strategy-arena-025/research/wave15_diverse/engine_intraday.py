# Wave-15 A1/A2/A3 intraday-carry engine -- SPEC.md's priority family. Operates on 1h bars
# (unlike every other wave15/prior-wave engine, which is daily) and adds a settlement-timed
# state machine on top of the SAME per-bar bookkeeping formulas (gap PnL / turnover cost /
# intraday PnL / trade-close accounting) engine_daily.py and every wave10-13 engine before it
# use -- see run_intraday_carry's docstring for exactly which lines are the state machine and
# which are the copied bookkeeping.
#
# Universe: BTCUSDT/ETHUSDT only. research/wave6 + research/wave11_yield's 1h caches also
# have a third symbol, SOLUSDT, but it was DROPPED after an interactively-confirmed data
# check: SOLUSDT's real funding settlements are NOT a clean fixed 8h schedule throughout its
# history -- diffing its raw funding timestamps shows periods with settlements every ~2h
# (first observed 2022-11-10, recurring since, consistent with Binance's documented dynamic
# funding-interval-shortening mechanism during volatile stretches for some symbols), mixed
# with ordinary 8h stretches. This engine's whole state machine is built on ONE fixed
# DECISION_HOURS/SETTLEMENT_HOURS pair shared by every symbol in the universe -- correct for
# BTC/ETH (both verified zero-exception 8h for their ENTIRE history) but silently wrong for a
# symbol that sometimes settles every 2h (it would ignore 9 of SOL's 12 daily settlements and
# mis-time the ones it does act on). Building a second, properly variable-schedule state
# machine just to add one more (already-liquid) symbol was judged not worth the added
# complexity/bug surface for this wave's actual question (turnover economics, not breadth) --
# BTC+ETH-only mirrors research/wave13_liquidity/configs13.py's own L1 precedent (a fixed
# 2-symbol pair is an established, legitimate scope choice in this repo, not a new one).
# SPEC.md pre-approves fetching more symbols' 1h history "if A계열에 더 필요" -- also
# deliberately not pursued: wave12-14 already ran the breadth axis to saturation, and this
# wave's point is the MECHANISM, not re-litigating breadth with a new multi-symbol fetch.
#
# BTCUSDT/ETHUSDT have used the standard 00:00/08:00/16:00 UTC funding schedule for their
# entire listed history with ZERO exceptions (interactively verified: diffing raw funding
# timestamps for both symbols finds 0 intervals under 4h across 7000+ settlements each) --
# DECISION_HOURS/SETTLEMENT_HOURS below are safe as fixed constants for this 2-symbol
# universe, not something that needs per-symbol detection.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final, Literal

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import CACHE_DIR as WAVE1_CACHE_DIR
from research.wave1.common import load_frame
from research.wave1.fam_funding import funding_score
from research.wave15_diverse.common15 import ACTIVE_CAPITAL, LEG_FRACTION, Wave10Result

WAVE6_CACHE_DIR: Final = Path(__file__).resolve().parents[2] / "research" / "wave6" / "cache"
WAVE11_CACHE_DIR: Final = Path(__file__).resolve().parents[2] / "research" / "wave11_yield" / "cache"

A_SERIES_SYMBOLS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT")  # SOLUSDT dropped -- see module docstring (irregular <8h funding-interval stretches)

DECISION_HOURS: Final = frozenset({23, 7, 15})  # T-1h for T in {0,8,16} -- new-entry evaluation only
SETTLEMENT_HOURS: Final = frozenset({0, 8, 16})
INTRADAY_HOLD: Final = pd.Timedelta(hours=2)  # entry at T-1h bar's open -> forced exit at T+1h bar's open

Mode = Literal["FLAT", "DAILY", "INTRADAY"]


@dataclass(frozen=True, slots=True)
class IntradayConfig:
    candidate_id: str
    fast_entry_threshold: float | None  # None -> A1's implicit floor (>0, i.e. any positive prior funding); 0.0003 -> A2/A3's SPEC.md filter (0.03% = ~33% APR)
    daily_entry_threshold: float | None  # None disables the hybrid daily-hold leg (A1, A2); 0.15 enables it (A3)
    daily_exit_threshold: float | None  # required together with daily_entry_threshold (A3: 0.075)


def hourly_cache_paths(symbol: str) -> tuple[Path, Path, Path]:
    return (
        WAVE11_CACHE_DIR / f"binance_spot_{symbol}_1h.csv.gz",
        WAVE6_CACHE_DIR / f"binance_fapi_{symbol}_1h.csv.gz",
        WAVE1_CACHE_DIR / f"binance_funding_{symbol}.csv.gz",
    )


def verify_hourly_cache(symbols: tuple[str, ...] = A_SERIES_SYMBOLS) -> None:
    missing = [str(p) for symbol in symbols for p in hourly_cache_paths(symbol) if not p.exists()]
    if missing:
        raise RuntimeError(f"wave15 A-series hourly cache incomplete: {', '.join(missing[:8])}")


@dataclass(frozen=True, slots=True)
class HourlyFrames:
    index: pd.DatetimeIndex
    symbols: tuple[str, ...]
    spot_open: pd.DataFrame
    spot_close: pd.DataFrame
    perp_open: pd.DataFrame
    perp_close: pd.DataFrame
    funding_actual: pd.DataFrame  # nonzero exactly at settlement hours, this period's REAL realized rate
    funding_known: pd.DataFrame  # ffilled -- "most recently realized rate as of this hour" (A1/A2's proxy)
    slow_score_known: pd.DataFrame  # ffilled funding_score(7d), point-in-time (rolling mean is causal by construction)


def build_hourly_frames(symbols: tuple[str, ...] = A_SERIES_SYMBOLS) -> HourlyFrames:
    verify_hourly_cache(symbols)
    spot_open: dict[str, pd.Series] = {}
    spot_close: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    funding_native: dict[str, pd.Series] = {}
    slow_score_native: dict[str, pd.Series] = {}
    for symbol in symbols:
        spot_path, perp_path, funding_path = hourly_cache_paths(symbol)
        spot = load_frame(spot_path)
        perp = load_frame(perp_path)
        funding = load_frame(funding_path)["funding_rate"]
        spot_open[symbol] = spot["open"]
        spot_close[symbol] = spot["close"]
        perp_open[symbol] = perp["open"]
        perp_close[symbol] = perp["close"]
        funding_native[symbol] = funding
        slow_score_native[symbol] = funding_score(funding, 7)  # rolling(21obs).mean()*3*365 -- causal, no lookahead

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    perp_open_frame = pd.DataFrame(perp_open).sort_index()
    index = spot_open_frame.index.union(perp_open_frame.index)
    spot_open_frame = spot_open_frame.reindex(index)
    spot_close_frame = pd.DataFrame(spot_close).reindex(index)
    perp_open_frame = perp_open_frame.reindex(index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(index)
    funding_native_frame = pd.DataFrame(funding_native)
    funding_actual = funding_native_frame.reindex(index, fill_value=0.0).fillna(0.0)
    funding_known = funding_native_frame.reindex(index, method="ffill")
    slow_score_known = pd.DataFrame(slow_score_native).reindex(index, method="ffill")
    return HourlyFrames(
        index=index,
        symbols=symbols,
        spot_open=spot_open_frame,
        spot_close=spot_close_frame,
        perp_open=perp_open_frame,
        perp_close=perp_close_frame,
        funding_actual=funding_actual,
        funding_known=funding_known,
        slow_score_known=slow_score_known,
    )


def build_hourly_cost_rate_frame(quote_volume_frame_daily: pd.DataFrame, symbols: tuple[str, ...], hourly_index: pd.DatetimeIndex, mapping, stress_multiplier: float = 1.0) -> pd.DataFrame:
    """Daily costs_measured mapping (point-in-time-safe trailing 30d volume, see
    costs_measured.point_in_time_known_avg), broadcast to every hour of its own calendar day
    -- never sharper than daily because the volume input itself is only known daily; this is
    conservative (same-day cost is applied even to the day's very first hour) not lookahead."""
    from research.wave13_liquidity import costs_measured

    known_avg = costs_measured.point_in_time_known_avg(quote_volume_frame_daily).reindex(columns=list(symbols))
    bp_frame = costs_measured.bp_frame_from_known_avg(known_avg, mapping)
    daily_rate = costs_measured.cost_rate_from_bp(bp_frame, stress_multiplier)
    floored = pd.DatetimeIndex(hourly_index).floor("D")
    daily_lookup = daily_rate.copy()
    daily_lookup.index = pd.DatetimeIndex(daily_lookup.index).floor("D")
    daily_lookup = daily_lookup[~daily_lookup.index.duplicated(keep="last")]
    hourly_rate = daily_lookup.reindex(floored)
    hourly_rate.index = hourly_index
    return hourly_rate.fillna(mapping.worst_bp * 0.0 + costs_measured.cost_rate_from_bp(mapping.worst_bp, stress_multiplier))


def _resolve_decision(
    mode: Mode,
    held_symbol: str | None,
    fast_scores: pd.Series,
    slow_scores: pd.Series,
    config: IntradayConfig,
) -> tuple[Mode, str | None]:
    """Priority order, evaluated ONLY at decision bars (T-1h): (1) keep an existing DAILY
    hold alive via hysteresis on the SLOW score: (2) open a NEW daily hold if some eligible
    symbol's slow score clears daily_entry_threshold; (3) fall back to a 2-bar INTRADAY entry
    gated on the FAST (prior-realized-period) score; (4) stay flat. By construction (see
    module docstring: an INTRADAY hold always resolves to FLAT strictly before the next
    decision bar, 6-8h later) `mode` passed in here is never "INTRADAY" -- pinned by
    test_wave15.py's state-machine test."""
    if mode == "DAILY" and held_symbol is not None:
        current_slow = slow_scores.get(held_symbol, float("nan"))
        if pd.notna(current_slow) and config.daily_exit_threshold is not None and current_slow > config.daily_exit_threshold:
            return "DAILY", held_symbol
    if config.daily_entry_threshold is not None:
        qualifying = slow_scores[slow_scores > config.daily_entry_threshold].dropna()
        if len(qualifying) > 0:
            return "DAILY", str(qualifying.idxmax())
    fast_bar = config.fast_entry_threshold if config.fast_entry_threshold is not None else 0.0
    qualifying_fast = fast_scores[fast_scores > fast_bar].dropna()
    if len(qualifying_fast) > 0:
        return "INTRADAY", str(qualifying_fast.idxmax())
    return "FLAT", None


def run_intraday_carry(
    frames: HourlyFrames,
    config: IntradayConfig,
    cost_rate_frame: pd.DataFrame,
    leg_fraction: float = LEG_FRACTION,
) -> tuple[Wave10Result, float, dict[str, float]]:
    symbols = frames.symbols
    n_symbols = len(symbols)
    index = frames.index
    n = len(index)

    spot_open = frames.spot_open.to_numpy(dtype=float)
    spot_close = frames.spot_close.to_numpy(dtype=float)
    perp_open = frames.perp_open.to_numpy(dtype=float)
    perp_close = frames.perp_close.to_numpy(dtype=float)
    funding_actual = frames.funding_actual.to_numpy(dtype=float)
    cost_rate = cost_rate_frame.reindex(index=index, columns=list(symbols)).to_numpy(dtype=float)
    spot_close_prev = np.vstack([np.full(n_symbols, np.nan), spot_close[:-1]])
    perp_close_prev = np.vstack([np.full(n_symbols, np.nan), perp_close[:-1]])

    available = np.isfinite(spot_open) & np.isfinite(spot_close) & np.isfinite(perp_open) & np.isfinite(perp_close)

    hours = np.array([ts.hour for ts in index])
    is_decision_bar = np.isin(hours, tuple(DECISION_HOURS))

    capital = ACTIVE_CAPITAL
    equity = np.empty(n, dtype=float)
    turnover_arr = np.empty(n, dtype=float)
    exposure_arr = np.empty(n, dtype=float)
    concurrent_arr = np.empty(n, dtype=int)
    trade_values: list[float] = []
    trade_times: list[pd.Timestamp] = []
    total_cost_usdt = 0.0

    previous_weights = np.zeros(n_symbols, dtype=float)
    trade_growth: dict[int, float] = {}
    trade_weight_at_open: dict[int, float] = {}

    mode: Mode = "FLAT"
    held_symbol_idx: int | None = None
    scheduled_exit_ts: pd.Timestamp | None = None
    n_daily_entries = 0
    n_intraday_entries = 0
    daily_bar_count = 0
    intraday_bar_count = 0
    flat_bar_count = 0
    non_flat_at_decision_check = 0  # invariant counter: should stay 0 (see _resolve_decision docstring)

    for i in range(n):
        timestamp = index[i]
        weights = previous_weights.copy()

        if mode == "INTRADAY" and scheduled_exit_ts is not None and timestamp == scheduled_exit_ts:
            weights[held_symbol_idx] = 0.0
            mode, held_symbol_idx, scheduled_exit_ts = "FLAT", None, None

        if is_decision_bar[i]:
            if mode == "INTRADAY":
                non_flat_at_decision_check += 1  # should never fire -- pinned by tests/test_wave15.py
            avail_row = available[i]
            fast_series = pd.Series(frames.funding_known.to_numpy()[i], index=symbols)[avail_row]
            slow_series = pd.Series(frames.slow_score_known.to_numpy()[i], index=symbols)[avail_row]
            held_symbol = symbols[held_symbol_idx] if held_symbol_idx is not None else None
            new_mode, new_symbol = _resolve_decision(mode, held_symbol, fast_series, slow_series, config)

            if mode == "DAILY" and held_symbol_idx is not None and not (new_mode == "DAILY" and new_symbol == held_symbol):
                weights[held_symbol_idx] = 0.0

            if new_mode == "DAILY":
                new_idx = symbols.index(new_symbol)
                weights[new_idx] = leg_fraction
                if not (mode == "DAILY" and held_symbol_idx == new_idx):
                    n_daily_entries += 1
                mode, held_symbol_idx, scheduled_exit_ts = "DAILY", new_idx, None
            elif new_mode == "INTRADAY":
                new_idx = symbols.index(new_symbol)
                weights[new_idx] = leg_fraction
                mode, held_symbol_idx = "INTRADAY", new_idx
                scheduled_exit_ts = timestamp + INTRADAY_HOLD
                n_intraday_entries += 1
            else:
                mode, held_symbol_idx, scheduled_exit_ts = "FLAT", None, None

        if mode == "DAILY":
            daily_bar_count += 1
        elif mode == "INTRADAY":
            intraday_bar_count += 1
        else:
            flat_bar_count += 1

        spot_gap = spot_open[i] / spot_close_prev[i] - 1.0
        perp_gap = perp_open[i] / perp_close_prev[i] - 1.0
        gap_by_symbol = np.nan_to_num(spot_gap - perp_gap, nan=0.0, posinf=0.0, neginf=0.0)
        capital *= 1.0 + float(np.dot(gap_by_symbol, previous_weights))

        weight_delta = weights - previous_weights
        turnover = float(np.abs(weight_delta).sum())
        cost_return = float(np.dot(np.abs(weight_delta), cost_rate[i]))
        capital_before_cost = capital
        capital *= 1.0 - cost_return
        total_cost_usdt += capital_before_cost - capital

        raw_intraday = spot_close[i] / spot_open[i] - perp_close[i] / perp_open[i] + funding_actual[i]
        intraday_ret = np.nan_to_num(raw_intraday, nan=0.0, posinf=0.0, neginf=0.0)
        capital *= 1.0 + float(np.dot(intraday_ret, weights))

        for symbol_idx in range(n_symbols):
            previous_weight = float(previous_weights[symbol_idx])
            current_weight = float(weights[symbol_idx])
            leg_rate = float(cost_rate[i, symbol_idx])
            if previous_weight > 0.0 and symbol_idx in trade_growth:
                trade_growth[symbol_idx] *= 1.0 + float(gap_by_symbol[symbol_idx])
            if previous_weight > 0.0 and current_weight == 0.0:
                trade_growth[symbol_idx] *= 1.0 - leg_rate
                trade_values.append((trade_growth.pop(symbol_idx) - 1.0) * trade_weight_at_open.pop(symbol_idx))
                trade_times.append(pd.Timestamp(timestamp))
            elif previous_weight == 0.0 and current_weight > 0.0:
                trade_growth[symbol_idx] = 1.0 - leg_rate
                trade_weight_at_open[symbol_idx] = current_weight
            elif previous_weight > 0.0 and current_weight > 0.0 and previous_weight != current_weight:
                trade_growth[symbol_idx] *= 1.0 - abs(current_weight - previous_weight) * leg_rate / max(current_weight, previous_weight)
                trade_weight_at_open[symbol_idx] = current_weight
            if current_weight > 0.0:
                trade_growth[symbol_idx] *= 1.0 + float(intraday_ret[symbol_idx])

        equity[i] = capital
        turnover_arr[i] = turnover
        exposure_arr[i] = float(np.abs(weights).sum())
        concurrent_arr[i] = int((weights != 0.0).sum())
        previous_weights = weights

    if n > 0 and float(np.abs(previous_weights).sum()) > 0.0:
        final_timestamp = pd.Timestamp(index[-1])
        final_cost = float(np.dot(previous_weights, cost_rate[-1]))
        capital_before_final_cost = capital
        capital *= 1.0 - final_cost
        total_cost_usdt += capital_before_final_cost - capital
        equity[-1] = capital
        turnover_arr[-1] += float(np.abs(previous_weights).sum())
        for symbol_idx, growth in trade_growth.items():
            leg_rate = float(cost_rate[-1, symbol_idx])
            trade_values.append((growth * (1.0 - leg_rate) - 1.0) * trade_weight_at_open[symbol_idx])
            trade_times.append(final_timestamp)

    equity_series = pd.Series(equity, index=index, dtype=float)
    positions_series = pd.Series(exposure_arr, index=index, dtype=float)
    turnover_series = pd.Series(turnover_arr, index=index, dtype=float)
    trades = pd.Series(trade_values, index=pd.DatetimeIndex(trade_times), dtype=float).sort_index()
    result = Wave10Result(
        equity=equity_series,
        positions=positions_series,
        turnover=turnover_series,
        trade_returns=trades,
        max_concurrent_positions=int(concurrent_arr.max()) if n > 0 else 0,
        symbols_used=symbols,
    )
    diagnostics = {
        "n_daily_entries": float(n_daily_entries),
        "n_intraday_entries": float(n_intraday_entries),
        "daily_bar_fraction": daily_bar_count / n if n else 0.0,
        "intraday_bar_fraction": intraday_bar_count / n if n else 0.0,
        "flat_bar_fraction": flat_bar_count / n if n else 0.0,
        "state_machine_invariant_violations": float(non_flat_at_decision_check),
    }
    return result, total_cost_usdt, diagnostics


__all__ = [
    "A_SERIES_SYMBOLS",
    "DECISION_HOURS",
    "INTRADAY_HOLD",
    "SETTLEMENT_HOURS",
    "HourlyFrames",
    "IntradayConfig",
    "build_hourly_cost_rate_frame",
    "build_hourly_frames",
    "hourly_cache_paths",
    "run_intraday_carry",
    "verify_hourly_cache",
]
