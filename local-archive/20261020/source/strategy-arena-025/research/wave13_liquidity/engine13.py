# Wave-13 liquidity-constrained-carry engine: the SAME per-timestamp bookkeeping as
# research.wave12_frontier.engine12._run_frontier_loop (gap PnL, intraday PnL, turnover,
# trade-close bookkeeping, final forced unwind -- copied, not reimplemented, continuing the
# wave10 -> wave11 -> wave12 precedent of copying the loop body verbatim across waves), with
# exactly one substantive change: where engine13 gets its per-day cost_rate_frame and
# liquidity_ok_frame from.
#
#   engine12: costs_tiered (assumed rank-tier bp table: 1/3/6/10/20bp) + a flat $2M floor
#             applied identically to every config.
#   engine13: costs_measured (Bitget-measured, volume-fitted bp mapping -- see
#             costs_measured.py) + a liquidity mask that DIFFERS by config: L1-L4 use a pure
#             data-availability mask (build_data_availability_mask), L5 additionally
#             requires the SPEC.md dynamic filter (build_dynamic_liquidity_mask: trailing
#             30d volume >= $20M AND mapped slippage <= 5bp, re-evaluated every day).
#
# Nothing else differs: a single shared `weights` value still drives both legs of every
# `intraday = spot_ret - perp_ret + funding` term (S1's delta-neutral invariant is
# structural), sizing is still a fixed leg_fraction of ACTIVE_CAPITAL per ranked symbol, and
# the funding score / entry-exit hysteresis signal (research.wave1.fam_funding.funding_score
# / carry_position) is imported unmodified.
# tests/test_wave13.py's engine-equivalence test pins this the same way
# test_wave12.py's own test_engine12_matches_wave10_when_cost_is_flat_and_liquidity_always_ok
# does: fed a CONSTANT cost-rate frame and an all-True liquidity mask, this engine reproduces
# research.wave10_carry100.engine.run_fixed_fraction_portfolio bit-for-bit.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.fam_funding import FundingMarket, carry_position, funding_score
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, OOS_SPLIT, RESERVE_FRACTION, TOTAL_CAPITAL, Wave10Result
from research.wave13_liquidity import costs_measured
from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.configs13 import Wave13Config
from research.wave13_liquidity.costs_measured import MeasuredCostMapping

DEFAULT_STRESS_MULTIPLIER: Final = 1.0
STRESS_MULTIPLIER: Final = 3.0  # SPEC.md S5: "실측슬리피지 x3 재실행" (wave12 used x2; wave13's base cost is a single live snapshot, so the stress bar is set higher)


# ---------------------------------------------------------------------------
# Frame assembly -- byte-for-byte copy of research.wave12_frontier.engine12.
# _build_aligned_frames (duplicated, not imported, matching wave11/wave12's own precedent
# of copying wave10's preamble rather than adding a new cross-wave abstraction for it).
# ---------------------------------------------------------------------------


def _build_aligned_frames(
    markets: dict[str, FundingMarket], candidate
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spot_open: dict[str, pd.Series] = {}
    spot_close: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    funding_returns: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        funding_daily = market.funding.resample("1D").sum()
        funding_apr = funding_score(market.funding, candidate.window_days).resample("1D").last()
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        spot_open[symbol] = spot_daily["open"]
        spot_close[symbol] = spot_daily["close"]
        perp_open[symbol] = perp_daily["open"]
        perp_close[symbol] = perp_daily["close"]
        funding_returns[symbol] = funding_daily
        scores[symbol] = funding_apr
        active[symbol] = carry_position(funding_apr, candidate)

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    spot_close_frame = pd.DataFrame(spot_close).reindex(spot_open_frame.index)
    perp_open_frame = pd.DataFrame(perp_open).reindex(spot_open_frame.index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(spot_open_frame.index)
    funding_frame = pd.DataFrame(funding_returns).reindex(spot_open_frame.index).fillna(0.0)
    score_frame = pd.DataFrame(scores).reindex(spot_open_frame.index).shift(1)
    active_frame = pd.DataFrame(active).reindex(spot_open_frame.index).fillna(0.0)
    return spot_open_frame, spot_close_frame, perp_open_frame, perp_close_frame, funding_frame, score_frame, active_frame


# ---------------------------------------------------------------------------
# The loop itself -- byte-for-byte copy of engine12._run_frontier_loop (see module
# docstring for the exact one-point diff: where cost_rate_frame/liquidity_ok_frame come
# from is decided by the CALLER, run_candidate, below; the loop itself is cost-model
# agnostic and needs no changes at all).
# ---------------------------------------------------------------------------


def _run_liquidity_loop(
    spot_open_frame: pd.DataFrame,
    spot_close_frame: pd.DataFrame,
    perp_open_frame: pd.DataFrame,
    perp_close_frame: pd.DataFrame,
    funding_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    active_frame: pd.DataFrame,
    top_k: int,
    leg_fraction: float,
    cost_rate_frame: pd.DataFrame,
    liquidity_ok_frame: pd.DataFrame,
) -> tuple[Wave10Result, float, pd.Series]:
    capital = ACTIVE_CAPITAL
    equity_values: list[float] = []
    turnover_values: list[float] = []
    exposures: list[float] = []
    concurrent_counts: list[int] = []
    eligible_counts: list[int] = []
    trade_values: list[float] = []
    trade_times: list[pd.Timestamp] = []
    previous_weights = pd.Series(0.0, index=spot_open_frame.columns)
    trade_growth: dict[str, float] = {}
    trade_weights: dict[str, float] = {}
    total_cost_usdt = 0.0

    def cost_for(symbol: str, timestamp: pd.Timestamp) -> float:
        return float(cost_rate_frame.loc[timestamp, symbol])

    for timestamp in spot_open_frame.index:
        available = (
            spot_open_frame.loc[timestamp].notna()
            & spot_close_frame.loc[timestamp].notna()
            & perp_open_frame.loc[timestamp].notna()
            & perp_close_frame.loc[timestamp].notna()
        )
        eligible = active_frame.loc[timestamp][active_frame.loc[timestamp] > 0.0].index
        eligible = eligible.intersection(available[available].index)
        # Liquidity mask -- re-checked every day, no "already entered" exemption (see
        # costs_measured.build_dynamic_liquidity_mask's own docstring for why: this loop
        # recomputes `eligible` fresh every single timestamp, so there is no separate
        # per-position entry-event state to hook a one-time-only check into).
        liquidity_row = liquidity_ok_frame.loc[timestamp]
        eligible = eligible.intersection(liquidity_row[liquidity_row].index)
        eligible_counts.append(int(len(eligible)))
        ranked = score_frame.loc[timestamp, eligible].dropna().nlargest(top_k).index
        weights = pd.Series(0.0, index=spot_open_frame.columns)
        if len(ranked) > 0:
            weights.loc[ranked] = leg_fraction
        spot_gap = spot_open_frame.loc[timestamp] / spot_close_frame.shift(1).loc[timestamp] - 1.0
        perp_gap = perp_open_frame.loc[timestamp] / perp_close_frame.shift(1).loc[timestamp] - 1.0
        gap_by_symbol = (spot_gap - perp_gap).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        capital *= 1.0 + float((gap_by_symbol * previous_weights).sum())
        turnover = float((weights - previous_weights).abs().sum())
        cost_return = sum(
            abs(float(weights[symbol] - previous_weights[symbol])) * cost_for(symbol, timestamp)
            for symbol in spot_open_frame.columns
        )
        capital_before_cost = capital
        capital *= 1.0 - cost_return
        total_cost_usdt += capital_before_cost - capital
        intraday = (
            spot_close_frame.loc[timestamp] / spot_open_frame.loc[timestamp]
            - perp_close_frame.loc[timestamp] / perp_open_frame.loc[timestamp]
        )
        intraday = (intraday + funding_frame.loc[timestamp]).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        capital *= 1.0 + float((intraday * weights).sum())
        for symbol in spot_open_frame.columns:
            previous_weight = float(previous_weights[symbol])
            current_weight = float(weights[symbol])
            leg_rate = cost_for(symbol, timestamp)
            if previous_weight > 0.0 and symbol in trade_growth:
                trade_growth[symbol] *= 1.0 + float(gap_by_symbol[symbol])
            if previous_weight > 0.0 and current_weight == 0.0:
                trade_growth[symbol] *= 1.0 - leg_rate
                trade_values.append((trade_growth.pop(symbol) - 1.0) * trade_weights.pop(symbol))
                trade_times.append(pd.Timestamp(timestamp))
            elif previous_weight == 0.0 and current_weight > 0.0:
                trade_growth[symbol] = 1.0 - leg_rate
                trade_weights[symbol] = current_weight
            elif previous_weight > 0.0 and current_weight > 0.0 and previous_weight != current_weight:
                trade_growth[symbol] *= 1.0 - abs(current_weight - previous_weight) * leg_rate / max(current_weight, previous_weight)
                trade_weights[symbol] = current_weight
            if current_weight > 0.0:
                trade_growth[symbol] *= 1.0 + float(intraday[symbol])
        equity_values.append(capital)
        turnover_values.append(turnover)
        exposures.append(float(weights.abs().sum()))
        concurrent_counts.append(int((weights != 0.0).sum()))
        previous_weights = weights

    if len(spot_open_frame.index) > 0 and float(previous_weights.abs().sum()) > 0.0:
        final_timestamp = pd.Timestamp(spot_open_frame.index[-1])
        final_cost = sum(float(previous_weights[symbol]) * cost_for(symbol, final_timestamp) for symbol in spot_open_frame.columns)
        capital_before_final_cost = capital
        capital *= 1.0 - final_cost
        total_cost_usdt += capital_before_final_cost - capital
        equity_values[-1] = capital
        turnover_values[-1] += float(previous_weights.abs().sum())
        for symbol, growth in trade_growth.items():
            leg_rate = cost_for(symbol, final_timestamp)
            trade_values.append((growth * (1.0 - leg_rate) - 1.0) * trade_weights[symbol])
            trade_times.append(final_timestamp)

    equity = pd.Series(equity_values, index=spot_open_frame.index, dtype=float)
    positions = pd.Series(exposures, index=spot_open_frame.index, dtype=float)
    turnover_series = pd.Series(turnover_values, index=spot_open_frame.index, dtype=float)
    trades = pd.Series(trade_values, index=pd.DatetimeIndex(trade_times), dtype=float).sort_index()
    eligible_series = pd.Series(eligible_counts, index=spot_open_frame.index, dtype=float)
    result = Wave10Result(
        equity=equity,
        positions=positions,
        turnover=turnover_series,
        trade_returns=trades,
        max_concurrent_positions=max(concurrent_counts, default=0),
        symbols_used=tuple(spot_open_frame.columns),
    )
    return result, total_cost_usdt, eligible_series


# ---------------------------------------------------------------------------
# Top-level entry points used by run_wave13.py.
# ---------------------------------------------------------------------------


def build_cost_and_liquidity_frames(
    config: Wave13Config,
    symbols: tuple[str, ...],
    index: pd.DatetimeIndex,
    mapping: MeasuredCostMapping,
    stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds this config's own cost-rate and liquidity-mask frames, reindexed to exactly
    `symbols` x `index`. L1-L4 (config.universe_kind in {"fixed","breadth"}) get a pure
    data-availability liquidity mask; L5 (config.universe_kind == "dynamic") additionally
    gets SPEC.md's dynamic tradability filter -- see costs_measured.py's
    build_data_availability_mask / build_dynamic_liquidity_mask docstrings."""
    quote_volume_frame = ul.load_quote_volume_frame(symbols)
    cost_rate_frame = costs_measured.build_cost_rate_frame(quote_volume_frame, symbols, mapping, stress_multiplier)
    if config.universe_kind == "dynamic":
        if config.dynamic_volume_floor_usdt is None or config.dynamic_slippage_cap_bp is None:
            raise ValueError(f"{config.candidate.candidate_id}: universe_kind='dynamic' requires both dynamic_volume_floor_usdt and dynamic_slippage_cap_bp")
        liquidity_ok_frame = costs_measured.build_dynamic_liquidity_mask(
            quote_volume_frame, symbols, mapping, config.dynamic_volume_floor_usdt, config.dynamic_slippage_cap_bp
        )
    else:
        liquidity_ok_frame = costs_measured.build_data_availability_mask(quote_volume_frame, symbols)

    cost_rate_frame = cost_rate_frame.reindex(index=index, columns=list(symbols))
    liquidity_ok_frame = liquidity_ok_frame.reindex(index=index, columns=list(symbols))
    # Fail-closed fallback only (should be a no-op: quote_volume_frame's own date range
    # already spans spot_open_frame's index for every symbol actually returned by
    # verify_cache_and_load_symbols) -- an unknown day/symbol gets the worst measured tier
    # and is treated as illiquid, mirroring engine12.build_cost_frames_for_symbols's own
    # fallback convention.
    fallback_rate = costs_measured.cost_rate_from_bp(mapping.worst_bp, stress_multiplier)
    cost_rate_frame = cost_rate_frame.fillna(fallback_rate)
    liquidity_ok_frame = liquidity_ok_frame.fillna(False)
    return cost_rate_frame, liquidity_ok_frame


def run_candidate(
    config: Wave13Config, mapping: MeasuredCostMapping, stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER
) -> tuple[Wave10Result, float, pd.Series]:
    """Dispatches a single L1-L5 config end to end: fail-closed cache load (no network --
    borrows research/wave12_frontier/cache, see universe_liquidity.py), build aligned
    frames, build this config's own measured-cost/liquidity frames, run the loop."""
    symbols = ul.verify_cache_and_load_symbols(config)
    markets = ul.load_markets_for_symbols(symbols)
    frames = _build_aligned_frames(markets, config.candidate)
    spot_open_frame = frames[0]
    cost_rate_frame, liquidity_ok_frame = build_cost_and_liquidity_frames(
        config, tuple(spot_open_frame.columns), spot_open_frame.index, mapping, stress_multiplier
    )
    return _run_liquidity_loop(*frames, config.candidate.top_k, config.leg_fraction, cost_rate_frame, liquidity_ok_frame)


__all__ = [
    "ACTIVE_CAPITAL",
    "MIN_ORDER_USDT",
    "OOS_SPLIT",
    "RESERVE_FRACTION",
    "STRESS_MULTIPLIER",
    "TOTAL_CAPITAL",
    "build_cost_and_liquidity_frames",
    "run_candidate",
]
