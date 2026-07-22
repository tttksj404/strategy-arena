# Wave-12 frontier engine: the SAME per-timestamp bookkeeping as
# research.wave10_carry100.engine.run_fixed_fraction_portfolio /
# research.wave11_yield.engine_y.run_daily_fixed_fraction (gap PnL, intraday PnL,
# turnover, trade-close bookkeeping, final forced unwind -- copied, not reimplemented,
# same as wave11 copied wave10's), with exactly the two changes SPEC.md's "필수 사전
# 조건" registers:
#
#   1. cost_for(symbol) becomes cost_for(symbol, timestamp): a per-day, per-symbol
#      lookup into a pre-computed tiered cost-rate frame (research.wave12_frontier.
#      costs_tiered), instead of the old time-invariant research.wave10_carry100.engine.
#      cost_rate(symbol).
#   2. the daily `eligible` set additionally intersects that day's liquidity-floor mask
#      (a symbol whose point-in-time trailing 30d average quote_volume is below $2M
#      cannot receive new weight that day -- see costs_tiered.py's docstring for why this
#      is checked every day rather than only at a tracked "entry" event).
#
# Nothing else about the loop differs: a single shared `weights` value still drives both
# legs of every `intraday = spot_ret - perp_ret + funding` term (S1's delta-neutral
# invariant is structural, unchanged), sizing is still a fixed leg_fraction of
# ACTIVE_CAPITAL per ranked symbol (unchanged from wave10/wave11), and the funding score
# / entry-exit hysteresis signal (research.wave1.fam_funding.funding_score /
# carry_position) is imported unmodified. tests/test_wave12.py's
# test_engine12_matches_wave10_when_cost_is_flat_and_liquidity_always_ok pins this: fed a
# CONSTANT cost-rate frame (equal to wave10's own flat cost_rate for that symbol) and an
# all-True liquidity mask, this engine reproduces wave10's run_fixed_fraction_portfolio
# bit-for-bit -- proof that the cost/liquidity model is the only thing that changed.

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
from research.wave12_frontier import costs_tiered
from research.wave12_frontier import universe_frontier as uf
from research.wave12_frontier.configs12 import Wave12Config

DEFAULT_STRESS_MULTIPLIER: Final = 1.0
STRESS_MULTIPLIER: Final = 2.0  # SPEC.md S5: "위 슬리피지 ×2 재실행"


# ---------------------------------------------------------------------------
# Frame assembly (identical pattern to research.wave10_carry100.engine.
# run_fixed_fraction_portfolio / research.wave11_yield.engine_y.run_daily_fixed_fraction's
# own preamble -- duplicated intentionally rather than imported, matching wave11's own
# precedent of copying wave10's loop body instead of adding a new cross-wave abstraction).
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
# The loop itself (see module docstring for the exact two-point diff vs wave10/wave11).
# ---------------------------------------------------------------------------


def _run_frontier_loop(
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
        # SPEC.md liquidity floor: a symbol below the $2M point-in-time 30d-avg-volume
        # bar cannot receive weight today -- see costs_tiered.py's docstring for why this
        # is applied every day (this engine has no separate "already entered" state to
        # exempt from the daily re-check).
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
# Top-level entry points used by run_wave12.py.
# ---------------------------------------------------------------------------


def build_cost_frames_for_symbols(
    symbols: tuple[str, ...], index: pd.DatetimeIndex, stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the shared tier-reference quote-volume panel, builds the point-in-time
    cost-rate/liquidity frames over it (research.wave12_frontier.costs_tiered), and
    reindexes to exactly `symbols` x `index` -- the shape research/wave12_frontier's loop
    needs for one specific config's run. The tier-reference panel spans a strict superset
    of any single config's own date range (every reference symbol's data reaches back
    close to research/wave1's 2019-09-01 start), so this reindex is a narrowing, not a
    fill -- documented in universe_frontier.py's module docstring."""
    tier_symbols = uf.verify_cache_and_load_tier_reference_symbols()
    quote_volume_frame = uf.load_quote_volume_frame(tier_symbols)
    cost_rate_frame, liquidity_ok_frame = costs_tiered.build_cost_and_liquidity_frames(quote_volume_frame, symbols, stress_multiplier)
    cost_rate_frame = cost_rate_frame.reindex(index=index, columns=list(symbols))
    liquidity_ok_frame = liquidity_ok_frame.reindex(index=index, columns=list(symbols))
    # Fail-closed fallback only -- should be a no-op given the superset relationship
    # above; if it ever isn't (e.g. a symbol's config-level date range somehow reaches
    # earlier than the tier-reference panel), an unknown day/symbol gets the worst tier
    # and is treated as illiquid, never silently priced cheap or assumed tradeable.
    fallback_rate = costs_tiered.cost_rate_from_bp(costs_tiered.TAIL_SLIPPAGE_BP, stress_multiplier)
    cost_rate_frame = cost_rate_frame.fillna(fallback_rate)
    liquidity_ok_frame = liquidity_ok_frame.fillna(False)
    return cost_rate_frame, liquidity_ok_frame


def run_candidate(config: Wave12Config, stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER) -> tuple[Wave10Result, float, pd.Series]:
    """Dispatches a single U0-U6 config end to end: fail-closed cache load (no network),
    build aligned frames, build the matching tiered-cost/liquidity frames, run the loop.
    Returns (result, total_cost_usdt, eligible_count_series) -- the extra series (vs.
    research.wave11_yield.engine_y.run_candidate's 2-tuple contract) is the per-day count
    of symbols that cleared every eligibility filter (active hysteresis + data
    availability + liquidity floor) that day, i.e. the actual size of the tradeable pool
    on that day -- SPEC.md's required frontier-curve column "유니버스 크기(편입 심볼수
    중앙값)" is this series' median, computed in reporting12.py."""
    symbols = uf.verify_cache_and_load_symbols(config.candidate.candidate_id)
    markets = uf.load_markets_for_symbols(symbols)
    frames = _build_aligned_frames(markets, config.candidate)
    spot_open_frame = frames[0]
    cost_rate_frame, liquidity_ok_frame = build_cost_frames_for_symbols(tuple(spot_open_frame.columns), spot_open_frame.index, stress_multiplier)
    return _run_frontier_loop(*frames, config.candidate.top_k, config.leg_fraction, cost_rate_frame, liquidity_ok_frame)


__all__ = [
    "ACTIVE_CAPITAL",
    "MIN_ORDER_USDT",
    "OOS_SPLIT",
    "RESERVE_FRACTION",
    "STRESS_MULTIPLIER",
    "TOTAL_CAPITAL",
    "build_cost_frames_for_symbols",
    "run_candidate",
]
