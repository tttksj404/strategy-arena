# Wave-15 D1: sector-internal pair mean-reversion. Both legs are PERPETUALS (no spot leg --
# SPEC.md: "양쪽 퍼프 델타중립") -- a dollar-neutral pair (long $45 notional one perp, short $45
# notional the other), a DIFFERENT risk shape than every other wave10-15 candidate's
# spot-vs-own-perp basis neutrality (residual basis/correlation risk between two DIFFERENT
# tokens' perps, not a single token's spot-perp basis) -- gates15's S1 disclosure records this
# distinction explicitly rather than reusing the word "델타중립" as if it meant the same thing.
#
# Sector pools are HARDCODED per the user's task packet (섹터 분류는 하드코딩): L1 =
# {SOL,AVAX,NEAR}, DeFi = {UNI,AAVE,LINK}, Meme = {DOGE,SHIB,...}. SHIBUSDT has no cache in
# this repo (confirmed absent from every one of the three fallback cache dirs) -- substituted
# with WIFUSDT (also a top-of-mind meme perp, cached with a full spot/perp/funding triple),
# documented here and in report/wave15_report.md rather than silently swapped. "상위 2종" per
# sector is resolved PROCEDURALLY (ranked by wave12_frontier's own reference_volume_30d_usdt
# snapshot, common15.reference_volume_30d) -- not a manually cherry-picked pair.
#
# SPEC.md's literal candidate text is one-directional ("z>2 진입, z<0.5 청산"). Implemented
# here as a SYMMETRIC |z| rule instead (entry |z|>2, exit |z|<0.5, direction frozen at entry
# by the sign of z) -- a deliberate, disclosed deviation: which of the two sector symbols is
# labelled "first" is an arbitrary alphabetical/selection artifact, so a literal one-sided
# "z>2 only" rule would make the strategy's tradable regime depend on that arbitrary label
# rather than on the (label-independent) economic content "the spread is 2 std devs from its
# own 30d norm, in either direction." Symmetric |z| removes that arbitrariness while keeping
# the exact same mean-reversion mechanism and the exact same 2.0/0.5 bars SPEC.md registered.

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

from research.wave1.fam_funding import FundingMarket
from research.wave13_liquidity import costs_measured
from research.wave15_diverse.common15 import ACTIVE_CAPITAL, LEG_FRACTION, TOP_K, Wave10Result
from research.wave2.funding import W2_MAKER_FEE_RATE

SECTOR_CANDIDATE_POOLS: Final[dict[str, tuple[str, ...]]] = {
    "L1": ("SOLUSDT", "AVAXUSDT", "NEARUSDT"),
    "DeFi": ("UNIUSDT", "AAVEUSDT", "LINKUSDT"),
    "Meme": ("DOGEUSDT", "WIFUSDT"),  # SHIBUSDT unavailable in cache -> WIFUSDT substituted (see module docstring)
}
MEME_SUBSTITUTION_NOTE: Final = "SHIBUSDT 캐시 없음 (3개 fallback 캐시 디렉터리 모두 부재 확인) -> WIFUSDT로 대체."

ZSCORE_WINDOW_DAYS: Final = 30
ENTRY_Z: Final = 2.0
EXIT_Z: Final = 0.5


@dataclass(frozen=True, slots=True)
class PairInstrument:
    sector: str
    symbol_a: str
    symbol_b: str

    @property
    def pair_id(self) -> str:
        return f"{self.sector}:{self.symbol_a.removesuffix('USDT')}-{self.symbol_b.removesuffix('USDT')}"


def select_sector_pairs(pool: dict) -> tuple[PairInstrument, ...]:
    from research.wave15_diverse.common15 import reference_volume_30d

    instruments: list[PairInstrument] = []
    for sector, candidates in SECTOR_CANDIDATE_POOLS.items():
        ranked = sorted(candidates, key=lambda symbol: reference_volume_30d(pool, symbol), reverse=True)
        top_two = ranked[:2]
        instruments.append(PairInstrument(sector, top_two[0], top_two[1]))
    return tuple(instruments)


def pair_position_and_direction(z: pd.Series, entry_z: float = ENTRY_Z, exit_z: float = EXIT_Z) -> tuple[pd.Series, pd.Series]:
    """Same hysteresis SHAPE as research.wave1.fam_funding.carry_position (unconditional
    scan, shift(1)+fillna(0.0) ending) generalized to (a) a symmetric |z| entry/exit band and
    (b) a direction that FREEZES at the entry bar and holds fixed until the position exits
    (never re-derived mid-hold from a later, possibly opposite-signed, z) -- tests/
    test_wave15.py's D1 pair-selection/direction test pins that a sign flip WHILE held does
    not flip the direction series."""
    active_values: list[float] = []
    direction_values: list[float] = []
    active = 0.0
    direction = 0.0
    for value in z:
        if pd.isna(value):
            pass
        elif active == 0.0 and abs(value) > entry_z:
            active = 1.0
            direction = 1.0 if value > 0.0 else -1.0
        elif active == 1.0 and abs(value) < exit_z:
            active = 0.0
            direction = 0.0
        active_values.append(active)
        direction_values.append(direction)
    idx = z.index
    active_series = pd.Series(active_values, index=idx, dtype=float).shift(1).fillna(0.0)
    direction_series = pd.Series(direction_values, index=idx, dtype=float).shift(1).fillna(0.0)
    return active_series, direction_series


@dataclass(frozen=True, slots=True)
class PairDailyFrames:
    index: pd.DatetimeIndex
    z: pd.Series  # raw (unshifted) daily z-score, for reporting
    active: pd.Series  # point-in-time (shifted) 0/1
    direction: pd.Series  # point-in-time (shifted), frozen at entry: +1 (short A/long B) | -1 (long A/short B) | 0
    combined_intraday: pd.Series  # today's REALIZED (unshifted) direction-adjusted return, incl. funding
    combined_gap: pd.Series  # today's REALIZED (unshifted) direction-adjusted overnight gap return
    cost_rate: pd.Series  # both-legs one-way cost rate (each leg's OWN measured bp, not assumed equal)


def build_pair_frames(pair: PairInstrument, markets: dict[str, FundingMarket], quote_volume_frame: pd.DataFrame, mapping: costs_measured.MeasuredCostMapping, stress_multiplier: float = 1.0) -> PairDailyFrames:
    market_a, market_b = markets[pair.symbol_a], markets[pair.symbol_b]
    perp_a = market_a.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
    perp_b = market_b.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
    index = perp_a.index.union(perp_b.index)
    perp_a = perp_a.reindex(index)
    perp_b = perp_b.reindex(index)
    funding_a = market_a.funding.resample("1D").sum().reindex(index).fillna(0.0)
    funding_b = market_b.funding.resample("1D").sum().reindex(index).fillna(0.0)

    log_spread = np.log(perp_a["close"]) - np.log(perp_b["close"])
    rolling_mean = log_spread.rolling(ZSCORE_WINDOW_DAYS, min_periods=ZSCORE_WINDOW_DAYS).mean()
    rolling_std = log_spread.rolling(ZSCORE_WINDOW_DAYS, min_periods=ZSCORE_WINDOW_DAYS).std()
    z = ((log_spread - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)

    active, direction = pair_position_and_direction(z)

    a_intraday = perp_a["close"] / perp_a["open"] - 1.0
    b_intraday = perp_b["close"] / perp_b["open"] - 1.0
    raw_intraday = (b_intraday - a_intraday) + (funding_a - funding_b)
    combined_intraday = (direction * raw_intraday).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    a_gap = perp_a["open"] / perp_a["close"].shift(1) - 1.0
    b_gap = perp_b["open"] / perp_b["close"].shift(1) - 1.0
    raw_gap = b_gap - a_gap
    combined_gap = (direction * raw_gap).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    known_a = costs_measured.point_in_time_known_avg(quote_volume_frame[[pair.symbol_a]])[pair.symbol_a]
    known_b = costs_measured.point_in_time_known_avg(quote_volume_frame[[pair.symbol_b]])[pair.symbol_b]
    bp_a = costs_measured.bp_frame_from_known_avg(known_a.to_frame(pair.symbol_a), mapping)[pair.symbol_a]
    bp_b = costs_measured.bp_frame_from_known_avg(known_b.to_frame(pair.symbol_b), mapping)[pair.symbol_b]
    cost_rate = (2.0 * W2_MAKER_FEE_RATE + (bp_a.reindex(index) + bp_b.reindex(index)) * 0.0001 * stress_multiplier).fillna(
        2.0 * W2_MAKER_FEE_RATE + 2.0 * mapping.worst_bp * 0.0001 * stress_multiplier
    )

    return PairDailyFrames(
        index=index,
        z=z,
        active=active,
        direction=direction,
        combined_intraday=combined_intraday,
        combined_gap=combined_gap,
        cost_rate=cost_rate,
    )


def run_sector_pairs(
    pairs: tuple[PairInstrument, ...],
    frames_by_pair: dict[str, PairDailyFrames],
    top_k: int = TOP_K,
    leg_fraction: float = LEG_FRACTION,
) -> tuple[Wave10Result, float]:
    pair_ids = tuple(pair.pair_id for pair in pairs)
    index = sorted(set().union(*(set(f.index) for f in frames_by_pair.values())))
    index = pd.DatetimeIndex(index)

    active_frame = pd.DataFrame({pid: frames_by_pair[pid].active.reindex(index).fillna(0.0) for pid in pair_ids})
    score_frame = pd.DataFrame({pid: frames_by_pair[pid].z.abs().reindex(index) for pid in pair_ids}).shift(1)
    intraday_frame = pd.DataFrame({pid: frames_by_pair[pid].combined_intraday.reindex(index).fillna(0.0) for pid in pair_ids})
    gap_frame = pd.DataFrame({pid: frames_by_pair[pid].combined_gap.reindex(index).fillna(0.0) for pid in pair_ids})
    cost_frame = pd.DataFrame({pid: frames_by_pair[pid].cost_rate.reindex(index) for pid in pair_ids}).ffill().fillna(0.0)

    capital = ACTIVE_CAPITAL
    equity_values: list[float] = []
    turnover_values: list[float] = []
    exposures: list[float] = []
    concurrent_counts: list[int] = []
    trade_values: list[float] = []
    trade_times: list[pd.Timestamp] = []
    previous_weights = pd.Series(0.0, index=pair_ids)
    trade_growth: dict[str, float] = {}
    trade_weights: dict[str, float] = {}
    total_cost_usdt = 0.0

    def cost_for(pid: str, timestamp: pd.Timestamp) -> float:
        return float(cost_frame.loc[timestamp, pid])

    for timestamp in index:
        eligible = active_frame.loc[timestamp][active_frame.loc[timestamp] > 0.0].index
        ranked = score_frame.loc[timestamp, eligible].dropna().nlargest(top_k).index
        weights = pd.Series(0.0, index=pair_ids)
        if len(ranked) > 0:
            weights.loc[ranked] = leg_fraction

        capital *= 1.0 + float((gap_frame.loc[timestamp] * previous_weights).sum())
        turnover = float((weights - previous_weights).abs().sum())
        cost_return = sum(abs(float(weights[pid] - previous_weights[pid])) * cost_for(pid, timestamp) for pid in pair_ids)
        capital_before_cost = capital
        capital *= 1.0 - cost_return
        total_cost_usdt += capital_before_cost - capital

        intraday = intraday_frame.loc[timestamp]
        capital *= 1.0 + float((intraday * weights).sum())

        for pid in pair_ids:
            previous_weight = float(previous_weights[pid])
            current_weight = float(weights[pid])
            leg_rate = cost_for(pid, timestamp)
            if previous_weight > 0.0 and pid in trade_growth:
                trade_growth[pid] *= 1.0 + float(gap_frame.loc[timestamp, pid])
            if previous_weight > 0.0 and current_weight == 0.0:
                trade_growth[pid] *= 1.0 - leg_rate
                trade_values.append((trade_growth.pop(pid) - 1.0) * trade_weights.pop(pid))
                trade_times.append(pd.Timestamp(timestamp))
            elif previous_weight == 0.0 and current_weight > 0.0:
                trade_growth[pid] = 1.0 - leg_rate
                trade_weights[pid] = current_weight
            elif previous_weight > 0.0 and current_weight > 0.0 and previous_weight != current_weight:
                trade_growth[pid] *= 1.0 - abs(current_weight - previous_weight) * leg_rate / max(current_weight, previous_weight)
                trade_weights[pid] = current_weight
            if current_weight > 0.0:
                trade_growth[pid] *= 1.0 + float(intraday[pid])

        equity_values.append(capital)
        turnover_values.append(turnover)
        exposures.append(float(weights.abs().sum()))
        concurrent_counts.append(int((weights != 0.0).sum()))
        previous_weights = weights

    if len(index) > 0 and float(previous_weights.abs().sum()) > 0.0:
        final_timestamp = pd.Timestamp(index[-1])
        final_cost = sum(float(previous_weights[pid]) * cost_for(pid, final_timestamp) for pid in pair_ids)
        capital_before_final_cost = capital
        capital *= 1.0 - final_cost
        total_cost_usdt += capital_before_final_cost - capital
        equity_values[-1] = capital
        turnover_values[-1] += float(previous_weights.abs().sum())
        for pid, growth in trade_growth.items():
            leg_rate = cost_for(pid, final_timestamp)
            trade_values.append((growth * (1.0 - leg_rate) - 1.0) * trade_weights[pid])
            trade_times.append(final_timestamp)

    equity = pd.Series(equity_values, index=index, dtype=float)
    positions = pd.Series(exposures, index=index, dtype=float)
    turnover_series = pd.Series(turnover_values, index=index, dtype=float)
    trades = pd.Series(trade_values, index=pd.DatetimeIndex(trade_times), dtype=float).sort_index()
    result = Wave10Result(
        equity=equity,
        positions=positions,
        turnover=turnover_series,
        trade_returns=trades,
        max_concurrent_positions=max(concurrent_counts, default=0),
        symbols_used=pair_ids,
    )
    return result, total_cost_usdt


__all__ = [
    "ENTRY_Z",
    "EXIT_Z",
    "MEME_SUBSTITUTION_NOTE",
    "SECTOR_CANDIDATE_POOLS",
    "ZSCORE_WINDOW_DAYS",
    "PairDailyFrames",
    "PairInstrument",
    "build_pair_frames",
    "pair_position_and_direction",
    "run_sector_pairs",
    "select_sector_pairs",
]
