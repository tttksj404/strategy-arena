# Wave-12 tiered slippage + liquidity-floor cost model (SPEC.md's "필수 사전 조건").
#
# Every prior carry-family wave (wave1 F1, wave2 W2, wave10 C1-C4, wave11 Y1-Y6) prices
# slippage with research.wave1.costs.slippage_rate: a STATIC, symbol-only lookup (1bp
# BTC/ETH/SOL, 3bp everything else) that never changes over the life of a backtest. That
# is exactly what SPEC.md's "필수 사전 조건" section forbids reusing once the universe
# expands past ~100 liquid names: a symbol that is genuinely thin for most of its history
# (or thin only in some multi-year stretch) would silently get majors-grade execution cost
# throughout. This module replaces the static lookup with a point-in-time, rank-based
# tier: each symbol's cost on day t depends on its OWN trailing 30-day average
# quote_volume ending the PRIOR day (computed with pandas .rolling(...).shift(1), i.e.
# using only data strictly before the day the rate is charged) -- the same t-close ->
# t+1-open lag every engine in this repo already uses for the funding score itself (see
# research.wave1.fam_funding.carry_position's own shift(1)).
#
# BTC/ETH keep a fixed 1bp regardless of rank (SPEC.md's table lists them as their own
# row, not "whatever their volume rank happens to be" -- in practice they are always
# rank 1-2 anyway, so this is a documentation/robustness convenience, not a departure).
# Every other symbol's tier is purely a function of its point-in-time rank among a shared
# reference pool (research.wave12_frontier.universe_frontier's tier-reference symbol set:
# every symbol that cleared the loosest history floor used anywhere across U0-U6, NOT
# just the symbols in whichever config happens to be running) -- so "rank 1-50" means the
# same 50 names on a given day regardless of which of U0-U6 is being priced that run.
#
# Liquidity floor: SPEC.md excludes any symbol whose "진입 시점" (entry-time) 30-day
# average quote_volume is below $2,000,000 from being tradeable that day. Because every
# cost-bearing rebalance in research.wave10_carry100.engine's loop already recomputes
# `eligible` fresh every single day (there is no persistent "this position was already
# open, skip re-checking eligibility" branch), "entry-time" and "every day this position
# would otherwise continue to be held" are the same check in this engine's actual
# mechanics -- there is no separate per-position entry-timestamp bookkeeping to hook a
# one-time-only check into without restructuring the loop (which SPEC.md's "룰 변경 금지"
# forbids). This module's liquidity mask is therefore applied uniformly, every day, not
# just on the first day a symbol turns active; research/wave12_frontier/engine12.py's
# docstring repeats this same note where the mask is actually wired into the loop.

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave2.funding import W2_MAKER_FEE_RATE

MAJOR_SYMBOLS: Final = frozenset({"BTCUSDT", "ETHUSDT"})

# (max_rank_inclusive, one-way slippage in basis points), checked in ascending order.
# Matches SPEC.md's literal table: 1-50 -> 3bp, 51-100 -> 6bp, 101-200 -> 10bp, 201+ -> 20bp.
RANK_TIER_BOUNDS: Final = (
    (50, 3.0),
    (100, 6.0),
    (200, 10.0),
)
TAIL_SLIPPAGE_BP: Final = 20.0  # rank 201+, and the conservative fallback for "no valid rank yet"
MAJOR_SLIPPAGE_BP: Final = 1.0

LIQUIDITY_FLOOR_USDT: Final = 2_000_000.0
ROLLING_WINDOW_DAYS: Final = 30


def slippage_bp_for_rank(symbol: str, rank: float | None) -> float:
    """Single source of truth for the tier lookup: SPEC.md's table, plus the BTC/ETH
    override and a fail-closed (worst-tier, never zero-cost) default when `rank` is
    None/NaN -- either because the symbol has fewer than ROLLING_WINDOW_DAYS of trailing
    history yet (freshly listed) or because it fell outside the tier-reference pool
    entirely. A symbol with an unknown rank is never silently priced at majors-grade cost."""
    if symbol in MAJOR_SYMBOLS:
        return MAJOR_SLIPPAGE_BP
    if rank is None or not pd.notna(rank):
        return TAIL_SLIPPAGE_BP
    for max_rank_inclusive, bp in RANK_TIER_BOUNDS:
        if rank <= max_rank_inclusive:
            return bp
    return TAIL_SLIPPAGE_BP


def cost_rate_from_bp(bp: float | pd.DataFrame, stress_multiplier: float = 1.0) -> float | pd.DataFrame:
    """One-way cost rate for a leg-pair rebalance unit -- same convention as
    research.wave10_carry100.engine.cost_rate: maker fee 0.02% on each leg (stress
    -invariant; SPEC.md's S5 stress re-run only doubles SLIPPAGE, never the maker fee)
    plus slippage on both legs, scaled by `stress_multiplier` (1.0 base, 2.0 for S5).
    `bp` may be a scalar or a whole DataFrame (pandas broadcasts either way)."""
    return 2.0 * W2_MAKER_FEE_RATE + 2.0 * (bp * 0.0001) * stress_multiplier


def rolling_trailing_avg_volume(quote_volume_frame: pd.DataFrame, window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """Trailing `window`-day average quote_volume per symbol per day, requiring a FULL
    window (min_periods=window) -- a symbol with fewer than `window` days of history so
    far gets NaN, not a partial-window estimate that would understate how new/unproven it
    is. This alone (before the shift(1) below) already only ever looks backward: pandas'
    rolling() at row t is a function of rows [t-window+1, t], never t+1 or later."""
    return quote_volume_frame.rolling(window, min_periods=window).mean()


def point_in_time_known_avg(quote_volume_frame: pd.DataFrame, window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """The trailing average as of the PREVIOUS bar's close -- i.e. what is actually known
    at the moment day t's t+1-open trade is decided, the same shift(1) lag
    research.wave1.fam_funding's score_frame applies to funding_score before
    carry_position ever sees it. tests/test_wave12.py pins this with a synthetic future
    volume spike that must not move any earlier day's rank."""
    return rolling_trailing_avg_volume(quote_volume_frame, window).shift(1)


def point_in_time_ranks(known_avg_frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank (1 = highest trailing volume), computed independently for
    each row/day over whichever columns have a valid value that day -- a symbol with NaN
    average that day (see point_in_time_known_avg) is excluded from ranking automatically
    (pandas .rank() leaves NaN cells NaN in the output), which slippage_bp_for_rank then
    treats as the fail-closed tail tier."""
    return known_avg_frame.rank(axis=1, ascending=False, method="min")


def _bp_from_rank_array(rank_values: np.ndarray) -> np.ndarray:
    bp = np.full(rank_values.shape, TAIL_SLIPPAGE_BP, dtype=float)
    assigned = np.zeros(rank_values.shape, dtype=bool)
    valid = ~np.isnan(rank_values)
    for max_rank_inclusive, bp_value in RANK_TIER_BOUNDS:
        mask = valid & ~assigned & (rank_values <= max_rank_inclusive)
        bp[mask] = bp_value
        assigned |= mask
    return bp


def bp_frame_from_ranks(ranks_frame: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Vectorized application of slippage_bp_for_rank's own tier boundaries (kept
    numerically identical to it -- tests/test_wave12.py cross-checks the two) over a
    whole rank frame at once, reindexed to exactly `symbols`, with the BTC/ETH override
    applied last so it always wins regardless of their (already-top) rank."""
    ranks = ranks_frame.reindex(columns=list(symbols))
    values = _bp_from_rank_array(ranks.to_numpy(dtype=float))
    frame = pd.DataFrame(values, index=ranks.index, columns=list(symbols))
    for major in MAJOR_SYMBOLS:
        if major in frame.columns:
            frame[major] = MAJOR_SLIPPAGE_BP
    return frame


def build_cost_and_liquidity_frames(
    quote_volume_frame: pd.DataFrame,
    symbols: tuple[str, ...],
    stress_multiplier: float = 1.0,
    window: int = ROLLING_WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the two frames research/wave12_frontier/engine12.py's loop needs, both
    reindexed to exactly `symbols` (columns) and `quote_volume_frame`'s own index (rows):

      - cost_rate_frame: one-way cost rate (maker + tiered slippage) per symbol per day.
        Never NaN -- any symbol/day with no valid rank yet falls back to the tail tier via
        slippage_bp_for_rank/bp_frame_from_ranks, so this frame is always safe to multiply
        directly into the engine's capital-compounding arithmetic without a separate NaN
        guard downstream.
      - liquidity_ok_frame: bool, True iff that symbol's point-in-time trailing average
        quote_volume is >= LIQUIDITY_FLOOR_USDT that day. NaN trailing average
        (insufficient history) compares False, the correct conservative outcome (a symbol
        without enough history to even measure its own liquidity cannot be assumed
        liquid) -- pandas' `NaN >= x` is False, so no explicit fillna is needed here.

    `quote_volume_frame` should be built over research/wave12_frontier's shared
    tier-reference pool (every symbol that cleared the loosest history floor used by any
    of U0-U6), NOT just `symbols` -- see this module's docstring for why rank must be
    computed against one consistent shared pool across every config.
    """
    known_avg = point_in_time_known_avg(quote_volume_frame, window)
    ranks = point_in_time_ranks(known_avg)
    bp_frame = bp_frame_from_ranks(ranks, symbols)
    cost_rate_frame = cost_rate_from_bp(bp_frame, stress_multiplier)
    liquidity_ok_frame = known_avg.reindex(columns=list(symbols)) >= LIQUIDITY_FLOOR_USDT
    return cost_rate_frame, liquidity_ok_frame


__all__ = [
    "LIQUIDITY_FLOOR_USDT",
    "MAJOR_SLIPPAGE_BP",
    "MAJOR_SYMBOLS",
    "RANK_TIER_BOUNDS",
    "ROLLING_WINDOW_DAYS",
    "TAIL_SLIPPAGE_BP",
    "bp_frame_from_ranks",
    "build_cost_and_liquidity_frames",
    "cost_rate_from_bp",
    "point_in_time_known_avg",
    "point_in_time_ranks",
    "rolling_trailing_avg_volume",
    "slippage_bp_for_rank",
]
