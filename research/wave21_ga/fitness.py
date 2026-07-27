# Wave-21 GA fitness engine.
#
# WHY a NEW engine instead of reusing research.wave13_liquidity.engine13 /
# research.wave18_idle.engine18 directly: those engines run a Python per-timestamp loop
# (~2,500 days x up to 300 symbols). Measured empirically before writing this module: a single
# engine13.run_candidate(L4, breadth=200) call takes ~164s wall-clock. This wave needs
# 1,500 evaluations x 5 seeds x 2 (GA + random control) = 15,000 backtests (SPEC.md's frozen GA
# settings) -- at 164s each that is roughly a month of wall-clock time, which is not a
# "make caching help" problem, it is an infeasible-by-two-orders-of-magnitude problem. This
# module therefore re-derives the SAME per-day economics (gap PnL, turnover cost, intraday PnL,
# entry/exit hysteresis, final forced unwind -- every formula is copied line-for-line from
# engine13._run_liquidity_loop / engine18._run_idle_overlay_loop, not reinvented) in a
# vectorized form with NO Python loop over days:
#   1. Entry/exit hysteresis (research.wave1.fam_funding.carry_position's per-timestamp
#      if/elif state machine) is provably equivalent to "mark +1 where score crosses above
#      entry, mark 0 where score crosses below exit, forward-fill everything in between" --
#      see vectorized_hysteresis's own docstring and
#      tests/test_wave21.py::test_vectorized_hysteresis_matches_reference_carry_position, which
#      pins this against the imported reference function bit-for-bit.
#   2. Top-k ranking per day (`score.nlargest(top_k)`) is a per-ROW top-k selection over a
#      (days x symbols) score matrix -- vectorized via one numpy argsort per call, not a loop
#      over days (_select_top_k).
#   3. Once the weights matrix is fixed, capital compounding is a PURE cumulative product:
#      every per-day term (gap return, cost, intraday return) depends only on that day's AND
#      the previous day's weights plus market data -- NEVER on the running capital level
#      itself. So the whole day-loop collapses to a handful of vectorized numpy ops (elementwise
#      products, one row-sum per day) followed by np.cumprod. This is the single biggest lever:
#      it removes the O(days) Python loop entirely.
# tests/test_wave21.py::test_vectorized_engine_matches_reference_engine13 backtests the SAME
# small synthetic market through BOTH this module's run_backtest AND engine13's own
# _run_liquidity_loop and asserts the equity curves agree to float precision -- that test is
# the correctness anchor this whole wave's numbers rest on.
#
# OOS SEALING (SPEC.md 오염 차단 장치 1, 최중요): the evolution loop (ga.py, random_search.py)
# calls ONLY evaluate_genome(), which internally hardcodes mode=MODE_IS and never accepts a
# caller-supplied mode -- there is no code path by which ga.py/random_search.py can request
# MODE_OOS_FINAL even by mistake (structural enforcement, same "cannot physically happen by
# construction" pattern research/wave18_idle/engine18.py's L4-first day-loop already uses for
# S6). run_backtest(mode=MODE_IS) additionally slices every market-data array to a CONTIGUOUS
# PREFIX ending at OOS_SPLIT before any ranking/weight/cost computation touches it, and asserts
# (raising OOSLeakageError) that the resulting equity index never exceeds OOS_SPLIT -- the
# empirical backstop, mirroring gates18.gate_s6_recoverability's "structural claim + empirical
# gate" pattern. oos_slice() is the ONLY function anywhere in this package allowed to read
# OOS-range (> OOS_SPLIT) data, and it raises OOSLeakageError unless called with
# mode=MODE_OOS_FINAL. final_evaluation() -- called at most ONCE, only by run_wave21.py's
# `gates` stage, only on the single selected final candidate -- is the only caller of
# run_backtest(mode=MODE_OOS_FINAL) and oos_slice(mode=MODE_OOS_FINAL) in this package.

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

from research.validation.deep_stats import DeepValidationError, TimedValue, deflated_sharpe
from research.wave1.fam_funding import FundingCandidate, funding_score
from research.wave10_carry100.engine import ACTIVE_CAPITAL, OOS_SPLIT
from research.wave10_carry100.regime import regime_breakdown
from research.wave13_liquidity import costs_measured, engine13
from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.configs13 import Wave13Config
from research.wave18_idle import engine18, fetch18
from research.wave18_idle.configs18 import MAJORS_ONLY_SYMBOLS, OVERLAY_CARRY_CANDIDATE
from research.wave21_ga import genome as genome_mod
from research.wave21_ga.genome import Genome, genome_key

MODE_IS: Final = "IS"
MODE_OOS_FINAL: Final = "OOS_FINAL"
VALID_MODES: Final[tuple[str, ...]] = (MODE_IS, MODE_OOS_FINAL)

N_FOLDS: Final = 4
MDD_FOLD_FLOOR: Final = 0.10
MDD_PENALTY_WEIGHT: Final = 0.5

_MAX_BREADTH: Final = max(genome_mod.UNIVERSE_BREADTH_CHOICES)
_GA_UNIVERSE_CONFIG: Final = Wave13Config(
    FundingCandidate("GA_UNIVERSE", 7, 0.15, 1),  # placeholder candidate: universe_liquidity/build_cost_and_liquidity_frames only read universe_kind/breadth/history_months off this config, never candidate's own fields
    0.50,
    "breadth",
    None,
    _MAX_BREADTH,
    12.0,
    None,
    None,
    "wave21_ga superset universe loader (breadth=300, 12mo) -- every genome's own (smaller) universe_breadth is a column SLICE of this, never a separate fetch/load.",
)


class OOSLeakageError(Exception):
    """Raised the moment OOS-range (> OOS_SPLIT) data is touched outside mode=MODE_OOS_FINAL."""


# ---------------------------------------------------------------------------
# Market cache: every genome-INDEPENDENT computation, built exactly once and reused across
# all 15,000 evaluations (task instruction: '유니버스 데이터를 사전 로드해 메모리에 두고
# 재사용'). Frame columns are always `symbols`, in descending-30d-reference-volume order
# (universe_liquidity.symbols_for_config's own ordering) -- universe_breadth's per-genome
# column slicing (breadth_masks) depends on this order being stable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketCache:
    symbols: tuple[str, ...]
    index: pd.DatetimeIndex
    is_row_mask: np.ndarray  # bool[days] -- index <= OOS_SPLIT; a CONTIGUOUS prefix (asserted at build time)
    spot_open: np.ndarray
    spot_close: np.ndarray
    perp_open: np.ndarray
    perp_close: np.ndarray
    funding: np.ndarray
    available: np.ndarray  # bool[days, symbols]
    liquidity_ok: np.ndarray  # bool[days, symbols]
    cost_rate: np.ndarray  # float[days, symbols], stress_multiplier=1.0
    cost_rate_stress: np.ndarray  # float[days, symbols], stress_multiplier=engine13.STRESS_MULTIPLIER (3.0)
    raw_score_by_window: dict[int, pd.DataFrame]  # window_days -> unshifted funding-APR score frame (pandas, needed for .shift/.ffill)
    breadth_masks: dict[int, np.ndarray]  # universe_breadth -> bool[symbols]
    overlay_selected: np.ndarray  # bool[days, symbols] -- I2/I5's OWN fixed overlay (window=7/entry=8%/exit=4%/top_k=1/majors-only), genome-independent
    lending_daily_rate: float


def _select_top_k(scores: np.ndarray, eligible: np.ndarray, top_k: int) -> np.ndarray:
    """Vectorized per-row top-k selection: True at the top_k highest `scores` cells among
    `eligible` cells, per row -- the array analogue of
    research.wave1.fam_funding's `score_frame.loc[t, eligible].dropna().nlargest(top_k).index`,
    applied to every row at once via one np.argsort call instead of a per-day Python loop.
    Fewer than top_k eligible on a given row selects only those (matches nlargest's own
    truncating behavior when fewer candidates exist than requested)."""
    masked = np.where(eligible, scores, -np.inf)
    n_days, n_symbols = masked.shape
    k = min(top_k, n_symbols)
    if k <= 0 or n_days == 0:
        return np.zeros((n_days, n_symbols), dtype=bool)
    order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n_days), k)
    cols = order.ravel()
    valid = np.isfinite(masked[rows, cols])
    selected = np.zeros((n_days, n_symbols), dtype=bool)
    selected[rows[valid], cols[valid]] = True
    return selected


def vectorized_hysteresis(raw_score_frame: pd.DataFrame, entry_threshold_apr: float, exit_threshold_apr: float) -> pd.DataFrame:
    """Vectorized generalization of research.wave1.fam_funding.carry_position (which hardcodes
    exit = entry/2.0; SPEC.md's exit_threshold_ratio gene needs an arbitrary exit level). The
    reference is a per-timestamp state machine: enter (state=1) when score > entry, exit
    (state=0) when score < exit, otherwise CARRY FORWARD the previous state; the whole result
    is then shift(1)'d (yesterday's decision governs today's position, avoiding lookahead).
    That recurrence is exactly what mark-then-forward-fill encodes: write 1.0 where score
    triggers entry, 0.0 where score triggers exit, leave every other cell NaN (no new decision
    that day), then ffill each NaN run forward to its last real decision -- NaN before the very
    first decision fills to 0.0, matching carry_position's own `active = 0.0` initial state.
    No Python loop over time; pandas' ffill is a single vectorized (Cython) pass.
    tests/test_wave21.py pins this against the imported reference function bit-for-bit at
    exit_threshold_apr == entry_threshold_apr/2.0 (carry_position's own fixed ratio)."""
    raw = pd.DataFrame(np.nan, index=raw_score_frame.index, columns=raw_score_frame.columns)
    raw = raw.mask(raw_score_frame > entry_threshold_apr, 1.0)
    raw = raw.mask(raw_score_frame < exit_threshold_apr, 0.0)
    active_today = raw.ffill().fillna(0.0)
    return active_today.shift(1).fillna(0.0)


def build_market_cache() -> MarketCache:
    pool = ul.load_candidate_pool()
    symbols = ul.verify_cache_and_load_symbols(_GA_UNIVERSE_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    missing = [symbol for symbol in symbols if symbol not in markets]
    if missing:
        raise RuntimeError(f"wave21_ga market cache: {len(missing)} symbols missing from load_markets_for_symbols despite passing the cache check: {missing[:5]}")

    spot_open_cols: dict[str, pd.Series] = {}
    spot_close_cols: dict[str, pd.Series] = {}
    perp_open_cols: dict[str, pd.Series] = {}
    perp_close_cols: dict[str, pd.Series] = {}
    funding_cols: dict[str, pd.Series] = {}
    raw_by_window: dict[int, dict[str, pd.Series]] = {window: {} for window in genome_mod.WINDOW_DAYS_CHOICES}
    for symbol in symbols:
        market = markets[symbol]
        funding_daily = market.funding.resample("1D").sum()
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        spot_open_cols[symbol] = spot_daily["open"]
        spot_close_cols[symbol] = spot_daily["close"]
        perp_open_cols[symbol] = perp_daily["open"]
        perp_close_cols[symbol] = perp_daily["close"]
        funding_cols[symbol] = funding_daily
        for window in genome_mod.WINDOW_DAYS_CHOICES:
            raw_by_window[window][symbol] = funding_score(market.funding, window).resample("1D").last()

    ordered = list(symbols)
    spot_open_frame = pd.DataFrame(spot_open_cols)[ordered].sort_index()
    index = spot_open_frame.index
    spot_close_frame = pd.DataFrame(spot_close_cols)[ordered].reindex(index)
    perp_open_frame = pd.DataFrame(perp_open_cols)[ordered].reindex(index)
    perp_close_frame = pd.DataFrame(perp_close_cols)[ordered].reindex(index)
    funding_frame = pd.DataFrame(funding_cols)[ordered].reindex(index).fillna(0.0)
    raw_score_by_window = {
        window: pd.DataFrame(raw_by_window[window])[ordered].reindex(index) for window in genome_mod.WINDOW_DAYS_CHOICES
    }

    available = (
        spot_open_frame.notna().to_numpy()
        & spot_close_frame.notna().to_numpy()
        & perp_open_frame.notna().to_numpy()
        & perp_close_frame.notna().to_numpy()
    )

    mapping = costs_measured.fit_mapping()
    cost_rate_frame, liquidity_ok_frame = engine13.build_cost_and_liquidity_frames(_GA_UNIVERSE_CONFIG, symbols, index, mapping, engine13.DEFAULT_STRESS_MULTIPLIER)
    cost_rate_frame_stress, _ = engine13.build_cost_and_liquidity_frames(_GA_UNIVERSE_CONFIG, symbols, index, mapping, engine13.STRESS_MULTIPLIER)

    breadth_masks: dict[int, np.ndarray] = {}
    for breadth in genome_mod.UNIVERSE_BREADTH_CHOICES:
        mask = np.zeros(len(symbols), dtype=bool)
        mask[: min(breadth, len(symbols))] = True
        breadth_masks[breadth] = mask

    majors_mask = np.array([symbol in MAJORS_ONLY_SYMBOLS for symbol in symbols], dtype=bool)
    overlay_raw = raw_score_by_window[OVERLAY_CARRY_CANDIDATE.window_days]
    overlay_entry = OVERLAY_CARRY_CANDIDATE.threshold_apr
    overlay_exit = overlay_entry / 2.0  # carry_position's own fixed exit ratio -- I2/I5's overlay is 100% frozen (wave18), not one of this wave's evolved genes
    overlay_active_lagged = vectorized_hysteresis(overlay_raw, overlay_entry, overlay_exit).to_numpy(dtype=float)
    overlay_ranking = overlay_raw.shift(1).to_numpy(dtype=float)
    liquidity_ok_array = liquidity_ok_frame.to_numpy(dtype=bool)
    overlay_eligible = (
        (overlay_active_lagged > 0.0) & available & liquidity_ok_array & majors_mask[None, :] & np.isfinite(overlay_ranking)
    )
    overlay_selected = _select_top_k(overlay_ranking, overlay_eligible, OVERLAY_CARRY_CANDIDATE.top_k)

    lending_apr = fetch18.load_usdt_lending_apr(conservative=True)
    lending_daily_rate = engine18.daily_rate_from_apr(lending_apr)

    is_row_mask = np.asarray(index <= OOS_SPLIT, dtype=bool)
    prefix_length = int(is_row_mask.sum())
    if prefix_length > 0 and not bool(is_row_mask[:prefix_length].all()):
        raise RuntimeError("wave21_ga market cache: IS row mask is not a contiguous prefix of `index` -- index must be sorted ascending")

    _ = pool  # loaded only to fail loudly/early if wave12_frontier's cache is missing; membership itself came from verify_cache_and_load_symbols
    return MarketCache(
        symbols=symbols,
        index=index,
        is_row_mask=is_row_mask,
        spot_open=spot_open_frame.to_numpy(dtype=float),
        spot_close=spot_close_frame.to_numpy(dtype=float),
        perp_open=perp_open_frame.to_numpy(dtype=float),
        perp_close=perp_close_frame.to_numpy(dtype=float),
        funding=funding_frame.to_numpy(dtype=float),
        available=available,
        liquidity_ok=liquidity_ok_array,
        cost_rate=cost_rate_frame.to_numpy(dtype=float),
        cost_rate_stress=cost_rate_frame_stress.to_numpy(dtype=float),
        raw_score_by_window=raw_score_by_window,
        breadth_masks=breadth_masks,
        overlay_selected=overlay_selected,
        lending_daily_rate=lending_daily_rate,
    )


def market_cache_from_markets(
    markets: dict[str, "engine18.FundingMarket"],  # noqa: F821 -- forward-referenced for the docstring only; see import note below
    *,
    majors: tuple[str, ...] = (),
    flat_cost_rate: float = 0.001,
    flat_stress_cost_rate: float | None = None,
    always_liquid: bool = True,
    lending_daily_rate: float = 0.0,
) -> MarketCache:
    """Test-only constructor: builds a MarketCache directly from an in-memory `markets` dict
    (research.wave1.fam_funding.FundingMarket values) with a FLAT constant cost rate and an
    always-liquid mask -- no disk I/O, no costs_measured/universe_liquidity dependency. Mirrors
    research/wave18_idle/tests/test_wave18.py's own synthetic-market + `flat_cost`/
    `always_liquid` fixture convention. Every engine-equivalence/unit test in
    tests/test_wave21.py builds its MarketCache this way; only build_market_cache() itself
    (exercised against the real repo cache by run_wave21.py, and by a slower cache-integration
    smoke test) reads from disk."""
    symbols = tuple(markets.keys())
    spot_open_cols: dict[str, pd.Series] = {}
    spot_close_cols: dict[str, pd.Series] = {}
    perp_open_cols: dict[str, pd.Series] = {}
    perp_close_cols: dict[str, pd.Series] = {}
    funding_cols: dict[str, pd.Series] = {}
    raw_by_window: dict[int, dict[str, pd.Series]] = {window: {} for window in genome_mod.WINDOW_DAYS_CHOICES}
    for symbol in symbols:
        market = markets[symbol]
        funding_daily = market.funding.resample("1D").sum()
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        spot_open_cols[symbol] = spot_daily["open"]
        spot_close_cols[symbol] = spot_daily["close"]
        perp_open_cols[symbol] = perp_daily["open"]
        perp_close_cols[symbol] = perp_daily["close"]
        funding_cols[symbol] = funding_daily
        for window in genome_mod.WINDOW_DAYS_CHOICES:
            raw_by_window[window][symbol] = funding_score(market.funding, window).resample("1D").last()

    ordered = list(symbols)
    spot_open_frame = pd.DataFrame(spot_open_cols)[ordered].sort_index()
    index = spot_open_frame.index
    spot_close_frame = pd.DataFrame(spot_close_cols)[ordered].reindex(index)
    perp_open_frame = pd.DataFrame(perp_open_cols)[ordered].reindex(index)
    perp_close_frame = pd.DataFrame(perp_close_cols)[ordered].reindex(index)
    funding_frame = pd.DataFrame(funding_cols)[ordered].reindex(index).fillna(0.0)
    raw_score_by_window = {
        window: pd.DataFrame(raw_by_window[window])[ordered].reindex(index) for window in genome_mod.WINDOW_DAYS_CHOICES
    }

    available = (
        spot_open_frame.notna().to_numpy()
        & spot_close_frame.notna().to_numpy()
        & perp_open_frame.notna().to_numpy()
        & perp_close_frame.notna().to_numpy()
    )
    n_days, n_symbols = available.shape
    cost_rate = np.full((n_days, n_symbols), flat_cost_rate, dtype=float)
    cost_rate_stress = np.full((n_days, n_symbols), flat_cost_rate if flat_stress_cost_rate is None else flat_stress_cost_rate, dtype=float)
    liquidity_ok = np.full((n_days, n_symbols), always_liquid, dtype=bool)
    breadth_masks = {breadth: np.ones(n_symbols, dtype=bool) for breadth in genome_mod.UNIVERSE_BREADTH_CHOICES}

    majors_mask = np.array([symbol in majors for symbol in ordered], dtype=bool)
    overlay_raw = raw_score_by_window[OVERLAY_CARRY_CANDIDATE.window_days]
    overlay_entry = OVERLAY_CARRY_CANDIDATE.threshold_apr
    overlay_exit = overlay_entry / 2.0
    overlay_active_lagged = vectorized_hysteresis(overlay_raw, overlay_entry, overlay_exit).to_numpy(dtype=float)
    overlay_ranking = overlay_raw.shift(1).to_numpy(dtype=float)
    overlay_eligible = (overlay_active_lagged > 0.0) & available & liquidity_ok & majors_mask[None, :] & np.isfinite(overlay_ranking)
    overlay_selected = _select_top_k(overlay_ranking, overlay_eligible, OVERLAY_CARRY_CANDIDATE.top_k)

    is_row_mask = np.asarray(index <= OOS_SPLIT, dtype=bool)

    return MarketCache(
        symbols=tuple(ordered),
        index=index,
        is_row_mask=is_row_mask,
        spot_open=spot_open_frame.to_numpy(dtype=float),
        spot_close=spot_close_frame.to_numpy(dtype=float),
        perp_open=perp_open_frame.to_numpy(dtype=float),
        perp_close=perp_close_frame.to_numpy(dtype=float),
        funding=funding_frame.to_numpy(dtype=float),
        available=available,
        liquidity_ok=liquidity_ok,
        cost_rate=cost_rate,
        cost_rate_stress=cost_rate_stress,
        raw_score_by_window=raw_score_by_window,
        breadth_masks=breadth_masks,
        overlay_selected=overlay_selected,
        lending_daily_rate=lending_daily_rate,
    )


# ---------------------------------------------------------------------------
# Per-genome backtest.
# ---------------------------------------------------------------------------


def _compound_factor(cache: MarketCache, cutoff: int, weights: np.ndarray, cost_rate: np.ndarray, lending_day: np.ndarray) -> np.ndarray:
    """The vectorized replacement for engine13/engine18's own per-day capital multiplication --
    see module docstring point 3 for why this is valid (every term below depends only on
    THIS day's and the PREVIOUS day's weights plus market data, never on the running capital
    level itself, so the whole day-loop reduces to elementwise array ops + one final cumprod
    in run_backtest)."""
    spot_open = cache.spot_open[:cutoff]
    spot_close = cache.spot_close[:cutoff]
    perp_open = cache.perp_open[:cutoff]
    perp_close = cache.perp_close[:cutoff]
    funding = cache.funding[:cutoff]
    n_symbols = weights.shape[1]

    if cutoff > 1:
        previous_weights = np.vstack([np.zeros((1, n_symbols)), weights[:-1]])
        spot_close_prev = np.vstack([np.full((1, n_symbols), np.nan), spot_close[:-1]])
        perp_close_prev = np.vstack([np.full((1, n_symbols), np.nan), perp_close[:-1]])
    else:
        previous_weights = np.zeros_like(weights)
        spot_close_prev = np.full_like(spot_close, np.nan)
        perp_close_prev = np.full_like(perp_close, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        spot_gap = spot_open / spot_close_prev - 1.0
        perp_gap = perp_open / perp_close_prev - 1.0
    gap_by_symbol = np.nan_to_num(spot_gap - perp_gap, nan=0.0, posinf=0.0, neginf=0.0)
    gap_return = np.sum(gap_by_symbol * previous_weights, axis=1)

    cost_return = np.sum(np.abs(weights - previous_weights) * cost_rate, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        intraday = spot_close / spot_open - perp_close / perp_open
    intraday = np.nan_to_num(intraday + funding, nan=0.0, posinf=0.0, neginf=0.0)
    intraday_return = np.sum(intraday * weights, axis=1)

    factor = (1.0 + gap_return) * (1.0 - cost_return) * (1.0 + intraday_return)
    return np.where(lending_day, factor * (1.0 + cache.lending_daily_rate), factor)


def run_backtest(genome: Genome, cache: MarketCache, mode: str, stress: bool = False) -> pd.Series:
    """Runs ONE genome end to end and returns its equity curve (native USD, starting at
    ACTIVE_CAPITAL). mode=MODE_IS truncates every array to the contiguous IS prefix BEFORE any
    ranking/weight/cost computation -- see module docstring's OOS SEALING section. This
    function is never called directly by ga.py/random_search.py (they only see
    evaluate_genome(), which hardcodes mode=MODE_IS) nor by anything computing the final
    candidate's OOS number except final_evaluation() (which hardcodes mode=MODE_OOS_FINAL)."""
    if mode not in VALID_MODES:
        raise ValueError(f"run_backtest: unknown mode {mode!r}, expected one of {VALID_MODES}")

    cutoff = int(cache.is_row_mask.sum()) if mode == MODE_IS else len(cache.index)
    if cutoff < 2 * N_FOLDS:
        raise ValueError(f"run_backtest: sliced date range too short ({cutoff} rows) for mode={mode!r}")
    index = cache.index[:cutoff]

    raw = cache.raw_score_by_window[genome.window_days].iloc[:cutoff]
    exit_threshold_apr = genome.entry_threshold_apr * genome.exit_threshold_ratio
    active_lagged = vectorized_hysteresis(raw, genome.entry_threshold_apr, exit_threshold_apr).to_numpy(dtype=float)
    ranking = raw.shift(1).to_numpy(dtype=float)

    available = cache.available[:cutoff]
    liquidity_ok = cache.liquidity_ok[:cutoff]
    breadth_mask = cache.breadth_masks[genome.universe_breadth]
    l4_eligible = (active_lagged > 0.0) & available & liquidity_ok & breadth_mask[None, :] & np.isfinite(ranking)
    l4_selected = _select_top_k(ranking, l4_eligible, genome.top_k_pairs)
    l4_weights = np.where(l4_selected, genome.leg_fraction, 0.0)
    l4_has = l4_selected.any(axis=1)

    n_symbols = len(cache.symbols)
    if genome.uses_overlay:
        overlay_selected = cache.overlay_selected[:cutoff]
        overlay_weights = np.where(overlay_selected, genome.leg_fraction, 0.0)
        overlay_has = overlay_selected.any(axis=1)
    else:
        overlay_weights = np.zeros((cutoff, n_symbols), dtype=float)
        overlay_has = np.zeros(cutoff, dtype=bool)

    use_overlay_today = overlay_has & (~l4_has)
    weights = np.where(l4_has[:, None], l4_weights, np.where(use_overlay_today[:, None], overlay_weights, 0.0))
    lending_day = (~l4_has) & (~use_overlay_today) if genome.uses_lending else np.zeros(cutoff, dtype=bool)

    cost_rate = (cache.cost_rate_stress if stress else cache.cost_rate)[:cutoff]
    factor = _compound_factor(cache, cutoff, weights, cost_rate, lending_day)

    equity_values = ACTIVE_CAPITAL * np.cumprod(factor)
    final_weights = weights[-1]
    if np.sum(np.abs(final_weights)) > 0.0:
        final_cost = float(np.sum(np.abs(final_weights) * cost_rate[-1]))
        equity_values[-1] *= 1.0 - final_cost
    equity = pd.Series(equity_values, index=index, dtype=float)

    if mode == MODE_IS and bool((equity.index > OOS_SPLIT).any()):
        # Unreachable given the prefix slice above (cutoff is derived from is_row_mask, which
        # is itself asserted contiguous at cache-build time) -- kept as a loud, immediate
        # failure if a future edit to the slicing above ever regresses this, matching
        # gates18.gate_s6_recoverability's "structural claim + empirical gate" convention.
        raise OOSLeakageError("run_backtest: IS-mode equity index extends past OOS_SPLIT")
    return equity


def oos_slice(equity: pd.Series, mode: str) -> pd.Series:
    """The ONLY function in this package allowed to read OOS-range (> OOS_SPLIT) data. Raises
    OOSLeakageError unless `mode` explicitly identifies itself as the one-time final
    evaluation. SPEC.md: 'OOS(2025-10~)는 최종 1개 개체에 대해 단 한 번만 평가... 진화 루프가
    OOS에 접근하면 코드 레벨에서 예외 발생'."""
    if mode != MODE_OOS_FINAL:
        raise OOSLeakageError(f"oos_slice: called with mode={mode!r} -- OOS data is sealed until mode={MODE_OOS_FINAL!r}")
    return equity[equity.index > OOS_SPLIT]


# ---------------------------------------------------------------------------
# Walk-forward IS fitness (SPEC.md 오염 차단 장치 2): median(fold CAGR) - std(fold CAGR) -
# 0.5*max(0, MDD-10%), folds = 4 contiguous chronological slices of the SAME single IS equity
# curve (no per-fold re-optimization -- the GA/random search itself IS the outer search loop;
# re-fitting per fold would be a nested search this wave's budget (1,500 evals/seed) has no
# room for, and SPEC.md's own formula reads as consistency-across-sub-periods of one continuous
# curve, not nested walk-forward optimization).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitnessResult:
    fitness: float
    fold_cagrs: tuple[float, ...]
    median_fold_cagr: float
    std_fold_cagr: float
    mdd: float
    mdd_penalty: float
    full_is_cagr: float


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start_value, end_value = float(equity.iloc[0]), float(equity.iloc[-1])
    if start_value <= 0.0:
        return 0.0
    days = max((pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).total_seconds() / 86_400.0, 1.0)
    growth = end_value / start_value
    return float(growth ** (365.0 / days) - 1.0) if growth > 0.0 else -1.0


def _max_drawdown(equity: pd.Series) -> float:
    values = equity.to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.nan_to_num(1.0 - values / peaks, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(drawdown))


def walk_forward_fitness(is_equity: pd.Series, n_folds: int = N_FOLDS) -> FitnessResult:
    n = len(is_equity)
    if n < n_folds * 2:
        raise ValueError(f"walk_forward_fitness: IS equity series too short ({n} obs) for {n_folds}-fold walk-forward")
    edges = [int(round(i * n / n_folds)) for i in range(n_folds + 1)]
    edges[0], edges[-1] = 0, n
    fold_cagrs: list[float] = []
    for fold_index in range(n_folds):
        lo, hi = edges[fold_index], edges[fold_index + 1]
        hi = max(hi, min(lo + 2, n))  # guard a degenerate 0/1-length fold from integer rounding
        fold_cagrs.append(cagr(is_equity.iloc[lo:hi]))
    fold_array = np.asarray(fold_cagrs, dtype=float)
    median_fold = float(np.median(fold_array))
    std_fold = float(np.std(fold_array, ddof=0))
    mdd = _max_drawdown(is_equity)
    mdd_penalty = MDD_PENALTY_WEIGHT * max(0.0, mdd - MDD_FOLD_FLOOR)
    fitness = median_fold - std_fold - mdd_penalty
    return FitnessResult(
        fitness=fitness,
        fold_cagrs=tuple(fold_cagrs),
        median_fold_cagr=median_fold,
        std_fold_cagr=std_fold,
        mdd=mdd,
        mdd_penalty=mdd_penalty,
        full_is_cagr=cagr(is_equity),
    )


def evaluate_genome(genome: Genome, cache: MarketCache) -> FitnessResult:
    """The ONLY fitness entry point ga.py/random_search.py ever call. Mode is hardcoded to
    MODE_IS -- there is no parameter here through which a caller could request OOS data."""
    equity = run_backtest(genome, cache, MODE_IS)
    return walk_forward_fitness(equity)


def evaluate_genome_cached(genome: Genome, cache: MarketCache, fitness_cache: dict[tuple, FitnessResult]) -> tuple[FitnessResult, bool]:
    """evaluate_genome with a caller-owned cache dict (task instruction: '평가 캐싱(동일 유전자
    -> 결과 재사용) 필수'). Returns (result, was_cache_hit) so callers can report how many of
    their nominal 1,500 evaluations were actual fresh backtests vs. reused lookups."""
    key = genome_key(genome)
    if key in fitness_cache:
        return fitness_cache[key], True
    result = evaluate_genome(genome, cache)
    fitness_cache[key] = result
    return result, False


# ---------------------------------------------------------------------------
# One-time final evaluation (OOS seal opened here, and ONLY here).
# ---------------------------------------------------------------------------


class _EquityOnly:
    """Minimal duck-typed stand-in so research.wave10_carry100.regime.regime_breakdown (which
    only ever reads `.equity`) can run on a bare pd.Series -- same pattern gates18.py's own
    _EquityOnly uses."""

    def __init__(self, equity: pd.Series) -> None:
        self.equity = equity


@dataclass(frozen=True, slots=True)
class FinalEvaluation:
    genome: Genome
    full_equity: pd.Series
    is_equity: pd.Series
    oos_equity: pd.Series
    stress_equity: pd.Series
    full_period_cagr: float
    is_cagr: float
    oos_cagr_self_contained: float  # cagr(oos_slice) -- OOS window's OWN start/end, ignoring IS continuity
    oos_cagr_regime_anchored: float | None  # regime_breakdown's OOS_SPLIT-anchored figure -- the apples-to-apples number vs I5.json's own saved current_low_funding.annualized_return
    mdd_full: float
    regime_breakdown: dict


def final_evaluation(genome: Genome, cache: MarketCache) -> FinalEvaluation:
    """Called AT MOST ONCE per wave run, on the single selected final candidate. This is the
    only function in this package that runs the backtest over the OOS range at all."""
    full_equity = run_backtest(genome, cache, MODE_OOS_FINAL, stress=False)
    stress_equity = run_backtest(genome, cache, MODE_OOS_FINAL, stress=True)
    is_equity = full_equity[full_equity.index <= OOS_SPLIT]  # reading the <= side is never gated; only the > OOS_SPLIT side goes through oos_slice
    oos_equity = oos_slice(full_equity, MODE_OOS_FINAL)
    regime = regime_breakdown(_EquityOnly(full_equity))
    current_low_funding = regime.get("current_low_funding")
    oos_anchored = current_low_funding.get("annualized_return") if isinstance(current_low_funding, dict) else None
    return FinalEvaluation(
        genome=genome,
        full_equity=full_equity,
        is_equity=is_equity,
        oos_equity=oos_equity,
        stress_equity=stress_equity,
        full_period_cagr=cagr(full_equity),
        is_cagr=cagr(is_equity),
        oos_cagr_self_contained=cagr(oos_equity),
        oos_cagr_regime_anchored=oos_anchored,
        mdd_full=_max_drawdown(full_equity),
        regime_breakdown=regime,
    )


def deflated_sharpe_for_trials(equity: pd.Series, trials: int) -> dict | None:
    clean = equity.dropna()
    if len(clean) < 4:
        return None
    timed = tuple(TimedValue(pd.Timestamp(ts).to_pydatetime(), float(value)) for ts, value in clean.items())
    try:
        dsr = deflated_sharpe(timed, trials=trials)
    except DeepValidationError:
        return None
    return {"score": dsr.score, "probability": dsr.probability, "trials": dsr.trials, "observed_sharpe": dsr.observed_sharpe}


__all__ = [
    "MDD_FOLD_FLOOR",
    "MDD_PENALTY_WEIGHT",
    "MODE_IS",
    "MODE_OOS_FINAL",
    "N_FOLDS",
    "VALID_MODES",
    "FinalEvaluation",
    "FitnessResult",
    "MarketCache",
    "OOSLeakageError",
    "build_market_cache",
    "cagr",
    "deflated_sharpe_for_trials",
    "evaluate_genome",
    "evaluate_genome_cached",
    "final_evaluation",
    "oos_slice",
    "run_backtest",
    "vectorized_hysteresis",
    "walk_forward_fitness",
]
