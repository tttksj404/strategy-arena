# Wave-24 GP fitness engine: delta-neutral carry backtest driven by a GP tree's OWN evaluated
# score instead of a fixed funding-score formula.
#
# POSITION STRUCTURE IS FROZEN, ONLY THE SIGNAL IS EVOLVED (task brief, byte-for-byte): "진입
# 구조는 델타중립 캐리로 고정(방향 베팅 금지 -- wave-9 6개중 5개 전소, wave-23 원금82% 손실로
# 실증). GP가 진화시키는 건 '어느 심볼에 언제 들어갈지 판단하는 신호 수식'이지 포지션 구조가
# 아니다." Concretely: spot-long/perp-short 2-leg carry, 1 pair at a time, 50% of active capital
# per leg, 200-symbol universe -- these are research.wave21_ga.genome's own L4/I5 baseline values
# (L4_TOP_K_PAIRS=1, L4_LEG_FRACTION=0.50, L4_UNIVERSE_BREADTH=200), hardcoded as FIXED_* below
# rather than exposed as genes the way wave21_ga's GA exposed them. The GP tree (research.
# wave24_gp.tree.Node) supplies exactly one thing: the per-(day, symbol) score used to decide
# WHEN a symbol enters and WHICH symbol gets picked -- see _score_to_active's own docstring for
# why entry/exit is a single scale-invariant zero-crossing rather than a tunable threshold band.
#
# ENGINE REUSE: the day-loop economics (_compound_factor) and top-k selection (_select_top_k)
# below are copied line-for-line from research.wave21_ga.fitness (minus its idle-capital-overlay
# branch, which this wave does not use -- SPEC.md registers no overlay for GP), REIMPLEMENTED
# LOCALLY rather than imported. This mirrors research.wave23_ga_short.engine23's own explicit
# precedent (its module docstring: "_carry_gap_intraday... copied formula-for-formula from
# research.wave21_ga.fitness._compound_factor") of not cross-importing engine internals between
# sibling GA/GP waves, so this wave stays auditable on its own without a dependency on wave21_ga's
# internals continuing to exist unchanged.
#
# OOS SEALING (same structural pattern as research.wave21_ga.fitness / research.wave23_ga_short.
# engine23): gp.py/random_trees.py call ONLY evaluate_tree()/evaluate_tree_cached(), which
# hardcode mode=MODE_IS -- neither function accepts a caller-supplied mode, so there is no code
# path by which the evolution loop can request OOS data even by mistake. run_backtest(mode=
# MODE_IS) slices every terminal panel to the CONTIGUOUS IS PREFIX (cache.is_row_mask, asserted
# contiguous at cache-build time) BEFORE tree.evaluate() ever touches it, and re-asserts (raising
# OOSLeakageError) that the resulting equity index never exceeds OOS_SPLIT. oos_slice() is the
# only function in this package allowed to read OOS-range data, and only under mode=MODE_OOS_FINAL
# -- final_evaluation() (called at most once, only by run_wave24.py's `gates` stage) is the only
# caller of run_backtest(mode=MODE_OOS_FINAL) / oos_slice(mode=MODE_OOS_FINAL) in this package.

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
from research.wave1.fam_funding import FundingCandidate, FundingMarket, funding_score
from research.wave10_carry100.engine import ACTIVE_CAPITAL, OOS_SPLIT
from research.wave10_carry100.regime import regime_breakdown
from research.wave13_liquidity import costs_measured, engine13
from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.configs13 import Wave13Config
from research.wave20_convex.engine20 import atr, realized_vol
from research.wave24_gp.tree import Node, evaluate, node_count

MODE_IS: Final = "IS"
MODE_OOS_FINAL: Final = "OOS_FINAL"
VALID_MODES: Final[tuple[str, ...]] = (MODE_IS, MODE_OOS_FINAL)

# Position-structure constants -- FIXED (see module docstring), matching
# research.wave21_ga.genome's own L4_TOP_K_PAIRS / L4_LEG_FRACTION / L4_UNIVERSE_BREADTH.
FIXED_TOP_K_PAIRS: Final = 1
FIXED_LEG_FRACTION: Final = 0.50
FIXED_UNIVERSE_BREADTH: Final = 200

# Terminal windows (SPEC.md terminal table).
FUNDING_WINDOWS: Final[dict[str, int]] = {"funding_1d": 1, "funding_7d": 7, "funding_14d": 14, "funding_30d": 30}
PRICE_RET_WINDOWS: Final[dict[str, int]] = {"price_ret_1d": 1, "price_ret_7d": 7, "price_ret_30d": 30}
REALIZED_VOL_WINDOW: Final = 20
ATR_WINDOW: Final = 14
QUOTE_VOLUME_WINDOW: Final = 30

N_FOLDS: Final = 4  # SPEC.md "워크포워드 4폴드"
MDD_FOLD_FLOOR: Final = 0.15  # SPEC.md fitness formula's own "MDD-0.15"
MDD_PENALTY_WEIGHT: Final = 5.0  # SPEC.md fitness formula's own "5*max(0, MDD-0.15)"
NODE_COUNT_PENALTY_WEIGHT: Final = 0.02  # SPEC.md fitness formula's own "0.02*노드수"

_MIN_CUTOFF_DAYS: Final = 120  # floor giving slack for the deepest terminal warmup (30d) plus a meaningful 4-fold split

_MAX_BREADTH: Final = FIXED_UNIVERSE_BREADTH
_GP_UNIVERSE_CONFIG: Final = Wave13Config(
    FundingCandidate("GP_UNIVERSE", 7, 0.15, 1),  # placeholder candidate: universe_liquidity/build_cost_and_liquidity_frames only read universe_kind/breadth/history_months off this config (matches research.wave21_ga.fitness._GA_UNIVERSE_CONFIG's own precedent)
    FIXED_LEG_FRACTION,
    "breadth",
    None,
    _MAX_BREADTH,
    12.0,
    None,
    None,
    "wave24_gp universe loader (breadth=200, matches L4/I5's own frozen position-structure baseline -- GP evolves the SIGNAL only, see module docstring).",
)


class OOSLeakageError(Exception):
    """Raised the moment OOS-range (> OOS_SPLIT) data is touched outside mode=MODE_OOS_FINAL."""


# ---------------------------------------------------------------------------
# Market cache: every tree-INDEPENDENT computation, built exactly once and reused across every
# evaluation this wave ever runs (evolve + control combined: up to ~60,000 tree evaluations).
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
    terminals: dict[str, pd.DataFrame]  # name -> [days, symbols], POINT-IN-TIME as of that row's own close (NOT pre-shifted -- see run_backtest)


def _assemble_terminals(
    ordered: list[str],
    index: pd.DatetimeIndex,
    funding_score_cols: dict[str, dict[str, pd.Series]],
    spot_close_frame: pd.DataFrame,
    perp_close_frame: pd.DataFrame,
    atr_cols: dict[str, pd.Series],
    quote_volume_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Shared by build_market_cache (real cache) and market_cache_from_markets (synthetic test
    fixture) so the terminal FORMULAS themselves can never drift between production and test --
    only how the raw ingredients (funding_score_cols/atr_cols/quote_volume_frame) get built
    differs between the two callers."""
    terminals: dict[str, pd.DataFrame] = {}
    for name in FUNDING_WINDOWS:
        terminals[name] = pd.DataFrame(funding_score_cols[name])[ordered].reindex(index)
    for name, window in PRICE_RET_WINDOWS.items():
        terminals[name] = perp_close_frame.pct_change(window).replace([np.inf, -np.inf], np.nan)
    terminals["realized_vol_20d"] = realized_vol(perp_close_frame, REALIZED_VOL_WINDOW)
    terminals["atr_14"] = pd.DataFrame(atr_cols)[ordered].reindex(index)
    # NOT costs_measured.point_in_time_known_avg -- that helper applies its OWN internal
    # shift(1) (documented as "AS OF THE PREVIOUS BAR'S CLOSE"), which would leave this ONE
    # terminal lagged by 2 days once run_backtest applies its own uniform shift(1) to the tree's
    # final output, while every other terminal is lagged by only 1. Using the un-shifted rolling
    # average keeps all 11 terminals on the identical "as of this row's own close" convention,
    # so a single shift at the score level (not the terminal level) lags everything uniformly.
    terminals["quote_volume_30d"] = costs_measured.rolling_trailing_avg_volume(quote_volume_frame, QUOTE_VOLUME_WINDOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        terminals["basis"] = (perp_close_frame / spot_close_frame - 1.0).replace([np.inf, -np.inf], np.nan)

    # Dtype hygiene (bug found running the real cache, 2026-07-28): a symbol with genuinely EMPTY
    # spot data (0 rows -- confirmed real case: AEROUSDT, whose futures/perp market has rows but
    # whose spot market has none at all) makes pandas infer that column as dtype=object (boxed
    # Python floats/NaN) rather than float64, all the way from the empty source frame through
    # resample/reindex/divide. A DataFrame with even one object-dtype column blows up downstream
    # the moment any node in a GP tree calls a numpy ufunc on it -- e.g. `log`/`abs`/`zscore` on
    # `basis` raises `TypeError: loop of ufunc does not support argument 0 of type float which
    # has no callable log method` (numpy's per-element object-loop fallback, since a plain float
    # has no `.log()` method). This is a pure STORAGE-representation fix, not a value change:
    # `.astype(float)` maps NaN->NaN and every real number to the identical value, just backed by
    # a proper float64 block -- it does not touch tree.py's own deliberate NaN-propagation
    # semantics (module docstring: "That NaN must propagate all the way to the tree's final
    # score"). Applied to every terminal (not just `basis`) defensively, since the same
    # empty-source-frame condition could in principle hit any of them for some other symbol.
    return {name: frame.astype(np.float64) for name, frame in terminals.items()}


def build_market_cache() -> MarketCache:
    pool = ul.load_candidate_pool()
    symbols = ul.verify_cache_and_load_symbols(_GP_UNIVERSE_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    missing = [symbol for symbol in symbols if symbol not in markets]
    if missing:
        raise RuntimeError(f"wave24_gp market cache: {len(missing)} symbols missing from load_markets_for_symbols despite passing the cache check: {missing[:5]}")

    spot_open_cols: dict[str, pd.Series] = {}
    spot_close_cols: dict[str, pd.Series] = {}
    perp_open_cols: dict[str, pd.Series] = {}
    perp_close_cols: dict[str, pd.Series] = {}
    funding_cols: dict[str, pd.Series] = {}
    funding_score_cols: dict[str, dict[str, pd.Series]] = {name: {} for name in FUNDING_WINDOWS}
    atr_cols: dict[str, pd.Series] = {}

    for symbol in symbols:
        market = markets[symbol]
        funding_daily = market.funding.resample("1D").sum()
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        spot_open_cols[symbol] = spot_daily["open"]
        spot_close_cols[symbol] = spot_daily["close"]
        perp_open_cols[symbol] = perp_daily["open"]
        perp_close_cols[symbol] = perp_daily["close"]
        funding_cols[symbol] = funding_daily
        for name, window in FUNDING_WINDOWS.items():
            funding_score_cols[name][symbol] = funding_score(market.funding, window).resample("1D").last()
        atr_cols[symbol] = atr(perp_daily[["high", "low", "close"]], ATR_WINDOW)

    ordered = list(symbols)
    spot_open_frame = pd.DataFrame(spot_open_cols)[ordered].sort_index()
    index = spot_open_frame.index
    spot_close_frame = pd.DataFrame(spot_close_cols)[ordered].reindex(index)
    perp_open_frame = pd.DataFrame(perp_open_cols)[ordered].reindex(index)
    perp_close_frame = pd.DataFrame(perp_close_cols)[ordered].reindex(index)
    funding_frame = pd.DataFrame(funding_cols)[ordered].reindex(index).fillna(0.0)

    available = (
        spot_open_frame.notna().to_numpy()
        & spot_close_frame.notna().to_numpy()
        & perp_open_frame.notna().to_numpy()
        & perp_close_frame.notna().to_numpy()
    )

    mapping = costs_measured.fit_mapping()
    cost_rate_frame, liquidity_ok_frame = engine13.build_cost_and_liquidity_frames(_GP_UNIVERSE_CONFIG, symbols, index, mapping, engine13.DEFAULT_STRESS_MULTIPLIER)
    cost_rate_stress_frame, _ = engine13.build_cost_and_liquidity_frames(_GP_UNIVERSE_CONFIG, symbols, index, mapping, engine13.STRESS_MULTIPLIER)

    quote_volume_frame = ul.load_quote_volume_frame(symbols)[ordered].reindex(index)
    terminals = _assemble_terminals(ordered, index, funding_score_cols, spot_close_frame, perp_close_frame, atr_cols, quote_volume_frame)

    is_row_mask = np.asarray(index <= OOS_SPLIT, dtype=bool)
    prefix_length = int(is_row_mask.sum())
    if prefix_length > 0 and not bool(is_row_mask[:prefix_length].all()):
        raise RuntimeError("wave24_gp market cache: IS row mask is not a contiguous prefix of `index` -- index must be sorted ascending")

    _ = pool  # loaded only to fail loudly/early if wave12_frontier's cache is missing
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
        liquidity_ok=liquidity_ok_frame.to_numpy(dtype=bool),
        cost_rate=cost_rate_frame.to_numpy(dtype=float),
        cost_rate_stress=cost_rate_stress_frame.to_numpy(dtype=float),
        terminals=terminals,
    )


def market_cache_from_markets(
    markets: dict[str, FundingMarket],
    *,
    flat_cost_rate: float = 0.001,
    flat_stress_cost_rate: float | None = None,
    always_liquid: bool = True,
    flat_quote_volume_usdt: float = 1_000_000.0,
) -> MarketCache:
    """Test-only constructor: builds a MarketCache directly from an in-memory `markets` dict
    (research.wave1.fam_funding.FundingMarket values) with a FLAT constant cost rate, an
    always-liquid mask, and a flat synthetic quote-volume series -- no disk I/O. Mirrors
    research.wave21_ga.fitness.market_cache_from_markets / research.wave23_ga_short.engine23.
    market_cache_from_markets's own synthetic-fixture convention. Every non-real-cache test in
    tests/test_wave24.py builds its MarketCache this way; only build_market_cache() itself reads
    from disk."""
    symbols = tuple(markets.keys())
    spot_open_cols: dict[str, pd.Series] = {}
    spot_close_cols: dict[str, pd.Series] = {}
    perp_open_cols: dict[str, pd.Series] = {}
    perp_close_cols: dict[str, pd.Series] = {}
    funding_cols: dict[str, pd.Series] = {}
    funding_score_cols: dict[str, dict[str, pd.Series]] = {name: {} for name in FUNDING_WINDOWS}
    atr_cols: dict[str, pd.Series] = {}

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
        perp_close_cols[symbol] = perp_daily["close"]
        funding_cols[symbol] = funding_daily
        for name, window in FUNDING_WINDOWS.items():
            funding_score_cols[name][symbol] = funding_score(market.funding, window).resample("1D").last()
        atr_cols[symbol] = atr(perp_daily[["high", "low", "close"]], ATR_WINDOW)

    ordered = list(symbols)
    spot_open_frame = pd.DataFrame(spot_open_cols)[ordered].sort_index()
    index = spot_open_frame.index
    spot_close_frame = pd.DataFrame(spot_close_cols)[ordered].reindex(index)
    perp_open_frame = pd.DataFrame(perp_open_cols)[ordered].reindex(index)
    perp_close_frame = pd.DataFrame(perp_close_cols)[ordered].reindex(index)
    funding_frame = pd.DataFrame(funding_cols)[ordered].reindex(index).fillna(0.0)

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

    quote_volume_frame = pd.DataFrame(flat_quote_volume_usdt, index=index, columns=ordered)
    terminals = _assemble_terminals(ordered, index, funding_score_cols, spot_close_frame, perp_close_frame, atr_cols, quote_volume_frame)

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
        terminals=terminals,
    )


# ---------------------------------------------------------------------------
# Per-tree backtest.
# ---------------------------------------------------------------------------


def _select_top_k(scores: np.ndarray, eligible: np.ndarray, top_k: int) -> np.ndarray:
    """Vectorized per-row top-k selection -- copied from research.wave21_ga.fitness._select_top_k
    (see module docstring for why this is reimplemented locally rather than imported)."""
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


def _score_to_active(raw: pd.DataFrame) -> pd.DataFrame:
    """Delta-neutral carry entry/exit, lagged. Structurally the entry=exit=0.0 degenerate case of
    research.wave21_ga.fitness.vectorized_hysteresis (reimplemented locally, not imported -- see
    module docstring): mark 1.0 where today's score is STRICTLY positive, 0.0 where STRICTLY
    negative; a score of exactly 0.0 or NaN gets NO new decision (stays NaN here) and is
    forward-filled from the last real decision, then the whole thing is shifted 1 day so a
    decision computable from day t's own close-of-day data only takes effect starting day t+1.

    WHY a single fixed threshold (0.0) instead of a tunable entry/exit band like wave21_ga's own
    2 genes: SPEC.md's terminal/function alphabet registers no 'entry_threshold' gene at all, and
    a GP tree's OUTPUT SCALE is arbitrary by construction (it might be a raw funding rate ~0.001,
    a price return ~0.1, or an unbounded product of several terminals) -- any fixed NONZERO
    threshold chosen ahead of time would be meaningless for most trees and would itself be an
    unregistered free parameter this wave's SPEC never lists. Zero is the one threshold that is
    scale-invariant (a tree's OWN sign is meaningful regardless of its units) and requires
    inventing nothing. Persistence/smoothing against a noisy near-zero score whipsawing (and
    paying turnover cost on every flip) is deliberately NOT hardcoded as a second threshold here
    -- that is exactly what the `ma`/`zscore` function nodes exist for (SPEC.md's own alphabet):
    evolution can build its own smoothing directly into the score if doing so improves the
    (turnover-cost-sensitive) fitness function, rather than this harness imposing an exogenous
    hysteresis band no formula asked for."""
    state = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    state = state.mask(raw > 0.0, 1.0)
    state = state.mask(raw < 0.0, 0.0)
    active_today = state.ffill().fillna(0.0)
    return active_today.shift(1).fillna(0.0)


def _compound_factor(cache: MarketCache, cutoff: int, weights: np.ndarray, cost_rate: np.ndarray) -> np.ndarray:
    """Delta-neutral carry per-day economics -- copied line-for-line from
    research.wave21_ga.fitness._compound_factor (see that module's own docstring for why this
    vectorized form is numerically equivalent to research.wave13_liquidity.engine13's per-day
    Python loop), minus the idle-capital-overlay branch (SPEC.md registers none for GP -- an
    inactive day simply earns 0, matching wave18_idle's own I0)."""
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

    return (1.0 + gap_return) * (1.0 - cost_return) * (1.0 + intraday_return)


def run_backtest(node: Node, cache: MarketCache, mode: str, stress: bool = False) -> pd.Series:
    """Runs ONE tree end to end and returns its equity curve (native USD, starting at
    ACTIVE_CAPITAL). Never called directly by gp.py/random_trees.py (they only see
    evaluate_tree()/evaluate_tree_cached(), which hardcode mode=MODE_IS) nor by anything
    computing the final candidate's OOS number except final_evaluation()."""
    if mode not in VALID_MODES:
        raise ValueError(f"run_backtest: unknown mode {mode!r}, expected one of {VALID_MODES}")

    cutoff = int(cache.is_row_mask.sum()) if mode == MODE_IS else len(cache.index)
    if cutoff < _MIN_CUTOFF_DAYS:
        raise ValueError(f"run_backtest: sliced date range too short ({cutoff} rows) for mode={mode!r}")
    index = cache.index[:cutoff]

    # OOS SEALING: terminals are sliced to the IS prefix BEFORE tree.evaluate() ever touches
    # them -- the only place any "computation" happens -- mirroring research.wave21_ga.fitness's
    # own "slice first, then compute" pattern (see that module's docstring).
    sliced_terminals = {name: frame.iloc[:cutoff] for name, frame in cache.terminals.items()}
    raw_score = evaluate(node, sliced_terminals)

    active_lagged = _score_to_active(raw_score).to_numpy(dtype=float)
    ranking = raw_score.shift(1).to_numpy(dtype=float)

    available = cache.available[:cutoff]
    liquidity_ok = cache.liquidity_ok[:cutoff]
    eligible = (active_lagged > 0.0) & available & liquidity_ok & np.isfinite(ranking)
    selected = _select_top_k(ranking, eligible, FIXED_TOP_K_PAIRS)
    weights = np.where(selected, FIXED_LEG_FRACTION, 0.0)

    cost_rate = (cache.cost_rate_stress if stress else cache.cost_rate)[:cutoff]
    factor = _compound_factor(cache, cutoff, weights, cost_rate)

    equity_values = ACTIVE_CAPITAL * np.cumprod(factor)
    final_weights = weights[-1]
    if np.sum(np.abs(final_weights)) > 0.0:
        final_cost = float(np.sum(np.abs(final_weights) * cost_rate[-1]))
        equity_values[-1] *= 1.0 - final_cost
    equity = pd.Series(equity_values, index=index, dtype=float)

    if mode == MODE_IS and bool((equity.index > OOS_SPLIT).any()):
        # Unreachable given the prefix slice above -- kept as a loud, immediate failure if a
        # future edit ever regresses this, matching research.wave21_ga.fitness's own precedent.
        raise OOSLeakageError("run_backtest: IS-mode equity index extends past OOS_SPLIT")
    return equity


def oos_slice(equity: pd.Series, mode: str) -> pd.Series:
    """The ONLY function in this package allowed to read OOS-range (> OOS_SPLIT) data."""
    if mode != MODE_OOS_FINAL:
        raise OOSLeakageError(f"oos_slice: called with mode={mode!r} -- OOS data is sealed until mode={MODE_OOS_FINAL!r}")
    return equity[equity.index > OOS_SPLIT]


# ---------------------------------------------------------------------------
# Walk-forward IS fitness (SPEC.md formula, byte-for-byte):
#   median(fold CAGR) - std(fold CAGR) - 5*max(0, MDD-0.15) - 0.02*node_count
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitnessResult:
    fitness: float
    fold_cagrs: tuple[float, ...]
    median_fold_cagr: float
    std_fold_cagr: float
    mdd: float
    mdd_penalty: float
    node_count: int
    node_count_penalty: float
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


def walk_forward_fitness(is_equity: pd.Series, node_count_value: int, n_folds: int = N_FOLDS) -> FitnessResult:
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
    node_penalty = NODE_COUNT_PENALTY_WEIGHT * node_count_value
    fitness = median_fold - std_fold - mdd_penalty - node_penalty
    return FitnessResult(
        fitness=fitness,
        fold_cagrs=tuple(fold_cagrs),
        median_fold_cagr=median_fold,
        std_fold_cagr=std_fold,
        mdd=mdd,
        mdd_penalty=mdd_penalty,
        node_count=node_count_value,
        node_count_penalty=node_penalty,
        full_is_cagr=cagr(is_equity),
    )


def evaluate_tree(node: Node, cache: MarketCache) -> FitnessResult:
    """The ONLY fitness entry point gp.py/random_trees.py ever call. Mode is hardcoded to
    MODE_IS -- there is no parameter here through which a caller could request OOS data."""
    equity = run_backtest(node, cache, MODE_IS)
    return walk_forward_fitness(equity, node_count(node))


def evaluate_tree_cached(node: Node, cache: MarketCache, fitness_cache: dict[Node, FitnessResult]) -> tuple[FitnessResult, bool]:
    """evaluate_tree with a caller-owned cache dict (task instruction: '평가 캐싱 필수'). `Node`
    is itself hashable/structurally-comparable (see tree.py's own module docstring), so it is
    used directly as the dict key -- no separate key-encoding step. Returns (result,
    was_cache_hit) so callers can report how many of their nominal evaluations were actual fresh
    backtests vs. reused lookups."""
    if node in fitness_cache:
        return fitness_cache[node], True
    result = evaluate_tree(node, cache)
    fitness_cache[node] = result
    return result, False


# ---------------------------------------------------------------------------
# One-time final evaluation (OOS seal opened here, and ONLY here).
# ---------------------------------------------------------------------------


class _EquityOnly:
    """Minimal duck-typed stand-in so research.wave10_carry100.regime.regime_breakdown (which
    only ever reads `.equity`) can run on a bare pd.Series -- same pattern research.wave21_ga.
    fitness's own _EquityOnly uses."""

    def __init__(self, equity: pd.Series) -> None:
        self.equity = equity


@dataclass(frozen=True, slots=True)
class FinalEvaluation:
    node: Node
    full_equity: pd.Series
    is_equity: pd.Series
    oos_equity: pd.Series
    stress_equity: pd.Series
    full_period_cagr: float
    is_cagr: float
    oos_cagr_self_contained: float  # cagr(oos_slice) -- OOS window's OWN start/end, ignoring IS continuity
    oos_cagr_regime_anchored: float | None  # regime_breakdown's OOS_SPLIT-anchored figure -- apples-to-apples vs I5.json's own saved current_low_funding.annualized_return
    mdd_full: float
    regime_breakdown: dict


def final_evaluation(node: Node, cache: MarketCache) -> FinalEvaluation:
    """Called AT MOST ONCE per wave run, on the single selected final candidate. This is the
    only function in this package that runs the backtest over the OOS range at all."""
    full_equity = run_backtest(node, cache, MODE_OOS_FINAL, stress=False)
    stress_equity = run_backtest(node, cache, MODE_OOS_FINAL, stress=True)
    is_equity = full_equity[full_equity.index <= OOS_SPLIT]  # reading the <= side is never gated; only the > OOS_SPLIT side goes through oos_slice
    oos_equity = oos_slice(full_equity, MODE_OOS_FINAL)
    regime = regime_breakdown(_EquityOnly(full_equity))
    current_low_funding = regime.get("current_low_funding")
    oos_anchored = current_low_funding.get("annualized_return") if isinstance(current_low_funding, dict) else None
    return FinalEvaluation(
        node=node,
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
    "ATR_WINDOW",
    "FIXED_LEG_FRACTION",
    "FIXED_TOP_K_PAIRS",
    "FIXED_UNIVERSE_BREADTH",
    "FUNDING_WINDOWS",
    "MDD_FOLD_FLOOR",
    "MDD_PENALTY_WEIGHT",
    "MODE_IS",
    "MODE_OOS_FINAL",
    "NODE_COUNT_PENALTY_WEIGHT",
    "N_FOLDS",
    "PRICE_RET_WINDOWS",
    "QUOTE_VOLUME_WINDOW",
    "REALIZED_VOL_WINDOW",
    "VALID_MODES",
    "FinalEvaluation",
    "FitnessResult",
    "MarketCache",
    "OOSLeakageError",
    "build_market_cache",
    "cagr",
    "deflated_sharpe_for_trials",
    "evaluate_tree",
    "evaluate_tree_cached",
    "final_evaluation",
    "market_cache_from_markets",
    "oos_slice",
    "run_backtest",
    "walk_forward_fitness",
]
