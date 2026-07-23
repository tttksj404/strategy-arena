# Wave-16 dual-layer-carry engine: the SAME per-timestamp bookkeeping as
# research.wave13_liquidity.engine13._run_liquidity_loop (gap PnL, cost accounting, trade-close
# bookkeeping, final forced unwind -- copied, not reimplemented, continuing this repo's own
# wave10->...->wave13 precedent of copying the loop body verbatim across waves), with exactly
# ONE substantive change: an extra `lending_daily_rate` term (constant per symbol, 0 for symbols
# with no current lending data) added into the SAME `intraday` return alongside `funding_frame` --
# structurally identical to how funding already works, because that is exactly what SPEC.md asks
# for ("현재 대여이자를 상수로 두고 과거 펀딩 시계열에 얹은 하한 추정치").
#
# What varies across E0-E4 (SPEC.md's frozen 5 candidates, configs16.py) is NOT engine structure --
# it is just two scalars per candidate: `ranking_lending_discount` (how much of the current
# lending snapshot enters the ranking/hysteresis score fed to carry_position + nlargest) and
# `pnl_lending_discount` (how much of it is actually added to realized daily PnL). Both default to
# 0.0, which makes this engine bit-for-bit identical to engine13's -- proven, not just claimed, by
# tests/test_wave16.py's own engine-equivalence test (same convention as test_wave13.py's own
# "reproduces wave10 bit-for-bit when cost/liquidity are held constant" test).
#
# SPEC.md 방법 4's gating instruction ("MC/블록셔플은 펀딩 부분에만 적용 가능... 대여이자
# 부분은 게이트 미적용") needs, for every candidate, a "same ranking/trade-selection, lending
# stripped from realized PnL" companion series. Representing every candidate as a
# (ranking_lending_discount, pnl_lending_discount) pair makes that companion just
# `(ranking_lending_discount, 0.0)` -- see configs16.funding_only_variant_key. `DualLayerRunner`
# below memoizes purely on this pair (+ stress_multiplier), so E1's companion literally IS E0's
# own cached run and E2's companion literally IS E4's own cached run -- not a coincidence asserted
# in a comment, but the SAME dict entry, because (0.0, 0.0) and (1.0, 0.0) are literally the same
# tuples in both places. See configs16.py's own module docstring for the full E0-E4 discount table.
#
# Universe/cost model: "top200 유니버스(L4 승계)" + "wave-13 실측비용" (SPEC.md "공통") are
# inherited by literally reusing research.wave13_liquidity's own L4 Wave13Config, universe cache
# loader, and costs_measured mapping -- never re-derived here, matching wave15's own precedent of
# calling engine13.run_candidate(get_config('L4')) in-memory for its own L4 reference rather than
# re-deriving the universe.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.fam_funding import FundingCandidate, FundingMarket, carry_position, funding_score
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, OOS_SPLIT, RESERVE_FRACTION, TOTAL_CAPITAL, Wave10Result
from research.wave13_liquidity import costs_measured
from research.wave13_liquidity import engine13
from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.costs_measured import MeasuredCostMapping
from research.wave13_liquidity.engine13 import build_cost_and_liquidity_frames
from research.wave16_duallayer.configs16 import CANDIDATE_IDS, CANDIDATES, L4_CONFIG, DualLayerCandidate, funding_only_variant_key, get_candidate

DEFAULT_STRESS_MULTIPLIER: Final = engine13.DEFAULT_STRESS_MULTIPLIER
STRESS_MULTIPLIER: Final = engine13.STRESS_MULTIPLIER


# ---------------------------------------------------------------------------
# Frame assembly -- adapted from engine13._build_aligned_frames. The ONLY diff: `ranking_apr`
# (funding_apr + lending_c) replaces bare `funding_apr` in the two places that feed the
# ranking/hysteresis machinery (`scores[symbol]`, `active[symbol]`); `funding_frame` (the REAL
# per-day funding payment used in PnL) is untouched. When ranking_lending_discount=0.0,
# ranking_apr == funding_apr exactly, so this reduces to engine13's own function bit-for-bit.
# ---------------------------------------------------------------------------


def _build_aligned_frames_dual(
    markets: dict[str, FundingMarket],
    candidate: FundingCandidate,
    lending_apr_by_symbol: dict[str, float],
    ranking_lending_discount: float,
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
        lending_c = lending_apr_by_symbol.get(symbol, 0.0) * ranking_lending_discount  # scalar per symbol -- SAME current snapshot value on every historical day (SPEC.md 치명적 한계 1: no historical lending series exists)
        ranking_apr = funding_apr + lending_c
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        spot_open[symbol] = spot_daily["open"]
        spot_close[symbol] = spot_daily["close"]
        perp_open[symbol] = perp_daily["open"]
        perp_close[symbol] = perp_daily["close"]
        funding_returns[symbol] = funding_daily
        scores[symbol] = ranking_apr
        active[symbol] = carry_position(ranking_apr, candidate)

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    spot_close_frame = pd.DataFrame(spot_close).reindex(spot_open_frame.index)
    perp_open_frame = pd.DataFrame(perp_open).reindex(spot_open_frame.index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(spot_open_frame.index)
    funding_frame = pd.DataFrame(funding_returns).reindex(spot_open_frame.index).fillna(0.0)
    score_frame = pd.DataFrame(scores).reindex(spot_open_frame.index).shift(1)
    active_frame = pd.DataFrame(active).reindex(spot_open_frame.index).fillna(0.0)
    return spot_open_frame, spot_close_frame, perp_open_frame, perp_close_frame, funding_frame, score_frame, active_frame


def _lending_daily_rate_series(columns: pd.Index, lending_apr_by_symbol: dict[str, float], pnl_lending_discount: float) -> pd.Series:
    """Constant (not day-varying) per-symbol daily-equivalent lending return added into
    `intraday` -- annualized_rate * discount / 365, exactly the same "APR -> daily rate" scaling
    convention already implicit in funding_score's own *3*365 annualization (just inverted)."""
    values = {symbol: lending_apr_by_symbol.get(symbol, 0.0) * pnl_lending_discount / 365.0 for symbol in columns}
    return pd.Series(values, index=columns, dtype=float)


# ---------------------------------------------------------------------------
# The loop itself -- byte-for-byte copy of engine13._run_liquidity_loop except the `intraday`
# line, which additionally adds `lending_daily_rate` (see module docstring). `lending_daily_rate`
# is indexed identically to spot_open_frame.columns by construction, so no reindex is needed at
# read time; when it is all-zero (pnl_lending_discount=0.0) this is arithmetically a no-op.
# ---------------------------------------------------------------------------


def _run_dual_layer_loop(
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
    lending_daily_rate: pd.Series,
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
        intraday = (intraday + funding_frame.loc[timestamp] + lending_daily_rate).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
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
# Runner: memoizes frame-builds (keyed by ranking_lending_discount only -- index/columns are
# identical regardless of discount, so cost/liquidity frames need building only once per
# stress_multiplier) and full variant runs (keyed by (ranking_discount, pnl_discount,
# stress_multiplier) -- see module docstring for why this makes companions free/aliased, not a
# separate code path).
# ---------------------------------------------------------------------------


class DualLayerRunner:
    def __init__(
        self,
        config: Any,
        mapping: MeasuredCostMapping,
        markets: dict[str, FundingMarket],
        lending_apr_by_symbol: dict[str, float],
    ) -> None:
        self._config = config
        self._mapping = mapping
        self._markets = markets
        self._lending_apr_by_symbol = lending_apr_by_symbol
        self._frames_cache: dict[float, tuple[pd.DataFrame, ...]] = {}
        self._cost_liquidity_cache: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._variant_cache: dict[tuple[float, float, float], tuple[Wave10Result, float, pd.Series]] = {}

    def _frames(self, ranking_lending_discount: float) -> tuple[pd.DataFrame, ...]:
        if ranking_lending_discount not in self._frames_cache:
            self._frames_cache[ranking_lending_discount] = _build_aligned_frames_dual(
                self._markets, self._config.candidate, self._lending_apr_by_symbol, ranking_lending_discount
            )
        return self._frames_cache[ranking_lending_discount]

    def _cost_and_liquidity(self, stress_multiplier: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        if stress_multiplier not in self._cost_liquidity_cache:
            any_frames = self._frames(0.0)
            spot_open_frame = any_frames[0]
            self._cost_liquidity_cache[stress_multiplier] = build_cost_and_liquidity_frames(
                self._config, tuple(spot_open_frame.columns), spot_open_frame.index, self._mapping, stress_multiplier
            )
        return self._cost_liquidity_cache[stress_multiplier]

    def symbols(self) -> tuple[str, ...]:
        return tuple(self._frames(0.0)[0].columns)

    def run_variant(self, ranking_lending_discount: float, pnl_lending_discount: float, stress_multiplier: float) -> tuple[Wave10Result, float, pd.Series]:
        key = (ranking_lending_discount, pnl_lending_discount, stress_multiplier)
        if key not in self._variant_cache:
            frames = self._frames(ranking_lending_discount)
            cost_rate_frame, liquidity_ok_frame = self._cost_and_liquidity(stress_multiplier)
            lending_daily_rate = _lending_daily_rate_series(frames[0].columns, self._lending_apr_by_symbol, pnl_lending_discount)
            self._variant_cache[key] = _run_dual_layer_loop(
                *frames, self._config.candidate.top_k, self._config.leg_fraction, cost_rate_frame, liquidity_ok_frame, lending_daily_rate
            )
        return self._variant_cache[key]


def build_runner(lending_snapshot: dict[str, Any]) -> DualLayerRunner:
    """Loads L4's own frozen 200-symbol universe/market cache (read-only, wave13_liquidity's own
    fail-closed loader -- no network) and wave13's fitted measured-cost mapping, joins in this
    session's fetched lending_snapshot['by_symbol']['lending_apr'] per symbol (0.0 for symbols
    with no OKX lending listing -- fail-closed, never invents a rate)."""
    symbols = ul.verify_cache_and_load_symbols(L4_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    mapping = costs_measured.fit_mapping()
    lending_apr_by_symbol = {
        symbol: float(info["lending_apr"]) for symbol, info in lending_snapshot["by_symbol"].items() if symbol in symbols
    }
    return DualLayerRunner(L4_CONFIG, mapping, markets, lending_apr_by_symbol)


# ---------------------------------------------------------------------------
# Top-level per-candidate result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VariantRun:
    result: Wave10Result
    total_cost_usdt: float
    eligible: pd.Series
    stress_result: Wave10Result
    stress_total_cost_usdt: float
    stress_eligible: pd.Series


@dataclass(frozen=True, slots=True)
class DualLayerResult:
    candidate_id: str
    ranking_lending_discount: float
    pnl_lending_discount: float
    combined: VariantRun  # this candidate's OWN headline definition (SPEC.md table)
    funding_only: VariantRun  # SAME ranking/trade-selection, lending stripped from PnL -- the ONLY series gates16.py's S2/S3/S5 may touch
    symbols_used: tuple[str, ...]


def _variant_run(runner: DualLayerRunner, ranking_discount: float, pnl_discount: float) -> VariantRun:
    result, total_cost, eligible = runner.run_variant(ranking_discount, pnl_discount, DEFAULT_STRESS_MULTIPLIER)
    stress_result, stress_total_cost, stress_eligible = runner.run_variant(ranking_discount, pnl_discount, STRESS_MULTIPLIER)
    return VariantRun(result, total_cost, eligible, stress_result, stress_total_cost, stress_eligible)


def run_candidate(candidate_id: str, runner: DualLayerRunner) -> DualLayerResult:
    candidate = get_candidate(candidate_id)
    combined = _variant_run(runner, candidate.ranking_lending_discount, candidate.pnl_lending_discount)
    fo_ranking, fo_pnl = funding_only_variant_key(candidate)
    funding_only = _variant_run(runner, fo_ranking, fo_pnl)
    return DualLayerResult(
        candidate_id=candidate_id,
        ranking_lending_discount=candidate.ranking_lending_discount,
        pnl_lending_discount=candidate.pnl_lending_discount,
        combined=combined,
        funding_only=funding_only,
        symbols_used=runner.symbols(),
    )


# ---------------------------------------------------------------------------
# SPEC.md 방법 2: current cross-sectional snapshot pick -- NOT a backtest (no engine loop, no
# hysteresis: a single day has no "already held" state). Uses the INSTANTANEOUS current Bitget
# funding rate annualized (fetch_lending.py's bitget_funding_apr_current), a documented
# approximation to the strategy's actual 7-day trailing realized-average signal (that would need
# a live rolling-window fetch this one-off snapshot does not build) -- kept entirely OUT of the
# historical backtest above, which always uses the true rolling signal from cache.
# ---------------------------------------------------------------------------


def current_snapshot_pick(ranking_lending_discount: float, entry_threshold_apr: float, lending_snapshot: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for symbol, info in lending_snapshot["by_symbol"].items():
        funding_apr = info.get("bitget_funding_apr_current")
        if funding_apr is None:
            continue
        lending_apr = float(info["lending_apr"]) if info["lending_available"] else 0.0
        ranking_score = float(funding_apr) + lending_apr * ranking_lending_discount
        rows.append(
            {
                "symbol": symbol,
                "funding_apr_current": float(funding_apr),
                "lending_apr": lending_apr,
                "lending_available": bool(info["lending_available"]),
                "ranking_score": ranking_score,
            }
        )
    rows.sort(key=lambda row: row["ranking_score"], reverse=True)
    clearing = [row for row in rows if row["ranking_score"] > entry_threshold_apr]
    return {
        "universe_n": len(rows),
        "entry_threshold_apr": entry_threshold_apr,
        "n_clearing_threshold": len(clearing),
        "top_pick": clearing[0] if clearing else None,
        "top_5_by_score": rows[:5],
    }


__all__ = [
    "ACTIVE_CAPITAL",
    "DEFAULT_STRESS_MULTIPLIER",
    "MIN_ORDER_USDT",
    "OOS_SPLIT",
    "RESERVE_FRACTION",
    "STRESS_MULTIPLIER",
    "TOTAL_CAPITAL",
    "DualLayerResult",
    "DualLayerRunner",
    "VariantRun",
    "build_runner",
    "current_snapshot_pick",
    "run_candidate",
]
