# Wave-18 idle-capital-overlay engine.
#
# Structure (SPEC.md "핵심 구조"): L4's own signal is left untouched and always gets FIRST
# priority every single day; idle-capital deployment (I1-I5) is a layer that only ever gets a
# chance on a day L4 itself ranks nothing. This is enforced STRUCTURALLY, not by a post-hoc
# check: the day-loop below (_run_idle_overlay_loop) evaluates the L4 layer first and only
# falls through to the overlay list / lending fallback when L4's own eligible-ranked set is
# empty -- which is exactly SPEC.md's S6 gate ("캐리 신호 발생일에 유휴자본이 즉시 회수돼 캐리
# 진입을 막지 않음"), made true by construction rather than asserted after the fact.
# gates18.gate_s6_recoverability still re-checks this empirically against the saved
# per-day layer_used series, the same "structural claim + empirical gate" pattern this repo
# uses everywhere else (e.g. S1's delta-neutrality).
#
# Why ONE parameterized loop for all six candidates, not six copies of engine13's loop (the
# wave10->...->wave17 lineage's own usual precedent): those waves differ from EACH OTHER in
# genuinely separate ways across DIFFERENT sessions. I0-I5 here are six tightly-coupled
# variants of the exact same idea, frozen together in one SPEC.md, sharing the identical
# primary layer, cost model, and $90 capital contract -- collapsing "which overlay(s) are
# tried, in what order, with what direction" into a small per-candidate config
# (configs18.IdleConfig) is the smaller, more auditable surface, and it is the ONLY way S6 can
# be a single shared engine-level guarantee instead of six separately-reviewed loop bodies.
#
# What is new here relative to engine13._run_liquidity_loop:
#   1. Frame-building is split into a candidate-INDEPENDENT half (_build_aligned_frames18:
#      spot/perp OHLC, funding, and the raw 7d funding-APR score -- window_days=7 is shared by
#      every wave18 layer, SPEC.md "L4 엔진의 threshold_apr만 변경") and a candidate-DEPENDENT
#      half (active_frame_for / reverse_active_frame_for, applying carry_position's or the new
#      reverse_carry_position's hysteresis to that SAME score at a given threshold). This
#      avoids recomputing funding_score three times (L4 at 15%, the carry overlay at 8%, the
#      reverse overlay at -15%) over an identical rolling-mean pass.
#   2. Weights are SIGNED (positive = long-spot/short-perp, the normal carry direction;
#      negative = long-perp/short-spot, I4's reverse direction) instead of engine13's
#      always-nonnegative weights. This is a deliberate, minimal generalization: engine13's
#      existing arithmetic (gap-PnL, intraday PnL) already produces the CORRECT sign for a
#      reverse position once weights carry a sign, because `spot_ret - perp_ret + funding`
#      multiplied by a negative weight is algebraically identical to
#      `-(spot_ret - perp_ret + funding)` -- exactly the mirrored construction I4 needs. Only
#      the trade-bookkeeping branch conditions (`weight > 0.0`) needed generalizing to
#      `abs(weight) > 0.0` so a reverse trade's own open/close is still recorded (see inline
#      comments at the trade_growth block). Turnover/cost math (`abs(weights - previous)`) was
#      ALREADY sign-correct with no changes: a same-symbol flip from +leg_fraction to
#      -leg_fraction costs 2x leg_fraction's worth of turnover, which is economically right
#      (close one direction, open the other) -- and tests/test_wave18.py pins this.
#      Structurally this never actually has to handle a same-symbol sign flip in practice: I4
#      is the only signed-negative layer any wave18 candidate uses, and it only ever competes
#      against "0" (L4 idle, nothing else eligible), never against a same-day positive-weight
#      layer for the SAME symbol -- see run_idle_candidate's per-candidate layer lists.
#   3. A lending fallback: when NEITHER L4 nor any overlay layer ranks anything on a given day,
#      weights stay all-zero (as in engine13) but capital additionally compounds by a FLAT
#      per-day rate (I1/I5 only) -- modeling flexible USDT lending, which this wave's
#      background (wave-17) established has no bid-ask/turnover cost of its own to model.
#
# Reused, not reimplemented: research.wave13_liquidity.engine13.build_cost_and_liquidity_frames
# (identical measured-cost + liquidity-mask construction, called once against L4's own top200
# universe and shared by every layer -- I2's BTC/ETH overlay and I3/I4's full-universe overlays
# are all COLUMN SUBSETS of that same frame, never a separately-fetched dataset).
# research.wave1.fam_funding.funding_score / carry_position are imported unmodified, matching
# every prior wave's own precedent.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

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
from research.wave13_liquidity.engine13 import DEFAULT_STRESS_MULTIPLIER, STRESS_MULTIPLIER, build_cost_and_liquidity_frames
from research.wave18_idle.configs18 import (
    L4_CONFIG,
    LEG_FRACTION,
    OVERLAY_CARRY_CANDIDATE,
    OVERLAY_REVERSE_CANDIDATE,
    TOP_K,
    IdleConfig,
)

LAYER_L4: Final = "L4"
LAYER_CARRY_OVERLAY: Final = "carry_overlay"
LAYER_REVERSE_OVERLAY: Final = "reverse_overlay"
LAYER_LENDING: Final = "lending"
LAYER_CASH: Final = "cash"


# ---------------------------------------------------------------------------
# Signal construction.
# ---------------------------------------------------------------------------


def reverse_carry_position(score: pd.Series, candidate: FundingCandidate) -> pd.Series:
    """Mirrors research.wave1.fam_funding.carry_position's own entry/exit-half hysteresis
    state machine with the SIGN FLIPPED: SPEC.md's I4 rule is 'funding < -threshold_apr'
    (shorts paying longs) triggers entry -- the mirror image of carry_position's own
    'score > +threshold_apr' entry rule around zero. Exit mirrors carry_position's own
    threshold/2 convention on the negative side (score > -threshold_apr/2 -- the reverse
    signal has decayed back to within half its entry magnitude of zero), not a new,
    separately-tuned exit rule. `candidate.threshold_apr` is read as a plain positive
    MAGNITUDE (see configs18.OVERLAY_REVERSE_CANDIDATE's own docstring)."""
    values: list[float] = []
    active = 0.0
    for value in score:
        if pd.notna(value) and value < -candidate.threshold_apr:
            active = 1.0
        elif pd.notna(value) and value > -candidate.threshold_apr / 2.0:
            active = 0.0
        values.append(active)
    return pd.Series(values, index=score.index, dtype=float).shift(1).fillna(0.0)


def _build_aligned_frames18(
    markets: dict[str, FundingMarket], window_days: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Candidate-INDEPENDENT half of engine13._build_aligned_frames: spot/perp OHLC + funding
    + the RAW (NOT yet shift(1)'d, NOT yet thresholded) 7d funding-APR score per symbol.
    window_days=7 is shared by every wave18 layer (L4 primary + every I1-I5 overlay) -- see
    module docstring point 1. Callers derive the ranking score (`.shift(1)` of the last
    returned frame) and as many differently-thresholded active_frames as they need from this
    SAME raw score via active_frame_for / reverse_active_frame_for, rather than this function
    baking in one candidate's own hysteresis the way engine13's version does."""
    spot_open: dict[str, pd.Series] = {}
    spot_close: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    funding_returns: dict[str, pd.Series] = {}
    raw_scores: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        funding_daily = market.funding.resample("1D").sum()
        funding_apr = funding_score(market.funding, window_days).resample("1D").last()
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        spot_open[symbol] = spot_daily["open"]
        spot_close[symbol] = spot_daily["close"]
        perp_open[symbol] = perp_daily["open"]
        perp_close[symbol] = perp_daily["close"]
        funding_returns[symbol] = funding_daily
        raw_scores[symbol] = funding_apr

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    spot_close_frame = pd.DataFrame(spot_close).reindex(spot_open_frame.index)
    perp_open_frame = pd.DataFrame(perp_open).reindex(spot_open_frame.index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(spot_open_frame.index)
    funding_frame = pd.DataFrame(funding_returns).reindex(spot_open_frame.index).fillna(0.0)
    raw_score_frame = pd.DataFrame(raw_scores).reindex(spot_open_frame.index)
    return spot_open_frame, spot_close_frame, perp_open_frame, perp_close_frame, funding_frame, raw_score_frame


def active_frame_for(raw_score_frame: pd.DataFrame, candidate: FundingCandidate) -> pd.DataFrame:
    """carry_position applied per-column over the SAME raw (unshifted) score -- carry_position
    does its own internal shift(1), matching engine13's own convention exactly (see that
    module's _build_aligned_frames: `active[symbol] = carry_position(funding_apr, candidate)`
    fed the RAW score, never the pre-shifted one)."""
    return pd.DataFrame(
        {symbol: carry_position(raw_score_frame[symbol], candidate) for symbol in raw_score_frame.columns},
        index=raw_score_frame.index,
    ).fillna(0.0)


def reverse_active_frame_for(raw_score_frame: pd.DataFrame, candidate: FundingCandidate) -> pd.DataFrame:
    return pd.DataFrame(
        {symbol: reverse_carry_position(raw_score_frame[symbol], candidate) for symbol in raw_score_frame.columns},
        index=raw_score_frame.index,
    ).fillna(0.0)


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OverlayLayer:
    label: str  # LAYER_CARRY_OVERLAY | LAYER_REVERSE_OVERLAY
    active_frame: pd.DataFrame  # already-thresholded carry_position/reverse_carry_position output, full (all-200-symbol) columns
    symbols: tuple[str, ...]  # eligibility is restricted to exactly these columns (MAJORS_ONLY_SYMBOLS for I2/I5; the full universe for I3/I4)
    direction: float  # +1.0 normal (long spot / short perp) or -1.0 reverse (long perp / short spot)


@dataclass(frozen=True, slots=True)
class Wave18Result:
    equity: pd.Series  # native USD, starts at ACTIVE_CAPITAL ($90) -- same convention as Wave10Result
    positions: pd.Series  # sum(|weights|) per day; 0.0 == no carry/reverse position held that day (may still be a lending day)
    turnover: pd.Series
    trade_returns: pd.Series
    layer_used: pd.Series  # object dtype, one of LAYER_L4/LAYER_CARRY_OVERLAY/LAYER_REVERSE_OVERLAY/LAYER_LENDING/LAYER_CASH per day
    max_concurrent_positions: int
    symbols_used: tuple[str, ...]


def _run_idle_overlay_loop(
    spot_open_frame: pd.DataFrame,
    spot_close_frame: pd.DataFrame,
    perp_open_frame: pd.DataFrame,
    perp_close_frame: pd.DataFrame,
    funding_frame: pd.DataFrame,
    ranking_score_frame: pd.DataFrame,  # raw_score_frame.shift(1) -- shared ranking metric for every layer
    l4_active_frame: pd.DataFrame,
    overlay_layers: tuple[OverlayLayer, ...],  # tried IN ORDER, first match wins, ONLY on a day L4 itself ranks nothing
    top_k: int,
    leg_fraction: float,
    cost_rate_frame: pd.DataFrame,
    liquidity_ok_frame: pd.DataFrame,
    lending_daily_rate: float | None,  # flat per-day rate applied when L4 AND every overlay layer miss; None => plain cash (0 return), matching I0/I2/I3/I4's own "no fallback" contract
) -> tuple[Wave18Result, float, pd.Series]:
    capital = ACTIVE_CAPITAL
    equity_values: list[float] = []
    turnover_values: list[float] = []
    exposures: list[float] = []
    concurrent_counts: list[int] = []
    l4_eligible_counts: list[int] = []
    layer_used_values: list[str] = []
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
        liquidity_row = liquidity_ok_frame.loc[timestamp]
        liquid_index = liquidity_row[liquidity_row].index
        available_index = available[available].index

        # 1) L4 primary layer -- ALWAYS checked first, exactly reproducing engine13's own
        #    `eligible`/`ranked` computation (S6's structural guarantee: nothing below this
        #    branch is even evaluated on a day this one succeeds).
        eligible_l4 = l4_active_frame.loc[timestamp][l4_active_frame.loc[timestamp] > 0.0].index
        eligible_l4 = eligible_l4.intersection(available_index).intersection(liquid_index)
        l4_eligible_counts.append(int(len(eligible_l4)))
        ranked_l4 = ranking_score_frame.loc[timestamp, eligible_l4].dropna().nlargest(top_k).index

        weights = pd.Series(0.0, index=spot_open_frame.columns)
        layer_used = LAYER_CASH
        if len(ranked_l4) > 0:
            weights.loc[ranked_l4] = leg_fraction
            layer_used = LAYER_L4
        else:
            for layer in overlay_layers:
                layer_row = layer.active_frame.loc[timestamp]
                eligible = layer_row[layer_row > 0.0].index
                eligible = eligible.intersection(pd.Index(layer.symbols)).intersection(available_index).intersection(liquid_index)
                scored = ranking_score_frame.loc[timestamp, eligible].dropna()
                ranked = (scored.nsmallest(top_k) if layer.direction < 0.0 else scored.nlargest(top_k)).index
                if len(ranked) > 0:
                    weights.loc[ranked] = leg_fraction * layer.direction
                    layer_used = layer.label
                    break
            else:
                layer_used = LAYER_LENDING if lending_daily_rate is not None else LAYER_CASH
        layer_used_values.append(layer_used)

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
        if layer_used == LAYER_LENDING:
            # weights are all zero this branch (nothing below L4/overlays matched), so the
            # `intraday * weights` term above contributed exactly 0.0 -- this multiplies the
            # SAME day's capital by the flat lending accrual instead, never both at once.
            capital *= 1.0 + float(lending_daily_rate)  # type: ignore[arg-type]

        for symbol in spot_open_frame.columns:
            previous_weight = float(previous_weights[symbol])
            current_weight = float(weights[symbol])
            leg_rate = cost_for(symbol, timestamp)
            # Generalized from engine13's `weight > 0.0` to `abs(weight) > 0.0` so a REVERSE
            # (negative-weight) trade's own open/close is still recorded in trade_growth --
            # see module docstring point 2. `sign` picks up the direction of whichever side of
            # the transition is currently held, and is applied only to gap/intraday (which are
            # computed in the "normal" long-spot/short-perp convention from raw price/funding
            # data); leg_rate/cost is never sign-flipped -- a cost is a cost regardless of
            # direction.
            if abs(previous_weight) > 0.0 and symbol in trade_growth:
                sign = 1.0 if previous_weight > 0.0 else -1.0
                trade_growth[symbol] *= 1.0 + sign * float(gap_by_symbol[symbol])
            if abs(previous_weight) > 0.0 and abs(current_weight) == 0.0:
                trade_growth[symbol] *= 1.0 - leg_rate
                trade_values.append((trade_growth.pop(symbol) - 1.0) * trade_weights.pop(symbol))
                trade_times.append(pd.Timestamp(timestamp))
            elif abs(previous_weight) == 0.0 and abs(current_weight) > 0.0:
                trade_growth[symbol] = 1.0 - leg_rate
                trade_weights[symbol] = abs(current_weight)
            elif abs(previous_weight) > 0.0 and abs(current_weight) > 0.0 and previous_weight != current_weight:
                # Same-sign resize only in practice (fixed leg_fraction/top_k=1 means this
                # branch is never actually hit in wave18 -- kept for structural parity with
                # engine13's own loop shape, and to fail safe rather than silently mis-cost a
                # same-symbol DIRECTION FLIP if a future candidate ever violates that
                # assumption: abs() below makes a flip's fraction computable rather than
                # nonsensical, though it does not by itself validate the flip is priced right
                # -- tests/test_wave18.py's turnover-cost test covers the flip case directly
                # via the raw weights/cost formula instead, not through this branch.
                trade_growth[symbol] *= 1.0 - abs(current_weight - previous_weight) * leg_rate / max(abs(current_weight), abs(previous_weight))
                trade_weights[symbol] = abs(current_weight)
            if abs(current_weight) > 0.0:
                sign = 1.0 if current_weight > 0.0 else -1.0
                trade_growth[symbol] *= 1.0 + sign * float(intraday[symbol])

        equity_values.append(capital)
        turnover_values.append(turnover)
        exposures.append(float(weights.abs().sum()))
        concurrent_counts.append(int((weights != 0.0).sum()))
        previous_weights = weights

    if len(spot_open_frame.index) > 0 and float(previous_weights.abs().sum()) > 0.0:
        final_timestamp = pd.Timestamp(spot_open_frame.index[-1])
        final_cost = sum(abs(float(previous_weights[symbol])) * cost_for(symbol, final_timestamp) for symbol in spot_open_frame.columns)
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
    layer_used_series = pd.Series(layer_used_values, index=spot_open_frame.index, dtype=object)
    eligible_series = pd.Series(l4_eligible_counts, index=spot_open_frame.index, dtype=float)
    result = Wave18Result(
        equity=equity,
        positions=positions,
        turnover=turnover_series,
        trade_returns=trades,
        layer_used=layer_used_series,
        max_concurrent_positions=max(concurrent_counts, default=0),
        symbols_used=tuple(spot_open_frame.columns),
    )
    return result, total_cost_usdt, eligible_series


# ---------------------------------------------------------------------------
# Top-level entry points used by run_wave18.py.
# ---------------------------------------------------------------------------


def daily_rate_from_apr(apr: float) -> float:
    """Compound-daily conversion, matching research.wave10_carry100.regime's own
    `growth ** (365.0/days) - 1.0` annualization convention (365.0, not 365.25) so a lending
    candidate's own reported CAGR contribution is computed on the identical day-count basis as
    every other regime/CAGR number in this report."""
    return (1.0 + apr) ** (1.0 / 365.0) - 1.0


def run_i0_reference(mapping: MeasuredCostMapping, stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER) -> tuple[Wave10Result, float, pd.Series]:
    """I0 = L4, with NOTHING layered on top. Calls research.wave13_liquidity.engine13.run_candidate
    directly (not the wave18 loop) so I0's reproduction of results/L4.json is byte-for-byte by
    CONSTRUCTION (same function, same config, same inputs) rather than relying on the wave18
    loop reducing to the same numbers -- that reduction is separately proven in
    tests/test_wave18.py (engine18 with zero overlays and no lending must ALSO match this),
    but is not what run_wave18.py's own I0 stage depends on."""
    return engine13.run_candidate(L4_CONFIG, mapping, stress_multiplier)


def run_idle_candidate(
    idle_config: IdleConfig,
    mapping: MeasuredCostMapping,
    lending_apr: float | None,
    stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER,
) -> tuple[Wave18Result, float, pd.Series]:
    """Dispatches one I1-I5 config end to end: fail-closed cache load (borrows L4's own top200
    universe/cache -- research.wave13_liquidity.universe_liquidity, no separate wave18 fetch of
    price/funding data), build the shared frames once, build this candidate's own overlay
    layer(s) from configs18.IdleConfig, run the loop. I0 does NOT go through this function --
    see run_i0_reference above."""
    symbols = ul.verify_cache_and_load_symbols(L4_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    spot_open_frame, spot_close_frame, perp_open_frame, perp_close_frame, funding_frame, raw_score_frame = _build_aligned_frames18(markets)
    ranking_score_frame = raw_score_frame.shift(1)
    l4_active_frame = active_frame_for(raw_score_frame, L4_CONFIG.candidate)
    cost_rate_frame, liquidity_ok_frame = build_cost_and_liquidity_frames(
        L4_CONFIG, tuple(spot_open_frame.columns), spot_open_frame.index, mapping, stress_multiplier
    )

    overlay_layers: list[OverlayLayer] = []
    if idle_config.uses_carry_overlay:
        carry_active = active_frame_for(raw_score_frame, OVERLAY_CARRY_CANDIDATE)
        overlay_symbols = idle_config.overlay_symbols if idle_config.overlay_symbols is not None else symbols
        overlay_layers.append(OverlayLayer(LAYER_CARRY_OVERLAY, carry_active, overlay_symbols, 1.0))
    if idle_config.uses_reverse_overlay:
        reverse_active = reverse_active_frame_for(raw_score_frame, OVERLAY_REVERSE_CANDIDATE)
        overlay_layers.append(OverlayLayer(LAYER_REVERSE_OVERLAY, reverse_active, symbols, -1.0))

    lending_daily_rate: float | None = None
    if idle_config.uses_lending_fallback:
        if lending_apr is None:
            raise ValueError(f"{idle_config.candidate_id}: uses_lending_fallback=True requires lending_apr (run `--stage fetch` first)")
        lending_daily_rate = daily_rate_from_apr(lending_apr)

    return _run_idle_overlay_loop(
        spot_open_frame,
        spot_close_frame,
        perp_open_frame,
        perp_close_frame,
        funding_frame,
        ranking_score_frame,
        l4_active_frame,
        tuple(overlay_layers),
        TOP_K,
        LEG_FRACTION,
        cost_rate_frame,
        liquidity_ok_frame,
        lending_daily_rate,
    )


__all__ = [
    "ACTIVE_CAPITAL",
    "LAYER_CARRY_OVERLAY",
    "LAYER_CASH",
    "LAYER_L4",
    "LAYER_LENDING",
    "LAYER_REVERSE_OVERLAY",
    "MIN_ORDER_USDT",
    "OOS_SPLIT",
    "RESERVE_FRACTION",
    "STRESS_MULTIPLIER",
    "TOTAL_CAPITAL",
    "OverlayLayer",
    "Wave18Result",
    "active_frame_for",
    "daily_rate_from_apr",
    "reverse_active_frame_for",
    "reverse_carry_position",
    "run_i0_reference",
    "run_idle_candidate",
]
