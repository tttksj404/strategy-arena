# Wave-14 multi-venue backtest engines. Two structurally different position types, both
# producing a research.wave10_carry100.engine.Wave10Result (unchanged dataclass, reused so
# every downstream consumer -- gates14.py, regime_breakdown, reporting14.py -- works
# identically regardless of which engine produced a given config's result):
#
#   run_carry_candidate   (M0-M5): the ORDINARY same-venue spot+perp carry trade,
#     multi-venue only in WHERE its candidate pool and per-symbol costs come from. The
#     per-timestamp bookkeeping (gap PnL, intraday PnL, turnover cost, trade-close
#     accounting, final forced unwind) is an ADAPTED COPY of
#     research.wave13_liquidity.engine13's own `_run_liquidity_loop` (_run_carry_loop,
#     below) -- economically IDENTICAL rules, not reimplemented logic, but a copy rather
#     than a cross-module import because engine13's own version bakes in
#     `capital = ACTIVE_CAPITAL` (a module-level import of wave10's fixed $90, not a
#     parameter) at its very first line, so it cannot serve M2/M4/M5/M6/M7's non-$90
#     capital tiers without either monkeypatching shared module state (rejected -- fragile,
#     not this repo's convention) or an adapted copy that takes `active_capital` as an
#     explicit parameter (this module's approach). Capital tier is exactly the one axis
#     SPEC.md's own task contract authorizes changing ("바꾸는 건 거래소 소스·동시 쌍
#     수·자본뿐"), and copying-rather-than-parameterizing the shared loop to accommodate a
#     sanctioned axis of variation is the SAME precedent engine13.py's own module docstring
#     documents for itself: "continuing the wave10 -> wave11 -> wave12 -> wave13 precedent
#     of copying the loop body verbatim across waves". tests/test_wave14.py's engine-
#     equivalence test pins _run_carry_loop against engine13's own
#     `_run_liquidity_loop`/wave10's `run_fixed_fraction_portfolio` at active_capital=$90
#     (byte-for-byte, same technique wave13's own equivalence test uses), so the ONLY
#     thing this copy is verified to have changed is the capital parameterization itself.
#     What ELSE changes, all upstream of that loop: (a) which symbols/venues feed the
#     aligned frames (_build_aligned_frames_multi, below -- a copy of engine13's own
#     frame-builder, unavoidably duplicated because it needs one line changed: see
#     daily_funding_score's own docstring for why), and (b) the cost-rate/liquidity frames
#     are venue-routed (build_multivenue_cost_and_liquidity_frames, below).
#
#   run_cross_venue_candidate (M6/M7 only): a GENUINELY NEW position structure per SPEC.md
#     ("M6/M7은 신규 구조(양쪽 퍼프)라 기존 캐리와 별도 패밀리로 등록") -- two perpetual legs
#     on TWO DIFFERENT venues for the SAME symbol (short the higher-funding venue, long the
#     lower-funding venue), no spot leg on either side. See _run_cross_venue_loop's own
#     docstring for the full PnL/cost derivation.
#
# Both engines restrict their tradable index to OVERLAP_START..OVERLAP_END (2024-01-01
# through FROZEN_END=2026-07-14, the only span with BOTH Binance and Bybit data -- SPEC.md:
# "장기 백테스트 구간은 Binance+Bybit 겹침(2024-01~2026-07)뿹이다... 공정 비교를 위해 M0도
# 반드시 동일 구간으로 절단해 산출") -- applied by SLICING the fully-warmed-up aligned
# frames' DatetimeIndex, never by truncating the raw input history: every rolling/score
# calculation still sees its full pre-2024 lookback for warmup, and the loop's own
# `capital = active_capital` / `previous_weights = 0` initialization at whatever the FIRST
# row of the sliced frame happens to be is what gives every M-config (M0 included) a fresh,
# comparable start exactly at 2024-01-01 -- see tests/test_wave14.py's window-slicing test.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.fam_funding import FundingCandidate, FundingMarket, carry_position
from research.wave10_carry100.engine import Wave10Result
from research.wave13_liquidity import costs_measured
from research.wave13_liquidity.costs_measured import MeasuredCostMapping
from research.wave13_liquidity.universe_liquidity import FROZEN_END
from research.wave14_multivenue import costs_venue
from research.wave14_multivenue import universe_multi as um
from research.wave14_multivenue.configs14 import Wave14Config
from research.wave14_multivenue.costs_venue import BybitCostMappings
from research.wave14_multivenue.universe_multi import CrossVenuePair

DEFAULT_STRESS_MULTIPLIER: Final = 1.0
STRESS_MULTIPLIER: Final = 3.0  # SPEC.md "S1~S5 승계" -- wave13's x3 measured-slippage stress, unchanged

OVERLAP_START: Final = pd.Timestamp("2024-01-01T00:00:00Z")  # SPEC.md "Binance+Bybit(2024-01~2026-07)"
OVERLAP_END: Final = FROZEN_END + pd.Timedelta(days=1)  # exclusive upper bound; FROZEN_END=2026-07-14, matches fetch_venues.END_TS


def daily_funding_score(funding_daily: pd.Series, window_days: int) -> pd.Series:
    """Interval-agnostic replacement for research.wave1.fam_funding.funding_score, needed
    because Bybit's own funding cadence is NOT uniformly 8h the way Binance's is (this
    wave's own probe: 66/135 Bybit symbols pay every 4h, 69/135 every 8h -- see
    fetch_venues.py's discover_bybit_universe). wave1's funding_score assumes exactly 3
    raw events/day (`window_days * 3` as its rolling window width); applying that
    event-COUNT window to a 4h-cadence symbol would look back HALF the intended calendar
    span. This function instead rolls over the ALREADY-DAILY-RESAMPLED series
    (`funding_daily = market.funding.resample("1D").sum()`, which every engine in this
    chain computes anyway for the PnL term) by CALENDAR DAYS, which is venue/cadence
    agnostic by construction.

    Mathematically identical to funding_score for Binance's own uniform 8h cadence (proof,
    also pinned by tests/test_wave14.py): with 3 events/day, funding_score's rolling
    window of `window_days*3` raw events, sampled at a given day's LAST (16:00) event via
    the caller's own `.resample("1D").last()`, spans exactly the `window_days` FULL
    calendar days ending on that day (event index arithmetic: the window's earliest event
    lands exactly on day `(d - window_days + 1)`'s FIRST (00:00) event) -- so its sum
    equals `funding_daily[d-window_days+1 : d+1].sum()`, and
    `rolling_mean * 3 * 365 == rolling_sum * 365/window_days`, which is exactly this
    function's own `rolling(window_days).sum() * 365/window_days`. This function is used
    uniformly for BOTH Binance- and Bybit-sourced symbols in this module (one formula, not
    two) precisely because it reproduces wave1's own signal exactly where wave1's formula
    was valid (uniform 8h) and correctly generalizes where it wasn't (Bybit's mixed
    cadence) -- not a rule change, a cadence-safe generalization of the same rule."""
    return funding_daily.rolling(window_days, min_periods=window_days).sum() * (365.0 / window_days)


# ---------------------------------------------------------------------------
# M0-M5: ordinary same-venue carry, multi-venue candidate pool.
# ---------------------------------------------------------------------------


def _build_aligned_frames_multi(
    markets: dict[str, FundingMarket], candidate: FundingCandidate
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Copy of research.wave13_liquidity.engine13._build_aligned_frames with exactly one
    line changed (funding_score -> daily_funding_score, see that function's own docstring
    for why this copy is unavoidable) -- otherwise byte-identical, continuing this repo's
    own established copy-the-frame-builder-verbatim convention."""
    spot_open: dict[str, pd.Series] = {}
    spot_close: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    funding_returns: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        funding_daily = market.funding.resample("1D").sum()
        funding_apr = daily_funding_score(funding_daily, candidate.window_days)
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


def build_multivenue_cost_and_liquidity_frames(
    symbols: tuple[str, ...],
    venue_of: dict[str, str],
    index: pd.DatetimeIndex,
    binance_mapping: MeasuredCostMapping,
    bybit_mappings: BybitCostMappings | None,
    stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds ONE cost-rate frame and ONE liquidity-mask frame spanning the WHOLE union
    symbol pool, routing each column to its own venue's cost model: Binance-keyed columns
    get wave13's Bitget-measured mapping (unchanged); Bybit-keyed columns
    (universe_multi.BYBIT_KEY_SUFFIX-suffixed) get this wave's own Bybit spot+linear
    mapping (costs_venue.bybit_pair_cost_rate). A Bybit-sourced slot is liquidity-eligible
    only when BOTH its spot leg and its linear leg individually clear the same pure
    data-availability bar wave13 L1-L4 use (no extra dollar floor -- see
    costs_measured.build_data_availability_mask's own docstring for why: this wave doesn't
    revisit that liquidity-vs-cost-model design choice, it only extends it to a second
    venue's two markets)."""
    binance_keys = tuple(sorted(key for key in symbols if venue_of[key] == "binance"))
    bybit_keys = tuple(sorted(key for key in symbols if venue_of[key] == "bybit"))

    cost_parts: list[pd.DataFrame] = []
    liquidity_parts: list[pd.DataFrame] = []

    if binance_keys:
        binance_symbols = binance_keys  # Binance-side keys carry no suffix -- identical to their raw symbol
        quote_volume = um.load_binance_quote_volume_frame(binance_symbols)
        binance_cost = costs_measured.build_cost_rate_frame(quote_volume, binance_symbols, binance_mapping, stress_multiplier)
        binance_liquidity = costs_measured.build_data_availability_mask(quote_volume, binance_symbols)
        cost_parts.append(binance_cost.reindex(index=index, columns=list(binance_symbols)))
        liquidity_parts.append(binance_liquidity.reindex(index=index, columns=list(binance_symbols)))

    if bybit_keys:
        if bybit_mappings is None:
            raise ValueError("bybit_mappings required when the symbol pool includes Bybit-keyed columns")
        bybit_symbols = tuple(um.base_symbol(key) for key in bybit_keys)
        spot_volume = um.load_bybit_quote_volume_frame(bybit_symbols, "spot")
        linear_volume = um.load_bybit_quote_volume_frame(bybit_symbols, "linear")
        spot_bp = costs_venue.build_bp_frame_for_market(spot_volume, bybit_symbols, bybit_mappings.spot)
        linear_bp = costs_venue.build_bp_frame_for_market(linear_volume, bybit_symbols, bybit_mappings.linear)
        bybit_cost = costs_venue.bybit_pair_cost_rate(spot_bp, linear_bp, stress_multiplier)
        spot_liquidity = costs_venue.build_liquidity_mask_for_market(spot_volume, bybit_symbols)
        linear_liquidity = costs_venue.build_liquidity_mask_for_market(linear_volume, bybit_symbols)
        bybit_liquidity = spot_liquidity & linear_liquidity
        cost_parts.append(bybit_cost.rename(columns=um.bybit_key).reindex(index=index, columns=list(bybit_keys)))
        liquidity_parts.append(bybit_liquidity.rename(columns=um.bybit_key).reindex(index=index, columns=list(bybit_keys)))

    cost_frame = pd.concat(cost_parts, axis=1).reindex(index=index, columns=list(symbols)) if cost_parts else pd.DataFrame(index=index, columns=list(symbols))
    liquidity_frame = (
        pd.concat(liquidity_parts, axis=1).reindex(index=index, columns=list(symbols)) if liquidity_parts else pd.DataFrame(index=index, columns=list(symbols))
    )
    fallback_rate = costs_measured.cost_rate_from_bp(binance_mapping.worst_bp, stress_multiplier)
    cost_frame = cost_frame.fillna(fallback_rate)
    liquidity_frame = liquidity_frame.fillna(False)
    return cost_frame, liquidity_frame


def _run_carry_loop(
    spot_open_frame: pd.DataFrame,
    spot_close_frame: pd.DataFrame,
    perp_open_frame: pd.DataFrame,
    perp_close_frame: pd.DataFrame,
    funding_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    active_frame: pd.DataFrame,
    top_k: int,
    leg_fraction: float,
    active_capital: float,
    cost_rate_frame: pd.DataFrame,
    liquidity_ok_frame: pd.DataFrame,
) -> tuple[Wave10Result, float, pd.Series]:
    """Adapted copy of research.wave13_liquidity.engine13._run_liquidity_loop -- IDENTICAL
    per-timestamp economics (gap PnL, intraday PnL, turnover cost, trade-close bookkeeping,
    final forced unwind), with exactly one change: `capital = active_capital` is a
    parameter here instead of engine13's hardcoded `capital = ACTIVE_CAPITAL` ($90) module
    import. See this module's own top-of-file docstring for why a copy (not a further
    import) is the correct way to accommodate SPEC.md's sanctioned capital-tier axis, and
    tests/test_wave14.py for the equivalence pin at active_capital=$90."""
    capital = active_capital
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


def compute_daily_bybit_share(
    score_frame: pd.DataFrame,
    active_frame: pd.DataFrame,
    liquidity_ok_frame: pd.DataFrame,
    available_mask_frame: pd.DataFrame,
    top_k: int,
    venue_of: dict[str, str],
) -> pd.Series:
    """Side-channel, capital-independent replay of _run_carry_loop's own eligible/ranked
    selection (identical rule: active AND available AND liquidity-ok, then top_k by score)
    -- used ONLY to derive what FRACTION of a day's filled slots are Bybit-sourced. Every
    filled slot carries the identical `leg_fraction` weight, so "fraction of filled slots
    that are Bybit" IS "fraction of that day's gross exposure that is Bybit" exactly, with
    no need to re-run the capital/cost bookkeeping just to get this one descriptive number.
    Feeds gates14.gate_s6_pool_venue_exposure's empirical figure (M1/M3/M4/M5 only,
    informational -- see that gate's own docstring for why it never blocks promotion).
    NaN on days with zero filled slots (no exposure to attribute a share to, not 0%)."""
    shares: list[float] = []
    for timestamp in score_frame.index:
        eligible = active_frame.loc[timestamp][active_frame.loc[timestamp] > 0.0].index
        eligible = eligible.intersection(available_mask_frame.loc[timestamp][available_mask_frame.loc[timestamp]].index)
        liquidity_row = liquidity_ok_frame.loc[timestamp]
        eligible = eligible.intersection(liquidity_row[liquidity_row].index)
        ranked = score_frame.loc[timestamp, eligible].dropna().nlargest(top_k).index
        if len(ranked) == 0:
            shares.append(float("nan"))
            continue
        bybit_count = sum(1 for symbol in ranked if venue_of.get(symbol) == "bybit")
        shares.append(bybit_count / len(ranked))
    return pd.Series(shares, index=score_frame.index, dtype=float)


def run_carry_candidate(
    config: Wave14Config,
    binance_mapping: MeasuredCostMapping,
    bybit_mappings: BybitCostMappings | None,
    stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER,
) -> tuple[Wave10Result, float, pd.Series, pd.Series]:
    """M0-M5 dispatch: assemble the (possibly Bybit-unioned) candidate pool, build aligned
    frames + venue-routed cost/liquidity frames, slice to OVERLAP_START..OVERLAP_END, hand
    off to this module's own _run_carry_loop (an active_capital-parameterized adaptation of
    engine13's `_run_liquidity_loop` -- see module docstring). Returns a 4-tuple (result,
    total_cost, eligible_series, daily_bybit_share) -- the extra 4th element vs.
    run_cross_venue_candidate's 3-tuple is deliberate, not an oversight: only M0-M5's pool
    structure has a "which venue was this slot filled from" question to answer at all."""
    markets, venue_of = um.markets_for_carry_config(config.include_bybit)
    candidate = config.candidate
    frames = _build_aligned_frames_multi(markets, candidate)
    spot_open_frame = frames[0]
    symbols = tuple(spot_open_frame.columns)
    cost_rate_frame, liquidity_ok_frame = build_multivenue_cost_and_liquidity_frames(
        symbols, venue_of, spot_open_frame.index, binance_mapping, bybit_mappings, stress_multiplier
    )
    window_mask = (spot_open_frame.index >= OVERLAP_START) & (spot_open_frame.index < OVERLAP_END)
    sliced_frames = tuple(frame.loc[window_mask] for frame in frames)
    sliced_cost = cost_rate_frame.loc[window_mask]
    sliced_liquidity = liquidity_ok_frame.loc[window_mask]
    result, total_cost, eligible = _run_carry_loop(
        *sliced_frames, candidate.top_k, config.leg_fraction, config.active_capital, sliced_cost, sliced_liquidity
    )
    available_mask_frame = sliced_frames[0].notna() & sliced_frames[1].notna() & sliced_frames[2].notna() & sliced_frames[3].notna()
    bybit_share = compute_daily_bybit_share(sliced_frames[5], sliced_frames[6], sliced_liquidity, available_mask_frame, candidate.top_k, venue_of)
    # engine13's Wave10Result.symbols_used is set from the (pre-slice) frame columns, which
    # is what we want (the full candidate pool the config could have drawn from, not just
    # the subset that happened to trade) -- but _run_liquidity_loop derives it from the
    # frame it was actually handed, so rebuild it explicitly here for clarity/robustness
    # rather than relying on that incidental equivalence.
    result = Wave10Result(
        equity=result.equity,
        positions=result.positions,
        turnover=result.turnover,
        trade_returns=result.trade_returns,
        max_concurrent_positions=result.max_concurrent_positions,
        symbols_used=symbols,
    )
    return result, total_cost, eligible, bybit_share


# ---------------------------------------------------------------------------
# M6/M7: cross-venue funding-spread structure (new, separate family).
# ---------------------------------------------------------------------------


def cross_venue_position(raw_spread: pd.Series, candidate: FundingCandidate) -> tuple[pd.Series, pd.Series]:
    """Entry/exit hysteresis for the cross-venue spread signal, analogous to
    research.wave1.fam_funding.carry_position but tracking a SIDE alongside the
    active/inactive state -- carry_position has no side to track (a single-venue carry
    trade only ever has one direction: long spot/short perp). `raw_spread` is SIGNED
    (score_binance - score_bybit, APR): positive means Binance's own funding is currently
    richer. Side is decided ONCE, at entry, and held fixed for the life of the trade
    (documented design choice, not SPEC.md-mandated): SPEC.md's "동일 심볼을 펀딩 높은
    거래소에서 숏퍼프 + 낮은 거래소에서 롱퍼프" describes which side to take, not what to do
    if the ranking transiently flips mid-trade. Re-flipping the side mid-trade would mean
    closing BOTH legs and reopening BOTH legs in the opposite direction (a full round-trip
    cost twice over) purely from a same-day ranking wobble; locking the side at entry and
    exiting via the SAME threshold/2 hysteresis band carry_position already uses (measured
    along the locked side's own direction) is the more conservative, cost-honest choice and
    keeps this function a direct structural sibling of carry_position rather than a new,
    untested state machine. Returns (active, side), both shift(1)'d exactly like
    carry_position's own output (signal known at t's close, acted on at t+1's open)."""
    active_values: list[float] = []
    side_values: list[float] = []
    active = 0.0
    side = 0.0
    for value in raw_spread:
        if pd.notna(value):
            if active == 0.0:
                if value > candidate.threshold_apr:
                    active, side = 1.0, 1.0  # Binance richer -> short Binance / long Bybit
                elif value < -candidate.threshold_apr:
                    active, side = 1.0, -1.0  # Bybit richer -> short Bybit / long Binance
            else:
                signed_in_locked_direction = value * side
                if signed_in_locked_direction < candidate.threshold_apr / 2.0:
                    active, side = 0.0, 0.0
        active_values.append(active)
        side_values.append(side)
    active_series = pd.Series(active_values, index=raw_spread.index, dtype=float).shift(1).fillna(0.0)
    side_series = pd.Series(side_values, index=raw_spread.index, dtype=float).shift(1).fillna(0.0)
    return active_series, side_series


def _build_cross_venue_frames(
    pairs: dict[str, CrossVenuePair], candidate: FundingCandidate
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (binance_open, binance_close, bybit_open, bybit_close, combined_funding_diff,
    abs_score_frame[ranking, shift(1)'d], active_frame, side_frame) -- deliberately NOT
    reusing _build_aligned_frames_multi's 7-tuple shape 1:1 (there is no single "spot" vs
    "perp" pair here, both legs are perpetuals on different venues, and there is a genuinely
    new 8th element -- side_frame, from cross_venue_position -- with no analogue in the
    ordinary carry structure) even though the underlying daily-resample/score/hysteresis
    machinery is otherwise the same idiom."""
    binance_open: dict[str, pd.Series] = {}
    binance_close: dict[str, pd.Series] = {}
    bybit_open: dict[str, pd.Series] = {}
    bybit_close: dict[str, pd.Series] = {}
    funding_diff: dict[str, pd.Series] = {}  # funding_binance_daily - funding_bybit_daily
    abs_scores: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    side: dict[str, pd.Series] = {}
    for symbol, pair in pairs.items():
        binance_daily = pair.binance_perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        bybit_daily = pair.bybit_perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        binance_funding_daily = pair.binance_funding.resample("1D").sum()
        bybit_funding_daily = pair.bybit_funding.resample("1D").sum()
        binance_open[symbol] = binance_daily["open"]
        binance_close[symbol] = binance_daily["close"]
        bybit_open[symbol] = bybit_daily["open"]
        bybit_close[symbol] = bybit_daily["close"]
        score_binance = daily_funding_score(binance_funding_daily, candidate.window_days)
        score_bybit = daily_funding_score(bybit_funding_daily, candidate.window_days)
        aligned_index = score_binance.index.union(score_bybit.index)
        raw_spread = score_binance.reindex(aligned_index) - score_bybit.reindex(aligned_index)
        funding_diff[symbol] = (binance_funding_daily.reindex(aligned_index) - bybit_funding_daily.reindex(aligned_index)).fillna(0.0)
        abs_scores[symbol] = raw_spread.abs()
        active_series, side_series = cross_venue_position(raw_spread, candidate)
        active[symbol] = active_series
        side[symbol] = side_series

    binance_open_frame = pd.DataFrame(binance_open).sort_index()
    binance_close_frame = pd.DataFrame(binance_close).reindex(binance_open_frame.index)
    bybit_open_frame = pd.DataFrame(bybit_open).reindex(binance_open_frame.index)
    bybit_close_frame = pd.DataFrame(bybit_close).reindex(binance_open_frame.index)
    funding_diff_frame = pd.DataFrame(funding_diff).reindex(binance_open_frame.index).fillna(0.0)
    abs_score_frame = pd.DataFrame(abs_scores).reindex(binance_open_frame.index).shift(1)
    active_frame = pd.DataFrame(active).reindex(binance_open_frame.index).fillna(0.0)
    side_frame = pd.DataFrame(side).reindex(binance_open_frame.index).fillna(0.0)
    return binance_open_frame, binance_close_frame, bybit_open_frame, bybit_close_frame, funding_diff_frame, abs_score_frame, active_frame, side_frame


def _run_cross_venue_loop(
    binance_open_frame: pd.DataFrame,
    binance_close_frame: pd.DataFrame,
    bybit_open_frame: pd.DataFrame,
    bybit_close_frame: pd.DataFrame,
    funding_diff_frame: pd.DataFrame,
    abs_score_frame: pd.DataFrame,
    active_frame: pd.DataFrame,
    side_frame: pd.DataFrame,
    top_k: int,
    leg_fraction: float,
    active_capital: float,
    cost_rate_frame: pd.DataFrame,
    liquidity_ok_frame: pd.DataFrame,
) -> tuple[Wave10Result, float, pd.Series]:
    """M6/M7's own loop -- structurally the same idiom as
    research.wave13_liquidity.engine13._run_liquidity_loop (gap PnL, intraday PnL, turnover
    cost, trade-close bookkeeping, final forced unwind) but with SIGNED weights (side *
    leg_fraction, side in {-1,0,+1}) instead of engine13's non-negative-only weights, and a
    combined price+funding return term derived from TWO PERP price series instead of
    spot-minus-perp. Per-unit-of-leg_fraction combined return (derivation in engine14.py's
    module-level comments / research/wave14_multivenue/SPEC.md discussion):

        E = (bybit_close/bybit_open - binance_close/binance_open) + (funding_binance_daily - funding_bybit_daily)

    is the return of "being +1 side" (short Binance / long Bybit, entered when Binance's
    funding is richer -- shorting the rich-funding venue earns its funding, exactly
    engine13's own "short perp earns +funding" sign convention, applied to Binance's leg;
    longing Bybit PAYS its own funding, hence the minus). A -1-side position (short Bybit /
    long Binance) earns the exact negation, -E -- so `weights[symbol] = side * leg_fraction`
    and `capital *= 1 + sum(weights * E)` covers both directions with one formula, no
    per-side branching in the hot loop. `cost_rate_frame` must already be the COMBINED
    (Binance-leg + Bybit-leg) rate from costs_venue.cross_venue_leg_cost_rate -- turnover
    cost is `abs(weight_change) * cost_for(symbol)` exactly as in engine13, which correctly
    prices a full entry/exit (weight_change == leg_fraction) since the hysteresis in
    cross_venue_position never flips side without first returning to 0 (no same-step
    double-leg-flip case to worry about -- see that function's own docstring)."""
    capital = active_capital
    equity_values: list[float] = []
    turnover_values: list[float] = []
    exposures: list[float] = []
    concurrent_counts: list[int] = []
    eligible_counts: list[int] = []
    trade_values: list[float] = []
    trade_times: list[pd.Timestamp] = []
    previous_weights = pd.Series(0.0, index=binance_open_frame.columns)
    trade_growth: dict[str, float] = {}
    trade_weights: dict[str, float] = {}
    total_cost_usdt = 0.0

    def cost_for(symbol: str, timestamp: pd.Timestamp) -> float:
        return float(cost_rate_frame.loc[timestamp, symbol])

    for timestamp in binance_open_frame.index:
        available = (
            binance_open_frame.loc[timestamp].notna()
            & binance_close_frame.loc[timestamp].notna()
            & bybit_open_frame.loc[timestamp].notna()
            & bybit_close_frame.loc[timestamp].notna()
        )
        eligible = active_frame.loc[timestamp][active_frame.loc[timestamp] > 0.0].index
        eligible = eligible.intersection(available[available].index)
        liquidity_row = liquidity_ok_frame.loc[timestamp]
        eligible = eligible.intersection(liquidity_row[liquidity_row].index)
        eligible_counts.append(int(len(eligible)))
        ranked = abs_score_frame.loc[timestamp, eligible].dropna().nlargest(top_k).index
        weights = pd.Series(0.0, index=binance_open_frame.columns)
        if len(ranked) > 0:
            weights.loc[ranked] = side_frame.loc[timestamp, ranked] * leg_fraction

        binance_gap = binance_open_frame.loc[timestamp] / binance_close_frame.shift(1).loc[timestamp] - 1.0
        bybit_gap = bybit_open_frame.loc[timestamp] / bybit_close_frame.shift(1).loc[timestamp] - 1.0
        gap_by_symbol = (bybit_gap - binance_gap).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        capital *= 1.0 + float((gap_by_symbol * previous_weights).sum())

        turnover = float((weights - previous_weights).abs().sum())
        cost_return = sum(abs(float(weights[symbol] - previous_weights[symbol])) * cost_for(symbol, timestamp) for symbol in binance_open_frame.columns)
        capital_before_cost = capital
        capital *= 1.0 - cost_return
        total_cost_usdt += capital_before_cost - capital

        price_diff = bybit_close_frame.loc[timestamp] / bybit_open_frame.loc[timestamp] - binance_close_frame.loc[timestamp] / binance_open_frame.loc[timestamp]
        combined = (price_diff + funding_diff_frame.loc[timestamp]).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        capital *= 1.0 + float((combined * weights).sum())

        for symbol in binance_open_frame.columns:
            previous_weight = float(previous_weights[symbol])
            current_weight = float(weights[symbol])
            leg_rate = cost_for(symbol, timestamp)
            trade_side = 1.0 if (current_weight > 0.0 or (current_weight == 0.0 and previous_weight > 0.0)) else -1.0
            signed_gap = trade_side * float(gap_by_symbol[symbol])
            signed_combined = trade_side * float(combined[symbol])
            if previous_weight != 0.0 and symbol in trade_growth:
                trade_growth[symbol] *= 1.0 + signed_gap
            if previous_weight != 0.0 and current_weight == 0.0:
                trade_growth[symbol] *= 1.0 - leg_rate
                trade_values.append((trade_growth.pop(symbol) - 1.0) * trade_weights.pop(symbol))
                trade_times.append(pd.Timestamp(timestamp))
            elif previous_weight == 0.0 and current_weight != 0.0:
                trade_growth[symbol] = 1.0 - leg_rate
                trade_weights[symbol] = abs(current_weight)
            elif previous_weight != 0.0 and current_weight != 0.0 and previous_weight != current_weight:
                trade_growth[symbol] *= 1.0 - abs(current_weight - previous_weight) * leg_rate / max(abs(current_weight), abs(previous_weight))
                trade_weights[symbol] = abs(current_weight)
            if current_weight != 0.0:
                trade_growth[symbol] *= 1.0 + signed_combined
        equity_values.append(capital)
        turnover_values.append(turnover)
        exposures.append(float(weights.abs().sum()))
        concurrent_counts.append(int((weights != 0.0).sum()))
        previous_weights = weights

    if len(binance_open_frame.index) > 0 and float(previous_weights.abs().sum()) > 0.0:
        final_timestamp = pd.Timestamp(binance_open_frame.index[-1])
        final_cost = sum(abs(float(previous_weights[symbol])) * cost_for(symbol, final_timestamp) for symbol in binance_open_frame.columns)
        capital_before_final_cost = capital
        capital *= 1.0 - final_cost
        total_cost_usdt += capital_before_final_cost - capital
        equity_values[-1] = capital
        turnover_values[-1] += float(previous_weights.abs().sum())
        for symbol, growth in trade_growth.items():
            leg_rate = cost_for(symbol, final_timestamp)
            trade_values.append((growth * (1.0 - leg_rate) - 1.0) * trade_weights[symbol])
            trade_times.append(final_timestamp)

    equity = pd.Series(equity_values, index=binance_open_frame.index, dtype=float)
    positions = pd.Series(exposures, index=binance_open_frame.index, dtype=float)
    turnover_series = pd.Series(turnover_values, index=binance_open_frame.index, dtype=float)
    trades = pd.Series(trade_values, index=pd.DatetimeIndex(trade_times), dtype=float).sort_index()
    eligible_series = pd.Series(eligible_counts, index=binance_open_frame.index, dtype=float)
    result = Wave10Result(
        equity=equity,
        positions=positions,
        turnover=turnover_series,
        trade_returns=trades,
        max_concurrent_positions=max(concurrent_counts, default=0),
        symbols_used=tuple(binance_open_frame.columns),
    )
    return result, total_cost_usdt, eligible_series


def run_cross_venue_candidate(
    config: Wave14Config,
    binance_mapping: MeasuredCostMapping,
    bybit_linear_mapping: MeasuredCostMapping,
    stress_multiplier: float = DEFAULT_STRESS_MULTIPLIER,
) -> tuple[Wave10Result, float, pd.Series]:
    pairs = um.cross_venue_pairs()
    candidate = config.candidate
    frames = _build_cross_venue_frames(pairs, candidate)
    binance_open_frame = frames[0]
    symbols = tuple(binance_open_frame.columns)

    binance_volume = um.load_binance_quote_volume_frame(symbols)
    bybit_volume = um.load_bybit_quote_volume_frame(symbols, "linear")
    binance_bp = costs_venue.build_bp_frame_for_market(binance_volume, symbols, binance_mapping)
    bybit_bp = costs_venue.build_bp_frame_for_market(bybit_volume, symbols, bybit_linear_mapping)
    cost_rate_frame = costs_venue.cross_venue_leg_cost_rate(binance_bp, bybit_bp, stress_multiplier)
    binance_liquidity = costs_venue.build_liquidity_mask_for_market(binance_volume, symbols)
    bybit_liquidity = costs_venue.build_liquidity_mask_for_market(bybit_volume, symbols)
    liquidity_ok_frame = binance_liquidity & bybit_liquidity
    cost_rate_frame = cost_rate_frame.reindex(index=binance_open_frame.index, columns=list(symbols))
    liquidity_ok_frame = liquidity_ok_frame.reindex(index=binance_open_frame.index, columns=list(symbols))
    fallback_rate = costs_venue.cross_venue_leg_cost_rate(binance_mapping.worst_bp, bybit_linear_mapping.worst_bp, stress_multiplier)
    cost_rate_frame = cost_rate_frame.fillna(fallback_rate)
    liquidity_ok_frame = liquidity_ok_frame.fillna(False)

    window_mask = (binance_open_frame.index >= OVERLAP_START) & (binance_open_frame.index < OVERLAP_END)
    sliced_frames = tuple(frame.loc[window_mask] for frame in frames)
    sliced_cost = cost_rate_frame.loc[window_mask]
    sliced_liquidity = liquidity_ok_frame.loc[window_mask]
    return _run_cross_venue_loop(*sliced_frames, candidate.top_k, config.leg_fraction, config.active_capital, sliced_cost, sliced_liquidity)


__all__ = [
    "DEFAULT_STRESS_MULTIPLIER",
    "OVERLAP_END",
    "OVERLAP_START",
    "STRESS_MULTIPLIER",
    "build_multivenue_cost_and_liquidity_frames",
    "compute_daily_bybit_share",
    "cross_venue_position",
    "daily_funding_score",
    "run_carry_candidate",
    "run_cross_venue_candidate",
]
