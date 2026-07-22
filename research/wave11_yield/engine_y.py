# Wave-11 engine dispatch: for 4 of 6 candidates (Y1/Y2/Y3/Y4) this module writes NO new
# backtest loop at all -- it calls research.wave10_carry100.engine.run_fixed_fraction_portfolio
# unmodified, only varying the FundingCandidate/leg_fraction (Y1/Y2/Y3, same as wave10's own
# C1-C4 pattern) or the `markets` dict fed into it (Y4, expanded universe -- the loop that
# consumes markets never changes). Two candidates need genuinely new mechanics that
# research.wave10_carry100.engine does not expose a hook for (and this wave may not modify
# that file): Y5 needs 8h-bar resampling instead of 1D, and Y6 needs a funding-spike
# active/eligibility rule instead of carry_position's threshold hysteresis. Both reuse the
# exact same per-timestamp bookkeeping (_run_fixed_fraction_loop below is a straight copy of
# wave10 engine.run_fixed_fraction_portfolio's per-timestamp loop body -- same gap/cost
# /intraday/turnover/trade-close formulas, same cost_rate import) so the only thing that
# actually differs between wave11's two new engines and wave10's original is exactly the one
# thing each candidate's SPEC.md row says should differ.
#
# Delta-neutral invariant (S1): unchanged from wave10 -- a single shared `weights` value
# drives both the long-spot and short-perp leg of every `intraday = spot_ret - perp_ret +
# funding` term, in every engine function here, exactly as in
# research/wave10_carry100/engine.py's own docstring. tests/test_wave11.py carries the same
# regression proof wave10's test suite uses (diverging-basis synthetic market).

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import CACHE_DIR as WAVE1_CACHE_DIR
from research.wave1.common import load_frame, load_json
from research.wave1.fam_funding import FundingCandidate, FundingMarket, carry_position, funding_score
from research.wave10_carry100.engine import (
    ACTIVE_CAPITAL,
    MIN_ORDER_USDT,
    OOS_SPLIT,
    RESERVE_FRACTION,
    TOTAL_CAPITAL,
    Wave10Result,
    cost_rate,
    load_universe_symbols,
)
from research.wave11_yield.configs import Wave11Config, Y6_SPIKE_ENTRY_RATE, Y6_SPIKE_HOLD_DAYS

WAVE6_CACHE_DIR: Final = Path(__file__).resolve().parents[1] / "wave6" / "cache"
WAVE11_CACHE_DIR: Final = Path(__file__).resolve().parent / "cache"
MAJOR_SYMBOLS_Y5: Final = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


# ---------------------------------------------------------------------------
# Market loading. Three universes: baseline40 (Y1/Y2/Y3/Y6), expanded_y4 (Y4), majors_8h (Y5).
# ---------------------------------------------------------------------------


def _first_existing(filename: str) -> Path | None:
    """wave11's own cache (a corrected, non-truncated spot-1d refetch for every symbol
    it touches -- see fetch_y11.py's module docstring) takes priority over wave1's when
    both exist; perp/funding are unaffected by that bug and are simply wherever they are."""
    wave11_path = WAVE11_CACHE_DIR / filename
    if wave11_path.exists():
        return wave11_path
    wave1_path = WAVE1_CACHE_DIR / filename
    if wave1_path.exists():
        return wave1_path
    return None


def _required_market_files(symbol: str) -> tuple[str, str, str]:
    return (f"binance_spot_{symbol}_1d.csv.gz", f"binance_fapi_{symbol}_1d.csv.gz", f"binance_funding_{symbol}.csv.gz")


def missing_market_files(symbols: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for symbol in symbols:
        missing.extend(filename for filename in _required_market_files(symbol) if _first_existing(filename) is None)
    return missing


def load_markets_merged(symbols: tuple[str, ...]) -> dict[str, FundingMarket]:
    """Same contract as research.wave1.fam_funding.load_markets, but each of a symbol's
    three files is resolved independently across research/wave11_yield/cache and
    research/wave1/cache (see _first_existing)."""
    markets: dict[str, FundingMarket] = {}
    for symbol in symbols:
        spot_name, perp_name, funding_name = _required_market_files(symbol)
        spot_path, perp_path, funding_path = _first_existing(spot_name), _first_existing(perp_name), _first_existing(funding_name)
        if spot_path is not None and perp_path is not None and funding_path is not None:
            markets[symbol] = FundingMarket(spot=load_frame(spot_path), perp=load_frame(perp_path), funding=load_frame(funding_path)["funding_rate"])
    return markets


def verify_cache_and_load_symbols_baseline40() -> tuple[str, ...]:
    """Fail-closed cache check for the baseline-40 universe (Y1/Y2/Y3/Y6), mirroring
    research.wave10_carry100.engine.verify_cache_and_load_symbols but resolving each
    file through load_markets_merged's two-directory search."""
    symbols = load_universe_symbols()
    missing = missing_market_files(symbols)
    if missing:
        raise RuntimeError(f"wave-1/wave-11 cache incomplete for baseline40 universe: {', '.join(sorted(set(missing))[:8])}")
    return symbols


def verify_cache_and_load_symbols_y4() -> tuple[str, ...]:
    """Fail-closed cache check for Y4's expanded universe. Requires
    research/wave11_yield/cache/universe_y4.json (written by
    fetch_y11.expand_universe_y4 during `--stage fetch`) -- performs no network access."""
    path = WAVE11_CACHE_DIR / "universe_y4.json"
    if not path.exists():
        raise RuntimeError("research/wave11_yield/cache/universe_y4.json missing -- run `--stage fetch` first (Y4 needs the expanded-universe fetch)")
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise RuntimeError("research/wave11_yield/cache/universe_y4.json is invalid")
    symbols = tuple(str(symbol) for symbol in payload["symbols"])
    missing = missing_market_files(symbols)
    if missing:
        raise RuntimeError(f"wave-11 Y4 cache incomplete: {', '.join(sorted(set(missing))[:8])}")
    return symbols


def verify_cache_and_load_markets_y5() -> dict[str, FundingMarket]:
    """Fail-closed cache check + load for Y5's majors-8h universe: spot 1h (wave11's own
    fetch), perp 1h (research/wave6/cache, reused read-only), funding (research/wave1/cache,
    native 8h, reused read-only)."""
    markets: dict[str, FundingMarket] = {}
    missing: list[str] = []
    for symbol in MAJOR_SYMBOLS_Y5:
        spot_path = WAVE11_CACHE_DIR / f"binance_spot_{symbol}_1h.csv.gz"
        perp_path = WAVE6_CACHE_DIR / f"binance_fapi_{symbol}_1h.csv.gz"
        funding_path = WAVE1_CACHE_DIR / f"binance_funding_{symbol}.csv.gz"
        paths = (spot_path, perp_path, funding_path)
        if all(path.exists() for path in paths):
            markets[symbol] = FundingMarket(spot=load_frame(spot_path), perp=load_frame(perp_path), funding=load_frame(funding_path)["funding_rate"])
        else:
            missing.extend(path.name for path in paths if not path.exists())
    if missing:
        raise RuntimeError(f"wave-11 Y5 cache incomplete: {', '.join(missing)} -- run `--stage fetch` first")
    return markets


# ---------------------------------------------------------------------------
# Shared per-timestamp bookkeeping (copied unmodified from
# research.wave10_carry100.engine.run_fixed_fraction_portfolio's loop body). Used by both
# new engines below so the ONLY thing that differs between them and wave10's original is
# the axis each one is actually testing (bar frequency for Y5, active-signal source for Y6).
# ---------------------------------------------------------------------------


def _run_fixed_fraction_loop(
    spot_open_frame: pd.DataFrame,
    spot_close_frame: pd.DataFrame,
    perp_open_frame: pd.DataFrame,
    perp_close_frame: pd.DataFrame,
    funding_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    active_frame: pd.DataFrame,
    top_k: int,
    leg_fraction: float,
) -> tuple[Wave10Result, float]:
    """Returns (result, total_cost_usdt). total_cost_usdt is the exact sum, in dollars,
    of every maker-fee+slippage deduction actually applied to `capital` over the run
    (entries, exits, rebalances, and the final forced unwind) -- captured as
    capital-before-cost minus capital-after-cost at each step, not a blended-rate
    approximation, so it is directly comparable to the equity path's own compounding.
    This is a new field wave10's own Wave10Result doesn't carry (its dataclass is reused
    unmodified for the equity/positions/turnover/trade_returns side); wave11's SPEC
    requires disclosing total cost per candidate, especially for the high-turnover Y5/Y6
    candidates, so it is tracked here where the loop already has exact access to it."""
    capital = ACTIVE_CAPITAL
    equity_values: list[float] = []
    turnover_values: list[float] = []
    exposures: list[float] = []
    concurrent_counts: list[int] = []
    trade_values: list[float] = []
    trade_times: list[pd.Timestamp] = []
    previous_weights = pd.Series(0.0, index=spot_open_frame.columns)
    trade_growth: dict[str, float] = {}
    trade_weights: dict[str, float] = {}
    total_cost_usdt = 0.0

    def cost_for(symbol: str) -> float:
        return cost_rate(symbol)

    for timestamp in spot_open_frame.index:
        available = (
            spot_open_frame.loc[timestamp].notna()
            & spot_close_frame.loc[timestamp].notna()
            & perp_open_frame.loc[timestamp].notna()
            & perp_close_frame.loc[timestamp].notna()
        )
        eligible = active_frame.loc[timestamp][active_frame.loc[timestamp] > 0.0].index
        eligible = eligible.intersection(available[available].index)
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
            abs(float(weights[symbol] - previous_weights[symbol])) * cost_for(symbol)
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
            leg_rate = cost_for(symbol)
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
        final_cost = sum(float(previous_weights[symbol]) * cost_for(symbol) for symbol in spot_open_frame.columns)
        capital_before_final_cost = capital
        capital *= 1.0 - final_cost
        total_cost_usdt += capital_before_final_cost - capital
        equity_values[-1] = capital
        turnover_values[-1] += float(previous_weights.abs().sum())
        final_timestamp = pd.Timestamp(spot_open_frame.index[-1])
        for symbol, growth in trade_growth.items():
            leg_rate = cost_for(symbol)
            trade_values.append((growth * (1.0 - leg_rate) - 1.0) * trade_weights[symbol])
            trade_times.append(final_timestamp)

    equity = pd.Series(equity_values, index=spot_open_frame.index, dtype=float)
    positions = pd.Series(exposures, index=spot_open_frame.index, dtype=float)
    turnover_series = pd.Series(turnover_values, index=spot_open_frame.index, dtype=float)
    trades = pd.Series(trade_values, index=pd.DatetimeIndex(trade_times), dtype=float).sort_index()
    result = Wave10Result(
        equity=equity,
        positions=positions,
        turnover=turnover_series,
        trade_returns=trades,
        max_concurrent_positions=max(concurrent_counts, default=0),
        symbols_used=tuple(spot_open_frame.columns),
    )
    return result, total_cost_usdt


# ---------------------------------------------------------------------------
# Y6: funding-spike active/eligibility rule (SPEC.md: single 8h funding print >0.05%
# triggers entry, 3-daily-mark hold or 7d APR<5% forces exit, whichever first).
# ---------------------------------------------------------------------------

ActiveBuilder = Callable[[FundingCandidate, FundingMarket, pd.Series], pd.Series]


def _default_active_builder(candidate: FundingCandidate, market: FundingMarket, funding_apr: pd.Series) -> pd.Series:
    del market  # unused: standard hysteresis only looks at the score, matching carry_position's own signature
    return carry_position(funding_apr, candidate)


def y6_spike_active_builder(candidate: FundingCandidate, market: FundingMarket, funding_apr: pd.Series) -> pd.Series:
    """3 consecutive active daily marks starting the entry day (hold_days 1,2,3), forced
    flat starting the 4th -- OR earlier the moment the 7d annualized score
    (candidate.threshold_apr, reused as Y6's exit-APR field -- see configs.py) drops
    below threshold. Re-entry the same day it goes flat is allowed if a fresh spike is
    observed. The shared shift(1) at the end applies the usual t-close-signal ->
    t+1-open-execution lag, matching carry_position's own contract."""
    daily_spike = market.funding.resample("1D").max()
    idx = funding_apr.index
    spike_today = daily_spike.reindex(idx).fillna(0.0)
    values: list[float] = []
    active = 0.0
    hold_days = 0
    for date in idx:
        apr = funding_apr.loc[date]
        spike = spike_today.loc[date]
        if active == 1.0:
            hold_days += 1
        elif spike > Y6_SPIKE_ENTRY_RATE:
            active = 1.0
            hold_days = 1
        values.append(active)
        if active == 1.0 and (hold_days >= Y6_SPIKE_HOLD_DAYS or (pd.notna(apr) and apr < candidate.threshold_apr)):
            active = 0.0
            hold_days = 0
    return pd.Series(values, index=idx, dtype=float).shift(1).fillna(0.0)


def run_daily_fixed_fraction(
    markets: dict[str, FundingMarket],
    candidate: FundingCandidate,
    leg_fraction: float,
    active_builder: ActiveBuilder = _default_active_builder,
) -> tuple[Wave10Result, float]:
    """Same per-symbol daily-bar frame construction and per-timestamp bookkeeping as
    research.wave10_carry100.engine.run_fixed_fraction_portfolio (numerically identical
    for the default active_builder -- see
    tests/test_wave11.py::test_default_daily_engine_matches_wave10_engine_exactly for a
    regression proof against the real, unmodified wave10 function on a synthetic
    market). Used directly (with the default carry_position-based active_builder) for
    Y1/Y2/Y3 (baseline40 universe) and Y4 (expanded universe, only `markets` differs);
    Y6 supplies its own active_builder (y6_spike_active_builder). Returns (result,
    total_cost_usdt) -- see _run_fixed_fraction_loop's docstring for what the cost
    figure is exactly."""
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
        active[symbol] = active_builder(candidate, market, funding_apr)

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    spot_close_frame = pd.DataFrame(spot_close).reindex(spot_open_frame.index)
    perp_open_frame = pd.DataFrame(perp_open).reindex(spot_open_frame.index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(spot_open_frame.index)
    funding_frame = pd.DataFrame(funding_returns).reindex(spot_open_frame.index).fillna(0.0)
    score_frame = pd.DataFrame(scores).reindex(spot_open_frame.index).shift(1)
    active_frame = pd.DataFrame(active).reindex(spot_open_frame.index).fillna(0.0)

    return _run_fixed_fraction_loop(
        spot_open_frame, spot_close_frame, perp_open_frame, perp_close_frame, funding_frame, score_frame, active_frame, candidate.top_k, leg_fraction
    )


# ---------------------------------------------------------------------------
# Y5: 8h-bar engine (rebalance/execution aligned to funding settlement cadence).
# ---------------------------------------------------------------------------


def run_8h_fixed_fraction(markets: dict[str, FundingMarket], candidate: FundingCandidate, leg_fraction: float) -> tuple[Wave10Result, float]:
    """Same formulas as wave10's daily loop, resampled to 8h bars (pandas' default '8h'
    resample origin divides the day evenly, landing on Binance's 00:00/08:00/16:00 UTC
    funding settlements) instead of 1D. Standard carry_position hysteresis, unchanged
    from the baseline. Caller must restrict `markets` to symbols with real 8h-resolution
    spot+perp data (verify_cache_and_load_markets_y5 enforces this for Y5)."""
    spot_open: dict[str, pd.Series] = {}
    spot_close: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    funding_returns: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        funding_8h = market.funding.resample("8h").sum()
        funding_apr = funding_score(market.funding, candidate.window_days).resample("8h").last()
        spot_8h = market.spot.resample("8h").agg({"open": "first", "close": "last"}).dropna()
        perp_8h = market.perp.resample("8h").agg({"open": "first", "close": "last"}).dropna()
        spot_open[symbol] = spot_8h["open"]
        spot_close[symbol] = spot_8h["close"]
        perp_open[symbol] = perp_8h["open"]
        perp_close[symbol] = perp_8h["close"]
        funding_returns[symbol] = funding_8h
        scores[symbol] = funding_apr
        active[symbol] = carry_position(funding_apr, candidate)

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    spot_close_frame = pd.DataFrame(spot_close).reindex(spot_open_frame.index)
    perp_open_frame = pd.DataFrame(perp_open).reindex(spot_open_frame.index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(spot_open_frame.index)
    funding_frame = pd.DataFrame(funding_returns).reindex(spot_open_frame.index).fillna(0.0)
    score_frame = pd.DataFrame(scores).reindex(spot_open_frame.index).shift(1)
    active_frame = pd.DataFrame(active).reindex(spot_open_frame.index).fillna(0.0)

    return _run_fixed_fraction_loop(
        spot_open_frame, spot_close_frame, perp_open_frame, perp_close_frame, funding_frame, score_frame, active_frame, candidate.top_k, leg_fraction
    )


# ---------------------------------------------------------------------------
# Top-level dispatch used by run_wave11.py
# ---------------------------------------------------------------------------


def run_candidate(config: Wave11Config) -> tuple[Wave10Result, float]:
    """Returns (result, total_cost_usdt). Dispatches on config.universe/config.axis;
    every path funnels into _run_fixed_fraction_loop (via run_daily_fixed_fraction or
    run_8h_fixed_fraction) so cost is tracked identically for all 6 candidates."""
    candidate = config.candidate
    if config.universe == "baseline40":
        symbols = verify_cache_and_load_symbols_baseline40()
        markets = load_markets_merged(symbols)
        active_builder = y6_spike_active_builder if config.axis == "spike" else _default_active_builder
        return run_daily_fixed_fraction(markets, candidate, config.leg_fraction, active_builder)
    if config.universe == "expanded_y4":
        symbols = verify_cache_and_load_symbols_y4()
        markets = load_markets_merged(symbols)
        return run_daily_fixed_fraction(markets, candidate, config.leg_fraction)
    if config.universe == "majors_8h":
        markets = verify_cache_and_load_markets_y5()
        return run_8h_fixed_fraction(markets, candidate, config.leg_fraction)
    raise ValueError(f"unknown wave11 universe: {config.universe}")


__all__ = [
    "ACTIVE_CAPITAL",
    "MIN_ORDER_USDT",
    "OOS_SPLIT",
    "RESERVE_FRACTION",
    "TOTAL_CAPITAL",
    "MAJOR_SYMBOLS_Y5",
    "load_markets_merged",
    "missing_market_files",
    "run_8h_fixed_fraction",
    "run_candidate",
    "run_daily_fixed_fraction",
    "verify_cache_and_load_markets_y5",
    "verify_cache_and_load_symbols_baseline40",
    "verify_cache_and_load_symbols_y4",
    "y6_spike_active_builder",
]
