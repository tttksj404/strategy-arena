# Wave-10 fixed-fraction carry engine.
#
# Reuses (imports, does not reimplement) the wave-1/wave-2 signal and cost primitives:
#   - research.wave1.fam_funding.funding_score   -- 7d rolling funding APR score
#   - research.wave1.fam_funding.carry_position  -- entry/exit hysteresis on threshold_apr
#   - research.wave1.fam_funding.load_markets    -- cached spot/perp/funding loader
#   - research.wave2.funding.W2_MAKER_FEE_RATE   -- 0.02%/leg maker fee (wave2 cost regime)
#   - research.wave1.costs.slippage_rate         -- 1bp majors / 3bp alts slippage schedule
#
# The only rule change versus research.wave1.fam_funding.run_portfolio is the weight
# assignment: instead of splitting 100% of capital evenly across up to top_k ranked
# symbols (weight = 1/len(ranked), which is how W2c's gross reaches 2x active capital),
# each ranked symbol gets a FIXED fraction of active capital
# (weight = config.leg_fraction), independent of how many symbols are ranked. This is
# the "sizing and concurrent pair count" change the wave10 task explicitly authorizes;
# everything else about the day-loop (gap PnL, turnover cost, intraday PnL, trade
# bookkeeping, final unwind) mirrors research.wave1.fam_funding.run_portfolio exactly,
# because a single shared `weights` value drives both the long-spot and short-perp leg
# of `intraday = spot_ret - perp_ret + funding`, which is what keeps every position
# delta-neutral by construction (see tests/test_wave10_engine.py for a market-moves
# but basis-stays-zero regression proof of that invariant).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import CACHE_DIR as WAVE1_CACHE_DIR
from research.wave1.common import load_json
from research.wave1.costs import slippage_rate
from research.wave1.fam_funding import (
    FundingMarket,
    carry_position,
    funding_score,
    load_markets,
)
from research.wave2.funding import W2_MAKER_FEE_RATE
from research.wave10_carry100.configs import Wave10Config

TOTAL_CAPITAL: Final = 100.0
RESERVE_FRACTION: Final = 0.10
ACTIVE_CAPITAL: Final = TOTAL_CAPITAL * (1.0 - RESERVE_FRACTION)  # $90
MIN_ORDER_USDT: Final = 5.0
OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")


def cost_rate(symbol: str, stress_multiplier: float = 1.0) -> float:
    """One-way cost rate for a single leg-pair rebalance unit (matches the convention of
    research.wave1.fam_funding.run_portfolio's cost_for() / research.wave2.funding._maker_cost()):
    maker fee 0.02% on the spot leg + maker fee 0.02% on the perp leg (research.wave2's cost
    route), PLUS slippage on both legs from research.wave1.costs.slippage_rate (1bp BTC/ETH/SOL,
    3bp everything else in this universe). wave2's own W2c run used zero slippage (pure
    maker-fill assumption); wave10's task contract explicitly specifies maker fee + slippage
    together, which is a more conservative cost model than W2c's, applied identically to every
    wave10 config so the four configs stay comparable to each other.
    """
    return 2.0 * W2_MAKER_FEE_RATE + 2.0 * slippage_rate(symbol, stress_multiplier)


def load_universe_symbols() -> tuple[str, ...]:
    payload = load_json(WAVE1_CACHE_DIR / "universe.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("wave-1 cache universe.json is invalid or missing")
    return tuple(str(symbol) for symbol in payload["symbols"])


def required_cache_files(symbols: tuple[str, ...]) -> set[Path]:
    required = {WAVE1_CACHE_DIR / "universe.json"}
    for symbol in symbols:
        required.update(
            {
                WAVE1_CACHE_DIR / f"binance_spot_{symbol}_1d.csv.gz",
                WAVE1_CACHE_DIR / f"binance_fapi_{symbol}_1d.csv.gz",
                WAVE1_CACHE_DIR / f"binance_funding_{symbol}.csv.gz",
            }
        )
    return required


def verify_cache_and_load_symbols() -> tuple[str, ...]:
    """Fail-closed cache check: no network calls, every required wave-1 cache file must
    already exist on disk. Mirrors research/wave2/run_wave2.py's _required_cache_files gate."""
    symbols = load_universe_symbols()
    required = required_cache_files(symbols)
    missing = sorted(str(path.relative_to(WAVE1_CACHE_DIR)) for path in required if not path.exists())
    if missing:
        raise RuntimeError(f"wave-1 cache incomplete for wave10_carry100: {', '.join(missing[:8])}")
    return symbols


@dataclass(frozen=True, slots=True)
class Wave10Result:
    equity: pd.Series  # native USD, starts at ACTIVE_CAPITAL ($90)
    positions: pd.Series  # sum(|weight|) per day; 0.0 == flat that day
    turnover: pd.Series
    trade_returns: pd.Series  # portfolio-return-equivalent contribution per closed trade
    max_concurrent_positions: int
    symbols_used: tuple[str, ...]


def run_fixed_fraction_portfolio(markets: dict[str, FundingMarket], config: Wave10Config) -> Wave10Result:
    candidate = config.candidate
    leg_fraction = config.leg_fraction
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

    def cost_for(symbol: str) -> float:
        return cost_rate(symbol)

    for timestamp in spot_open_frame.index:
        start_capital = capital
        available = (
            spot_open_frame.loc[timestamp].notna()
            & spot_close_frame.loc[timestamp].notna()
            & perp_open_frame.loc[timestamp].notna()
            & perp_close_frame.loc[timestamp].notna()
        )
        eligible = active_frame.loc[timestamp][active_frame.loc[timestamp] > 0.0].index
        eligible = eligible.intersection(available[available].index)
        ranked = score_frame.loc[timestamp, eligible].dropna().nlargest(candidate.top_k).index
        weights = pd.Series(0.0, index=spot_open_frame.columns)
        if len(ranked) > 0:
            # The only sizing rule change vs research.wave1.fam_funding.run_portfolio:
            # a FIXED fraction of active capital per leg, not 1.0 / len(ranked).
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
        capital *= 1.0 - cost_return
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
        _ = start_capital  # kept for parity with the source loop; not separately reported
        equity_values.append(capital)
        turnover_values.append(turnover)
        exposures.append(float(weights.abs().sum()))
        concurrent_counts.append(int((weights != 0.0).sum()))
        previous_weights = weights

    if len(spot_open_frame.index) > 0 and float(previous_weights.abs().sum()) > 0.0:
        final_cost = sum(float(previous_weights[symbol]) * cost_for(symbol) for symbol in spot_open_frame.columns)
        capital *= 1.0 - final_cost
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
    return Wave10Result(
        equity=equity,
        positions=positions,
        turnover=turnover_series,
        trade_returns=trades,
        max_concurrent_positions=max(concurrent_counts, default=0),
        symbols_used=tuple(markets),
    )


def run_config(cache_dir: Path, symbols: tuple[str, ...], config: Wave10Config) -> Wave10Result:
    markets = load_markets(cache_dir, symbols)
    return run_fixed_fraction_portfolio(markets, config)
