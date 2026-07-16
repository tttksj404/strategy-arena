# F1 funding-carry grid, scoring, universe filtering, and hedged PnL.

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Callable, Final

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import StrategyResult, load_frame
from research.wave1.costs import PERP_TAKER_RATE, SPOT_TAKER_RATE, slippage_rate


@dataclass(frozen=True, slots=True)
class FundingCandidate:
    candidate_id: str
    window_days: int
    threshold_apr: float
    top_k: int
    majors_only: bool = False


@dataclass(frozen=True, slots=True)
class FundingResult:
    equity: pd.Series
    daily_pnl: pd.Series
    positions: pd.Series
    trade_returns: pd.Series
    max_concurrent_positions: int
    max_position_weight: float
    min_position_weight: float
    turnover: pd.Series


@dataclass(frozen=True, slots=True)
class FundingMarket:
    spot: pd.DataFrame
    perp: pd.DataFrame
    funding: pd.Series


@dataclass(frozen=True, slots=True)
class UniverseInputs:
    funding_months: pd.Series
    quote_volume: pd.Series
    spot_symbols: set[str]
    bitget_symbols: set[str]


F1_CANDIDATES: Final = (
    FundingCandidate("F1a", 3, 0.08, 2),
    FundingCandidate("F1b", 3, 0.15, 2),
    FundingCandidate("F1c", 7, 0.08, 2),
    FundingCandidate("F1d", 7, 0.15, 2),
    FundingCandidate("F1e", 7, 0.08, 2, True),
    FundingCandidate("F1f", 7, 0.15, 4),
)


def neighbor_candidates(candidate: FundingCandidate) -> tuple[FundingCandidate, ...]:
    windows = {max(1, round(candidate.window_days * 0.8)), round(candidate.window_days * 1.2)}
    thresholds = {candidate.threshold_apr * 0.8, candidate.threshold_apr * 1.2}
    top_ks = {max(1, int(candidate.top_k * 0.8)), ceil(candidate.top_k * 1.2)}
    return tuple(
        FundingCandidate(candidate.candidate_id, window, threshold, top_k, candidate.majors_only)
        for window in sorted(windows)
        for threshold in sorted(thresholds)
        for top_k in sorted(top_ks)
    )


def funding_score(funding: pd.Series, window_days: int) -> pd.Series:
    observations = window_days * 3
    return funding.rolling(observations, min_periods=observations).mean() * 3.0 * 365.0


def eligible_universe(inputs: UniverseInputs) -> tuple[str, ...]:
    eligible = inputs.funding_months.index[
        (inputs.funding_months >= 24.0)
        & inputs.funding_months.index.to_series().isin(inputs.spot_symbols)
        & inputs.funding_months.index.to_series().isin(inputs.bitget_symbols)
    ]
    ranked = inputs.quote_volume.reindex(eligible).dropna().sort_values(ascending=False).head(40)
    return tuple(str(symbol) for symbol in ranked.index)


def carry_position(score: pd.Series, candidate: FundingCandidate) -> pd.Series:
    values: list[float] = []
    active = 0.0
    for value in score:
        if pd.notna(value) and value > candidate.threshold_apr:
            active = 1.0
        elif pd.notna(value) and value < candidate.threshold_apr / 2.0:
            active = 0.0
        values.append(active)
    return pd.Series(values, index=score.index, dtype=float).shift(1).fillna(0.0)


def load_markets(cache_dir: Path, symbols: tuple[str, ...]) -> dict[str, FundingMarket]:
    markets: dict[str, FundingMarket] = {}
    for symbol in symbols:
        spot_path = cache_dir / f"binance_spot_{symbol}_1d.csv.gz"
        perp_path = cache_dir / f"binance_fapi_{symbol}_1d.csv.gz"
        funding_path = cache_dir / f"binance_funding_{symbol}.csv.gz"
        if spot_path.exists() and perp_path.exists() and funding_path.exists():
            funding_frame = load_frame(funding_path)
            markets[symbol] = FundingMarket(
                spot=load_frame(spot_path),
                perp=load_frame(perp_path),
                funding=funding_frame["funding_rate"],
            )
    return markets


def run_portfolio(
    markets: dict[str, FundingMarket],
    candidate: FundingCandidate,
    stress_multiplier: float = 1.0,
    position_builder: Callable[[pd.Series, pd.Series, FundingCandidate], pd.Series] | None = None,
    cost_model: Callable[[str, float], float] | None = None,
) -> FundingResult:
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
        active[symbol] = position_builder(market.funding, funding_apr, candidate) if position_builder is not None else carry_position(funding_apr, candidate)
    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    spot_close_frame = pd.DataFrame(spot_close).reindex(spot_open_frame.index)
    perp_open_frame = pd.DataFrame(perp_open).reindex(spot_open_frame.index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(spot_open_frame.index)
    funding_frame = pd.DataFrame(funding_returns).reindex(spot_open_frame.index).fillna(0.0)
    score_frame = pd.DataFrame(scores).reindex(spot_open_frame.index).shift(1)
    active_frame = pd.DataFrame(active).reindex(spot_open_frame.index).fillna(0.0)
    capital = 300.0
    equity_values: list[float] = []
    daily_pnl: list[float] = []
    exposures: list[float] = []
    trade_values: list[float] = []
    trade_times: list[pd.Timestamp] = []
    turnover_values: list[float] = []
    concurrent_counts: list[int] = []
    nonzero_weights: list[float] = []
    previous_weights = pd.Series(0.0, index=spot_open_frame.columns)
    trade_growth: dict[str, float] = {}
    trade_weights: dict[str, float] = {}
    slippage_factor = stress_multiplier
    def cost_for(symbol: str) -> float:
        return cost_model(symbol, slippage_factor) if cost_model is not None else SPOT_TAKER_RATE + PERP_TAKER_RATE + 2.0 * slippage_rate(symbol, slippage_factor)
    for timestamp in spot_open_frame.index:
        start_capital = capital
        available = spot_open_frame.loc[timestamp].notna() & spot_close_frame.loc[timestamp].notna() & perp_open_frame.loc[timestamp].notna() & perp_close_frame.loc[timestamp].notna()
        eligible = active_frame.loc[timestamp][active_frame.loc[timestamp] > 0.0].index
        eligible = eligible.intersection(available[available].index)
        if candidate.majors_only:
            eligible = eligible.intersection(pd.Index(["BTCUSDT", "ETHUSDT"]))
        ranked = score_frame.loc[timestamp, eligible].dropna().nlargest(candidate.top_k).index
        weights = pd.Series(0.0, index=spot_open_frame.columns)
        if len(ranked) > 0:
            weights.loc[ranked] = 1.0 / len(ranked)
        spot_gap = spot_open_frame.loc[timestamp] / spot_close_frame.shift(1).loc[timestamp] - 1.0
        perp_gap = perp_open_frame.loc[timestamp] / perp_close_frame.shift(1).loc[timestamp] - 1.0
        gap_by_symbol = (spot_gap - perp_gap).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        capital *= 1.0 + float((gap_by_symbol * previous_weights).sum())
        turnover = float((weights - previous_weights).abs().sum())
        cost_return = sum(
            abs(float(weights[symbol] - previous_weights[symbol]))
            * cost_for(symbol)
            for symbol in spot_open_frame.columns
        )
        capital *= 1.0 - cost_return
        intraday = spot_close_frame.loc[timestamp] / spot_open_frame.loc[timestamp] - perp_close_frame.loc[timestamp] / perp_open_frame.loc[timestamp]
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
        pnl = capital - start_capital
        equity_values.append(capital)
        daily_pnl.append(pnl)
        exposures.append(float(weights.abs().sum()))
        turnover_values.append(turnover)
        concurrent_counts.append(int((weights != 0.0).sum()))
        nonzero_weights.extend(float(value) for value in weights[weights != 0.0].abs())
        previous_weights = weights
    if len(spot_open_frame.index) > 0 and float(previous_weights.abs().sum()) > 0.0:
        final_cost = sum(
            float(previous_weights[symbol]) * cost_for(symbol)
            for symbol in spot_open_frame.columns
        )
        capital *= 1.0 - final_cost
        equity_values[-1] = capital
        turnover_values[-1] += float(previous_weights.abs().sum())
        final_timestamp = pd.Timestamp(spot_open_frame.index[-1])
        for symbol, growth in trade_growth.items():
            leg_rate = cost_for(symbol)
            trade_values.append((growth * (1.0 - leg_rate) - 1.0) * trade_weights[symbol])
            trade_times.append(final_timestamp)
    equity = pd.Series(equity_values, index=spot_open_frame.index, dtype=float)
    pnl_series = equity.diff().fillna(equity - 300.0)
    positions = pd.Series(exposures, index=spot_open_frame.index, dtype=float)
    turnover_series = pd.Series(turnover_values, index=spot_open_frame.index, dtype=float)
    trades = pd.Series(trade_values, index=pd.DatetimeIndex(trade_times), dtype=float).sort_index()
    return FundingResult(
        equity,
        pnl_series,
        positions,
        trades,
        max(concurrent_counts, default=0),
        max(nonzero_weights, default=0.0),
        min(nonzero_weights, default=0.0),
        turnover_series,
    )


def run_cached(cache_dir: Path, symbols: tuple[str, ...], candidate: FundingCandidate) -> StrategyResult:
    markets = load_markets(cache_dir, symbols)
    result = run_portfolio(markets, candidate)
    stress = run_portfolio(markets, candidate, stress_multiplier=2.0)
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    stress_is = stress.equity[stress.equity.index <= split]
    oos_stress = stress.equity[stress.equity.index > split]
    anchor = float(stress_is.iloc[-1]) if not stress_is.empty else 300.0
    stress_return = float(oos_stress.iloc[-1] / anchor - 1.0) if not oos_stress.empty else 0.0
    return StrategyResult(
        candidate_id=candidate.candidate_id,
        family="F1",
        equity=result.equity,
        trade_returns=result.trade_returns,
        positions=result.positions,
        turnover=result.turnover,
        stress_total_return=stress_return,
        metadata={
            "symbols": list(markets),
            "exploratory_only": False,
            "cost_model_valid": True,
            "intended_factor": "funding_carry",
            "max_concurrent_positions": result.max_concurrent_positions,
            "max_position_weight": result.max_position_weight,
            "min_position_weight": result.min_position_weight,
            "min_order_usdt": 5.0,
            "candidate_config": {
                "window_days": candidate.window_days,
                "threshold_apr": candidate.threshold_apr,
                "top_k": candidate.top_k,
                "majors_only": candidate.majors_only,
            },
        },
    )
