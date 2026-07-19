"""Cache-only leverage sweep over the validated W2c/F1f carry engines.

The strategy rules stay in wave-1/wave-2.  This module only adds capital
structure, conservative liquidation, borrow-interest, and Monte Carlo layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.costs import PERP_TAKER_RATE, SPOT_TAKER_RATE, slippage_rate
from research.wave1.fam_funding import (
    F1_CANDIDATES,
    FundingCandidate,
    FundingMarket,
    carry_position,
    funding_score,
    load_markets,
    run_portfolio,
)
from research.wave2.funding import W2_FUNDING_CANDIDATES, _maker_cost, run_maker_portfolio


Structure = Literal["SYM", "ASYM"]
VALID_LEVERAGES: Final[tuple[float, ...]] = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
STRUCTURES: Final[tuple[Structure, ...]] = ("SYM", "ASYM")
INITIAL_CAPITAL: Final = 300.0
BORROW_APR: Final = 0.10
MAINTENANCE_RATE: Final = 0.005
LIQUIDATION_FEE_RATE: Final = 0.0006
MC_PATHS: Final = 10_000
MC_BANKRUPTCY_THRESHOLD: Final = INITIAL_CAPITAL / 2.0
DAYS_PER_YEAR: Final = 365.0


@dataclass(frozen=True, slots=True)
class PortfolioTrace:
    index: pd.DatetimeIndex
    gap_returns: pd.DataFrame
    intraday_returns: pd.DataFrame
    worst_basis_moves: pd.DataFrame
    weights: pd.DataFrame
    pair_costs: dict[str, float]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    candidate_id: str
    structure: Structure
    leverage: float
    equity: pd.Series
    daily_returns: pd.Series
    cagr: float
    mdd: float
    mc_p05: float
    bankruptcy_probability: float
    liquidation_count: int
    borrowing_cost_total: float
    engine_equity_final: float
    cache_symbols: tuple[str, ...]


def _candidate(candidate_id: str) -> FundingCandidate:
    candidates = (*F1_CANDIDATES, *W2_FUNDING_CANDIDATES)
    return next(candidate for candidate in candidates if candidate.candidate_id == candidate_id)


def _pair_cost(candidate_id: str, symbol: str) -> float:
    if candidate_id == "W2c":
        return _maker_cost(symbol, 1.0)
    return SPOT_TAKER_RATE + PERP_TAKER_RATE + 2.0 * slippage_rate(symbol)


def _position_series(candidate: FundingCandidate, market: FundingMarket) -> pd.Series:
    # W2c and F1f are both the default carry-position rule.  The candidate
    # objects and the wave-1 helpers are imported instead of redefining them.
    daily_score = funding_score(market.funding, candidate.window_days).resample("1D").last()
    return carry_position(daily_score, candidate)


def _daily_market_parts(market: FundingMarket) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    spot = market.spot.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    perp = market.perp.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    funding = market.funding.resample("1D").sum()
    return spot, funding, perp


def build_trace(cache_dir: Path, symbols: tuple[str, ...], candidate: FundingCandidate) -> tuple[PortfolioTrace, dict[str, FundingMarket]]:
    markets = load_markets(cache_dir, symbols)
    if not markets:
        raise ValueError("no cached markets available for leverage sweep")
    spot_open: dict[str, pd.Series] = {}
    spot_close: dict[str, pd.Series] = {}
    spot_low: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    perp_high: dict[str, pd.Series] = {}
    funding_daily: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        spot, funding, perp = _daily_market_parts(market)
        spot_open[symbol], spot_close[symbol], spot_low[symbol] = spot["open"], spot["close"], spot["low"]
        perp_open[symbol], perp_close[symbol], perp_high[symbol] = perp["open"], perp["close"], perp["high"]
        funding_daily[symbol] = funding
        scores[symbol] = funding_score(market.funding, candidate.window_days).resample("1D").last()
        active[symbol] = _position_series(candidate, market)

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    spot_close_frame = pd.DataFrame(spot_close).reindex(spot_open_frame.index)
    spot_low_frame = pd.DataFrame(spot_low).reindex(spot_open_frame.index)
    perp_open_frame = pd.DataFrame(perp_open).reindex(spot_open_frame.index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(spot_open_frame.index)
    perp_high_frame = pd.DataFrame(perp_high).reindex(spot_open_frame.index)
    funding_frame = pd.DataFrame(funding_daily).reindex(spot_open_frame.index).fillna(0.0)
    score_frame = pd.DataFrame(scores).reindex(spot_open_frame.index).shift(1)
    active_frame = pd.DataFrame(active).reindex(spot_open_frame.index).fillna(0.0)

    gap = (spot_open_frame / spot_close_frame.shift(1) - perp_open_frame / perp_close_frame.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    intraday = (spot_close_frame / spot_open_frame - perp_close_frame / perp_open_frame).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # Conservative, intentionally asynchronous bar worst case: spot low and
    # perp high are combined even if they did not occur at the same instant.
    worst_basis = (spot_low_frame / spot_open_frame - perp_high_frame / perp_open_frame).replace([np.inf, -np.inf], np.nan)

    weights_rows: list[pd.Series] = []
    for timestamp in spot_open_frame.index:
        available = spot_open_frame.loc[timestamp].notna() & spot_close_frame.loc[timestamp].notna() & perp_open_frame.loc[timestamp].notna() & perp_close_frame.loc[timestamp].notna()
        eligible = active_frame.loc[timestamp][active_frame.loc[timestamp] > 0.0].index
        eligible = eligible.intersection(available[available].index)
        if candidate.majors_only:
            eligible = eligible.intersection(pd.Index(["BTCUSDT", "ETHUSDT"]))
        ranked = score_frame.loc[timestamp, eligible].dropna().nlargest(candidate.top_k).index
        weights = pd.Series(0.0, index=spot_open_frame.columns, dtype=float)
        if len(ranked) > 0:
            weights.loc[ranked] = 1.0 / len(ranked)
        weights_rows.append(weights)
    weights_frame = pd.DataFrame(weights_rows, index=spot_open_frame.index).fillna(0.0)
    return PortfolioTrace(
        spot_open_frame.index,
        gap,
        intraday + funding_frame,
        worst_basis,
        weights_frame,
        {symbol: _pair_cost(candidate.candidate_id, symbol) for symbol in markets},
    ), markets


def notional_multiplier(structure: Structure, leverage: float) -> float:
    if leverage not in VALID_LEVERAGES:
        raise ValueError(f"leverage is not preregistered: {leverage}")
    if structure == "SYM":
        return leverage / 2.0
    if structure == "ASYM":
        return leverage / (leverage + 1.0)
    raise ValueError(f"unknown structure: {structure}")


def perp_margin_fraction(structure: Structure, leverage: float) -> float:
    if structure not in STRUCTURES or leverage not in VALID_LEVERAGES:
        raise ValueError("invalid preregistered capital structure")
    return 0.5 if structure == "SYM" else 1.0 / (leverage + 1.0)


def spot_borrow_fraction(structure: Structure, leverage: float) -> float:
    if structure == "SYM":
        return max(0.0, leverage / 2.0 - 0.5)
    if structure == "ASYM":
        return 0.0
    raise ValueError(f"unknown structure: {structure}")


def asym_capital_efficiency(leverage: float) -> float:
    if leverage not in VALID_LEVERAGES:
        raise ValueError(f"leverage is not preregistered: {leverage}")
    return 2.0 / (1.0 + 1.0 / leverage)


def liquidation_threshold(notional: float, initial_margin: float) -> float:
    return initial_margin - MAINTENANCE_RATE * notional


def liquidation_loss(notional: float, worst_basis_move: float, initial_margin: float) -> float | None:
    adverse_move = max(0.0, -worst_basis_move)
    loss_before_fee = notional * adverse_move
    if loss_before_fee < liquidation_threshold(notional, initial_margin):
        return None
    return loss_before_fee + notional * LIQUIDATION_FEE_RATE


def _mdd(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float((1.0 - equity / running_max.replace(0.0, np.nan)).max()) if not equity.empty else 0.0


def _mc(daily_returns: pd.Series, seed: int) -> tuple[float, float]:
    values = np.clip(daily_returns.to_numpy(dtype=float), -0.999999, None)
    if values.size == 0:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    finals = np.empty(MC_PATHS, dtype=float)
    for start in range(0, MC_PATHS, 500):
        stop = min(start + 500, MC_PATHS)
        samples = rng.choice(values, size=(stop - start, values.size), replace=True)
        finals[start:stop] = INITIAL_CAPITAL * np.prod(1.0 + samples, axis=1)
    return float(np.quantile(finals, 0.05)), float(np.mean(finals < MC_BANKRUPTCY_THRESHOLD))


def replay_engine_equity(trace: PortfolioTrace) -> pd.Series:
    capital = INITIAL_CAPITAL
    previous_weights = pd.Series(0.0, index=trace.weights.columns, dtype=float)
    values: list[float] = []
    for timestamp in trace.index:
        weights = trace.weights.loc[timestamp]
        capital *= 1.0 + float((trace.gap_returns.loc[timestamp] * previous_weights).sum())
        cost_return = sum(
            abs(float(weights[symbol] - previous_weights[symbol])) * trace.pair_costs[symbol]
            for symbol in trace.weights.columns
        )
        capital *= 1.0 - cost_return
        capital *= 1.0 + float((trace.intraday_returns.loc[timestamp] * weights).sum())
        values.append(capital)
        previous_weights = weights
    if values and float(previous_weights.abs().sum()) > 0.0:
        final_cost = sum(float(previous_weights[symbol]) * trace.pair_costs[symbol] for symbol in trace.weights.columns)
        values[-1] *= 1.0 - final_cost
    return pd.Series(values, index=trace.index, dtype=float)


def replay_engine_equity(trace: PortfolioTrace) -> pd.Series:
    """Replay the imported wave-1/wave-2 portfolio accounting from the trace."""
    capital = INITIAL_CAPITAL
    previous_weights = pd.Series(0.0, index=trace.weights.columns, dtype=float)
    values: list[float] = []
    for timestamp in trace.index:
        weights = trace.weights.loc[timestamp]
        capital *= 1.0 + float((trace.gap_returns.loc[timestamp] * previous_weights).sum())
        cost_return = sum(
            abs(float(weights[symbol] - previous_weights[symbol])) * trace.pair_costs[symbol]
            for symbol in trace.weights.columns
        )
        capital *= 1.0 - cost_return
        capital *= 1.0 + float((trace.intraday_returns.loc[timestamp] * weights).sum())
        values.append(capital)
        previous_weights = weights
    if values and float(previous_weights.abs().sum()) > 0.0:
        final_cost = sum(float(previous_weights[symbol]) * trace.pair_costs[symbol] for symbol in trace.weights.columns)
        values[-1] *= 1.0 - final_cost
    return pd.Series(values, index=trace.index, dtype=float)


def simulate(
    trace: PortfolioTrace,
    candidate_id: str,
    structure: Structure,
    leverage: float,
    engine_equity_final: float,
    cache_symbols: tuple[str, ...],
    seed: int,
) -> SimulationResult:
    multiplier = notional_multiplier(structure, leverage)
    margin_fraction = perp_margin_fraction(structure, leverage)
    borrow_fraction = spot_borrow_fraction(structure, leverage)
    capital = INITIAL_CAPITAL
    previous_weights = pd.Series(0.0, index=trace.weights.columns, dtype=float)
    equity_values: list[float] = []
    liquidations = 0
    borrowing_total = 0.0
    for timestamp in trace.index:
        start_capital = capital
        if start_capital <= 0.0:
            equity_values.append(0.0)
            previous_weights = pd.Series(0.0, index=trace.weights.columns, dtype=float)
            continue
        target_weights = trace.weights.loc[timestamp]
        effective_weights = target_weights.copy()
        liquidation_dollars = 0.0
        for symbol, weight in target_weights[target_weights > 0.0].items():
            worst_move = float(trace.worst_basis_moves.loc[timestamp, symbol])
            notional = start_capital * float(weight) * multiplier
            initial_margin = start_capital * float(weight) * margin_fraction
            if notional <= 0.0:
                continue
            loss = liquidation_loss(notional, worst_move, initial_margin)
            if loss is not None:
                effective_weights.loc[symbol] = 0.0
                liquidation_dollars += loss
                liquidations += 1

        gap_return = float((trace.gap_returns.loc[timestamp] * previous_weights).sum())
        capital = max(0.0, capital * (1.0 + gap_return))
        cost_return = sum(
            abs(float(effective_weights[symbol] - previous_weights[symbol]))
            * trace.pair_costs[symbol]
            * multiplier
            for symbol in trace.weights.columns
        )
        capital = max(0.0, capital * (1.0 - cost_return))
        intraday_return = float((trace.intraday_returns.loc[timestamp] * effective_weights).sum()) * multiplier
        capital = max(0.0, capital * (1.0 + intraday_return))
        borrow_cost = start_capital * float(effective_weights.sum()) * borrow_fraction * BORROW_APR / DAYS_PER_YEAR
        borrowing_total += borrow_cost
        capital = max(0.0, capital - borrow_cost - liquidation_dollars)
        equity_values.append(capital)
        previous_weights = effective_weights

    equity = pd.Series(equity_values, index=trace.index, dtype=float)
    daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    days = max(1, (trace.index[-1] - trace.index[0]).days) if len(trace.index) else 1
    final = float(equity.iloc[-1]) if not equity.empty else 0.0
    cagr = float((final / INITIAL_CAPITAL) ** (DAYS_PER_YEAR / days) - 1.0) if final > 0.0 else -1.0
    p05, bankruptcy = _mc(daily_returns, seed)
    return SimulationResult(
        candidate_id,
        structure,
        leverage,
        equity,
        daily_returns,
        cagr,
        _mdd(equity),
        p05,
        bankruptcy,
        liquidations,
        borrowing_total,
        engine_equity_final,
        cache_symbols,
    )


def load_candidate_symbols(cache_dir: Path) -> tuple[str, ...]:
    payload = json.loads((cache_dir / "universe.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("invalid cached universe.json")
    return tuple(str(symbol) for symbol in payload["symbols"])


def run_sweep(root: Path) -> tuple[SimulationResult, ...]:
    cache_dir = root / "research" / "wave1" / "cache"
    output_dir = root / "research" / "wave4_leverage" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = load_candidate_symbols(cache_dir)
    results: list[SimulationResult] = []
    for candidate_index, candidate_id in enumerate(("W2c", "F1f")):
        candidate = _candidate(candidate_id)
        trace, markets = build_trace(cache_dir, symbols, candidate)
        engine_result = run_maker_portfolio(markets, candidate) if candidate_id == "W2c" else run_portfolio(markets, candidate)
        engine_path_match = bool(np.allclose(replay_engine_equity(trace).to_numpy(), engine_result.equity.to_numpy(), rtol=1e-10, atol=1e-8))
        if not engine_path_match:
            raise RuntimeError(f"trace does not reproduce imported engine equity path: {candidate_id}")
        for structure_index, structure in enumerate(STRUCTURES):
            for leverage in VALID_LEVERAGES:
                seed = 20_260_716 + candidate_index * 1_000 + structure_index * 100 + int(leverage * 10)
                result = simulate(trace, candidate_id, structure, leverage, float(engine_result.equity.iloc[-1]), tuple(markets), seed)
                results.append(result)
                payload = {
                    "candidate_id": candidate_id,
                    "structure": structure,
                    "valid_leverage": leverage,
                    "grid_preregistered": True,
                    "initial_capital": INITIAL_CAPITAL,
                    "metrics": {
                        "cagr": result.cagr,
                        "mdd": result.mdd,
                        "mc_paths": MC_PATHS,
                        "mc_p05": result.mc_p05,
                        "bankruptcy_probability_final_below_150": result.bankruptcy_probability,
                        "liquidation_count": result.liquidation_count,
                        "borrowing_cost_total": result.borrowing_cost_total,
                    },
                    "model": {
                        "borrow_apr": BORROW_APR,
                        "maintenance_rate": MAINTENANCE_RATE,
                        "liquidation_fee_rate": LIQUIDATION_FEE_RATE,
                        "asym_capital_efficiency": asym_capital_efficiency(leverage) if structure == "ASYM" else None,
                        "notional_multiplier": notional_multiplier(structure, leverage),
                        "worst_basis_definition": "spot_low/spot_open - perp_high/perp_open",
                        "liquidation_loss": "notional*max(0,-worst_basis_move)+notional*0.0006",
                    },
                    "source": {
                        "cache_only": True,
                        "cache_dir": "research/wave1/cache",
                        "engine_result_final": result.engine_equity_final,
                        "engine_equity_path_match": engine_path_match,
                        "symbols": list(result.cache_symbols),
                    },
                }
                (output_dir / f"{candidate_id}_{structure}_L{str(leverage).replace('.', 'p')}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
    return tuple(results)
