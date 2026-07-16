from __future__ import annotations

from pathlib import Path
from typing import Callable, Final

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import JsonValue, StrategyResult
from research.wave1.fam_funding import (
    FundingCandidate,
    FundingMarket,
    FundingResult,
    carry_position,
    load_markets,
    neighbor_candidates,
    run_portfolio,
)
from research.wave1.gates import MetricInput, calculate_metrics
from research.wave2.spike import hysteresis_position, spike_position


W2_MAKER_FEE_RATE: Final = 0.0002
W2_FUNDING_CANDIDATES: Final = (
    FundingCandidate("W2a", 7, 0.08, 2, True),
    FundingCandidate("W2b", 7, 0.05, 2, True),
    FundingCandidate("W2c", 7, 0.15, 4),
    FundingCandidate("W2d", 7, 0.025, 2),
    FundingCandidate("W2e", 7, 0.08, 2, True),
    FundingCandidate("W2f", 7, 0.08, 2),
)
W2_FUNDING_IDS: Final = tuple(candidate.candidate_id for candidate in W2_FUNDING_CANDIDATES)


def _spike_builder(funding: pd.Series, score: pd.Series, _candidate: FundingCandidate) -> pd.Series:
    return spike_position(funding, score, entry_rate=0.0005, exit_threshold_apr=0.025)


def _hysteresis_builder(_funding: pd.Series, score: pd.Series, _candidate: FundingCandidate) -> pd.Series:
    return hysteresis_position(score, entry_threshold_apr=0.08, exit_threshold_apr=0.02)


_SPECIAL_POSITION_BUILDERS: Final = {"W2d": _spike_builder, "W2e": _hysteresis_builder}


def _position_builder(funding: pd.Series, score: pd.Series, candidate: FundingCandidate) -> pd.Series:
    builder = _SPECIAL_POSITION_BUILDERS.get(candidate.candidate_id)
    return builder(funding, score, candidate) if builder is not None else carry_position(score, candidate)


def _maker_cost(_symbol: str, _slippage_factor: float) -> float:
    return 2.0 * W2_MAKER_FEE_RATE


def run_maker_portfolio(
    markets: dict[str, FundingMarket],
    candidate: FundingCandidate,
    position_builder: Callable[[pd.Series, pd.Series, FundingCandidate], pd.Series] = _position_builder,
) -> FundingResult:
    return run_portfolio(
        markets,
        candidate,
        position_builder=position_builder,
        cost_model=_maker_cost,
    )


def _neighbor_sharpes(markets: dict[str, FundingMarket], candidate: FundingCandidate) -> list[float]:
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    sharpes: list[float] = []
    for neighbor in neighbor_candidates(candidate):
        result = run_maker_portfolio(markets, neighbor)
        equity = result.equity[result.equity.index <= split]
        trades = tuple(float(value) for value in result.trade_returns[result.trade_returns.index <= split])
        sharpes.append(calculate_metrics(MetricInput(equity, trades)).sharpe)
    return sharpes


def _candidate_config(candidate: FundingCandidate) -> dict[str, JsonValue]:
    return {
        "window_days": candidate.window_days,
        "threshold_apr": candidate.threshold_apr,
        "top_k": candidate.top_k,
        "majors_only": candidate.majors_only,
        "allowed_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"] if candidate.candidate_id == "W2f" else None,
        "entry_funding_rate": 0.0005 if candidate.candidate_id == "W2d" else None,
        "exit_threshold_apr": 0.025 if candidate.candidate_id == "W2d" else (0.02 if candidate.candidate_id == "W2e" else None),
    }


def run_funding_variant(cache_dir: Path, symbols: tuple[str, ...], candidate: FundingCandidate) -> StrategyResult:
    selected_symbols = symbols
    if candidate.majors_only:
        selected_symbols = tuple(symbol for symbol in symbols if symbol in ("BTCUSDT", "ETHUSDT"))
    if candidate.candidate_id == "W2f":
        selected_symbols = tuple(symbol for symbol in symbols if symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"))
    if candidate.candidate_id == "W2d":
        selected_symbols = symbols[:10]
    markets = load_markets(cache_dir, selected_symbols)
    result = run_maker_portfolio(markets, candidate)
    stress = run_portfolio(
        markets,
        candidate,
        stress_multiplier=2.0,
        position_builder=_position_builder,
        cost_model=_maker_cost,
    )
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    stress_is = stress.equity[stress.equity.index <= split]
    stress_oos = stress.equity[stress.equity.index > split]
    anchor = float(stress_is.iloc[-1]) if not stress_is.empty else 300.0
    stress_return = float(stress_oos.iloc[-1] / anchor - 1.0) if not stress_oos.empty else 0.0
    metadata: dict[str, JsonValue] = {
        "symbols": list(markets),
        "exploratory_only": False,
        "cost_model_valid": True,
        "cost_route": "maker_0.02pct_per_leg_zero_slippage",
        "intended_factor": "funding_carry",
        "max_concurrent_positions": result.max_concurrent_positions,
        "max_position_weight": result.max_position_weight,
        "min_position_weight": result.min_position_weight,
        "min_order_usdt": 5.0,
        "candidate_config": _candidate_config(candidate),
        "neighbor_is_sharpes": _neighbor_sharpes(markets, candidate),
        "data_valid": bool(markets),
    }
    return StrategyResult(
        candidate.candidate_id,
        "F1",
        result.equity,
        result.trade_returns,
        result.positions,
        result.turnover,
        stress_return,
        metadata,
    )
