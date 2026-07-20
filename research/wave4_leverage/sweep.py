"""Cache-only leverage sweep over the validated W2c/F1f carry engines.

The strategy rules stay in wave-1/wave-2.  This module only adds capital
structure, conservative liquidation, borrow-interest, and Monte Carlo layers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
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
    stress_basis_moves: pd.DataFrame
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
    baseline_cagr: float | None
    baseline_mdd: float | None
    cagr_relative_error: float | None
    mdd_relative_error: float | None
    reconciliation_pass: bool | None
    stress_liquidation_count: int


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
    spot_high: dict[str, pd.Series] = {}
    spot_low: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    perp_low: dict[str, pd.Series] = {}
    perp_high: dict[str, pd.Series] = {}
    funding_daily: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    active: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        spot, funding, perp = _daily_market_parts(market)
        spot_open[symbol], spot_close[symbol] = spot["open"], spot["close"]
        spot_high[symbol], spot_low[symbol] = spot["high"], spot["low"]
        perp_open[symbol], perp_close[symbol] = perp["open"], perp["close"]
        perp_low[symbol], perp_high[symbol] = perp["low"], perp["high"]
        funding_daily[symbol] = funding
        scores[symbol] = funding_score(market.funding, candidate.window_days).resample("1D").last()
        active[symbol] = _position_series(candidate, market)

    spot_open_frame = pd.DataFrame(spot_open).sort_index()
    spot_close_frame = pd.DataFrame(spot_close).reindex(spot_open_frame.index)
    spot_high_frame = pd.DataFrame(spot_high).reindex(spot_open_frame.index)
    spot_low_frame = pd.DataFrame(spot_low).reindex(spot_open_frame.index)
    perp_open_frame = pd.DataFrame(perp_open).reindex(spot_open_frame.index)
    perp_close_frame = pd.DataFrame(perp_close).reindex(spot_open_frame.index)
    perp_low_frame = pd.DataFrame(perp_low).reindex(spot_open_frame.index)
    perp_high_frame = pd.DataFrame(perp_high).reindex(spot_open_frame.index)
    funding_frame = pd.DataFrame(funding_daily).reindex(spot_open_frame.index).fillna(0.0)
    score_frame = pd.DataFrame(scores).reindex(spot_open_frame.index).shift(1)
    active_frame = pd.DataFrame(active).reindex(spot_open_frame.index).fillna(0.0)

    gap = (spot_open_frame / spot_close_frame.shift(1) - perp_open_frame / perp_close_frame.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    intraday = (spot_close_frame / spot_open_frame - perp_close_frame / perp_open_frame).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    close_basis = (perp_close_frame / spot_close_frame).replace([np.inf, -np.inf], np.nan)
    close_basis_move = (close_basis / close_basis.shift(1) - 1.0).abs()
    spot_range = ((spot_high_frame - spot_low_frame) / spot_open_frame).replace([np.inf, -np.inf], np.nan)
    perp_range = ((perp_high_frame - perp_low_frame) / perp_open_frame).replace([np.inf, -np.inf], np.nan)
    range_spread = (perp_range - spot_range).clip(lower=0.0) * 0.5
    worst_basis = (close_basis_move + range_spread).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stress_basis = worst_basis * 1.5

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
        stress_basis,
        weights_frame,
        {symbol: _pair_cost(candidate.candidate_id, symbol) for symbol in markets},
    ), markets


def notional_multiplier(structure: Structure, leverage: float) -> float:
    if leverage not in VALID_LEVERAGES:
        raise ValueError(f"leverage is not preregistered: {leverage}")
    if structure == "SYM":
        return leverage
    if structure == "ASYM":
        return asym_capital_efficiency(leverage)
    raise ValueError(f"unknown structure: {structure}")


def perp_margin_fraction(structure: Structure, leverage: float) -> float:
    if structure not in STRUCTURES or leverage not in VALID_LEVERAGES:
        raise ValueError("invalid preregistered capital structure")
    return 1.0 / notional_multiplier(structure, leverage)


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
    adverse_move = abs(worst_basis_move)
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


def _relative_error(actual: float, baseline: float) -> float:
    return abs(actual - baseline) / max(abs(baseline), 1e-12)


def simulate(
    trace: PortfolioTrace,
    candidate_id: str,
    structure: Structure,
    leverage: float,
    engine_equity_final: float,
    cache_symbols: tuple[str, ...],
    seed: int,
    engine_equity: pd.Series | None = None,
    engine_trade_returns: pd.Series | None = None,
    basis_moves: pd.DataFrame | None = None,
    include_mc: bool = True,
) -> SimulationResult:
    multiplier = notional_multiplier(structure, leverage)
    margin_fraction = perp_margin_fraction(structure, leverage)
    borrow_fraction = spot_borrow_fraction(structure, leverage)
    capital = INITIAL_CAPITAL
    previous_weights = pd.Series(0.0, index=trace.weights.columns, dtype=float)
    equity_values: list[float] = []
    liquidations = 0
    borrowing_total = 0.0
    selected_basis_moves = trace.worst_basis_moves if basis_moves is None else basis_moves
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
            worst_move = float(selected_basis_moves.loc[timestamp, symbol])
            notional = start_capital * float(weight) * multiplier
            initial_margin = start_capital * float(weight) * margin_fraction
            if notional <= 0.0:
                continue
            loss = liquidation_loss(notional, worst_move, initial_margin)
            if loss is not None:
                effective_weights.loc[symbol] = 0.0
                liquidation_dollars += loss
                liquidations += 1

        gap_return = float((trace.gap_returns.loc[timestamp] * previous_weights).sum()) * multiplier
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
    mc_returns = daily_returns if engine_trade_returns is None else engine_trade_returns * multiplier
    p05, bankruptcy = _mc(mc_returns, seed) if include_mc else (0.0, 0.0)
    baseline_cagr: float | None = None
    baseline_mdd: float | None = None
    cagr_relative_error: float | None = None
    mdd_relative_error: float | None = None
    reconciliation_pass: bool | None = None
    if leverage == 1.0 and engine_equity is not None:
        baseline_days = max(1, (engine_equity.index[-1] - engine_equity.index[0]).days)
        baseline_cagr = float((engine_equity.iloc[-1] / INITIAL_CAPITAL) ** (DAYS_PER_YEAR / baseline_days) - 1.0)
        baseline_mdd = _mdd(engine_equity)
        cagr_relative_error = _relative_error(cagr, baseline_cagr)
        mdd_relative_error = _relative_error(_mdd(equity), baseline_mdd)
        reconciliation_pass = cagr_relative_error <= 0.01 and mdd_relative_error <= 0.01
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
        baseline_cagr,
        baseline_mdd,
        cagr_relative_error,
        mdd_relative_error,
        reconciliation_pass,
        0,
    )


def load_candidate_symbols(cache_dir: Path) -> tuple[str, ...]:
    payload = json.loads((cache_dir / "universe.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("invalid cached universe.json")
    symbols: list[str] = []
    for symbol in payload["symbols"]:
        if not isinstance(symbol, str) or re.fullmatch(r"[A-Z0-9]+", symbol) is None:
            raise ValueError(f"invalid cached symbol: {symbol!r}")
        symbols.append(symbol)
    return tuple(symbols)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(root: Path, cache_dir: Path, symbols: tuple[str, ...]) -> dict[str, object]:
    cache_root = cache_dir.resolve()
    relative_files = [Path("research/wave1/cache/universe.json")]
    for symbol in symbols:
        relative_files.extend(
            Path(f"research/wave1/cache/{prefix}{symbol}{suffix}")
            for prefix, suffix in (
                ("binance_spot_", "_1d.csv.gz"),
                ("binance_fapi_", "_1d.csv.gz"),
                ("binance_funding_", ".csv.gz"),
            )
        )
    input_sha256: dict[str, str] = {}
    for relative_path in relative_files:
        path = root / relative_path
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(cache_root):
            raise ValueError(f"cached input escapes cache directory: {relative_path}")
        input_sha256[relative_path.as_posix()] = _sha256(resolved)
    manifest = "\n".join(f"{key}\t{input_sha256[key]}" for key in sorted(input_sha256))
    source_files = {
        "research/wave4_leverage/sweep.py": _sha256(Path(__file__).resolve()),
        "research/wave4_leverage/run_wave4.py": _sha256(Path(__file__).with_name("run_wave4.py").resolve()),
    }
    return {
        "cache_manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        "input_sha256": input_sha256,
        "source_sha256": source_files,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def run_sweep(root: Path) -> tuple[SimulationResult, ...]:
    cache_dir = root / "research" / "wave1" / "cache"
    output_dir = root / "research" / "wave4_leverage" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = load_candidate_symbols(cache_dir)
    provenance = _source_provenance(root, cache_dir, symbols)
    results: list[SimulationResult] = []
    payloads: list[tuple[Path, dict[str, object]]] = []
    for candidate_index, candidate_id in enumerate(("W2c", "F1f")):
        candidate = _candidate(candidate_id)
        trace, markets = build_trace(cache_dir, symbols, candidate)
        engine_result = run_maker_portfolio(markets, candidate) if candidate_id == "W2c" else run_portfolio(markets, candidate)
        engine_path_match = bool(np.allclose(replay_engine_equity(trace).to_numpy(), engine_result.equity.to_numpy(), rtol=1e-10, atol=1e-8))
        if not engine_path_match:
            raise RuntimeError(f"trace does not reproduce imported engine equity path: {candidate_id}")
        for structure_index, structure in enumerate(STRUCTURES):
            for leverage in VALID_LEVERAGES:
                seed = 20_260_715 if leverage == 1.0 else 20_260_716 + candidate_index * 1_000 + structure_index * 100 + int(leverage * 10)
                result = simulate(
                    trace,
                    candidate_id,
                    structure,
                    leverage,
                    float(engine_result.equity.iloc[-1]),
                    tuple(markets),
                    seed,
                    engine_equity=engine_result.equity,
                    engine_trade_returns=engine_result.trade_returns,
                )
                stress_result = simulate(
                    trace,
                    candidate_id,
                    structure,
                    leverage,
                    float(engine_result.equity.iloc[-1]),
                    tuple(markets),
                    seed,
                    engine_equity=engine_result.equity,
                    engine_trade_returns=engine_result.trade_returns,
                    basis_moves=trace.stress_basis_moves,
                    include_mc=False,
                )
                result = replace(result, stress_liquidation_count=stress_result.liquidation_count)
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
                        "stress_liquidation_count": result.stress_liquidation_count,
                        "borrowing_cost_total": result.borrowing_cost_total,
                    },
                    "reconciliation": {
                        "baseline_cagr": result.baseline_cagr,
                        "baseline_mdd": result.baseline_mdd,
                        "cagr_relative_error": result.cagr_relative_error,
                        "mdd_relative_error": result.mdd_relative_error,
                        "status": "PASS" if result.reconciliation_pass is True else ("FAIL" if result.reconciliation_pass is False else "N/A"),
                    },
                    "model": {
                        "borrow_apr": BORROW_APR,
                        "maintenance_rate": MAINTENANCE_RATE,
                        "liquidation_fee_rate": LIQUIDATION_FEE_RATE,
                        "asym_capital_efficiency": asym_capital_efficiency(leverage) if structure == "ASYM" else None,
                        "notional_multiplier": notional_multiplier(structure, leverage),
                        "worst_basis_definition": "abs(close_basis_change) + max(0, perp_range_pct - spot_range_pct) * 0.5",
                        "stress_worst_basis_definition": "baseline_worst_basis * 1.5",
                        "liquidation_loss": "notional*abs(worst_basis_move)+notional*0.0006",
                    },
                    "source": {
                        "cache_only": True,
                        "cache_dir": "research/wave1/cache",
                        "engine_result_final": result.engine_equity_final,
                        "engine_equity_path_match": engine_path_match,
                        "symbols": list(result.cache_symbols),
                        "rng_seed": seed,
                        **provenance,
                    },
                }
                payloads.append(
                    (
                        output_dir / f"{candidate_id}_{structure}_L{str(leverage).replace('.', 'p')}.json",
                        payload,
                    )
                )
    reconciliation_results = [result for result in results if result.leverage == 1.0]
    if len(reconciliation_results) != 4 or any(result.reconciliation_pass is not True for result in reconciliation_results):
        details = ", ".join(
            f"{result.candidate_id}/{result.structure}: CAGR={result.cagr_relative_error!r}, MDD={result.mdd_relative_error!r}"
            for result in reconciliation_results
        )
        report_path = root / "research" / "wave4_leverage" / "LEVERAGE_REPORT.md"
        report_path.unlink(missing_ok=True)
        for path in output_dir.glob("*.json"):
            path.unlink()
        raise RuntimeError(f"baseline reconciliation gate failed; report publication blocked ({details})")
    staging_dir = Path(tempfile.mkdtemp(prefix=".wave4-staging-", dir=output_dir))
    try:
        for final_path, payload in payloads:
            staged_path = staging_dir / final_path.name
            staged_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for final_path, _ in payloads:
            os.replace(staging_dir / final_path.name, final_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return tuple(results)
