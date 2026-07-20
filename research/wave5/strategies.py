from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import JsonValue, PipelineError, StrategyResult, load_frame, validate_symbol
from research.wave1.costs import PERP_TAKER_RATE, slippage_rate
from research.wave1.fam_funding import FundingCandidate, load_markets
from research.wave1.fam_tsmom import vol_target_fraction
from research.wave2.funding import run_maker_portfolio
from research.wave3.engine import AssetMarket, CandidateConfig, run_candidate
from research.wave3.universe import AssetListing
from research.wave5.engine import (
    basis_round_trip_cost,
    cached_frame,
    equity_from_returns,
    funding_capitulation_position,
    pair_round_trip_cost,
    rolling_zscore,
    rsi,
    zscore_hysteresis_position,
)


OOS_SPLIT = pd.Timestamp("2025-09-30T23:59:59Z")
W2C = FundingCandidate("W2c", 7, 0.15, 4)


def _records_metadata(symbols: tuple[str, ...], factor: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "symbols": list(symbols),
        "exploratory_only": False,
        "data_valid": bool(symbols),
        "cost_model_valid": True,
        "intended_factor": factor,
        "max_concurrent_positions": 2,
        "max_position_weight": 0.5,
        "min_position_weight": 0.5,
        "min_order_usdt": 5.0,
        "neighbor_is_sharpes": [],
        **extra,
    }


def _result(
    candidate_id: str,
    family: str,
    returns: pd.Series,
    positions: pd.Series,
    turnover: pd.Series,
    metadata: dict[str, JsonValue],
) -> StrategyResult:
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = equity_from_returns(clean_returns)
    trades = clean_returns[positions.abs().reindex(clean_returns.index).fillna(0.0) > 0.0]
    stress_returns = clean_returns.copy()
    stress_cost = turnover.reindex(clean_returns.index).fillna(0.0) * 0.0002
    stress_returns = stress_returns - stress_cost
    stress_equity = equity_from_returns(stress_returns)
    stress_is_equity = stress_equity[stress_equity.index <= OOS_SPLIT]
    oos_equity = stress_equity[stress_equity.index > OOS_SPLIT]
    anchor = float(stress_is_equity.iloc[-1]) if not stress_is_equity.empty else 300.0
    stress_oos = float(oos_equity.iloc[-1] / anchor - 1.0) if not oos_equity.empty else 0.0
    return StrategyResult(candidate_id, family, equity, trades, positions, turnover, stress_oos, metadata)


def run_w2c(cache_dir: Path, symbols: tuple[str, ...]) -> StrategyResult:
    markets = load_markets(cache_dir, symbols)
    result = run_maker_portfolio(markets, W2C)
    metadata = _records_metadata(
        tuple(sorted(markets)),
        "funding_carry",
        max_concurrent_positions=result.max_concurrent_positions,
        max_position_weight=result.max_position_weight,
        min_position_weight=result.min_position_weight,
        cost_route="wave2_maker_0.02pct_per_leg_zero_slippage",
        candidate_config={"window_days": 7, "threshold_apr": 0.15, "top_k": 4},
    )
    return StrategyResult("W2c", "F1", result.equity, result.trade_returns, result.positions, result.turnover, 0.0, metadata)


def _pair_returns(cache_dir: Path, entry_z: float, exit_z: float, candidate_id: str) -> StrategyResult:
    eth = cached_frame(cache_dir, "binance_fapi_", "ETHUSDT", "_1d.csv.gz")["close"]
    btc = cached_frame(cache_dir, "binance_fapi_", "BTCUSDT", "_1d.csv.gz")["close"]
    prices = pd.concat([eth.rename("eth"), btc.rename("btc")], axis=1).dropna()
    ratio = np.log(prices["eth"] / prices["btc"])
    zscore = rolling_zscore(ratio, 20)
    raw = zscore_hysteresis_position(zscore, entry_z, exit_z)
    realized = (prices["eth"].pct_change() - prices["btc"].pct_change()).rolling(20, min_periods=20).std(ddof=1)
    scale = vol_target_fraction(realized, target_vol=0.015, leverage_cap=2.0).shift(1).fillna(0.0)
    position = raw * scale
    gross = position.shift(1).fillna(0.0) * (prices["eth"].pct_change() - prices["btc"].pct_change())
    turnover = position.diff().abs().fillna(position.abs())
    event_cost = pair_round_trip_cost(1.0, slippage_rate("BTCUSDT"))
    returns = gross - turnover * event_cost
    return _result(
        candidate_id,
        "F5",
        returns,
        position,
        turnover,
        _records_metadata(("ETHUSDT", "BTCUSDT"), "eth_btc_mean_reversion", entry_z=entry_z, exit_z=exit_z, leverage_cap=2.0),
    )


def run_w5a(cache_dir: Path) -> StrategyResult:
    return _pair_returns(cache_dir, 2.0, 0.5, "W5a")


def run_w5b(cache_dir: Path) -> StrategyResult:
    return _pair_returns(cache_dir, 2.5, 0.25, "W5b")


def run_w5c(cache_dir: Path) -> StrategyResult:
    pieces: list[pd.DataFrame] = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        perp = cached_frame(cache_dir, "binance_fapi_", symbol, "_1d.csv.gz")["close"].rename("perp")
        spot = cached_frame(cache_dir, "binance_spot_", symbol, "_1d.csv.gz")["close"].rename("spot")
        frame = pd.concat([perp, spot], axis=1).dropna()
        frame["basis"] = frame["perp"] / frame["spot"] - 1.0
        frame["z"] = rolling_zscore(frame["basis"], 30)
        frame["position"] = zscore_hysteresis_position(frame["z"], 2.0, 0.5)
        frame["spread_return"] = frame["perp"].pct_change() - frame["spot"].pct_change()
        pieces.append(frame)
    combined = pd.concat(pieces, axis=1, keys=("BTCUSDT", "ETHUSDT")).dropna()
    pair_positions = combined.xs("position", axis=1, level=1)
    pair_spreads = combined.xs("spread_return", axis=1, level=1)
    position = pair_positions.mean(axis=1)
    turnover = pair_positions.diff().abs().fillna(pair_positions.abs()).mean(axis=1)
    event_cost = basis_round_trip_cost(1.0, slippage_rate("BTCUSDT"))
    returns = (pair_positions.shift(1).fillna(0.0) * pair_spreads).mean(axis=1) - turnover * event_cost
    return _result(
        "W5c",
        "F5",
        returns,
        position,
        turnover,
        _records_metadata(("BTCUSDT", "ETHUSDT"), "spot_perp_basis_mean_reversion", basis_window=30, entry_z=2.0, exit_z=0.5, four_leg_round_trip=True),
    )


def _range_position(bars: pd.DataFrame) -> pd.Series:
    close = bars["close"]
    returns = close.pct_change()
    volatility = returns.rolling(20, min_periods=20).std(ddof=1)
    percentile = volatility.shift(1).expanding(min_periods=30).quantile(0.30)
    average = close.rolling(50, min_periods=50).mean()
    previous_close = close.shift(1)
    ranges = pd.concat([bars["high"] - bars["low"], (bars["high"] - previous_close).abs(), (bars["low"] - previous_close).abs()], axis=1).max(axis=1)
    atr = ranges.rolling(20, min_periods=20).mean()
    rsi_two = rsi(close, 2)
    signal: list[float] = []
    active = 0.0
    entry = float("nan")
    for day in close.index:
        price = float(close.loc[day])
        stop = bool(pd.notna(entry) and pd.notna(atr.loc[day]) and abs(price - entry) >= 2.0 * float(atr.loc[day]))
        eligible = pd.notna(percentile.loc[day]) and volatility.loc[day] < percentile.loc[day] and pd.notna(atr.loc[day]) and abs(price - average.loc[day]) / float(atr.loc[day]) < 1.0
        if active != 0.0 and (stop or 45.0 <= rsi_two.loc[day] <= 55.0):
            active = 0.0
            entry = float("nan")
        elif active == 0.0 and eligible and rsi_two.loc[day] < 10.0:
            active, entry = 1.0, price
        elif active == 0.0 and eligible and rsi_two.loc[day] > 90.0:
            active, entry = -1.0, price
        signal.append(active)
    return pd.Series(signal, index=close.index, dtype=float)


def run_w5d(cache_dir: Path) -> StrategyResult:
    returns: list[pd.Series] = []
    positions: list[pd.Series] = []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        bars = cached_frame(cache_dir, "binance_fapi_", symbol, "_1d.csv.gz")
        position = _range_position(bars)
        returns.append(position.shift(1).fillna(0.0) * bars["close"].pct_change())
        positions.append(position)
    frame = pd.concat(returns, axis=1).mean(axis=1).fillna(0.0)
    exposure = pd.concat(positions, axis=1).mean(axis=1).fillna(0.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    returns_frame = frame - turnover * pair_round_trip_cost(1.0, slippage_rate("BTCUSDT")) / 2.0
    return _result("W5d", "F5", returns_frame, exposure, turnover, _records_metadata(("BTCUSDT", "ETHUSDT", "SOLUSDT"), "low_volatility_mean_reversion", rsi_window=2, stop_atr=2.0))


def run_w5e(cache_dir: Path, symbols: tuple[str, ...]) -> StrategyResult:
    candidates: list[tuple[str, pd.DataFrame]] = []
    for symbol in symbols:
        perp_path = cache_dir / f"binance_fapi_{symbol}_1d.csv.gz"
        funding_path = cache_dir / f"binance_funding_{symbol}.csv.gz"
        if not perp_path.exists() or not funding_path.exists():
            continue
        perp = load_frame(perp_path)
        funding = load_frame(funding_path)["funding_rate"]
        candidates.append((symbol, perp.assign(daily_funding=funding.resample("1D").sum().reindex(perp.index).fillna(0.0))))
    positions: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}
    for symbol, bars in candidates:
        funding = load_frame(cache_dir / f"binance_funding_{symbol}.csv.gz")["funding_rate"]
        active_8h = funding_capitulation_position(funding, -0.0005, 6)
        active_daily = active_8h.resample("1D").max().reindex(bars.index).fillna(0.0)
        positions[symbol] = active_daily
        returns[symbol] = active_daily.shift(1).fillna(0.0) * (bars["close"].pct_change() - bars["daily_funding"])
    position_frame = pd.DataFrame(positions).sort_index()
    return_frame = pd.DataFrame(returns).reindex(position_frame.index).fillna(0.0)
    volume_rank = pd.DataFrame(
        {
            symbol: bars["quote_volume"].rolling(20, min_periods=20).mean()
            for symbol, bars in candidates
        }
    ).reindex(position_frame.index)
    selected_frame = select_top_active_positions(position_frame, volume_rank, max_universe=20, max_positions=2)
    portfolio_returns = (return_frame * selected_frame.shift(1).fillna(0.0)).sum(axis=1)
    turnover = selected_frame.diff().abs().sum(axis=1).fillna(selected_frame.abs().sum(axis=1))
    portfolio_returns -= turnover * pair_round_trip_cost(1.0, slippage_rate("BTCUSDT")) / 2.0
    return _result("W5e", "F5", portfolio_returns, selected_frame.sum(axis=1), turnover, _records_metadata(tuple(symbol for symbol, _ in candidates), "funding_capitulation", threshold=-0.0005, hold_hours=48, max_universe=20, max_concurrent_positions=2))


def run_w5f(cache_dir: Path, symbols: tuple[str, ...]) -> StrategyResult:
    markets: dict[str, AssetMarket] = {}
    for symbol in symbols:
        perp_path = cache_dir / f"binance_fapi_{symbol}_1d.csv.gz"
        funding_path = cache_dir / f"binance_funding_{symbol}.csv.gz"
        if not perp_path.exists() or not funding_path.exists():
            continue
        perp = cached_frame(cache_dir, "binance_fapi_", symbol, "_1d.csv.gz")
        funding = cached_frame(cache_dir, "binance_funding_", symbol, ".csv.gz")["funding_rate"]
        listing = AssetListing(symbol, "crypto", pd.Timestamp(perp.index.min()), False, True)
        markets[symbol] = AssetMarket(listing, perp, None, funding)
    result = run_candidate(markets, CandidateConfig("W5f", "momentum", 3, 7, long_short=True))
    btc = cached_frame(cache_dir, "binance_fapi_", "BTCUSDT", "_1d.csv.gz")["close"].pct_change().reindex(result.equity.index).fillna(0.0)
    gross_returns = result.equity.pct_change().fillna(0.0)
    beta = gross_returns.rolling(60, min_periods=60).cov(btc).div(btc.rolling(60, min_periods=60).var()).shift(1).fillna(0.0).clip(-2.0, 2.0)
    hedge_turnover = beta.diff().abs().fillna(beta.abs())
    hedge_cost = hedge_turnover * (PERP_TAKER_RATE + slippage_rate("BTCUSDT"))
    neutral_returns = gross_returns - beta * btc - hedge_cost
    neutral_equity = equity_from_returns(neutral_returns)
    metadata = {
        **result.metadata,
        "oos_label": "OOS_CONTAMINATED_IS_ONLY",
        "intended_factor": "w3c_momentum_vol_target_beta_neutral_overlay",
        "btc_beta_window": 60,
        "btc_beta_cap": 2.0,
        "btc_beta_neutral": True,
    }
    return StrategyResult(
        "W5f",
        "F4",
        neutral_equity,
        neutral_returns[neutral_returns != 0.0],
        result.positions.add(beta.abs(), fill_value=0.0),
        result.turnover.add(hedge_turnover, fill_value=0.0),
        result.stress_total_return,
        metadata,
    )


def load_symbols(cache_dir: Path) -> tuple[str, ...]:
    payload = json.loads((cache_dir / "universe.json").read_text(encoding="utf-8"))
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(symbols, list) or not all(isinstance(symbol, str) for symbol in symbols):
        raise PipelineError("wave-1 cache universe.json is invalid")
    normalized = tuple(validate_symbol(symbol) for symbol in symbols)
    if len(normalized) != len(set(normalized)):
        raise PipelineError("wave-1 cache universe.json contains duplicate symbols")
    return normalized


def select_top_active_positions(
    positions: pd.DataFrame,
    volume_rank: pd.DataFrame,
    max_universe: int,
    max_positions: int,
) -> pd.DataFrame:
    """Select active symbols using only volume known at each decision timestamp."""
    if max_universe < 1 or max_positions < 1:
        raise PipelineError("universe and position limits must be positive")
    selected = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)
    for timestamp in positions.index:
        ranked = volume_rank.loc[timestamp].dropna().nlargest(max_universe).index
        active = positions.loc[timestamp].reindex(ranked).fillna(0.0)
        names = active[active > 0.0].nlargest(max_positions).index
        if len(names) > 0:
            selected.loc[timestamp, names] = 1.0 / len(names)
    return selected


def run_candidates(cache_dir: Path) -> tuple[StrategyResult, ...]:
    symbols = load_symbols(cache_dir)
    return (run_w2c(cache_dir, symbols), run_w5a(cache_dir), run_w5b(cache_dir), run_w5c(cache_dir), run_w5d(cache_dir), run_w5e(cache_dir, symbols), run_w5f(cache_dir, symbols))


__all__ = ["load_symbols", "run_candidates", "run_w2c", "run_w5a", "run_w5b", "run_w5c", "run_w5d", "run_w5e", "run_w5f", "select_top_active_positions"]
