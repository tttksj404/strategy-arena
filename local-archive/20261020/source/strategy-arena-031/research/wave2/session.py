from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import JsonValue, StrategyResult, load_frame
from research.wave2.funding import W2_MAKER_FEE_RATE
from research.wave1.fam_session import effect_estimate


W2G_SYMBOLS: Final = ("SPY", "QQQ", "TSLA", "NVDA", "MSTR")


def _symbol_returns(cache_dir: Path, symbol: str) -> tuple[pd.Series, dict[str, JsonValue]]:
    mark = load_frame(cache_dir / f"bitget_{symbol}USDT_1H.csv.gz").sort_index()
    funding = load_frame(cache_dir / f"bitget_funding_{symbol}USDT.csv.gz")["funding_rate"].sort_index()
    mark = mark.loc[mark.index >= funding.index.min()]
    mark_return = mark["close"].pct_change().clip(lower=-0.999)
    aligned_funding = funding.reindex(mark.index, method="ffill").shift(1)
    spot_path = cache_dir / f"yahoo_{symbol}_1d.csv.gz"
    spot_return = pd.Series(float("nan"), index=mark.index, dtype=float)
    if spot_path.exists():
        spot = load_frame(spot_path).sort_index()
        spot_close = spot["close"].reindex(mark.index, method="ffill")
        spot_return = spot_close.pct_change().clip(lower=-0.999)
    basis_returns = (spot_return - mark_return + aligned_funding).replace([np.inf, -np.inf], np.nan).dropna()
    directional_returns = (-mark_return + aligned_funding).replace([np.inf, -np.inf], np.nan).dropna()
    spot_available = len(basis_returns) > 0
    returns = basis_returns if spot_available else directional_returns
    estimate = effect_estimate(returns, 2.0 * W2_MAKER_FEE_RATE)
    return returns, {
        "observations": len(returns),
        "mark_to_market_mean": float((-mark_return).dropna().mean()),
        "funding_mean": float(aligned_funding.mean()),
        "spot_proxy_available": spot_available,
        "basis_mode": "mark_minus_spot" if spot_available else "directional_mark_only",
        "basis_mark_to_market_included": spot_available,
        "effect": {
            "mean": estimate.mean,
            "t_stat": estimate.t_stat,
            "cost_after_mean": estimate.cost_after_mean,
            "cost_after_sign": "positive" if estimate.cost_after_mean > 0.0 else "non_positive",
        },
        "delta_neutral": False,
        "directional_risk_exposed": not spot_available,
    }


def run_w2g(cache_dir: Path) -> StrategyResult:
    samples: list[pd.Series] = []
    analysis: dict[str, JsonValue] = {}
    basis_symbols: list[str] = []
    directional_symbols: list[str] = []
    for symbol in W2G_SYMBOLS:
        returns, details = _symbol_returns(cache_dir, symbol)
        samples.append(returns.rename(symbol))
        analysis[symbol] = details
        if details["basis_mark_to_market_included"] is True:
            basis_symbols.append(symbol)
        else:
            directional_symbols.append(symbol)
    returns = pd.concat(samples, axis=1).mean(axis=1).dropna()
    if not returns.empty:
        returns.iloc[0] -= 2.0 * W2_MAKER_FEE_RATE
        returns.iloc[-1] -= 2.0 * W2_MAKER_FEE_RATE
    aggregate_effect = effect_estimate(returns, 2.0 * W2_MAKER_FEE_RATE)
    equity = 300.0 * (1.0 + returns.clip(lower=-0.999)).cumprod()
    stress_returns = returns.copy()
    stress_equity = 300.0 * (1.0 + stress_returns.clip(lower=-0.999)).cumprod()
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    stress_is = stress_equity[stress_equity.index <= split]
    stress_oos = stress_equity[stress_equity.index > split]
    anchor = float(stress_is.iloc[-1]) if not stress_is.empty else 300.0
    stress_total = float(stress_oos.iloc[-1] / anchor - 1.0) if not stress_oos.empty else 0.0
    positions = pd.Series(1.0, index=equity.index, dtype=float)
    turnover = pd.Series(0.0, index=equity.index, dtype=float)
    if not turnover.empty:
        turnover.iloc[0] = 1.0
        turnover.iloc[-1] += 1.0
    return StrategyResult(
        "W2g",
        "F3",
        equity,
        returns,
        positions,
        turnover,
        stress_total,
        {
            "analysis": analysis,
            "exploratory_only": True,
            "cost_model_valid": len(returns) > 0,
            "cost_route": "maker_0.02pct_per_leg_zero_slippage",
            "intended_factor": "equity_perp_funding_bias",
            "max_concurrent_positions": 1,
            "max_position_weight": 1.0,
            "min_position_weight": 1.0,
            "min_order_usdt": 5.0,
            "basis_mark_to_market_included": bool(basis_symbols),
            "basis_symbols": basis_symbols,
            "directional_only_symbols": directional_symbols,
            "basis_complete": not directional_symbols,
            "delta_neutral": False,
            "directional_risk_exposed": True,
            "data_window": "Bitget 133d cache",
            "data_valid": not returns.empty and not directional_symbols,
            "effect": {
                "mean": aggregate_effect.mean,
                "t_stat": aggregate_effect.t_stat,
                "cost_after_mean": aggregate_effect.cost_after_mean,
                "cost_after_sign": "positive" if aggregate_effect.cost_after_mean > 0.0 else "non_positive",
            },
        },
    )
