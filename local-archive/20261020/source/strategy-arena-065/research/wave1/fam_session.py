# F3 session-effect statistics and conservative cost-after simulations.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from pathlib import Path
from typing import Final, assert_never

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import JsonValue, StrategyResult, load_frame
from research.wave1.costs import PERP_TAKER_RATE, slippage_rate


class SessionKind(StrEnum):
    OVERNIGHT = "overnight_drift"
    WEEKEND = "weekend_crypto_beta"
    FUNDING = "equity_perp_funding_bias"


@dataclass(frozen=True, slots=True)
class SessionCandidate:
    candidate_id: str
    hypothesis: SessionKind


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    mean: float
    t_stat: float
    observations: int
    cost_after_mean: float


F3_CANDIDATES: Final = (
    SessionCandidate("F3-H1", SessionKind.OVERNIGHT),
    SessionCandidate("F3-H2", SessionKind.WEEKEND),
    SessionCandidate("F3-H3", SessionKind.FUNDING),
)


def effect_estimate(values: pd.Series, round_trip_cost: float) -> EffectEstimate:
    clean = values.dropna().astype(float)
    if clean.empty:
        return EffectEstimate(float("nan"), float("nan"), 0, float("nan"))
    mean = float(clean.mean())
    standard_error = float(clean.std(ddof=1) / sqrt(len(clean))) if len(clean) > 1 else float("nan")
    t_stat = mean / standard_error if standard_error > 0.0 else float("nan")
    return EffectEstimate(mean, t_stat, len(clean), mean - round_trip_cost)


def overnight_effect(daily: pd.DataFrame, symbol: str) -> dict[str, EffectEstimate]:
    overnight = daily["open"] / daily["close"].shift(1) - 1.0
    intraday = daily["close"] / daily["open"] - 1.0
    cost = 2.0 * (PERP_TAKER_RATE + slippage_rate(symbol))
    return {
        "overnight": effect_estimate(overnight, cost),
        "intraday": effect_estimate(intraday, cost),
    }


def weekend_beta(equity_returns: pd.Series, btc_returns: pd.Series) -> dict[str, float]:
    aligned = pd.concat(
        [equity_returns.rename("equity"), btc_returns.rename("btc")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2 or float(aligned["btc"].var(ddof=1)) == 0.0:
        return {"beta": float("nan"), "drift": float("nan"), "observations": float(len(aligned))}
    beta = float(aligned["equity"].cov(aligned["btc"]) / aligned["btc"].var(ddof=1))
    drift = float((aligned["equity"] - beta * aligned["btc"]).mean())
    return {"beta": beta, "drift": drift, "observations": float(len(aligned))}


def funding_bias(funding: pd.Series, symbol: str) -> EffectEstimate:
    annualized = funding.astype(float) * 3.0 * 365.0
    return effect_estimate(annualized, 2.0 * (PERP_TAKER_RATE + slippage_rate(symbol)))


def _estimate_payload(estimate: EffectEstimate) -> dict[str, JsonValue]:
    return {
        "mean": estimate.mean,
        "t_stat": estimate.t_stat,
        "observations": estimate.observations,
        "cost_after_mean": estimate.cost_after_mean,
    }


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=-0.999)
    return 300.0 * (1.0 + clean).cumprod()


def _weekend_returns(hourly: pd.DataFrame) -> pd.Series:
    close = hourly["close"].sort_index()
    samples: dict[pd.Timestamp, float] = {}
    for timestamp in close.index[close.index.weekday == 4]:
        if timestamp.hour != 21:
            continue
        end = timestamp + pd.Timedelta(days=2, hours=16, minutes=30)
        future = close.loc[timestamp:end]
        if len(future) > 1:
            samples[pd.Timestamp(timestamp)] = float(future.iloc[-1] / future.iloc[0] - 1.0)
    return pd.Series(samples, dtype=float).sort_index()


def run_cached(cache_dir: Path, candidate: SessionCandidate) -> StrategyResult:
    match candidate.hypothesis:
        case SessionKind.OVERNIGHT:
            samples: list[pd.Series] = []
            metadata: dict[str, JsonValue] = {}
            for symbol in ("SPY", "QQQ"):
                baseline = load_frame(cache_dir / f"yahoo_{symbol}_1d.csv.gz")
                estimate = overnight_effect(baseline, symbol)
                metadata[symbol] = {name: _estimate_payload(value) for name, value in estimate.items()}
                bitget = load_frame(cache_dir / f"bitget_{symbol}USDT_1H.csv.gz")
                daily = bitget.resample("1D").agg({"open": "first", "close": "last"}).dropna()
                cost = 2.0 * (PERP_TAKER_RATE + slippage_rate(symbol))
                samples.append((daily["open"] / daily["close"].shift(1) - 1.0 - cost).dropna())
            returns = pd.concat(samples, axis=1).mean(axis=1).dropna()
            stress_returns = returns - 2.0 * (slippage_rate("SPY") + slippage_rate("QQQ")) / 2.0
        case SessionKind.WEEKEND:
            equity_weekend = _weekend_returns(load_frame(cache_dir / "bitget_SPYUSDT_1H.csv.gz"))
            btc_weekend = _weekend_returns(load_frame(cache_dir / "bitget_BTCUSDT_1H.csv.gz"))
            metadata = weekend_beta(equity_weekend, btc_weekend)
            returns = equity_weekend - 2.0 * (PERP_TAKER_RATE + slippage_rate("SPY"))
            stress_returns = returns - 2.0 * slippage_rate("SPY")
        case SessionKind.FUNDING:
            funding_samples: list[pd.Series] = []
            metadata = {}
            for symbol in ("SPY", "QQQ", "TSLA", "NVDA", "MSTR"):
                funding = load_frame(cache_dir / f"bitget_funding_{symbol}USDT.csv.gz")["funding_rate"]
                metadata[symbol] = _estimate_payload(funding_bias(funding, symbol))
                funding_samples.append(funding.rename(symbol))
            returns = pd.concat(funding_samples, axis=1).mean(axis=1).dropna()
            stress_returns = returns.copy()
            if len(returns) > 0:
                returns.iloc[0] -= PERP_TAKER_RATE + slippage_rate("SPY")
                returns.iloc[-1] -= PERP_TAKER_RATE + slippage_rate("SPY")
                average_slippage = float(np.mean([slippage_rate(symbol) for symbol in ("SPY", "QQQ", "TSLA", "NVDA", "MSTR")]))
                stress_returns.iloc[0] -= PERP_TAKER_RATE + 2.0 * average_slippage
                stress_returns.iloc[-1] -= PERP_TAKER_RATE + 2.0 * average_slippage
        case unreachable:
            assert_never(unreachable)
    equity = _equity_from_returns(returns)
    stress_equity = _equity_from_returns(stress_returns)
    split = pd.Timestamp("2025-09-30T23:59:59Z")
    stress_is = stress_equity[stress_equity.index <= split]
    oos_stress = stress_equity[stress_equity.index > split]
    stress_anchor = float(stress_is.iloc[-1]) if not stress_is.empty else 300.0
    stress_total = float(oos_stress.iloc[-1] / stress_anchor - 1.0) if not oos_stress.empty else 0.0
    positions = pd.Series(1.0, index=equity.index, dtype=float)
    turnover = pd.Series(0.0, index=equity.index, dtype=float)
    if not turnover.empty:
        turnover.iloc[0] = 1.0
        turnover.iloc[-1] += 1.0
    return StrategyResult(
        candidate_id=candidate.candidate_id,
        family="F3",
        equity=equity,
        trade_returns=returns.astype(float),
        positions=positions,
        turnover=turnover,
        stress_total_return=stress_total,
        metadata={
            "analysis": metadata,
            "exploratory_only": True,
            "cost_model_valid": True,
            "intended_factor": candidate.hypothesis.value,
            "max_concurrent_positions": 1,
            "max_position_weight": 1.0,
            "min_position_weight": 1.0,
            "min_order_usdt": 5.0,
            "candidate_config": {"hypothesis": candidate.hypothesis.value},
        },
    )
