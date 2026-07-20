from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.common import PipelineError
from research.wave1.costs import PERP_TAKER_RATE, f1_round_trip_cost


def zscore_hysteresis_position(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
) -> pd.Series:
    if entry_z <= exit_z or exit_z < 0.0:
        raise PipelineError("z-score thresholds must satisfy entry > exit >= 0")
    values: list[float] = []
    active = 0.0
    for value in zscore.astype(float):
        if pd.notna(value):
            if value < -entry_z:
                active = 1.0
            elif value > entry_z:
                active = -1.0
            elif abs(value) < exit_z:
                active = 0.0
        values.append(active)
    return pd.Series(values, index=zscore.index, dtype=float)


def funding_capitulation_position(
    funding: pd.Series,
    threshold: float,
    hold_bars: int,
) -> pd.Series:
    if threshold >= 0.0 or hold_bars < 1:
        raise PipelineError("funding threshold must be negative and hold_bars positive")
    values: list[float] = []
    remaining = 0
    for value in funding.astype(float):
        if pd.notna(value) and value < threshold:
            remaining = hold_bars
        values.append(1.0 if remaining > 0 else 0.0)
        remaining = max(0, remaining - 1)
    return pd.Series(values, index=funding.index, dtype=float)


def rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    if window < 2:
        raise PipelineError("z-score window must be at least two")
    mean = values.rolling(window, min_periods=window).mean()
    deviation = values.rolling(window, min_periods=window).std(ddof=0)
    return ((values - mean) / deviation.where(deviation > 0.0)).replace([np.inf, -np.inf], np.nan)


def basis_round_trip_cost(notional: float, slippage_rate: float) -> float:
    return f1_round_trip_cost(notional, slippage_rate)


def pair_round_trip_cost(notional: float, slippage_rate: float) -> float:
    return abs(notional) * 4.0 * (PERP_TAKER_RATE + slippage_rate)


def combine_returns(
    baseline: pd.Series,
    candidate: pd.Series,
    weight: float = 0.5,
) -> pd.Series:
    if not 0.0 <= weight <= 1.0:
        raise PipelineError("portfolio weight must be between zero and one")
    aligned = pd.concat([baseline.rename("baseline"), candidate.rename("candidate")], axis=1).fillna(0.0)
    return (aligned["baseline"] * weight + aligned["candidate"] * (1.0 - weight)).rename(None)


def equity_from_returns(returns: pd.Series, initial_capital: float = 300.0) -> pd.Series:
    return initial_capital * (1.0 + returns.astype(float).clip(lower=-0.999999)).cumprod()


def maximum_drawdown(equity: pd.Series) -> float:
    clean = equity.dropna().astype(float)
    if clean.empty:
        return 0.0
    return abs(float((clean / clean.cummax() - 1.0).min()))


def annualized_cagr(equity: pd.Series) -> float:
    clean = equity.dropna().sort_index()
    if len(clean) < 2 or clean.iloc[0] <= 0.0 or clean.iloc[-1] <= 0.0:
        return 0.0
    days = max((clean.index[-1] - clean.index[0]).total_seconds() / 86_400.0, 1.0)
    return float((clean.iloc[-1] / clean.iloc[0]) ** (365.0 / days) - 1.0)


def aligned_correlation(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    return float(aligned["left"].corr(aligned["right"])) if len(aligned) >= 2 else float("nan")


def rsi(close: pd.Series, window: int = 2) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = (-change.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    relative = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + relative)).where(loss > 0.0, 100.0).fillna(50.0)


def cached_frame(cache_dir: Path, prefix: str, symbol: str, suffix: str) -> pd.DataFrame:
    from research.wave1.common import validate_symbol

    safe_symbol = validate_symbol(symbol)
    cache_root = cache_dir.resolve()
    path = (cache_root / f"{prefix}{safe_symbol}{suffix}").resolve()
    if not path.is_file() or path.parent != cache_root:
        raise PipelineError(f"required wave-1 cache file is missing: {path.name}")

    from research.wave1.common import load_frame
    return load_frame(path)


__all__ = [
    "aligned_correlation",
    "annualized_cagr",
    "basis_round_trip_cost",
    "cached_frame",
    "combine_returns",
    "equity_from_returns",
    "funding_capitulation_position",
    "maximum_drawdown",
    "pair_round_trip_cost",
    "rolling_zscore",
    "rsi",
    "zscore_hysteresis_position",
]
