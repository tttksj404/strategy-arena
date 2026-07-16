"""W3e cross-sectional max-z selection and rank-based hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK


FACTOR_ORDER: Final = ("carry", "momentum")


@dataclass(frozen=True, slots=True)
class RankedSignal:
    """A selected factor signal with its cross-sectional rank."""

    symbol: str
    factor: str
    score: float
    rank: int


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    """Standardize one date's factor values without inventing scores for NaNs."""
    clean = values.astype(float)
    mean = clean.mean(skipna=True)
    std = clean.std(skipna=True, ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=clean.index, dtype=float).where(clean.notna())
    return (clean - mean) / std


def select_max_z_candidates(
    carry_z: pd.Series,
    momentum_z: pd.Series,
    top_k: int = 3,
) -> tuple[RankedSignal, ...]:
    """Select the top ``top_k`` symbols by max(carry-z, momentum-z)."""
    if top_k < 1:
        return ()
    candidates: list[tuple[str, str, float]] = []
    symbols = sorted(set(carry_z.index.astype(str)) | set(momentum_z.index.astype(str)))
    for symbol in symbols:
        factor_values = {"carry": carry_z.get(symbol), "momentum": momentum_z.get(symbol)}
        available = [(factor, float(value)) for factor, value in factor_values.items() if pd.notna(value)]
        if available:
            factor, score = max(available, key=lambda item: (item[1], -FACTOR_ORDER.index(item[0])))
            candidates.append((symbol, factor, score))
    candidates.sort(key=lambda item: (-item[2], item[0], FACTOR_ORDER.index(item[1])))
    return tuple(RankedSignal(symbol, factor, score, rank) for rank, (symbol, factor, score) in enumerate(candidates[:top_k], 1))


def update_hysteresis(
    previous: tuple[RankedSignal, ...],
    ranked_today: tuple[RankedSignal, ...],
    entry_count: int = 3,
    exit_rank: int = 10,
) -> tuple[RankedSignal, ...]:
    """Enter top-ranked signals and retain existing signals through rank ``exit_rank``."""
    today = {signal.symbol: signal for signal in ranked_today}
    selected = list(ranked_today[:entry_count])
    selected_symbols = {signal.symbol for signal in selected}
    for held in previous:
        current = today.get(held.symbol)
        if current is not None and current.rank <= exit_rank and held.symbol not in selected_symbols:
            selected.append(current)
            selected_symbols.add(held.symbol)
    return tuple(selected)


__all__ = ["RankedSignal", "cross_sectional_zscore", "select_max_z_candidates", "update_hysteresis"]
