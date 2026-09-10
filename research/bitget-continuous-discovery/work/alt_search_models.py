"""Typed value objects for the paper-only altcoin strategy search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

FloatMatrix: TypeAlias = npt.NDArray[np.float64]
FloatVector: TypeAlias = npt.NDArray[np.float64]
BASE_FEE: Final = 0.0006
BASE_SLIPPAGE: Final = 0.0003
STRESS_SLIPPAGE: Final = 0.0006
LIQUIDATION_RESERVE: Final = 0.02


class Family(StrEnum):
    """Closed set of predeclared signal families."""

    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"
    DONCHIAN = "donchian"
    EMA = "ema"
    BOLLINGER = "bollinger"
    RANGE = "range"
    PRICE_VOLUME = "price_volume"
    PAIR = "pair"


@dataclass(frozen=True, slots=True)
class Candidate:
    """A signal parameterization known before validation is inspected."""

    family: Family
    lookback: int
    secondary: int
    top_k: int
    direction: int
    pair: tuple[str, str] | None
    entry_z: float
    rebalance_days: int = 1
    minimum_daily_quote_volume: float = 0.0

    @property
    def identifier(self) -> str:
        """Return a deterministic audit identifier."""

        pair_label = "none" if self.pair is None else f"{self.pair[0]}-{self.pair[1]}"
        return (
            f"{self.family.value}|lb={self.lookback}|sec={self.secondary}|k={self.top_k}"
            f"|dir={self.direction}|pair={pair_label}|z={self.entry_z:g}|reb={self.rebalance_days}"
            f"|liq={self.minimum_daily_quote_volume:g}"
        )


@dataclass(frozen=True, slots=True)
class RiskOverlay:
    """A lagged volatility target and maximum leverage cap."""

    max_leverage: float
    target_daily_vol: float
    volatility_window: int

    @property
    def identifier(self) -> str:
        """Return a deterministic audit identifier."""

        return f"L={self.max_leverage:g}|tv={self.target_daily_vol:g}|vw={self.volatility_window}"


@dataclass(frozen=True, slots=True)
class MarketData:
    """Aligned closed daily bars for the fixed altcoin universe."""

    dates: tuple[datetime, ...]
    symbols: tuple[str, ...]
    open_price: FloatMatrix
    high_price: FloatMatrix
    low_price: FloatMatrix
    close_price: FloatMatrix
    quote_volume: FloatMatrix


@dataclass(frozen=True, slots=True)
class SegmentMetrics:
    """Cost-adjusted return statistics for one chronological segment."""

    net_return: float
    max_drawdown: float
    sharpe: float
    positive_blocks: int
    observations: int


@dataclass(frozen=True, slots=True)
class PathResult:
    """A simulated path plus conservative liquidation evidence."""

    returns: FloatVector
    stress_returns: FloatVector
    leverage: FloatVector
    liquidations: int
    first_liquidation_index: int
    liquidation_indices: tuple[int, ...] = ()
