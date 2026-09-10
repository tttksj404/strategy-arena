"""Typed experiment outputs, sizing policies, and validation-only selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, assert_never

import numpy as np
from pydantic import BaseModel, ConfigDict

from selection_bias_stats import DeflatedSharpeResult, PboResult
from x_discovery_account import AccountPath, daily_returns, kelly_schedule
from x_discovery_pairs import PairChoice
from x_discovery_signals import SignalSpec
from x_discovery_types import FloatMatrix, FloatVector


class Sizing(StrEnum):
    FIXED1 = "fixed_1"
    FIXED2 = "fixed_2"
    FIXED5 = "fixed_5"
    KELLY = "quarter_kelly"
    CAUTIOUS = "quarter_kelly_minus_se"


@dataclass(frozen=True, slots=True)
class Configuration:
    interval: str
    signal: SignalSpec
    sizing: Sizing
    volume_exit: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.interval}|{self.signal.identifier}|{self.sizing}|vexit={self.volume_exit:g}"

    @property
    def identifier(self) -> str:
        return sha256(self.label.encode()).hexdigest()[:16]


class Metrics(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    compounded_return: float
    max_drawdown: float
    liquidation_proxy_index: int
    active_days: int
    mean_gross_exposure: float
    turnover_equity_units: float
    daily_sharpe: float


class Trial(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    config: Configuration
    identifier: str
    parent_signal: str
    base: Metrics
    stress: Metrics
    eligible: bool
    failure_reasons: tuple[str, ...]
    pair_pool: tuple[PairChoice, ...]


class TestEvaluation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    trial: Trial
    dsr_local_approximation: DeflatedSharpeResult | None
    dsr_cumulative_approximation: DeflatedSharpeResult | None


class DataEvidence(BaseModel):
    interval: str
    rows: int
    start: str
    end: str
    snapshot_ms: int
    partition_hashes: tuple[str, ...]


class DiscoveryReport(BaseModel):
    schema_version: str = "x-discovery-v2"
    candidate_count: int
    preregistration_sha256: str
    selection_independent: bool = False
    exact_bitget_liquidation: bool = False
    live_promotable: bool = False
    train_end: str
    validation_start: str
    historical_test_start: str
    evaluation_end: str
    winner_id: str | None
    test_ids_sha256: str
    validation: tuple[Trial, ...]
    historical_test: tuple[TestEvaluation, ...]
    pbo: PboResult | None
    pbo_candidate_count: int
    pbo_unique_paths: int
    data: tuple[DataEvidence, ...]
    limitations: tuple[str, ...]


def size_schedule(sizing: Sizing, shadow: FloatVector, bars_per_day: int) -> FloatVector:
    match sizing:
        case Sizing.FIXED1:
            return np.ones_like(shadow)
        case Sizing.FIXED2:
            return np.full_like(shadow, 2.)
        case Sizing.FIXED5:
            return np.full_like(shadow, 5.)
        case Sizing.KELLY:
            return kelly_schedule(shadow, 60 * bars_per_day, False)
        case Sizing.CAUTIOUS:
            return kelly_schedule(shadow, 60 * bars_per_day, True)
        case _:
            assert_never(sizing)


def metrics(path: AccountPath, segment_weights: FloatMatrix, bars_per_day: int) -> Metrics:
    daily = daily_returns(path, bars_per_day)
    peaks = np.maximum.accumulate(path.equity)
    volatility = float(daily.std(ddof=1)) if len(daily) > 1 else 0.
    active_days = sum(bool(np.any(segment_weights[index:index+bars_per_day]))
                      for index in range(0, len(segment_weights), bars_per_day))
    return Metrics(compounded_return=path.equity.item(-1) - 1,
                   max_drawdown=float(np.max(1 - path.equity / peaks)),
                   liquidation_proxy_index=path.liquidation_index, active_days=active_days,
                   mean_gross_exposure=path.average_gross, turnover_equity_units=path.turnover,
                   daily_sharpe=float(daily.mean() / volatility) if volatility > 0 else 0.)


def select_winner(trials: tuple[Trial, ...]) -> Trial | None:
    eligible = tuple(item for item in trials if item.eligible)
    return max(eligible, key=lambda item: item.stress.compounded_return) if eligible else None
