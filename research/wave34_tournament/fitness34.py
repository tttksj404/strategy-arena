# Wave-34 objective: ONE scalar, identical for all seven arms.
#
#   fitness = log(final $ on a $100 account)  -  penalty(constraint violations)
#
# ---------------------------------------------------------------------------------------
# Why log, and why the account is still FIXED-BASE sized
# ---------------------------------------------------------------------------------------
# The user asked to maximise the money, so the objective is the account's final dollar value.
# log() is a monotone transform of it, so the ARGMAX is unchanged; it is used only because the
# raw value spans several orders of magnitude and every optimizer here (CMA-ES step control,
# PSO velocity, SA acceptance temperature, TPE's good/bad split) behaves badly on a scalar
# whose scale drifts by 10^3 between candidates.
#
# Sizing stays engine30.SizingMode(fixed_base=True) -- the same evaluation path as wave33 --
# for two reasons that survived that wave's review: (a) compounding re-introduces the capacity
# fiction (wave30 reached $1.9M notional against a cost model fitted at $45), and (b) keeping
# the path identical to wave33 is what makes these numbers comparable to wave33's. Under a
# fixed base the account still grows and still dies: P&L accumulates into equity, and when
# equity can no longer fund a full-size position the engine stops trading. So "final $" is a
# real, measured account value, not a growth multiple of a fiction.
#
# ---------------------------------------------------------------------------------------
# Why infeasible genomes are penalised rather than discarded
# ---------------------------------------------------------------------------------------
# CMA-ES, PSO and SA all need a defined value at every point of the box -- discarding would
# hand them a NaN landscape. The penalty is GRADED (it shrinks as the frequency shortfall
# shrinks) so those methods can walk downhill into the feasible region instead of wandering a
# flat plateau. The base offset is large enough that no infeasible point can ever outrank a
# feasible one: feasible fitness is bounded above by log(account) and, at the $100 start with
# a fixed base, has never come near +BASE_PENALTY in any wave.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import MarketCache
from research.wave30_qd.engine30 import run_genome
from research.wave30_qd.genome30 import Genome
from research.wave33_frequency.fitness33 import (
    FIXED_SIZING,
    MIN_TRADES_PER_ACTIVE_DAY,
    MIN_TRADES_FOR_STATS,
    entry_profile,
)
from research.wave34_tournament.encoding34 import DIM, decode

BASE_PENALTY: Final = 50.0
EQUITY_FLOOR: Final = 1e-3  # $ -- keeps log() finite for a fully drained account
START_CAPITAL: Final = 100.0


@dataclass(frozen=True)
class Trial:
    """One evaluated point. Everything the report needs, nothing the search can leak OOS from."""

    x: tuple[float, ...]
    fitness: float
    feasible: bool
    final_usdt: float
    trades_per_active_day: float
    survived: bool
    n_trades: int
    mean_usdt: float
    median_usdt: float
    share_ge_target: float
    account_mdd: float
    leverage: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": list(self.x),
            "genome": decode(np.array(self.x)).to_dict(),
            "fitness": self.fitness,
            "feasible": self.feasible,
            "final_usdt": self.final_usdt,
            "trades_per_active_day": self.trades_per_active_day,
            "survived": self.survived,
            "n_trades": self.n_trades,
            "mean_usdt": self.mean_usdt,
            "median_usdt": self.median_usdt,
            "share_ge_target": self.share_ge_target,
            "account_mdd": self.account_mdd,
            "leverage": self.leverage,
            "reasons": list(self.reasons),
        }


def evaluate_vector(cache: MarketCache, x: np.ndarray) -> Trial:
    genome: Genome = decode(x)
    result = run_genome(cache, genome, mode="is", sizing=FIXED_SIZING)
    profile = entry_profile(cache, result)

    final_usdt = max(float(profile.account_final_usdt), EQUITY_FLOOR)
    growth = float(np.log(final_usdt / START_CAPITAL))

    reasons: list[str] = []
    shortfall = 0.0
    if profile.n_trades < MIN_TRADES_FOR_STATS:
        reasons.append("too few trades")
        shortfall += 1.0 - profile.n_trades / MIN_TRADES_FOR_STATS
    if profile.trades_per_active_day < MIN_TRADES_PER_ACTIVE_DAY:
        reasons.append("below 1 entry per active day")
        shortfall += 1.0 - profile.trades_per_active_day / MIN_TRADES_PER_ACTIVE_DAY
    if not profile.survived_full_span:
        reasons.append("account died before the span ended")
        shortfall += 1.0

    fitness = growth if not reasons else growth - BASE_PENALTY - 10.0 * shortfall
    return Trial(
        x=tuple(float(v) for v in np.clip(x, 0.0, 1.0)),
        fitness=float(fitness),
        feasible=not reasons,
        final_usdt=float(profile.account_final_usdt),
        trades_per_active_day=float(profile.trades_per_active_day),
        survived=bool(profile.survived_full_span),
        n_trades=int(profile.n_trades),
        mean_usdt=float(profile.mean_usdt),
        median_usdt=float(profile.median_usdt),
        share_ge_target=float(profile.share_ge_target),
        account_mdd=float(profile.account_mdd),
        leverage=float(genome.leverage),
        reasons=tuple(reasons),
    )


class BudgetExhausted(RuntimeError):
    """Raised the moment an optimizer asks for one evaluation past its budget."""


class Objective:
    """Shared, memoised, budget-capped objective handed to every optimizer.

    Only DISTINCT genomes count against the budget -- a cache hit is free, exactly as in
    wave30's Evaluator, so a method cannot buy extra search by re-proposing the same point.
    Vectors are keyed by the DECODED genome, not by the float vector, because two nearby
    vectors that decode to the same genome are the same experiment.
    """

    def __init__(self, cache: MarketCache, budget: int, deadline_seconds: float | None = None) -> None:
        self.cache = cache
        self.budget = int(budget)
        # Hard wall-clock backstop. The budget is the thing held equal between arms, so this
        # must NOT fire in a healthy run -- it exists because per-evaluation cost is a
        # property of the genomes a method proposes and can rise several-fold once a method
        # starts finding survivors. If it does fire, the result file records `truncated: true`
        # and the affected arm is reported as having spent LESS than the shared budget, which
        # is a caveat on that arm's number rather than something to be smoothed over.
        self.deadline_seconds = deadline_seconds
        self.started = time.monotonic()
        self.truncated = False
        self.n_evaluations = 0
        self._memo: dict[tuple, Trial] = {}
        self.best: Trial | None = None
        self.best_feasible: Trial | None = None
        self.history: list[float] = []  # best-feasible fitness after each distinct evaluation

    @property
    def exhausted(self) -> bool:
        if self.n_evaluations >= self.budget:
            return True
        if self.deadline_seconds is not None and time.monotonic() - self.started > self.deadline_seconds:
            self.truncated = True
            return True
        return False

    def __call__(self, x: np.ndarray) -> float:
        return self.trial(x).fitness

    def trial(self, x: np.ndarray) -> Trial:
        x = np.clip(np.asarray(x, dtype=float).reshape(DIM), 0.0, 1.0)
        key = decode(x).key()
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        if self.exhausted:
            raise BudgetExhausted(f"budget {self.budget} exhausted")
        result = evaluate_vector(self.cache, x)
        self._memo[key] = result
        self.n_evaluations += 1
        if self.best is None or result.fitness > self.best.fitness:
            self.best = result
        if result.feasible and (self.best_feasible is None or result.fitness > self.best_feasible.fitness):
            self.best_feasible = result
        self.history.append(self.best_feasible.fitness if self.best_feasible else float("-inf"))
        return result
