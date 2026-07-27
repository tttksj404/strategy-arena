# Wave-22 thin evaluation wrapper around research/wave21_ga/fitness.py's own engine (task
# instruction: reuse build_market_cache/run_backtest/cagr/oos_slice/_max_drawdown verbatim,
# never reimplement the backtest). This module adds nothing numeric of its own -- it only
# packages fitness.py's outputs into one small, JSON-friendly summary per genome (full-period
# CAGR, OOS CAGR [self-contained AND regime-anchored], MDD, gross exposure/feasibility) and a
# genome_key-based cache so the ~150 genome evaluations wave22's 6 validations need never
# backtest the same genome twice (mirrors fitness.evaluate_genome_cached's own convention).
#
# mode=MODE_OOS_FINAL is used for every evaluation in this module, deliberately. This is NOT a
# re-opening of wave21's evolutionary OOS seal: that seal exists to stop a GA/random-search LOOP
# from selecting genomes based on OOS performance. wave22 runs no search at all -- every genome
# evaluated here (G1 itself, its +-10/20% neighbors, I5, shuffled random genomes, one-at-a-time
# attribution genomes) is either already-fixed (G1, I5) or drawn independently of any OOS-based
# selection criterion. Auditing an already-frozen candidate's OOS behavior is the explicit
# purpose of this wave (task: "OOS를 통과했지만... 이 판정이 실전 투입 여부를 가른다"), matching
# the same one-time-final-evaluation pattern research/wave21_ga/fitness.py's own
# final_evaluation() already uses for the identical purpose.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import pandas as pd  # noqa: PANDAS_OK

from research.wave10_carry100.engine import ACTIVE_CAPITAL
from research.wave10_carry100.regime import regime_breakdown
from research.wave21_ga import fitness, gates21
from research.wave21_ga.genome import Genome, genome_key


class _EquityOnly:
    """Duck-typed stand-in so regime_breakdown (which only reads `.equity`) accepts a bare
    pd.Series -- same pattern fitness.py's own _EquityOnly / gates18.py's _EquityOnly use."""

    def __init__(self, equity: pd.Series) -> None:
        self.equity = equity


@dataclass(frozen=True, slots=True)
class GenomeMetrics:
    full_cagr: float
    oos_cagr_self_contained: float  # cagr(fitness.oos_slice(equity)) -- matches STRATEGY_CARD.md's reported G1 OOS figure (4.04%), see tests/test_wave22.py
    oos_cagr_regime_anchored: float | None  # regime_breakdown's OOS_SPLIT-anchored figure -- secondary cross-check, matches wave21_report.md's H2 methodology
    mdd_full: float
    gross_usdt: float
    gross_feasible_1x: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_cagr": self.full_cagr,
            "oos_cagr_self_contained": self.oos_cagr_self_contained,
            "oos_cagr_regime_anchored": self.oos_cagr_regime_anchored,
            "mdd_full": self.mdd_full,
            "gross_usdt": self.gross_usdt,
            "gross_feasible_1x": self.gross_feasible_1x,
        }


def full_equity(genome: Genome, cache: fitness.MarketCache) -> pd.Series:
    """The one place this module calls run_backtest directly -- everything else derives from
    this equity curve."""
    return fitness.run_backtest(genome, cache, fitness.MODE_OOS_FINAL)


def metrics_from_equity(genome: Genome, equity: pd.Series) -> GenomeMetrics:
    oos_equity = fitness.oos_slice(equity, fitness.MODE_OOS_FINAL)
    regime = regime_breakdown(_EquityOnly(equity))
    current_low_funding = regime.get("current_low_funding")
    oos_anchored = current_low_funding.get("annualized_return") if isinstance(current_low_funding, dict) else None
    gross = gates21.gross_usdt(genome)
    return GenomeMetrics(
        full_cagr=fitness.cagr(equity),
        oos_cagr_self_contained=fitness.cagr(oos_equity),
        oos_cagr_regime_anchored=oos_anchored,
        mdd_full=fitness._max_drawdown(equity),
        gross_usdt=gross,
        gross_feasible_1x=bool(gross <= ACTIVE_CAPITAL + 1e-9),
    )


def evaluate(genome: Genome, cache: fitness.MarketCache) -> GenomeMetrics:
    return metrics_from_equity(genome, full_equity(genome, cache))


class MetricsCache:
    """genome_key-keyed memo across an entire wave22 run (task-spec precedent: fitness.py's own
    evaluate_genome_cached). Tracks hit/miss counts so run_wave22.py can report how many of the
    ~150 nominal evaluations were fresh backtests vs. reused lookups (same disclosure wave21's
    own evolve stage prints)."""

    def __init__(self) -> None:
        self._store: dict[tuple, GenomeMetrics] = {}
        self.hits = 0
        self.misses = 0

    def get(self, genome: Genome, cache: fitness.MarketCache) -> GenomeMetrics:
        key = genome_key(genome)
        cached = self._store.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = evaluate(genome, cache)
        self._store[key] = result
        return result

    def __len__(self) -> int:
        return len(self._store)


__all__ = ["GenomeMetrics", "MetricsCache", "evaluate", "full_equity", "metrics_from_equity"]
