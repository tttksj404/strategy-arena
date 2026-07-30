# Wave-34 genome <-> [0,1]^13 encoding, plus the shared objective.
#
# ---------------------------------------------------------------------------------------
# Why an encoding at all, and why it is not a "repair"
# ---------------------------------------------------------------------------------------
# CMA-ES, PSO, simulated annealing, TPE and MCTS all operate on real vectors, while the frozen
# genome is a mix of categoricals and coupled continuous genes. A fair tournament requires every
# method -- including the random control -- to see the SAME space, so all six sample from
# [0,1]^13 and decode identically.
#
# The delicate part is stop_pct, which is feasible only inside an interval that depends on
# risk_frac (genome30's derived-leverage constraints). Earlier waves handled infeasibility by
# DISCARDING genomes, deliberately never repairing them, so that the search could not drift into
# a region it cannot actually trade. Discarding would wreck a tournament, though: CMA-ES and PSO
# move points continuously and would spend most of their budget on rejected samples, and the
# rejection rate would differ per method, which is exactly the confound the tournament exists to
# remove.
#
# So instead of repairing after the fact, the coordinate is REPARAMETERISED: u[3] means "where in
# the feasible stop range for this risk_frac", not "stop_pct in absolute terms". Every decoded
# vector is feasible by construction and the infeasible region is simply not representable. That
# preserves the fail-closed property (no genome is ever silently edited into validity) while
# giving all six methods an identical, dense, box-constrained domain.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import MarketCache
from research.wave30_qd.engine30 import COMPOUNDING, run_genome
from research.wave30_qd.fitness30 import Evaluation
from research.wave30_qd.genome30 import (
    COOLDOWN_CHOICES,
    ENTRY_THRESHOLD_RANGE,
    LOOKBACK_CHOICES,
    MAX_HOLD_CHOICES,
    RISK_FRAC_RANGE,
    SIGNAL_FAMILIES,
    SLEEVE_FRACTION_CHOICES,
    SYMBOL_SETS,
    TARGET_R_RANGE,
    CONCURRENCY_CHOICES,
    Genome,
    _max_stop_for_risk,
    _min_stop_for_risk,
)
from research.wave33_frequency.fitness33 import entry_profile

DIMENSIONS: Final = 13
MIN_TRADES_PER_ACTIVE_DAY: Final = 1.0
INFEASIBLE_PENALTY: Final = 100.0
LOG_FLOOR: Final = -10.0  # log growth of a wiped account, clipped so optimisers see a finite value
CAPACITY_LIMIT_USDT: Final = 50_000.0  # gate G6; reported here, enforced at judgement


def _pick(u: float, choices: tuple) -> object:
    index = int(np.clip(int(u * len(choices)), 0, len(choices) - 1))
    return choices[index]


def _lerp(u: float, low: float, high: float) -> float:
    return float(low + np.clip(u, 0.0, 1.0) * (high - low))


def decode(vector: np.ndarray) -> Genome:
    """[0,1]^13 -> Genome. Always feasible (see module docstring)."""
    u = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
    risk_frac = _lerp(u[6], *RISK_FRAC_RANGE)
    low = _min_stop_for_risk(risk_frac)
    high = _max_stop_for_risk(risk_frac)
    stop_pct = _lerp(u[3], low, high) if high > low else low
    return Genome(
        signal_family=str(_pick(u[0], SIGNAL_FAMILIES)),
        lookback_bars=int(_pick(u[1], LOOKBACK_CHOICES)),
        entry_threshold=_lerp(u[2], *ENTRY_THRESHOLD_RANGE),
        stop_pct=stop_pct,
        target_r=_lerp(u[4], *TARGET_R_RANGE),
        trail_enabled=bool(u[5] >= 0.5),
        risk_frac=risk_frac,
        max_hold_bars=int(_pick(u[7], MAX_HOLD_CHOICES)),
        allow_short=bool(u[8] >= 0.5),
        symbols=tuple(_pick(u[9], SYMBOL_SETS)),  # type: ignore[arg-type]
        max_concurrent=int(_pick(u[10], CONCURRENCY_CHOICES)),
        cooldown_bars_after_loss=int(_pick(u[11], COOLDOWN_CHOICES)),
        sleeve_fraction=float(_pick(u[12], SLEEVE_FRACTION_CHOICES)),
    ).validate()


def random_vector(rng: np.random.Generator) -> np.ndarray:
    return rng.random(DIMENSIONS)


def objective(cache: MarketCache, vector: np.ndarray) -> Evaluation:
    """Shared objective for all six methods (SPEC.md):

        fitness = log(IS final total capital / 100) - 100 if the frequency/survival
                  constraint is violated

    Compounding sizing, because the request is to maximise profit on a $100 account. The
    capacity number that compounding inflates (wave30 reached $1.9M notional) is recorded on
    every evaluation so gate G6 can enforce a ceiling at judgement time.
    """
    genome = decode(vector)
    result = run_genome(cache, genome, mode="is", sizing=COMPOUNDING)
    profile = entry_profile(cache, result)

    final_total = float(result.total_equity_daily[result.daily_valid][-1])
    log_growth = float(np.clip(np.log(max(final_total, 1e-9) / 100.0), LOG_FLOOR, None))

    reasons: list[str] = []
    if profile.trades_per_active_day < MIN_TRADES_PER_ACTIVE_DAY:
        reasons.append(f"{profile.trades_per_active_day:.3f} entries/active-day < 1.0")
    if not profile.survived_full_span:
        reasons.append("account died before the IS span ended")

    fitness = log_growth - (0.0 if not reasons else INFEASIBLE_PENALTY)
    max_notional = float(max((t.notional_usdt for t in result.trades), default=0.0))
    return Evaluation(
        genome=genome,
        fitness=float(fitness),
        fold_cagrs=(),
        is_total_cagr=0.0,
        is_total_final=final_total,
        sleeve_mdd=profile.account_mdd,
        total_mdd=profile.account_mdd,
        trades_per_year=profile.trades_per_active_day * 365.0,
        n_trades=profile.n_trades,
        n_liquidations=result.n_liquidations,
        wipe_probability=0.0,
        descriptor=(0, 0, 0),  # unused in wave34: this is a scalar tournament, not an archive
        mean_leverage=float(result.mean_realized_leverage),
        min_notional_usdt=float(result.min_notional_usdt),
        sleeve_survived=profile.survived_full_span,
        extras={
            "vector": u_to_list(vector),
            "log_growth": log_growth,
            "infeasible_reasons": reasons,
            "entry_profile": profile.as_dict(),
            "max_notional_usdt": max_notional,
        },
    )


def u_to_list(vector: np.ndarray) -> list[float]:
    return [float(x) for x in np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)]
