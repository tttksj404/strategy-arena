# Wave-31 gates Q1-Q7 exactly as frozen in SPEC.md.
#
# Q1, Q5 and Q6 are wave30's P1/P5/P6 reused UNMODIFIED (only the DSR trial count differs, and
# that count is the cumulative one SPEC.md registers) -- the method-validity, deflated-Sharpe
# and executability questions are identical across the two waves and there is no reason to
# have two implementations that could drift apart. Q2/Q3/Q4/Q7 are the sprint-specific ones.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave30_qd.engine30 import TOTAL_CAPITAL
from research.wave30_qd.fitness30 import bootstrap_wipe_probability
from research.wave30_qd.gates30 import (
    GateOutcome,
    _daily_returns,
    gate_p1_method_validity,
    gate_p5_deflated_sharpe,
    gate_p6_executability,
)
from research.wave30_qd.genome30 import Genome
from research.wave31_sprint.fitness31 import FITNESS_WINDOW, window_statistics

Q3_MAX_PROB_HALVING: Final = 0.10
Q3_MAX_PROB_DECIMATION: Final = 0.01
Q3_MAX_WIPE_PROBABILITY: Final = 0.05
Q4_RUIN_FLOOR_USDT: Final = 50.0
Q4_MAX_RUIN_PROBABILITY: Final = 0.05
Q5_CUMULATIVE_TRIALS: Final = 255_621
Q7_MIN_POSITIVE_SHARE: Final = 0.50
MC_PATHS: Final = 10_000


def gate_q2_oos_sprint(candidate_oos: dict, baseline_oos: dict) -> GateOutcome:
    candidate = candidate_oos["windows"][str(FITNESS_WINDOW)]["p50"]
    baseline = baseline_oos["windows"][str(FITNESS_WINDOW)]["p50"]
    return GateOutcome(
        "Q2_oos_sprint_beats_i5",
        "PASS" if candidate > baseline else "FAIL",
        {
            "window_days": FITNESS_WINDOW,
            "candidate_oos_median": candidate,
            "baseline_oos_median": baseline,
            "gap_pp": (candidate - baseline) * 100.0,
            "candidate_oos_p95": candidate_oos["windows"][str(FITNESS_WINDOW)]["p95"],
            "baseline_oos_p95": baseline_oos["windows"][str(FITNESS_WINDOW)]["p95"],
        },
    )


def gate_q3_sprint_survival(oos_and_is_curve: np.ndarray, trade_returns: np.ndarray, seed: int) -> GateOutcome:
    """The sprint-specific risk gate: over the FULL measured span, how often does a 30-day
    window halve or decimate the account, and can the sleeve be bootstrapped to zero."""
    stats = window_statistics(oos_and_is_curve, FITNESS_WINDOW)
    rng = np.random.default_rng(seed)
    wipe = bootstrap_wipe_probability(np.asarray(trade_returns, dtype=float), rng, paths=MC_PATHS)
    reasons: list[str] = []
    if stats["prob_loss_over_50"] >= Q3_MAX_PROB_HALVING:
        reasons.append(f"P(30d loss>50%) {stats['prob_loss_over_50']:.4f} >= {Q3_MAX_PROB_HALVING}")
    if stats["prob_loss_over_90"] >= Q3_MAX_PROB_DECIMATION:
        reasons.append(f"P(30d loss>90%) {stats['prob_loss_over_90']:.4f} >= {Q3_MAX_PROB_DECIMATION}")
    if wipe >= Q3_MAX_WIPE_PROBABILITY:
        reasons.append(f"sleeve wipe prob {wipe:.4f} >= {Q3_MAX_WIPE_PROBABILITY}")
    return GateOutcome(
        "Q3_sprint_survival",
        "PASS" if not reasons else "FAIL",
        {
            "window_days": FITNESS_WINDOW,
            "n_windows": stats["n_windows"],
            "prob_loss_over_50": stats["prob_loss_over_50"],
            "max_prob_loss_over_50": Q3_MAX_PROB_HALVING,
            "prob_loss_over_90": stats["prob_loss_over_90"],
            "max_prob_loss_over_90": Q3_MAX_PROB_DECIMATION,
            "sleeve_wipe_probability": wipe,
            "max_wipe_probability": Q3_MAX_WIPE_PROBABILITY,
            "reasons": reasons,
        },
    )


def gate_q4_ruin(total_curve: np.ndarray, seed: int) -> GateOutcome:
    rng = np.random.default_rng(seed + 1)
    daily = _daily_returns(total_curve)
    if len(daily) < 3:
        return GateOutcome("Q4_ruin", "FAIL", {"reason": "insufficient daily history"})
    draws = rng.integers(0, len(daily), size=(MC_PATHS, len(daily)))
    finals = (TOTAL_CAPITAL * np.cumprod(1.0 + daily[draws], axis=1))[:, -1]
    ruin = float((finals < Q4_RUIN_FLOOR_USDT).mean())
    return GateOutcome(
        "Q4_ruin",
        "PASS" if ruin < Q4_MAX_RUIN_PROBABILITY else "FAIL",
        {
            "paths": MC_PATHS,
            "ruin_probability": ruin,
            "max_ruin_probability": Q4_MAX_RUIN_PROBABILITY,
            "ruin_floor_usdt": Q4_RUIN_FLOOR_USDT,
            "p05_usdt": float(np.percentile(finals, 5)),
            "median_usdt": float(np.median(finals)),
        },
    )


def gate_q7_short_horizon_consistency(candidate_oos: dict) -> GateOutcome:
    share = candidate_oos["windows"][str(FITNESS_WINDOW)]["positive_share"]
    n = candidate_oos["windows"][str(FITNESS_WINDOW)]["n_windows"]
    return GateOutcome(
        "Q7_short_horizon_consistency",
        "PASS" if share > Q7_MIN_POSITIVE_SHARE else "FAIL",
        {
            "window_days": FITNESS_WINDOW,
            "oos_positive_share": share,
            "minimum": Q7_MIN_POSITIVE_SHARE,
            "oos_windows_counted": n,
        },
    )


def evaluate_all_gates(
    seed_rows: list[dict],
    genome: Genome,
    total_curve: np.ndarray,
    daily_index: pd.DatetimeIndex,
    trade_returns: np.ndarray,
    candidate_oos_profile: dict,
    baseline_oos_profile: dict,
    min_notional_usdt: float,
    n_trades: int,
    seed: int,
) -> dict:
    outcomes = [
        gate_p1_method_validity(seed_rows),
        gate_q2_oos_sprint(candidate_oos_profile, baseline_oos_profile),
        gate_q3_sprint_survival(total_curve, trade_returns, seed),
        gate_q4_ruin(total_curve, seed),
        gate_p5_deflated_sharpe(total_curve, daily_index, trials=Q5_CUMULATIVE_TRIALS),
        gate_p6_executability(genome, min_notional_usdt, n_trades),
        gate_q7_short_horizon_consistency(candidate_oos_profile),
    ]
    renamed = {"P1_method_validity": "Q1_method_validity", "P5_deflated_sharpe": "Q5_deflated_sharpe", "P6_executability": "Q6_executability"}
    gates: dict[str, dict] = {}
    failures: list[str] = []
    for outcome in outcomes:
        name = renamed.get(outcome.name, outcome.name)
        gates[name] = {"status": outcome.status, **outcome.detail}
        if not outcome.passed:
            failures.append(name)
    return {
        "gates": gates,
        "overall": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "promoted": not failures,
    }
