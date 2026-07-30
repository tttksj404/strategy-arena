# Wave-30 gates P1-P7 exactly as frozen in SPEC.md.
#
# The Deflated Sharpe calculation is IMPORTED from research.validation.deep_stats rather than
# reimplemented, so wave30's P5 number is produced by the same Bailey-Lopez de Prado code that
# produced every DSR figure already on the strategy card (and the same code
# research/validation/tests/test_deep_validate.py pins). Only the `trials` count differs, and
# that count is the cumulative one SPEC.md registers.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.validation.deep_stats import TimedValue, deflated_sharpe
from research.wave30_qd.engine30 import TOTAL_CAPITAL
from research.wave30_qd.fitness30 import bootstrap_wipe_probability
from research.wave30_qd.genome30 import LEV_CAP, STOP_BAND_MARGIN, Genome

# Frozen gate thresholds (SPEC.md)
P1_MIN_SEED_WINS: Final = 4
P2_BASELINE_LABEL: Final = "I5 $100-basis OOS"
P3_RUIN_FLOOR_USDT: Final = 50.0
P3_MAX_RUIN_PROBABILITY: Final = 0.01
P3_MAX_WIPE_PROBABILITY: Final = 0.05
P4_MAX_BLOCK_MDD_P95: Final = 0.35
P4_BLOCK_DAYS: Final = 90
P4_PATHS: Final = 1_000
P5_CUMULATIVE_TRIALS: Final = 213_621
P6_MIN_ORDER_USDT: Final = 5.0
P7_WORST_YEAR_FLOOR: Final = -0.20
MC_PATHS: Final = 10_000


@dataclass
class GateOutcome:
    name: str
    status: str  # PASS / FAIL
    detail: dict

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _daily_returns(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    if len(curve) < 2:
        return np.zeros(0)
    previous = curve[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(previous > 0, curve[1:] / np.maximum(previous, 1e-12) - 1.0, 0.0)
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def gate_p1_method_validity(seed_rows: list[dict]) -> GateOutcome:
    """(a) evolved best fitness beats the matched-budget random control, and (b) the archive
    covers more behaviour cells than the control's archive -- both in >= 4 of 5 seeds.

    Part (b) is new relative to wave21/23/24, which only ever compared best fitness. Coverage
    is what distinguishes an ILLUMINATING search from a lucky one: an algorithm can win on
    best fitness by accident, but it cannot map more of the behaviour space by accident.
    """
    fitness_wins = sum(1 for row in seed_rows if row["evolved_best_fitness"] > row["random_best_fitness"])
    coverage_wins = sum(1 for row in seed_rows if row["archive_coverage"] > row["random_coverage"])
    ok = fitness_wins >= P1_MIN_SEED_WINS and coverage_wins >= P1_MIN_SEED_WINS
    return GateOutcome(
        "P1_method_validity",
        "PASS" if ok else "FAIL",
        {
            "seeds": len(seed_rows),
            "fitness_wins": fitness_wins,
            "coverage_wins": coverage_wins,
            "required_wins": P1_MIN_SEED_WINS,
            "per_seed": [
                {
                    "seed": row["seed"],
                    "evolved_best_fitness": row["evolved_best_fitness"],
                    "random_best_fitness": row["random_best_fitness"],
                    "fitness_win": bool(row["evolved_best_fitness"] > row["random_best_fitness"]),
                    "archive_coverage": row["archive_coverage"],
                    "random_coverage": row["random_coverage"],
                    "coverage_win": bool(row["archive_coverage"] > row["random_coverage"]),
                    "qd_score": row["qd_score"],
                    "random_qd_score": row["random_qd_score"],
                }
                for row in seed_rows
            ],
        },
    )


def gate_p2_oos(final: dict) -> GateOutcome:
    candidate = final["oos"]["annualized"]
    baseline = final["baseline_oos"]["annualized"]
    return GateOutcome(
        "P2_oos_beats_i5",
        "PASS" if candidate > baseline else "FAIL",
        {
            "candidate_oos_annualized": candidate,
            "baseline_oos_annualized": baseline,
            "baseline_label": P2_BASELINE_LABEL,
            "gap_pp": (candidate - baseline) * 100.0,
            "oos_window": {"start": final["oos_start_day"], "days": final["oos"]["days"]},
            "candidate_oos_start_usdt": final["oos"]["start_usdt"],
            "candidate_oos_end_usdt": final["oos"]["end_usdt"],
        },
    )


def gate_p3_survival(total_curve: np.ndarray, trade_returns: np.ndarray, seed: int) -> GateOutcome:
    """MC bootstrap of the TOTAL system's daily returns for ruin, plus the sleeve's own trade
    returns for a wipe. Two separate questions: "can the $100 fall below $50" and "can the
    leveraged sleeve go to zero" -- a system can easily pass the first while failing the
    second if the sleeve is small, and SPEC.md gates both."""
    rng = np.random.default_rng(seed)
    daily = _daily_returns(total_curve)
    if len(daily) < 3:
        return GateOutcome("P3_survival", "FAIL", {"reason": "insufficient daily history"})

    draws = rng.integers(0, len(daily), size=(MC_PATHS, len(daily)))
    paths = TOTAL_CAPITAL * np.cumprod(1.0 + daily[draws], axis=1)
    finals = paths[:, -1]
    ruin_probability = float((finals < P3_RUIN_FLOOR_USDT).mean())

    wipe_probability = bootstrap_wipe_probability(np.asarray(trade_returns, dtype=float), rng, paths=MC_PATHS)

    ok = ruin_probability < P3_MAX_RUIN_PROBABILITY and wipe_probability < P3_MAX_WIPE_PROBABILITY
    return GateOutcome(
        "P3_survival",
        "PASS" if ok else "FAIL",
        {
            "paths": MC_PATHS,
            "p05_usdt": float(np.percentile(finals, 5)),
            "median_usdt": float(np.median(finals)),
            "ruin_probability": ruin_probability,
            "ruin_floor_usdt": P3_RUIN_FLOOR_USDT,
            "max_ruin_probability": P3_MAX_RUIN_PROBABILITY,
            "sleeve_wipe_probability": wipe_probability,
            "max_wipe_probability": P3_MAX_WIPE_PROBABILITY,
        },
    )


def gate_p4_block_shuffle(total_curve: np.ndarray, seed: int) -> GateOutcome:
    """Reorder complete 90-day blocks of daily returns 1,000 times. This asks whether the
    observed drawdown depended on the historical ORDER of regimes -- the same question
    gate_s3/gate_g4 asked in earlier waves, on the same block length."""
    daily = _daily_returns(total_curve)
    if len(daily) < P4_BLOCK_DAYS * 2:
        return GateOutcome("P4_block_shuffle", "FAIL", {"reason": "history shorter than two blocks"})
    n_blocks = len(daily) // P4_BLOCK_DAYS
    trimmed = daily[: n_blocks * P4_BLOCK_DAYS].reshape(n_blocks, P4_BLOCK_DAYS)
    rng = np.random.default_rng(seed)
    mdds = np.empty(P4_PATHS)
    finals = np.empty(P4_PATHS)
    for index in range(P4_PATHS):
        path = trimmed[rng.permutation(n_blocks)].reshape(-1)
        curve = TOTAL_CAPITAL * np.cumprod(1.0 + path)
        peak = np.maximum.accumulate(np.concatenate(([TOTAL_CAPITAL], curve)))[1:]
        mdds[index] = float(np.max(1.0 - curve / np.maximum(peak, 1e-12)))
        finals[index] = curve[-1]
    mdd_p95 = float(np.percentile(mdds, 95))
    return GateOutcome(
        "P4_block_shuffle",
        "PASS" if mdd_p95 <= P4_MAX_BLOCK_MDD_P95 else "FAIL",
        {
            "block_days": P4_BLOCK_DAYS,
            "block_count": n_blocks,
            "paths": P4_PATHS,
            "mdd_p95": mdd_p95,
            "max_mdd_p95": P4_MAX_BLOCK_MDD_P95,
            "final_p05_usdt": float(np.percentile(finals, 5)),
        },
    )


def gate_p5_deflated_sharpe(total_curve: np.ndarray, daily_index: pd.DatetimeIndex) -> GateOutcome:
    series = [
        TimedValue(timestamp=daily_index[i].to_pydatetime(), value=float(total_curve[i]))
        for i in range(len(total_curve))
    ]
    try:
        result = deflated_sharpe(series, trials=P5_CUMULATIVE_TRIALS)
    except Exception as error:  # deep_stats raises DeepValidationError on degenerate input
        return GateOutcome("P5_deflated_sharpe", "FAIL", {"reason": f"{type(error).__name__}: {error}"})
    return GateOutcome(
        "P5_deflated_sharpe",
        "PASS" if result.score > 0.0 else "FAIL",
        {
            "score": result.score,
            "probability": result.probability,
            "observed_sharpe": result.observed_sharpe,
            "benchmark_sharpe": result.benchmark_sharpe,
            "n_days": result.n_days,
            "trials": P5_CUMULATIVE_TRIALS,
            "skew": result.skew,
            "kurtosis": result.kurtosis,
        },
    )


def gate_p6_executability(genome: Genome, min_notional_usdt: float, n_trades: int) -> GateOutcome:
    reasons: list[str] = []
    if not np.isfinite(min_notional_usdt):
        reasons.append("no trade was ever opened, so executability is undetermined")
    elif min_notional_usdt < P6_MIN_ORDER_USDT:
        reasons.append(f"smallest notional ${min_notional_usdt:.2f} < ${P6_MIN_ORDER_USDT:.2f} minimum order")
    if genome.leverage > LEV_CAP + 1e-9:
        reasons.append(f"leverage {genome.leverage:.4f}x exceeds {LEV_CAP}x cap")
    if genome.stop_pct > STOP_BAND_MARGIN * genome.liquidation_band + 1e-12:
        reasons.append("stop is not interior to the liquidation band")
    return GateOutcome(
        "P6_executability",
        "PASS" if not reasons else "FAIL",
        {
            "min_notional_usdt": min_notional_usdt,
            "min_order_usdt": P6_MIN_ORDER_USDT,
            "leverage": genome.leverage,
            "leverage_cap": LEV_CAP,
            "stop_pct": genome.stop_pct,
            "liquidation_band": genome.liquidation_band,
            "stop_band_limit": STOP_BAND_MARGIN * genome.liquidation_band,
            "n_trades": n_trades,
            "reasons": reasons,
        },
    )


def gate_p7_worst_year(total_curve: np.ndarray, daily_index: pd.DatetimeIndex, baseline_curve: np.ndarray) -> GateOutcome:
    frame = pd.DataFrame(
        {"total": np.asarray(total_curve, dtype=float), "baseline": np.asarray(baseline_curve, dtype=float)},
        index=pd.DatetimeIndex(daily_index),
    )
    rows = []
    for year, group in frame.groupby(frame.index.year):
        if len(group) < 2:
            continue
        rows.append(
            {
                "year": int(year),
                "candidate_return": float(group["total"].iloc[-1] / group["total"].iloc[0] - 1.0),
                "baseline_return": float(group["baseline"].iloc[-1] / group["baseline"].iloc[0] - 1.0),
            }
        )
    if not rows:
        return GateOutcome("P7_worst_year", "FAIL", {"reason": "no complete calendar year"})
    worst = min(rows, key=lambda row: row["candidate_return"])
    return GateOutcome(
        "P7_worst_year",
        "PASS" if worst["candidate_return"] >= P7_WORST_YEAR_FLOOR else "FAIL",
        {
            "worst_year": worst["year"],
            "worst_year_return": worst["candidate_return"],
            "floor": P7_WORST_YEAR_FLOOR,
            "baseline_same_year": worst["baseline_return"],
            "years_worse_than_baseline": sum(1 for row in rows if row["candidate_return"] < row["baseline_return"]),
            "per_year": rows,
        },
    )


def evaluate_all_gates(
    seed_rows: list[dict],
    final: dict,
    genome: Genome,
    total_curve: np.ndarray,
    baseline_curve: np.ndarray,
    daily_index: pd.DatetimeIndex,
    trade_returns: np.ndarray,
    seed: int,
) -> dict:
    outcomes = [
        gate_p1_method_validity(seed_rows),
        gate_p2_oos(final),
        gate_p3_survival(total_curve, trade_returns, seed),
        gate_p4_block_shuffle(total_curve, seed),
        gate_p5_deflated_sharpe(total_curve, daily_index),
        gate_p6_executability(genome, final["min_notional_usdt"], final["n_trades_full"]),
        gate_p7_worst_year(total_curve, daily_index, baseline_curve),
    ]
    failures = [outcome.name for outcome in outcomes if not outcome.passed]
    return {
        "gates": {outcome.name: {"status": outcome.status, **outcome.detail} for outcome in outcomes},
        "overall": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "promoted": not failures,
    }
