# Wave-15 gate evaluation -- S1-S5, same numeric bars and simulation methodology as
# research/wave13_liquidity/gates13.py (ruin P(<$50)<1%, MC p05>$100, block MDD p95<=10%,
# leg>=$5/gross<=$90, S5 stress x3 sign+MDD<=15%), reimplemented locally rather than imported
# -- continuing gates13.py's own documented precedent of reimplementing rather than
# cross-importing wave-specific gate logic across waves.
#
# Two changes vs gates13.py, both required by wave15 having candidates that don't share one
# config dataclass and one bar frequency:
#   1. Every function takes explicit `leg_usdt`/`gross_usdt`/`delta_neutral` instead of a
#      Wave13Config -- A1-A3/B1/B2/C1/D1 have no shared config type, so gates15 is config-
#      agnostic. `delta_neutral` is a new REQUIRED, explicit disclosure (SPEC.md: "S1
#      구조(델타중립 여부는 후보별 명시)") -- it does NOT gate PASS/FAIL (leverage feasibility
#      still does, unchanged), it is recorded so B2's directional exposure and D1's
#      dollar-neutral-pair (a different risk shape than spot-perp basis neutrality) are never
#      silently reported as if they were the same delta-neutral guarantee wave10-14 built.
#   2. `resample_daily`: A1-A3 trade hourly bars (~60k rows over the backtest span, vs ~2.2k
#      for every daily candidate). Bootstrapping 10,000 MC paths at native hourly resolution
#      was benchmarked here at ~40s/call before writing this flag -- 6 calls (3 candidates x
#      base+stress) would cost several minutes purely on RNG/array work with no change in
#      economic content over resampling first. When True, gate_s2_mc/gate_s3_block_mdd/
#      gate_s5_stress bootstrap over the equity curve's DAILY closes (`.resample("1D").last()`)
#      instead of every hourly bar -- this changes ONLY the bootstrap's observation grid, not
#      the reported regime_breakdown annualized returns (those always use the full-resolution
#      equity elsewhere, in run_wave15.py, never resampled).

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.validation.deep_stats import DeepValidationError, TimedValue, deflated_sharpe
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, RESERVE_FRACTION, TOTAL_CAPITAL, Wave10Result
from research.wave10_carry100.regime import regime_breakdown

MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SEED: Final = 20_260_722  # matches wave10-13's freeze-date seed convention

S2_RUIN_THRESHOLD_USDT: Final = 50.0
S2_RUIN_PROBABILITY_MAX: Final = 0.01
S2_P05_FLOOR_USDT: Final = 100.0
S3_BLOCK_MDD_P95_MAX: Final = 0.10
S5_BLOCK_MDD_P95_MAX: Final = 0.15
STRESS_MULTIPLIER: Final = 3.0

DSR_CUMULATIVE_TRIALS: Final = 91  # wave14's own disclosed 84 (research/wave14_multivenue/gates14.py: wave13's 76 + M0-M7's 8) + this wave's 7 ACTUAL candidates (A1-A3,B1,B2,C1,D1).
# NOTE: SPEC.md's own text says "누적 92회" (84+8) -- an artifact of the SAME 8-vs-7 mismatch
# configs15.py's docstring documents (SPEC.md's header says 8 candidates, its table lists 7).
# 91, not 92, is used here because only 7 wave15 backtests actually exist to count -- DSR
# trial-counting should track what was ACTUALLY run, not a stale header number.


def leg_usdt_of(leg_fraction: float) -> float:
    return leg_fraction * ACTIVE_CAPITAL


def gross_usdt_of(leg_fraction: float, top_k: int = 1) -> float:
    return 2.0 * top_k * leg_fraction * ACTIVE_CAPITAL


def _bar_returns(equity: pd.Series, resample_daily: bool) -> tuple[tuple[pd.Timestamp, ...], np.ndarray]:
    series = equity.resample("1D").last().dropna() if resample_daily else equity
    clean = series.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave15 equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave15 equity returns contain invalid values")
    timestamps = tuple(pd.Timestamp(ts) for ts in clean.index[1:])
    return timestamps, returns


def _simulate_mc(returns: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    finals = np.empty(MC_PATHS, dtype=float)
    chunk_size = 250
    for start in range(0, MC_PATHS, chunk_size):
        stop = min(start + chunk_size, MC_PATHS)
        samples = rng.choice(returns, size=(stop - start, returns.size), replace=True)
        growth = np.prod(1.0 + np.clip(samples, -0.999999, None), axis=1)
        finals[start:stop] = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * growth
    return {
        "p05": float(np.quantile(finals, 0.05)),
        "ruin_probability": float(np.mean(finals < S2_RUIN_THRESHOLD_USDT)),
        "mean": float(np.mean(finals)),
        "median": float(np.median(finals)),
        "paths": MC_PATHS,
    }


def _blocks(timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray) -> tuple[np.ndarray, ...]:
    anchor = timestamps[0]
    grouped: dict[int, list[float]] = {}
    for timestamp, value in zip(timestamps, returns):
        index = (timestamp - anchor).days // BLOCK_DAYS
        grouped.setdefault(index, []).append(float(value))
    return tuple(np.asarray(grouped[key], dtype=float) for key in sorted(grouped))


def _mdd(values: np.ndarray) -> float:
    curve = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * np.cumprod(1.0 + np.clip(values, -0.999999, None))
    peaks = np.maximum.accumulate(np.concatenate(([TOTAL_CAPITAL], curve)))
    return float(np.max(1.0 - curve / peaks[1:]))


def _block_shuffle(timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray, seed: int) -> dict[str, float]:
    blocks = _blocks(timestamps, returns)
    rng = np.random.default_rng(seed)
    mdds = np.empty(BLOCK_PATHS, dtype=float)
    finals = np.empty(BLOCK_PATHS, dtype=float)
    for index in range(BLOCK_PATHS):
        path = np.concatenate([blocks[item] for item in rng.permutation(len(blocks))])
        mdds[index] = _mdd(path)
        finals[index] = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * np.prod(1.0 + np.clip(path, -0.999999, None))
    return {
        "block_days": BLOCK_DAYS,
        "block_count": len(blocks),
        "paths": BLOCK_PATHS,
        "mdd_p95": float(np.quantile(mdds, 0.95)),
        "final_p05": float(np.quantile(finals, 0.05)),
    }


def gate_s1_structure(leg_usdt: float, gross_usdt: float, delta_neutral: bool, note: str) -> dict[str, Any]:
    leverage_ok = gross_usdt <= ACTIVE_CAPITAL + 1e-9
    return {
        "delta_neutral_by_construction": delta_neutral,
        "structure_note": note,
        "gross_usdt": gross_usdt,
        "leverage_multiplier_of_active_capital": gross_usdt / ACTIVE_CAPITAL,
        "leverage_1x_ok": leverage_ok,
        "status": "PASS" if leverage_ok else "FAIL",
    }


def gate_s2_mc(equity: pd.Series, seed_offset: int, resample_daily: bool = False) -> dict[str, Any]:
    _, returns = _bar_returns(equity, resample_daily)
    mc = _simulate_mc(returns, SEED + seed_offset * 101)
    p05_ok = mc["p05"] > S2_P05_FLOOR_USDT
    ruin_ok = mc["ruin_probability"] < S2_RUIN_PROBABILITY_MAX
    return {**mc, "resampled_daily_for_bootstrap": resample_daily, "p05_floor_ok": p05_ok, "ruin_ok": ruin_ok, "status": "PASS" if (p05_ok and ruin_ok) else "FAIL"}


def gate_s3_block_mdd(equity: pd.Series, seed_offset: int, resample_daily: bool = False) -> dict[str, Any]:
    timestamps, returns = _bar_returns(equity, resample_daily)
    block = _block_shuffle(timestamps, returns, SEED + seed_offset * 103)
    ok = block["mdd_p95"] <= S3_BLOCK_MDD_P95_MAX
    return {**block, "resampled_daily_for_bootstrap": resample_daily, "status": "PASS" if ok else "FAIL"}


def gate_s4_feasibility(leg_usdt: float, gross_usdt: float, result: Wave10Result) -> dict[str, Any]:
    min_order_ok = leg_usdt >= MIN_ORDER_USDT
    gross_ok = gross_usdt <= ACTIVE_CAPITAL + 1e-9
    positioned_equity = result.equity[result.positions > 0.0]
    dynamic_min_leg = float(positioned_equity.min()) * (leg_usdt / ACTIVE_CAPITAL) if len(positioned_equity) else None
    dynamic_ok = None if dynamic_min_leg is None else bool(dynamic_min_leg >= MIN_ORDER_USDT)
    passed = bool(min_order_ok and gross_ok)
    return {
        "leg_usdt_nominal": leg_usdt,
        "gross_usdt_nominal": gross_usdt,
        "min_order_usdt": MIN_ORDER_USDT,
        "min_order_feasible": min_order_ok,
        "gross_exposure_feasible": gross_ok,
        "dynamic_min_leg_usdt": dynamic_min_leg,
        "dynamic_min_order_feasible": dynamic_ok,
        "status": "PASS" if passed else "FAIL",
    }


def gate_s5_stress(
    base_high_funding: float | None,
    stress_high_funding: float | None,
    stress_equity: pd.Series,
    seed_offset: int,
    resample_daily: bool = False,
) -> dict[str, Any]:
    sign_ok = stress_high_funding is not None and stress_high_funding > 0.0
    stress_block = None
    mdd_ok = False
    try:
        timestamps, returns = _bar_returns(stress_equity, resample_daily)
        stress_block = _block_shuffle(timestamps, returns, SEED + seed_offset * 107)
        mdd_ok = stress_block["mdd_p95"] <= S5_BLOCK_MDD_P95_MAX
    except ValueError:
        stress_block = None
        mdd_ok = False
    ok = sign_ok and mdd_ok
    return {
        "base_high_funding_annualized": base_high_funding,
        "stress_high_funding_annualized": stress_high_funding,
        "sign_preserved": sign_ok,
        "stress_block_mdd_p95": stress_block["mdd_p95"] if stress_block is not None else None,
        "stress_mdd_ok": mdd_ok,
        "stress_mdd_p95_max": S5_BLOCK_MDD_P95_MAX,
        "status": "PASS" if ok else "FAIL",
    }


def utilization(result: Wave10Result) -> float:
    if len(result.positions) == 0:
        return 0.0
    return float((result.positions.abs() > 0.0).mean())


def annualized_round_trips(result: Wave10Result) -> float:
    if len(result.equity) < 2:
        return 0.0
    span_days = max((pd.Timestamp(result.equity.index[-1]) - pd.Timestamp(result.equity.index[0])).total_seconds() / 86_400.0, 1.0)
    return float(len(result.trade_returns)) / (span_days / 365.0)


def deflated_sharpe_reference(result: Wave10Result) -> dict[str, Any] | None:
    equity = result.equity.dropna()
    if len(equity) < 4:
        return None
    timed = tuple(TimedValue(pd.Timestamp(idx).to_pydatetime(), float(value)) for idx, value in equity.items())
    try:
        dsr = deflated_sharpe(timed, trials=DSR_CUMULATIVE_TRIALS)
    except DeepValidationError:
        return None
    return {"score": dsr.score, "probability": dsr.probability, "trials": dsr.trials, "observed_sharpe": dsr.observed_sharpe}


FAILURE_LEVERAGE = "레버리지/gross"
FAILURE_MIN_ORDER = "최소주문"
FAILURE_PRINCIPAL = "원금보존(MC)"
FAILURE_DRAWDOWN = "MDD초과"
FAILURE_STRESS_SIGN = "스트레스부호반전"
FAILURE_STRESS_MDD = "스트레스MDD초과"


def _classify_failures(s1: dict, s2: dict, s3: dict, s4: dict, s5: dict) -> tuple[str, ...]:
    reasons: list[str] = []
    if s1["status"] == "FAIL":
        reasons.append(FAILURE_LEVERAGE)
    if s2["status"] == "FAIL":
        reasons.append(FAILURE_PRINCIPAL)
    if s3["status"] == "FAIL":
        reasons.append(FAILURE_DRAWDOWN)
    if s4["status"] == "FAIL":
        if not s4["min_order_feasible"]:
            reasons.append(FAILURE_MIN_ORDER)
        if not s4["gross_exposure_feasible"]:
            reasons.append(FAILURE_LEVERAGE)
    if s5["status"] == "FAIL":
        if not s5["sign_preserved"]:
            reasons.append(FAILURE_STRESS_SIGN)
        if not s5["stress_mdd_ok"]:
            reasons.append(FAILURE_STRESS_MDD)
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return tuple(seen)


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    high_funding_mean_annualized_return: float | None
    promoted_gate_only: bool  # S1-S5 all PASS; the L4-beat clause is applied separately in run_wave15.py (needs the L4 reference, which gates15.py has no access to)


def promotion_check(regime: dict[str, Any], overall_status: str) -> PromotionCheck:
    high_funding = regime.get("high_funding_mean_annualized_return")
    return PromotionCheck(high_funding, overall_status == "PASS")


@dataclass(frozen=True, slots=True)
class GateReport:
    gate_s1: dict[str, Any]
    gate_s2: dict[str, Any]
    gate_s3: dict[str, Any]
    gate_s4: dict[str, Any]
    gate_s5: dict[str, Any]
    overall: str
    failure_reasons: tuple[str, ...]
    promotion: PromotionCheck


def evaluate_gates(
    result: Wave10Result,
    stress_result: Wave10Result,
    leg_usdt: float,
    gross_usdt: float,
    delta_neutral: bool,
    structure_note: str,
    seed_offset: int,
    resample_daily: bool = False,
) -> GateReport:
    gate_s1 = gate_s1_structure(leg_usdt, gross_usdt, delta_neutral, structure_note)
    gate_s2 = gate_s2_mc(result.equity, seed_offset, resample_daily)
    gate_s3 = gate_s3_block_mdd(result.equity, seed_offset, resample_daily)
    gate_s4 = gate_s4_feasibility(leg_usdt, gross_usdt, result)
    regime = regime_breakdown(result)
    stress_regime = regime_breakdown(stress_result)
    gate_s5 = gate_s5_stress(
        regime.get("high_funding_mean_annualized_return"),
        stress_regime.get("high_funding_mean_annualized_return"),
        stress_result.equity,
        seed_offset,
        resample_daily,
    )
    all_pass = all(gate["status"] == "PASS" for gate in (gate_s1, gate_s2, gate_s3, gate_s4, gate_s5))
    overall = "PASS" if all_pass else "FAIL"
    promotion = promotion_check(regime, overall)
    reasons = _classify_failures(gate_s1, gate_s2, gate_s3, gate_s4, gate_s5)
    return GateReport(gate_s1, gate_s2, gate_s3, gate_s4, gate_s5, overall, reasons, promotion)


def gate_report_payload(report: GateReport) -> dict[str, Any]:
    return {
        "gate_s1": report.gate_s1,
        "gate_s2": report.gate_s2,
        "gate_s3": report.gate_s3,
        "gate_s4": report.gate_s4,
        "gate_s5": report.gate_s5,
        "overall": report.overall,
        "failure_reasons": list(report.failure_reasons),
        "promotion": asdict(report.promotion),
    }


__all__ = [
    "DSR_CUMULATIVE_TRIALS",
    "S2_P05_FLOOR_USDT",
    "S2_RUIN_PROBABILITY_MAX",
    "S2_RUIN_THRESHOLD_USDT",
    "S3_BLOCK_MDD_P95_MAX",
    "S5_BLOCK_MDD_P95_MAX",
    "STRESS_MULTIPLIER",
    "GateReport",
    "PromotionCheck",
    "annualized_round_trips",
    "deflated_sharpe_reference",
    "evaluate_gates",
    "gate_report_payload",
    "gate_s1_structure",
    "gate_s2_mc",
    "gate_s3_block_mdd",
    "gate_s4_feasibility",
    "gate_s5_stress",
    "gross_usdt_of",
    "leg_usdt_of",
    "promotion_check",
    "utilization",
]
