# Wave-13 gate evaluation. S1 (delta-neutral/1x) / S2 (MC bootstrap) / S3 (block-shuffle
# 90-day MDD) / S4 ($45 executability) reuse the EXACT SAME numeric bars and simulation
# methodology as research/wave12_frontier/gates12.py (ruin P(<$50)<1%, MC p05>$100, block
# MDD p95<=10%, leg>=$5/gross<=$90) -- reimplemented locally rather than imported, matching
# gates12.py's own precedent ("이 모듈은 wave10의 것을 가져오는 대신 로컬로 재구현한다").
#
# S5 is NEW relative to gates12.py in two ways SPEC.md registers explicitly:
#   1. stress_multiplier is 3.0 (research.wave13_liquidity.engine13.STRESS_MULTIPLIER),
#      not wave12's 2.0 -- wave13's BASE cost is a single live snapshot (not an
#      already-conservative rank-tier assumption), so SPEC.md asks for a larger stress
#      multiplier to compensate for that snapshot risk (see SPEC.md's own 한계 section).
#   2. S5 additionally requires the STRESSED run's own block-shuffle MDD p95 <= 15% --
#      wave12's S5 only checked sign preservation; wave13's SPEC.md explicitly conjoins a
#      drawdown bar onto the stress gate ("고펀딩기 연환산 부호 유지 AND MDD p95 <= 15%").
#
# Promotion is also SIMPLER than gates12.py's: SPEC.md states "승격 = S1~S5 전부 PASS" with
# no additional "beat a baseline's return" clause (unlike wave12's U0-relative bar) --
# wave13's question is "does anything survive under measured cost," not relative ranking
# among L1-L5. reporting13.py separately picks the highest high-funding return AMONG
# promoted configs as "최종 추천," but that selection is a report-time convenience, not a
# gate condition.

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
from research.wave13_liquidity.configs13 import Wave13Config

MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SEED: Final = 20_260_722  # matches wave10/wave11/wave12's freeze-date seed convention

S2_RUIN_THRESHOLD_USDT: Final = 50.0
S2_RUIN_PROBABILITY_MAX: Final = 0.01
S2_P05_FLOOR_USDT: Final = 100.0
S3_BLOCK_MDD_P95_MAX: Final = 0.10
S5_BLOCK_MDD_P95_MAX: Final = 0.15  # SPEC.md S5: "MDD p95 <= 15%" (looser than S3's 10% -- this is the STRESSED run)

DSR_CUMULATIVE_TRIALS: Final = 76  # wave12's own disclosed 71 (research/wave12_frontier/gates12.py) + this wave's 5 new candidates (L1-L5)


def leg_usdt(config: Wave13Config) -> float:
    return config.leg_fraction * ACTIVE_CAPITAL


def gross_usdt(config: Wave13Config) -> float:
    return 2.0 * config.candidate.top_k * config.leg_fraction * ACTIVE_CAPITAL


def _daily_returns(equity: pd.Series) -> tuple[tuple[pd.Timestamp, ...], np.ndarray]:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave13 equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave13 equity returns contain invalid values")
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


def gate_s1_structure(config: Wave13Config) -> dict[str, Any]:
    """S1: delta-neutral 2-leg + 1x leverage. Delta-neutrality is structural (a single
    shared `weights` value drives both legs in engine13.py's loop, copied unchanged from
    wave10/11/12); tests/test_wave13.py regression-tests this. 1x leverage means
    gross <= ACTIVE_CAPITAL, checked here."""
    gross = gross_usdt(config)
    leverage_ok = gross <= ACTIVE_CAPITAL + 1e-9
    return {
        "delta_neutral_by_construction": True,
        "gross_usdt": gross,
        "leverage_multiplier_of_active_capital": gross / ACTIVE_CAPITAL,
        "leverage_1x_ok": leverage_ok,
        "status": "PASS" if leverage_ok else "FAIL",
    }


def gate_s2_mc(result: Wave10Result, seed_offset: int) -> dict[str, Any]:
    _, returns = _daily_returns(result.equity)
    mc = _simulate_mc(returns, SEED + seed_offset * 101)
    p05_ok = mc["p05"] > S2_P05_FLOOR_USDT
    ruin_ok = mc["ruin_probability"] < S2_RUIN_PROBABILITY_MAX
    return {**mc, "p05_floor_ok": p05_ok, "ruin_ok": ruin_ok, "status": "PASS" if (p05_ok and ruin_ok) else "FAIL"}


def gate_s3_block_mdd(result: Wave10Result, seed_offset: int) -> dict[str, Any]:
    timestamps, returns = _daily_returns(result.equity)
    block = _block_shuffle(timestamps, returns, SEED + seed_offset * 103)
    ok = block["mdd_p95"] <= S3_BLOCK_MDD_P95_MAX
    return {**block, "status": "PASS" if ok else "FAIL"}


def gate_s4_feasibility(config: Wave13Config, result: Wave10Result) -> dict[str, Any]:
    leg = leg_usdt(config)
    gross = gross_usdt(config)
    min_order_ok = leg >= MIN_ORDER_USDT
    gross_ok = gross <= ACTIVE_CAPITAL + 1e-9
    positioned_equity = result.equity[result.positions > 0.0]
    dynamic_min_leg = float(positioned_equity.min()) * config.leg_fraction if len(positioned_equity) else None
    dynamic_ok = None if dynamic_min_leg is None else bool(dynamic_min_leg >= MIN_ORDER_USDT)
    passed = bool(min_order_ok and gross_ok)
    return {
        "leg_usdt_nominal": leg,
        "gross_usdt_nominal": gross,
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
    stress_result: Wave10Result,
    seed_offset: int,
) -> dict[str, Any]:
    """S5: SPEC.md "실측슬리피지 x3에서 (a) 고펀딩기 연환산 부호 유지 AND (b) 블록셔플
    MDD p95 <= 15%" -- both legs of the AND are required to PASS, not just sign
    preservation (the wave12 precedent this gate is modeled on only checked sign)."""
    sign_ok = stress_high_funding is not None and stress_high_funding > 0.0
    stress_block = None
    mdd_ok = False
    try:
        timestamps, returns = _daily_returns(stress_result.equity)
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
    """가동률: fraction of days the strategy held any position."""
    if len(result.positions) == 0:
        return 0.0
    return float((result.positions.abs() > 0.0).mean())


def annualized_round_trips(result: Wave10Result) -> float:
    if len(result.equity) < 2:
        return 0.0
    span_days = max((pd.Timestamp(result.equity.index[-1]) - pd.Timestamp(result.equity.index[0])).total_seconds() / 86_400.0, 1.0)
    return float(len(result.trade_returns)) / (span_days / 365.0)


def deflated_sharpe_reference(result: Wave10Result) -> dict[str, Any] | None:
    """Reference-only DSR, corrected for DSR_CUMULATIVE_TRIALS=76. Never used for
    promotion (wave10/11/12's shared principle)."""
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
    """Simpler than gates12.PromotionCheck: SPEC.md's wave13 promotion rule is
    unconditional "S1-S5 전부 PASS", not "beat a baseline's return" -- `promoted` is
    therefore just an alias for the caller's own `overall == PASS`, kept as its own field
    (rather than inlined at every call site) so reporting13.py has one stable attribute
    name to read regardless of which wave's gate report it's rendering."""

    high_funding_mean_annualized_return: float | None
    promoted: bool


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
    config: Wave13Config,
    result: Wave10Result,
    stress_result: Wave10Result,
    seed_offset: int,
) -> GateReport:
    gate_s1 = gate_s1_structure(config)
    gate_s2 = gate_s2_mc(result, seed_offset)
    gate_s3 = gate_s3_block_mdd(result, seed_offset)
    gate_s4 = gate_s4_feasibility(config, result)
    regime = regime_breakdown(result)
    stress_regime = regime_breakdown(stress_result)
    gate_s5 = gate_s5_stress(
        regime.get("high_funding_mean_annualized_return"), stress_regime.get("high_funding_mean_annualized_return"), stress_result, seed_offset
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
    "gross_usdt",
    "leg_usdt",
    "promotion_check",
    "utilization",
]
