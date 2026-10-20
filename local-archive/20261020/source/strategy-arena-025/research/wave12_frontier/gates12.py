# Wave-12 gate evaluation: S1 (delta-neutral/1x) / S2 (MC bootstrap) / S3 (block-shuffle
# 90-day MDD) / S4 ($100 executability) are "승계" (carried over) from
# research/wave11_yield/gates_y.py verbatim -- SPEC.md: "게이트 (wave-11 S1~S4 승계)",
# same literal numeric bars (ruin P(<$50)<1%, MC p05>$100, block MDD p95<=10%, leg>=$5 /
# gross<=$90). S5 is new (SPEC.md: "슬리피지 ×2에서 고펀딩기 연환산 부호 유지").
#
# MC bootstrap and block-shuffle METHODOLOGY (path counts, $100/$90/$10 capital basis, 90
# -day block length, resample-with-replacement) is reimplemented locally rather than
# imported from gates_y.py's own module-private helpers -- the same boundary convention
# gates_y.py itself used relative to research/wave10_carry100/gates.py ("이 모듈은 wave10의
# 것을 가져오는 대신 로컬로 재구현한다").
#
# Promotion bar (the one thing that is NOT carried over unchanged): wave11's promotion
# bar was a frozen literal number (C1's own SPEC.md-registered figure). Wave12's SPEC.md
# instead requires beating U0's OWN tiered-cost-model result ("고펀딩기 연환산 > U0(계층
# 비용 기준 Y4 재산출값)") -- a run-time comparison, not a constant, because the whole
# point of this wave is that U0 itself must be recomputed under the new cost model before
# anything can be compared against it. promotion_check therefore takes the bar as a
# parameter (U0's own already-evaluated high-funding annualized return) rather than
# reading a module constant; research/wave12_frontier/run_wave12.py's gates stage runs U0
# first and threads its regime figure into every other config's promotion_check call. U0
# itself has no bar to beat (it defines the baseline) -- passed as None, and
# PromotionCheck.is_baseline records that distinctly from "attempted promotion and failed".

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
from research.wave12_frontier.configs12 import Wave12Config

MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SEED: Final = 20_260_722  # matches wave10/wave11's freeze-date seed convention

S2_RUIN_THRESHOLD_USDT: Final = 50.0
S2_RUIN_PROBABILITY_MAX: Final = 0.01  # SPEC: "파산확률 P(최종<$50) < 1%" (wave11 S2, carried over)
S2_P05_FLOOR_USDT: Final = 100.0  # SPEC: "MC 1e4 p05 > $100"
S3_BLOCK_MDD_P95_MAX: Final = 0.10  # SPEC: "블록셔플 90일 MDD p95 <= 10%" (wave11 S3, carried over)

DSR_CUMULATIVE_TRIALS: Final = 71  # SPEC: "누적 시행 71회 기준 DSR 보정 표기"


def leg_usdt(config: Wave12Config) -> float:
    return config.leg_fraction * ACTIVE_CAPITAL


def gross_usdt(config: Wave12Config) -> float:
    return 2.0 * config.candidate.top_k * config.leg_fraction * ACTIVE_CAPITAL


def _daily_returns(equity: pd.Series) -> tuple[tuple[pd.Timestamp, ...], np.ndarray]:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave12 equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave12 equity returns contain invalid values")
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


def gate_s1_structure(config: Wave12Config) -> dict[str, Any]:
    """S1: delta-neutral 2-leg + 1x leverage. Delta-neutrality is structural (a single
    shared `weights` value drives both legs in research/wave12_frontier/engine12.py's
    loop, unchanged from wave10/wave11 -- see that module's docstring); regression-tested
    in tests/test_wave12.py. 1x leverage means gross <= ACTIVE_CAPITAL, checked here."""
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


def gate_s4_feasibility(config: Wave12Config, result: Wave10Result) -> dict[str, Any]:
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


def gate_s5_stress(base_high_funding: float | None, stress_high_funding: float | None) -> dict[str, Any]:
    """S5 (new): SPEC.md "슬리피지 ×2에서 고펀딩기 연환산 부호 유지" -- re-run the exact
    same config with research.wave12_frontier.engine12.STRESS_MULTIPLIER (2.0) fed into
    the tiered cost model (doubles slippage only, never the maker fee -- see
    costs_tiered.cost_rate_from_bp), then require the stressed high-funding-regime
    annualized return to still be positive. Promotion separately requires the BASE run to
    beat U0, so the only outcome this gate actually discriminates in practice is "stayed
    positive under 2x slippage" vs "flipped negative" -- literal, unconditional
    `stress_high_funding > 0` rather than a base-relative sign comparison."""
    ok = stress_high_funding is not None and stress_high_funding > 0.0
    return {
        "base_high_funding_annualized": base_high_funding,
        "stress_high_funding_annualized": stress_high_funding,
        "sign_preserved": ok,
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
    """Reference-only DSR, corrected for DSR_CUMULATIVE_TRIALS=71 per SPEC's multiple
    -testing disclosure requirement ("누적 시행 71회 기준 DSR 보정 표기")."""
    equity = result.equity.dropna()
    if len(equity) < 4:
        return None
    timed = tuple(TimedValue(pd.Timestamp(idx).to_pydatetime(), float(value)) for idx, value in equity.items())
    try:
        dsr = deflated_sharpe(timed, trials=DSR_CUMULATIVE_TRIALS)
    except DeepValidationError:
        return None
    return {"score": dsr.score, "probability": dsr.probability, "trials": dsr.trials, "observed_sharpe": dsr.observed_sharpe}


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    high_funding_mean_annualized_return: float | None
    high_funding_bar: float | None  # None for U0 itself -- the baseline has nothing to beat
    high_funding_ok: bool
    is_baseline: bool
    promoted: bool


def promotion_check(regime: dict[str, Any], high_funding_bar: float | None) -> PromotionCheck:
    high_funding = regime.get("high_funding_mean_annualized_return")
    if high_funding_bar is None:
        return PromotionCheck(high_funding, None, False, True, False)
    ok = high_funding is not None and high_funding > high_funding_bar
    return PromotionCheck(high_funding, high_funding_bar, bool(ok), False, bool(ok))


FAILURE_LEVERAGE = "레버리지/gross"
FAILURE_MIN_ORDER = "최소주문"
FAILURE_PRINCIPAL = "원금보존(MC)"
FAILURE_DRAWDOWN = "MDD초과"
FAILURE_STRESS = "스트레스부호반전"


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
        reasons.append(FAILURE_STRESS)
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return tuple(seen)


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
    config: Wave12Config,
    result: Wave10Result,
    stress_result: Wave10Result,
    seed_offset: int,
    high_funding_bar: float | None,
) -> GateReport:
    gate_s1 = gate_s1_structure(config)
    gate_s2 = gate_s2_mc(result, seed_offset)
    gate_s3 = gate_s3_block_mdd(result, seed_offset)
    gate_s4 = gate_s4_feasibility(config, result)
    regime = regime_breakdown(result)
    stress_regime = regime_breakdown(stress_result)
    gate_s5 = gate_s5_stress(regime.get("high_funding_mean_annualized_return"), stress_regime.get("high_funding_mean_annualized_return"))
    all_pass = all(gate["status"] == "PASS" for gate in (gate_s1, gate_s2, gate_s3, gate_s4, gate_s5))
    promotion = promotion_check(regime, high_funding_bar)
    overall = "PASS" if all_pass else "FAIL"
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
