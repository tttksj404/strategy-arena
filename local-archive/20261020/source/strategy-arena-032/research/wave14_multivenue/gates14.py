# Wave-14 gate evaluation. S1-S5 use the EXACT SAME numeric bars and simulation
# methodology as research/wave13_liquidity/gates13.py (ruin P(<0.5x initial)<1%, MC
# p05>initial capital, block MDD p95<=10%, leg>=$5/gross<=active capital, stressed(x3)
# sign+MDD<=15% survival) -- SPEC.md: "게이트(wave-13 S1~S5 승계 + 신규)". Reimplemented
# locally (not imported) for the same reason gates13.py itself reimplements rather than
# imports gates12's: those numeric helpers (_simulate_mc/_block_shuffle/_mdd) close over
# wave10's FIXED $100/$90/10% capital constants, which this wave's variable capital tiers
# ($100/$300/$1,000/$3,000) cannot use unmodified -- see engine14.py's own module
# docstring for the identical reasoning applied to the backtest loop itself. Two of
# gates13's helpers genuinely ARE capital-agnostic (utilization, annualized_round_trips)
# and are imported unchanged below, not copied.
#
# S6 is NEW (SPEC.md): "거래소간 구조는 양쪽 거래소에 자본이 분리 예치된다. 한 거래소 전액
# 손실... 남은 자본이 초기의 50% 이상인지 명시... 확률이 아니라 구조적 노출로 보고(백테스트로
# 확률 추정 금지)". Interpretation (SPEC.md's own S6 sentence textually scopes itself to
# "M6/M7 거래소간 구조", documented here rather than silently generalized or silently
# ignored elsewhere):
#   - M6/M7 (structure="cross_venue_spread"): S6 is a REAL, GATING, STRUCTURAL (not
#     empirical) computation -- gate_s6_cross_venue_structure. By construction every active
#     position always has EXACTLY HALF its gross notional on each venue (one whole perp leg
#     per venue, symmetric), so the worst case ("one venue's custody goes to zero") is a
#     FIXED fraction of capital for every M6/M7 tier, not something a backtest path could
#     make look better or worse -- consistent with SPEC's own "확률 추정 금지".
#   - M1/M3/M4/M5 (structure="carry", include_bybit=True): capital IS in fact custodied at
#     two venues (whichever venue each position's BOTH legs happen to sit on), but SPEC's
#     own S6 sentence never describes this shape ("거래소간 구조" describes M6/M7's
#     split-leg structure specifically, not "some fraction of the book happens to be on
#     Bybit today"). This module still COMPUTES a parallel structural-exposure figure for
#     these four (gate_s6_pool_venue_exposure) -- useful for SPEC.md's own required
#     "거래소 추가의 순효과: 기회 증가분 vs 비용·운영 복잡도" writeup -- but reports it as
#     INFO, not PASS/FAIL, and it never blocks promotion. Both the structural worst case
#     (100% concentration is not ruled out by construction -- nothing in the top_k ranking
#     guarantees venue diversification) and the empirical historical max are reported side
#     by side so neither number is mistaken for the other.
#   - M0/M2 (single venue only): N/A, not computed.

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
from research.wave10_carry100.engine import MIN_ORDER_USDT, Wave10Result
from research.wave10_carry100.regime import regime_breakdown
from research.wave13_liquidity.gates13 import annualized_round_trips, utilization  # capital-agnostic, reused unchanged
from research.wave14_multivenue.configs14 import LEG_USDT, RESERVE_FRACTION, Wave14Config

MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SEED: Final = 20_260_722  # matches wave10-13's freeze-date seed convention

S2_RUIN_PROBABILITY_MAX: Final = 0.01
S3_BLOCK_MDD_P95_MAX: Final = 0.10
S5_BLOCK_MDD_P95_MAX: Final = 0.15
S6_RESIDUAL_CAPITAL_FLOOR: Final = 0.50  # SPEC.md "남은 자본이 초기의 50% 이상"

DSR_CUMULATIVE_TRIALS: Final = 84  # task contract: wave13's own disclosed 76 + this wave's 8 (M0-M7) new candidates


def leg_usdt(config: Wave14Config) -> float:
    return LEG_USDT  # fixed at every tier by construction -- see configs14.py's own module docstring


def gross_usdt(config: Wave14Config) -> float:
    return 2.0 * config.candidate.top_k * LEG_USDT


def _daily_returns(equity: pd.Series) -> tuple[tuple[pd.Timestamp, ...], np.ndarray]:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave14 equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave14 equity returns contain invalid values")
    timestamps = tuple(pd.Timestamp(ts) for ts in clean.index[1:])
    return timestamps, returns


def _simulate_mc(returns: np.ndarray, seed: int, total_capital: float, active_capital: float, reserve_fraction: float) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    finals = np.empty(MC_PATHS, dtype=float)
    chunk_size = 250
    reserve_usdt = total_capital * reserve_fraction
    for start in range(0, MC_PATHS, chunk_size):
        stop = min(start + chunk_size, MC_PATHS)
        samples = rng.choice(returns, size=(stop - start, returns.size), replace=True)
        growth = np.prod(1.0 + np.clip(samples, -0.999999, None), axis=1)
        finals[start:stop] = reserve_usdt + active_capital * growth
    ruin_threshold = total_capital * 0.5  # SPEC.md "P(<초기x0.5)"
    return {
        "p05": float(np.quantile(finals, 0.05)),
        "ruin_probability": float(np.mean(finals < ruin_threshold)),
        "ruin_threshold_usdt": ruin_threshold,
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


def _mdd(values: np.ndarray, total_capital: float, active_capital: float, reserve_fraction: float) -> float:
    reserve_usdt = total_capital * reserve_fraction
    curve = reserve_usdt + active_capital * np.cumprod(1.0 + np.clip(values, -0.999999, None))
    peaks = np.maximum.accumulate(np.concatenate(([total_capital], curve)))
    return float(np.max(1.0 - curve / peaks[1:]))


def _block_shuffle(
    timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray, seed: int, total_capital: float, active_capital: float, reserve_fraction: float
) -> dict[str, float]:
    blocks = _blocks(timestamps, returns)
    rng = np.random.default_rng(seed)
    mdds = np.empty(BLOCK_PATHS, dtype=float)
    finals = np.empty(BLOCK_PATHS, dtype=float)
    reserve_usdt = total_capital * reserve_fraction
    for index in range(BLOCK_PATHS):
        path = np.concatenate([blocks[item] for item in rng.permutation(len(blocks))])
        mdds[index] = _mdd(path, total_capital, active_capital, reserve_fraction)
        finals[index] = reserve_usdt + active_capital * np.prod(1.0 + np.clip(path, -0.999999, None))
    return {
        "block_days": BLOCK_DAYS,
        "block_count": len(blocks),
        "paths": BLOCK_PATHS,
        "mdd_p95": float(np.quantile(mdds, 0.95)),
        "final_p05": float(np.quantile(finals, 0.05)),
    }


def gate_s1_structure(config: Wave14Config) -> dict[str, Any]:
    """S1: delta-neutral 2-leg + 1x leverage. For M0-M5 this is the ordinary long-spot/
    short-perp structure (a single shared `weights` value drives both legs -- structural by
    construction, unchanged from wave10-13). For M6/M7, "delta-neutral" means the two
    perpetual legs carry EQUAL AND OPPOSITE notional in the SAME underlying (short one
    venue's perp, long the other's, both sized at `leg_fraction` -- net USD delta to the
    underlying's price is ~0 by construction, though NOT price-identical since the two
    venues' own mark prices can diverge -- that residual basis risk is priced into the
    backtest via two REAL price series, not assumed away; see engine14.py's own PnL
    derivation). Leverage 1x means gross <= active_capital, checked here for all 8."""
    gross = gross_usdt(config)
    leverage_ok = gross <= config.active_capital + 1e-6
    return {
        "delta_neutral_by_construction": True,
        "structure": config.structure,
        "gross_usdt": gross,
        "active_capital_usdt": config.active_capital,
        "leverage_multiplier_of_active_capital": gross / config.active_capital,
        "leverage_1x_ok": leverage_ok,
        "status": "PASS" if leverage_ok else "FAIL",
    }


def gate_s2_mc(result: Wave10Result, config: Wave14Config, seed_offset: int) -> dict[str, Any]:
    _, returns = _daily_returns(result.equity)
    mc = _simulate_mc(returns, SEED + seed_offset * 101, config.total_capital, config.active_capital, RESERVE_FRACTION)
    p05_floor = config.total_capital  # SPEC.md "p05 > 초기자본"
    p05_ok = mc["p05"] > p05_floor
    ruin_ok = mc["ruin_probability"] < S2_RUIN_PROBABILITY_MAX
    return {**mc, "p05_floor_usdt": p05_floor, "p05_floor_ok": p05_ok, "ruin_ok": ruin_ok, "status": "PASS" if (p05_ok and ruin_ok) else "FAIL"}


def gate_s3_block_mdd(result: Wave10Result, config: Wave14Config, seed_offset: int) -> dict[str, Any]:
    timestamps, returns = _daily_returns(result.equity)
    block = _block_shuffle(timestamps, returns, SEED + seed_offset * 103, config.total_capital, config.active_capital, RESERVE_FRACTION)
    ok = block["mdd_p95"] <= S3_BLOCK_MDD_P95_MAX
    return {**block, "status": "PASS" if ok else "FAIL"}


def gate_s4_feasibility(config: Wave14Config, result: Wave10Result) -> dict[str, Any]:
    leg = leg_usdt(config)
    gross = gross_usdt(config)
    min_order_ok = leg >= MIN_ORDER_USDT
    gross_ok = gross <= config.active_capital + 1e-6
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
    config: Wave14Config,
    seed_offset: int,
) -> dict[str, Any]:
    sign_ok = stress_high_funding is not None and stress_high_funding > 0.0
    stress_block = None
    mdd_ok = False
    try:
        timestamps, returns = _daily_returns(stress_result.equity)
        stress_block = _block_shuffle(timestamps, returns, SEED + seed_offset * 107, config.total_capital, config.active_capital, RESERVE_FRACTION)
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


def gate_s6_cross_venue_structure(config: Wave14Config) -> dict[str, Any]:
    """M6/M7 only: STRUCTURAL (not empirical/probabilistic -- SPEC.md forbids estimating a
    probability here) worst-case residual capital if ONE venue's custody goes to zero.
    Every active M6/M7 position has exactly HALF its gross notional on each venue by
    construction (one whole perp leg per venue) -- so per-venue exposure = gross/2 =
    active_capital/2 always, independent of which symbols are actually held or what the
    market did. The 10% cash reserve is assumed held off either exchange (this repo's
    existing convention throughout -- RESERVE_FRACTION has never been described as
    exchange-custodied capital in any prior wave)."""
    if config.structure != "cross_venue_spread":
        return {"applicable": False, "status": "N/A", "note": "cross_venue_spread 구조가 아님 -- S6 미적용."}
    per_venue_exposure_usdt = gross_usdt(config) / 2.0
    residual_capital_fraction = 1.0 - per_venue_exposure_usdt / config.total_capital
    ok = residual_capital_fraction >= S6_RESIDUAL_CAPITAL_FLOOR
    return {
        "applicable": True,
        "basis": "structural_worst_case_not_probability",
        "gross_usdt": gross_usdt(config),
        "per_venue_exposure_usdt": per_venue_exposure_usdt,
        "total_capital_usdt": config.total_capital,
        "residual_capital_fraction_if_one_venue_wiped": residual_capital_fraction,
        "residual_capital_floor": S6_RESIDUAL_CAPITAL_FLOOR,
        "status": "PASS" if ok else "FAIL",
    }


def gate_s6_pool_venue_exposure(config: Wave14Config, daily_bybit_share: pd.Series | None) -> dict[str, Any]:
    """M1/M3/M4/M5 only (informational -- see module docstring for why this never gates
    promotion): (a) the STRUCTURAL worst case (nothing in top_k ranking prevents every
    concurrent slot from landing on the same venue on any given day, so the worst case is
    ALL of active_capital on one venue = residual 1 - active_capital/total_capital = 10%
    for every tier, always < the 50% floor) and (b) the EMPIRICAL historical Bybit-vs-
    Binance split actually realized in this run (engine14.compute_daily_bybit_share, a
    capital-independent replay of the loop's own eligible/ranked selection), reported side
    by side and clearly labeled so neither is mistaken for the other -- (a) is a provable
    bound, (b) is merely what happened to occur in this one historical path."""
    if not (config.structure == "carry" and config.include_bybit):
        return {"applicable": False, "status": "N/A", "note": "단일거래소 구성 -- 교차거래소 노출 없음."}
    worst_case_residual = 1.0 - config.active_capital / config.total_capital
    empirical: dict[str, Any] = {"available": False}
    if daily_bybit_share is not None:
        active_days = daily_bybit_share.dropna()
        if len(active_days) > 0:
            empirical = {
                "available": True,
                "mean_bybit_share_of_filled_slots": float(active_days.mean()),
                "median_bybit_share_of_filled_slots": float(active_days.median()),
                "max_bybit_share_of_filled_slots": float(active_days.max()),
                "days_with_any_position": int(len(active_days)),
            }
    return {
        "applicable": True,
        "basis": "informational_not_gating (SPEC.md's S6 문장은 M6/M7 '거래소간 구조'를 명시적으로 지칭 -- gates14.py 모듈 docstring 참조)",
        "structural_worst_case_residual_if_one_venue_wiped": worst_case_residual,
        "structural_note": "top_k 랭킹은 거래소 분산을 보장하지 않음 -- 이론상 전 슬롯이 한 거래소에 몰릴 수 있어 구조적 하한은 항상 (1 - active_capital/total_capital) = 10%.",
        "residual_capital_floor_reference": S6_RESIDUAL_CAPITAL_FLOOR,
        "empirical_bybit_share": empirical,
        "status": "INFO",
    }


def utilization_wrap(result: Wave10Result) -> float:
    return utilization(result)


def annualized_round_trips_wrap(result: Wave10Result) -> float:
    return annualized_round_trips(result)


def deflated_sharpe_reference(result: Wave10Result) -> dict[str, Any] | None:
    """Reference-only DSR corrected for DSR_CUMULATIVE_TRIALS=84 (wave13's own disclosed 76
    + this wave's 8 new M0-M7 candidates). Never used for promotion -- same principle
    wave10-13 all share. research.validation.deep_stats.deflated_sharpe itself is fully
    capital-agnostic (works off equity returns directly), so only the `trials` count needed
    overriding here -- not worth a full gates13.deflated_sharpe_reference copy for that."""
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
FAILURE_VENUE_STRUCTURE = "거래소구조노출초과"


def _classify_failures(s1: dict, s2: dict, s3: dict, s4: dict, s5: dict, s6: dict) -> tuple[str, ...]:
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
    if s6["status"] == "FAIL":
        reasons.append(FAILURE_VENUE_STRUCTURE)
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return tuple(seen)


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    high_funding_mean_annualized_return: float | None
    gates_pass: bool  # S1-S6 (as applicable) all PASS
    baseline_candidate_id: str | None
    baseline_high_funding_mean_annualized_return: float | None
    beats_baseline: bool | None  # None when baseline_candidate_id is None (M0/M2: nothing to beat)
    promoted: bool


def promotion_check(
    regime: dict[str, Any], overall_status: str, config: Wave14Config, baseline_regime: dict[str, Any] | None
) -> PromotionCheck:
    high_funding = regime.get("high_funding_mean_annualized_return")
    gates_pass = overall_status == "PASS"
    baseline_high_funding = baseline_regime.get("high_funding_mean_annualized_return") if baseline_regime is not None else None
    beats_baseline: bool | None
    if config.baseline_candidate_id is None:
        beats_baseline = None  # M0/M2 -- nothing external to beat, they ARE the tier's own baseline
    elif high_funding is None or baseline_high_funding is None:
        beats_baseline = False  # missing data fails closed, never assumed to pass
    else:
        beats_baseline = high_funding > baseline_high_funding
    promoted = gates_pass and (beats_baseline is None or beats_baseline is True)
    return PromotionCheck(high_funding, gates_pass, config.baseline_candidate_id, baseline_high_funding, beats_baseline, promoted)


@dataclass(frozen=True, slots=True)
class GateReport:
    gate_s1: dict[str, Any]
    gate_s2: dict[str, Any]
    gate_s3: dict[str, Any]
    gate_s4: dict[str, Any]
    gate_s5: dict[str, Any]
    gate_s6: dict[str, Any]
    overall: str
    failure_reasons: tuple[str, ...]
    promotion: PromotionCheck


def evaluate_gates(
    config: Wave14Config,
    result: Wave10Result,
    stress_result: Wave10Result,
    seed_offset: int,
    daily_bybit_share: pd.Series | None,
    baseline_regime: dict[str, Any] | None,
) -> GateReport:
    gate_s1 = gate_s1_structure(config)
    gate_s2 = gate_s2_mc(result, config, seed_offset)
    gate_s3 = gate_s3_block_mdd(result, config, seed_offset)
    gate_s4 = gate_s4_feasibility(config, result)
    regime = regime_breakdown(result)
    stress_regime = regime_breakdown(stress_result)
    gate_s5 = gate_s5_stress(regime.get("high_funding_mean_annualized_return"), stress_regime.get("high_funding_mean_annualized_return"), stress_result, config, seed_offset)
    if config.structure == "cross_venue_spread":
        gate_s6 = gate_s6_cross_venue_structure(config)
    else:
        gate_s6 = gate_s6_pool_venue_exposure(config, daily_bybit_share)
    blocking_gates = (gate_s1, gate_s2, gate_s3, gate_s4, gate_s5)
    all_pass = all(gate["status"] == "PASS" for gate in blocking_gates) and gate_s6["status"] in {"PASS", "N/A", "INFO"}
    overall = "PASS" if all_pass else "FAIL"
    promotion = promotion_check(regime, overall, config, baseline_regime)
    reasons = _classify_failures(gate_s1, gate_s2, gate_s3, gate_s4, gate_s5, gate_s6)
    return GateReport(gate_s1, gate_s2, gate_s3, gate_s4, gate_s5, gate_s6, overall, reasons, promotion)


def gate_report_payload(report: GateReport) -> dict[str, Any]:
    return {
        "gate_s1": report.gate_s1,
        "gate_s2": report.gate_s2,
        "gate_s3": report.gate_s3,
        "gate_s4": report.gate_s4,
        "gate_s5": report.gate_s5,
        "gate_s6": report.gate_s6,
        "overall": report.overall,
        "failure_reasons": list(report.failure_reasons),
        "promotion": asdict(report.promotion),
    }


__all__ = [
    "DSR_CUMULATIVE_TRIALS",
    "S2_RUIN_PROBABILITY_MAX",
    "S3_BLOCK_MDD_P95_MAX",
    "S5_BLOCK_MDD_P95_MAX",
    "S6_RESIDUAL_CAPITAL_FLOOR",
    "GateReport",
    "PromotionCheck",
    "annualized_round_trips_wrap",
    "deflated_sharpe_reference",
    "evaluate_gates",
    "gate_report_payload",
    "gate_s1_structure",
    "gate_s2_mc",
    "gate_s3_block_mdd",
    "gate_s4_feasibility",
    "gate_s5_stress",
    "gate_s6_cross_venue_structure",
    "gate_s6_pool_venue_exposure",
    "gross_usdt",
    "leg_usdt",
    "promotion_check",
    "utilization_wrap",
]
