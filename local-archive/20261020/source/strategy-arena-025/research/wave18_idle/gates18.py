# Wave-18 gate evaluation. S1-S5 reuse the EXACT SAME numeric bars and simulation
# methodology as research/wave13_liquidity/gates13.py (ruin P(<$50)<1%, MC p05>$100, block MDD
# p95<=10%, leg>=$5/gross<=$90, S5 stress sign+MDD<=15% under x3 measured slippage) --
# reimplemented locally rather than imported, matching gates13.py's own precedent ("이 모듈은
# wave10의 것을 가져오는 대신 로컬로 재구현한다"). S6 is NEW (SPEC.md): idle-capital exposure
# must never linger into (or block) a day L4 itself would trade -- engine18.py's day-loop
# guarantees this STRUCTURALLY (L4 checked first, unconditionally); gate_s6_recoverability
# re-checks it EMPIRICALLY against the saved per-day layer_used series, the same
# "structural claim + empirical gate" pattern S1 (delta-neutral-by-construction) already uses.
#
# Promotion is SPEC.md's own: "S1~S6 전부 PASS ∧ 전기간 CAGR > I0(9.37%) ∧ 고펀딩기 연환산이
# I0(22.01%) 대비 -1%p 이내" -- unlike gates13 (no baseline-beating clause) and closer to
# wave11_yield's own Y-series rule (beat a named baseline). The I0 numbers this module compares
# against are always the FRESHLY COMPUTED ones passed in by run_wave18.py (I0's own saved
# results/I0.json), never the SPEC.md prose figures hard-coded -- SPEC.md's own 9.37%/22.01%
# are themselves just a report of an earlier I0 run, and this wave's own "I0 재현이 L4와
# 정확히 일치하는지 먼저 검증" instruction means the freshly-computed number is the one that
# actually governs promotion.

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
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, RESERVE_FRACTION, TOTAL_CAPITAL
from research.wave10_carry100.regime import regime_breakdown
from research.wave18_idle import engine18
from research.wave18_idle.configs18 import LEG_FRACTION, TOP_K, IdleConfig

MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SEED: Final = 20_260_722  # matches wave10-17's freeze-date seed convention (kept as the same literal, not re-derived)

S2_RUIN_THRESHOLD_USDT: Final = 50.0
S2_RUIN_PROBABILITY_MAX: Final = 0.01
S2_P05_FLOOR_USDT: Final = 100.0
S3_BLOCK_MDD_P95_MAX: Final = 0.10
S5_BLOCK_MDD_P95_MAX: Final = 0.15  # inherited from gates13's own S5 bar -- SPEC.md's own wave18 S5 text ("실측슬리피지 x3 스트레스") does not restate a number, and this wave inherits wave13's cost model wholesale, so its stress-testing convention is inherited too (documented assumption, not silently invented)

PROMOTION_HIGH_FUNDING_TOLERANCE_PP: Final = -1.0  # SPEC.md: "고펀딩기 연환산이 I0 대비 -1%p 이내"

# SPEC.md line 36 states "누적 106회" as this wave's own pre-registered cumulative-trials
# figure. It does NOT chain cleanly from wave16's own disclosed 96 (research/wave16_duallayer/
# gates16.py) + wave18's 6 new candidates (96+6=102, not 106) -- wave17 added 0 (no new
# statistical trials, per its own gates17.py/reporting17.py). This module uses SPEC.md's frozen
# figure VERBATIM (not re-derived) -- same "trust the frozen SPEC number, do not re-derive it"
# precedent research/wave11_yield/gates_y.py itself set ("SPEC: '누적 시행 64회 기준' (frozen,
# not re-derived)"). This is reference-only and never used for promotion either way (see
# deflated_sharpe_reference below).
DSR_CUMULATIVE_TRIALS: Final = 106


def leg_usdt() -> float:
    return LEG_FRACTION * ACTIVE_CAPITAL


def gross_usdt() -> float:
    return 2.0 * TOP_K * LEG_FRACTION * ACTIVE_CAPITAL


def _daily_returns(equity: pd.Series) -> tuple[tuple[pd.Timestamp, ...], np.ndarray]:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave18 equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave18 equity returns contain invalid values")
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


def full_period_annualized(equity: pd.Series) -> float | None:
    """전기간 CAGR -- this wave's own primary metric (SPEC.md "이번 wave의 목적함수는 전기간
    CAGR"). Same formula as research.wave10_carry100.regime's own `_annualize` /
    research.wave17_lending_verified.gates17._full_period_annualized, reimplemented locally
    per this repo's established per-wave-gates-module convention."""
    if len(equity) < 2:
        return None
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    if start_value <= 0.0:
        return None
    days = max((pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).total_seconds() / 86_400.0, 1.0)
    growth = end_value / start_value
    if growth <= 0.0:
        return -1.0
    return float(growth ** (365.0 / days) - 1.0)


def utilization(positions: pd.Series) -> float:
    """가동률: fraction of days ANY carry/reverse position (L4 or an overlay) was held --
    does NOT count lending days (weights stay 0 while lending), matching Wave10Result's own
    positions convention exactly (sum(|weights|) > 0)."""
    if len(positions) == 0:
        return 0.0
    return float((positions.abs() > 0.0).mean())


def layer_breakdown(layer_used: pd.Series) -> dict[str, Any]:
    """Per-day layer attribution -- SPEC.md's required '대기일 수익 기여' breakdown. Counts +
    fractions of L4 / carry_overlay / reverse_overlay / lending / cash days."""
    total = int(len(layer_used))
    if total == 0:
        return {"total_days": 0, "counts": {}, "fractions": {}}
    counts = layer_used.value_counts()
    return {
        "total_days": total,
        "counts": {str(key): int(value) for key, value in counts.items()},
        "fractions": {str(key): float(value) / total for key, value in counts.items()},
    }


def annualized_round_trips(equity: pd.Series, trade_returns: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    span_days = max((pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).total_seconds() / 86_400.0, 1.0)
    return float(len(trade_returns)) / (span_days / 365.0)


def deflated_sharpe_reference(equity: pd.Series) -> dict[str, Any] | None:
    """Reference-only DSR, corrected for DSR_CUMULATIVE_TRIALS=106 (SPEC.md's own frozen
    figure). Never used for promotion (wave10-17's shared principle)."""
    clean = equity.dropna()
    if len(clean) < 4:
        return None
    timed = tuple(TimedValue(pd.Timestamp(idx).to_pydatetime(), float(value)) for idx, value in clean.items())
    try:
        dsr = deflated_sharpe(timed, trials=DSR_CUMULATIVE_TRIALS)
    except DeepValidationError:
        return None
    return {"score": dsr.score, "probability": dsr.probability, "trials": dsr.trials, "observed_sharpe": dsr.observed_sharpe}


# ---------------------------------------------------------------------------
# S1-S6.
# ---------------------------------------------------------------------------


def gate_s1_structure(idle_config: IdleConfig) -> dict[str, Any]:
    """S1: delta-neutral 2-leg + 1x leverage. Delta-neutrality is structural for every
    carry/reverse layer (a single shared, possibly-signed `weights` value drives both legs in
    engine18._run_idle_overlay_loop, copied from engine13/wave10-17's own precedent); 1x
    leverage means gross <= ACTIVE_CAPITAL, checked here -- the SAME $90 cap every layer shares
    (LEG_FRACTION/TOP_K are wave18-wide constants, not per-candidate). Lending (I1/I5's
    fallback) carries NO directional exposure at all by construction (a stablecoin balance,
    weights stay exactly 0.0 on lending days) -- SPEC.md's own "유휴 시간에 넣는 것은 캐리보다
    리스크가 낮거나 같아야 한다" principle is satisfied trivially for the lending leg, and
    explicitly NOT satisfied for I4's reverse-carry leg (flagged in gate_s4, not here -- S1 is
    about THIS backtest's own leverage/delta bookkeeping, not the borrow-market risk a live
    short-spot leg would add)."""
    gross = gross_usdt()
    leverage_ok = gross <= ACTIVE_CAPITAL + 1e-9
    return {
        "delta_neutral_by_construction": True,
        "lending_has_zero_directional_exposure_by_construction": idle_config.uses_lending_fallback,
        "gross_usdt": gross,
        "leverage_multiplier_of_active_capital": gross / ACTIVE_CAPITAL,
        "leverage_1x_ok": leverage_ok,
        "status": "PASS" if leverage_ok else "FAIL",
    }


def gate_s2_mc(equity: pd.Series, seed_offset: int) -> dict[str, Any]:
    _, returns = _daily_returns(equity)
    mc = _simulate_mc(returns, SEED + seed_offset * 101)
    p05_ok = mc["p05"] > S2_P05_FLOOR_USDT
    ruin_ok = mc["ruin_probability"] < S2_RUIN_PROBABILITY_MAX
    return {**mc, "p05_floor_ok": p05_ok, "ruin_ok": ruin_ok, "status": "PASS" if (p05_ok and ruin_ok) else "FAIL"}


def gate_s3_block_mdd(equity: pd.Series, seed_offset: int) -> dict[str, Any]:
    timestamps, returns = _daily_returns(equity)
    block = _block_shuffle(timestamps, returns, SEED + seed_offset * 103)
    ok = block["mdd_p95"] <= S3_BLOCK_MDD_P95_MAX
    return {**block, "status": "PASS" if ok else "FAIL"}


# SPEC.md task instruction: I4's short-spot leg "실행가능성(S4)을 엄격히 검증하고 불가하면
# 그렇게 보고". This wave fetched OKX's public `interest-rate-loan-quota` endpoint live
# (2026-07-24) and could NOT confirm from public documentation whether its `rate` field is
# already annualized or needs x8760 (both readings were tried; neither is confirmed by an API
# doc that says "annualized" in so many words) -- see report/wave18_report.md's I4 section for
# the raw fetched numbers under both readings. Regardless of that unresolved unit question, the
# STRUCTURAL blockers below are independently well-documented and sufficient on their own.
I4_SHORT_SPOT_INFEASIBLE_REASONS: Final[tuple[str, ...]] = (
    "차입금리 단위(시간당 raw vs 연환산) 미확정 -- OKX public/interest-rate-loan-quota를 실측(2026-07-24)했으나 "
    "공개 문서에서 'annualized'라고 명시하는 원문을 확보하지 못했다 (두 해석 모두 report에 병기).",
    "숏현물 실행에는 이 wave의 다른 모든 후보(L4/I0-I3/I5)가 쓰는 단순 현물+무기한 헤지 계좌와 별도인 "
    "마진(교차/포트폴리오 마진) 계좌 모드가 필요하다 -- 계좌 모드 전환 자체가 이 백테스트의 "
    "capital_contract가 가정하는 단순 구조를 벗어난다.",
    "차입한 현물은 상환 시 현물로 되갚아야 한다(repay-in-kind) -- 포지션 종료 시점의 가격 변동이 "
    "추가 리스크로 얹힌다. I1(USDT 대여, flexible 즉시상환 확인됨)이나 L4/I2/I3(단순 현물+선물 헤지)에는 "
    "없는 리스크 축이다.",
    "SPEC.md 원칙 '유휴 시간에 넣는 것은 캐리보다 리스크가 낮거나 같아야 한다'를 마진 차입-기반 숏현물은 "
    "구조적으로 충족하지 못한다 -- 마진콜/청산 리스크가 캐리 자체의 델타중립 구조에는 없는 새로운 리스크다.",
)


def gate_s4_feasibility(idle_config: IdleConfig, positions: pd.Series) -> dict[str, Any]:
    leg = leg_usdt()
    gross = gross_usdt()
    min_order_ok = leg >= MIN_ORDER_USDT
    gross_ok = gross <= ACTIVE_CAPITAL + 1e-9
    base_passed = bool(min_order_ok and gross_ok)

    lending_note = (
        "OKX 공개 도움말(introduction-to-okx-savings-and-its-rules, wave17 실측/인용): 개인 예치·회수에 "
        "일반적인 한도 없음 -- $90 예치는 이 기준상 실행가능."
        if idle_config.uses_lending_fallback or idle_config.candidate_id == "I1"
        else None
    )

    short_spot_feasible: bool | None = None
    reasons: tuple[str, ...] = ()
    if idle_config.uses_reverse_overlay:
        short_spot_feasible = False
        reasons = I4_SHORT_SPOT_INFEASIBLE_REASONS

    passed = base_passed and (short_spot_feasible is not False)
    return {
        "leg_usdt_nominal": leg,
        "gross_usdt_nominal": gross,
        "min_order_usdt": MIN_ORDER_USDT,
        "min_order_feasible": min_order_ok,
        "gross_exposure_feasible": gross_ok,
        "lending_deposit_feasible_note": lending_note,
        "short_spot_borrow_feasible": short_spot_feasible,
        "short_spot_infeasibility_reasons": list(reasons),
        "status": "PASS" if passed else "FAIL",
    }


def gate_s5_stress(
    base_high_funding: float | None,
    stress_high_funding: float | None,
    stress_equity: pd.Series,
    seed_offset: int,
) -> dict[str, Any]:
    """S5: SPEC.md "실측슬리피지 x3 스트레스". Numeric bar inherited from gates13's own S5
    (sign preservation AND block-shuffle MDD p95 <= 15%) -- see module-level S5_BLOCK_MDD_P95_MAX
    comment for why this is an explicit, documented inheritance rather than a SPEC.md-restated
    number."""
    sign_ok = stress_high_funding is not None and stress_high_funding > 0.0
    stress_block = None
    mdd_ok = False
    try:
        timestamps, returns = _daily_returns(stress_equity)
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


def gate_s6_recoverability(layer_used: pd.Series, i0_reference_positions: pd.Series) -> dict[str, Any]:
    """S6 (신규, SPEC.md): 캐리 신호 발생일에 유휴자본이 즉시 회수돼 캐리 진입을 막지 않음.
    `i0_reference_positions` is I0's OWN saved positions series (= L4's own standalone signal).
    PASS requires layer_used == engine18.LAYER_L4 on EVERY day I0 itself is active; any other
    value on such a day means an overlay/lending exposure lingered into (or blocked) an L4
    entry. Structurally impossible given engine18._run_idle_overlay_loop's own unconditional
    L4-first priority -- this gate is the empirical confirmation, not the mechanism itself."""
    aligned_layer = layer_used.reindex(i0_reference_positions.index)
    l4_active_days = i0_reference_positions.abs() > 0.0
    n_l4_active_days = int(l4_active_days.sum())
    if n_l4_active_days == 0:
        return {"l4_active_days": 0, "violations": 0, "status": "PASS", "note": "reference has zero active days"}
    on_active_days = aligned_layer[l4_active_days]
    violations = on_active_days.isna() | (on_active_days != engine18.LAYER_L4)
    n_violations = int(violations.sum())
    return {
        "l4_active_days": n_l4_active_days,
        "violations": n_violations,
        "status": "PASS" if n_violations == 0 else "FAIL",
    }


# ---------------------------------------------------------------------------
# Promotion + failure classification.
# ---------------------------------------------------------------------------

FAILURE_LEVERAGE = "레버리지/gross"
FAILURE_MIN_ORDER = "최소주문"
FAILURE_PRINCIPAL = "원금보존(MC)"
FAILURE_DRAWDOWN = "MDD초과"
FAILURE_STRESS_SIGN = "스트레스부호반전"
FAILURE_STRESS_MDD = "스트레스MDD초과"
FAILURE_RECOVERABILITY = "회수성위반(S6)"
FAILURE_SHORT_SPOT_INFEASIBLE = "숏현물실행불가"
FAILURE_BELOW_I0_CAGR = "전기간CAGR미달(I0대비)"
FAILURE_ACTIVE_REGIME_DAMAGED = "고펀딩기활성구간훼손(I0대비-1%p초과)"


def _classify_gate_failures(s1: dict, s2: dict, s3: dict, s4: dict, s5: dict, s6: dict) -> list[str]:
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
        if s4.get("short_spot_borrow_feasible") is False:
            reasons.append(FAILURE_SHORT_SPOT_INFEASIBLE)
    if s5["status"] == "FAIL":
        if not s5["sign_preserved"]:
            reasons.append(FAILURE_STRESS_SIGN)
        if not s5["stress_mdd_ok"]:
            reasons.append(FAILURE_STRESS_MDD)
    if s6["status"] == "FAIL":
        reasons.append(FAILURE_RECOVERABILITY)
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return seen


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    full_period_annualized: float | None
    high_funding_mean_annualized_return: float | None
    beats_i0_full_period: bool
    i0_full_period_annualized: float | None
    within_tolerance_of_i0_high_funding: bool
    i0_high_funding_annualized: float | None
    high_funding_gap_pp: float | None
    promoted: bool


def promotion_check(
    full_period: float | None,
    high_funding: float | None,
    i0_full_period: float | None,
    i0_high_funding: float | None,
    gates_overall: str,
) -> PromotionCheck:
    beats_i0 = full_period is not None and i0_full_period is not None and full_period > i0_full_period
    gap_pp = (high_funding - i0_high_funding) * 100.0 if (high_funding is not None and i0_high_funding is not None) else None
    # 1e-9 slack guards the exact "-1%p 이내" boundary against float noise from the *100.0
    # subtraction above (e.g. 0.21-0.22 != -0.01 exactly in binary float) -- same
    # boundary-inclusive convention gates13.py's own `<= ACTIVE_CAPITAL + 1e-9` checks use.
    within_tolerance = gap_pp is not None and gap_pp >= PROMOTION_HIGH_FUNDING_TOLERANCE_PP - 1e-9
    promoted = gates_overall == "PASS" and beats_i0 and within_tolerance
    return PromotionCheck(full_period, high_funding, beats_i0, i0_full_period, within_tolerance, i0_high_funding, gap_pp, promoted)


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
    idle_config: IdleConfig,
    equity: pd.Series,
    positions: pd.Series,
    stress_equity: pd.Series,
    layer_used: pd.Series,
    i0_reference_positions: pd.Series,
    i0_full_period_cagr: float | None,
    i0_high_funding_annualized: float | None,
    seed_offset: int,
) -> GateReport:
    gate_s1 = gate_s1_structure(idle_config)
    gate_s2 = gate_s2_mc(equity, seed_offset)
    gate_s3 = gate_s3_block_mdd(equity, seed_offset)
    gate_s4 = gate_s4_feasibility(idle_config, positions)
    base_regime = regime_breakdown(_EquityOnly(equity))
    stress_regime = regime_breakdown(_EquityOnly(stress_equity))
    gate_s5 = gate_s5_stress(
        base_regime.get("high_funding_mean_annualized_return"), stress_regime.get("high_funding_mean_annualized_return"), stress_equity, seed_offset
    )
    gate_s6 = gate_s6_recoverability(layer_used, i0_reference_positions)

    all_pass = all(gate["status"] == "PASS" for gate in (gate_s1, gate_s2, gate_s3, gate_s4, gate_s5, gate_s6))
    overall = "PASS" if all_pass else "FAIL"

    full_period = full_period_annualized(equity)
    high_funding = base_regime.get("high_funding_mean_annualized_return")
    promotion = promotion_check(full_period, high_funding, i0_full_period_cagr, i0_high_funding_annualized, overall)

    reasons = _classify_gate_failures(gate_s1, gate_s2, gate_s3, gate_s4, gate_s5, gate_s6)
    if not promotion.beats_i0_full_period:
        reasons.append(FAILURE_BELOW_I0_CAGR)
    if not promotion.within_tolerance_of_i0_high_funding:
        reasons.append(FAILURE_ACTIVE_REGIME_DAMAGED)

    return GateReport(gate_s1, gate_s2, gate_s3, gate_s4, gate_s5, gate_s6, overall, tuple(reasons), promotion)


class _EquityOnly:
    """Minimal duck-typed stand-in so research.wave10_carry100.regime.regime_breakdown
    (which only ever reads `.equity`) can be called on a bare pd.Series without needing a full
    Wave10Result/Wave18Result constructed just for that one field."""

    def __init__(self, equity: pd.Series) -> None:
        self.equity = equity


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
    "I4_SHORT_SPOT_INFEASIBLE_REASONS",
    "PROMOTION_HIGH_FUNDING_TOLERANCE_PP",
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
    "full_period_annualized",
    "gate_report_payload",
    "gate_s1_structure",
    "gate_s2_mc",
    "gate_s3_block_mdd",
    "gate_s4_feasibility",
    "gate_s5_stress",
    "gate_s6_recoverability",
    "gross_usdt",
    "layer_breakdown",
    "leg_usdt",
    "promotion_check",
    "utilization",
]
