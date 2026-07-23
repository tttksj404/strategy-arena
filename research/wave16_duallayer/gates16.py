# Wave-16 gate evaluation. Two DIFFERENT kinds of judgment live here, and SPEC.md is explicit
# that they must not be blurred into one PASS/FAIL number:
#
# 1. S1-S5 STRUCTURAL/STATISTICAL gates (leverage, MC bootstrap, block-shuffle MDD, $45
#    executability, x3-stress survival) -- reused VERBATIM from research.wave13_liquidity.gates13
#    (identical numeric bars: SPEC.md never registers new ones for wave16), but evaluated ONLY on
#    each candidate's `funding_only` companion series (engine16.DualLayerResult.funding_only) --
#    SPEC.md 방법 4: "MC/블록셔플은 펀딩 부분에만 적용 가능함을 표기... 대여이자 부분은 게이트
#    미적용." The lending-inclusive `combined` series is NEVER passed to gates13's MC/block
#    -shuffle math -- doing so would smear a hand-set constant's false certainty into a
#    resampling procedure that only has empirical grounding for the funding component.
#
# 2. The wave16-SPECIFIC "구조 유효" (structure valid) promotion rule from SPEC.md's own 판정
#    section -- "E2가 E0 대비 개선 ∧ E3(50%할인) 개선 유지 ∧ E4(대여0%) >= E0" -- compared on
#    each candidate's OWN `combined` (lending-inclusive) high-funding-regime annualized return.
#    This is NOT a gates13-style PASS/FAIL; it is a cross-candidate scalar comparison, and it is
#    the ONLY place in this module the lending-inclusive numbers are used for a promotion
#    decision (reporting16.py may print them elsewhere, but never as a "gate").
#
# A third, non-probabilistic item is reported alongside (never gated): the OKX(lend)/Bitget
# (execute) cross-venue exposure this structure requires, using the SAME formula wave14's own S6
# established for its dual-perp M6/M7 candidates (research/wave14_multivenue/gates14.py: "거래소
# 1곳 전액 손실 시 잔존 자본 ... 시장 경로와 무관하게 고정") -- a different REASON for the same
# fixed 50/50-of-active-capital split, so the same arithmetic applies verbatim.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave10_carry100.engine import TOTAL_CAPITAL, Wave10Result
from research.wave10_carry100.regime import regime_breakdown
from research.wave13_liquidity import gates13
from research.wave16_duallayer.configs16 import L4_CONFIG

DSR_CUMULATIVE_TRIALS: Final = 96  # wave15's own disclosed 91 (research/wave15_diverse/gates15.py: wave14's 84 + wave15's 7 ACTUAL candidates) + this wave's 5 (E0-E4)

# NOTE: every function below takes plain Wave10Result objects (never engine16.DualLayerResult
# directly) -- deliberately, so the SAME functions work both in-memory (run_wave16.py's `run`
# stage, fed straight from engine16.run_candidate(...).funding_only.result) and reloaded from
# saved JSON (run_wave16.py's `gates` stage, fed from research.wave1.gate_reporting._series --
# same convention run_wave13.py's own _evaluate_and_save/_result_from_payload split uses).


# ---------------------------------------------------------------------------
# 1. S1-S5 on the funding-only companion.
# ---------------------------------------------------------------------------


def evaluate_funding_only_gates(funding_only_result: Wave10Result, funding_only_stress_result: Wave10Result, seed_offset: int) -> gates13.GateReport:
    """S1(구조/1x)/S2(MC)/S3(블록MDD)/S4(실행가능)/S5(x3스트레스) on the funding_only companion
    (SAME ranking/trade-selection as the candidate's own combined series, lending stripped out of
    realized PnL). `L4_CONFIG` is the literal same Wave13Config object every E0-E4 candidate uses
    (capital contract never varies across this wave), so gates13's own S1/S4 math applies
    unmodified."""
    return gates13.evaluate_gates(L4_CONFIG, funding_only_result, funding_only_stress_result, seed_offset)


def deflated_sharpe_reference(funding_only_result: Wave10Result) -> dict[str, Any] | None:
    """Reference-only DSR (never used for promotion, wave10-15's shared principle), corrected
    for DSR_CUMULATIVE_TRIALS=96, evaluated on the funding_only companion for the same reason
    S2/S3/S5 are: Sharpe on a series with a hand-set constant baked in would not mean what DSR's
    correction assumes it means."""
    return gates13.deflated_sharpe_reference(funding_only_result)


def utilization(funding_only_result: Wave10Result) -> float:
    return gates13.utilization(funding_only_result)


# ---------------------------------------------------------------------------
# 2. SPEC.md's own "구조 유효" promotion rule -- combined (lending-inclusive) high-funding-regime
#    annualized return, cross-candidate comparison, never gated.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructureVerdict:
    e0_high_funding_annualized: float | None
    e1_high_funding_annualized: float | None
    e2_high_funding_annualized: float | None
    e3_high_funding_annualized: float | None
    e4_high_funding_annualized: float | None
    e2_beats_e0: bool
    e3_beats_e0: bool
    e4_at_least_e0: bool
    structure_valid: bool
    reasons: tuple[str, ...]


def _high_funding_annualized(combined_result: Wave10Result) -> float | None:
    regime = regime_breakdown(combined_result)
    value = regime.get("high_funding_mean_annualized_return")
    return float(value) if value is not None else None


def evaluate_structure_validity(combined_results_by_id: dict[str, Wave10Result]) -> StructureVerdict:
    """SPEC.md 판정: "E2가 E0 대비 개선 ∧ E3(50% 할인) 개선 유지 ∧ E4(대여 0%) >= E0 -> 구조
    유효로 등급 부여(단 '단면 근거, 시계열 미검증' 라벨 필수)." "E4 < E0면 합산 랭킹 기각."
    Compared on regime_breakdown()'s high_funding_mean_annualized_return -- this repo's own
    established cross-wave promotion metric (wave14/15's own bar is the identical quantity) --
    computed on each candidate's OWN `combined` (lending-inclusive, at that candidate's own
    discount) equity. Requires E0/E2/E3/E4 all present; E1 is carried for completeness/reporting
    only (SPEC.md's 판정 rule itself never references E1)."""
    missing = {"E0", "E1", "E2", "E3", "E4"} - set(combined_results_by_id)
    if missing:
        raise KeyError(f"evaluate_structure_validity needs all of E0-E4, missing: {sorted(missing)}")
    e0 = _high_funding_annualized(combined_results_by_id["E0"])
    e1 = _high_funding_annualized(combined_results_by_id["E1"])
    e2 = _high_funding_annualized(combined_results_by_id["E2"])
    e3 = _high_funding_annualized(combined_results_by_id["E3"])
    e4 = _high_funding_annualized(combined_results_by_id["E4"])
    e2_beats_e0 = e0 is not None and e2 is not None and e2 > e0
    e3_beats_e0 = e0 is not None and e3 is not None and e3 > e0
    e4_at_least_e0 = e0 is not None and e4 is not None and e4 >= e0
    structure_valid = e2_beats_e0 and e3_beats_e0 and e4_at_least_e0
    reasons: list[str] = []
    if not e2_beats_e0:
        reasons.append("E2가 E0 대비 개선 실패")
    if not e3_beats_e0:
        reasons.append("E3(50%할인)가 E0 대비 개선 유지 실패")
    if not e4_at_least_e0:
        reasons.append("E4(대여0%)가 E0 미달 -- SPEC.md: 합산 랭킹 자체가 해롭다는 뜻, 기각")
    return StructureVerdict(
        e0_high_funding_annualized=e0,
        e1_high_funding_annualized=e1,
        e2_high_funding_annualized=e2,
        e3_high_funding_annualized=e3,
        e4_high_funding_annualized=e4,
        e2_beats_e0=e2_beats_e0,
        e3_beats_e0=e3_beats_e0,
        e4_at_least_e0=e4_at_least_e0,
        structure_valid=structure_valid,
        reasons=tuple(reasons),
    )


def structure_verdict_payload(verdict: StructureVerdict) -> dict[str, Any]:
    return {
        "e0_high_funding_annualized": verdict.e0_high_funding_annualized,
        "e1_high_funding_annualized": verdict.e1_high_funding_annualized,
        "e2_high_funding_annualized": verdict.e2_high_funding_annualized,
        "e3_high_funding_annualized": verdict.e3_high_funding_annualized,
        "e4_high_funding_annualized": verdict.e4_high_funding_annualized,
        "e2_beats_e0": verdict.e2_beats_e0,
        "e3_beats_e0": verdict.e3_beats_e0,
        "e4_at_least_e0": verdict.e4_at_least_e0,
        "structure_valid": verdict.structure_valid,
        "reasons": list(verdict.reasons),
        "label": "단면 근거, 시계열 미검증 (cross-sectional evidence, time-series unvalidated)",
    }


# ---------------------------------------------------------------------------
# 3. Cross-venue (OKX-lend / Bitget-execute) structural exposure -- reported, never gated,
#    never treated as a probability (wave14 M6/M7 S6 precedent).
# ---------------------------------------------------------------------------


def exchange_separation_remaining_fraction() -> float:
    """wave14 gates14.py S6 precedent verbatim ('거래소 1곳 전액 손실 시 잔존 자본 55.00% ...
    시장 경로와 무관하게 고정'): wave16's OWN cross-venue split is OKX(spot custody+lending) vs
    Bitget(perp execution), a DIFFERENT reason for the SAME fixed 50/50-of-active-capital
    structure ($45 spot-lend leg / $45 perp leg / $10 untouched reserve) -- so the identical
    arithmetic applies: if the OKX-side leg is a TOTAL loss while a position happens to be open,
    only the reserve + the surviving Bitget leg remain. This is a STRUCTURAL ratio conditional on
    a position being open, not a backtested probability of OKX failing -- SPEC.md 치명적 한계 3
    explicitly forbids treating it as one."""
    leg = gates13.leg_usdt(L4_CONFIG)
    return (TOTAL_CAPITAL - leg) / TOTAL_CAPITAL


__all__ = [
    "DSR_CUMULATIVE_TRIALS",
    "StructureVerdict",
    "deflated_sharpe_reference",
    "evaluate_funding_only_gates",
    "evaluate_structure_validity",
    "exchange_separation_remaining_fraction",
    "structure_verdict_payload",
    "utilization",
]
