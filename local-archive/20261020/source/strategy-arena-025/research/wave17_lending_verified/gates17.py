# Wave-17 gate evaluation. Deliberately NARROWER than wave16's gates16.py: SPEC.md 방법 5
# registers exactly THREE comparisons (F1 beats F0, F2 still beats F0, F3 == F0), not wave16's
# full S1-S5 statistical battery. This is not a shortcut taken silently -- SPEC.md's own 방법
# section explains why re-running S1-S5 here would be redundant: every F0-F3 candidate in this
# wave shares `ranking_lending_discount=0.0` (this wave only re-derives E1-style candidates,
# SPEC.md 범위), so EVERY candidate's funding-only companion is bit-identical to F0/wave16-E0,
# which already passed S1-S5 in research/wave16_duallayer/results/E0.json
# (`gates_funding_only.overall`) -- re-running the MC bootstrap / block-shuffle on an identical
# series would reproduce the exact same PASS, not new evidence.
#
# Comparison metric: `regime_breakdown(...).get('high_funding_mean_annualized_return')` on each
# candidate's OWN `combined` (lending-inclusive) equity -- the SAME established cross-wave
# promotion metric wave14/15/16 all use (research.wave10_carry100.regime.regime_breakdown),
# imported, not redefined.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave10_carry100.engine import Wave10Result
from research.wave10_carry100.regime import regime_breakdown

IDENTITY_ABS_TOLERANCE: Final = 1e-9  # F3 vs F0: expected to be an EXACT memoized-cache alias (see recompute17.py docstring); tolerance only guards against future code drift, not real noise


@dataclass(frozen=True, slots=True)
class Wave17Verdict:
    f0_high_funding_annualized: float | None
    f1_high_funding_annualized: float | None
    f2_high_funding_annualized: float | None
    f3_high_funding_annualized: float | None
    f0_full_period_annualized: float | None
    f1_full_period_annualized: float | None
    f1_beats_f0: bool
    f2_beats_f0: bool
    f3_equals_f0: bool
    f3_abs_diff: float | None
    verdict_valid: bool
    reasons: tuple[str, ...]


def _high_funding_annualized(result: Wave10Result) -> float | None:
    regime = regime_breakdown(result)
    value = regime.get("high_funding_mean_annualized_return")
    return float(value) if value is not None else None


def _full_period_annualized(result: Wave10Result) -> float | None:
    equity = result.equity
    if len(equity) < 2:
        return None
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    if start_value <= 0.0:
        return None
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0, 1.0)
    total_return = end_value / start_value - 1.0
    growth = 1.0 + total_return
    if growth <= 0.0:
        return -1.0
    return float(growth ** (365.0 / days) - 1.0)


def evaluate_wave17(combined_results_by_id: dict[str, Wave10Result]) -> Wave17Verdict:
    """SPEC.md 판정: 'F1 > F0 ∧ F2 > F0 ∧ F3 == F0(허용오차) -> 실수취 기준 재확인 유효.'
    Requires F0/F1/F2/F3 all present (F_min is reference-only, never gated -- SPEC.md 후보
    표: '추가 참고(게이트 대상 아님)')."""
    missing = {"F0", "F1", "F2", "F3"} - set(combined_results_by_id)
    if missing:
        raise KeyError(f"evaluate_wave17 needs F0-F3, missing: {sorted(missing)}")

    f0 = _high_funding_annualized(combined_results_by_id["F0"])
    f1 = _high_funding_annualized(combined_results_by_id["F1"])
    f2 = _high_funding_annualized(combined_results_by_id["F2"])
    f3 = _high_funding_annualized(combined_results_by_id["F3"])

    f1_beats_f0 = f0 is not None and f1 is not None and f1 > f0
    f2_beats_f0 = f0 is not None and f2 is not None and f2 > f0
    f3_abs_diff = abs(f3 - f0) if (f0 is not None and f3 is not None) else None
    f3_equals_f0 = f3_abs_diff is not None and f3_abs_diff <= IDENTITY_ABS_TOLERANCE

    verdict_valid = f1_beats_f0 and f2_beats_f0 and f3_equals_f0
    reasons: list[str] = []
    if not f1_beats_f0:
        reasons.append("F1(실측 lendingRate 중앙값)이 F0 대비 개선 실패")
    if not f2_beats_f0:
        reasons.append("F2(50% 보수 할인)가 F0 대비 개선 유지 실패")
    if not f3_equals_f0:
        reasons.append(f"F3가 F0와 동일하지 않음(diff={f3_abs_diff}) -- 무결성 실패, 엔진 재사용 자체를 재점검할 것")

    return Wave17Verdict(
        f0_high_funding_annualized=f0,
        f1_high_funding_annualized=f1,
        f2_high_funding_annualized=f2,
        f3_high_funding_annualized=f3,
        f0_full_period_annualized=_full_period_annualized(combined_results_by_id["F0"]),
        f1_full_period_annualized=_full_period_annualized(combined_results_by_id["F1"]),
        f1_beats_f0=f1_beats_f0,
        f2_beats_f0=f2_beats_f0,
        f3_equals_f0=f3_equals_f0,
        f3_abs_diff=f3_abs_diff,
        verdict_valid=verdict_valid,
        reasons=tuple(reasons),
    )


def verdict_payload(verdict: Wave17Verdict) -> dict[str, Any]:
    return {
        "f0_high_funding_annualized": verdict.f0_high_funding_annualized,
        "f1_high_funding_annualized": verdict.f1_high_funding_annualized,
        "f2_high_funding_annualized": verdict.f2_high_funding_annualized,
        "f3_high_funding_annualized": verdict.f3_high_funding_annualized,
        "f0_full_period_annualized": verdict.f0_full_period_annualized,
        "f1_full_period_annualized": verdict.f1_full_period_annualized,
        "f1_beats_f0": verdict.f1_beats_f0,
        "f2_beats_f0": verdict.f2_beats_f0,
        "f3_equals_f0": verdict.f3_equals_f0,
        "f3_abs_diff": verdict.f3_abs_diff,
        "verdict_valid": verdict.verdict_valid,
        "reasons": list(verdict.reasons),
        "label": "단면 근거, 시계열 미검증 (cross-sectional evidence, time-series unvalidated) -- wave16과 동일 라벨 유지",
    }


__all__ = ["IDENTITY_ABS_TOLERANCE", "Wave17Verdict", "evaluate_wave17", "verdict_payload"]
