# Wave-22 pre-registered verdict combiner (task's own "판정 기준 (사전등록)" table, copied
# verbatim):
#   PASS (진짜 엣지):  stability_ratio >= 0.8  AND  rolling_win_rate >= 55%  AND
#                      improvement spread across >=2 axes  AND  DSR > 0  AND
#                      random-control top 5%
#   CONDITIONAL:       partially met -> "paper 전진검증 필수" grade
#   FAIL (과최적화):    stability_ratio < 0.6  OR  improvement concentrated in a single axis  OR
#                      rolling_win_rate < 50%
#
# Precedence (this module's own, undisclosed-in-the-task-text but necessary to make the two
# rules jointly total): FAIL's three OR-conditions are checked FIRST and, if any fires,
# override everything else -- they are the task's own bright-line "this alone is disqualifying"
# rules. Only if none of them fire does this module check whether ALL FIVE PASS conditions hold;
# if not, the result is CONDITIONAL. This ordering is the only reading under which every input
# combination maps to exactly one of the three verdicts with no gap and no ambiguity.

from __future__ import annotations

from typing import Any, Final

STABILITY_PASS_THRESHOLD: Final = 0.8
STABILITY_FAIL_THRESHOLD: Final = 0.6
ROLLING_WIN_RATE_PASS_THRESHOLD: Final = 0.55
ROLLING_WIN_RATE_FAIL_THRESHOLD: Final = 0.50


def combine(sensitivity: dict[str, Any], rolling: dict[str, Any], regime: dict[str, Any], dsr: dict[str, Any], attribution: dict[str, Any], shuffle_control: dict[str, Any]) -> dict[str, Any]:
    stability_ratio = sensitivity["overall"]["primary_value"]
    rolling_win_rate = rolling["g1_win_rate"]
    spread_2plus = attribution["forward_concentration"]["spread_across_2plus"]
    concentrated_single_axis = attribution["forward_concentration"]["concentrated_single_axis"]
    dsr_score = dsr["g1_dsr_score_cumulative"]
    dsr_positive = dsr["g1_dsr_positive_at_cumulative_trials"]
    shuffle_top5 = shuffle_control["g1_in_top_5pct_full_cagr"]

    missing = [name for name, value in {
        "stability_ratio": stability_ratio, "rolling_win_rate": rolling_win_rate,
        "concentrated_single_axis": concentrated_single_axis, "dsr_score": dsr_score, "shuffle_top5": shuffle_top5,
    }.items() if value is None]
    if missing:
        raise RuntimeError(f"verdict.combine: required inputs missing from upstream validation results: {missing}")

    fail_reasons: list[str] = []
    if stability_ratio < STABILITY_FAIL_THRESHOLD:
        fail_reasons.append(f"stability_ratio {stability_ratio:.3f} < {STABILITY_FAIL_THRESHOLD} (worst axis: {sensitivity['overall']['worst_axis']})")
    if concentrated_single_axis:
        fail_reasons.append(f"improvement concentrated in single axis '{attribution['forward_concentration']['top_axis']}' (share={attribution['forward_concentration']['top_share']:.1%} > {attribution['methodology']['concentration_threshold']:.0%})")
    if rolling_win_rate < ROLLING_WIN_RATE_FAIL_THRESHOLD:
        fail_reasons.append(f"rolling_win_rate {rolling_win_rate:.1%} < {ROLLING_WIN_RATE_FAIL_THRESHOLD:.0%}")

    pass_checks = {
        "stability_ratio_ge_0_8": stability_ratio >= STABILITY_PASS_THRESHOLD,
        "rolling_win_rate_ge_55pct": rolling_win_rate >= ROLLING_WIN_RATE_PASS_THRESHOLD,
        "spread_across_2plus_axes": bool(spread_2plus),
        "dsr_positive": bool(dsr_positive),
        "shuffle_control_top5pct": bool(shuffle_top5),
    }
    all_pass_checks_met = all(pass_checks.values())

    if fail_reasons:
        overall = "FAIL"
    elif all_pass_checks_met:
        overall = "PASS"
    else:
        overall = "CONDITIONAL"

    unmet_pass_checks = [name for name, met in pass_checks.items() if not met]

    if overall == "PASS":
        recommendation = "G1 승격 유지 -- 6종 검증 전부 사전등록 기준 충족. 실거래 투입 가능(기존 wave21 H1-H5 게이트와 별개로 이 wave의 판정도 통과)."
    elif overall == "FAIL":
        recommendation = "G1 승격 철회 권고 -- 과최적화 신호(사전등록 FAIL 기준 " + "; ".join(fail_reasons) + ")가 확인됨. 현재 파라미터 조합을 실거래에 투입하지 말 것. I5를 현행 운용 상한으로 유지."
    else:
        recommendation = "G1 승격을 'paper 전진검증 필수' 조건부로 유지 -- 일부 기준만 충족(미충족: " + ", ".join(unmet_pass_checks) + "). 실거래 자본 투입 전 paper 전진검증(실시간, out-of-sample) 결과를 추가로 확인할 것."

    return {
        "methodology": {
            "source": "task pre-registered criteria table (verbatim), FAIL-first precedence (see module docstring)",
            "stability_pass_threshold": STABILITY_PASS_THRESHOLD,
            "stability_fail_threshold": STABILITY_FAIL_THRESHOLD,
            "rolling_win_rate_pass_threshold": ROLLING_WIN_RATE_PASS_THRESHOLD,
            "rolling_win_rate_fail_threshold": ROLLING_WIN_RATE_FAIL_THRESHOLD,
        },
        "inputs": {
            "stability_ratio": stability_ratio,
            "rolling_win_rate": rolling_win_rate,
            "spread_across_2plus_axes": spread_2plus,
            "concentrated_single_axis": concentrated_single_axis,
            "concentration_top_axis": attribution["forward_concentration"]["top_axis"],
            "dsr_score_cumulative": dsr_score,
            "dsr_positive": dsr_positive,
            "shuffle_top5pct_full_cagr": shuffle_top5,
            "shuffle_g1_top_pct": shuffle_control["rank_by_full_cagr"]["g1_top_pct_of_pooled_31"],
            "dominant_regime": regime["dominant_regime_by_magnitude"],
        },
        "fail_reasons": fail_reasons,
        "pass_checks": pass_checks,
        "unmet_pass_checks": unmet_pass_checks,
        "overall": overall,
        "recommendation": recommendation,
    }


__all__ = ["ROLLING_WIN_RATE_FAIL_THRESHOLD", "ROLLING_WIN_RATE_PASS_THRESHOLD", "STABILITY_FAIL_THRESHOLD", "STABILITY_PASS_THRESHOLD", "combine"]
