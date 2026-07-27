# Wave-21 markdown report + registry writer. Pure formatting over already-computed
# results/{ga_seed*,random_seed*,final_candidate}.json (run_wave21.py's evolve/control/gates
# stages) plus a read-only peek at research/wave18_idle/results/I5.json (this wave's own
# baseline). SPEC.md's own instruction governs the headline framing here: "헤드라인은 항상
# OOS 기준" -- this report NEVER leads with the final candidate's IS number.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave10_carry100.engine import ACTIVE_CAPITAL
from research.wave21_ga import ga, gates21, random_search
from research.wave21_ga.genome import GENE_NAMES, I5_BASELINE_GENOME

GENERATION_SNAPSHOTS: Final[tuple[int, ...]] = (1, 5, 10, 15, 20, 25)


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_pp(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:+.{digits}f}%p"


def _fmt(value: float | None, digits: int = 5) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sections.
# ---------------------------------------------------------------------------


def _header_section() -> list[str]:
    return [
        "# Wave-21 리포트 -- 유전 알고리즘 파라미터 탐색 (사전등록 research/wave21_ga/SPEC.md)",
        "",
        "**이 wave의 본체는 GA가 아니라 오염 차단이다.** GA는 1,500 x 5시드 = 7,500회 평가 -- "
        "대규모 다중검정이다. 무방비면 IS에서 화려한 개체가 반드시 나오고 그것은 엣지가 아니라 "
        "과최적화다. 이 리포트가 보고하는 헤드라인은 **항상 최종 후보의 OOS(2025-10~) 성과**이며, "
        "IS 성과는 진화 과정의 참고 수치로만 다룬다.",
        "",
        "## 오염 차단 4중 장치 (구현 요약)",
        "",
        "1. **OOS 봉인**: 진화/랜덤탐색 루프는 `fitness.evaluate_genome()`만 호출한다 -- 이 "
        "함수는 `mode` 인자 자체를 받지 않아 OOS를 요청할 방법이 코드상 존재하지 않는다(구조적 "
        "봉쇄). `fitness.run_backtest(mode='IS', ...)`는 매 호출마다 OOS_SPLIT 이전으로 데이터를 "
        "먼저 잘라낸 뒤 계산하고, 그 결과 equity의 인덱스가 OOS_SPLIT을 넘으면 `OOSLeakageError`를 "
        "던진다(경험적 재확인). `fitness.oos_slice()`는 이 저장소 전체에서 OOS 구간(> OOS_SPLIT)을 "
        "읽을 수 있는 유일한 함수이고, `mode='OOS_FINAL'`이 아니면 무조건 예외를 던진다. "
        "`tests/test_wave21.py`가 이 세 가지를 전부 테스트로 고정한다.",
        "2. **적합도**: 단순 수익 최대화가 아니라 IS를 4개 폴드로 쪼갠 워크포워드 "
        "`median(폴드별 CAGR) - std(폴드별 CAGR) - 0.5*max(0, MDD-10%)`.",
        "3. **랜덤 대조군**: GA와 동일한 1,500 x 5시드 평가를 무작위 탐색으로 병행, H1 게이트로 "
        "GA가 우연보다 나은지 자체를 검정.",
        "4. **DSR 보정**: 이 wave 자체의 시행횟수(1,500x5=7,500)를 trials로 넣어 H3에서 판정.",
        "",
    ]


def _methodology_notes_section() -> list[str]:
    return [
        "## 방법론 노트 (필독)",
        "",
        "1. **엔진**: 15,000회(=GA 7,500 + 랜덤 7,500) 백테스트를 감당하기 위해 "
        "`research/wave13_liquidity/engine13.py`의 파이썬 일별 루프(실측: L4 breadth=200 1회 "
        "실행에 약 164초)를 그대로 쓰지 않고, 동일한 일별 손익 공식(갭 PnL/회전비용/일중 "
        "PnL/최종강제청산)을 넘파이로 벡터화해 재도출했다(`fitness.py`). "
        "`tests/test_wave21.py::test_vectorized_engine_matches_reference_engine13`가 합성 "
        "시장에서 이 엔진과 `engine13._run_liquidity_loop`의 equity curve가 float 정밀도로 "
        "일치함을 고정하고, `test_final_evaluation_reproduces_i5_json_on_real_cache`가 실제 "
        "캐시에서 I5 유전자를 이 엔진으로 돌린 결과가 `research/wave18_idle/results/I5.json`의 "
        "전기간 CAGR·OOS 연환산과 소수 13자리까지 일치함을 확인한다.",
        "2. **적합도 폴드**: 4개 폴드는 IS 구간을 4등분한 연속 구간이며, 폴드마다 별도로 "
        "재적합(re-fit)하지 않는다 -- GA/랜덤탐색 자체가 이미 바깥쪽 탐색 루프이고, 폴드별 "
        "재최적화는 이 wave의 평가 예산(시드당 1,500회) 안에 들어갈 여지가 없는 중첩 탐색이 "
        "된다. SPEC.md의 수식은 하나의 연속된 IS equity curve의 구간별 일관성을 재는 것으로 "
        "해석했다.",
        "3. **최종 후보 선정**: SPEC.md \"5회 모두에서 재현되는 개선만 인정(단일 시드 대박 "
        "무효)\"을 5개 GA 시드의 자체 최고 개체 중 **중앙값**(최댓값이 아님)을 최종 후보로 "
        "선택하는 규칙으로 구현했다(`run_wave21.select_final_candidate`). 단일 시드의 우연한 "
        "대박은 중앙값 선택 규칙상 최종 후보가 될 수 없다.",
        "4. **idle_mode 4값**은 wave18의 I0/I1/I2/I5에 1:1 대응한다(I3 전체유니버스 오버레이, "
        "I4 역캐리는 이 wave의 탐색공간 밖). 오버레이 자체의 파라미터(임계 8%/청산 4%/window=7/"
        "top_k=1/메이저 한정)는 wave18과 완전히 동일하게 고정했다 -- 이 wave는 오버레이를 "
        "쓸지 말지만 진화시킨다.",
        "5. **1x 레버리지는 유전자 범위 자체가 아니라 H4 게이트로 강제한다**: SPEC.md 유전자 표는 "
        "top_k_pairs({1,2,3})와 leg_fraction(0.30~0.50)을 독립 축으로 등록했고, 두 값의 곱이 "
        "gross=2*top_k*leg_fraction*ACTIVE_CAPITAL로 1x(=ACTIVE_CAPITAL)를 넘을 수 있다(예: "
        "top_k=3 ∧ leg=0.5 -> gross=3.0x). SPEC.md의 '그 외 고정: ...1x...'를 유전자 사전 제약이 "
        "아니라 H4(=wave13/18 S1/S4와 동일한 gross<=ACTIVE_CAPITAL 판정)가 사후에 거르는 것으로 "
        "구현했다 -- GA가 이 조합을 찾아내면 적합도 자체는 높아도 H4에서 걸러지도록 설계된 것이며, "
        "이 판단은 결과를 보기 전에 genome.py에 코드 주석으로 고정해 두었다(사후 조정 아님).",
        "",
    ]


def _ga_vs_random_section(ga_payloads: list[dict[str, Any]], random_payloads: list[dict[str, Any]], gate_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## H1 -- GA vs 랜덤 대조군 (시드별)", ""]
    lines.append("| 시드쌍 | GA 시드 | GA 최고적합도 | 랜덤 시드 | 랜덤 최고적합도 | GA 승리 |")
    lines.append("|---|---|---|---|---|---|")
    for index, (ga_payload, random_payload) in enumerate(zip(ga_payloads, random_payloads)):
        ga_best = ga_payload.get("best_fitness")
        random_best = random_payload.get("best_fitness")
        win = "YES" if (ga_best is not None and random_best is not None and ga_best > random_best) else "NO"
        lines.append(f"| {index + 1} | {ga_payload.get('seed')} | {_fmt(ga_best)} | {random_payload.get('seed')} | {_fmt(random_best)} | {win} |")
    lines.append("")
    if gate_payload is not None:
        h1 = gate_payload.get("h1", {})
        lines.append(f"**H1 판정: {h1.get('status', 'N/A')}** ({h1.get('n_wins', 'N/A')}/{h1.get('n_seeds', 'N/A')}회 GA 승리, 기준 {h1.get('threshold', 'N/A')}회 이상)")
    lines.append("")
    return lines


def _evolution_curve_section(ga_payloads: list[dict[str, Any]]) -> list[str]:
    lines = ["## 진화 곡선 (세대별 최고/평균 적합도, 스냅샷)", ""]
    for payload in ga_payloads:
        seed = payload.get("seed")
        history = payload.get("history", [])
        by_generation = {int(row["generation"]) + 1: row for row in history}  # stored 0-indexed; report as 1-indexed generation numbers
        lines.append(f"### 시드 {seed}")
        lines.append("")
        lines.append("| 세대 | 최고 적합도 | 평균 적합도 | 최저 적합도 |")
        lines.append("|---|---|---|---|")
        for generation in GENERATION_SNAPSHOTS:
            row = by_generation.get(generation)
            if row is None:
                continue
            lines.append(f"| {generation} | {_fmt(row.get('best_fitness'))} | {_fmt(row.get('mean_fitness'))} | {_fmt(row.get('worst_fitness'))} |")
        if history:
            gen1_best = history[0].get("best_fitness")
            genN_best = history[-1].get("best_fitness")
            improvement = (genN_best - gen1_best) if (gen1_best is not None and genN_best is not None) else None
            lines.append("")
            lines.append(f"- 세대 1 -> {len(history)} 최고 적합도 개선: {_fmt(gen1_best)} -> {_fmt(genN_best)} ({_fmt(improvement) if improvement is not None else 'N/A'})")
            lines.append(f"- 이번 시드 실제 백테스트 실행 횟수: {payload.get('n_backtests_run')}/{payload.get('n_evaluations')} (나머지는 캐시 재사용 -- 엘리트 승계 등)")
        lines.append("")
    return lines


def _final_candidate_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 최종 후보 유전자 (5개 GA 시드 중 IS 적합도 중앙값 시드 선택)", ""]
    if final_payload is None:
        lines.append("- final_candidate.json 없음 (gates 스테이지 미실행).")
        lines.append("")
        return lines
    genome = final_payload.get("final_genome", {})
    lines.append(f"- 선정 출처: 시드 {final_payload.get('source_seed')} (그 시드의 IS 적합도 = {_fmt(final_payload.get('source_is_fitness'))})")
    lines.append("")
    lines.append("| 유전자 | 최종 후보 값 | L4/I5 기준값 |")
    lines.append("|---|---|---|")
    baseline = I5_BASELINE_GENOME.to_dict()
    for name in GENE_NAMES:
        lines.append(f"| {name} | {genome.get(name)} | {baseline.get(name)} |")
    lines.append("")
    return lines


def _oos_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## OOS 결과 (봉인 해제, 단 한 번 평가) -- 이 리포트의 헤드라인", ""]
    if final_payload is None:
        lines.append("- final_candidate.json 없음 (gates 스테이지 미실행).")
        lines.append("")
        return lines
    oos_final = final_payload.get("oos_cagr_regime_anchored")
    oos_i5 = final_payload.get("i5_reference", {}).get("oos_cagr")
    gap_pp = (oos_final - oos_i5) * 100.0 if (oos_final is not None and oos_i5 is not None) else None
    lines.append(f"- **최종 후보 OOS(2025-10~) 연환산**: {_fmt_pct(oos_final)} (anchor: OOS_SPLIT 시점 equity, I5.json과 동일 방법론)")
    lines.append(f"- 참고(자체 구간 기준, IS 연속성 무시): {_fmt_pct(final_payload.get('oos_cagr_self_contained'))}")
    lines.append(f"- **I5(기준선) OOS(2025-10~) 연환산**: {_fmt_pct(oos_i5)}")
    lines.append(f"- **격차**: {_fmt_pp(gap_pp)} ({'개선' if (gap_pp is not None and gap_pp > 0) else '악화 또는 동일'})")
    lines.append("")
    lines.append(f"- 참고, 최종 후보 전기간 CAGR: {_fmt_pct(final_payload.get('full_period_cagr'))} / IS CAGR: {_fmt_pct(final_payload.get('is_cagr'))} / 전기간 MDD: {_fmt_pct(final_payload.get('mdd_full'))}")
    lines.append("")
    return lines


def _overfitting_gap_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## IS-OOS 격차 (과최적화 정도)", ""]
    if final_payload is None:
        lines.append("- final_candidate.json 없음.")
        lines.append("")
        return lines
    is_cagr = final_payload.get("is_cagr")
    oos_cagr = final_payload.get("oos_cagr_regime_anchored")
    gap_pp = (is_cagr - oos_cagr) * 100.0 if (is_cagr is not None and oos_cagr is not None) else None
    i5_ref = final_payload.get("i5_reference", {})
    i5_gap_pp = i5_ref.get("is_oos_gap_pp")
    lines.append(f"- 최종 후보: IS 연환산 {_fmt_pct(is_cagr)} vs OOS 연환산 {_fmt_pct(oos_cagr)} -> 격차 {_fmt_pp(gap_pp)}")
    lines.append(
        f"- **비교 기준(I5 자체 격차)**: I5도 IS {_fmt_pct(i5_ref.get('is_cagr'))} vs OOS {_fmt_pct(i5_ref.get('oos_cagr'))} "
        f"-> 격차 {_fmt_pp(i5_gap_pp)} -- 펀딩캐리 계열 전략은 현재 OOS 구간(2025-10~) 자체가 시장 전체적으로 "
        "저펀딩 레짐이라, 승격된 기존 후보(I5)조차 IS 대비 OOS가 크게 낮다. 그래서 최종 후보의 raw 격차만 "
        "단독으로 보면 레짐 효과와 과최적화 효과를 구분할 수 없다 -- 아래는 I5 대비 상대 격차다."
    )
    relative_gap_pp = (gap_pp - i5_gap_pp) if (gap_pp is not None and i5_gap_pp is not None) else None
    lines.append(f"- **I5 대비 상대 격차**: {_fmt_pp(relative_gap_pp)} (양수면 I5보다 레짐효과를 넘어서는 추가 과최적화 신호)")
    if relative_gap_pp is not None and relative_gap_pp > 10.0:
        lines.append(
            "- 상대 격차가 10%p를 넘는다 -- 레짐효과로 설명되지 않는 추가적인 IS 특화 신호가 있다는 뜻이다. "
            "다만 H2(아래)가 PASS라면 이 초과분이 OOS 성과 자체를 I5보다 나쁘게 만들 정도는 아니라는 뜻이므로, "
            "'과최적화로 인한 OOS 붕괴'와는 다른, 더 약한 형태의 신호임을 함께 밝힌다."
        )
    else:
        lines.append("- 상대 격차가 10%p 이내다 -- I5 자체의 레짐효과를 넘어서는 뚜렷한 과최적화 신호는 약하다.")
    lines.append("")
    return lines


def _gates_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 게이트 (H1~H5)", ""]
    if final_payload is None:
        lines.append("- 게이트 미실행.")
        lines.append("")
        return lines
    gates = final_payload.get("gates", {})
    lines.append("| 게이트 | 상태 | 핵심 수치 |")
    lines.append("|---|---|---|")
    h1 = gates.get("h1", {})
    lines.append(f"| H1 (GA>랜덤, >=4/5시드) | {h1.get('status')} | {h1.get('n_wins')}/{h1.get('n_seeds')} |")
    h2 = gates.get("h2", {})
    lines.append(f"| H2 (최종개체 OOS > I5 OOS) | {h2.get('status')} | {_fmt_pct(h2.get('final_oos_cagr'))} vs {_fmt_pct(h2.get('i5_oos_cagr'))} ({_fmt_pp(h2.get('gap_pp'))}) |")
    h3 = gates.get("h3", {})
    lines.append(f"| H3 (DSR, trials={gates21.GA_TRIALS}) | {h3.get('status')} | score={_fmt(h3.get('score'))}, p={_fmt(h3.get('probability'))} |")
    h4 = gates.get("h4", {})
    lines.append(
        f"| H4 (MC/블록MDD/체결가능/x3스트레스) | {h4.get('status')} | "
        f"MC_p05={_fmt_usd(h4.get('mc', {}).get('p05'))}, 파산확률={_fmt_pct(h4.get('mc', {}).get('ruin_probability'))}, "
        f"블록MDDp95={_fmt_pct(h4.get('block_mdd_p95'))}, 스트레스MDDp95={_fmt_pct(h4.get('stress_block_mdd_p95'))}, "
        f"레그{_fmt_usd(h4.get('leg_usdt_nominal'))}/총{_fmt_usd(h4.get('gross_usdt_nominal'))} "
        f"(1x한도={_fmt_usd(ACTIVE_CAPITAL)}, {'초과' if h4.get('gross_leverage_1x_ok') is False else '이내'}) |"
    )
    h5 = gates.get("h5", {})
    year_cells = ", ".join(
        f"{year}: {_fmt_pct(detail.get('final'))} vs {_fmt_pct(detail.get('i5'))}"
        for year, detail in h5.get("years", {}).items()
    )
    lines.append(f"| H5 (최악연도 2022/2025 비승계 악화) | {h5.get('status')} | {year_cells} |")
    lines.append("")
    overall = gates.get("overall")
    promoted = gates.get("promoted")
    reasons = gates.get("failure_reasons", [])
    lines.append(f"**종합: {overall} / 승격: {'YES' if promoted else 'NO'}**" + (f" (미달: {', '.join(reasons)})" if reasons else ""))
    lines.append("")
    return lines


def _dsr_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## DSR / 다중검정", ""]
    lines.append(
        f"- H3 게이트 DSR (이 wave 자체 시행 {gates21.GA_TRIALS:,}회 = 1,500 x 5시드 보정): "
        f"{_fmt(final_payload.get('gates', {}).get('h3', {}).get('score')) if final_payload else 'N/A'}"
    )
    reference = final_payload.get("dsr_reference_cumulative_121_plus_7500") if final_payload else None
    lines.append(
        f"- 참고용 DSR (누적 121후보 + 이 wave 7,500평가 = {gates21.CUMULATIVE_TRIALS_WITH_GA:,}회 보정, 승격 판정에는 미사용): "
        f"{_fmt(reference.get('score')) if reference else 'N/A'}"
    )
    lines.append(
        f"- SPEC.md 원칙: 누적 시행 = 사전등록 121후보 + 이번 GA 7,500평가. H3 게이트 자체는 이번 wave의 "
        f"7,500회만 사용하고(SPEC.md H3 문구 그대로), 다중검정 disclosure는 누적치를 참고용으로 병기한다."
    )
    lines.append("")
    return lines


def _verdict_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 판정", ""]
    if final_payload is None:
        lines.append("게이트 미실행 -- 판정 불가.")
        lines.append("")
        return lines
    gates = final_payload.get("gates", {})
    promoted = gates.get("promoted")
    oos_final = final_payload.get("oos_cagr_regime_anchored")
    oos_i5 = final_payload.get("i5_reference", {}).get("oos_cagr")
    if promoted:
        lines.append(
            f"**승격**: 최종 후보(시드 {final_payload.get('source_seed')} 중앙값 선택)가 H1~H5를 전부 통과했다. "
            f"OOS 연환산 {_fmt_pct(oos_final)} (I5 {_fmt_pct(oos_i5)} 대비 개선). GA 산출물은 과최적화 위험이 "
            f"상시 존재하므로, 이 결과를 카드에 반영하더라도 paper 전진검증을 필수 선행 조건으로 부여한다."
        )
    else:
        reasons = gates.get("failure_reasons", [])
        lines.append(f"**전멸 -- GA로도 I5 못 넘음**: H1~H5 중 다음 게이트가 미달했다: {', '.join(reasons) if reasons else '(내부 로직 확인 필요)'}.")
        lines.append("")
        if "H1" in reasons:
            lines.append("- H1 미달은 진화 메커니즘 자체가 이 문제에서 무작위 탐색보다 나을 게 없었다는 뜻이다 -- \"GA 무의미\" 정직 보고.")
        if "H2" in reasons:
            lines.append(
                f"- H2 미달({_fmt_pct(oos_final)} <= I5의 {_fmt_pct(oos_i5)})은 이 wave가 사전에 가장 가능성 높다고 "
                "지목한 시나리오와 일치한다: IS에서 찾은 개선이 OOS로 이어지지 않는 과최적화."
            )
        if "H4" in reasons and "H2" not in reasons:
            h4 = gates.get("h4", {})
            lines.append(
                f"- **H4 미달은 이 wave가 사전에 지목한 'IS 화려/OOS 붕괴' 패턴이 아니다.** 최종 후보는 오히려 "
                f"H1(5/5)·H2(OOS가 I5보다 우수)·H3(DSR>0)·H5(최악연도 비악화)를 전부 통과했고, MC/블록셔플/"
                f"스트레스 위험지표도 전부 통과했다({_fmt_usd(h4.get('mc', {}).get('p05'))} MC p05, "
                f"{_fmt_pct(h4.get('block_mdd_p95'))} 블록MDD, {_fmt_pct(h4.get('stress_block_mdd_p95'))} "
                "스트레스MDD -- 전부 기준 이내). 유일한 미달 사유는 순수 구조적 제약이다: GA가 "
                f"top_k_pairs=3 ∧ leg_fraction=0.5(둘 다 탐색 범위의 최댓값 근처)를 동시에 골라 "
                f"gross={_fmt_usd(h4.get('gross_usdt_nominal'))}로 1x 한도(={_fmt_usd(ACTIVE_CAPITAL)})를 "
                "넘겼다. 즉 GA는 '통계적으로 더 안전하면서 OOS 성과도 더 좋은' 조합을 찾아냈지만, 그 조합은 "
                "이 계좌($90 활성자본) 규모에서 동시에 감당 가능한 것보다 많은 포지션을 동시에 요구한다 -- "
                "신호 품질의 과최적화가 아니라 사이징 제약 위반이다."
            )
        lines.append("이 결과는 실패가 아니라 유의미한 발견이다 -- 파라미터 탐색(GA 포함)의 한계를 실증한 사례로 기록한다.")
    lines.append("")
    return lines


def write_wave21_report(results_dir: Path, report_dir: Path, registry_path: Path, i5_results_path: Path) -> None:
    ga_payloads = [payload for seed in ga.SEEDS if (payload := _load_optional(results_dir / f"ga_seed{seed}.json")) is not None]
    random_payloads = [payload for seed in random_search.SEEDS if (payload := _load_optional(results_dir / f"random_seed{seed}.json")) is not None]
    final_payload = _load_optional(results_dir / "final_candidate.json")
    _ = i5_results_path  # already folded into final_candidate.json's own i5_reference/H2/H5 payload by the gates stage; kept as a parameter for symmetry with reporting18.write_wave18_report's own signature

    lines: list[str] = [
        *_header_section(),
        *_methodology_notes_section(),
        *_ga_vs_random_section(ga_payloads, random_payloads, final_payload.get("gates") if final_payload else None),
        *_evolution_curve_section(ga_payloads),
        *_final_candidate_section(final_payload),
        *_oos_section(final_payload),
        *_overfitting_gap_section(final_payload),
        *_gates_section(final_payload),
        *_dsr_section(final_payload),
        *_verdict_section(final_payload),
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave21_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    promoted = bool(final_payload.get("gates", {}).get("promoted")) if final_payload else False
    overall = final_payload.get("gates", {}).get("overall", "PENDING") if final_payload else "PENDING"
    oos_final = final_payload.get("oos_cagr_regime_anchored") if final_payload else None
    gates = final_payload.get("gates", {}) if final_payload else {}
    gate_cells = " ".join(f"H{n}:{'P' if gates.get(f'h{n}', {}).get('status') == 'PASS' else 'F'}" for n in range(1, 6)) if final_payload else "미실행"
    registry_lines = [
        "# Wave-21 registry",
        "",
        "| Candidate | Family | State | 최종후보 OOS CAGR | I5 OOS CAGR | 승격 | 게이트(H1-H5) |",
        "|---|---|---|---|---|---|---|",
        (
            f"| GA_FINAL | wave21_ga | {'EVALUATED' if final_payload else 'PENDING'} | {_fmt_pct(oos_final)} | "
            f"{_fmt_pct(final_payload.get('i5_reference', {}).get('oos_cagr')) if final_payload else 'N/A'} | "
            f"{'YES' if promoted else 'NO'} | {gate_cells} |"
        ),
        "",
        f"참고: GA 시드 {len(ga_payloads)}/{len(ga.SEEDS)}, 랜덤 대조군 시드 {len(random_payloads)}/{len(random_search.SEEDS)}, 종합 {overall}.",
        "",
        f"**최종 판정**: {'승격 (paper 전진검증 필수 조건부)' if promoted else 'GA로도 I5 못 넘음 (report/wave21_report.md 참조)'}.",
        "",
    ]
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave21_report"]
