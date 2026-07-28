# Wave-23 markdown report + registry writer. Pure formatting over already-computed
# results/{ga_seed*,random_seed*,final_candidate}.json (run_wave23.py's evolve/control/gates
# stages). Task instruction: "헤드라인은 OOS 기준" -- this report NEVER leads with the final
# candidate's IS number, and always reports the SPEC.md fitness metric (not CAGR/Sharpe) as
# the headline, since CAGR is explicitly NOT this wave's objective.

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
from research.wave23_ga_short import ga23, gates23, random_search23
from research.wave23_ga_short.genome23 import GENE_NAMES

GENERATION_SNAPSHOTS: Final[tuple[int, ...]] = (1, 5, 10, 15, 20, 25)


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


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
        "# Wave-23 리포트 -- GA 재탐색: 단기 수익최대화 (사전등록 research/wave23_ga_short/SPEC.md)",
        "",
        "**wave-21과 목표가 다르다**: wave-21은 워크포워드 CAGR 중앙값-표준편차(안정형)를 적합도로 "
        "썼고 캐리 변형(G1)을 찾았지만, 그 G1은 정식 DSR 재계산에서 -0.7597(누적시행 보정)로 조건부 "
        "격하됐다(research/wave22_overfit). **이번 wave는 적합도 자체를 60일 롤링창 상위 25% 평균 "
        "수익률 - 3xP(창 < -20%)로 재정의**하고, 전략 종류(strategy_kind) 자체를 유전자에 포함시켜 "
        "GA가 5개 전략 계열(carry/momentum/breakout/funding_spike/convex_dual) 중 무엇을 선호하는지도 "
        "함께 관찰한다. **헤드라인은 항상 최종 후보의 OOS(2025-10~) 단기창 성과**이며, IS 성과는 "
        "진화 과정의 참고 수치로만 다룬다.",
        "",
        "## 오염 차단 (구현 요약)",
        "",
        "1. **OOS 봉인**: 진화/랜덤탐색 루프는 `fitness23.evaluate_genome()`만 호출한다 -- 이 함수는 "
        "`mode` 인자 자체를 받지 않는다(구조적 봉쇄). `engine23.run_backtest(mode='IS', ...)`는 매 "
        "호출마다 OOS_SPLIT 이전으로 먼저 잘라낸 뒤 계산하고, 결과 equity 인덱스가 OOS_SPLIT을 넘으면 "
        "`OOSLeakageError`를 던진다. `engine23.oos_slice()`가 OOS 구간을 읽을 수 있는 유일한 함수다.",
        "2. **랜덤 대조군**: GA와 동일한 1,500 x 5시드 평가를 무작위 탐색으로 병행(K1).",
        "3. **시드 5개**: 5개 GA 시드의 자체 최고 개체 중 **중앙값**(최댓값이 아님)을 최종 후보로 "
        "선택 -- 단일 시드의 우연한 대박은 최종 후보가 될 수 없다.",
        "4. **DSR을 최종 후보 자신의 equity curve로 계산**(K3) -- wave-21이 놓친 검증을 이번엔 처음부터 "
        "게이트에 내장했다: H3 게이트는 항상 `final_evaluation()`이 반환한 바로 그 개체의 곡선만 쓴다.",
        "",
    ]


def _methodology_notes_section() -> list[str]:
    return [
        "## 방법론 노트 (필독)",
        "",
        "1. **엔진**: 5개 전략 종류가 SPEC.md의 8축 유전자(strategy_kind 포함)를 공유하므로, "
        "`engine23.py`는 신호 계산(전략별)과 포지션 라이프사이클(공통, 손절/익절/보유기간상한/동시보유수)을 "
        "분리했다. 손절/익절/보유기간 종료는 경로의존적이라(며칠째 보유 중인지, 진입가 대비 누적수익률이 "
        "얼마인지) wave21_ga의 순수 히스테리시스처럼 룩어헤드 없이 완전히 벡터화할 수 없다 -- 날짜 축에 "
        "대한 파이썬 루프가 불가피하지만, 그 루프 내부는 심볼 축에 대해 넘파이로 벡터화했다(실측 "
        "~0.15-0.27초/평가, 15,000회 평가 예산 안에 들어오는 속도).",
        "2. **신호 표준화**: 5개 전략 모두 '진입강도(entry_z)'라는 동일한 유전자 축을 공유하므로, 각 "
        "전략의 원신호(캐리=funding_score 7일, funding_spike=funding_score 3일, momentum=30일 가격모멘텀, "
        "breakout=(종가-SMA20)/ATR14, convex_dual=모멘텀·브레이크아웃 부호 일치시 평균)를 심볼별 트레일링 "
        "180일(최소 60일) 롤링 z-score로 통일해 entry_z가 5개 전략에서 동일한 의미(자기 자신의 최근 "
        "이력 대비 얼마나 극단적인가)를 갖도록 설계했다. carry/funding_spike는 롱온리(캐리는 스프레드를 "
        "역방향으로 파는 개념이 없고, funding_spike는 wave20 V2의 극단펀딩 스퀴즈 추격 컨벤션을 그대로 "
        "따른다), momentum/breakout/convex_dual은 양방향이다.",
        "3. **레버리지 1x는 유전자 자체에서 구조적으로 강제**(사후 게이트 아님): "
        "`genome23.Genome.normalized_weight`가 `position_fraction x max_concurrent`가 1.0을 넘지 "
        "못하도록 항상 나눠 낮춘다. wave21_ga는 이 조합을 H4 게이트로 사후에 걸렀는데, 그 결과 GA가 "
        "예산을 실행불가능한 구간 탐색에 낭비했고, 사후에 유전자를 수동 수정한 G1의 DSR은 원래 게이트를 "
        "통과한 적이 없었다(wave22_overfit). 이번엔 그 사고 자체가 재발할 수 없도록 설계했다.",
        "4. **최종 후보 선정**: 5개 GA 시드의 자체 최고 개체 중 **중앙값**을 최종 후보로 선택한다"
        "(`run_wave23.select_final_candidate`, wave21_ga와 동일 규칙).",
        "5. **누적 다중검정**: K3 게이트 자체가 누적시행(121+7,500+15,000=22,621)으로 보정된 DSR을 "
        "쓴다 -- wave21_ga가 게이트용 DSR과 disclosure용 DSR을 분리해서 결과적으로 혼선을 만들었던 것과 "
        "달리, 이번엔 게이트 자체를 가장 보수적인 숫자로 고정했다.",
        "",
    ]


def _k1_section(ga_payloads: list[dict[str, Any]], random_payloads: list[dict[str, Any]], gate_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## K1 -- GA vs 랜덤 대조군 (시드별)", ""]
    lines.append("| 시드쌍 | GA 시드 | GA 최고적합도 | 랜덤 시드 | 랜덤 최고적합도 | GA 승리 |")
    lines.append("|---|---|---|---|---|---|")
    for index, (ga_payload, random_payload) in enumerate(zip(ga_payloads, random_payloads)):
        ga_best = ga_payload.get("best_fitness")
        random_best = random_payload.get("best_fitness")
        win = "YES" if (ga_best is not None and random_best is not None and ga_best > random_best) else "NO"
        lines.append(f"| {index + 1} | {ga_payload.get('seed')} | {_fmt(ga_best)} | {random_payload.get('seed')} | {_fmt(random_best)} | {win} |")
    lines.append("")
    if gate_payload is not None:
        k1 = gate_payload.get("k1", {})
        lines.append(f"**K1 판정: {k1.get('status', 'N/A')}** ({k1.get('n_wins', 'N/A')}/{k1.get('n_seeds', 'N/A')}회 GA 승리, 기준 {k1.get('threshold', 'N/A')}회 이상)")
    lines.append("")
    return lines


def _evolution_curve_section(ga_payloads: list[dict[str, Any]]) -> list[str]:
    lines = ["## 진화 곡선 (세대별 최고/평균 적합도, 스냅샷)", ""]
    for payload in ga_payloads:
        seed = payload.get("seed")
        history = payload.get("history", [])
        by_generation = {int(row["generation"]) + 1: row for row in history}
        lines.append(f"### 시드 {seed}")
        lines.append("")
        lines.append("| 세대 | 최고 적합도 | 평균 적합도 | 최저 적합도 | 최고개체 전략 |")
        lines.append("|---|---|---|---|---|")
        for generation in GENERATION_SNAPSHOTS:
            row = by_generation.get(generation)
            if row is None:
                continue
            kind = row.get("best_genome", {}).get("strategy_kind", "N/A")
            lines.append(f"| {generation} | {_fmt(row.get('best_fitness'))} | {_fmt(row.get('mean_fitness'))} | {_fmt(row.get('worst_fitness'))} | {kind} |")
        if history:
            gen1_best = history[0].get("best_fitness")
            genN_best = history[-1].get("best_fitness")
            improvement = (genN_best - gen1_best) if (gen1_best is not None and genN_best is not None) else None
            lines.append("")
            lines.append(f"- 세대 1 -> {len(history)} 최고 적합도 개선: {_fmt(gen1_best)} -> {_fmt(genN_best)} ({_fmt(improvement) if improvement is not None else 'N/A'})")
            lines.append(f"- 이번 시드 실제 백테스트 실행 횟수: {payload.get('n_backtests_run')}/{payload.get('n_evaluations')} (나머지는 캐시 재사용)")
        lines.append("")
    return lines


def _strategy_kind_distribution_section(ga_payloads: list[dict[str, Any]], final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 전략 종류 분포 (진화가 무엇을 선호했는가)", ""]
    lines.append("### 5개 GA 시드, 최종세대(25) 개체 60개씩의 전략 종류 분포")
    lines.append("")
    lines.append("| 전략 | " + " | ".join(f"시드{p.get('seed')}" for p in ga_payloads) + " | 합계 |")
    lines.append("|---|" + "---|" * (len(ga_payloads) + 1))
    kinds = ["carry", "momentum", "breakout", "funding_spike", "convex_dual"]
    totals = {kind: 0 for kind in kinds}
    for kind in kinds:
        row_counts = []
        for payload in ga_payloads:
            count = int(payload.get("final_population_kind_counts", {}).get(kind, 0))
            row_counts.append(count)
            totals[kind] += count
        lines.append(f"| {kind} | " + " | ".join(str(c) for c in row_counts) + f" | {totals[kind]} |")
    grand_total = sum(totals.values())
    lines.append("")
    if grand_total > 0:
        ranked = sorted(totals.items(), key=lambda item: -item[1])
        lines.append("- 선호 순위: " + ", ".join(f"{kind}({count}, {count / grand_total * 100.0:.1f}%)" for kind, count in ranked))
    if final_payload is not None:
        lines.append(f"- **최종 후보의 전략 종류**: {final_payload.get('final_genome', {}).get('strategy_kind', 'N/A')}")
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
    lines.append("| 유전자 | 최종 후보 값 |")
    lines.append("|---|---|")
    for name in GENE_NAMES:
        lines.append(f"| {name} | {genome.get(name)} |")
    lines.append("")
    return lines


def _oos_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## OOS 결과 (봉인 해제, 단 한 번 평가) -- 이 리포트의 헤드라인", ""]
    if final_payload is None:
        lines.append("- final_candidate.json 없음 (gates 스테이지 미실행).")
        lines.append("")
        return lines
    oos_fitness = final_payload.get("oos_fitness")
    i5_oos_fitness = final_payload.get("i5_oos_fitness")
    gap = (oos_fitness["fitness"] - i5_oos_fitness) if (oos_fitness is not None and i5_oos_fitness is not None) else None
    lines.append(f"- **최종 후보 OOS(2025-10~) 단기창 적합도** (SPEC.md 공식, 60일 롤링창 상위25%평균 - 3xP(<-20%)): {_fmt(oos_fitness['fitness']) if oos_fitness else 'N/A'}")
    if oos_fitness is not None:
        lines.append(f"  - 상위25%창 평균수익률: {_fmt_pct(oos_fitness.get('top_quantile_mean_return'))} / P(창<-20%): {_fmt_pct(oos_fitness.get('p_window_ruin'))} / 창 개수: {oos_fitness.get('n_windows')}")
    lines.append(f"- **I5(기준선) OOS 단기창 적합도**: {_fmt(i5_oos_fitness)}")
    lines.append(f"- **격차**: {_fmt(gap) if gap is not None else 'N/A'} ({'개선' if (gap is not None and gap > 0) else '악화 또는 동일'})")
    lines.append("")
    lines.append(f"- 참고, 최종 후보 전기간 CAGR: {_fmt_pct(final_payload.get('full_period_cagr'))} / 전기간 MDD: {_fmt_pct(final_payload.get('mdd_full'))} (CAGR은 이 wave의 목표가 아니라 참고 수치일 뿐이다)")
    is_fitness = final_payload.get("is_fitness")
    if is_fitness is not None:
        lines.append(f"- 참고, 최종 후보 IS 단기창 적합도: {_fmt(is_fitness.get('fitness'))} (상위25%평균 {_fmt_pct(is_fitness.get('top_quantile_mean_return'))}, P(창<-20%) {_fmt_pct(is_fitness.get('p_window_ruin'))})")
    lines.append("")
    return lines


def _gates_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 게이트 (K1~K6)", ""]
    if final_payload is None:
        lines.append("- 게이트 미실행.")
        lines.append("")
        return lines
    gates = final_payload.get("gates", {})
    lines.append("| 게이트 | 상태 | 핵심 수치 |")
    lines.append("|---|---|---|")
    k1 = gates.get("k1", {})
    lines.append(f"| K1 (GA>랜덤, >=4/5시드) | {k1.get('status')} | {k1.get('n_wins')}/{k1.get('n_seeds')} |")
    k2 = gates.get("k2", {})
    lines.append(f"| K2 (최종개체 OOS단기창 > I5 OOS단기창) | {k2.get('status')} | {_fmt(k2.get('final_oos_fitness'))} vs {_fmt(k2.get('i5_oos_fitness'))} |")
    k3 = gates.get("k3", {})
    lines.append(f"| K3 (DSR, trials={gates23.CUMULATIVE_TRIALS:,}) | {k3.get('status')} | score={_fmt(k3.get('score'))}, p={_fmt(k3.get('probability'))} |")
    k4 = gates.get("k4", {})
    lines.append(
        f"| K4 (파산방어: MC P(<$50)<10%, 최대손실<=원금30%) | {k4.get('status')} | "
        f"파산확률={_fmt_pct(k4.get('mc', {}).get('ruin_probability_below_50'))}, "
        f"MC p05={_fmt_usd(k4.get('mc', {}).get('p05'))}, "
        f"최대손실={_fmt_pct(k4.get('max_loss_fraction_of_principal'))}(원금대비) |"
    )
    k5 = gates.get("k5", {})
    lines.append(
        f"| K5 (실행가능: 레그>=$5, gross<=1x, x3스트레스 부호유지) | {k5.get('status')} | "
        f"레그{_fmt_usd(k5.get('leg_usdt'))}, gross(구조){_fmt_usd(k5.get('gross_usdt_by_construction'))}/"
        f"실측최대{_fmt_usd(k5.get('realized_gross_max_usdt'))} (1x한도={_fmt_usd(ACTIVE_CAPITAL)}), "
        f"base상위25%={_fmt_pct(k5.get('base_top25_mean_return'))} -> stress상위25%={_fmt_pct(k5.get('stress_top25_mean_return'))} |"
    )
    k6 = gates.get("k6", {})
    lines.append(
        f"| K6 (사양-실행 정합성: paper 트래커 재현가능성) | {k6.get('status')} | "
        f"전략지원={k6.get('strategy_kind_supported')}, 유니버스폭 {k6.get('universe_breadth')}<=한도{k6.get('paper_universe_cap')}: {k6.get('universe_breadth_ok')}, "
        f"손절/익절 재현가능={k6.get('exit_mechanics_reproducible')} (동시보유수 max_concurrent={k6.get('max_concurrent')}는 게이트 대상 아님 -- paper의 기존 top_k 메커니즘이 이미 지원) |"
    )
    lines.append("")
    if k6.get("reasons"):
        lines.append("**K6 미달 사유 상세**:")
        for reason in k6.get("reasons", []):
            lines.append(f"- {reason}")
        lines.append("")
    overall = gates.get("overall")
    promoted = gates.get("promoted")
    reasons = gates.get("failure_reasons", [])
    lines.append(f"**종합: {overall} / 승격: {'YES' if promoted else 'NO'}**" + (f" (미달: {', '.join(reasons)})" if reasons else ""))
    lines.append("")
    return lines


def _dsr_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## DSR / 다중검정", ""]
    k3 = final_payload.get("gates", {}).get("k3", {}) if final_payload else {}
    lines.append(
        f"- K3 게이트 DSR (누적시행 {gates23.CUMULATIVE_TRIALS:,}회 = 사전 121 + wave21 GA 7,500 + "
        f"이번 wave 총평가 {gates23.THIS_WAVE_TOTAL_TRIALS:,}(GA {gates23.GA_TRIALS:,} + 랜덤 {gates23.RANDOM_TRIALS:,})): "
        f"{_fmt(k3.get('score'))} (p={_fmt(k3.get('probability'))}, observed_sharpe={_fmt(k3.get('observed_sharpe'))})"
    )
    lines.append(
        "- 이 DSR은 **최종 승격 후보 자신의 equity curve**(full_equity, IS+OOS 전체)로 계산했다 -- "
        "wave-21의 실수(게이트 탈락한 다른 개체의 수치를 보고)를 반복하지 않기 위해, K3 게이트 자체가 "
        "gates23.gate_k3_dsr(final.full_equity, ...) 한 곳에서만 계산되고 다른 개체로 대체될 코드 경로가 없다."
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
    genome = final_payload.get("final_genome", {})
    oos_fitness = final_payload.get("oos_fitness")
    if promoted:
        lines.append(
            f"**승격**: 최종 후보(시드 {final_payload.get('source_seed')} 중앙값 선택, 전략종류={genome.get('strategy_kind')})가 "
            f"K1~K6를 전부 통과했다. OOS 단기창 적합도 {_fmt(oos_fitness['fitness']) if oos_fitness else 'N/A'}. "
            "K6 통과 확인 후 paper 트래커에 정합성 검증과 함께 추가 검토 대상이다."
        )
    else:
        reasons = gates.get("failure_reasons", [])
        lines.append(f"**단기 수익최대화 목표로도 검증 통과 후보 없음**: K1~K6 중 다음 게이트가 미달했다: {', '.join(reasons) if reasons else '(내부 로직 확인 필요)'}.")
        lines.append("")
        if "K1" in reasons:
            lines.append("- K1 미달은 진화 메커니즘 자체가 무작위 탐색보다 나을 게 없었다는 뜻이다 -- \"GA 무의미\" 정직 보고.")
        if "K3" in reasons:
            k3 = gates.get("k3", {})
            lines.append(
                f"- K3(DSR) 미달(score={_fmt(k3.get('score'))})은 이 wave가 사전에 가장 경계한 시나리오다: "
                "\"단기 고수익은 우연으로 만들기 가장 쉬운 결과\"라는 SPEC.md의 우려가 누적 다중검정 보정 앞에서 "
                "그대로 확인됐다는 뜻이다."
            )
        if "K4" in reasons:
            lines.append("- K4(파산방어) 미달은 단기 수익 극대화를 추구한 대가로 파산 위험이 SPEC.md의 허용선(P(<$50)<10%, 최대손실<=원금30%)을 넘었다는 뜻이다.")
        if "K6" in reasons:
            k6 = gates.get("k6", {})
            lines.append(f"- K6(사양-실행 정합성) 미달은 최종 후보가 통계적으로 유효하더라도 **오늘의 research/paper/ 인프라가 그 사양을 있는 그대로 자동 운영할 수 없다**는 뜻이다: {'; '.join(k6.get('reasons', []))}")
        lines.append("")
        lines.append("이 결과는 실패가 아니라 유의미한 발견이다 -- I5(기존 승격 후보)는 그대로 유지한다.")
    lines.append("")
    return lines


def write_wave23_report(results_dir: Path, report_dir: Path, registry_path: Path) -> None:
    ga_payloads = [payload for seed in ga23.SEEDS if (payload := _load_optional(results_dir / f"ga_seed{seed}.json")) is not None]
    random_payloads = [payload for seed in random_search23.SEEDS if (payload := _load_optional(results_dir / f"random_seed{seed}.json")) is not None]
    final_payload = _load_optional(results_dir / "final_candidate.json")

    lines: list[str] = [
        *_header_section(),
        *_methodology_notes_section(),
        *_k1_section(ga_payloads, random_payloads, final_payload.get("gates") if final_payload else None),
        *_evolution_curve_section(ga_payloads),
        *_strategy_kind_distribution_section(ga_payloads, final_payload),
        *_final_candidate_section(final_payload),
        *_oos_section(final_payload),
        *_gates_section(final_payload),
        *_dsr_section(final_payload),
        *_verdict_section(final_payload),
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave23_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    promoted = bool(final_payload.get("gates", {}).get("promoted")) if final_payload else False
    overall = final_payload.get("gates", {}).get("overall", "PENDING") if final_payload else "PENDING"
    oos_fitness = final_payload.get("oos_fitness") if final_payload else None
    gates = final_payload.get("gates", {}) if final_payload else {}
    gate_cells = " ".join(f"K{n}:{'P' if gates.get(f'k{n}', {}).get('status') == 'PASS' else 'F'}" for n in range(1, 7)) if final_payload else "미실행"
    registry_lines = [
        "# Wave-23 registry",
        "",
        "| Candidate | Family | State | 최종후보 전략종류 | 최종후보 OOS 단기창 적합도 | I5 OOS 단기창 적합도 | 승격 | 게이트(K1-K6) |",
        "|---|---|---|---|---|---|---|---|",
        (
            f"| GA23_FINAL | wave23_ga_short | {'EVALUATED' if final_payload else 'PENDING'} | "
            f"{final_payload.get('final_genome', {}).get('strategy_kind') if final_payload else 'N/A'} | "
            f"{_fmt(oos_fitness['fitness']) if oos_fitness else 'N/A'} | "
            f"{_fmt(final_payload.get('i5_oos_fitness')) if final_payload else 'N/A'} | "
            f"{'YES' if promoted else 'NO'} | {gate_cells} |"
        ),
        "",
        f"참고: GA 시드 {len(ga_payloads)}/{len(ga23.SEEDS)}, 랜덤 대조군 시드 {len(random_payloads)}/{len(random_search23.SEEDS)}, 종합 {overall}.",
        "",
        f"**최종 판정**: {'승격 (K6 정합성 확인 후 paper 검토 대상)' if promoted else '단기 목표로도 통과 후보 없음 -- I5 유지 (report/wave23_report.md 참조)'}.",
        "",
    ]
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave23_report"]
