# Wave-24 markdown report + registry writer. Pure formatting/analysis over already-computed
# results/{gp_seed*,random_seed*,final_candidate,timing_estimate}.json (run_wave24.py's
# evolve/control/gates stages). SPEC.md's own instruction governs the headline framing here:
# "헤드라인은 항상 OOS 기준" -- this report NEVER leads with the final candidate's IS number.
#
# Two analyses live ONLY here, not in gates24.py, because they are DISCLOSURE requirements
# (SPEC.md 과최적화 방어 3/5), not promotion gates (L1-L7 has no "5-seed structural
# reproducibility" or "formula interpretability" gate):
#   - _seed_formula_comparison_section: compares all 5 GP seeds' own best formulas against each
#     other (terminal-kind Jaccard + economic-category grouping) -- task instruction: "5시드
#     수식 유사성: 시드마다 완전히 다른 수식이 나오면 그 자체가 노이즈 학습 증거이므로 반드시
#     비교·보고".
#   - _economic_interpretability_section: classifies the final candidate's terminals into
#     economic categories (funding/momentum/risk/liquidity/const) via a fixed, auditable rule
#     table and prints "해석 불가 -- 과최적화 의심" whenever the combination has no clean
#     carry-economics story -- task instruction: "경제적 해석 가능성 평가... 설명 불가면
#     '해석 불가 -- 과최적화 의심'으로 명시".

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave10_carry100.engine import ACTIVE_CAPITAL
from research.wave24_gp import gates24, gp, random_trees
from research.wave24_gp.tree import MAX_DEPTH, depth, from_dict, terminal_kinds_used, to_formula_string

GENERATION_SNAPSHOTS: Final[tuple[int, ...]] = (1, 5, 10, 15, 20, 25, 30)

# Read-only cross-reference for the verdict section's "탐색 방법론 소진" check (SPEC.md 판정:
# "GA(파라미터)·GP(규칙) 모두 실패면... 남은 길은 새 데이터뿐이다") -- derived from __file__, not
# from any results_dir a caller (e.g. a test) might pass in, since these sibling waves' own
# results live at their fixed real repo locations regardless of where THIS wave's outputs go.
_RESEARCH_DIR: Final = Path(__file__).resolve().parent.parent
_WAVE21_REGISTRY: Final = _RESEARCH_DIR / "wave21_ga" / "REGISTRY.md"
_WAVE23_REGISTRY: Final = _RESEARCH_DIR / "wave23_ga_short" / "REGISTRY.md"

# Economic categorization of terminal kinds (report-only judgment call, disclosed and fixed in
# code -- not tuned after seeing results). 'basis' (perp/spot spread) is grouped with 'funding':
# perpetual funding payments exist specifically to pull basis toward zero, so the two are the
# same underlying carry-economics quantity. 'const' is its own bucket -- a constant carries no
# market-economics meaning of its own and is excluded from every interpretability judgment below.
_TERMINAL_ECON_CATEGORY: Final[dict[str, str]] = {
    "funding_1d": "funding", "funding_7d": "funding", "funding_14d": "funding", "funding_30d": "funding",
    "basis": "funding",
    "price_ret_1d": "momentum", "price_ret_7d": "momentum", "price_ret_30d": "momentum",
    "realized_vol_20d": "risk", "atr_14": "risk",
    "quote_volume_30d": "liquidity",
    "const": "const",
}


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_pp(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:+.{digits}f}%p"


def _fmt(value: float | None, digits: int = 5) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 3, 0)] + "..."


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sibling_promoted(registry_path: Path) -> bool | None:
    """Read-only, best-effort peek at a sibling wave's own REGISTRY.md 승격 cell (rendered as
    literal `| YES |` / `| NO |` by every wave21_ga/wave23_ga_short-style reporting module in
    this repo). Returns None (== 'unknown', never guessed) if the file is missing or the
    expected cell text is not found, so a malformed/relocated sibling file can never silently
    manufacture a false "탐색 방법론 소진" claim."""
    try:
        text = registry_path.read_text(encoding="utf-8")
    except OSError:
        return None
    has_yes, has_no = "| YES |" in text, "| NO |" in text
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


# ---------------------------------------------------------------------------
# Economic interpretability (task requirement -- see module docstring).
# ---------------------------------------------------------------------------


def _econ_categories(node: Any) -> frozenset[str]:
    return frozenset(_TERMINAL_ECON_CATEGORY.get(kind, "unknown") for kind in terminal_kinds_used(node))


def _interpretability_verdict(node: Any) -> tuple[str, frozenset[str], frozenset[str]]:
    """Fixed, auditable rule table (disclosed in the report itself, see
    _economic_interpretability_section) -- not a free-text judgment. `market_categories` is the
    econ-category set EXCLUDING 'const'. Returns (verdict_text, terminal_kinds, market_categories)."""
    kinds = terminal_kinds_used(node)
    market_categories = _econ_categories(node) - {"const"}
    if not market_categories:
        verdict = "해석 불가 -- 과최적화 의심 (상수항뿐, 시장 정보를 전혀 사용하지 않는 신호)"
    elif market_categories <= {"funding"}:
        verdict = "해석 가능 -- 순수 캐리/펀딩 신호 (funding/basis가 클수록 진입 -- 캐리 논리와 직접 일치)"
    elif market_categories <= {"funding", "risk"}:
        verdict = "해석 가능 -- 리스크조정 캐리 신호 (펀딩을 변동성/ATR로 정규화 -- 표준적인 리스크조정 캐리 직관과 일치)"
    elif market_categories <= {"funding", "liquidity"}:
        verdict = "해석 가능 -- 유동성가중 캐리 신호 (펀딩을 거래량으로 걸러내거나 가중 -- 유동성 높은 캐리를 선호하는 것은 합리적)"
    elif market_categories <= {"funding", "risk", "liquidity"}:
        verdict = "부분적으로 해석 가능 -- 캐리가 주도하지만 리스크·유동성이 동시에 얽혀 단일 서사로 검증하기는 약함"
    elif "funding" not in market_categories:
        verdict = "해석 불가 -- 과최적화 의심 (펀딩/베이시스 성분이 전혀 없다 -- 델타중립 캐리 진입 신호로서의 경제적 근거 부재)"
    elif "momentum" in market_categories:
        verdict = "해석 불가 -- 과최적화 의심 (가격모멘텀 성분이 캐리 신호에 섞여 있다 -- 방향성 신호가 진입을 결정하는 구조는 SPEC.md가 배제한 방향베팅과 경제적으로 유사한 우려를 남긴다)"
    else:
        verdict = "해석 불가 -- 과최적화 의심 (터미널 조합이 표준적 캐리 직관과 대응되지 않는다)"
    return verdict, kinds, market_categories


# ---------------------------------------------------------------------------
# Sections.
# ---------------------------------------------------------------------------


def _header_section() -> list[str]:
    return [
        "# Wave-24 리포트 -- 유전 프로그래밍: 규칙(신호 수식) 자체를 진화 (사전등록 research/wave24_gp/SPEC.md)",
        "",
        "**wave-21(GA, 파라미터)·wave-23(GA, 단기목표)이 5개 전략 계열 안에서만 탐색했다면, 이 wave는 "
        "탐색 공간 자체를 신호 수식 트리로 바꿨다** -- 내가 상상하지 못한 조합을 기계가 만들도록. 포지션 "
        "구조(델타중립 캐리, spot롱/perp숏, 1페어, 레그 50%, 유니버스 200종)는 L4/I5 기준값 그대로 "
        "고정했고, GP가 진화시키는 것은 '언제 진입할지 판단하는 신호 수식' 하나뿐이다(방향 베팅이 아니다). "
        "**헤드라인은 항상 최종 후보의 OOS(2025-10~) CAGR**이며, IS 적합도는 진화 과정의 참고 수치로만 "
        "다룬다.",
        "",
        "## 과최적화 방어 5중 장치 (구현 요약, SPEC.md 그대로)",
        "",
        "1. **OOS 봉인**: 진화/랜덤탐색 루프는 `fitness24.evaluate_tree()/evaluate_tree_cached()`만 "
        "호출한다 -- 이 함수들은 `mode` 인자 자체를 받지 않아 OOS를 요청할 방법이 코드상 존재하지 않는다"
        "(구조적 봉쇄). `run_backtest(mode='IS')`는 매 호출마다 OOS_SPLIT 이전으로 먼저 잘라낸 뒤 "
        "계산하고, 결과 equity 인덱스가 OOS_SPLIT을 넘으면 `OOSLeakageError`를 던진다. `oos_slice()`가 "
        "OOS 구간을 읽을 수 있는 유일한 함수이고, `final_evaluation()`이 이 패키지에서 OOS를 여는 유일한 "
        "호출자다(gates 스테이지, wave당 정확히 1회).",
        "2. **랜덤 트리 대조군**: GP와 동일 예산(200x30x5=30,000평가)의 무작위 트리를 동일 분포"
        "(ramped half-and-half)에서 뽑아 병행 평가한다(L1).",
        "3. **5시드 재현성**: 5개 GP 시드의 자체 최고 개체 중 **중앙값**(최댓값이 아님)을 최종 후보로 "
        "선택하고, 5개 시드의 수식을 서로 비교해 구조적으로 유사한지 아래 별도 섹션에서 직접 검증한다 "
        "(수식이 매번 완전히 다르면 그 자체가 노이즈 학습 증거).",
        "4. **DSR을 최종 후보 자신의 equity curve로 계산**(L3) -- wave-21의 실수(게이트 탈락한 다른 "
        "개체의 수치를 보고)를 반복하지 않도록, 최종 후보가 아닌 다른 트리의 곡선이 대신 쓰일 코드 경로가 "
        "없다.",
        "5. **수식 해석 가능성 검사**: 최종 수식이 경제적으로 설명 가능한지 아래 별도 섹션에서 판정하고, "
        "설명 불가능한 조합이면 '해석 불가 -- 과최적화 의심'으로 그대로 기록한다(통계적 게이트 통과 "
        "여부와 무관하게 항상 병기).",
        "",
        "## L7 (신규 게이트, 이 wave만의 것)",
        "",
        f"수식 단순성: 노드수 <= {gates24.L7_MAX_NODE_COUNT} ∧ 사용 터미널 종류 <= {gates24.L7_MAX_TERMINAL_KINDS}종. "
        "GP는 GA(고정된 유전자 개수)보다 표현력이 커서 과최적화 위험도 그만큼 크므로, 트리 복잡도 자체를 "
        "엄격히 제한하는 게이트를 SPEC.md가 이 wave에 신규로 등록했다.",
        "",
    ]


def _methodology_notes_section(
    timing_payload: dict[str, Any] | None, gp_payloads: list[dict[str, Any]], random_payloads: list[dict[str, Any]]
) -> list[str]:
    lines = ["## 방법론 노트 (필독)", ""]
    lines.append(
        "1. **적합도 != CAGR**: 진화 루프가 최적화하는 적합도는 `median(4폴드 IS CAGR) - std(4폴드) - "
        "5*max(0, MDD-15%) - 0.02*노드수`(SPEC.md 공식)이지 CAGR 자체가 아니다 -- 'L1'과 '진화 곡선' "
        "표의 수치는 전부 이 적합도이며 CAGR로 착각하면 안 된다. 헤드라인(OOS CAGR)은 별도 섹션에서만 "
        "다룬다."
    )
    lines.append(
        "2. **엔진 재사용**: 일별 경제 공식(갭 PnL/회전비용/일중 PnL)은 "
        "`research.wave21_ga.fitness._compound_factor`와 라인 단위로 동일하게 이 패키지 안에서 "
        "재구현했다(교차 import 대신 -- wave23_ga_short의 engine23 선례와 동일한 이유: 이 wave가 "
        "자기 완결적으로 감사 가능하도록). top-k 선택(`_select_top_k`)도 마찬가지다."
    )
    lines.append(
        "3. **최종 후보 선정**: 5개 GP 시드의 자체 최고 개체 중 **중앙값**을 최종 후보로 선택한다"
        "(`run_wave24.select_final_candidate`, wave21_ga/wave23_ga_short와 동일 규칙 -- 홀수 시드수라 "
        "중앙값이 항상 실제 트리 하나와 정확히 일치한다)."
    )
    lines.append(
        f"4. **누적 다중검정**: L3 게이트 자체가 누적시행({gates24.CUMULATIVE_TRIALS:,}회 = 사전 "
        f"{gates24.PRIOR_CUMULATIVE_TRIALS_BEFORE_WAVE24:,}(wave1-23 누적) + 이번 wave "
        f"{gates24.THIS_WAVE_TOTAL_TRIALS:,}(GP {gates24.GP_TRIALS:,} + 랜덤 {gates24.RANDOM_TRIALS:,}))으로 "
        "보정된 DSR을 쓴다 -- wave23_ga_short의 K3 선례를 그대로 계승해, 게이트 자체를 가장 보수적인 "
        "숫자로 고정했다."
    )
    est_minutes = timing_payload.get("estimated_total_minutes") if timing_payload else None
    actual_seconds = sum(float(p.get("wall_seconds", 0.0)) for p in gp_payloads) + sum(float(p.get("wall_seconds", 0.0)) for p in random_payloads)
    actual_minutes = actual_seconds / 60.0 if (gp_payloads or random_payloads) else None
    timing_bits = []
    if est_minutes is not None:
        timing_bits.append(f"사전 추정 상한 {est_minutes:.1f}분(1세대 실측 기반, 캐시 적중 무시)")
    if actual_minutes is not None:
        timing_bits.append(f"실제 소요 {actual_minutes:.1f}분(완료된 시드 기준)")
    if timing_bits:
        lines.append("5. **실행시간**: " + ", ".join(timing_bits) + ".")
    lines.append("")
    return lines


def _l1_section(gp_payloads: list[dict[str, Any]], random_payloads: list[dict[str, Any]], gates_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## L1 -- GP vs 랜덤 트리 대조군 (시드별)", ""]
    lines.append("| 시드쌍 | GP 시드 | GP 최고적합도 | 랜덤 시드 | 랜덤 최고적합도 | GP 승리 |")
    lines.append("|---|---|---|---|---|---|")
    for index, (gp_payload, random_payload) in enumerate(zip(gp_payloads, random_payloads)):
        gp_best = gp_payload.get("best_fitness")
        random_best = random_payload.get("best_fitness")
        win = "YES" if (gp_best is not None and random_best is not None and gp_best > random_best) else "NO"
        lines.append(f"| {index + 1} | {gp_payload.get('seed')} | {_fmt(gp_best)} | {random_payload.get('seed')} | {_fmt(random_best)} | {win} |")
    lines.append("")
    if gates_payload is not None:
        l1 = gates_payload.get("l1", {})
        lines.append(f"**L1 판정: {l1.get('status', 'N/A')}** ({l1.get('n_wins', 'N/A')}/{l1.get('n_seeds', 'N/A')}회 GP 승리, 기준 {l1.get('threshold', 'N/A')}회 이상)")
    lines.append("")
    return lines


def _evolution_curve_section(gp_payloads: list[dict[str, Any]]) -> list[str]:
    lines = ["## 진화 곡선 (세대별 최고/평균 적합도, 스냅샷)", ""]
    for payload in gp_payloads:
        seed = payload.get("seed")
        history = payload.get("history", [])
        by_generation = {int(row["generation"]) + 1: row for row in history}  # stored 0-indexed; report as 1-indexed
        lines.append(f"### 시드 {seed}")
        lines.append("")
        lines.append("| 세대 | 최고 적합도 | 평균 적합도 | 최저 적합도 | 평균 노드수 | 최고개체 노드수 |")
        lines.append("|---|---|---|---|---|---|")
        for generation in GENERATION_SNAPSHOTS:
            row = by_generation.get(generation)
            if row is None:
                continue
            best_nodes = row.get("best_tree", {}).get("node_count")
            lines.append(
                f"| {generation} | {_fmt(row.get('best_fitness'))} | {_fmt(row.get('mean_fitness'))} | "
                f"{_fmt(row.get('worst_fitness'))} | {_fmt(row.get('mean_node_count'), 1)} | {best_nodes} |"
            )
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
    lines = ["## 최종 후보 수식 (5개 GP 시드 중 IS 적합도 중앙값 시드 선택)", ""]
    if final_payload is None:
        lines.append("- final_candidate.json 없음 (gates 스테이지 미실행).")
        lines.append("")
        return lines
    tree_payload = final_payload.get("final_tree", {})
    node = from_dict(tree_payload["tree"]) if "tree" in tree_payload else None
    formula = tree_payload.get("formula") or (to_formula_string(node) if node is not None else "N/A")
    lines.append(f"- 선정 출처: 시드 {final_payload.get('source_seed')} (그 시드의 IS 적합도 = {_fmt(final_payload.get('source_is_fitness'))})")
    lines.append("")
    lines.append(f"- **최종 수식**: `{formula}`")
    if node is not None:
        kinds = terminal_kinds_used(node)
        lines.append(f"- 노드수: {tree_payload.get('node_count')} (L7 한도 {gates24.L7_MAX_NODE_COUNT}) / 깊이: {depth(node)} (구조 한도 {MAX_DEPTH})")
        lines.append(f"- 사용 터미널 종류({len(kinds)}종, L7 한도 {gates24.L7_MAX_TERMINAL_KINDS}): {', '.join(sorted(kinds)) if kinds else '(none)'}")
    lines.append("")
    lines.append("### 5개 GP 시드의 자체 최고 적합도 (중앙값 선택 근거)")
    lines.append("")
    lines.append("| 시드 | IS 최고적합도 |")
    lines.append("|---|---|")
    for seed_str, fitness_value in final_payload.get("gp_best_by_seed", {}).items():
        marker = " <- 선택" if str(seed_str) == str(final_payload.get("source_seed")) else ""
        lines.append(f"| {seed_str} | {_fmt(fitness_value)}{marker} |")
    lines.append("")
    return lines


def _seed_formula_comparison_section(gp_payloads: list[dict[str, Any]]) -> list[str]:
    lines = ['## 5시드 수식 비교 (과최적화 방어 3 -- SPEC.md: "수식이 매번 완전히 다르면 노이즈 학습")', ""]
    if not gp_payloads:
        lines.append("- GP 시드 결과 없음 (evolve 스테이지 미실행).")
        lines.append("")
        return lines
    lines.append("| 시드 | IS 최고적합도 | 노드수 | 깊이 | 사용 터미널 종류 | 경제 카테고리 | 수식 |")
    lines.append("|---|---|---|---|---|---|---|")
    kind_sets: list[frozenset[str]] = []
    category_sets: list[frozenset[str]] = []
    for payload in gp_payloads:
        tree_payload = payload.get("best_tree", {})
        node = from_dict(tree_payload["tree"])
        kinds = terminal_kinds_used(node)
        categories = _econ_categories(node) - {"const"}
        kind_sets.append(kinds)
        category_sets.append(categories)
        formula = tree_payload.get("formula") or to_formula_string(node)
        lines.append(
            f"| {payload.get('seed')} | {_fmt(payload.get('best_fitness'))} | {tree_payload.get('node_count')} | "
            f"{depth(node)} | {', '.join(sorted(kinds)) if kinds else '(none)'} | "
            f"{', '.join(sorted(categories)) if categories else '(none)'} | `{formula}` |"
        )
    lines.append("")

    n_seeds = len(gp_payloads)
    if n_seeds >= 2:
        pair_jaccard = []
        for set_a, set_b in combinations(kind_sets, 2):
            union = set_a | set_b
            pair_jaccard.append(len(set_a & set_b) / len(union) if union else 1.0)
        mean_jaccard = sum(pair_jaccard) / len(pair_jaccard)

        groups: dict[frozenset[str], list[int]] = {}
        for index, categories in enumerate(category_sets):
            groups.setdefault(categories, []).append(index)
        largest_categories, largest_indices = max(groups.items(), key=lambda item: len(item[1]))
        threshold = 4  # SPEC.md "4/5 이상에서 유사 구조가 나와야 인정"
        reproducible = len(largest_indices) >= threshold

        lines.append(f"- **터미널 종류(kind) 집합 자카드 유사도**: 평균 {mean_jaccard:.3f} ({len(pair_jaccard)}개 시드쌍, {n_seeds}시드)")
        seed_list = ", ".join(str(gp_payloads[i].get("seed")) for i in largest_indices)
        category_label = ", ".join(sorted(largest_categories)) if largest_categories else "(none, 상수뿐)"
        lines.append(f"- **경제 카테고리 집합 기준 최대 동일 그룹**: {len(largest_indices)}/{n_seeds}시드가 동일 카테고리 {{{category_label}}} 사용 (시드: {seed_list})")
        verdict = "구조 재현성 있음 (SPEC.md 기준 4/5 이상 충족)" if reproducible else "시드마다 상이 -- 노이즈 학습 가능성 (SPEC.md 기준 4/5 미달)"
        lines.append(f"- **판정: {verdict}**")
    else:
        lines.append("- 시드가 2개 미만이라 유사도 계산 불가.")
    lines.append("")
    return lines


def _economic_interpretability_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 경제적 해석 가능성 평가 (최종 후보)", ""]
    if final_payload is None:
        lines.append("- final_candidate.json 없음 (gates 스테이지 미실행).")
        lines.append("")
        return lines
    tree_payload = final_payload.get("final_tree", {})
    node = from_dict(tree_payload["tree"]) if "tree" in tree_payload else None
    if node is None:
        lines.append("- final_tree 없음.")
        lines.append("")
        return lines
    verdict, kinds, market_categories = _interpretability_verdict(node)
    formula = tree_payload.get("formula") or to_formula_string(node)
    lines.append(f"- **수식**: `{formula}`")
    lines.append(f"- **사용 터미널**: {', '.join(sorted(kinds)) if kinds else '(none)'}")
    lines.append(f"- **경제 카테고리** (const 제외): {', '.join(sorted(market_categories)) if market_categories else '(none)'}")
    lines.append(f"- **판정: {verdict}**")
    lines.append("")
    lines.append(
        "(판정 규칙, 이 리포트 코드에 고정: funding/basis만 쓰면 순수 캐리, +risk(vol/ATR)는 "
        "리스크조정 캐리, +liquidity(거래량)는 유동성가중 캐리 -- 전부 표준적 캐리 직관과 합치한다. "
        "funding 성분이 아예 없거나 momentum(가격모멘텀)이 섞이면 델타중립 캐리 신호로서의 경제적 서사가 "
        "약해지므로 '해석 불가 -- 과최적화 의심'으로 자동 표시한다. 이 규칙은 결과를 보기 전에 코드로 "
        "고정한 것이며, 특정 결과를 통과시키기 위해 사후에 조정하지 않는다.)"
    )
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
    lines.append(
        f"- 참고, 최종 후보 전기간 CAGR: {_fmt_pct(final_payload.get('full_period_cagr'))} / "
        f"IS CAGR: {_fmt_pct(final_payload.get('is_cagr'))} / 전기간 MDD: {_fmt_pct(final_payload.get('mdd_full'))}"
    )
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
        "저펀딩 레짐이라, 승격된 기존 후보(I5)조차 IS 대비 OOS가 크게 낮다. 최종 후보의 raw 격차만 단독으로 "
        "보면 레짐효과와 과최적화효과를 구분할 수 없다 -- 아래는 I5 대비 상대 격차다."
    )
    relative_gap_pp = (gap_pp - i5_gap_pp) if (gap_pp is not None and i5_gap_pp is not None) else None
    lines.append(f"- **I5 대비 상대 격차**: {_fmt_pp(relative_gap_pp)} (양수면 I5보다 레짐효과를 넘어서는 추가 과최적화 신호)")
    if relative_gap_pp is not None and relative_gap_pp > 10.0:
        lines.append(
            "- 상대 격차가 10%p를 넘는다 -- 레짐효과로 설명되지 않는 추가적인 IS 특화 신호가 있다는 뜻이다. "
            "다만 아래 L2가 PASS라면 이 초과분이 OOS 성과 자체를 I5보다 나쁘게 만들 정도는 아니라는 뜻이므로, "
            "'과최적화로 인한 OOS 붕괴'와는 다른, 더 약한 형태의 신호임을 함께 밝힌다."
        )
    else:
        lines.append("- 상대 격차가 10%p 이내다 -- I5 자체의 레짐효과를 넘어서는 뚜렷한 과최적화 신호는 약하다.")
    lines.append("")
    return lines


def _gates_section(final_payload: dict[str, Any] | None) -> list[str]:
    lines = ["## 게이트 (L1~L7)", ""]
    if final_payload is None:
        lines.append("- 게이트 미실행.")
        lines.append("")
        return lines
    gates = final_payload.get("gates", {})
    lines.append("| 게이트 | 상태 | 핵심 수치 |")
    lines.append("|---|---|---|")
    l1 = gates.get("l1", {})
    lines.append(f"| L1 (GP>랜덤, >=4/5시드) | {l1.get('status')} | {l1.get('n_wins')}/{l1.get('n_seeds')} |")
    l2 = gates.get("l2", {})
    lines.append(f"| L2 (최종개체 OOS CAGR > I5 OOS CAGR) | {l2.get('status')} | {_fmt_pct(l2.get('final_oos_cagr'))} vs {_fmt_pct(l2.get('i5_oos_cagr'))} ({_fmt_pp(l2.get('gap_pp'))}) |")
    l3 = gates.get("l3", {})
    lines.append(f"| L3 (DSR, 자기 곡선, trials={gates24.CUMULATIVE_TRIALS:,}) | {l3.get('status')} | score={_fmt(l3.get('score'))}, p={_fmt(l3.get('probability'))} |")
    l4 = gates.get("l4", {})
    mc = l4.get("mc") or {}
    lines.append(
        f"| L4 (MC p05>{_fmt_usd(gates24.L4_P05_FLOOR_USDT)} ∧ 파산확률<{_fmt_pct(gates24.L4_RUIN_PROBABILITY_MAX)} ∧ "
        f"블록MDD p95<={_fmt_pct(gates24.L4_BLOCK_MDD_P95_MAX)}) | {l4.get('status')} | "
        f"MC p05={_fmt_usd(mc.get('p05'))}, 파산확률={_fmt_pct(mc.get('ruin_probability'))}, 블록MDD p95={_fmt_pct(l4.get('block_mdd_p95'))} |"
    )
    l5 = gates.get("l5", {})
    lines.append(
        f"| L5 (실행가능: 레그>=$5, gross<=1x, x3스트레스 부호유지) | {l5.get('status')} | "
        f"레그{_fmt_usd(l5.get('leg_usdt_nominal'))}/총{_fmt_usd(l5.get('gross_usdt_nominal'))} (1x한도={_fmt_usd(ACTIVE_CAPITAL)}), "
        f"스트레스 {_fmt_usd(l5.get('stress_start_usdt'))}->{_fmt_usd(l5.get('stress_end_usdt'))} |"
    )
    l6 = gates.get("l6", {})
    lines.append(
        f"| L6 (paper 재현가능성) | {l6.get('status')} | "
        f"터미널 데이터매핑={l6.get('data_ok')}, 유니버스폭 {l6.get('universe_breadth')}<=한도{l6.get('paper_carry_universe_cap')}: {l6.get('universe_breadth_ok')} |"
    )
    l7 = gates.get("l7", {})
    lines.append(
        f"| L7 (수식단순성: 노드<={gates24.L7_MAX_NODE_COUNT} ∧ 터미널종류<={gates24.L7_MAX_TERMINAL_KINDS}) | {l7.get('status')} | "
        f"노드수={l7.get('node_count')}, 터미널종류={l7.get('n_terminal_kinds')}, 깊이={l7.get('depth')} |"
    )
    lines.append("")
    if l6.get("reasons"):
        lines.append("**L6 미달 사유 상세**:")
        for reason in l6.get("reasons", []):
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
    if final_payload is None:
        lines.append("- final_candidate.json 없음.")
        lines.append("")
        return lines
    l3 = final_payload.get("gates", {}).get("l3", {})
    reference = final_payload.get("dsr_reference_cumulative")
    lines.append(
        f"- **L3 게이트 DSR** (누적시행 {gates24.CUMULATIVE_TRIALS:,}회 = 사전 "
        f"{gates24.PRIOR_CUMULATIVE_TRIALS_BEFORE_WAVE24:,}(wave1-23 누적) + 이번 wave "
        f"{gates24.THIS_WAVE_TOTAL_TRIALS:,}(GP {gates24.GP_TRIALS:,} + 랜덤 {gates24.RANDOM_TRIALS:,})): "
        f"{_fmt(l3.get('score'))} (p={_fmt(l3.get('probability'))}, observed_sharpe={_fmt(l3.get('observed_sharpe'))})"
    )
    lines.append(
        "- 이 DSR은 **최종 승격 후보 자신의 equity curve**(full_equity, IS+OOS 전체)로 계산했다 -- "
        "`gates24.gate_l3_dsr(final.full_equity)` 한 곳에서만 계산되고 다른 개체로 대체될 코드 경로가 없다 "
        "(wave-21의 실수 -- 게이트 탈락한 다른 개체의 수치를 보고 -- 재발 방지)."
    )
    if reference is not None:
        score_a, score_b = reference.get("score"), l3.get("score")
        consistent = score_a is not None and score_b is not None and abs(score_a - score_b) < 1e-9
        lines.append(
            f"- 대조 계산(dsr_reference_cumulative, 동일 equity/trials로 별도 산출): {_fmt(score_a)} "
            f"({'게이트 값과 일치 -- 내부 정합성 확인됨' if consistent else '게이트 값과 불일치 -- 확인 필요'})."
        )
    lines.append(
        "- wave24는 wave23_ga_short의 K3 선례를 그대로 계승해 **게이트 자체가 이미 누적(가장 보수적인) "
        "trials를 쓴다** -- wave21_ga처럼 '이번 wave 전용 DSR'과 '누적 disclosure용 DSR'을 별도로 "
        "분리하지 않는다."
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
    tree_payload = final_payload.get("final_tree", {})
    formula = tree_payload.get("formula", "N/A")
    node = from_dict(tree_payload["tree"]) if "tree" in tree_payload else None
    interp_verdict = _interpretability_verdict(node)[0] if node is not None else "N/A"
    oos_final = final_payload.get("oos_cagr_regime_anchored")
    oos_i5 = final_payload.get("i5_reference", {}).get("oos_cagr")

    if promoted:
        lines.append(
            f"**승격**: 최종 후보(시드 {final_payload.get('source_seed')} 중앙값 선택) `{formula}`가 "
            f"L1~L7을 전부 통과했다. OOS 연환산 {_fmt_pct(oos_final)} (I5 {_fmt_pct(oos_i5)} 대비 개선). "
            f"경제적 해석: {interp_verdict}."
        )
        if interp_verdict.startswith("해석 불가"):
            lines.append(
                "- **경고**: 게이트는 전부 통과했지만 수식 자체는 경제적으로 해석하기 어렵다는 판정이다 -- "
                "통계적 게이트 통과가 경제적 타당성을 보증하지 않는다는 점을 그대로 기록한다. paper 전진검증 "
                "착수 전 재검토를 권장한다."
            )
        lines.append(
            "GP 산출물은 GA보다도 표현력이 커서 과최적화 위험이 상시 존재하므로, 이 결과를 카드에 반영하더라도 "
            "paper 전진검증(L6 확인 후)을 필수 선행 조건으로 부여한다."
        )
    else:
        reasons = gates.get("failure_reasons", [])
        lines.append(f"**미승격**: L1~L7 중 다음 게이트가 미달했다: {', '.join(reasons) if reasons else '(내부 로직 확인 필요)'}.")
        lines.append("")
        if "L1" in reasons:
            lines.append('- L1 미달은 진화 메커니즘 자체가 무작위 트리 생성보다 나을 게 없었다는 뜻이다 -- "GP 무의미" 정직 보고.')
        if "L2" in reasons or "L3" in reasons:
            lines.append(
                "- L2/L3 미달은 이 wave가 사전에 가장 가능성 높다고 지목한 시나리오와 일치한다: **IS에서 화려한 "
                "수식을 찾았지만 OOS·DSR에서 무너지는 과최적화**. SPEC.md 자체 문구대로, 이는 실패가 아니라 "
                "**표현력을 트리 구조로 키워도 이 데이터엔 추가 엣지가 없다는 탐색 종료 근거**로 기록한다."
            )
        if "L4" in reasons:
            lines.append("- L4(파산방어) 미달은 몬테카를로/블록셔플 관점에서 원금 보존 실패 위험이 SPEC.md 허용선을 넘었다는 뜻이다.")
        if "L7" in reasons:
            l7 = gates.get("l7", {})
            lines.append(
                f"- L7(수식 단순성) 미달(노드수={l7.get('node_count')}, 터미널종류={l7.get('n_terminal_kinds')})은 "
                "진화가 SPEC.md 자체가 정한 복잡도 예산을 넘어서야만 IS 적합도를 개선할 수 있었다는 뜻이며, "
                "그 자체로 과최적화 신호다."
            )
        lines.append(f"- 경제적 해석: {interp_verdict}.")
        lines.append("")

        ga_promoted = _sibling_promoted(_WAVE21_REGISTRY)
        ga_short_promoted = _sibling_promoted(_WAVE23_REGISTRY)
        if ga_promoted is False and ga_short_promoted is False:
            lines.append(
                "**탐색 방법론 소진 선언** (SPEC.md \"판정\" 사전등록 기준): wave21_ga(파라미터 GA)· "
                "wave23_ga_short(단기목표 GA)·wave24_gp(규칙 자체를 진화시킨 GP) 세 차례 모두 I5를 넘지 "
                "못했다. 탐색 알고리즘을 파라미터 튜닝(GA)에서 규칙 자체의 진화(GP)로 바꿔도 결과가 같다는 "
                "것은, 이 데이터에서 짜낼 수 있는 구조적 엣지가 이미 소진되었다는 강한 증거다. 남은 길은 이 "
                "캐시 안에서 더 정교한 탐색 알고리즘을 시도하는 것이 아니라 **새 데이터(전진수집)**뿐이다."
            )
        else:
            lines.append("이 결과는 실패가 아니라 유의미한 발견이다 -- I5(기존 승격 후보)는 그대로 유지한다.")
    lines.append("")
    return lines


def write_wave24_report(results_dir: Path, report_dir: Path, registry_path: Path, i5_results_path: Path) -> None:
    gp_payloads = [payload for seed in gp.SEEDS if (payload := _load_optional(results_dir / f"gp_seed{seed}.json")) is not None]
    random_payloads = [payload for seed in random_trees.SEEDS if (payload := _load_optional(results_dir / f"random_seed{seed}.json")) is not None]
    final_payload = _load_optional(results_dir / "final_candidate.json")
    timing_payload = _load_optional(results_dir / "timing_estimate.json")
    _ = i5_results_path  # already folded into final_candidate.json's own i5_reference/L2 payload by the gates stage; kept as a parameter for symmetry with reporting21.write_wave21_report / reporting18.write_wave18_report's own signature

    lines: list[str] = [
        *_header_section(),
        *_methodology_notes_section(timing_payload, gp_payloads, random_payloads),
        *_l1_section(gp_payloads, random_payloads, final_payload.get("gates") if final_payload else None),
        *_evolution_curve_section(gp_payloads),
        *_final_candidate_section(final_payload),
        *_seed_formula_comparison_section(gp_payloads),
        *_economic_interpretability_section(final_payload),
        *_oos_section(final_payload),
        *_overfitting_gap_section(final_payload),
        *_gates_section(final_payload),
        *_dsr_section(final_payload),
        *_verdict_section(final_payload),
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave24_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    promoted = bool(final_payload.get("gates", {}).get("promoted")) if final_payload else False
    overall = final_payload.get("gates", {}).get("overall", "PENDING") if final_payload else "PENDING"
    oos_final = final_payload.get("oos_cagr_regime_anchored") if final_payload else None
    gates = final_payload.get("gates", {}) if final_payload else {}
    gate_cells = " ".join(f"L{n}:{'P' if gates.get(f'l{n}', {}).get('status') == 'PASS' else 'F'}" for n in range(1, 8)) if final_payload else "미실행"
    formula = final_payload.get("final_tree", {}).get("formula", "N/A") if final_payload else "N/A"
    node_count_value = final_payload.get("final_tree", {}).get("node_count", "N/A") if final_payload else "N/A"
    registry_lines = [
        "# Wave-24 registry",
        "",
        "| Candidate | Family | State | 최종후보 수식 | 노드수 | 최종후보 OOS CAGR | I5 OOS CAGR | 승격 | 게이트(L1-L7) |",
        "|---|---|---|---|---|---|---|---|---|",
        (
            f"| GP_FINAL | wave24_gp | {'EVALUATED' if final_payload else 'PENDING'} | `{_truncate(formula, 80)}` | {node_count_value} | "
            f"{_fmt_pct(oos_final)} | "
            f"{_fmt_pct(final_payload.get('i5_reference', {}).get('oos_cagr')) if final_payload else 'N/A'} | "
            f"{'YES' if promoted else 'NO'} | {gate_cells} |"
        ),
        "",
        f"참고: GP 시드 {len(gp_payloads)}/{len(gp.SEEDS)}, 랜덤 트리 대조군 시드 {len(random_payloads)}/{len(random_trees.SEEDS)}, 종합 {overall}.",
        "",
        f"**최종 판정**: {'승격 (L6 정합성 확인 후 paper 검토 대상)' if promoted else '미승격 -- I5 유지 (report/wave24_report.md 참조)'}.",
        "",
    ]
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave24_report"]
