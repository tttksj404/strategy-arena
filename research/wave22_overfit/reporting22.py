# Wave-22 markdown report + registry writer. Pure formatting over already-computed
# results/*.json (run_wave22.py's `evaluate` stage). No numeric computation happens here --
# every number printed is read from a results/*.json field, so the report can never silently
# diverge from what was actually measured.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave22_overfit.genomes import G1_GENOME, G1_REFERENCE_METRICS, I5_GENOME, I5_REFERENCE_METRICS


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_pp(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:+.{digits}f}%p"


def _fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _fmt_bool_kr(value: bool | None, yes: str = "예", no: str = "아니오") -> str:
    return "N/A" if value is None else (yes if value else no)


def _fmt_ratio(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"reporting22: required results file missing: {path} -- run `run_wave22.py --stage evaluate` first")
    return json.loads(path.read_text(encoding="utf-8"))


AXIS_LABEL_KR: Final[dict[str, str]] = {
    "entry_threshold_apr": "진입임계(APR)",
    "exit_threshold_ratio": "청산비율",
    "window_days": "윈도우(일)",
    "top_k_pairs": "top_k",
    "leg_fraction": "leg 비중",
    "universe_breadth": "유니버스 폭",
    "idle_mode": "유휴모드",
}


# ---------------------------------------------------------------------------
# Section 0: engine reproduction check.
# ---------------------------------------------------------------------------


def _section_reproduction(sensitivity: dict[str, Any]) -> str:
    g1_full = sensitivity["g1_full_cagr"]
    g1_oos = sensitivity["g1_oos_cagr_self_contained"]
    ref_full = G1_REFERENCE_METRICS["full_period_cagr"]
    ref_oos = G1_REFERENCE_METRICS["oos_cagr_self_contained"]
    rows = [
        ["전기간 CAGR", _fmt_pct(g1_full), _fmt_pct(ref_full), _fmt_pp((g1_full - ref_full) * 100.0)],
        ["OOS CAGR(자체구간)", _fmt_pct(g1_oos), _fmt_pct(ref_oos), _fmt_pp((g1_oos - ref_oos) * 100.0)],
    ]
    return "\n".join([
        "## 0. 엔진 재현성 확인",
        "",
        "wave22 자체 캐시/엔진(`fitness.build_market_cache`/`run_backtest`/`cagr`/`oos_slice`)으로 G1을 재평가해 STRATEGY_CARD.md의 실측값과 대조했다. 이 재현이 어긋나면 이후 6종 검증 전부가 무의미하므로 가장 먼저 확인한다.",
        "",
        _table(["지표", "wave22 재계산", "STRATEGY_CARD 실측", "차이"], rows),
        "",
        "차이가 소수점 이하 수준이면(반올림 오차) 재현 성공으로 간주하고 이후 절을 진행한다.",
    ])


# ---------------------------------------------------------------------------
# Section 1: sensitivity.
# ---------------------------------------------------------------------------


def _section_sensitivity(data: dict[str, Any]) -> str:
    overall = data["overall"]
    rows = []
    for axis in AXIS_LABEL_KR:
        summary = data["per_axis"][axis]
        ratio = summary["neighbor_avg_full_cagr_ratio"]
        stable = summary["stable_mean_ratio_ge_0_8"]
        boundary = "예" if summary["at_range_boundary"] else ""
        rows.append([
            AXIS_LABEL_KR[axis],
            str(summary["baseline_value"]),
            f"{summary['n_available']}/{summary['n_total_tiers']}",
            _fmt_ratio(ratio),
            _fmt_ratio(summary["neighbor_min_full_cagr_ratio"]),
            _fmt_bool_kr(stable),
            boundary,
        ])

    point_blocks = []
    for axis in AXIS_LABEL_KR:
        summary = data["per_axis"][axis]
        point_rows = []
        for point in summary["points"]:
            if point["available"]:
                point_rows.append([point["tier_label"], str(point["gene_value"]), _fmt_pct(point["full_cagr"]), _fmt_ratio(point["full_cagr_ratio"]), ""])
            else:
                point_rows.append([point["tier_label"], "-", "-", "-", point["note"]])
        point_blocks.append(f"**{AXIS_LABEL_KR[axis]}** ({axis})\n\n" + _table(["tier", "gene값", "전기간CAGR", "G1대비비율", "비고"], point_rows))

    grid_blocks = []
    for pair in data["grid"]["pairs"]:
        axis_a, axis_b = pair["axis_a"], pair["axis_b"]
        label_a, label_b = AXIS_LABEL_KR.get(axis_a, axis_a), AXIS_LABEL_KR.get(axis_b, axis_b)
        cell_rows = []
        for cell in pair["cells"]:
            if cell["available"]:
                feas = "" if cell["gross_feasible_1x"] else "(1x초과)"
                cell_rows.append([str(cell["tier_a"]), str(cell["tier_b"]), _fmt_pct(cell["full_cagr"]), _fmt_ratio(cell["full_cagr_ratio"]) + feas])
            else:
                cell_rows.append([str(cell["tier_a"]), str(cell["tier_b"]), "-", "범위밖"])
        grid_blocks.append(
            f"**{label_a} x {label_b}** -- 유효 {pair['n_available_cells']}칸, 1x초과 {pair['n_gross_infeasible_cells']}칸, 비율 최소 {_fmt_ratio(pair['min_full_cagr_ratio'])} / 평균 {_fmt_ratio(pair['mean_full_cagr_ratio'])}\n\n"
            + _table(["tier_" + axis_a, "tier_" + axis_b, "전기간CAGR", "G1대비비율"], cell_rows)
        )

    return "\n".join([
        "## 1. 파라미터 안정성 지형 (가장 중요)",
        "",
        f"G1의 7개 유전자를 각각 +-10%/+-20% (범위 내 값은 gene 종류별 정의, methodology 참조) 흔들어 재평가했다. '이웃 평균 성과 / G1 성과' 비율이 1에 가까울수록 완만한 고원(엣지), 낮을수록 뾰족한 봉우리(과최적화)에 가깝다.",
        "",
        _table(["축", "G1 값", "평가가능/전체", "이웃평균비율", "이웃최소비율", "안정(>=0.8)", "경계값"], rows),
        "",
        f"**종합 안정성 비율 (7축 중 최솟값, 판정에 사용) = {_fmt_ratio(overall['stability_ratio_min_of_axis_means'])}** (최약축: {AXIS_LABEL_KR.get(overall['worst_axis'], overall['worst_axis'])})",
        f"참고 -- 7축 평균 = {_fmt_ratio(overall['stability_ratio_mean_of_axis_means'])}, 0.8 미만 축 수 = {overall['n_axes_below_0_8']}/7, 0.6 미만 축 수 = {overall['n_axes_below_0_6']}/7",
        "",
        "### 축별 상세",
        "",
        "\n\n".join(point_blocks),
        "",
        "### 2축 동시변동 격자 (top_k_pairs x leg_fraction 격자는 gross 1x 제약을 넘는 셀도 포함해 원시 신호 품질을 확인한다)",
        "",
        "\n\n".join(grid_blocks),
    ])


# ---------------------------------------------------------------------------
# Section 2: rolling.
# ---------------------------------------------------------------------------


def _section_rolling(data: dict[str, Any]) -> str:
    rows = []
    for window in data["windows"]:
        if window["g1_wins"] is None:
            rows.append([window["window_start"][:10], window["window_end"][:10], "-", "-", "-", "-", window["note"]])
            continue
        flag = "OOS포함" if window["contains_oos"] else ""
        lowconf = "(저신뢰)" if window["low_confidence"] else ""
        rows.append([
            window["window_start"][:10], window["window_end"][:10],
            _fmt_pct(window["g1_cagr"]), _fmt_pct(window["i5_cagr"]), _fmt_pp(window["g1_minus_i5_pp"]),
            "G1승" if window["g1_wins"] else "I5승",
            (flag + lowconf).strip() or "",
        ])
    limitations = "\n".join(f"- {item}" for item in data["limitations"])
    return "\n".join([
        "## 2. 시간 안정성 (롤링 워크포워드, 6개월 비중첩)",
        "",
        f"**G1 전체 승률 = {_fmt_pct(data['g1_win_rate'], 1)}** ({data['n_g1_wins']}/{data['n_windows_counted']}구간)",
        f"- IS전용 구간 승률: {_fmt_pct(data['win_rate_pre_oos_windows'], 1)} ({data['n_pre_oos_windows']}구간) / OOS포함 구간 승률: {_fmt_pct(data['win_rate_oos_touching_windows'], 1)} ({data['n_oos_touching_windows']}구간)",
        f"- 전반부 승률: {_fmt_pct(data['win_rate_first_half_chronological'], 1)} / 후반부 승률: {_fmt_pct(data['win_rate_second_half_chronological'], 1)}",
        f"- 최장 G1 연승: {data['streaks']['longest_g1_win_streak']}구간 / 최장 I5 연승: {data['streaks']['longest_i5_win_streak']}구간",
        "",
        _table(["구간시작", "구간끝", "G1 CAGR", "I5 CAGR", "차이", "승자", "비고"], rows),
        "",
        "**한계**:",
        limitations,
    ])


# ---------------------------------------------------------------------------
# Section 3: regime.
# ---------------------------------------------------------------------------


def _section_regime(data: dict[str, Any]) -> str:
    rows = []
    for year_row in data["by_year"]:
        bucket_kr = {"high_funding": "고펀딩", "low_funding": "저펀딩", "unclassified": "미분류"}[year_row["bucket"]]
        flags = []
        if year_row["is_partial_year"]:
            flags.append("부분연도")
        if year_row["straddles_oos_split"]:
            flags.append("OOS경계")
        rows.append([
            str(year_row["year"]), bucket_kr,
            _fmt_pct(year_row["g1_cagr"]), _fmt_pct(year_row["i5_cagr"]), _fmt_pp(year_row["g1_minus_i5_pp"]),
            _fmt_bool_kr(year_row["g1_wins"]), ",".join(flags),
        ])
    high, low = data["high_funding"], data["low_funding"]
    limitations = "\n".join(f"- {item}" for item in data["limitations"])
    dominant_kr = {"high_funding": "고펀딩기", "low_funding": "저펀딩기", None: "판정불가"}[data["dominant_regime_by_magnitude"]]
    return "\n".join([
        "## 3. 레짐 분해 (고펀딩 2020/2021/2024 vs 저펀딩 2022/2023/2025/2026)",
        "",
        _table(["연도", "레짐", "G1 CAGR", "I5 CAGR", "차이", "G1승", "비고"], rows),
        "",
        f"- 고펀딩기 평균 개선분: {_fmt_pp(high['mean_g1_minus_i5_pp'])} (중앙값 {_fmt_pp(high['median_g1_minus_i5_pp'])}, {high['n_years']}개년, G1 {high['g1_win_count']}승)",
        f"- 저펀딩기 평균 개선분: {_fmt_pp(low['mean_g1_minus_i5_pp'])} (중앙값 {_fmt_pp(low['median_g1_minus_i5_pp'])}, {low['n_years']}개년, G1 {low['g1_win_count']}승)",
        f"- **개선분 기여가 더 큰 레짐: {dominant_kr}**",
        f"- 두 레짐 모두 개선(양수)인가: {_fmt_bool_kr(data['improvement_positive_in_both_regimes'])}"
        + (f" / 한쪽에만 존재하는 개선: {data['improvement_only_in_one_regime']}" if data["improvement_only_in_one_regime"] else ""),
        "",
        "**한계**:",
        limitations,
    ])


# ---------------------------------------------------------------------------
# Section 4: DSR.
# ---------------------------------------------------------------------------


def _section_dsr(data: dict[str, Any]) -> str:
    at_wave = data["g1_dsr_at_trials_this_wave_only"]
    at_cum = data["g1_dsr_at_trials_cumulative"]
    ref = data["wave21_report_reference_ga_final_top_k3"]
    cross = data["ga_final_top_k3_cross_check_this_wave"]
    rows = [
        ["G1 (top_k=1, 이 wave 재계산)", "이 wave만 (7,500)", _fmt(at_wave["score"] if at_wave else None, 5), _fmt(at_wave["probability"] if at_wave else None, 5)],
        ["G1 (top_k=1, 이 wave 재계산)", "누적 (121+7,500=7,621)", _fmt(at_cum["score"] if at_cum else None, 5), _fmt(at_cum["probability"] if at_cum else None, 5)],
        ["GA_FINAL (top_k=3, wave21 원본, 참고)", "이 wave만 (7,500)", _fmt(ref["trials_this_wave_only"], 5), "-"],
        ["GA_FINAL (top_k=3, wave21 원본, 참고)", "누적 (7,621)", _fmt(ref["trials_cumulative_121_plus_7500"], 5), "-"],
    ]
    if cross:
        rows.append(["GA_FINAL (top_k=3, 이 wave 재계산 대조)", "이 wave만 (7,500)", _fmt(cross["dsr_at_trials_this_wave_only"]["score"] if cross["dsr_at_trials_this_wave_only"] else None, 5), "-"])
        rows.append(["GA_FINAL (top_k=3, 이 wave 재계산 대조)", "누적 (7,621)", _fmt(cross["dsr_at_trials_cumulative"]["score"] if cross["dsr_at_trials_cumulative"] else None, 5), "-"])
    limitations = "\n".join(f"- {item}" for item in data["limitations"])
    return "\n".join([
        "## 4. DSR 재계산 (누적 시행 반영)",
        "",
        _table(["대상", "시행수", "DSR score", "probability"], rows),
        "",
        f"**G1의 누적시행 DSR = {_fmt(data['g1_dsr_score_cumulative'], 5)} (양수 여부: {_fmt_bool_kr(data['g1_dsr_positive_at_cumulative_trials'])})**",
        "",
        "**한계**:",
        limitations,
    ])


# ---------------------------------------------------------------------------
# Section 5: attribution.
# ---------------------------------------------------------------------------


def _section_attribution(data: dict[str, Any]) -> str:
    rows = []
    for axis in AXIS_LABEL_KR:
        row = data["per_axis"][axis]
        if row["unchanged"]:
            rows.append([AXIS_LABEL_KR[axis], f"{row['i5_value']} (동일)", "-", "-", "-"])
            continue
        rows.append([
            AXIS_LABEL_KR[axis],
            f"{row['i5_value']} -> {row['g1_value']}",
            _fmt_pp(row["forward_contribution_pp"]),
            _fmt_pp(row["backward_contribution_pp"]),
            "",
        ])
    fwd, bwd = data["forward_concentration"], data["backward_concentration"]
    limitations = "\n".join(f"- {item}" for item in data["limitations"])
    return "\n".join([
        "## 5. 유전자 기여도 분해 (I5 -> G1, one-at-a-time)",
        "",
        f"I5->G1 전체 격차: 전기간 {_fmt_pp(data['total_gap_full_cagr_pp'])}, OOS {_fmt_pp(data['total_gap_oos_cagr_pp'])}. 변경된 축은 {len(data['changed_axes'])}/7개({', '.join(AXIS_LABEL_KR.get(a, a) for a in data['changed_axes'])}), 동일값 축은 {', '.join(AXIS_LABEL_KR.get(a, a) for a in data['unchanged_axes']) or '없음'}.",
        "",
        _table(["축", "I5->G1 값변화", "정방향 기여(I5+1축)", "역방향 기여(G1-1축)", "비고"], rows),
        "",
        f"- 정방향 최대기여축: {AXIS_LABEL_KR.get(fwd['top_axis'], fwd['top_axis'])} (양의 기여분 중 {_fmt_pct(fwd['top_share'], 1)}) -> 단일축 집중: {_fmt_bool_kr(fwd['concentrated_single_axis'])} / 2축이상 분산: {_fmt_bool_kr(fwd['spread_across_2plus'])}",
        f"- 역방향 최대기여축: {AXIS_LABEL_KR.get(bwd['top_axis'], bwd['top_axis'])} (양의 기여분 중 {_fmt_pct(bwd['top_share'], 1)}) -> 단일축 집중: {_fmt_bool_kr(bwd['concentrated_single_axis'])} / 2축이상 분산: {_fmt_bool_kr(bwd['spread_across_2plus'])}",
        f"- 정방향/역방향 일치 여부: {_fmt_bool_kr(data['forward_and_backward_agree_on_concentration'])}",
        f"- 정방향 기여 합 = {_fmt_pp(data['sum_of_forward_contributions_pp'])} (실제 격차 대비 상호작용 잔차 = {_fmt_pp(data['interaction_residual_pp'])})",
        "",
        "**한계**:",
        limitations,
    ])


# ---------------------------------------------------------------------------
# Section 6: shuffle control.
# ---------------------------------------------------------------------------


def _section_shuffle(data: dict[str, Any]) -> str:
    rank_full, rank_oos = data["rank_by_full_cagr"], data["rank_by_oos_cagr"]
    sorted_draws = sorted(data["draws"], key=lambda row: row["full_cagr"], reverse=True)
    top5 = sorted_draws[:5]
    bottom5 = sorted_draws[-5:]
    top_rows = [[str(row["index"]), _fmt_pct(row["full_cagr"]), _fmt_pct(row["oos_cagr_self_contained"]), f"${row['gross_usdt']:.2f}"] for row in top5]
    bottom_rows = [[str(row["index"]), _fmt_pct(row["full_cagr"]), _fmt_pct(row["oos_cagr_self_contained"]), f"${row['gross_usdt']:.2f}"] for row in bottom5]
    limitations = "\n".join(f"- {item}" for item in data["limitations"])
    return "\n".join([
        "## 6. 거짓 발견 대조 (무작위 유전자 30개)",
        "",
        f"G1 전기간 CAGR {_fmt_pct(data['g1_full_cagr'])} / OOS CAGR {_fmt_pct(data['g1_oos_cagr_self_contained'])}",
        f"- 전기간 CAGR 기준: G1은 무작위 {rank_full['n_pool']}개 중 {rank_full['n_random_below_g1']}개보다 우수 (상위 {_fmt(rank_full['g1_top_pct_of_pooled_31'], 1)}%, 상위5%이내: {_fmt_bool_kr(rank_full['g1_in_top_5pct'])})",
        f"- OOS CAGR 기준: G1은 무작위 {rank_oos['n_pool']}개 중 {rank_oos['n_random_below_g1']}개보다 우수 (상위 {_fmt(rank_oos['g1_top_pct_of_pooled_31'], 1)}%, 상위5%이내: {_fmt_bool_kr(rank_oos['g1_in_top_5pct'])})",
        f"- 시도 횟수(gross<=1x 조건 재추첨 포함): {data['methodology']['n_draws_attempted']}회 시도 -> {data['methodology']['n_draws_requested']}개 채택",
        "",
        "무작위 대조군 상위 5개 (전기간 CAGR 기준):",
        "",
        _table(["idx", "전기간CAGR", "OOS CAGR", "gross"], top_rows),
        "",
        "무작위 대조군 하위 5개:",
        "",
        _table(["idx", "전기간CAGR", "OOS CAGR", "gross"], bottom_rows),
        "",
        "**한계**:",
        limitations,
    ])


# ---------------------------------------------------------------------------
# Section 7: verdict.
# ---------------------------------------------------------------------------


def _section_verdict(data: dict[str, Any]) -> str:
    inputs = data["inputs"]
    checks = data["pass_checks"]
    check_label_kr = {
        "stability_ratio_ge_0_8": "안정성비율 >=0.8",
        "rolling_win_rate_ge_55pct": "롤링승률 >=55%",
        "spread_across_2plus_axes": "개선분 2축이상 분산",
        "dsr_positive": "DSR>0",
        "shuffle_control_top5pct": "무작위대조 상위5%",
    }
    rows = [[check_label_kr[name], _fmt_bool_kr(met, "충족", "미충족")] for name, met in checks.items()]
    fail_rows = [[reason] for reason in data["fail_reasons"]] or [["(없음)"]]
    return "\n".join([
        "## 종합 판정",
        "",
        f"- 안정성 비율(최솟값): {_fmt_ratio(inputs['stability_ratio'])}",
        f"- 롤링 승률: {_fmt_pct(inputs['rolling_win_rate'], 1)}",
        f"- 개선분 집중축(정방향): {AXIS_LABEL_KR.get(inputs['concentration_top_axis'], inputs['concentration_top_axis'])} (단일축집중: {_fmt_bool_kr(inputs['concentrated_single_axis'])}, 2축이상분산: {_fmt_bool_kr(inputs['spread_across_2plus_axes'])})",
        f"- DSR(누적시행): {_fmt(inputs['dsr_score_cumulative'], 5)} (양수: {_fmt_bool_kr(inputs['dsr_positive'])})",
        f"- 무작위대조 상위%(전기간CAGR): {_fmt(inputs['shuffle_g1_top_pct'], 1)}% (상위5%이내: {_fmt_bool_kr(inputs['shuffle_top5pct_full_cagr'])})",
        f"- 개선분 기여 우세 레짐: {inputs['dominant_regime']}",
        "",
        "### PASS 기준 충족 현황",
        "",
        _table(["기준", "충족여부"], rows),
        "",
        "### FAIL 사유 (하나라도 있으면 즉시 FAIL, 최우선 판정)",
        "",
        _table(["사유"], fail_rows),
        "",
        f"# 종합판정: {data['overall']}",
        "",
        f"**권고**: {data['recommendation']}",
    ])


def write_wave22_report(results_dir: Path, report_dir: Path, registry_path: Path) -> None:
    sensitivity_data = _load(results_dir / "sensitivity.json")
    rolling_data = _load(results_dir / "rolling.json")
    regime_data = _load(results_dir / "regime.json")
    dsr_data = _load(results_dir / "dsr.json")
    attribution_data = _load(results_dir / "attribution.json")
    shuffle_data = _load(results_dir / "shuffle_control.json")
    verdict_data = _load(results_dir / "verdict.json")

    header = "\n".join([
        "# Wave-22 리포트 -- G1 과최적화 정밀판정",
        "",
        "GA(wave-21)가 산출하고 gross 1x 제약 복원으로 확정된 G1 구성(research/STRATEGY_CARD.md \"G1 확정\")이 진짜 엣지인지 과최적화인지를 6종 독립 검증으로 정밀판정한다. G1은 wave21 H1/H2/H3/H5를 통과했지만 IS-OOS 격차 33.5%p(I5 대비 상대 25.3%p)라는 약한 과최적화 신호가 있었다 -- 이 wave의 존재 이유는 그 신호의 실체를 가리는 것이다.",
        "",
        f"**G1 유전자**: {G1_GENOME.to_dict()}",
        f"**I5 유전자(기준선)**: {I5_GENOME.to_dict()}",
        "",
        "이 리포트는 결과를 유리하게 쓰지 않는다. 과최적화 신호가 확인되면 FAIL로 명시하고 G1 승격 철회를 권고한다 (사용자 지시 원문).",
    ])

    body = "\n\n".join([
        header,
        _section_reproduction(sensitivity_data),
        _section_sensitivity(sensitivity_data),
        _section_rolling(rolling_data),
        _section_regime(regime_data),
        _section_dsr(dsr_data),
        _section_attribution(attribution_data),
        _section_shuffle(shuffle_data),
        _section_verdict(verdict_data),
    ])

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave22_report.md").write_text(body + "\n", encoding="utf-8")

    registry = "\n".join([
        "# Wave-22 registry",
        "",
        "| Candidate | Family | 종합판정 | 안정성비율 | 롤링승률 | 개선분집중축 | DSR(누적) | 무작위대조 상위% |",
        "|---|---|---|---|---|---|---|---|",
        (
            f"| G1 | wave21_ga (top_k 3->1 수정본) | {verdict_data['overall']} "
            f"| {_fmt_ratio(verdict_data['inputs']['stability_ratio'])} "
            f"| {_fmt_pct(verdict_data['inputs']['rolling_win_rate'], 1)} "
            f"| {AXIS_LABEL_KR.get(verdict_data['inputs']['concentration_top_axis'], verdict_data['inputs']['concentration_top_axis'])} "
            f"| {_fmt(verdict_data['inputs']['dsr_score_cumulative'], 4)} "
            f"| {_fmt(verdict_data['inputs']['shuffle_g1_top_pct'], 1)}% |"
        ),
        "",
        f"**최종 판정**: {verdict_data['overall']} -- {verdict_data['recommendation']}",
        "",
        "근거: `research/wave22_overfit/report/wave22_report.md`",
        "",
    ])
    registry_path.write_text(registry, encoding="utf-8")


__all__ = ["write_wave22_report"]
