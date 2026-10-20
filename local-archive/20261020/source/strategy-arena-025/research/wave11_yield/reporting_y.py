# Wave-11 markdown report + registry writer. Pure formatting over already-computed
# results/Y{1..6}.json payloads written by run_wave11.py --stage run/gates.
# Mirrors research/wave10_carry100/reporting.py's structure, extended with the
# axis-breakdown / C1-comparison / cost-vs-edge classification SPEC.md requires.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, TOTAL_CAPITAL
from research.wave10_carry100.regime import HIGH_FUNDING_YEARS
from research.wave11_yield.configs import CONFIG_IDS, get_config
from research.wave11_yield.gates_y import (
    DSR_CUMULATIVE_TRIALS,
    PROMOTION_BLOCK_MDD_MAX,
    PROMOTION_HIGH_FUNDING_ANNUALIZED_MIN,
    gross_usdt,
    leg_usdt,
)

# C1 reference figures. Cited from research/wave10_carry100/results/C1.json (read once,
# frozen as literal constants here rather than re-read on every report build, so a future
# unrelated change to wave10's own results cannot silently move wave11's comparison
# baseline underneath it) -- exact values, not the SPEC.md-rounded "16.84%"/"2.29%" text.
#
# 2026-07-22 data-integrity phase 2 update: research/wave1/fetch_binance.py's spot-klines
# pagination bug (limit=1500 used against spot's real 1000-row cap, see
# research/validation/DATA_FIX_REPORT.md) truncated most of the universe's spot legs years
# before OOS even starts. Phase 1 refetched research/wave1/cache and recomputed C1 on the
# corrected cache; these constants are refreshed to that corrected C1.json (previously
# cited values, computed on the still-truncated cache, kept below for traceability):
#   C1_HIGH_FUNDING_ANNUALIZED          0.16835051871984544  -> 0.15943511612278694
#   C1_CURRENT_LOW_FUNDING_ANNUALIZED   0.0027133647851274034 -> 0.0027133647851269593 (unchanged to 1e-14)
#   C1_BLOCK_MDD_P95                    0.022891183254975085 -> 0.0268839144059019
#   C1_MC_P05_USDT                      130.3110544444444    -> 127.95065509576276
#   C1_MC_RUIN_PROBABILITY              0.0                   -> 0.0 (unchanged)
#   C1_FULL_PERIOD_TOTAL_RETURN         0.5937760912889094   -> 0.5618683390214398
# This closes the "C1을 교정 데이터로 재계산하는 것은 이 wave의 등록된 범위 밖" gap the
# data-quality section below used to flag (see _data_quality_section): Y1-Y6's own cache
# was already self-corrected (see fetch_y11.py docstring), so with this update both sides
# of the C1 comparison are now the same data generation. gates_y.py's PROMOTION_* bar is
# intentionally NOT updated here -- that is SPEC.md's pre-registered literal pass/fail
# threshold, frozen on purpose so a downstream data fix cannot retroactively move it; only
# these descriptive/informational comparison figures are refreshed.
C1_HIGH_FUNDING_ANNUALIZED: float = 0.15943511612278694
C1_CURRENT_LOW_FUNDING_ANNUALIZED: float = 0.0027133647851269593
C1_BLOCK_MDD_P95: float = 0.0268839144059019
C1_MC_P05_USDT: float = 127.95065509576276
C1_MC_RUIN_PROBABILITY: float = 0.0
C1_FULL_PERIOD_TOTAL_RETURN: float = 0.5618683390214398

AXIS_LABELS: dict[str, str] = {
    "threshold_down": "임계↓ (가동률 시험)",
    "width_up": "폭↑ (동시기회 시험)",
    "threshold_down_width_up": "임계↓+폭↑ (결합)",
    "universe_up": "유니버스↑ (종목풀 시험)",
    "speed_up": "속도↑ (리밸런스 주기 시험)",
    "spike": "스파이크 (단발 이벤트 포착)",
}


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_usd(value: float | None) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def _config_table() -> list[str]:
    lines = [
        "| Candidate | 축 | 유니버스 | 바 | 쌍 수 | 레그당 비중 | 레그 $ | gross $ | gross배수 | 진입임계 | 정의 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate_id in CONFIG_IDS:
        config = get_config(candidate_id)
        leg = leg_usdt(config)
        gross = gross_usdt(config)
        lines.append(
            f"| {candidate_id} | {AXIS_LABELS.get(config.axis, config.axis)} | {config.universe} | {config.bar} | "
            f"{config.candidate.top_k} | {config.leg_fraction:.3%} | {_fmt_usd(leg)} | {_fmt_usd(gross)} | "
            f"{gross / ACTIVE_CAPITAL:.2f}x | {config.candidate.threshold_apr:.0%} | {config.note} |"
        )
    return lines


def _gate_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | S1 구조 | S2 MC(p05/ruin) | S3 블록MDD p95 | S4 실행가능성 | Overall | 승격여부 | 실패/미승격 사유 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        if not gates:
            lines.append(f"| {candidate_id} | - | - | - | - | NOT_EVALUATED | - | run --stage gates 필요 |")
            continue
        s1, s2, s3, s4 = gates["gate_s1"], gates["gate_s2"], gates["gate_s3"], gates["gate_s4"]
        promo = gates["promotion"]
        s1_cell = f"{s1['status']} ({s1['leverage_multiplier_of_active_capital']:.2f}x)"
        s2_cell = f"{s2['status']} (p05={_fmt_usd(s2['p05'])}, ruin={_fmt_pct(s2['ruin_probability'])})"
        s3_cell = f"{s3['status']} ({_fmt_pct(s3['mdd_p95'])})"
        s4_cell = f"{s4['status']} (레그{_fmt_usd(s4['leg_usdt_nominal'])})"
        promoted = "YES" if promo["promoted"] else "no"
        reasons = ", ".join(gates.get("failure_reasons", [])) or "-"
        if not promo["promoted"] and gates["overall"] == "PASS":
            missing = []
            if not promo["high_funding_ok"]:
                missing.append(f"고펀딩기 {_fmt_pct(promo['high_funding_mean_annualized_return'])} <= 기준 {_fmt_pct(promo['high_funding_bar'])}")
            if not promo["block_mdd_ok"]:
                missing.append(f"블록MDD {_fmt_pct(promo['block_mdd_p95'])} > 기준 {_fmt_pct(promo['block_mdd_bar'])}")
            reasons = "; ".join(missing)
        lines.append(f"| {candidate_id} | {s1_cell} | {s2_cell} | {s3_cell} | {s4_cell} | **{gates['overall']}** | {promoted} | {reasons} |")
    return lines


def _yield_table(payloads: dict[str, dict]) -> list[str]:
    year_headers = " | ".join(f"{year}" for year in HIGH_FUNDING_YEARS)
    lines = [
        f"| Candidate | {year_headers} | 고펀딩기 평균 | 저펀딩기(OOS) | 가동률 | 연간회전(왕복) | 총비용$ | 순수익$(전기간) |",
        "|---|" + "---:|" * (len(HIGH_FUNDING_YEARS) + 6),
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        regime = payload.get("regime_breakdown")
        metadata = payload.get("metadata", {})
        equity = payload.get("equity") or []
        net_profit = None
        if equity:
            start_v = float(equity[0]["value"])
            end_v = float(equity[-1]["value"])
            net_profit = end_v - start_v
        if not regime:
            lines.append(f"| {candidate_id} | " + " | ".join(["-"] * (len(HIGH_FUNDING_YEARS) + 6)) + " |")
            continue
        year_cells = []
        for year in HIGH_FUNDING_YEARS:
            entry = regime["high_funding_years"].get(str(year))
            year_cells.append(_fmt_pct(entry["annualized_return"]) if entry else "no data")
        mean_ann = regime.get("high_funding_mean_annualized_return")
        current = regime.get("current_low_funding")
        current_ann = current["annualized_return"] if current else None
        util = metadata.get("utilization")
        round_trips = metadata.get("annualized_round_trips")
        cost = metadata.get("total_cost_usdt")
        lines.append(
            f"| {candidate_id} | " + " | ".join(year_cells) + f" | {_fmt_pct(mean_ann)} | {_fmt_pct(current_ann)} | "
            f"{_fmt_pct(util, 1)} | {round_trips:.1f}회/년 | {_fmt_usd(cost)} | {_fmt_usd(net_profit)} |"
        )
    return lines


def _c1_comparison_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | 고펀딩기 연환산 (C1=15.94%) | Δ vs C1 | 블록MDD p95 (C1=2.69%) | Δ vs C1 | MC p05 (C1=$127.95) | Δ vs C1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **C1(기준선)** | {_fmt_pct(C1_HIGH_FUNDING_ANNUALIZED)} | - | {_fmt_pct(C1_BLOCK_MDD_P95)} | - | {_fmt_usd(C1_MC_P05_USDT)} | - |",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        regime = payload.get("regime_breakdown")
        gates = payload.get("gates")
        if not regime or not gates:
            lines.append(f"| {candidate_id} | - | - | - | - | - | - |")
            continue
        high_funding = regime.get("high_funding_mean_annualized_return")
        mdd = gates["gate_s3"]["mdd_p95"]
        mc_p05 = gates["gate_s2"]["p05"]
        delta_hf = None if high_funding is None else high_funding - C1_HIGH_FUNDING_ANNUALIZED
        delta_mdd = mdd - C1_BLOCK_MDD_P95
        delta_p05 = mc_p05 - C1_MC_P05_USDT
        hf_arrow = "-" if delta_hf is None else ("개선" if delta_hf > 0 else "악화")
        mdd_arrow = "개선" if delta_mdd < 0 else "악화"
        p05_arrow = "개선" if delta_p05 > 0 else "악화"
        lines.append(
            f"| {candidate_id} | {_fmt_pct(high_funding)} | {_fmt_pct(delta_hf)} ({hf_arrow}) | {_fmt_pct(mdd)} | "
            f"{_fmt_pct(delta_mdd)} ({mdd_arrow}) | {_fmt_usd(mc_p05)} | {_fmt_usd(delta_p05)} ({p05_arrow}) |"
        )
    return lines


def _dsr_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        f"| Candidate | 바 | DSR score | DSR P(진짜 엣지) | 관측 샤프 | 누적시행수 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        ref = payload.get("reference_metrics") or {}
        dsr = ref.get("dsr")
        bar = payload.get("bar", "1D")
        flag = " *" if bar != "1D" else ""
        if not dsr:
            lines.append(f"| {candidate_id} | {bar} | N/A | N/A | N/A | {DSR_CUMULATIVE_TRIALS} |")
            continue
        lines.append(f"| {candidate_id} | {bar} | {dsr['score']:.3f}{flag} | {_fmt_pct(dsr['probability'])}{flag} | {dsr['observed_sharpe']:.3f} | {dsr['trials']} |")
    return lines


def _dsr_caveat_lines(payloads: dict[str, dict]) -> list[str]:
    non_daily = [cid for cid in CONFIG_IDS if payloads[cid].get("bar", "1D") != "1D"]
    if not non_daily:
        return []
    return [
        "",
        f"\\* **{', '.join(non_daily)}는 8h바** -- `research.validation.deep_stats.deflated_sharpe`는 "
        "샤프 연환산(sqrt(365))과 표준오차의 관측치 수(n) 둘 다 '관측치=일봉'을 가정한다. 8h바를 그대로 "
        "넣으면 같은 기간에 관측치 수가 약 3배가 되어 표준오차 항이 기계적으로 작아지고 DSR score(z값)가 "
        "부풀려진다 -- 위 표의 해당 행은 다른 후보와 액면가로 비교하면 안 되는, 방법론상 과대추정된 "
        "참고값이다 (게이트/승격 판정에는 애초에 DSR을 쓰지 않으므로 판정 자체에는 영향 없음).",
    ]


def _classify_blockage(candidate_id: str, payload: dict) -> str:
    """Per-candidate honest classification of why a non-promoted candidate didn't clear
    the bar: 비용초과 (turnover/cost demonstrably ate the edge) / 기회부재 (utilization
    too low to matter) / 게이트위반 (a hard S1-S4 gate failed) / 고펀딩기미달
    (cleared all gates but the high-funding annualized return still didn't beat C1) /
    MDD초과 (cleared all gates but drawdown exceeded the promotion bar)."""
    gates = payload.get("gates")
    if not gates:
        return "미평가"
    if gates["overall"] == "FAIL":
        return f"게이트위반({', '.join(gates.get('failure_reasons', [])) or '미분류'})"
    promo = gates["promotion"]
    if promo["promoted"]:
        return "승격"
    metadata = payload.get("metadata", {})
    cost = float(metadata.get("total_cost_usdt", 0.0))
    equity = payload.get("equity") or []
    gross_profit_before_cost_proxy = None
    if equity:
        gross_profit_before_cost_proxy = (float(equity[-1]["value"]) - float(equity[0]["value"])) + cost
    utilization = float(metadata.get("utilization", 0.0))
    reasons = []
    if not promo["high_funding_ok"]:
        if gross_profit_before_cost_proxy is not None and cost > max(gross_profit_before_cost_proxy * 0.3, 0.0):
            reasons.append("비용초과(총비용이 비용전이익의 30%+ 잠식)")
        elif utilization < 0.05:
            reasons.append(f"기회부재(가동률 {utilization:.1%})")
        else:
            reasons.append("고펀딩기미달(게이트는 통과했으나 C1 상회 실패)")
    if not promo["block_mdd_ok"]:
        reasons.append("MDD초과")
    return ", ".join(reasons) if reasons else "미분류"


def _verdict_section(payloads: dict[str, dict]) -> list[str]:
    promoted = [cid for cid in CONFIG_IDS if payloads[cid].get("gates", {}).get("promotion", {}).get("promoted")]
    gate_pass = [cid for cid in CONFIG_IDS if payloads[cid].get("gates", {}).get("overall") == "PASS"]
    failed = [cid for cid in CONFIG_IDS if payloads[cid].get("gates", {}).get("overall") == "FAIL"]
    lines = ["## 판정", ""]
    lines.append(f"- S1~S4 게이트 전부 PASS: **{len(gate_pass)}/6** ({', '.join(gate_pass) if gate_pass else '없음'})")
    lines.append(f"- 승격 조건까지 만족 (고펀딩기 연환산 > {_fmt_pct(PROMOTION_HIGH_FUNDING_ANNUALIZED_MIN)} ∧ 블록MDD <= {_fmt_pct(PROMOTION_BLOCK_MDD_MAX)}): **{len(promoted)}/6** ({', '.join(promoted) if promoted else '없음'})")
    lines.append(f"- 게이트 FAIL: {', '.join(failed) if failed else '없음'}")
    lines.append("")
    lines.append("### 후보별 미승격/실패 사유 분류 (비용초과 / 기회부재 / 게이트위반 / 고펀딩기미달 / MDD초과)")
    lines.append("")
    lines.append("| Candidate | 분류 |")
    lines.append("|---|---|")
    for candidate_id in CONFIG_IDS:
        lines.append(f"| {candidate_id} | {_classify_blockage(candidate_id, payloads[candidate_id])} |")
    lines.append("")
    if not promoted:
        lines.append(
            "**전멸**: 포착률(activity) 축을 6개 방향으로 넓혀도 C1(고펀딩기 연 15.94%, 블록MDD 2.69%)을 "
            "동시에 넘어서는 후보가 없었다. 아래 축별 분해가 각 축이 구체적으로 어디서 막혔는지 보여준다 "
            "-- C1이 이번 시행 범위 내 상한이라는 것이 정직한 결론이다."
        )
    else:
        lines.append(f"**승격 후보 존재**: {', '.join(promoted)}가 C1을 고펀딩기 연환산·MDD 양쪽에서 동시에 상회했다.")
    lines.append("")
    return lines


def _axis_breakdown_section(payloads: dict[str, dict]) -> list[str]:
    lines = ["## 축별 분해", ""]
    groups: dict[str, list[str]] = {}
    for candidate_id in CONFIG_IDS:
        axis = get_config(candidate_id).axis
        groups.setdefault(axis, []).append(candidate_id)
    for axis, ids in groups.items():
        lines.append(f"### {AXIS_LABELS.get(axis, axis)} ({', '.join(ids)})")
        lines.append("")
        for candidate_id in ids:
            payload = payloads[candidate_id]
            metadata = payload.get("metadata", {})
            regime = payload.get("regime_breakdown") or {}
            gates = payload.get("gates") or {}
            util = metadata.get("utilization")
            cost = metadata.get("total_cost_usdt")
            high_funding = regime.get("high_funding_mean_annualized_return")
            mdd = gates.get("gate_s3", {}).get("mdd_p95")
            classification = _classify_blockage(candidate_id, payload)
            lines.append(
                f"- **{candidate_id}**: 가동률 {_fmt_pct(util, 1)}, 연간회전 "
                f"{metadata.get('annualized_round_trips', 0.0):.1f}회, 총비용 {_fmt_usd(cost)}, "
                f"고펀딩기 연환산 {_fmt_pct(high_funding)} (C1 {_fmt_pct(C1_HIGH_FUNDING_ANNUALIZED)}), "
                f"블록MDD {_fmt_pct(mdd)} (C1 {_fmt_pct(C1_BLOCK_MDD_P95)}) -> {classification}"
            )
        lines.append("")
    return lines


def _data_quality_section() -> list[str]:
    return [
        "## 데이터 품질 참고사항 (이번 wave에서 발견, 정직성 고지)",
        "",
        "`research/wave1/fetch_binance.py`의 `fetch_klines()`가 모든 요청에 `limit=1500`을 "
        "고정해 사용한다. fapi(퍼프) klines 엔드포인트는 실제로 1500을 허용하지만, spot "
        "klines 엔드포인트(`/api/v3/klines`)의 실제 한도는 1000이다. 그 결과 spot 응답이 "
        "1000행으로 조용히 잘리고, `len(page) < 1500` 조건이 '더 이상 데이터 없음'으로 "
        "잘못 해석되어 페이지네이션이 1페이지 만에 멈춘다. `research/wave1/cache/`를 "
        "실측한 결과 유니버스 40종목 중 32종목의 `binance_spot_{symbol}_1d.csv.gz`가 정확히 "
        "1000행에서 멈춰 있었다 (BTC/ETH/XRP/DOGE/ADA 등 주요 종목 포함, 종료일이 "
        "2022-05-27 근방). 반면 퍼프·펀딩 데이터는 전부 2026-07-14까지 정상 완주한다 "
        "(`integrity_report()`는 내부 정합성만 검사해 '너무 일찍 끝남'은 잡아내지 못한다). "
        "즉 wave1 이후 이 유니버스를 재사용한 모든 캐리 백테스트(wave1 F1, wave2 W2, "
        "wave10 C1~C4 포함)에서 이들 종목의 스팟 레그가 실제로는 2022년 중반경부터 "
        "NaN이 되어 조용히 유니버스에서 탈락했을 가능성이 있다.",
        "",
        "이 wave는 `research/wave1/fetch_binance.py`나 `research/wave1/cache/`를 직접 "
        "수정하지 않는다(범위 밖). 대신 `research/wave11_yield/fetch_y11.py`에 올바르게 "
        "페이지네이션하는 로컬 fetcher를 구현했고, 이 wave가 새로 수집하는 모든 데이터"
        "(Y4 확장 종목, Y5 스팟 1h)뿐 아니라 baseline-40 전체의 스팟 일봉도 교정본으로 "
        "재수집해 `research/wave11_yield/cache/`에 저장했다. `engine_y.py`는 이 교정본을 "
        "wave1 캐시보다 우선 사용하므로 Y1~Y6 6개 후보는 모두 동일하게 교정된 데이터 위에서 "
        "돌았다 -- 6개 축 비교의 공정성은 유지된다. 이 리포트가 비교 기준으로 쓰는 C1은 "
        "당초 wave10에서 이미 계산된 JSON을 그대로 인용한 것이라 구버전(절단된) 데이터 "
        "기준이었다 -- 즉 Y1~Y6끼리의 상대 비교는 깨끗했지만 C1 자체의 절대 수치는 실제보다 "
        "낮게 측정돼 있었다. 2026-07-22 데이터 정합성 후속 작업(phase 2)에서 "
        "`research/wave1/cache`를 스팟 페이지네이션 수정본으로 재수집하고 wave10 C1을 "
        "재계산했으므로(`research/validation/DATA_FIX_REPORT.md`), 위 표의 C1 기준선은 "
        "이제 그 교정된 C1.json을 그대로 인용한다 -- Y1~Y6 자체는 처음부터 자체 교정 캐시를 "
        "썼으므로 이 wave의 결과값(equity/gates/promotion 판정)은 이번 갱신으로 전혀 "
        "바뀌지 않았고, 바뀐 것은 비교 기준선(C1)의 절대 수치뿐이다. gates_y.py의 승격 "
        "문턱(PROMOTION_HIGH_FUNDING_ANNUALIZED_MIN=16.84%, PROMOTION_BLOCK_MDD_MAX=4.6%)은 "
        "SPEC.md에 사전등록된 리터럴 값이라 의도적으로 갱신하지 않았다 -- 데이터 교정이 "
        "사후에 판정 문턱 자체를 움직이면 사전등록의 의미가 없어지기 때문이다. Y4는 이 "
        "고정 문턱 기준으로도, 아래 표의 교정된 C1 절대 수치 기준으로도 두 조건을 모두 "
        "통과한다(교정 후 C1 대비 격차는 오히려 소폭 확대: 고펀딩기 연환산 +1.94%p, "
        "MC p05 $4.95).",
        "",
    ]


def write_wave11_report(results_dir: Path, report_dir: Path, registry_path: Path) -> None:
    payloads = {candidate_id: _load(results_dir, candidate_id) for candidate_id in CONFIG_IDS}
    report_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Wave-11 리포트 -- 캐리 수익률 최대화 (Y1~Y6)",
        "",
        "사전등록: research/wave11_yield/SPEC.md. 목적함수: S1~S4(델타중립/1x, MC 원금보존, "
        "블록MDD, $100실행가능) 제약 하에서 연환산 수익률 최대화. wave-10 C1을 넘는 것이 "
        "'더 벌기'가 아니라 '포착률(activity) 증가'로만 달성 가능한지 시험한다.",
        "",
        f"자본 규약: 총자본 ${TOTAL_CAPITAL:.0f} / 현금버퍼 10% / 활성자본 ${ACTIVE_CAPITAL:.0f} / "
        f"최소주문 ${MIN_ORDER_USDT:.2f}. 비용: 메이커 0.02%/레그 + 슬리피지(메이저 1bp/알트 3bp), "
        "펀딩 8h 실적립, 신호는 t종가 확정 -> t+1바 시가 체결 (룩어헤드 없음) -- wave10과 "
        "동일, 유리하게 바꾸지 않았다.",
        "",
        *_data_quality_section(),
        "## 구성 (사전등록 6개, 사후 추가 없음)",
        "",
        *_config_table(),
        "",
        "## 게이트 결과 (S1 구조 / S2 MC 1e4부트스트랩 원금보존 / S3 블록셔플90일MDD / S4 실행가능성)",
        "",
        *_gate_table(payloads),
        "",
        "## 수익률·가동률·비용 (필수 산출 열)",
        "",
        *_yield_table(payloads),
        "",
        "## C1 대비 개선/악화",
        "",
        *_c1_comparison_table(payloads),
        "",
        *_verdict_section(payloads),
        *_axis_breakdown_section(payloads),
        "## 다중검정 보정 (참고 지표, DSR)",
        "",
        f"누적 시행 {DSR_CUMULATIVE_TRIALS}회 기준 (SPEC.md 사전등록치). 샤프·DSR은 참고 지표로만 "
        "기록하며 게이트 판정에는 쓰지 않는다 (S1~S4 및 승격조건만 판정에 사용).",
        "",
        *_dsr_table(payloads),
        *_dsr_caveat_lines(payloads),
        "",
        "## 모델링 노트",
        "",
        "- 바뀐 것은 SPEC.md의 5개 축(임계/폭/유니버스/속도/스파이크)뿐: 델타중립 2레그, "
        "레버리지 1x, 비용모델(메이커+슬리피지), 펀딩 8h 실적립, 신호 t종가->t+1바 체결은 "
        "6개 후보 전부 동일. Y1/Y2/Y3/Y4는 "
        "research.wave10_carry100.engine.run_fixed_fraction_portfolio와 수치적으로 동일한 "
        "엔진(research.wave11_yield.engine_y._run_fixed_fraction_loop, "
        "tests/test_wave11.py가 두 엔진의 출력이 합성시장에서 정확히 일치함을 회귀검증)으로 "
        "실행했고, Y5(8h바)·Y6(스파이크 규칙)만 축이 요구하는 만큼만 엔진을 갈래쳤다.",
        "- Y5는 스팟+퍼프 실캔들이 모두 있는 BTC/ETH/SOL로 유니버스를 제한했다 (다른 종목은 "
        "스팟 인트라데이 데이터가 캐시에도, 이 wave의 수집 범위에도 없어 가정으로 채우지 "
        "않았다).",
        "- Y6의 2쌍 사이징(25%/레그)은 SPEC.md에 레그 $가 명시되지 않아 wave10 C3의 2쌍 "
        "선례를 그대로 적용한 것 -- configs.py에 명시.",
        "- MC/블록셔플 방법론은 research/wave10_carry100/gates.py(및 그 뿌리인 wave8)와 "
        "동일 경로수/자본기준/블록길이를 쓰되, S2/S3의 PASS 임계값은 SPEC.md가 등록한 "
        "wave11 자체 기준(파산<1%, 블록MDD<=10%)으로 wave10보다 엄격하다.",
        "- 이 리포트는 paper 백테스트 결과이며 실계좌 주문을 실행하지 않는다. 메이커 체결률은 "
        "실측이 아닌 가정이다 (wave1/wave2/wave10과 동일한 한계).",
        "",
    ]
    (report_dir / "wave11_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = ["# Wave-11 registry", "", "| Candidate | Family | State | Verdict | 승격 | 분류 |", "|---|---|---|---|---|---|"]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        state = "EVALUATED" if gates else "RUN_ONLY"
        verdict = gates["overall"] if gates else "UNTESTED"
        promoted = "YES" if gates and gates.get("promotion", {}).get("promoted") else "no"
        classification = _classify_blockage(candidate_id, payload) if gates else "run --stage gates 필요"
        registry_lines.append(f"| {candidate_id} | wave11_yield | {state} | {verdict} | {promoted} | {classification} |")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave11_report"]
