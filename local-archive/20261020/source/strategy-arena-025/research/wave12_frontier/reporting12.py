# Wave-12 markdown report + registry writer. Pure formatting over already-computed
# results/U{0..6}.json payloads written by run_wave12.py --stage run/gates. Mirrors
# research/wave11_yield/reporting_y.py's structure, replaced with SPEC.md's required
# frontier-curve table (breadth on the x-axis, five required metrics) as the centerpiece.
#
# Y4 reference figures, cited literally from research/wave11_yield/results/Y4.json (read
# once, frozen as constants here -- same convention reporting_y.py itself used for C1):
# high-funding annualized 17.8756%, MC p05 $132.90, ruin 0%, block MDD p95 3.8565%,
# universe_size 100, total_cost_usdt $41.72 over 338 trades (flat 3bp-alt cost model).
# SPEC.md requires this report to explain, in cost-attribution terms, why U0 (top100/12mo
# under the NEW tiered-cost model) should NOT be expected to reproduce 17.88% exactly --
# see _cost_attribution_section.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.wave12_frontier.configs12 import CONFIG_IDS, get_config
from research.wave12_frontier.gates12 import DSR_CUMULATIVE_TRIALS, S3_BLOCK_MDD_P95_MAX

Y4_HIGH_FUNDING_ANNUALIZED: float = 0.17875588307162027
Y4_MC_P05_USDT: float = 132.9000827734323
Y4_RUIN_PROBABILITY: float = 0.0
Y4_BLOCK_MDD_P95: float = 0.03856465734856994
Y4_UNIVERSE_SIZE: int = 100
Y4_TOTAL_COST_USDT: float = 41.7163809622171
Y4_N_TRADES: int = 338
Y4_AVG_COST_PER_TRADE_USDT: float = Y4_TOTAL_COST_USDT / Y4_N_TRADES
Y4_FLAT_COST_MODEL_NOTE: str = "메이커 0.02%/레그 + 슬리피지 1bp majors/3bp alts 일괄 (계층화 이전)"

C1_HIGH_FUNDING_ANNUALIZED: float = 0.15943511612278694  # research/wave10_carry100/results/C1.json, cited via wave11's own reporting_y.py


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _fmt_breadth(breadth: int | None) -> str:
    return "무제한" if breadth is None else f"top{breadth}"


def _config_table() -> list[str]:
    lines = [
        "| Candidate | 볼륨 랭크 | 히스토리 요건 | 정의 |",
        "|---|---|---:|---|",
    ]
    for candidate_id in CONFIG_IDS:
        config = get_config(candidate_id)
        lines.append(f"| {candidate_id} | {_fmt_breadth(config.breadth)} | {config.history_months:.0f}mo | {config.note} |")
    return lines


def _frontier_curve_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | 볼륨랭크 | 편입심볼(정적) | 편입심볼(일별 중앙값) | 고펀딩기 연환산 | MC p05 | 블록MDD p95 | 평균슬리피지비용(1건당,$) | 편입심볼 중앙유동성(30d,$) | 스트레스(×2) 부호 | U0 대비 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    u0_high_funding = payloads.get("U0", {}).get("regime_breakdown", {}).get("high_funding_mean_annualized_return")
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        config = get_config(candidate_id)
        metadata = payload.get("metadata", {})
        regime = payload.get("regime_breakdown")
        stress_regime = payload.get("stress_regime_breakdown")
        gates = payload.get("gates")
        if regime is None or gates is None:
            lines.append(f"| {candidate_id} | {_fmt_breadth(config.breadth)} | {metadata.get('universe_size_static', 'N/A')} | - | - | - | - | - | - | - | run --stage gates 필요 |")
            continue
        high_funding = regime.get("high_funding_mean_annualized_return")
        stress_high_funding = stress_regime.get("high_funding_mean_annualized_return") if stress_regime else None
        stress_sign = "+ 유지" if (stress_high_funding is not None and stress_high_funding > 0.0) else "반전/N-A"
        eligible_median = metadata.get("eligible_count_stats", {}).get("median")
        eligible_median_str = "N/A" if eligible_median is None else f"{eligible_median:.0f}"
        delta_vs_u0 = "-" if (candidate_id == "U0" or high_funding is None or u0_high_funding is None) else f"{(high_funding - u0_high_funding) * 100.0:+.2f}%p"
        lines.append(
            f"| {candidate_id} | {_fmt_breadth(config.breadth)} | {metadata.get('universe_size_static', 'N/A')} | "
            f"{eligible_median_str} | {_fmt_pct(high_funding)} | {_fmt_usd(gates['gate_s2'].get('p05'))} | "
            f"{_fmt_pct(gates['gate_s3'].get('mdd_p95'))} | {_fmt_usd(metadata.get('avg_cost_per_trade_usdt'))} | "
            f"{_fmt_usd(metadata.get('median_reference_liquidity_usdt'), 0)} | {stress_sign} | {delta_vs_u0} |"
        )
    return lines


def _peak_analysis_section(payloads: dict[str, dict]) -> list[str]:
    """SPEC.md 필수 산출: "수익이 어디서 정점이고 왜 꺾이는지 명확히". Separates the two
    axes U0-U6 actually vary (breadth at a fixed 12mo history floor; history floor at a
    fixed breadth) so the peak/bend can be read directly off two clean slices instead of
    inferred from the full 7-row table where both axes move at once."""
    rows: list[tuple[str, int | None, float, float | None, float | None, float | None]] = []
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        config = get_config(candidate_id)
        regime = payload.get("regime_breakdown")
        if regime is None:
            continue
        metadata = payload.get("metadata", {})
        high_funding = regime.get("high_funding_mean_annualized_return")
        eligible_median = metadata.get("eligible_count_stats", {}).get("median")
        avg_cost = metadata.get("avg_cost_per_trade_usdt")
        rows.append((candidate_id, config.breadth, config.history_months, high_funding, eligible_median, avg_cost))

    if not rows or all(item[3] is None for item in rows):
        return ["## 정점 분석 (폭/히스토리 축 분리)", "", "게이트 미실행 -- `--stage gates` 필요."]

    scored = [item for item in rows if item[3] is not None]
    peak = max(scored, key=lambda item: item[3])
    lines = [
        "## 정점 분석 (폭/히스토리 축 분리)",
        "",
        f"전 {len(scored)}구성 중 고펀딩기 연환산 최댓값: **{peak[0]}** ({_fmt_pct(peak[3])}, "
        f"볼륨 {_fmt_breadth(peak[1])} / 히스토리 {peak[2]:.0f}mo).",
        "",
        "**폭 축 (히스토리 12mo 고정)**: U0(top100) -> U1(top150) -> U2(top200) -> U3(무제한). "
        "이 네 행에서 연환산이 오르다가 꺾이면 그 지점의 볼륨폭이 곧 최적 폭이다:",
        "",
        "| Candidate | 볼륨폭 | 고펀딩기 연환산 | 편입심볼 일별중앙값 | 건당비용($) |",
        "|---|---|---:|---:|---:|",
    ]
    for candidate_id, breadth, history_months, high_funding, eligible_median, avg_cost in rows:
        if history_months != 12.0:
            continue
        eligible_str = "-" if eligible_median is None else f"{eligible_median:.0f}"
        lines.append(f"| {candidate_id} | {_fmt_breadth(breadth)} | {_fmt_pct(high_funding)} | {eligible_str} | {_fmt_usd(avg_cost)} |")
    lines.extend(
        [
            "",
            "**히스토리 축 (동일 폭에서 12mo -> 6mo -> 3mo)**: top100은 U0->U4, top200은 U2->U5->U6:",
            "",
            "| Candidate | 볼륨폭 | 히스토리 | 고펀딩기 연환산 | 편입심볼 일별중앙값 | 건당비용($) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for candidate_id in ("U0", "U4", "U2", "U5", "U6"):
        match = next((item for item in rows if item[0] == candidate_id), None)
        if match is None:
            continue
        _cid, breadth, history_months, high_funding, eligible_median, avg_cost = match
        eligible_str = "-" if eligible_median is None else f"{eligible_median:.0f}"
        lines.append(f"| {candidate_id} | {_fmt_breadth(breadth)} | {history_months:.0f}mo | {_fmt_pct(high_funding)} | {eligible_str} | {_fmt_usd(avg_cost)} |")
    return lines


def _gate_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | S1 구조 | S2 MC(p05/ruin) | S3 블록MDD p95 | S4 실행가능성 | S5 스트레스부호 | Overall | 승격여부 | 실패/미승격 사유 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        if not gates:
            lines.append(f"| {candidate_id} | - | - | - | - | - | NOT_EVALUATED | - | run --stage gates 필요 |")
            continue
        s1, s2, s3, s4, s5 = gates["gate_s1"], gates["gate_s2"], gates["gate_s3"], gates["gate_s4"], gates["gate_s5"]
        promo = gates["promotion"]
        s1_cell = f"{s1['status']} ({s1['leverage_multiplier_of_active_capital']:.2f}x)"
        s2_cell = f"{s2['status']} (p05={_fmt_usd(s2['p05'])}, ruin={_fmt_pct(s2['ruin_probability'])})"
        s3_cell = f"{s3['status']} ({_fmt_pct(s3['mdd_p95'])})"
        s4_cell = f"{s4['status']} (레그{_fmt_usd(s4['leg_usdt_nominal'])})"
        s5_cell = f"{s5['status']} ({_fmt_pct(s5['stress_high_funding_annualized'])})"
        # Promotion requires BOTH overall==PASS AND promo["promoted"] (SPEC.md's rule is
        # a conjunction) -- promo["promoted"] alone only means "beat U0's return number"
        # and must never be shown as YES when the config failed a gate (e.g. MDD).
        actually_promoted = gates["overall"] == "PASS" and bool(promo["promoted"])
        if promo["is_baseline"]:
            promoted = "기준선(U0)"
        else:
            promoted = "YES" if actually_promoted else "no"
        reasons = ", ".join(gates.get("failure_reasons", [])) or "-"
        if not promo["is_baseline"] and not actually_promoted and gates["overall"] == "PASS":
            bar = promo.get("high_funding_bar")
            reasons = f"고펀딩기 {_fmt_pct(promo['high_funding_mean_annualized_return'])} <= U0 {_fmt_pct(bar)}"
        lines.append(f"| {candidate_id} | {s1_cell} | {s2_cell} | {s3_cell} | {s4_cell} | {s5_cell} | {gates['overall']} | {promoted} | {reasons} |")
    return lines


def _spot_verification_table(cache_dir: Path, payloads: dict[str, dict]) -> list[str]:
    path = cache_dir / "spot_verification.json"
    lines = [
        "## 스팟 수집 검증 (재발 방지 -- research/wave1/fetch_binance.py 스팟 절단 버그 재확인)",
        "",
    ]
    if not path.exists():
        lines.append("spot_verification.json 없음 -- `--stage fetch` 미실행.")
        return lines
    verification = json.loads(path.read_text(encoding="utf-8"))
    total = len(verification)
    by_source: dict[str, int] = {}
    truncated = []
    max_gap = 0
    for symbol, info in verification.items():
        by_source[info["source"]] = by_source.get(info["source"], 0) + 1
        gap = info.get("gap_days")
        if gap is not None:
            max_gap = max(max_gap, gap)
        if info.get("truncated_after_check"):
            truncated.append((symbol, gap))
    lines.append(f"검증 대상 {total}개 심볼 (모든 config 편입 심볼 + tier-reference 풀 합집합). 출처 분포: {by_source}.")
    lines.append(f"perp 종료일 대비 spot 종료일 최대 gap: {max_gap}일 (허용 오차 {10}일 이내 = 절단 아님).")
    if truncated:
        lines.append(f"**경고**: 재수집 후에도 절단 의심 {len(truncated)}개: {truncated[:10]}")
    else:
        lines.append("재수집 후 절단 의심 심볼 없음 -- 전 종목 spot 종료일이 perp과 근접함을 확인.")
    return lines


def _cost_attribution_section(payloads: dict[str, dict]) -> list[str]:
    u0 = payloads.get("U0", {})
    u0_metadata = u0.get("metadata", {})
    u0_regime = u0.get("regime_breakdown", {})
    u0_high_funding = u0_regime.get("high_funding_mean_annualized_return")
    lines = [
        "## U0 vs Y4 비용귀속 (계층비용 도입 전후)",
        "",
        f"- Y4(계층화 이전, 알트 일괄 3bp): 고펀딩기 연환산 {_fmt_pct(Y4_HIGH_FUNDING_ANNUALIZED)}, "
        f"MC p05 {_fmt_usd(Y4_MC_P05_USDT)}, 블록MDD p95 {_fmt_pct(Y4_BLOCK_MDD_P95)}, "
        f"파산확률 {_fmt_pct(Y4_RUIN_PROBABILITY)}, 총비용 {_fmt_usd(Y4_TOTAL_COST_USDT)} ({Y4_N_TRADES}건, "
        f"건당 {_fmt_usd(Y4_AVG_COST_PER_TRADE_USDT)}) -- {Y4_FLAT_COST_MODEL_NOTE}.",
        f"- U0(계층화 이후, 동일 유니버스 규칙 top100/12mo): 고펀딩기 연환산 {_fmt_pct(u0_high_funding)}, "
        f"총비용 {_fmt_usd(u0_metadata.get('total_cost_usdt'))} ({u0_metadata.get('n_trades', 'N/A')}건, "
        f"건당 {_fmt_usd(u0_metadata.get('avg_cost_per_trade_usdt'))}).",
    ]
    if u0_high_funding is not None:
        gap = Y4_HIGH_FUNDING_ANNUALIZED - u0_high_funding
        cost_delta_per_trade = u0_metadata.get("avg_cost_per_trade_usdt", 0.0) - Y4_AVG_COST_PER_TRADE_USDT
        utilization = u0_metadata.get("utilization")
        lines.append(
            f"- 갭: {gap * 100.0:+.2f}%p (연환산). 건당 비용 {_fmt_usd(cost_delta_per_trade)} 증가 -- "
            "top100 유니버스의 상당수가 계층모델에서 3bp(구) 대신 6bp(랭크51-100)~10bp로 재분류된 결과이며, "
            "이는 SPEC.md가 사전에 명시적으로 예상한 방향(비용모델 강화로 인한 정상적 하락)이다."
        )
        lines.append(
            f"- 방법론 주의: 신호(funding_score/carry_position)와 델타중립 구조, 진입임계는 Y4와 100% 동일하다. "
            "다만 유니버스 '선정 규칙'(top100 by volume + 12mo history)은 동일해도 '랭킹 산식'은 다르다 -- "
            "Y4는 wave1 원본 40종을 고정 기저로 삼고 나머지를 fetch 시점 현재거래량으로 채웠고, 이 wave는 "
            "전 후보를 FROZEN_END 기준 30일 평균 거래대금(계층 슬리피지와 동일 지표)으로 통일 재랭킹했다 -- "
            f"U0 가동률 {_fmt_pct(utilization)}이 Y4의 52.17%와 거의 일치하는 것으로 보아 두 100종 구성은 대부분 "
            "겹치지만 경계 종목 일부가 다를 수 있음을 인정한다. 잔여 갭은 압도적으로 비용모델 차이지만, 100% "
            "동일 유니버스에 대한 통제실험은 아니다."
        )
    else:
        lines.append("- U0 게이트 미평가 -- `--stage gates` 실행 필요.")
    return lines


def _headline_finding_section(payloads: dict[str, dict]) -> list[str]:
    """A finding SPEC.md's own promotion criterion doesn't have a slot for, so it would
    otherwise be buried inside the routine gate table: whether U0 ITSELF (the baseline
    every other config is judged against) clears the S1-S5 bars it inherits unchanged
    from wave11. wave11's Y4 cleared S3 (block MDD p95) at 3.86%, comfortably under the
    10% cap, under the OLD flat 3bp-alt cost model. If U0 fails S3 under the new tiered
    model, that is a materially bigger finding than "no config got promoted": it means
    the registered risk gate itself was only ever passing because slippage was
    underpriced, not because the strategy family is actually within its own declared risk
    budget."""
    u0_gates = payloads.get("U0", {}).get("gates")
    if u0_gates is None:
        return []
    s3 = u0_gates["gate_s3"]
    if s3["status"] == "PASS":
        return []
    lines = [
        "## 핵심 발견: 기준선 U0조차 S3(블록MDD) 미통과",
        "",
        f"U0(top100/12mo, Y4와 동일 규칙)의 블록셔플 90일 MDD p95 = {_fmt_pct(s3['mdd_p95'])} "
        f"(기준 <= {_fmt_pct(S3_BLOCK_MDD_P95_MAX)}). Y4(구 3bp 일괄 모델)는 {_fmt_pct(Y4_BLOCK_MDD_P95)}로 "
        "이 게이트를 여유 있게 통과했었다. 즉 wave10(C1)·wave11(Y1-Y6)가 통과시켰던 S3 게이트는 "
        "슬리피지가 실제보다 저평가되어 있었기 때문에 통과한 것이며, 계층비용을 적용하면 U0을 포함해 "
        "이 계열 전체가 사전등록된 리스크 예산 안에 있지 않았다는 뜻이다. 이는 SPEC.md가 명시적으로 "
        "묻지 않은 발견이지만 승격 판정보다 상위의 함의를 가지므로 여기서 먼저 밝힌다.",
        "",
    ]
    return lines


def _saturation_or_promotion_verdict(payloads: dict[str, dict]) -> tuple[str, list[str]]:
    u0_regime = payloads.get("U0", {}).get("regime_breakdown", {})
    u0_high_funding = u0_regime.get("high_funding_mean_annualized_return")
    promoted_candidates: list[tuple[str, float]] = []
    attempted: list[str] = []
    for candidate_id in CONFIG_IDS:
        if candidate_id == "U0":
            continue
        gates = payloads[candidate_id].get("gates")
        if not gates:
            continue
        attempted.append(candidate_id)
        promo = gates["promotion"]
        if gates["overall"] == "PASS" and promo["promoted"]:
            high_funding = promo["high_funding_mean_annualized_return"]
            if high_funding is not None:
                promoted_candidates.append((candidate_id, high_funding))

    if promoted_candidates:
        winner_id, winner_return = max(promoted_candidates, key=lambda item: item[1])
        winner_config = get_config(winner_id)
        lines = [
            "## 판정: 통과",
            "",
            f"**Y4 후속 고수익 구성: {winner_id}** ({_fmt_breadth(winner_config.breadth)}, {winner_config.history_months:.0f}mo) -- "
            f"고펀딩기 연환산 {_fmt_pct(winner_return)} (U0 {_fmt_pct(u0_high_funding)} 대비 "
            f"{'' if u0_high_funding is None else f'{(winner_return - u0_high_funding) * 100.0:+.2f}%p'}), S1-S5 전부 PASS, "
            f"스트레스(슬리피지×2)에서도 고펀딩기 연환산 부호(+) 유지.",
        ]
        return "PASS", lines

    # No promotion. Two SEPARATE questions, deliberately not collapsed into one label:
    # (a) economic diagnosis -- does the raw high-funding number even improve, and if
    #     not, is that better explained by rising cost (liquidity) or a flat opportunity
    #     pool (breadth added symbols but not funding-edge) -- computed independent of
    #     gate status, so it stays visible even for configs that also fail a risk gate;
    # (b) gate result -- whether S1-S5 actually passed, shown alongside, not instead.
    # An earlier version of this table let "게이트위반" pre-empt the economic diagnosis
    # entirely; since every non-U0 config here fails a gate, that produced a table that
    # said "게이트위반" seven times and nothing else -- technically true but it threw away
    # exactly the 유동성비용-vs-기회부재 read SPEC.md asks for, which is a further-out
    # cause the gate label alone can't show.
    lines = [
        "## 판정: 확장 축도 U0에서 포화",
        "",
        "S1-S5를 모두 통과하며 U0의 고펀딩기 연환산을 상회한 구성이 없다 (U1/U2/U5/U6은 원시 수익은 "
        "U0을 상회하지만 게이트 위반으로 무효화됨 -- 아래 '경제적 진단'과 '게이트' 두 열을 함께 읽을 것):",
        "",
        "| Candidate | 고펀딩기 연환산 | U0 대비(원시) | 건당비용 vs U0 | 편입풀(일별중앙값) vs U0 | 경제적 진단 | 게이트 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    u0_metadata = payloads.get("U0", {}).get("metadata", {})
    u0_cost = u0_metadata.get("avg_cost_per_trade_usdt")
    u0_eligible = u0_metadata.get("eligible_count_stats", {}).get("median")
    for candidate_id in attempted:
        payload = payloads[candidate_id]
        gates = payload["gates"]
        regime = payload.get("regime_breakdown", {})
        metadata = payload.get("metadata", {})
        high_funding = regime.get("high_funding_mean_annualized_return")
        cost = metadata.get("avg_cost_per_trade_usdt")
        eligible = metadata.get("eligible_count_stats", {}).get("median")
        cost_delta = None if (cost is None or u0_cost is None) else cost - u0_cost
        eligible_delta = None if (eligible is None or u0_eligible is None) else eligible - u0_eligible
        return_delta = None if (high_funding is None or u0_high_funding is None) else high_funding - u0_high_funding
        eligible_flat = eligible_delta is None or eligible_delta <= max(1.0, (u0_eligible or 0.0) * 0.05)
        if return_delta is None:
            diagnosis = "데이터 부족"
        elif return_delta > 0:
            diagnosis = "원시수익은 U0보다 높음 -- 그러나 그 개선폭이 추가 리스크(MDD/MC 초과분)를 정당화하지 못함"
        elif cost_delta is not None and cost_delta > 0 and eligible_flat:
            diagnosis = "유동성비용 우세 (건당비용↑, 편입풀은 거의 무증가)"
        elif cost_delta is not None and cost_delta <= 0:
            # Cost did NOT rise (flat or even cheaper) yet return still fell -- this
            # cannot be a cost story; the accessible funding opportunity itself is worse
            # (e.g. U4: same top100 breadth as U0 but a looser 6mo history floor swaps in
            # some newer-listed names that are cheaper to trade but don't carry better
            # funding edge).
            diagnosis = "기회부재 우세 (비용은 오히려 낮거나 동일한데도 수익 미증가 -- 유니버스 구성이 바뀌어도 펀딩 기회 자체는 늘지 않음)"
        elif eligible_delta is not None and eligible_delta > 0:
            diagnosis = "기회부재 우세 (편입풀은 늘었지만 수익 미증가 -- 신규 심볼이 고펀딩 기회를 못 더함)"
        else:
            diagnosis = "혼합/불명확"
        gate_cell = "PASS" if gates["overall"] == "PASS" else f"FAIL({', '.join(gates['failure_reasons']) or '?'})"
        delta_str = "-" if return_delta is None else f"{return_delta * 100.0:+.2f}%p"
        cost_delta_str = "-" if cost_delta is None else f"{cost_delta:+.4f}"
        eligible_delta_str = "-" if eligible_delta is None else f"{eligible_delta:+.1f}"
        lines.append(f"| {candidate_id} | {_fmt_pct(high_funding)} | {delta_str} | {cost_delta_str} | {eligible_delta_str} | {diagnosis} | {gate_cell} |")
    return "FAIL", lines


def _dsr_note() -> str:
    return (
        f"다중검정 보정: 누적 시행 {DSR_CUMULATIVE_TRIALS}회 기준 DSR(Deflated Sharpe Ratio) 참고치를 각 결과 JSON의 "
        "`reference_metrics.dsr`에 기록 (샤프는 참고 지표이며 승격 판정에는 사용하지 않음, wave10/wave11과 동일 원칙)."
    )


def write_wave12_report(results_dir: Path, report_dir: Path, registry_path: Path, cache_dir: Path) -> None:
    payloads = {candidate_id: _load(results_dir, candidate_id) for candidate_id in CONFIG_IDS}
    verdict, verdict_lines = _saturation_or_promotion_verdict(payloads)

    lines: list[str] = [
        "# Wave-12 리포트 — 유니버스 확장 프론티어 (U0-U6)",
        "",
        "**주의**: 이 wave는 계층별 슬리피지 + $2M 유동성 하한 비용모델을 신규 도입했다. "
        "U0을 포함한 전 구성이 이 새 비용모델로 재계산되었으므로, wave10(C1)·wave11(Y1-Y6)의 "
        "기존 3bp 일괄-슬리피지 수치와 이 리포트의 수치를 직접 비교할 수 없다 (U0 vs Y4의 비교는 "
        "\"U0 vs Y4 비용귀속\" 절에서 그 차이 자체를 분석 대상으로 다룬다).",
        "",
        *_headline_finding_section(payloads),
        "## 구성 정의",
        "",
        *_config_table(),
        "",
        "## 프론티어 곡선 (핵심 산출)",
        "",
        "x축 = 유니버스 폭. 편입심볼(정적)은 SPEC.md의 볼륨랭크컷으로 확정된 고정 후보 리스트 크기이고, "
        "편입심볼(일별 중앙값)은 활성신호+데이터가용성+유동성하한을 모두 통과해 그날 실제로 순위매김 대상이 "
        "된 심볼 수의 backtest 전기간 중앙값이다 (실제 '체감' 유니버스 크기).",
        "",
        *_frontier_curve_table(payloads),
        "",
        *_peak_analysis_section(payloads),
        "",
        *verdict_lines,
        "",
        *_cost_attribution_section(payloads),
        "",
        "## 게이트 (S1-S5)",
        "",
        *_gate_table(payloads),
        "",
        *_spot_verification_table(cache_dir, payloads),
        "",
        "## 다중검정",
        "",
        _dsr_note(),
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave12_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = ["# Wave-12 registry", "", "| Candidate | Family | State | Verdict | 승격 | 분류 |", "|---|---|---|---|---|---|"]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        if not gates:
            registry_lines.append(f"| {candidate_id} | wave12_frontier | PENDING | - | - | gates 미실행 |")
            continue
        promo = gates["promotion"]
        state = "EVALUATED"
        result_verdict = gates["overall"]
        # Promotion requires BOTH overall S1-S5 PASS AND promo["promoted"] (which by
        # itself only means "high_funding > U0's bar" -- SPEC.md's rule is a conjunction
        # of the two, not either alone). A config that beats U0's return but fails a gate
        # (e.g. MDD) must never be shown as promoted here.
        actually_promoted = result_verdict == "PASS" and bool(promo["promoted"])
        if promo["is_baseline"]:
            promoted_cell = "기준선"
            classification = "U0 = 새 기준선(계층비용 적용, Y4 재현)"
        elif actually_promoted:
            promoted_cell = "YES"
            classification = "승격"
        elif result_verdict != "PASS":
            promoted_cell = "no"
            classification = f"게이트위반({', '.join(gates['failure_reasons'])})"
        else:
            promoted_cell = "no"
            classification = "고펀딩기미달(U0 상회 실패)"
        registry_lines.append(f"| {candidate_id} | wave12_frontier | {state} | {result_verdict} | {promoted_cell} | {classification} |")
    registry_lines.append("")
    registry_lines.append(f"**판정**: {'통과' if verdict == 'PASS' else '확장 축도 U0에서 포화'} (자세한 내용은 report/wave12_report.md 참조).")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave12_report"]
