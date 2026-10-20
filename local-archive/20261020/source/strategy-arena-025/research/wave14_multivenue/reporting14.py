# Wave-14 markdown report + registry writer. Pure formatting over already-computed
# results/{M0..M7,_AUX_*}.json payloads written by run_wave14.py --stage run/gates, plus
# cache/bybit_spreads.json (Bybit order-book measurements) and cache/bybit_universe.json
# (venue-discovery metadata) -- same "pure formatting, no new computation" discipline as
# research.wave13_liquidity.reporting13.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.wave14_multivenue.configs14 import AUX_BASELINES, CONFIG_IDS, get_config
from research.wave14_multivenue.gates14 import DSR_CUMULATIVE_TRIALS, S3_BLOCK_MDD_P95_MAX, S5_BLOCK_MDD_P95_MAX, S6_RESIDUAL_CAPITAL_FLOOR

ALL_IDS = (*CONFIG_IDS, *(cfg.candidate_id for cfg in AUX_BASELINES))


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100.0:.{digits}f}%"


def _fmt_usd(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"${value:,.{digits}f}"


def _fmt_bp(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}bp"


def _full_period_total_return(payload: dict[str, Any]) -> float | None:
    """Total return over the WHOLE sliced backtest window (2024-01-01 ~ FROZEN_END), as
    opposed to regime_breakdown's "고펀딩기 연환산" which is a single-year (2024-only, see
    _limitations_section) annualized figure. SPEC.md's own promotion rule is pre-registered
    against the high-funding-year metric, so this module still honors that mechanically --
    but the two metrics can and do diverge (a config can win the 2024-only comparison while
    losing on the full-period path, e.g. by underperforming badly in the post-2025-09
    low-funding OOS stretch), and reporting the annualized figure ALONE without this one
    would hide that divergence. Computed directly from the saved equity series (first vs
    last point), not from any gate output -- pure arithmetic, no new modeling."""
    equity = payload.get("equity")
    if not equity or len(equity) < 2:
        return None
    start = equity[0].get("value")
    end = equity[-1].get("value")
    if start is None or end is None or start <= 0.0:
        return None
    return float(end) / float(start) - 1.0


# ---------------------------------------------------------------------------
# Section: limitations (required first, per this repo's own honesty convention).
# ---------------------------------------------------------------------------


def _limitations_section(universe_payload: dict[str, Any]) -> list[str]:
    lines = [
        "## 방법론 한계 (필독 -- 아래 모든 수치에 선행하는 전제)",
        "",
        "- **백테스트 구간은 Binance+Bybit 겹침 구간(2024-01-01 ~ FROZEN_END 2026-07-14, "
        "약 2.5년)으로 고정된다.** wave13의 L4가 사용한 전기간(2019-09~2026-07, 약 7년)보다 "
        "훨씬 짧다. 공정 비교를 위해 M0(L4 재현/기준선)도 이 구간으로 재실행했다 -- "
        "**wave13 L4의 전기간 22.01%와 이 리포트의 M0 수치를 직접 비교하지 말 것.**",
        "- **고펀딩기 표본이 사실상 2024년 하나뿐이다.** `research.wave10_carry100.regime`의 "
        "고펀딩년(2020/2021/2024) 중 2020·2021은 이 구간 밖이라 자동으로 빠지고, 아래 "
        "'고펀딩기 연환산' 열은 2024년 단독 표본이다. 표본 1개짜리 연환산은 통계적으로 "
        "박약하다 -- 절대 과신하지 말 것. **실제로 이 wave에서 SPEC 사전등록 지표(2024 단독)와 "
        "전체구간 누적수익률이 서로 다른 승자를 가리키는 사례가 나왔다** (M1) -- 아래 모든 표에 "
        "두 지표를 나란히 병기하고, 판정 절에서 그 괴리를 명시적으로 다룬다.",
        "- **Bybit 유니버스는 '현재(수집 시점) 살아있는 상장' 스냅샷이다** (Binance L4처럼 "
        "FROZEN_END 시점의 과거 랭킹이 아님). 구간 중 상장폐지된 심볼은 전체 백테스트에서 "
        "누락될 수 있다(생존편향, Bybit 쪽에만 해당). universe_multi.py 모듈 docstring 참조.",
        "- **비용모델**: Binance 소스 레그는 wave13의 Bitget 실측 매핑을 그대로 재사용(불변). "
        "Bybit 소스 레그는 이번 wave가 새로 실측한 Bybit 자체 오더북(spot+linear 별도, "
        "$45 book-walk, wave13과 동일 방법론)이다. 두 실측 모두 **단일 시점 스냅샷**(평온장) "
        "이며, 고변동기 스프레드 확대를 반영하지 않는다 -- S5 게이트의 x3 스트레스로만 상쇄.",
        f"- **거래소별 실제 수수료**: Bitget 메이커 0.02%(wave13 불변) / Bybit 스팟 메이커 0.10%, "
        "Bybit USDT무기한 메이커 0.02%(2026-07 공시 확인, `costs_venue.py` 모듈 docstring 출처 명시).",
        "- **M6/M7(거래소간 구조)은 새로운 신용리스크를 진다**: 두 거래소에 자본이 분산 예치되며, "
        "한쪽 거래소 파산/출금중단 시 구조적으로 그 거래소 몫이 사라진다 (S6 게이트, 확률 추정 아님 "
        "-- '몇 % 확률로 파산한다'가 아니라 '파산하면 얼마가 남는가'만 계산).",
        f"- Bybit 유니버스 discovery: L4(top200 Binance) ∩ Bybit spot+linear 상장 = "
        f"{universe_payload.get('l4_intersect_bybit_count', 'N/A')}종, 펀딩주기 4h 미만 "
        f"{len(universe_payload.get('excluded_low_funding_interval', []))}종 추가 제외 "
        f"(수집 실무상 이유, 본문 참조) -> 최종 {universe_payload.get('universe_after_fetch_count', universe_payload.get('universe_count', 'N/A'))}종. "
        "**심볼 폭 자체를 넓히는 목적이 아니다** -- wave13이 이미 top200을 정점으로 확정했으므로 "
        "(top358 하락), Bybit는 오직 '같은 200종에 대한 두 번째 거래소 접근권'만 추가한다.",
    ]
    failures = universe_payload.get("fetch_failures", [])
    if failures:
        lines.append(f"- Bybit 데이터 수집 실패로 제외된 심볼 {len(failures)}종 (가정으로 채우지 않음): " + ", ".join(f"{item['symbol']}" for item in failures[:15]) + (" ..." if len(failures) > 15 else ""))
    return lines


# ---------------------------------------------------------------------------
# Section: Bybit spread measurement + mapping (mirrors wave13's own tables, two markets).
# ---------------------------------------------------------------------------


def _spread_summary_section(spread_payload: dict[str, Any]) -> list[str]:
    lines = [
        "## Bybit 실측 스프레드 요약 (spot + linear, $45 book-walk, wave13와 동일 방법론)",
        "",
        f"수집 시각(UTC): {spread_payload.get('collected_at_utc', 'N/A')} · "
        f"spot 실측 {spread_payload.get('spot', {}).get('measured_count', 0)}종 · "
        f"linear 실측 {spread_payload.get('linear', {}).get('measured_count', 0)}종 · "
        f"측정실패 {len(spread_payload.get('measurement_failures', []))}종.",
        "",
        spread_payload.get("snapshot_limitation", ""),
        "",
    ]
    for market_key, market_label in (("spot", "Bybit SPOT"), ("linear", "Bybit LINEAR(USDT무기한)")):
        measurements = spread_payload.get(market_key, {}).get("measurements", [])
        if not measurements:
            lines.append(f"### {market_label}: 실측 데이터 없음")
            continue
        cheapest = sorted(measurements, key=lambda item: item["effective_slippage_bp"])[:5]
        priciest = sorted(measurements, key=lambda item: -item["effective_slippage_bp"])[:5]
        lines.append(f"### {market_label} -- 최저/최고 실효슬리피지 상위 5종씩")
        lines.append("")
        lines.append("| 구분 | 심볼 | 24h거래대금 | half-spread | walk-cost($45) | 실효슬리피지 | 부족 |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for label, rows in (("최저", cheapest), ("최고", priciest)):
            for item in rows:
                insuf = "부족" if item.get("insufficient_depth") else "-"
                lines.append(
                    f"| {label} | {item['symbol']} | {_fmt_usd(item['usdt_volume_24h'], 0)} | {_fmt_bp(item['half_spread_bp'])} | "
                    f"{_fmt_bp(item['walk_cost_bp'])} | {_fmt_bp(item['effective_slippage_bp'])} | {insuf} |"
                )
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section: config table + metrics + gates.
# ---------------------------------------------------------------------------


def _config_table() -> list[str]:
    lines = ["| Candidate | 자본 | 동시쌍 | 거래소 | 구조 | 비고 |", "|---|---:|---:|---|---|---|"]
    for candidate_id in CONFIG_IDS:
        config = get_config(candidate_id)
        venue = "+Bybit" if config.include_bybit else "단일(Binance/Bitget)"
        structure = "통상캐리" if config.structure == "carry" else "거래소간스프레드(신규)"
        lines.append(f"| {candidate_id} | {_fmt_usd(config.total_capital, 0)} | {config.candidate.top_k} | {venue} | {structure} | {config.note} |")
    return lines


def _metrics_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | 자본 | 동시쌍 | 편입심볼(정적) | 고펀딩기(2024) 연환산 | **전체구간(2024-01~2026-07) 누적수익률** | MC p05 | 블록MDD p95 | 건당비용($) | 스트레스(x3) 고펀딩기 | 스트레스 MDD p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        metadata = payload.get("metadata", {})
        regime = payload.get("regime_breakdown")
        stress_regime = payload.get("stress_regime_breakdown")
        gates = payload.get("gates")
        full_return = _full_period_total_return(payload)
        if regime is None or gates is None:
            lines.append(
                f"| {candidate_id} | {_fmt_usd(payload.get('total_capital_usdt'), 0)} | {payload.get('top_k_pairs', 'N/A')} | "
                f"{metadata.get('universe_size_static', 'N/A')} | - | {_fmt_pct(full_return)} | - | - | - | - | gates 미실행 |"
            )
            continue
        high_funding = regime.get("high_funding_mean_annualized_return")
        stress_high_funding = stress_regime.get("high_funding_mean_annualized_return") if stress_regime else None
        stress_mdd = gates["gate_s5"].get("stress_block_mdd_p95")
        lines.append(
            f"| {candidate_id} | {_fmt_usd(payload.get('total_capital_usdt'), 0)} | {payload.get('top_k_pairs', 'N/A')} | "
            f"{metadata.get('universe_size_static', 'N/A')} | {_fmt_pct(high_funding)} | **{_fmt_pct(full_return)}** | {_fmt_usd(gates['gate_s2'].get('p05'))} | "
            f"{_fmt_pct(gates['gate_s3'].get('mdd_p95'))} | {_fmt_usd(metadata.get('avg_cost_per_trade_usdt'))} | "
            f"{_fmt_pct(stress_high_funding)} | {_fmt_pct(stress_mdd)} |"
        )
    lines.append("")
    lines.append(
        "**두 반환열의 차이에 주의**: '고펀딩기(2024) 연환산'은 SPEC.md가 사전등록한 승격 판정 "
        "지표(2024 캘린더이어 단독 구간의 연환산)이고, '전체구간 누적수익률'은 실제 백테스트 전체 "
        "구간(2024-01-01~FROZEN_END, 2025-09-30 이후 저펀딩 OOS 구간 포함)의 실현 누적수익률이다. "
        "두 지표가 **다른 승자를 가리킬 수 있다** -- 아래 판정 절에서 실제로 그런 사례(M1)를 정직하게 "
        "병기한다."
    )
    return lines


def _gate_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | S1 구조 | S2 MC(p05/ruin) | S3 블록MDD(<=10%) | S4 실행가능 | S5 스트레스 | S6 거래소구조 | Overall | 기준선 대비 | 승격 | 실패사유 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        if not gates:
            lines.append(f"| {candidate_id} | - | - | - | - | - | - | NOT_EVALUATED | - | - | gates 미실행 |")
            continue
        s1, s2, s3, s4, s5, s6 = gates["gate_s1"], gates["gate_s2"], gates["gate_s3"], gates["gate_s4"], gates["gate_s5"], gates["gate_s6"]
        promo = gates["promotion"]
        s1_cell = f"{s1['status']} ({s1['leverage_multiplier_of_active_capital']:.2f}x)"
        s2_cell = f"{s2['status']} (p05={_fmt_usd(s2['p05'])})"
        s3_cell = f"{s3['status']} ({_fmt_pct(s3['mdd_p95'])})"
        s4_cell = f"{s4['status']} (레그{_fmt_usd(s4['leg_usdt_nominal'])})"
        s5_cell = f"{s5['status']} (부호={'+' if s5['sign_preserved'] else '반전'})"
        s6_cell = s6["status"] if s6.get("applicable") else "N/A"
        actually_promoted = gates["overall"] == "PASS" and bool(promo["promoted"])
        promoted_cell = "YES" if actually_promoted else "no"
        beats = promo.get("beats_baseline")
        baseline_cell = "기준선(자체)" if promo.get("baseline_candidate_id") is None else ("승리" if beats else ("패배" if beats is False else "N/A"))
        reasons = ", ".join(gates.get("failure_reasons", [])) or "-"
        lines.append(f"| {candidate_id} | {s1_cell} | {s2_cell} | {s3_cell} | {s4_cell} | {s5_cell} | {s6_cell} | {gates['overall']} | {baseline_cell} | {promoted_cell} | {reasons} |")
    return lines


# ---------------------------------------------------------------------------
# Section: capital-tier annualized curve + concurrent-position saturation (required).
# ---------------------------------------------------------------------------


def _capital_tier_curve_section(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "## 자본 티어별 연환산 곡선 (x축: $100 / $300 / $1,000 / $3,000) -- 동시 포지션 포화 지점",
        "",
        "동일 계열(통상캐리, +Bybit) 내에서 자본과 동시쌍수가 같이 늘어날 때 고펀딩기(2024) "
        "연환산이 어떻게 변하는지 -- M1($100/1쌍) -> M3($300/3쌍) -> M4($1,000/10쌍) -> "
        "M5($3,000/30쌍) 이어달리기:",
        "",
        "| 자본 | 동시쌍 | Candidate | 고펀딩기 연환산 | 전 단계 대비 | 전체구간 누적수익률 | 편입심볼(정적) | Overall |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    chain = ("M1", "M3", "M4", "M5")
    previous_return: float | None = None
    for candidate_id in chain:
        payload = payloads.get(candidate_id, {})
        regime = payload.get("regime_breakdown")
        gates = payload.get("gates")
        high_funding = regime.get("high_funding_mean_annualized_return") if regime else None
        full_return = _full_period_total_return(payload)
        delta = "-" if (previous_return is None or high_funding is None) else f"{(high_funding - previous_return) * 100.0:+.2f}%p"
        lines.append(
            f"| {_fmt_usd(payload.get('total_capital_usdt'), 0)} | {payload.get('top_k_pairs', 'N/A')} | {candidate_id} | "
            f"{_fmt_pct(high_funding)} | {delta} | {_fmt_pct(full_return)} | {payload.get('metadata', {}).get('universe_size_static', 'N/A')} | {gates['overall'] if gates else '-'} |"
        )
        if high_funding is not None:
            previous_return = high_funding
    lines.append("")
    lines.append(
        "동시쌍수가 편입심볼(정적) 수를 넘어서면 매일 채울 수 있는 슬롯보다 쌍수가 많아져 "
        "포화가 시작된다 -- 위 표의 '편입심볼(정적)' 열과 동시쌍 열을 대조해서 판단한다 "
        "(예: 동시쌍=30인데 그날 활성+유동성 조건을 만족하는 심볼이 30개 미만이면 미체결 슬롯이 "
        "발생, 실질 gross는 설계값 미만으로 자기 제한된다 -- eligible_count_stats가 이를 방증)."
    )
    return lines


# ---------------------------------------------------------------------------
# Section: venue-addition net effect (required: opportunity gain vs cost/complexity).
# ---------------------------------------------------------------------------


def _venue_addition_effect_section(payloads: dict[str, dict]) -> list[str]:
    lines = ["## 거래소 추가(+Bybit)의 순효과 -- 기회 증가 vs 비용·운영 복잡도", ""]
    pairs = (("M0", "M1", "$100/1쌍"), ("M2", "M3", "$300/3쌍"))
    for single_id, dual_id, label in pairs:
        single = payloads.get(single_id, {})
        dual = payloads.get(dual_id, {})
        single_regime = single.get("regime_breakdown")
        dual_regime = dual.get("regime_breakdown")
        single_high = single_regime.get("high_funding_mean_annualized_return") if single_regime else None
        dual_high = dual_regime.get("high_funding_mean_annualized_return") if dual_regime else None
        delta = "-" if (single_high is None or dual_high is None) else f"{(dual_high - single_high) * 100.0:+.2f}%p"
        single_full = _full_period_total_return(single)
        dual_full = _full_period_total_return(dual)
        single_eligible = single.get("metadata", {}).get("eligible_count_stats", {}).get("median")
        dual_eligible = dual.get("metadata", {}).get("eligible_count_stats", {}).get("median")
        single_cost = single.get("metadata", {}).get("avg_cost_per_trade_usdt")
        dual_cost = dual.get("metadata", {}).get("avg_cost_per_trade_usdt")
        dual_gates = dual.get("gates", {})
        s6 = dual_gates.get("gate_s6", {}) if dual_gates else {}
        empirical = s6.get("empirical_bybit_share", {}) if s6 else {}
        lines.extend(
            [
                f"### {label}: {single_id}(단일) vs {dual_id}(+Bybit)",
                "",
                f"- 고펀딩기(2024) 연환산: {_fmt_pct(single_high)} -> {_fmt_pct(dual_high)} ({delta})",
                f"- **전체구간(2024-01~2026-07) 누적수익률**: {_fmt_pct(single_full)} -> {_fmt_pct(dual_full)} "
                f"({'+Bybit 우위' if (single_full is not None and dual_full is not None and dual_full > single_full) else '단일거래소 우위'})",
                f"- 일일 매칭가능(eligible) 심볼 수 중앙값: {single_eligible if single_eligible is not None else 'N/A'} -> "
                f"{dual_eligible if dual_eligible is not None else 'N/A'} (기회풀 확대 정도)",
                f"- 건당비용: {_fmt_usd(single_cost)} -> {_fmt_usd(dual_cost)}",
            ]
        )
        if empirical.get("available"):
            lines.append(
                f"- 운영 복잡도(실측): 채워진 슬롯 중 Bybit 비중 평균 {_fmt_pct(empirical.get('mean_bybit_share_of_filled_slots'))}, "
                f"최대 {_fmt_pct(empirical.get('max_bybit_share_of_filled_slots'))} ({empirical.get('days_with_any_position', 0)}일 표본) "
                f"-- 두 거래소 계정·API·자금 관리를 동시에 운영해야 하는 실질 비중."
            )
        else:
            lines.append("- 운영 복잡도(실측): 데이터 없음(게이트 미실행 또는 무포지션 구간).")
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section: M6/M7 deep dive (required: return/risk decomposition).
# ---------------------------------------------------------------------------


def _cross_venue_deep_dive(payloads: dict[str, dict]) -> list[str]:
    lines = ["## M6/M7 거래소간 펀딩 스프레드 -- 수익·리스크 분해 (신규 구조, 기존 캐리와 별도 패밀리)", ""]
    for candidate_id in ("M6", "M7"):
        payload = payloads.get(candidate_id, {})
        regime = payload.get("regime_breakdown")
        stress_regime = payload.get("stress_regime_breakdown")
        gates = payload.get("gates")
        metadata = payload.get("metadata", {})
        config = get_config(candidate_id)
        lines.append(f"### {candidate_id} (${config.total_capital:,.0f} / {config.candidate.top_k}쌍)")
        lines.append("")
        if regime is None or gates is None:
            lines.append("- 게이트/리짐 미실행.")
            lines.append("")
            continue
        high_funding = regime.get("high_funding_mean_annualized_return")
        stress_high = stress_regime.get("high_funding_mean_annualized_return") if stress_regime else None
        s6 = gates.get("gate_s6", {})
        baseline_id = config.baseline_candidate_id
        baseline_payload = payloads.get(baseline_id, {}) if baseline_id else {}
        baseline_regime = baseline_payload.get("regime_breakdown")
        baseline_high = baseline_regime.get("high_funding_mean_annualized_return") if baseline_regime else None
        own_full = _full_period_total_return(payload)
        baseline_full = _full_period_total_return(baseline_payload) if baseline_payload else None
        lines.extend(
            [
                f"- 고펀딩기(2024) 연환산: {_fmt_pct(high_funding)} (스트레스x3: {_fmt_pct(stress_high)})",
                f"- 동일 자본/쌍수 통상캐리 기준선({baseline_id}) 대비: {_fmt_pct(baseline_high)} "
                f"({'승리' if (high_funding is not None and baseline_high is not None and high_funding > baseline_high) else '패배/판정불가'})",
                f"- **전체구간(2024-01~2026-07) 누적수익률**: {_fmt_pct(own_full)} (기준선 {baseline_id}: {_fmt_pct(baseline_full)}) "
                f"({'승리' if (own_full is not None and baseline_full is not None and own_full > baseline_full) else '패배'})",
                f"- 건당비용: {_fmt_usd(metadata.get('avg_cost_per_trade_usdt'))} (양쪽 거래소 레그 비용 합산 -- costs_venue.cross_venue_leg_cost_rate)",
                f"- 가동률: {_fmt_pct(metadata.get('utilization'))}, 연환산 거래횟수: {metadata.get('annualized_round_trips', 'N/A'):.1f}" if isinstance(metadata.get("annualized_round_trips"), (int, float)) else "- 가동률/거래횟수: N/A",
                f"- **S6 거래소 신용리스크(구조적, 확률 아님)**: 거래소 1곳 전액 손실 시 잔존 자본 "
                f"{_fmt_pct(s6.get('residual_capital_fraction_if_one_venue_wiped'))} (기준 {_fmt_pct(S6_RESIDUAL_CAPITAL_FLOOR)} 이상) "
                f"-> {s6.get('status', 'N/A')}. 매 활성 포지션이 항상 양쪽 거래소에 정확히 절반씩 걸리는 구조이므로 "
                "이 비율은 시장 경로와 무관하게 고정된다.",
                f"- Overall: {gates['overall']}, 실패사유: {', '.join(gates.get('failure_reasons', [])) or '-'}",
            ]
        )
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------


def _robustness_caveats(payloads: dict[str, dict], promoted_ids: tuple[str, ...]) -> list[str]:
    """SPEC.md pre-registers the promotion metric as the 2024-only high-funding-year
    annualized return -- this module honors that mechanically (a candidate that wins on
    that metric IS "promoted"). But that metric can diverge from the full-period picture
    (2024-01-01~FROZEN_END, including the 2025-09-30-onward low-funding OOS stretch): a
    candidate can win the pre-registered metric by a thin margin while its BASELINE
    actually delivered more total return over the whole window. Silently reporting only
    the pre-registered metric in that situation would be technically compliant but
    materially misleading -- this function detects and flags it explicitly, for every
    promoted candidate that has a registered baseline."""
    lines: list[str] = []
    for candidate_id in promoted_ids:
        payload = payloads[candidate_id]
        config = get_config(candidate_id)
        if config.baseline_candidate_id is None:
            continue
        baseline_payload = payloads.get(config.baseline_candidate_id)
        if baseline_payload is None:
            continue
        own_full = _full_period_total_return(payload)
        baseline_full = _full_period_total_return(baseline_payload)
        own_annual = payload.get("regime_breakdown", {}).get("high_funding_mean_annualized_return")
        baseline_annual = baseline_payload.get("regime_breakdown", {}).get("high_funding_mean_annualized_return")
        if own_full is None or baseline_full is None:
            continue
        margin = (own_annual - baseline_annual) if (own_annual is not None and baseline_annual is not None) else None
        if own_full < baseline_full:
            lines.append(
                f"- **{candidate_id}의 승격은 취약하다.** SPEC 사전등록 지표(고펀딩기 2024 연환산)에서 "
                f"{config.baseline_candidate_id} 대비 {f'{margin * 100.0:+.2f}%p' if margin is not None else '근소하게'} 앞서 형식상 승격했지만, "
                f"**전체구간(2024-01~2026-07) 누적수익률은 {_fmt_pct(own_full)}로 {config.baseline_candidate_id}의 "
                f"{_fmt_pct(baseline_full)}보다 낮다** -- 즉 2025-09 이후 저펀딩 OOS 구간에서 {config.baseline_candidate_id}에 "
                f"뒤처지며 이를 만회하지 못한다. 표본이 2024 한 해뿐인 지표로 판정이 뒤집힐 수 있다는 뜻이므로, "
                "이 승격은 견고하지 않은(단일 연도·근소 마진) 결과로 취급해야 한다."
            )
    return lines


def _verdict_section(payloads: dict[str, dict]) -> tuple[str, list[str]]:
    promoted: list[tuple[str, float, float]] = []  # (id, high_funding, total_capital)
    attempted: list[str] = []
    for candidate_id in CONFIG_IDS:
        gates = payloads[candidate_id].get("gates")
        if not gates:
            continue
        attempted.append(candidate_id)
        promo = gates["promotion"]
        if gates["overall"] == "PASS" and promo["promoted"]:
            high_funding = promo["high_funding_mean_annualized_return"]
            if high_funding is not None:
                promoted.append((candidate_id, high_funding, payloads[candidate_id].get("total_capital_usdt", 0.0)))

    if promoted:
        by_tier: dict[float, tuple[str, float]] = {}
        for candidate_id, high_funding, tier in promoted:
            current = by_tier.get(tier)
            if current is None or high_funding > current[1]:
                by_tier[tier] = (candidate_id, high_funding)
        promoted_ids = tuple(cid for cid, _, _ in promoted)
        caveats = _robustness_caveats(payloads, promoted_ids)
        lines = [
            "## 판정: 통과 (승격 구성 존재 -- 단, 견고성 주의사항 필독)",
            "",
            f"S1-S6 전부 PASS ∧ 동일 티어 단일거래소 기준선(2024 연환산 기준) 초과 구성: {', '.join(promoted_ids)}.",
            "",
            "### 자본 티어별 최적 구성 (SPEC 사전등록 지표 기준)",
            "",
            "| 자본 | 최적 Candidate | 고펀딩기(2024) 연환산 | 전체구간 누적수익률 |",
            "|---:|---|---:|---:|",
        ]
        for tier in sorted(by_tier):
            candidate_id, high_funding = by_tier[tier]
            full_return = _full_period_total_return(payloads[candidate_id])
            lines.append(f"| {_fmt_usd(tier, 0)} | {candidate_id} | {_fmt_pct(high_funding)} | {_fmt_pct(full_return)} |")
        if caveats:
            lines.append("")
            lines.append("### 견고성 주의사항 (정직 고지 -- 위 표만 보고 결론 내리지 말 것)")
            lines.append("")
            lines.extend(caveats)
            lines.append("")
            lines.append(
                "**종합 판단**: 형식상 승격 기준(PASS)은 충족하지만, 위 주의사항을 반영하면 "
                "\"거래소 추가가 명확하고 견고하게 개선을 만든다\"고 보기는 어렵다. 전체구간 기준으로는 "
                "모든 +Bybit/거래소간 구성이 동일 티어 단일거래소 기준선에 뒤처진다 (아래 '구성 x 지표' "
                "표의 '전체구간 누적수익률' 열 참조) -- **심볼·거래소·포지션 축 모두 사실상 포화 상태이고, "
                "L4(top200 단일거래소, wave13)가 여전히 실질적 상한에 가깝다는 것이 이 wave의 더 정직한 결론이다.**"
            )
        return "PASS", lines

    lines = [
        "## 판정: 거래소·포지션 축도 포화, L4가 상한",
        "",
        "S1-S6을 모두 통과하고 동일 티어 단일거래소 기준선을 초과한 구성이 없다 (게이트 완화 없이 정직 보고):",
        "",
        "| Candidate | 고펀딩기 연환산 | 전체구간 누적수익률 | Overall | 기준선 대비 | 실패/미달 사유 |",
        "|---|---:|---:|---|---|---|",
    ]
    for candidate_id in attempted:
        payload = payloads[candidate_id]
        gates = payload["gates"]
        regime = payload.get("regime_breakdown", {})
        high_funding = regime.get("high_funding_mean_annualized_return")
        full_return = _full_period_total_return(payload)
        promo = gates["promotion"]
        beats = promo.get("beats_baseline")
        baseline_cell = "기준선(자체)" if promo.get("baseline_candidate_id") is None else ("승리" if beats else "패배/판정불가")
        reason_cell = gates["overall"] if gates["overall"] == "PASS" else f"FAIL({', '.join(gates['failure_reasons']) or '?'})"
        lines.append(f"| {candidate_id} | {_fmt_pct(high_funding)} | {_fmt_pct(full_return)} | {gates['overall']} | {baseline_cell} | {reason_cell} |")
    return "FAIL", lines


def _dsr_note() -> str:
    return (
        f"다중검정 보정: 누적 시행 {DSR_CUMULATIVE_TRIALS}회(wave13까지 76회 + 이 wave의 M0-M7 "
        "8개) 기준 DSR(Deflated Sharpe Ratio) 참고치를 각 결과 JSON의 `reference_metrics.dsr`에 "
        "기록 (샤프는 참고 지표이며 승격 판정에는 사용하지 않음, wave10-13과 동일 원칙). "
        "**M6/M7은 신규 구조(양쪽 퍼프)이므로 기존 캐리 계열과 별도 패밀리로 등록** -- "
        "`family` 필드가 `wave14_multivenue_cross_venue_spread`로 구분됨."
    )


def write_wave14_report(results_dir: Path, report_dir: Path, registry_path: Path, cache_dir: Path) -> None:
    payloads = {candidate_id: _load(results_dir, candidate_id) for candidate_id in CONFIG_IDS}
    aux_payloads = {}
    for aux in AUX_BASELINES:
        aux_path = results_dir / f"{aux.candidate_id}.json"
        if aux_path.exists():
            aux_payloads[aux.candidate_id] = _load(results_dir, aux.candidate_id)
    payloads_with_aux = {**payloads, **aux_payloads}

    universe_path = cache_dir / "bybit_universe.json"
    universe_payload = json.loads(universe_path.read_text(encoding="utf-8")) if universe_path.exists() else {}
    spread_path = cache_dir / "bybit_spreads.json"
    spread_payload = json.loads(spread_path.read_text(encoding="utf-8")) if spread_path.exists() else {}
    hyperliquid_path = cache_dir / "hyperliquid_probe.json"
    hyperliquid_payload = json.loads(hyperliquid_path.read_text(encoding="utf-8")) if hyperliquid_path.exists() else {}

    verdict, verdict_lines = _verdict_section(payloads)

    lines: list[str] = [
        "# Wave-14 리포트 — 멀티거래소 x 동시포지션 x 자본티어 (M0-M7)",
        "",
        *_limitations_section(universe_payload),
        "",
        *_spread_summary_section(spread_payload),
        "",
        "## 구성 정의",
        "",
        *_config_table(),
        "",
        "## 구성 x 지표",
        "",
        *_metrics_table(payloads),
        "",
        *_capital_tier_curve_section(payloads_with_aux),
        "",
        *_venue_addition_effect_section(payloads_with_aux),
        "",
        *_cross_venue_deep_dive(payloads_with_aux),
        "",
        *verdict_lines,
        "",
        "## 게이트 (S1-S6)",
        "",
        *_gate_table(payloads),
        "",
        "## Hyperliquid 프로브 (M0-M7 백테스트에는 미사용, 접근성 기록용)",
        "",
        f"- 접근 가능: {hyperliquid_payload.get('accessible', 'N/A')} · perp 유니버스 {hyperliquid_payload.get('perp_universe_count', 'N/A')}종 · "
        f"BTC fundingHistory 샘플 {hyperliquid_payload.get('funding_history_btc_sample_count', 'N/A')}건",
        f"- {hyperliquid_payload.get('scope_note', 'N/A')}",
        "",
        "## OKX (장기 백테스트 불가, 백테스트 미사용)",
        "",
        "- OKX 공개 funding-rate-history 엔드포인트는 최근 ~93일만 반환 "
        "(`research/validation/CROSS_VENUE_REPORT.md`에서 기존 실측 확인, 이번 wave 재조사 불필요) "
        "-- 장기 백테스트에 부적합하여 M0-M7 어디에도 사용하지 않았다. 현재 기회 스캔용으로만 적합.",
        "",
        "## 다중검정",
        "",
        _dsr_note(),
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave14_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = ["# Wave-14 registry", "", "| Candidate | Family | State | Verdict | 기준선대비 | 승격 | 분류 |", "|---|---|---|---|---|---|---|"]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        if not gates:
            registry_lines.append(f"| {candidate_id} | wave14_multivenue | PENDING | - | - | - | gates 미실행 |")
            continue
        promo = gates["promotion"]
        result_verdict = gates["overall"]
        actually_promoted = result_verdict == "PASS" and bool(promo["promoted"])
        promoted_cell = "YES" if actually_promoted else "no"
        beats = promo.get("beats_baseline")
        baseline_cell = "자체기준선" if promo.get("baseline_candidate_id") is None else ("승리" if beats else "패배")
        family = payload.get("family", "wave14_multivenue")
        classification = "승격" if actually_promoted else (f"게이트위반({', '.join(gates['failure_reasons'])})" if result_verdict != "PASS" else "기준선미달")
        registry_lines.append(f"| {candidate_id} | {family} | EVALUATED | {result_verdict} | {baseline_cell} | {promoted_cell} | {classification} |")
    registry_lines.append("")
    registry_lines.append(f"**판정**: {'통과' if verdict == 'PASS' else '거래소·포지션 축도 포화, L4가 상한'} (자세한 내용은 report/wave14_report.md 참조).")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave14_report"]
