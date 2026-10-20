# Wave-10 markdown report + registry writer. Pure formatting over already-computed
# results/C{1..4}.json payloads written by run_wave10.py --stage run/gates.
# Mirrors research/wave9_100usd/reporting_w9.py's structure.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.wave10_carry100.configs import CONFIG_IDS, get_config
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, TOTAL_CAPITAL
from research.wave10_carry100.regime import HIGH_FUNDING_YEARS


def _load(results_dir: Path, candidate_id: str) -> dict[str, Any]:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100.0:.2f}%"


def _fmt_usd(value: float | None) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def _config_table() -> list[str]:
    lines = [
        "| Config | 쌍 수 | 레그당 비중 | 레그 $ (@ $90 활성자본) | gross $ | gross 배수 | 진입 임계 | 정의 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate_id in CONFIG_IDS:
        config = get_config(candidate_id)
        leg = config.leg_fraction * ACTIVE_CAPITAL
        gross = 2.0 * config.candidate.top_k * config.leg_fraction * ACTIVE_CAPITAL
        lines.append(
            f"| {candidate_id} | {config.candidate.top_k} | {config.leg_fraction:.0%} | {_fmt_usd(leg)} | "
            f"{_fmt_usd(gross)} | {gross / ACTIVE_CAPITAL:.2f}x | {config.candidate.threshold_apr:.0%} APR | "
            f"{config.note} |"
        )
    return lines


def _gate_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Config | A 실행가능성 | B MC(p05/ruin) | C 블록MDD p95 | D 전기간수익 | E OOS(휴면기) | Overall | 실패/라벨 사유 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        if not gates:
            lines.append(f"| {candidate_id} | - | - | - | - | - | NOT_EVALUATED | run --stage gates 필요 |")
            continue
        a, b, c, d, e = gates["gate_a"], gates["gate_b"], gates["gate_c"], gates["gate_d"], gates["gate_e"]
        a_cell = f"{a['status']} (레그{_fmt_usd(a['leg_usdt_nominal'])}, gross{a['gross_multiplier_of_active_capital']:.2f}x)"
        b_cell = f"{b['status']} (p05={_fmt_usd(b['p05'])}, ruin={_fmt_pct(b['ruin_probability'])})"
        c_cell = f"{c['status']} ({_fmt_pct(c['mdd_p95'])})"
        d_cell = f"{d['status']} ({_fmt_pct(d['total_return'])})"
        if e["status"] == "UNTESTED_IN_OOS":
            e_cell = "UNTESTED_IN_OOS (무포지션)"
        else:
            e_cell = f"{e['status']} (return={_fmt_pct(e['oos_return'])}, trades={e.get('oos_trade_count', 0)})"
        reasons = ", ".join(gates.get("failure_reasons", [])) or "-"
        lines.append(f"| {candidate_id} | {a_cell} | {b_cell} | {c_cell} | {d_cell} | {e_cell} | **{gates['overall']}** | {reasons} |")
    return lines


def _regime_table(payloads: dict[str, dict]) -> list[str]:
    year_headers = " | ".join(f"{year} 연환산" for year in HIGH_FUNDING_YEARS)
    lines = [
        f"| Config | {year_headers} | 고펀딩기 평균(연환산) | 고펀딩기 평균 $100기준 연이익 | 현재(저펀딩 OOS) 연환산 | 현재 $100기준 연이익 |",
        "|---|" + "---:|" * (len(HIGH_FUNDING_YEARS) + 4),
    ]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        regime = payload.get("regime_breakdown")
        if not regime:
            lines.append(f"| {candidate_id} | " + " | ".join(["-"] * (len(HIGH_FUNDING_YEARS) + 4)) + " |")
            continue
        year_cells = []
        for year in HIGH_FUNDING_YEARS:
            entry = regime["high_funding_years"].get(str(year))
            year_cells.append(_fmt_pct(entry["annualized_return"]) if entry else "no data")
        mean_ann = regime.get("high_funding_mean_annualized_return")
        mean_profit = regime.get("high_funding_mean_annual_profit_usdt_at_100_basis")
        current = regime.get("current_low_funding")
        current_ann = current["annualized_return"] if current else None
        current_profit = current["active_capital_annual_profit_usdt"] if current else None
        lines.append(
            f"| {candidate_id} | " + " | ".join(year_cells) + f" | {_fmt_pct(mean_ann)} | {_fmt_usd(mean_profit)} | "
            f"{_fmt_pct(current_ann)} | {_fmt_usd(current_profit)} |"
        )
    return lines


def _passing_candidates(payloads: dict[str, dict]) -> list[str]:
    return [cid for cid in CONFIG_IDS if payloads[cid].get("gates", {}).get("overall") == "PASS"]


def _untested_candidates(payloads: dict[str, dict]) -> list[str]:
    return [cid for cid in CONFIG_IDS if payloads[cid].get("gates", {}).get("overall") == "UNTESTED_IN_OOS"]


def _failed_candidates(payloads: dict[str, dict]) -> list[tuple[str, str]]:
    out = []
    for cid in CONFIG_IDS:
        gates = payloads[cid].get("gates", {})
        if gates.get("overall") == "FAIL":
            reasons = ", ".join(gates.get("failure_reasons", [])) or "미분류"
            out.append((cid, reasons))
    return out


def _verdict_section(payloads: dict[str, dict]) -> list[str]:
    passing = _passing_candidates(payloads)
    untested = _untested_candidates(payloads)
    failed = _failed_candidates(payloads)
    lines = ["## 판정", ""]
    lines.append(f"- 전체 게이트(A-D) 통과: **{len(passing) + len(untested)}/4** (그중 OOS까지 실측 확인된 완전 PASS: **{len(passing)}/4**)")
    lines.append(f"- PASS: {', '.join(passing) if passing else '없음'}")
    lines.append(f"- UNTESTED_IN_OOS (A-D 통과, 휴면기 무포지션이라 OOS 실측 불가): {', '.join(untested) if untested else '없음'}")
    lines.append(f"- FAIL: {', '.join(cid for cid, _ in failed) if failed else '없음'}")
    lines.append("")
    if failed:
        lines.append("실패 원인 분류 (gross / 최소주문 / 수익부족 / 휴면):")
        lines.append("")
        lines.append("| Config | 실패 사유 |")
        lines.append("|---|---|")
        for cid, reasons in failed:
            lines.append(f"| {cid} | {reasons} |")
        lines.append("")
    lines.append(
        "- wave-8 판정(4쌍 풀사이즈, gross $180 > 활성자본 $90 -> 전량 실행 불가)과 달리, "
        "wave-10의 4개 구성은 모두 사전 설계 단계에서 gross <= 활성자본이 되도록 사이징했다 "
        "(게이트 A는 사실상 설계로 보장됨 -- 진짜 판정은 B/C/D/E에서 갈린다)."
    )
    lines.append("")
    return lines


def _c4_caveat_section() -> list[str]:
    return [
        "## C4 스펙 불일치 참고사항 (정직성 고지, 수치 보정 없음)",
        "",
        "원 지시문: \"C4: 1쌍, 레그당 45% + 진입임계 완화 25%APR(가동률↑ 시도)\". W2c/기본 임계값은 "
        "15% APR이며, 이 코드베이스의 기존 용례(W2b: W2a의 8%APR -> 5%APR을 \"가동률↑\"로 명시, "
        "research/wave2/SPEC.md)에서 \"완화\"는 **더 낮은** 임계값을 의미한다. 25%APR은 15%APR보다 "
        "**높으므로** 기계적으로 진입 조건이 더 까다로워져 가동률이 낮아진다 -- 지시문의 수치(25%)와 "
        "의도(가동률↑) 문구가 서로 모순된다. 사전등록 수치를 사후에 임의로 고치는 것 자체가 이 "
        "wave의 금지 사항이므로, **지시된 수치(25% APR)를 문자 그대로 구현**하고 그 결과(대개 "
        "가동률이 W2c 기준선보다 낮아짐)를 정직하게 아래 표에 반영했다. 의도대로 \"가동률을 높이는\" "
        "실험이 필요하다면 임계값을 15%APR보다 낮춰(예: 10%APR 이하) 재등록해야 한다.",
        "",
    ]


def write_wave10_report(results_dir: Path, report_dir: Path, registry_path: Path) -> None:
    payloads = {candidate_id: _load(results_dir, candidate_id) for candidate_id in CONFIG_IDS}
    report_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Wave-10 리포트 -- $100 자본 반노셔널 캐리 ($90 활성자본)",
        "",
        "사전등록: research/wave10_carry100/SPEC.md. 배경: wave-8은 W2c(델타중립 숏퍼프+롱현물, "
        f"7d 펀딩 APR>15% 진입/7.5% 청산)를 4쌍 풀사이즈(top4 균등분할, 레그당 최대 100%)로 "
        "판정했고 gross notional이 활성자본의 2배($180 > $90)라 $100에서 실행 불가라고 결론지었다. "
        "wave-10은 신호·비용·체결 규약은 그대로 두고 **사이징(레그당 고정 비중)과 동시 쌍 수만** "
        f"바꿔, gross를 활성자본(${ACTIVE_CAPITAL:.0f}) 이하로 강제한 4개 구성이 나머지 강건성 "
        "게이트(MC/블록셔플/전기간수익/OOS)까지 통과하는지 검증한다.",
        "",
        f"자본 규약: 총자본 ${TOTAL_CAPITAL:.0f} / 현금버퍼 10% / 활성자본 ${ACTIVE_CAPITAL:.0f} / "
        f"최소주문 ${MIN_ORDER_USDT:.2f}. 비용: 메이커 0.02%/레그 + 슬리피지(메이저 1bp/알트 3bp), "
        "펀딩 8h 실적립, 신호는 t종가 확정 -> t+1시가 체결 (룩어헤드 없음). 신호·유니버스·체결 로직은 "
        "research.wave1.fam_funding (funding_score/carry_position/load_markets) 임포트 재사용, "
        "비용 상수는 research.wave2.funding.W2_MAKER_FEE_RATE + research.wave1.costs.slippage_rate "
        "임포트 재사용.",
        "",
        "## 구성 (사전등록 4개, 사후 추가 없음)",
        "",
        *_config_table(),
        "",
        *_c4_caveat_section(),
        "## 게이트 결과 (A 실행가능성 / B MC 1e4부트스트랩 / C 블록셔플90일MDD / D 전기간비용후수익 / E 휴면기OOS)",
        "",
        *_gate_table(payloads),
        "",
        "## 펀딩 레짐별 $100 기준 연환산 기대수익률",
        "",
        "고펀딩기 = 2020/2021/2024 (실측 이력 슬라이스, W2c 자체 연도별 수익도 이 3개 연도가 최고치). "
        "저펀딩기(현재) = OOS 2025-10-01~데이터 종료(2026-07-14) (실측). \"$100기준 연이익\" = "
        f"활성자본(${ACTIVE_CAPITAL:.0f}) x 해당 구간 연환산수익률 (현금버퍼 $10은 무이자 대기 가정, "
        "보수적).",
        "",
        *_regime_table(payloads),
        "",
        *_verdict_section(payloads),
        "## 모델링 노트",
        "",
        "- 사이징 규칙 변경만 허용: 신호(funding_score/carry_position), 유니버스(wave-1 40심볼 "
        "eligible universe), 체결 타이밍(t종가 신호 -> t+1시가 체결)은 W2c와 동일 임포트. 유일한 "
        "차이는 랭킹 상위 K(top_k=쌍 수)까지의 각 심볼에 1/len(ranked) 대신 **고정 레그 비중**을 "
        "배정하는 것 (research/wave10_carry100/engine.py의 run_fixed_fraction_portfolio).",
        "- 델타중립: 스팟 롱과 퍼프 숏이 동일한 weight 값 하나로 함께 구동되므로(intraday = "
        "spot_ret - perp_ret + funding, weights 곱) 구조적으로 델타중립이 보장된다. "
        "tests/test_wave10_engine.py가 가격이 움직이되 베이시스가 0인 합성 시장으로 이를 회귀 검증한다.",
        "- MC/블록셔플 방법론은 research/wave8_capital/run_capital100.py의 _simulate_mc/_block_shuffle과 "
        "동일 (경로 수, $100/$90/$10 기준, 90일 블록) -- wave-8 판정과 비교 가능하도록 유지.",
        "- 이 리포트는 paper 백테스트 결과이며 실계좌 주문을 실행하지 않는다. 메이커 체결률은 실측이 "
        "아닌 가정이다 (W2c와 동일한 한계, research/wave2/SPEC.md).",
        "",
    ]
    (report_dir / "wave10_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = ["# Wave-10 registry", "", "| Candidate | Family | State | Verdict | 실패/라벨 사유 |", "|---|---|---|---|---|"]
    for candidate_id in CONFIG_IDS:
        payload = payloads[candidate_id]
        gates = payload.get("gates")
        state = "EVALUATED" if gates else "RUN_ONLY"
        verdict = gates["overall"] if gates else "UNTESTED"
        reasons = ", ".join(gates.get("failure_reasons", [])) if gates else "run --stage gates 필요"
        registry_lines.append(f"| {candidate_id} | wave10_carry100 | {state} | {verdict} | {reasons or '-'} |")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave10_report"]
