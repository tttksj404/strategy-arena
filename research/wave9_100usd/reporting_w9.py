# Wave-9 markdown report + registry writer. Pure formatting over already-computed
# results/W9{a..f}.json payloads (written by the run/gates CLI stages) -- no
# additional computation happens here, mirroring research/wave7/reporting_w7.py.

from __future__ import annotations

import json
from pathlib import Path

from research.wave9_100usd.engine_w9 import W9_CANDIDATE_IDS, W9_CANDIDATES


CANDIDATE_DEFINITIONS = {config.candidate_id: config.definition for config in W9_CANDIDATES}
CANDIDATE_CONFIGS = {config.candidate_id: config for config in W9_CANDIDATES}


def _load(results_dir: Path, candidate_id: str) -> dict:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


def _fmt4(value: float) -> str:
    return f"{value:.4f}"


def _total_return(payload: dict) -> float:
    equity = payload.get("equity") or []
    if len(equity) < 2:
        return 0.0
    return float(equity[-1]["value"]) / float(equity[0]["value"]) - 1.0


def _overview_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | 정의 | Mode | Lev | Hold | Trades | 최종자본 | 총수익률 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate_id in W9_CANDIDATE_IDS:
        payload = payloads[candidate_id]
        meta = payload.get("metadata", {})
        cfg = payload.get("candidate_config", {})
        lines.append(
            f"| {candidate_id} | {payload.get('definition', CANDIDATE_DEFINITIONS[candidate_id])} "
            f"| {cfg.get('mode', '-')} | {cfg.get('leverage', 0):g}x | {cfg.get('hold_days', 0)}d "
            f"| {meta.get('trades_executed', 0)} | {_fmt_usd(meta.get('final_equity', 0.0))} "
            f"| {_fmt_pct(_total_return(payload))} |"
        )
    return lines


def _deep_validation_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | MC 중앙값 | MC p05 | P(<$30) | Block MDD p95 | OOS 수익 | Sharpe(참고) | Gates | Overall | Hard(H1/H2/H4) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for candidate_id in W9_CANDIDATE_IDS:
        validation = payloads[candidate_id].get("validation")
        if not validation:
            lines.append(f"| {candidate_id} | - | - | - | - | - | - | - | NOT_EVALUATED (run --stage gates) | - |")
            continue
        mc = validation["mc_bootstrap"]
        blocks = validation.get("block_shuffle") or {}
        gates = {gate["gate_id"]: gate for gate in validation["gates"]}
        overall = validation["overall"]
        ref = validation["reference_metrics"]
        oos_detail = gates.get("H3", {}).get("detail", "-")
        mdd_p95 = blocks.get("mdd_p95")
        lines.append(
            f"| {candidate_id} | {_fmt_usd(mc['median'])} | {_fmt_usd(mc['p05'])} "
            f"| {_fmt_pct(mc['p_bankrupt'])} | {_fmt_pct(mdd_p95) if mdd_p95 is not None else 'N/A'} "
            f"| {oos_detail} | {_fmt4(ref['sharpe_trade_level'])} "
            f"| {overall['passed']}/{overall['total']} | {overall['status']} | {overall['hard_gates_status']} |"
        )
    return lines


def _gate_detail_table(payloads: dict[str, dict]) -> list[str]:
    lines = ["| Candidate | Gate | Status | Detail |", "|---|---|---|---|"]
    for candidate_id in W9_CANDIDATE_IDS:
        validation = payloads[candidate_id].get("validation")
        if not validation:
            continue
        for gate in validation["gates"]:
            lines.append(f"| {candidate_id} | {gate['gate_id']} {gate['name']} | {gate['status']} | {gate['detail']} |")
    return lines


def _feasibility_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | 동시포지션 | 레버리지 | 시작 최소노셔널 | gross(활성자본 배수) | 단일레그 | H4 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for candidate_id in W9_CANDIDATE_IDS:
        meta = payloads[candidate_id].get("metadata", {})
        validation = payloads[candidate_id].get("validation")
        h4_status = next((g["status"] for g in validation["gates"] if g["gate_id"] == "H4"), "N/A") if validation else "N/A"
        lines.append(
            f"| {candidate_id} | {meta.get('num_concurrent_positions', '-')} | {meta.get('leverage', 0):g}x "
            f"| {_fmt_usd(meta.get('min_notional_at_start', 0.0))} | {meta.get('gross_fraction_at_start', 0.0):.2f}x "
            f"| {meta.get('single_leg')} | {h4_status} |"
        )
    return lines


def _failure_reason(payload: dict) -> str:
    """Classify a FAILing candidate as one of 엣지 부재 / 비용 초과 / 청산 발생 (task
    contract: honesty requirement -- distinguish these three, not just report FAIL).
    Priority: liquidation first (most decisive/catastrophic), then gross-vs-net P&L
    (edge never existed even before costs, vs. edge existed but costs consumed it)."""
    meta = payload.get("metadata", {})
    trades = payload.get("trades") or []
    liquidation_events = int(meta.get("liquidation_events", 0))
    if not trades:
        return "엣지 부재 (신호 없음: 사전등록 조건을 충족하는 거래가 한 번도 발생하지 않음)"
    gross_total = sum(float(trade["gross_pnl_dollars"]) for trade in trades)
    net_total = sum(float(trade["pnl_dollars"]) for trade in trades)
    fee_total = sum(float(trade["fee_dollars"]) for trade in trades)
    liquidated_trade_count = sum(1 for trade in trades if trade.get("liquidated_legs"))
    if liquidation_events > 0 and (liquidated_trade_count / max(len(trades), 1)) >= 0.05:
        return (
            f"청산 발생 (거래 {len(trades)}건 중 청산 {liquidated_trade_count}건, "
            f"이벤트 {liquidation_events}회; gross P&L={_fmt_usd(gross_total)}, net P&L={_fmt_usd(net_total)})"
        )
    if gross_total <= 0.0:
        return f"엣지 부재 (비용 차감 전 gross P&L={_fmt_usd(gross_total)} <= 0; 신호 자체가 우위 없음)"
    if net_total <= 0.0:
        return (
            f"비용 초과 (gross P&L={_fmt_usd(gross_total)} > 0 이지만 수수료·슬리피지 {_fmt_usd(fee_total)} 차감 후 "
            f"net P&L={_fmt_usd(net_total)} <= 0)"
        )
    return f"거래 자체는 순이익(net P&L={_fmt_usd(net_total)})이지만 MC/블록셔플/OOS 게이트 중 하나 이상 미달"


def _passing_candidates(payloads: dict[str, dict]) -> list[tuple[str, float]]:
    passing = []
    for candidate_id in W9_CANDIDATE_IDS:
        validation = payloads[candidate_id].get("validation")
        if validation and validation["overall"]["status"] == "PASS":
            passing.append((candidate_id, validation["mc_bootstrap"]["median"]))
    passing.sort(key=lambda item: item[1], reverse=True)
    return passing


def _verdict_section(payloads: dict[str, dict]) -> list[str]:
    passing = _passing_candidates(payloads)
    lines = ["## 판정", ""]
    if passing:
        lines.append(f"**통과 후보 {len(passing)}/6 (H1-H5 전부 PASS)** -- MC 중앙값 최종자본 내림차순:")
        lines.append("")
        for rank, (candidate_id, median) in enumerate(passing, 1):
            lines.append(f"{rank}. **{candidate_id}** -- MC 중앙값 {_fmt_usd(median)} ({CANDIDATE_DEFINITIONS[candidate_id]})")
        lines.append("")
        lines.append(
            "**리스크 등급 주의**: 위 통과 후보는 \"$100 티어 고변동 후보\"로, 기존 캐리(W2c 등) 후보와 "
            "**리스크 등급이 다르다**. 목적함수 자체가 안정성(샤프/MDD 최소화)이 아니라 MC 중앙값 최종자본 "
            "최대화이며, 레버리지·단일심볼 집중·청산 리스크를 명시적으로 감수한다. 샤프/Calmar는 참고 "
            "지표일 뿐 채택 기준이 아니다 (SPEC.md)."
        )
    else:
        lines.append("**전멸: 6개 후보 모두 H1-H5 게이트 중 하나 이상 미달. $100·단기·고수익 조건에서 검증된 엣지 없음.**")
        lines.append("")
        lines.append("후보별 실패 원인 (엣지 부재 / 비용 초과 / 청산 발생 중 구분):")
        lines.append("")
        lines.append("| Candidate | 실패 원인 |")
        lines.append("|---|---|")
        for candidate_id in W9_CANDIDATE_IDS:
            payload = payloads[candidate_id]
            validation = payload.get("validation")
            if validation and validation["overall"]["status"] == "PASS":
                continue
            lines.append(f"| {candidate_id} | {_failure_reason(payload)} |")
    lines.append("")
    return lines


def _multiple_testing_section(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "## 다중검정 보정 (참고)",
        "",
        "SPEC.md: \"이 6개가 전부. 사후 파라미터 조정 금지. 기존 52후보와 합산한 시행횟수(58)로 DSR 보정 표기.\" "
        "DSR(deflated Sharpe ratio)은 trials=58로 계산했으며 참고 지표일 뿐 채택 기준이 아니다.",
        "",
        "| Candidate | DSR score | DSR probability | trials |",
        "|---|---:|---:|---:|",
    ]
    for candidate_id in W9_CANDIDATE_IDS:
        validation = payloads[candidate_id].get("validation")
        dsr = (validation or {}).get("reference_metrics", {}).get("dsr")
        if dsr is None:
            lines.append(f"| {candidate_id} | N/A | N/A | - |")
        else:
            lines.append(f"| {candidate_id} | {_fmt4(dsr['score'])} | {_fmt4(dsr['probability'])} | {dsr['trials']} |")
    lines.append("")
    return lines


def write_wave9_report(results_dir: Path, report_dir: Path, registry_path: Path) -> None:
    payloads = {candidate_id: _load(results_dir, candidate_id) for candidate_id in W9_CANDIDATE_IDS}
    report_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Wave-9 리포트 -- $100 네이티브 단일레그 퍼프 (고변동 티어)",
        "",
        "사전등록: research/wave9_100usd/SPEC.md. 배경: wave-8은 $300 기준 후보(캐리/모멘텀)가 $100에서 "
        "실행 불가함을 확인했다 (캐리 gross 2x, 모멘텀 레그 $0.10 < 최소주문 $5). 이 wave는 처음부터 "
        "$100·단일레그 퍼프·동시 포지션 <=2 제약 안에서 설계된 6개 후보를 캐시(research/wave3/cache, "
        "Binance USDT-M, crypto-only)만으로 백테스트한다. 목적함수는 MC 중앙값 최종자본 최대화 "
        "(샤프/Calmar는 참고 지표).",
        "",
        "## 후보 개요",
        "",
        *_overview_table(payloads),
        "",
        "## 심층검증 (MC 트레이드부트스트랩 1e4 / 블록셔플 90일 1e3 / OOS 2025-10~ / H1-H5)",
        "",
        *_deep_validation_table(payloads),
        "",
        "### 게이트별 상세",
        "",
        *_gate_detail_table(payloads),
        "",
        "## 자본 현실성 (H4: 총자본 $100, 현금버퍼 10% -> 활성자본 $90, 최소주문 $5)",
        "",
        *_feasibility_table(payloads),
        "",
        *_verdict_section(payloads),
        *_multiple_testing_section(payloads),
        "## 모델링 노트",
        "",
        "- 체결: 시그널 바 종가 확정 -> 다음 바 시가 체결, 룩어헤드 없음.",
        "- 비용: 테이커 0.06%/사이드 + 슬리피지(메이저 1bp, 알트 3bp); 메이커 가정 없음.",
        "- 청산: research/wave4_leverage/sweep.py의 liquidation_loss를 그대로 재사용 "
        "(유지증거금 0.5%, 청산수수료 0.06%); 진입가 대비 누적 최악역행을 매 보유일마다 점검.",
        "- 펀딩은 전 후보에 동일하게 적용(모든 퍼프 포지션은 실제로 펀딩을 주고받음); W9e는 방향손익과 "
        "펀딩수취가 모두 반영된다 (헤지 없음).",
        "- 유니버스: research/wave3/cache 기반 Binance USDT-M 퍼프, crypto-only (토큰화 주식 제외). "
        "SPEC.md의 'Bitget USDT-M'은 목표 실거래 venue의 계약 성격을 서술한 것이며, 이 백테스트의 "
        "캐시 데이터 소스는 아니다 (Bitget 크립토 퍼프 OHLC 캐시 자체가 저장소에 없음).",
        "- 비용 보수화: 동일 심볼이 다음 주기에도 재선정되더라도 매 주기 진입+청산 수수료를 새로 부과한다 "
        "(포지션 이월 최적화 없음) -- 실제보다 비용을 과소평가하지 않기 위한 보수적 단순화.",
        "- 매 결과 JSON의 equity 시리즈는 거래 경계(진입/청산)에서만 기록된다 (일별 스무딩 없음); "
        "청산 판정 자체는 보유 중 매일 점검한다.",
        "",
    ]
    (report_dir / "wave9_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = ["# Wave-9 registry", "", "| Candidate | Family | State | Verdict | Hard(H1/H2/H4) |", "|---|---|---|---|---|"]
    for candidate_id in W9_CANDIDATE_IDS:
        validation = payloads[candidate_id].get("validation")
        state = "EVALUATED" if validation else "RUN_ONLY"
        verdict = validation["overall"]["status"] if validation else "UNTESTED"
        hard = validation["overall"]["hard_gates_status"] if validation else "UNTESTED"
        registry_lines.append(f"| {candidate_id} | F9 | {state} | {verdict} | {hard} |")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave9_report"]
