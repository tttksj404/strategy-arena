# Wave-7 markdown report + registry writer. Pure formatting over already-computed
# results/W7{a..d}.json payloads (written by the run/gates CLI stages) -- no
# additional computation happens here.

from __future__ import annotations

import json
from pathlib import Path

from research.wave7.engine_w7 import CANDIDATE_DEFINITIONS, W7_CANDIDATE_IDS


def _load(results_dir: Path, candidate_id: str) -> dict:
    path = results_dir / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _fmt4(value: float) -> str:
    return f"{value:.4f}"


def _overview_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | Definition | Total Ret | CAGR | Sharpe | MDD | Calmar |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate_id in W7_CANDIDATE_IDS:
        payload = payloads[candidate_id]
        metrics = payload.get("deep_validation", {}).get("metrics", {})
        lines.append(
            f"| {candidate_id} | {payload.get('definition', CANDIDATE_DEFINITIONS[candidate_id])} "
            f"| {_fmt_pct(metrics.get('total_ret', 0.0))} | {_fmt_pct(metrics.get('cagr', 0.0))} "
            f"| {_fmt4(metrics.get('sharpe', 0.0))} | {_fmt_pct(metrics.get('mdd', 0.0))} "
            f"| {_fmt4(metrics.get('calmar', 0.0))} |"
        )
    return lines


def _deep_validation_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | MC p05 | Ruin P(<150) | Block MDD p95 | Dormant OOS | Sharpe (combined vs carry-alone) | Corr w/ W2c | Gates | Overall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate_id in W7_CANDIDATE_IDS:
        deep = payloads[candidate_id].get("deep_validation")
        if not deep:
            lines.append(f"| {candidate_id} | - | - | - | - | - | - | - | NOT_EVALUATED (run --stage gates) |")
            continue
        mc = deep["bootstrap_mc"]
        blocks = deep["block_shuffle"]
        overall = deep["overall"]
        lines.append(
            f"| {candidate_id} | ${mc['p05']:.2f} | {_fmt_pct(mc['ruin_probability'])} "
            f"| {_fmt_pct(blocks['mdd_p95'])} | {_fmt_pct(deep['dormant_oos_return'])} "
            f"| {_fmt4(deep['metrics']['sharpe'])} vs {_fmt4(deep['carry_alone_sharpe'])} "
            f"| {_fmt4(deep['correlation_with_carry'])} "
            f"| {overall['passed_gates']}/{overall['total_gates']} | {overall['status']} |"
        )
    return lines


def _gate_detail_table(payloads: dict[str, dict]) -> list[str]:
    lines = ["| Candidate | Gate | Status | Value |", "|---|---|---|---|"]
    for candidate_id in W7_CANDIDATE_IDS:
        deep = payloads[candidate_id].get("deep_validation")
        if not deep:
            continue
        for gate in deep["gates"]:
            lines.append(f"| {candidate_id} | {gate['name']} | {gate['status']} | {gate['value']} |")
    return lines


def _capital_reality_table(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "| Candidate | Max combined weight | Buffer<=90% | Carry min leg | Momentum min leg | Status |",
        "|---|---:|---|---:|---:|---|",
    ]
    for candidate_id in W7_CANDIDATE_IDS:
        reality = payloads[candidate_id].get("capital_reality", {})
        lines.append(
            f"| {candidate_id} | {reality.get('max_combined_weight', float('nan')):.2f} "
            f"| {reality.get('buffer_ok')} | ${reality.get('carry_min_leg_usd', float('nan')):.2f} "
            f"| ${reality.get('momentum_min_leg_usd', float('nan')):.2f} | {reality.get('status', 'N/A')} |"
        )
    return lines


def _passing_candidates(payloads: dict[str, dict]) -> list[str]:
    passing = []
    for candidate_id in W7_CANDIDATE_IDS:
        deep = payloads[candidate_id].get("deep_validation")
        if deep and deep["overall"]["status"] == "PASS":
            passing.append(candidate_id)
    return passing


def _max_safe_momentum_weight(payloads: dict[str, dict], passing: list[str]) -> float:
    best = 0.0
    for candidate_id in passing:
        weights = payloads[candidate_id].get("weights", {}).get("momentum", [])
        if weights:
            best = max(best, max(item["value"] for item in weights))
    return best


def _verdict_section(payloads: dict[str, dict]) -> list[str]:
    passing = _passing_candidates(payloads)
    lines = ["## 판정", ""]
    if passing:
        max_weight = _max_safe_momentum_weight(payloads, passing)
        lines.append(f"통과 구성: {', '.join(passing)} ({len(passing)}/4).")
        lines.append("")
        lines.append(f"**최대 안전 모멘텀 비중: {max_weight:.2f}** (통과 구성 중 실사용된 모멘텀 슬리브 최대 블렌드 비중).")
        lines.append("")
        lines.append(
            "카드 갱신 노트: SPEC.md은 통과 시 STRATEGY_CARD.md에 \"코어 캐리 + 저확신 모멘텀 슬리브\" 결합 카드 추가를 "
            "요구하지만, 이번 작업 지시는 research/wave7/ 밖 파일 수정을 금지했으므로 STRATEGY_CARD.md 갱신은 "
            "이 실행 범위에서 수행하지 않았다. 별도 세션에서 위 통과 구성/최대 안전 모멘텀 비중을 반영해 갱신 필요."
        )
    else:
        lines.append("**전멸: 4개 구성 모두 심층검증 게이트 미달 -> 캐리 단독(W2c)이 최적.**")
        lines.append("")
        lines.append("개별 구성 수치는 위 표에 사실대로 기록. 상세 사유는 게이트별 상세표 참조.")
    lines.append("")
    lines.append(
        "모멘텀 슬리브 주의: W3c는 게이트 2(overfit_sensitivity)·게이트 3 계열(IS 일관성) 개별 FAIL, "
        "전체 후보군 자체 딥밸리데이션에서 MC p05 $227.99(UNDETERMINED, trade_returns가 daily-active 시맨틱), "
        "블록셔플 MDD p95 42.71%로 기록된 저확신 슬리브다. 위 결합 결과가 통과하더라도 모멘텀은 "
        "보조 비중으로만 취급해야 한다 (SPEC.md 핵심 규율)."
    )
    return lines


def write_wave7_report(results_dir: Path, report_dir: Path, registry_path: Path) -> None:
    payloads = {candidate_id: _load(results_dir, candidate_id) for candidate_id in W7_CANDIDATE_IDS}
    report_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Wave-7 리포트 -- 캐리+모멘텀 결합 포트폴리오",
        "",
        "사전등록: research/wave7/SPEC.md. 구성요소는 W2c(캐리, research/wave2/results/W2c.json)와 "
        "W3c(모멘텀, research/wave3/results/W3c.json)의 일수익 시리즈만 사용, 신규 신호 탐색 없음.",
        "",
        "## 구성 개요 + 결합 자산곡선 지표",
        "",
        *_overview_table(payloads),
        "",
        "## 심층검증 배터리 (MC 1e4 / 블록셔플 90일 1e3 / 휴면기 OOS / Sharpe 비교 / W2c 상관)",
        "",
        *_deep_validation_table(payloads),
        "",
        "### 게이트별 상세",
        "",
        *_gate_detail_table(payloads),
        "",
        "## 자본 현실성 ($300 x 0.9 = $270 동시마진 버퍼, 최소주문 5 USDT)",
        "",
        *_capital_reality_table(payloads),
        "",
        *_verdict_section(payloads),
        "",
    ]
    (report_dir / "wave7_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = ["# Wave-7 registry", "", "| Candidate | Family | State | Verdict |", "|---|---|---|---|"]
    for candidate_id in W7_CANDIDATE_IDS:
        deep = payloads[candidate_id].get("deep_validation")
        state = "EVALUATED" if deep else "RUN_ONLY"
        verdict = deep["overall"]["status"] if deep else "UNTESTED"
        registry_lines.append(f"| {candidate_id} | F7 | {state} | {verdict} |")
    registry_path.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")


__all__ = ["write_wave7_report"]
