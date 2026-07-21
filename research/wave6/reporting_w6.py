# Renders research/wave6/REGISTRY.md and research/wave6/report/wave6_report.md from the JSON
# results and gates_summary_w6.json produced by the run/gates stages.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave1.common import JsonValue, PipelineError, load_json
from research.wave6.engine_w6 import BASE_DIR, REPORT_DIR, RESULTS_DIR
from research.wave6.strategies_w6 import EXPLORATORY_IDS, STANDARD_IDS


EMPHASIS_GATES: Final = (2, 3, 4, 5, 7, 9, 16)
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"
REPORT_PATH: Final = REPORT_DIR / "wave6_report.md"


def _payload(candidate_id: str) -> dict[str, JsonValue]:
    path = RESULTS_DIR / f"{candidate_id}.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def _gate_status_map(payload: dict[str, JsonValue]) -> dict[int, str]:
    gates = payload.get("gates")
    if not isinstance(gates, list):
        return {}
    return {int(row["gate"]): str(row["status"]) for row in gates if isinstance(row, dict) and isinstance(row.get("gate"), int) and isinstance(row.get("status"), str)}


def _oos_metrics(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    metrics = payload.get("metrics")
    oos = metrics.get("oos") if isinstance(metrics, dict) else None
    return oos if isinstance(oos, dict) else {}


def _fmt(value: JsonValue, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _final_verdict(standard_verdicts: dict[str, JsonValue], combination: dict[str, JsonValue] | None, survivors: list[JsonValue]) -> str:
    if not survivors:
        failed = sorted(cid for cid, verdict in standard_verdicts.items() if verdict == "FAIL")
        untested = sorted(cid for cid, verdict in standard_verdicts.items() if verdict not in ("FAIL", "PASS"))
        detail = f"FAIL 확정 {len(failed)}건({', '.join(failed)})"
        if untested:
            detail += f"; 판정 보류 {len(untested)}건({', '.join(untested)} — OOS 구간에 트리거 0회, 데이터 부족으로 UNTESTED이지 FAIL 확정 아님)"
        return f"신규 정보원에서도 보완 없음 — 표준 4후보(W6a-d) 중 PASS 0건. {detail}. 다음 발굴 축: 전진 수집 데이터(호가·심도)."
    survivor_names = ", ".join(str(item) for item in survivors)
    if combination is not None and combination.get("verdict") == "PASS":
        return f"배포 후보: {survivor_names} + W2c 결합 (correlation·MDD·CAGR·OOS 전부 통과)."
    combo_verdict = combination.get("verdict") if combination is not None else "N/A"
    return f"단독 생존: {survivor_names} — W2c 결합 게이트 미통과({combo_verdict}). 결합 개선 없이는 배포 보류."


def write_reports() -> Path:
    summary_path = RESULTS_DIR / "gates_summary_w6.json"
    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        raise PipelineError("gates_summary_w6.json is missing or invalid; run the gates stage first")
    standard_verdicts_value = summary.get("standard_verdicts")
    standard_verdicts = standard_verdicts_value if isinstance(standard_verdicts_value, dict) else {}
    exploratory_value = summary.get("exploratory")
    exploratory = exploratory_value if isinstance(exploratory_value, dict) else {}
    combination = summary.get("combination") if isinstance(summary.get("combination"), dict) else None
    survivors_value = summary.get("survivors")
    survivors = survivors_value if isinstance(survivors_value, list) else []

    registry_lines = ["# Wave-6 registry", "", "| Candidate | Grade | Family | Verdict |", "|---|---|---|---|"]
    for candidate_id in STANDARD_IDS:
        payload = _payload(candidate_id)
        family = str(payload.get("family", "F6"))
        verdict = str(standard_verdicts.get(candidate_id, "MISSING"))
        registry_lines.append(f"| {candidate_id} | STANDARD | {family} | {verdict} |")
    for candidate_id in EXPLORATORY_IDS:
        payload = _payload(candidate_id)
        family = str(payload.get("family", "F6"))
        summary_row = exploratory.get(candidate_id, {})
        verdict = str(summary_row.get("verdict", "MISSING")) if isinstance(summary_row, dict) else "MISSING"
        registry_lines.append(f"| {candidate_id} | EXPLORATORY | {family} | {verdict} |")
    if combination is not None:
        registry_lines.append(f"| W6+W2c combo | COMBINATION | F1+F6 | {combination.get('verdict')} |")
    REGISTRY_PATH.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Wave-6 report",
        "",
        "New information sources at 1h resolution: Binance BTC/ETH/SOL funding-window timing, an "
        "intraday equity-open spillover, a weekend seasonal drift, a stock-token/underlying deviation "
        "fade, and a Bitget new-listing effect. IS ends 2025-09-30, OOS begins 2025-10-01 (wave-1 "
        "split, inherited). Intraday candidates carry 2x base slippage per the wave-6 preamble.",
        "",
        "## Known proxy / approximation limitations (declared in SPEC.md, restated here)",
        "",
        "- **W6a/W6b**: no historical predicted-funding series exists, so the entry signal uses the "
        "funding rate realized at the *prior* settlement as a proxy for the upcoming payment.",
        "- **W6c**: hourly bars do not align to the pre-registered 12:30/13:30 UTC boundaries; the "
        "signal is approximated by the [12:00,13:00) bar's own return and the entry by the 13:00 bar's "
        "open.",
        "- **W6f**: see its dedicated section below if UNDETERMINED — Bitget's `launchTime` field is "
        "empty across the entire contracts payload (verified against both the cached wave-3 snapshot "
        "and a live refetch).",
        "",
        "## Standard candidates (19-gate table; gates " + ", ".join(str(g) for g in EMPHASIS_GATES) + " emphasized)",
        "",
        "| Candidate | Verdict | " + " | ".join(f"G{g}" for g in EMPHASIS_GATES) + " | OOS Sharpe | OOS trades |",
        "|---|---|" + "---|" * len(EMPHASIS_GATES) + "---:|---:|",
    ]
    for candidate_id in STANDARD_IDS:
        payload = _payload(candidate_id)
        statuses = _gate_status_map(payload)
        oos = _oos_metrics(payload)
        verdict = str(standard_verdicts.get(candidate_id, "MISSING"))
        gate_cells = " | ".join(statuses.get(g, "?") for g in EMPHASIS_GATES)
        sharpe = _fmt(oos.get("sharpe"))
        n_trades = oos.get("n_trades", "?")
        report_lines.append(f"| {candidate_id} | {verdict} | {gate_cells} | {sharpe} | {n_trades} |")

    report_lines.extend(["", "## Exploratory candidates (effect stats only; no deployment claim)", "", "| Candidate | Verdict | Direction | t-stat | Cost-after mean | Sample |", "|---|---|---|---:|---:|---:|"])
    for candidate_id in EXPLORATORY_IDS:
        row = exploratory.get(candidate_id, {})
        row = row if isinstance(row, dict) else {}
        report_lines.append(
            f"| {candidate_id} | {row.get('verdict', 'MISSING')} | {row.get('direction', '-')} | "
            f"{_fmt(row.get('t_stat'))} | {_fmt(row.get('cost_after_mean'), 6)} | {row.get('sample_size', 0)} |"
        )
        if candidate_id == "W6f" and row.get("verdict") == "UNDETERMINED":
            payload = _payload(candidate_id)
            metadata = payload.get("metadata")
            reason = metadata.get("reason") if isinstance(metadata, dict) else None
            if reason:
                report_lines.append(f"  - W6f reason: {reason}")

    report_lines.extend(["", "## W2c combination check"])
    if combination is None:
        report_lines.append("")
        report_lines.append("No standard candidate passed all 19 gates, so the W2c combination step (SPEC.md: "
                             "\"생존자는 W2c와 결합 게이트 추가 판정\") did not run. This is not a failed test -- "
                             "there was no survivor to combine.")
    else:
        report_lines.extend(
            [
                "",
                f"- Survivor combined with W2c: `{combination.get('survivor')}`",
                f"- Correlation: `{combination.get('correlation')}`; pass=`{combination.get('correlation_pass')}`",
                f"- MDD: `{combination.get('mdd')}` vs baseline `{combination.get('baseline_mdd')}`; pass=`{combination.get('mdd_pass')}`",
                f"- CAGR: `{combination.get('cagr')}` vs baseline `{combination.get('baseline_cagr')}`; pass=`{combination.get('cagr_pass')}`",
                f"- OOS return: `{combination.get('oos_return')}` vs baseline `{combination.get('baseline_oos_return')}`; pass=`{combination.get('oos_pass')}`",
                f"- Final verdict: **{combination.get('verdict')}**",
            ]
        )

    report_lines.extend(["", "## Verdict", "", _final_verdict(standard_verdicts, combination, survivors)])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return REPORT_PATH


__all__ = ["REGISTRY_PATH", "REPORT_PATH", "write_reports"]
