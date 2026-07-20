from __future__ import annotations

from pathlib import Path

from research.wave1.common import JsonValue, load_json
from research.wave5.gates import SINGLE_IDS


def _status_map(payload: dict[str, JsonValue]) -> dict[int, str]:
    gates = payload.get("gates")
    if not isinstance(gates, list):
        return {}
    return {int(row["gate"]): str(row["status"]) for row in gates if isinstance(row, dict) and isinstance(row.get("gate"), int) and isinstance(row.get("status"), str)}


def _verdict(payload: dict[str, JsonValue], candidate_id: str) -> str:
    if candidate_id == "W5g":
        combination = payload.get("combination_gates")
        if isinstance(combination, dict):
            return str(combination.get("verdict", "FAIL"))
    validation = payload.get("validation")
    if isinstance(validation, dict) and isinstance(validation.get("oos_label"), str):
        return str(validation["oos_label"])
    statuses = _status_map(payload)
    return "PASS" if all(statuses.get(gate) == "PASS" for gate in range(1, 20)) else "FAIL"


def write_reports(results_dir: Path, report_dir: Path, registry_path: Path, selected: str, combination: dict[str, JsonValue]) -> None:
    candidate_ids = ("W2c", *SINGLE_IDS, "W5g")
    registry = ["# Wave-5 registry", "", f"Selected single candidate: `{selected}`", "", "| Candidate | Family | State | Verdict |", "|---|---|---|---|"]
    report = ["# Wave-5 report", "", "Wave-5 uses only the frozen wave-1 cache. No fetch stage or network call is part of this pipeline.", "", f"Single-candidate tie-break selection: **{selected}**.", "", "| Candidate | Family | Verdict | OOS label |", "|---|---|---|---|"]
    for candidate_id in candidate_ids:
        path = results_dir / f"{candidate_id}.json"
        if not path.exists():
            registry.append(f"| {candidate_id} | ? | MISSING | FAIL |")
            report.append(f"| {candidate_id} | ? | FAIL | MISSING |")
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        verdict = _verdict(payload, candidate_id)
        validation = payload.get("validation")
        label = validation.get("oos_label", "") if isinstance(validation, dict) else ""
        family = str(payload.get("family", "?"))
        state = "COMBINED" if candidate_id == "W5g" else ("BASELINE" if candidate_id == "W2c" else "EVALUATED")
        registry.append(f"| {candidate_id} | {family} | {state} | {verdict} |")
        report.append(f"| {candidate_id} | {family} | {verdict} | {label} |")
    report.extend(["", "## W5g combination decision", "", f"- Selected candidate: `{selected}`", f"- Correlation: `{combination.get('correlation')}`; pass=`{combination.get('correlation_pass')}`", f"- MDD: `{combination.get('mdd')}` vs baseline `{combination.get('baseline_mdd')}`; pass=`{combination.get('mdd_pass')}`", f"- CAGR: `{combination.get('cagr')}` vs baseline `{combination.get('baseline_cagr')}`; pass=`{combination.get('cagr_pass')}`", f"- OOS return: `{combination.get('oos_return')}` vs baseline `{combination.get('baseline_oos_return')}`; pass=`{combination.get('oos_pass')}`", f"- Final verdict: **{combination.get('verdict')}**", "", "A missing OOS interval remains untested; it is not converted into a performance win."])
    registry_path.write_text("\n".join(registry) + "\n", encoding="utf-8")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave5_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


__all__ = ["write_reports"]
