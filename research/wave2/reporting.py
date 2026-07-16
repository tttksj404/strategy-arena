from __future__ import annotations

from pathlib import Path

from research.wave1.common import JsonValue, load_json


def _status_map(payload: dict[str, JsonValue]) -> dict[int, str]:
    gates = payload.get("gates")
    if not isinstance(gates, list):
        return {}
    return {
        int(row["gate"]): str(row["status"])
        for row in gates
        if isinstance(row, dict) and isinstance(row.get("gate"), int) and isinstance(row.get("status"), str)
    }


def write_wave2_reports(results_dir: Path, report_dir: Path, registry_path: Path, candidate_ids: tuple[str, ...]) -> None:
    registry = ["# Wave-2 registry", "", "| Candidate | Family | State | Required gates |", "|---|---|---|---|"]
    report = ["# Wave-2 report", "", "| Candidate | Family | State | Verdict | OOS label | Effect cost-after sign | Basis MTM | Basis complete |", "|---|---|---|---|---|---|---|---|"]
    for candidate_id in candidate_ids:
        path = results_dir / f"{candidate_id}.json"
        if not path.exists():
            registry.append(f"| {candidate_id} | ? | MISSING | FAIL |")
            report.append(f"| {candidate_id} | ? | MISSING | FAIL | | | | |")
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        statuses = _status_map(payload)
        family = str(payload.get("family", "?"))
        metadata = payload.get("metadata")
        exploratory = metadata.get("exploratory_only") is True if isinstance(metadata, dict) else False
        oos_label = payload.get("validation", {}).get("oos_label", "") if isinstance(payload.get("validation"), dict) else ""
        verdict = oos_label if oos_label else ("PASS" if not exploratory and all(statuses.get(gate) == "PASS" for gate in range(1, 20)) else "FAIL")
        state = "EXPLORATORY" if exploratory else (oos_label or "EVALUATED")
        effect = metadata.get("effect") if isinstance(metadata, dict) else None
        effect_sign = effect.get("cost_after_sign", "") if isinstance(effect, dict) else ""
        basis = metadata.get("basis_mark_to_market_included", "") if isinstance(metadata, dict) else ""
        basis_complete = metadata.get("basis_complete", "") if isinstance(metadata, dict) else ""
        registry.append(f"| {candidate_id} | {family} | {state} | {verdict} |")
        report.append(f"| {candidate_id} | {family} | {state} | {verdict} | {oos_label} | {effect_sign} | {basis} | {basis_complete} |")
    registry_path.write_text("\n".join(registry) + "\n", encoding="utf-8")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
