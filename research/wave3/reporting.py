"""Wave-3 registry and report writer."""

from __future__ import annotations

from pathlib import Path

from research.wave1.common import JsonValue, load_json
from research.wave3.engine import W3_CANDIDATE_IDS


def _statuses(payload: dict[str, JsonValue]) -> dict[int, str]:
    gates = payload.get("gates")
    if not isinstance(gates, list):
        return {}
    return {
        int(row["gate"]): str(row["status"])
        for row in gates
        if isinstance(row, dict) and isinstance(row.get("gate"), int) and isinstance(row.get("status"), str)
    }


def _yearly_returns(payload: dict[str, JsonValue]) -> dict[int, float]:
    equity = payload.get("equity")
    if not isinstance(equity, list):
        return {}
    grouped: dict[int, list[float]] = {}
    for point in equity:
        if not isinstance(point, dict) or not isinstance(point.get("timestamp"), str) or not isinstance(point.get("value"), (int, float)):
            continue
        grouped.setdefault(int(point["timestamp"][:4]), []).append(float(point["value"]))
    return {year: values[-1] / values[0] - 1.0 for year, values in grouped.items() if values and values[0] != 0.0}


def write_wave3_reports(results_dir: Path, report_dir: Path, registry_path: Path) -> None:
    """Write a deterministic registry and candidate summary."""
    registry = ["# Wave-3 registry", "", "| Candidate | Family | State | Verdict |", "|---|---|---|---|"]
    report = ["# Wave-3 report", "", "| Candidate | Verdict | OOS label | Stress OOS | Stock-token gross contribution |", "|---|---|---|---:|---:|"]
    complete = True
    for candidate_id in W3_CANDIDATE_IDS:
        path = results_dir / f"{candidate_id}.json"
        if not path.exists():
            complete = False
            registry.append(f"| {candidate_id} | F4 | MISSING | FAIL |")
            report.append(f"| {candidate_id} | FAIL | | | |")
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            complete = False
            continue
        statuses = _statuses(payload)
        validation = payload.get("validation")
        oos_label = validation.get("oos_label", "") if isinstance(validation, dict) else ""
        verdict = str(oos_label) if oos_label else ("PASS" if complete and all(statuses.get(gate) == "PASS" for gate in range(1, 20)) else "FAIL")
        registry.append(f"| {candidate_id} | {payload.get('family', 'F4')} | {'EVALUATED' if not oos_label else oos_label} | {verdict} |")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        contribution = metadata.get("asset_type_gross_contribution") if isinstance(metadata, dict) else {}
        stock_contribution = contribution.get("stock_token", 0.0) if isinstance(contribution, dict) else 0.0
        report.append(f"| {candidate_id} | {verdict} | {oos_label} | {float(payload.get('stress_total_return', 0.0)):.4f} | {float(stock_contribution):.6f} |")
        yearly = _yearly_returns(payload)
        if yearly:
            report.append(f"| {candidate_id} yearly |  |  |  | {', '.join(f'{year}: {value:.4f}' for year, value in sorted(yearly.items()))} |")
    e_path = results_dir / "W3e.json"
    f_path = results_dir / "W3f.json"
    if e_path.exists() and f_path.exists():
        e = load_json(e_path)
        f = load_json(f_path)
        if isinstance(e, dict) and isinstance(f, dict):
            report.extend(["", f"W3e vs W3f stress OOS: {float(e.get('stress_total_return', 0.0)):.4f} vs {float(f.get('stress_total_return', 0.0)):.4f}"])
    registry_path.write_text("\n".join(registry) + "\n", encoding="utf-8")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "wave3_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


__all__ = ["write_wave3_reports"]
