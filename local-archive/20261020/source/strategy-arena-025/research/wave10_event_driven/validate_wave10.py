from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
result_path = BASE_DIR / "results" / "wave10_results.json"
report_path = BASE_DIR / "report" / "wave10_results.md"
validation_path = BASE_DIR / "results" / "validation.json"
errors: list[str] = []
if not result_path.exists():
    errors.append("results JSON is missing")
if not report_path.exists():
    errors.append("report is missing")
if not errors:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if len(results) != 8:
        errors.append("registered result count is not eight")
    if payload.get("eligible_candidates") != []:
        errors.append("historical screen unexpectedly claims promotion eligibility")
    if not all(item.get("all_gates_pass") is False for item in results):
        errors.append("all historical results must remain non-promotional")
    if not all(item.get("oos_independent") is False for item in results):
        errors.append("OOS independence must be explicitly false")
status = {"status": "passed" if not errors else "failed", "passed": not errors, "summary": "Wave-10 event-driven historical screen completed with explicit non-promotion gating." if not errors else "; ".join(errors)}
validation_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(status, ensure_ascii=False))
raise SystemExit(0 if not errors else 1)
