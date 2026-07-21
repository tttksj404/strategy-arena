from __future__ import annotations

import json
import hashlib
from pathlib import Path

from run_wave10 import IDS, REPORT_DIR, RESULTS_DIR


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"wave10 validation failed: {message}")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    results = RESULTS_DIR / "wave10_results.json"
    report = REPORT_DIR / "wave10_report.md"
    manifest_path = REPORT_DIR / "wave10_manifest.json"
    payload = json.loads(results.read_text(encoding="utf-8"))
    _require(tuple(payload["candidate_ids"]) == IDS, "aggregate candidate_ids")
    _require(len(payload["results"]) == len(IDS), "aggregate count")
    for result in payload["results"]:
        _require(result["data"]["data_valid"] is True, f"data validity {result['candidate_id']}")
        _require(result["gates"]["selection_independent"]["passed"] is False, f"selection independence {result['candidate_id']}")
        _require(result["all_gates_pass"] is False, f"fail-closed result {result['candidate_id']}")
        _require(result["execution"]["max_gross"] <= 0.600000001, f"gross cap {result['candidate_id']}")
        _require(json.loads((RESULTS_DIR / f"{result['candidate_id']}.json").read_text(encoding="utf-8")) == result, f"standalone equality {result['candidate_id']}")
    _require(report.is_file(), "report exists")
    _require(manifest_path.is_file(), "manifest exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = Path(__file__).resolve().parents[0]
    _require(manifest["report_sha256"] == _hash(report), "report hash")
    _require(manifest["results_sha256"] == _hash(results), "results hash")
    _require(manifest["spec_sha256"] == _hash(base / "SPEC.md"), "spec hash")
    _require(manifest["runner_sha256"] == _hash(base / "run_wave10.py"), "runner hash")
    _require(tuple(manifest["result_ids"]) == IDS, "manifest ids")
    _require(manifest["eligible"] == [], "manifest eligible")
    _require(manifest["selection_independent"] is False, "manifest selection independence")
    print(f"WAVE10_VALIDATION_PASS candidates={len(IDS)} eligible=[] selection_independent=false results={results} report={report} manifest={manifest_path} report_sha256={manifest['report_sha256']} results_sha256={manifest['results_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
