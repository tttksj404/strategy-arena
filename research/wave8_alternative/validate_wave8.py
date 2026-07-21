from __future__ import annotations

import json
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results" / "wave8_results.json"
REPORT = BASE_DIR / "report" / "wave8_report.md"
EXPECTED = ("R8a", "R8b", "R8c", "R8d", "V8a", "V8b", "V8c", "V8d", "Q8a", "Q8b", "Q8c", "Q8d", "F8a", "F8b", "F8c", "F8d")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"wave8 validation failed: {message}")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    ids = tuple(item["candidate_id"] for item in payload["results"])
    _require(tuple(payload["candidate_ids"]) == EXPECTED, "aggregate candidate_ids")
    _require(ids == EXPECTED, "aggregate result order")
    _require(len(set(ids)) == len(EXPECTED), "duplicate candidate ids")
    for item in payload["results"]:
        _require(item["data"]["data_valid"] is True, f"data validity {item['candidate_id']}")
        _require(item["data"]["selection_independent"] is False, f"selection independence {item['candidate_id']}")
        _require(item["gates"]["oos_independent"]["passed"] is False, f"oos independence gate {item['candidate_id']}")
        _require(item["all_gates_pass"] is False, f"fail-closed result {item['candidate_id']}")
        _require(item["metrics"]["mdd"] >= 0.0, f"MDD sign {item['candidate_id']}")
        _require(item["execution"]["max_gross"] <= 0.6000001, f"gross cap {item['candidate_id']}")
        _require(all(gate["status"] in {"PASS", "FAIL"} for gate in item["gates"].values()), f"gate status {item['candidate_id']}")
        _require(json.loads((BASE_DIR / "results" / f"{item['candidate_id']}.json").read_text(encoding="utf-8")) == item, f"standalone equality {item['candidate_id']}")
    _require(REPORT.is_file(), "report exists")
    manifest_path = BASE_DIR / "report" / "wave8_manifest.json"
    _require(manifest_path.is_file(), "manifest exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest["report_sha256"] == _hash(REPORT), "report hash")
    _require(manifest["results_sha256"] == _hash(RESULTS), "results hash")
    _require(manifest["spec_sha256"] == _hash(BASE_DIR / "SPEC.md"), "spec hash")
    _require(manifest["runner_sha256"] == _hash(BASE_DIR / "run_wave8.py"), "runner hash")
    _require(tuple(manifest["result_ids"]) == EXPECTED, "manifest ids")
    _require(manifest["eligible"] == [], "manifest eligible")
    _require(manifest["selection_independent"] is False, "manifest selection independence")
    print(f"WAVE8_VALIDATION_PASS candidates={len(ids)} eligible=[] results={RESULTS} report={REPORT} manifest={manifest_path} report_sha256={manifest['report_sha256']} results_sha256={manifest['results_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
