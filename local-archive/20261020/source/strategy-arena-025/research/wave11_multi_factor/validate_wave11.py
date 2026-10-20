import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results" / "wave11_results.json"
REPORT = BASE_DIR / "report" / "wave11_results.md"
VALIDATION = BASE_DIR / "results" / "validation.json"


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = payload.get("results")
    ids = [row.get("candidate_id") for row in rows] if isinstance(rows, list) else []
    passed = (
        len(ids) == 16
        and len(set(ids)) == 16
        and payload.get("eligible_candidates") == []
        and all(row.get("oos_independent") is False for row in rows)
        and all(row.get("all_gates_pass") is False for row in rows)
        and REPORT.exists()
    )
    result = {"status": "passed" if passed else "failed", "passed": passed, "summary": "Wave-11 multi-family screen completed with explicit non-promotion gating."}
    VALIDATION.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
