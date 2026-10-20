from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research" / "wave4_leverage" / "results"
REPORT = ROOT / "research" / "wave4_leverage" / "LEVERAGE_REPORT.md"
LEVERAGES = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
expected = {
    f"{candidate}_{structure}_L{leverage:.1f}.json".replace(f"L{leverage:.1f}", f"L{leverage:.1f}".replace(".", "p")): None
    for candidate in ("W2c", "F1f")
    for structure in ("SYM", "ASYM")
    for leverage in LEVERAGES
}
files = sorted(RESULTS.glob("*.json"))
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
combos = {
    (item["candidate_id"], item["structure"], float(item["valid_leverage"]))
    for item in payloads
}
required = {"candidate_id", "structure", "valid_leverage", "grid_preregistered", "initial_capital", "metrics", "model", "source"}
metric_keys = {
    "cagr",
    "mdd",
    "mc_paths",
    "mc_p05",
    "bankruptcy_probability_final_below_150",
    "liquidation_count",
    "borrowing_cost_total",
}
all_required = all(required <= set(item) and metric_keys <= set(item["metrics"]) for item in payloads)
all_finite = all(
    math.isfinite(float(item["metrics"][key]))
    for item in payloads
    for key in metric_keys - {"mc_paths", "liquidation_count"}
)
json_gates = sum(
    float(item["metrics"]["mc_p05"]) > 300.0
    and float(item["metrics"]["bankruptcy_probability_final_below_150"]) < 0.05
    and float(item["metrics"]["mdd"]) <= 0.25
    for item in payloads
)
report_text = REPORT.read_text(encoding="utf-8")
rows = [line for line in report_text.splitlines() if re.match(r"^\|\s*(W2c|F1f)\s*\|", line)]
report_pass = sum(bool(re.search(r"\|\s*PASS\s*\|$", line)) for line in rows)
report_fail = sum(bool(re.search(r"\|\s*FAIL\s*\|$", line)) for line in rows)
print(
    json.dumps(
        {
            "json_count": len(files),
            "expected_count": len(expected),
            "exact_names": {path.name for path in files} == set(expected),
            "unique_combinations": len(combos),
            "required_fields": all_required,
            "cache_only_all_true": all(item["source"]["cache_only"] for item in payloads),
            "cache_dir_all_wave1": all(item["source"]["cache_dir"] == "research/wave1/cache" for item in payloads),
            "grid_all_preregistered": all(item["grid_preregistered"] for item in payloads),
            "mc_paths_all_10000": all(item["metrics"]["mc_paths"] == 10000 for item in payloads),
            "model_constants_all_match": all(
                item["model"]["borrow_apr"] == 0.10
                and item["model"]["maintenance_rate"] == 0.005
                and item["model"]["liquidation_fee_rate"] == 0.0006
                for item in payloads
            ),
            "all_metrics_finite": all_finite,
            "json_gate_pass_count": json_gates,
            "report_nonempty": REPORT.stat().st_size > 0,
            "report_table_rows": len(rows),
            "report_pass_rows": report_pass,
            "report_fail_rows": report_fail,
            "report_gate_count_matches": report_pass == json_gates and report_fail == len(payloads) - json_gates,
            "report_declares_24": "24" in report_text,
        },
        sort_keys=True,
    )
)
