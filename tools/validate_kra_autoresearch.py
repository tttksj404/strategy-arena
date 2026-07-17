from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kra_promotion_policy import MIN_ABSOLUTE_TOP1_LIFT_PP, clears_absolute_top1_lift


class CompletionResult(TypedDict):
    status: str
    passed: bool
    summary: str
    output_artifact_path: str


def validate_report(report_path: str) -> CompletionResult:
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    selected_result = report.get("selected_result") or {}
    bootstrap = selected_result.get("pooled_bootstrap") or {}
    fresh_holdout = selected_result.get("fresh_holdout") or {}
    fresh_bootstrap = fresh_holdout.get("bootstrap") or {}
    passed = bool(
        report.get("selected")
        and report.get("promotion_pass") is True
        and selected_result.get("promotion_pass") is True
        and clears_absolute_top1_lift(float(bootstrap.get("mean_pp", 0.0)))
        and float(bootstrap.get("ci95_low_pp", 0.0)) > 0.0
        and clears_absolute_top1_lift(float(fresh_bootstrap.get("mean_pp", 0.0)))
        and float(fresh_bootstrap.get("ci95_low_pp", 0.0)) > 0.0
        and float(selected_result.get("pooled_logloss_delta", 1.0)) <= 0.0
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "summary": (
            f"validated candidate clears +{MIN_ABSOLUTE_TOP1_LIFT_PP:.1f}pp v4 gate"
            if passed
            else f"no candidate cleared +{MIN_ABSOLUTE_TOP1_LIFT_PP:.1f}pp v4 gate"
        ),
        "output_artifact_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--completion", required=True, type=Path)
    args = parser.parse_args()
    result = validate_report(args.report)
    args.completion.parent.mkdir(parents=True, exist_ok=True)
    args.completion.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
