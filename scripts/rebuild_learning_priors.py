#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import update_prediction_feedback as feedback

DEFAULT_REPORT = ROOT / "data" / "roster_audit_report.json"
DEFAULT_PRIORS = ROOT / "data" / "participant_learning_priors.json"
DEFAULT_SUMMARY = ROOT / "data" / "prediction_feedback_summary.json"


def mismatch_keys_from_report(report_path: Path) -> set[tuple[str, str, str, str]]:
    if not report_path.exists():
        raise FileNotFoundError(f"roster audit report missing: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row.get("status") != "mismatch":
            continue
        keys.add((
            str(row.get("sport") or "keirin"),
            str(row.get("date") or ""),
            str(row.get("meet") or ""),
            str(row.get("race_no") or ""),
        ))
    return keys


def backup_priors(priors_path: Path) -> Path | None:
    if not priors_path.exists():
        return None
    stamp = dt.datetime.now().strftime("%Y%m%d")
    backup_path = priors_path.with_name(f"{priors_path.stem}.backup_{stamp}{priors_path.suffix}")
    shutil.copy2(priors_path, backup_path)
    return backup_path


def rebuild(report_path: Path = DEFAULT_REPORT, priors_path: Path = DEFAULT_PRIORS, summary_path: Path = DEFAULT_SUMMARY) -> dict:
    excluded = mismatch_keys_from_report(report_path)
    backup_path = backup_priors(priors_path)
    result = feedback.build_feedback(
        feedback.DEFAULT_SQLITE,
        feedback.DEFAULT_SNAPSHOTS,
        kcycle_outcome_path=feedback.DEFAULT_KCYCLE_OUTCOMES,
        kra_db_path=feedback.DEFAULT_KRA_DB,
        exclude_keys=excluded,
    )
    feedback.write_feedback(result, priors_path, summary_path)
    return {
        "excluded": len(excluded),
        "backup_path": str(backup_path) if backup_path else None,
        "matched_races": result["summary"]["matched_races"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--out-priors", default=str(DEFAULT_PRIORS))
    parser.add_argument("--out-summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()
    try:
        result = rebuild(Path(args.report), Path(args.out_priors), Path(args.out_summary))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
