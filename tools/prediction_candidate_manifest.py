from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, NotRequired, TypedDict


SCHEMA: Final = "prediction_candidate_manifest_v1"
REPO_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_KRA_META_GATE_PATH: Final = Path("/tmp/kra_meta_gate_20260717.json")
KRA_CONTEXT_RERANK_EVIDENCE_PATH: Final = "runs/kra_context_rerank_v6_results.json"
DEFAULT_KRA_CONTEXT_RERANK_PATH: Final = REPO_ROOT / KRA_CONTEXT_RERANK_EVIDENCE_PATH


class CandidateManifestRow(TypedDict):
    candidate_id: str
    lane: str
    provenance: str
    decision: str
    reason: str
    evidence_path: str | None


class CandidateManifest(TypedDict):
    schema: str
    candidates: list[CandidateManifestRow]


class PromotionReport(TypedDict):
    promotion_pass: NotRequired[bool]


def _static_candidate_rows() -> list[CandidateManifestRow]:
    return [
        {
            "candidate_id": "kcycle_trifecta_ensemble_v1",
            "lane": "kcycle_trifecta",
            "provenance": "validation_only_baseline",
            "decision": "validation_only_baseline",
            "reason": "baseline_retained",
            "evidence_path": "data/kcycle_ensemble_gating_results.json",
        },
        {
            "candidate_id": "gen2_mut_436",
            "lane": "candidate_tournament",
            "provenance": "final_test_selected",
            "decision": "rejected",
            "reason": "selection_scope_must_be_train_val",
            "evidence_path": None,
        },
        {
            "candidate_id": "kcycle_trifecta_ensemble_v2_candidate",
            "lane": "kcycle_trifecta",
            "provenance": "audit_pending_candidate",
            "decision": "data_blocked",
            "reason": "audit_pending",
            "evidence_path": "data/kcycle_trifecta_snapshot_audit_latest.json",
        },
        {
            "candidate_id": "kcycle_timed_market_lane",
            "lane": "kcycle_timed_market",
            "provenance": "timed_market_lane",
            "decision": "data_blocked",
            "reason": "outcome_linked_timed_snapshot_unavailable",
            "evidence_path": "data/kcycle_market_timing_policy_results.json",
        },
    ]


def _promotion_report_row(
    *,
    candidate_id: str,
    lane: str,
    provenance: str,
    report_path: Path,
    evidence_path: str | None = None,
) -> CandidateManifestRow:
    displayed_evidence_path = evidence_path if evidence_path is not None else str(report_path)
    if not report_path.exists():
        return {
            "candidate_id": candidate_id,
            "lane": lane,
            "provenance": provenance,
            "decision": "data_blocked",
            "reason": "evidence_unavailable",
            "evidence_path": displayed_evidence_path,
        }

    try:
        parsed = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "candidate_id": candidate_id,
            "lane": lane,
            "provenance": provenance,
            "decision": "data_blocked",
            "reason": "evidence_unavailable",
            "evidence_path": displayed_evidence_path,
        }

    if not isinstance(parsed, dict):
        return {
            "candidate_id": candidate_id,
            "lane": lane,
            "provenance": provenance,
            "decision": "data_blocked",
            "reason": "evidence_unavailable",
            "evidence_path": displayed_evidence_path,
        }

    report: PromotionReport = parsed
    promotion_pass = report.get("promotion_pass")
    if promotion_pass is False:
        return {
            "candidate_id": candidate_id,
            "lane": lane,
            "provenance": provenance,
            "decision": "rejected",
            "reason": "promotion_pass_false",
            "evidence_path": displayed_evidence_path,
        }
    if promotion_pass is True:
        return {
            "candidate_id": candidate_id,
            "lane": lane,
            "provenance": provenance,
            "decision": "review_required",
            "reason": "promotion_pass_true_requires_separate_change",
            "evidence_path": displayed_evidence_path,
        }
    return {
        "candidate_id": candidate_id,
        "lane": lane,
        "provenance": provenance,
        "decision": "data_blocked",
        "reason": "evidence_unavailable",
        "evidence_path": displayed_evidence_path,
    }


def build_manifest(
    *,
    kra_meta_gate_path: Path = DEFAULT_KRA_META_GATE_PATH,
    kra_context_rerank_path: Path = DEFAULT_KRA_CONTEXT_RERANK_PATH,
) -> CandidateManifest:
    candidates = _static_candidate_rows()
    candidates.append(
        _promotion_report_row(
            candidate_id="kra_nested_meta_gate",
            lane="kra_top1",
            provenance="nested_meta_gate_report",
            report_path=kra_meta_gate_path,
        )
    )
    candidates.append(
        _promotion_report_row(
            candidate_id="kra_context_rerank",
            lane="kra_top1",
            provenance="context_rerank_report",
            report_path=kra_context_rerank_path,
            evidence_path=(
                KRA_CONTEXT_RERANK_EVIDENCE_PATH
                if kra_context_rerank_path == DEFAULT_KRA_CONTEXT_RERANK_PATH
                else None
            ),
        )
    )
    return {"schema": SCHEMA, "candidates": candidates}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify prediction candidates without changing production artifacts.",
    )
    parser.add_argument("--report", type=Path, help="Explicit JSON report path to write.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manifest()
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write(f"{rendered}\n")
    if args.report is not None:
        args.report.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
