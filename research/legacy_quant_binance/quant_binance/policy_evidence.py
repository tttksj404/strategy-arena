from __future__ import annotations

from copy import deepcopy
from typing import Any


POLICY_EVIDENCE_BUCKET_NAMES = (
    "active_policy",
    "previous_policy",
    "staged_candidate",
    "baseline_control",
)


def _copy_payload(value: Any) -> Any:
    return deepcopy(value)


def _policy_evidence_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw_payload = dict(payload or {})
    if "policy_evidence_buckets" in raw_payload:
        return raw_payload
    policy_context = dict(raw_payload.get("policy_context", {}) or {})
    if "policy_evidence_buckets" in policy_context:
        return policy_context
    policy_validation = dict(raw_payload.get("policy_validation", {}) or {})
    validation_evidence = dict(policy_validation.get("evidence", {}) or {})
    if validation_evidence:
        return validation_evidence
    return raw_payload


def normalize_policy_evidence_buckets(payload: dict[str, Any] | None) -> dict[str, dict[str, object]]:
    raw_payload = _policy_evidence_payload(payload)
    raw_buckets = dict(raw_payload.get("policy_evidence_buckets", {}) or {})
    buckets: dict[str, dict[str, object]] = {}
    for bucket_name in POLICY_EVIDENCE_BUCKET_NAMES:
        bucket = dict(raw_buckets.get(bucket_name, {}) or {})
        buckets[bucket_name] = {
            **bucket,
            "evidence": dict(bucket.get("evidence", {}) or {}),
            "comparison": dict(bucket.get("comparison", {}) or {}),
            "policy_lineage": dict(bucket.get("policy_lineage", {}) or {}),
            "evidence_lineage": dict(bucket.get("evidence_lineage", {}) or {}),
            "alignment": dict(bucket.get("alignment", {}) or {}),
            "policy_application": dict(bucket.get("policy_application", {}) or {}),
            "replay_summary": dict(bucket.get("replay_summary", {}) or {}),
        }
    return buckets


def policy_evidence_bucket(payload: dict[str, Any] | None, bucket_name: str) -> dict[str, object]:
    return dict(normalize_policy_evidence_buckets(payload).get(bucket_name, {}) or {})


def policy_evidence_bucket_evidence(
    payload: dict[str, Any] | None,
    bucket_name: str,
    *,
    fallback_to_root: bool = False,
) -> dict[str, object]:
    bucket = policy_evidence_bucket(payload, bucket_name)
    evidence = dict(bucket.get("evidence", {}) or {})
    if evidence or not fallback_to_root:
        return evidence
    fallback = _policy_evidence_payload(payload)
    fallback = dict(fallback or {})
    fallback.pop("policy_evidence_buckets", None)
    fallback.pop("policy_context", None)
    return fallback


def baseline_control_bucket_comparison(payload: dict[str, Any] | None) -> dict[str, object]:
    bucket = policy_evidence_bucket(payload, "baseline_control")
    comparison = dict(bucket.get("comparison", {}) or {})
    if comparison:
        return comparison
    return dict(_policy_evidence_payload(payload).get("baseline_control_comparison", {}) or {})


def with_policy_evidence_buckets(
    payload: dict[str, Any] | None,
    buckets: dict[str, dict[str, object]] | None,
) -> dict[str, object]:
    updated = dict(payload or {})
    updated["policy_evidence_buckets"] = {
        str(bucket_name): _copy_payload(bucket_payload)
        for bucket_name, bucket_payload in dict(buckets or {}).items()
    }
    return updated


def replay_summary_provenance(summary: dict[str, Any] | None) -> dict[str, object]:
    """Classify the provenance of a replay summary for diagnostic purposes."""
    payload = dict(summary or {})
    source = str(payload.get("source", "") or "")
    bucket_name = str(payload.get("bucket_name", "") or "")
    runtime_anchor = dict(payload.get("runtime_summary_anchor", {}) or {})

    has_bucket_reference = bool(bucket_name and bucket_name != "not_available")
    anchor_source = str(runtime_anchor.get("source", "") or "")
    has_runtime_summary_anchor = bool(anchor_source and anchor_source != "artifact_only")

    uses_direct_bucket_replay = "bucket_replay" in source or has_bucket_reference
    uses_mixed_root_summary_fallback = "mixed" in source or "fallback" in source
    uses_projected_evidence = not uses_direct_bucket_replay

    if uses_direct_bucket_replay:
        classification = "bucket_replay_evidence"
        reason_code = "DIRECT_BUCKET_REPLAY_SOURCE"
        decision_surface = bucket_name or "policy_bucket"
    elif uses_projected_evidence:
        classification = "projected_evidence"
        reason_code = "PROJECTED_RUNTIME_REPLAY_SOURCE"
        decision_surface = "staged_candidate"
    else:
        classification = "runtime_artifact_evidence"
        reason_code = "RUNTIME_ARTIFACT_REPLAY_SOURCE"
        decision_surface = "runtime_artifact"

    policy_bucket = bucket_name if has_bucket_reference else "not_available"
    return {
        "classification": classification,
        "decision_surface": decision_surface,
        "has_bucket_reference": has_bucket_reference,
        "has_runtime_summary_anchor": has_runtime_summary_anchor,
        "policy_bucket": policy_bucket,
        "reason_code": reason_code,
        "source": source,
        "summary": f"{classification}:{reason_code}",
        "uses_direct_bucket_replay": uses_direct_bucket_replay,
        "uses_mixed_root_summary_fallback": uses_mixed_root_summary_fallback,
        "uses_projected_evidence": uses_projected_evidence,
    }


def baseline_control_replay_provenance(payload: dict[str, Any] | None) -> dict[str, object]:
    """Classify the provenance of a baseline control replay for diagnostic purposes."""
    p = dict(payload or {})
    source = str(p.get("evidence_source", "") or p.get("source", "") or "")
    current_bucket_replay = dict(p.get("current_policy_bucket_replay", {}) or {})
    staged_bucket_replay = dict(p.get("staged_candidate_bucket_replay", {}) or {})
    runtime_anchor = dict(p.get("runtime_summary_anchor", {}) or {})

    has_bucket_reference = bool(
        current_bucket_replay.get("available") or staged_bucket_replay.get("available")
    )
    anchor_source = str(runtime_anchor.get("source", "") or "")
    has_runtime_summary_anchor = bool(anchor_source and anchor_source != "artifact_only")
    uses_direct_bucket_replay = has_bucket_reference
    uses_mixed_root_summary_fallback = "mixed" in source or "fallback" in source
    uses_projected_evidence = not has_bucket_reference

    if uses_direct_bucket_replay:
        classification = "bucket_replay_evidence"
        reason_code = "BASELINE_BUCKET_REPLAY_SOURCE"
    else:
        classification = "projected_evidence"
        reason_code = "PROJECTED_BASELINE_REPLAY_SOURCE"

    policy_bucket = "baseline_control" if has_bucket_reference else "not_available"
    return {
        "classification": classification,
        "decision_surface": "baseline_control",
        "has_bucket_reference": has_bucket_reference,
        "has_runtime_summary_anchor": has_runtime_summary_anchor,
        "policy_bucket": policy_bucket,
        "reason_code": reason_code,
        "source": source,
        "summary": f"{classification}:{reason_code}",
        "uses_direct_bucket_replay": uses_direct_bucket_replay,
        "uses_mixed_root_summary_fallback": uses_mixed_root_summary_fallback,
        "uses_projected_evidence": uses_projected_evidence,
    }


def checkpoint_replay_provenance(info: dict[str, Any] | None) -> dict[str, object]:
    """Classify the provenance of a checkpoint replay for diagnostic purposes."""
    p = dict(info or {})
    evidence_source = str(p.get("evidence_source", "") or "")
    evidence_policy_bucket = str(p.get("evidence_policy_bucket", "not_available") or "not_available")

    has_bucket_reference = (
        evidence_source == "policy_bucket" and evidence_policy_bucket != "not_available"
    )
    uses_direct_bucket_replay = has_bucket_reference
    uses_mixed_root_summary_fallback = (
        evidence_source in {"root", "summary_artifact", "strategy_comparison_recent_summary"}
        or "mixed" in evidence_source
        or "fallback" in evidence_source
    )
    uses_projected_evidence = not has_bucket_reference and not uses_mixed_root_summary_fallback
    has_runtime_summary_anchor = (
        "runtime_summary" in evidence_source or "summary_artifact" in evidence_source
    )

    if uses_direct_bucket_replay:
        classification = "bucket_replay_evidence"
        reason_code = "CHECKPOINT_BUCKET_REPLAY_SOURCE"
        decision_surface = evidence_policy_bucket
    elif uses_mixed_root_summary_fallback:
        classification = "mixed_root_summary_fallback"
        reason_code = "root"
        decision_surface = "not_available"
    else:
        classification = "projected_evidence"
        reason_code = "PROJECTED_CHECKPOINT_REPLAY_SOURCE"
        decision_surface = "staged_candidate"

    return {
        "classification": classification,
        "decision_surface": decision_surface,
        "has_bucket_reference": has_bucket_reference,
        "has_runtime_summary_anchor": has_runtime_summary_anchor,
        "policy_bucket": evidence_policy_bucket,
        "reason_code": reason_code,
        "source": evidence_source,
        "summary": f"{classification}:{reason_code}",
        "uses_direct_bucket_replay": uses_direct_bucket_replay,
        "uses_mixed_root_summary_fallback": uses_mixed_root_summary_fallback,
        "uses_projected_evidence": uses_projected_evidence,
    }


def build_replay_provenance() -> dict[str, object]:
    """Return a stable fallback replay provenance payload.

    Several reporting paths use this as a no-data/default replay provenance.
    Keep the shape aligned with other provenance builders.
    """
    return {
        "classification": "mixed_root_summary_fallback",
        "decision_surface": "not_available",
        "has_bucket_reference": False,
        "has_runtime_summary_anchor": True,
        "policy_bucket": "not_available",
        "reason_code": "MIXED_ROOT_SUMMARY_FALLBACK",
        "source": "root",
        "summary": "mixed_root_summary_fallback:root",
        "uses_direct_bucket_replay": False,
        "uses_mixed_root_summary_fallback": True,
        "uses_projected_evidence": False,
    }
