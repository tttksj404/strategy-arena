from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_iso8601(value: object) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_adjustments(
    adjustments: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in list(adjustments or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "symbol": str(item.get("symbol", "") or ""),
                "action": str(item.get("action", "keep") or "keep"),
                "size_multiplier": round(_safe_float(item.get("size_multiplier"), 1.0), 6),
                "leverage_multiplier": round(_safe_float(item.get("leverage_multiplier"), 1.0), 6),
                "entry_threshold_bps": round(_safe_float(item.get("entry_threshold_bps")), 6),
                "expected_profit_floor_bps": round(_safe_float(item.get("expected_profit_floor_bps")), 6),
                "symbol_bias": str(item.get("symbol_bias", "neutral") or "neutral"),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["symbol"]),
            str(row["action"]),
            float(row["size_multiplier"]),
            float(row["leverage_multiplier"]),
            float(row["entry_threshold_bps"]),
            float(row["expected_profit_floor_bps"]),
            str(row["symbol_bias"]),
        )
    )
    return rows


def _profile_payload(profile: dict[str, Any] | None) -> dict[str, object]:
    payload = dict(profile or {})
    symbols = sorted(str(symbol) for symbol in list(payload.get("symbols", []) or []) if str(symbol))
    return {
        "symbols": symbols,
        "adjustment_count": max(_safe_int(payload.get("adjustment_count"), 0) or 0, len(symbols)),
        "promote_count": max(_safe_int(payload.get("promote_count"), 0) or 0, 0),
        "aggressive_promote_count": max(_safe_int(payload.get("aggressive_promote_count"), 0) or 0, 0),
        "demote_count": max(_safe_int(payload.get("demote_count"), 0) or 0, 0),
        "disabled_count": max(_safe_int(payload.get("disabled_count"), 0) or 0, 0),
        "avg_size_multiplier": round(_safe_float(payload.get("avg_size_multiplier"), 1.0), 6),
        "avg_leverage_multiplier": round(_safe_float(payload.get("avg_leverage_multiplier"), 1.0), 6),
        "avg_entry_threshold_bps": round(_safe_float(payload.get("avg_entry_threshold_bps")), 6),
        "avg_expected_profit_floor_bps": round(_safe_float(payload.get("avg_expected_profit_floor_bps")), 6),
        "net_size_delta": round(_safe_float(payload.get("net_size_delta")), 6),
        "net_leverage_delta": round(_safe_float(payload.get("net_leverage_delta")), 6),
    }


def infer_policy_status_from_profile(profile: dict[str, Any] | None) -> str:
    payload = _profile_payload(profile)
    if int(payload["disabled_count"]) > 0:
        return "disabled"
    if int(payload["promote_count"]) > 0 or int(payload["aggressive_promote_count"]) > 0:
        return "promote"
    if int(payload["demote_count"]) > 0:
        return "demote"
    if int(payload["adjustment_count"]) <= 0:
        return "baseline"
    return "keep"


def build_policy_lineage_snapshot(
    *,
    policy: dict[str, Any] | None = None,
    rollout_phase: str = "",
    policy_status: str = "",
    version: object = None,
    updated_at: object = "",
    source: str = "",
) -> dict[str, object]:
    payload = dict(policy or {})
    adjustments = _normalize_adjustments(payload.get("adjustments", []))
    normalized_status = str(policy_status or payload.get("status", "") or ("baseline" if not adjustments else "keep"))
    normalized_phase = str(rollout_phase or dict(payload.get("rollout_progression", {}) or {}).get("execution_phase", "") or "")
    if not normalized_phase:
        normalized_phase = "baseline" if not adjustments else "full"
    symbols = sorted(str(item["symbol"]) for item in adjustments if str(item["symbol"]))
    adjustment_signature = _stable_hash(adjustments)
    structural_basis = {
        "policy_status": normalized_status,
        "rollout_phase": normalized_phase,
        "adjustment_count": len(adjustments),
        "symbols": symbols,
        "adjustment_signature": adjustment_signature,
    }
    structural_key = _stable_hash(structural_basis)
    version_value = _safe_int(version)
    versioned_key = _stable_hash({"structural_key": structural_key, "version": version_value}) if version_value is not None else ""
    return {
        "available": bool(adjustments or normalized_status or normalized_phase or version_value is not None),
        "source": str(source or ""),
        "basis": "adjustments",
        "policy_status": normalized_status,
        "rollout_phase": normalized_phase,
        "version": version_value,
        "updated_at": str(updated_at or ""),
        "adjustment_count": len(adjustments),
        "symbols": symbols,
        "adjustment_signature": adjustment_signature,
        "structural_key": structural_key,
        "versioned_key": versioned_key,
    }


def build_policy_profile_lineage_snapshot(
    *,
    policy_profile: dict[str, Any] | None,
    policy_status: str = "",
    updated_at: object = "",
    source: str = "",
) -> dict[str, object]:
    payload = dict(policy_profile or {})
    normalized_status = str(policy_status or infer_policy_status_from_profile(payload))
    normalized_phase = str(payload.get("rollout_phase", "") or ("baseline" if _safe_int(payload.get("adjustment_count"), 0) in {None, 0} else "full"))
    profile_basis = _profile_payload(payload)
    structural_basis = {
        "policy_status": normalized_status,
        "rollout_phase": normalized_phase,
        **profile_basis,
    }
    return {
        "available": bool(payload),
        "source": str(source or ""),
        "basis": "profile",
        "policy_status": normalized_status,
        "rollout_phase": normalized_phase,
        "version": None,
        "updated_at": str(updated_at or ""),
        "adjustment_count": int(profile_basis["adjustment_count"]),
        "symbols": list(profile_basis["symbols"]),
        "adjustment_signature": _stable_hash(profile_basis),
        "structural_key": _stable_hash(structural_basis),
        "versioned_key": "",
    }


def build_policy_state_lineage_snapshot(
    policy_state: dict[str, Any] | None,
    *,
    source: str = "policy_state",
) -> dict[str, object]:
    payload = dict(policy_state or {})
    active_policy = dict(payload.get("active_policy", {}) or {})
    active_adjustments = list(active_policy.get("adjustments", []) or [])
    rollout_progression = dict(payload.get("rollout_progression", dict(active_policy.get("rollout_progression", {}) or {})) or {})
    rollout_phase = str(rollout_progression.get("execution_phase", "") or "")
    if not rollout_phase:
        rollout_phase = "baseline" if not active_adjustments else "full"
    policy_status = str(
        active_policy.get("status")
        or payload.get("status")
        or ("baseline" if not active_adjustments else "keep")
    )
    return build_policy_lineage_snapshot(
        policy=active_policy,
        rollout_phase=rollout_phase,
        policy_status=policy_status,
        version=payload.get("version"),
        updated_at=payload.get("updated_at", ""),
        source=source,
    )


def policy_lineage_alignment(
    expected: dict[str, Any] | None,
    observed: dict[str, Any] | None,
    *,
    max_age_days: int = 10,
) -> dict[str, object]:
    expected_payload = dict(expected or {})
    observed_payload = dict(observed or {})
    if not bool(expected_payload.get("available")):
        return {
            "aligned": False,
            "status": "no_expected_lineage",
            "reason": "NO_EXPECTED_POLICY_LINEAGE",
        }
    if not bool(observed_payload.get("available")):
        return {
            "aligned": False,
            "status": "unknown",
            "reason": "NO_OBSERVED_POLICY_LINEAGE",
        }
    expected_updated_at = parse_iso8601(expected_payload.get("updated_at"))
    observed_updated_at = parse_iso8601(observed_payload.get("updated_at"))
    fresh = True
    if expected_updated_at is not None and observed_updated_at is not None:
        fresh = abs(expected_updated_at - observed_updated_at) <= timedelta(days=max_age_days)
    version_match = True
    expected_version = _safe_int(expected_payload.get("version"))
    observed_version = _safe_int(observed_payload.get("version"))
    if expected_version is not None and observed_version is not None:
        version_match = expected_version == observed_version
    structural_match = (
        bool(expected_payload.get("structural_key"))
        and expected_payload.get("structural_key") == observed_payload.get("structural_key")
    )
    if structural_match and fresh:
        return {
            "aligned": True,
            "status": "aligned",
            "reason": "POLICY_LINEAGE_MATCH",
        }
    if structural_match:
        return {
            "aligned": False,
            "status": "stale",
            "reason": "POLICY_LINEAGE_STALE",
        }
    if not version_match:
        return {
            "aligned": False,
            "status": "mismatch",
            "reason": "POLICY_VERSION_MISMATCH",
        }
    return {
        "aligned": False,
        "status": "mismatch",
        "reason": "POLICY_LINEAGE_MISMATCH",
    }
