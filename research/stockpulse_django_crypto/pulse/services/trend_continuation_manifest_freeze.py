from __future__ import annotations

import json
from datetime import datetime
from typing import cast

from django.core.exceptions import ValidationError

from ..trend_continuation_models import TrendContinuationSelectionManifest
from .trend_continuation_contract import Direction, stable_manifest_digest
from .trend_continuation_evaluation_types import FeePolicySnapshot
from .trend_continuation_manifest_types import (
    JsonObject,
    ManifestCandidate,
    TrendContinuationSelectionSnapshot,
)


def _candidate_payload(
    snapshot: TrendContinuationSelectionSnapshot,
    frozen_at: datetime,
) -> ManifestCandidate:
    candidate = snapshot.payload.get("candidate")
    if not isinstance(candidate, dict):
        raise ValidationError({"payload": "selection snapshot requires a candidate object"})
    required = (
        "contract",
        "profile",
        "trend_direction",
        "rule_name",
        "candidate_horizon_bars",
        "strategy_version",
        "feature_schema_version",
    )
    if any(name not in candidate for name in required):
        raise ValidationError({"payload": "candidate identity is incomplete"})
    if snapshot.payload.get("schema") != "trend-continuation-selection.v1":
        raise ValidationError({"payload": "selection schema must be trend-continuation-selection.v1"})
    if snapshot.payload.get("exploratory_only") is not True:
        raise ValidationError({"payload": "historical selection must remain exploratory_only"})
    raw_fee_policy = snapshot.payload.get("fee_policy")
    if not isinstance(raw_fee_policy, dict):
        raise ValidationError({"payload": "versioned fee policy snapshot is required"})
    try:
        fee_policy = FeePolicySnapshot(
            version=str(raw_fee_policy["version"]),
            recorded_at=datetime.fromisoformat(str(raw_fee_policy["recorded_at"])),
            source_url=str(raw_fee_policy["source_url"]),
            raw_sha256=str(raw_fee_policy["raw_sha256"]),
            effective_from=datetime.fromisoformat(str(raw_fee_policy["effective_from"])),
            effective_until=datetime.fromisoformat(str(raw_fee_policy["effective_until"])),
            taker_fee_rate=float(raw_fee_policy["taker_fee_rate"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError({"payload": "fee policy snapshot is malformed"}) from error
    if not fee_policy.valid_at(frozen_at):
        raise ValidationError({"payload": "fee policy is not valid at the manifest freeze time"})
    try:
        horizon = int(candidate["candidate_horizon_bars"])
        parsed_direction = Direction(str(candidate["trend_direction"]))
    except (TypeError, ValueError) as error:
        raise ValidationError({"payload": "candidate direction or horizon is invalid"}) from error
    if horizon <= 0:
        raise ValidationError({"payload": "candidate horizon must be positive"})
    if parsed_direction is Direction.NEUTRAL:
        raise ValidationError({"payload": "candidate direction cannot be neutral"})
    return ManifestCandidate(
        contract=str(candidate["contract"]),
        profile=str(candidate["profile"]),
        trend_direction=parsed_direction,
        rule_name=str(candidate["rule_name"]),
        candidate_horizon_bars=horizon,
        strategy_version=str(candidate["strategy_version"]),
        feature_schema_version=str(candidate["feature_schema_version"]),
    )


def freeze_selection(
    snapshot: TrendContinuationSelectionSnapshot,
    *,
    frozen_at: datetime,
) -> TrendContinuationSelectionManifest:
    if frozen_at.tzinfo is None:
        raise ValidationError({"frozen_at": "manifest freeze time must be timezone-aware"})
    candidate = _candidate_payload(snapshot, frozen_at)
    payload = cast(JsonObject, json.loads(json.dumps(snapshot.payload, sort_keys=True, separators=(",", ":"))))
    return TrendContinuationSelectionManifest.objects.create(
        digest=stable_manifest_digest(payload),
        payload=payload,
        frozen_at=frozen_at,
        contract_symbol=candidate.contract,
        profile=candidate.profile,
        trend_direction=candidate.trend_direction.value,
        rule_name=candidate.rule_name,
        candidate_horizon_bars=candidate.candidate_horizon_bars,
        strategy_version=candidate.strategy_version,
        feature_schema_version=candidate.feature_schema_version,
        exploratory_only=True,
    )
