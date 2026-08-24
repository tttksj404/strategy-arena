"""Stable public facade for manifest freezing and forward-result lifecycle."""

from .trend_continuation_consumption import register_forward_consumption
from .trend_continuation_lifecycle import append_holdout_result, append_validation_result
from .trend_continuation_manifest_freeze import freeze_selection
from .trend_continuation_manifest_types import (
    ForwardIntervalOverlapError,
    JsonObject,
    JsonScalar,
    JsonValue,
    ManifestCandidate,
    ManifestLifecycleError,
    TrendContinuationSelectionSnapshot,
)

__all__ = [
    "ForwardIntervalOverlapError",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "ManifestCandidate",
    "ManifestLifecycleError",
    "TrendContinuationSelectionSnapshot",
    "append_holdout_result",
    "append_validation_result",
    "freeze_selection",
    "register_forward_consumption",
]
