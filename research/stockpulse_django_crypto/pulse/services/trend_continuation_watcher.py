from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import assert_never

from ..trend_continuation_models import TrendContinuationSelectionManifest
from .bitget_market import TextFetcher
from .trend_continuation_forward_collection import collect_manifest_forward_evidence
from .trend_continuation_manifest import JsonObject
from .trend_continuation_settlement import (
    capture_forward_observation,
    settle_ready_observations,
)
from .trend_continuation_watcher_lifecycle import (
    WatcherLifecyclePhase,
    advance_watcher_lifecycle,
    watcher_lifecycle_phase,
)


@dataclass(frozen=True, slots=True)
class WatcherCycleOutcome:
    collected_at: datetime
    manifests: tuple[JsonObject, ...]
    failure_codes: tuple[str, ...]

    @property
    def successful(self) -> bool:
        return not self.failure_codes

    def payload(self) -> JsonObject:
        return {
            "collected_at": self.collected_at.isoformat(),
            "manifests": list(self.manifests),
            "failure_codes": list(self.failure_codes),
        }


def run_watcher_cycle(
    fetch_text: TextFetcher,
    *,
    collected_at: datetime,
) -> WatcherCycleOutcome:
    outcomes: list[JsonObject] = []
    failures: list[str] = []
    manifests = TrendContinuationSelectionManifest.objects.order_by("frozen_at", "digest")
    for manifest in manifests:
        phase = watcher_lifecycle_phase(manifest)
        result, failure = _run_manifest_phase(manifest, phase, fetch_text, collected_at)
        outcomes.append(result)
        if failure is not None:
            failures.append(failure)
    return WatcherCycleOutcome(collected_at, tuple(outcomes), tuple(sorted(failures)))


def _run_manifest_phase(
    manifest: TrendContinuationSelectionManifest,
    phase: WatcherLifecyclePhase,
    fetch_text: TextFetcher,
    collected_at: datetime,
) -> tuple[JsonObject, str | None]:
    match phase:
        case WatcherLifecyclePhase.COLLECT_VALIDATION | WatcherLifecyclePhase.COLLECT_HOLDOUT:
            return _collect_phase(manifest, phase, fetch_text, collected_at)
        case WatcherLifecyclePhase.SETTLE_VALIDATION | WatcherLifecyclePhase.SETTLE_HOLDOUT:
            settled = settle_ready_observations(manifest, fetch_text, collected_at=collected_at)
            current = advance_watcher_lifecycle(manifest)
            return _manifest_payload(manifest, phase, current, False, settled, None), None
        case WatcherLifecyclePhase.FINALIZE_VALIDATION | WatcherLifecyclePhase.FINALIZE_HOLDOUT:
            current = advance_watcher_lifecycle(manifest)
            failure = "terminal_rejected" if current is WatcherLifecyclePhase.TERMINAL_REJECTED else None
            return _manifest_payload(manifest, phase, current, False, 0, failure), failure
        case WatcherLifecyclePhase.COMPLETE:
            return _manifest_payload(manifest, phase, phase, False, 0, None), None
        case WatcherLifecyclePhase.TERMINAL_REJECTED:
            return _manifest_payload(manifest, phase, phase, False, 0, "terminal_rejected"), "terminal_rejected"
        case unreachable:
            assert_never(unreachable)


def _collect_phase(
    manifest: TrendContinuationSelectionManifest,
    phase: WatcherLifecyclePhase,
    fetch_text: TextFetcher,
    collected_at: datetime,
) -> tuple[JsonObject, str | None]:
    evidence = collect_manifest_forward_evidence(
        manifest,
        fetch_text,
        collected_at=collected_at,
    )
    if evidence.draft is None:
        settled = settle_ready_observations(manifest, fetch_text, collected_at=collected_at)
        current = advance_watcher_lifecycle(manifest)
        return (
            _manifest_payload(
                manifest,
                phase,
                current,
                False,
                settled,
                evidence.rejection_code,
            ),
            None,
        )
    lookup_started_at = collected_at + timedelta(seconds=1)
    capture = capture_forward_observation(
        evidence.draft,
        fetch_text,
        watcher_collected_at=collected_at,
        settlement_lookup_started_at=lookup_started_at,
        settlement_lookup=lambda: settle_ready_observations(
            manifest,
            fetch_text,
            collected_at=collected_at,
        ),
    )
    settled = capture.settled_count
    if not capture.created:
        settled += settle_ready_observations(manifest, fetch_text, collected_at=collected_at)
    current = advance_watcher_lifecycle(manifest)
    failure = "terminal_rejected" if current is WatcherLifecyclePhase.TERMINAL_REJECTED else None
    return _manifest_payload(manifest, phase, current, capture.created, settled, failure), failure


def _manifest_payload(
    manifest: TrendContinuationSelectionManifest,
    phase_before: WatcherLifecyclePhase,
    phase_after: WatcherLifecyclePhase,
    observation_created: bool,
    settled_count: int,
    rejection_code: str | None,
) -> JsonObject:
    return {
        "manifest_digest": manifest.digest,
        "contract": manifest.contract_symbol,
        "phase_before": phase_before.value,
        "phase_after": phase_after.value,
        "observation_created": observation_created,
        "settled_count": settled_count,
        "rejection_code": rejection_code,
    }
