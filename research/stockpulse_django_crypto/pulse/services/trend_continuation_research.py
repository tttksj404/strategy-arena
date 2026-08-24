"""Research-only public facade for historical evaluation and forward capture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .trend_continuation_fixture import (
    HistoricalResearchError,
    HistoricalResearchRequest,
)
from .trend_continuation_historical_research import (
    HistoricalResearchOutcome,
    eligible_local_us_symbols,
    evaluate_historical_research,
)
from .trend_continuation_historical_selection import freeze_selection_idempotently
from .trend_continuation_manifest import JsonObject
from .trend_continuation_settlement import (
    ForwardObservationCapture,
    ForwardObservationDraft,
    capture_forward_observation,
)


@dataclass(frozen=True, slots=True)
class TrendContinuationReport:
    as_of: datetime
    diagnostics: tuple[JsonObject, ...]
    rejections: tuple[tuple[str, int], ...]

    def payload(self) -> JsonObject:
        return {
            "mode": "research_only",
            "execution": {"enabled": False},
            "as_of": self.as_of.isoformat(),
            "diagnostics": list(self.diagnostics),
            "rejections": [
                {"reason": reason, "count": count} for reason, count in self.rejections
            ],
        }


def build_trend_continuation_report(now: datetime | None = None) -> TrendContinuationReport:
    reference = now or datetime.now(UTC)
    return TrendContinuationReport(reference, (), (("forward_watcher_required", 1),))


__all__ = [
    "ForwardObservationCapture",
    "ForwardObservationDraft",
    "HistoricalResearchError",
    "HistoricalResearchOutcome",
    "HistoricalResearchRequest",
    "TrendContinuationReport",
    "build_trend_continuation_report",
    "capture_forward_observation",
    "eligible_local_us_symbols",
    "evaluate_historical_research",
    "freeze_selection_idempotently",
]
