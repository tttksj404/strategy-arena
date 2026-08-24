from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from .trend_continuation_contract import CandidateIdentity
from .trend_continuation_evaluation_types import (
    MIN_SELECTION_OBSERVATIONS,
    MIN_WILSON_LOWER_95_PCT,
    STANDARD_EMBARGO,
    TRIPLE_EMBARGO,
    CandidateDevelopment,
    DevelopmentFold,
    EvaluationRow,
    HistoricalSelection,
    ObservationSource,
)
from .trend_continuation_features import STANDARD_PROFILE, TRIPLE_PROFILE
from .trend_continuation_statistics import evaluate_rows


@dataclass(frozen=True, slots=True)
class UnsupportedTrendProfileError(Exception):
    profile: str

    def __str__(self) -> str:
        return f"unsupported trend-continuation profile: {self.profile}"


@dataclass(frozen=True, slots=True)
class CandidateDevelopmentInputError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


def purge_crosses_boundary(row: EvaluationRow, boundary: datetime) -> bool:
    return row.evidence_time < boundary <= row.label_end_time


def stream_is_non_overlapping(rows: Sequence[EvaluationRow]) -> bool:
    ordered = tuple(sorted(rows, key=lambda row: (row.evidence_time, row.label_end_time)))
    return all(previous.label_end_time < current.evidence_time for previous, current in zip(ordered, ordered[1:]))


def profile_embargo(profile: str) -> timedelta:
    if profile == STANDARD_PROFILE:
        return STANDARD_EMBARGO
    if profile == TRIPLE_PROFILE:
        return TRIPLE_EMBARGO
    raise UnsupportedTrendProfileError(profile)


def build_development_folds(rows: Sequence[EvaluationRow], *, profile: str) -> tuple[DevelopmentFold, ...]:
    ordered = tuple(sorted(rows, key=lambda row: (row.evidence_time, row.label_end_time)))
    if len(ordered) < 10:
        return ()
    embargo = profile_embargo(profile)
    folds: list[DevelopmentFold] = []
    for start_pct in (50, 60, 70):
        start_index = len(ordered) * start_pct // 100
        end_index = len(ordered) * (start_pct + 10) // 100
        if start_index >= len(ordered) or end_index >= len(ordered):
            continue
        boundary = ordered[start_index].evidence_time
        next_boundary = ordered[end_index].evidence_time
        training = tuple(
            row
            for row in ordered[:start_index]
            if row.evidence_time < boundary and not purge_crosses_boundary(row, boundary)
        )
        validation = tuple(
            row
            for row in ordered[start_index:end_index]
            if boundary + embargo <= row.evidence_time < next_boundary
        )
        folds.append(DevelopmentFold(boundary, next_boundary, embargo, training, validation))
    return tuple(folds)


def build_candidate_development(
    identity: CandidateIdentity,
    rows: Sequence[EvaluationRow],
) -> CandidateDevelopment:
    ordered = tuple(sorted(rows, key=lambda row: (row.evidence_time, row.label_end_time)))
    if len(ordered) < 10:
        raise CandidateDevelopmentInputError("candidate development requires at least ten chronological rows")
    if any(row.source is not ObservationSource.HISTORICAL_EXPLORATORY for row in ordered):
        raise CandidateDevelopmentInputError("historical candidate selection accepts historical exploratory rows only")
    if not stream_is_non_overlapping(ordered):
        raise CandidateDevelopmentInputError("candidate development rows must be non-overlapping")
    folds = build_development_folds(ordered, profile=identity.profile)
    validation_rows = tuple(row for fold in folds for row in fold.validation)
    audit_index = len(ordered) * 80 // 100
    return CandidateDevelopment(
        identity=identity,
        metrics=evaluate_rows(validation_rows),
        split_boundaries=tuple(fold.boundary for fold in folds),
        historical_audit_start=ordered[audit_index].evidence_time,
    )


def select_historical_candidate(candidates: Sequence[CandidateDevelopment]) -> HistoricalSelection | None:
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.metrics.statistics_valid
        and candidate.metrics.resolved_count >= MIN_SELECTION_OBSERVATIONS
        and candidate.metrics.wilson_lower_95 >= MIN_WILSON_LOWER_95_PCT
        and candidate.metrics.mean_return_lower_95 > 0
    )
    if not eligible:
        return None
    selected = min(
        eligible,
        key=lambda candidate: (
            -candidate.metrics.wilson_lower_95,
            -candidate.metrics.mean_return_lower_95,
            -candidate.metrics.resolved_count,
            _candidate_lexical_key(candidate.identity),
        ),
    )
    return HistoricalSelection(
        identity=selected.identity,
        metrics=selected.metrics,
        split_boundaries=selected.split_boundaries,
        historical_audit_start=selected.historical_audit_start,
    )


def _candidate_lexical_key(identity: CandidateIdentity) -> tuple[str, str, str, str, int, str, str]:
    return (
        identity.contract,
        identity.profile,
        identity.trend_direction.value,
        identity.rule_name,
        identity.candidate_horizon_bars,
        identity.strategy_version,
        identity.feature_schema_version,
    )
