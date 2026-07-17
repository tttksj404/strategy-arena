from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal, assert_never


SelectionScope = Literal["train", "val", "test"]
ACCEPTED: Final = "accepted"


class ContractValidationError(ValueError):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _utc_validation_reason(
    value: datetime,
    naive_reason: str,
    non_utc_reason: str,
) -> str | None:
    offset = value.utcoffset()
    if offset is None:
        return naive_reason
    if offset != timedelta(0):
        return non_utc_reason
    return None


@dataclass(frozen=True, slots=True)
class LaneContract:
    lane_id: str
    tournament_started_at_utc: datetime
    max_candidate_age: timedelta

    def __post_init__(self) -> None:
        reason = _utc_validation_reason(
            self.tournament_started_at_utc,
            "lane_naive_start",
            "lane_non_utc_start",
        )
        if reason is not None:
            raise ContractValidationError(reason)
        if self.max_candidate_age <= timedelta(0):
            raise ContractValidationError("max_candidate_age_must_be_positive")


@dataclass(frozen=True, slots=True)
class CandidateContract:
    candidate_id: str
    lane_id: str
    selection_scope: SelectionScope
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        reason = _utc_validation_reason(
            self.generated_at_utc,
            "candidate_naive_generated_at",
            "candidate_non_utc_generated_at",
        )
        if reason is not None:
            raise ContractValidationError(reason)


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    accepted: bool
    reason: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RaceObservation:
    observed_at_utc: datetime
    baseline_hit: bool
    candidate_hit: bool
    candidate_probability: float

    def __post_init__(self) -> None:
        reason = _utc_validation_reason(
            self.observed_at_utc,
            "observation_naive_observed_at",
            "observation_non_utc_observed_at",
        )
        if reason is not None:
            raise ContractValidationError(reason)
        if not 0 <= self.candidate_probability <= 1:
            raise ContractValidationError("candidate_probability_out_of_bounds")


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    train_ended_at_utc: datetime
    validation_ended_at_utc: datetime
    locked_test_started_at_utc: datetime
    selection_cutoff_utc: datetime

    def __post_init__(self) -> None:
        utc_fields = (
            (self.train_ended_at_utc, "train_naive_end", "train_non_utc_end"),
            (self.validation_ended_at_utc, "validation_naive_end", "validation_non_utc_end"),
            (self.locked_test_started_at_utc, "locked_test_naive_start", "locked_test_non_utc_start"),
            (self.selection_cutoff_utc, "selection_cutoff_naive", "selection_cutoff_non_utc"),
        )
        for value, naive_reason, non_utc_reason in utc_fields:
            reason = _utc_validation_reason(value, naive_reason, non_utc_reason)
            if reason is not None:
                raise ContractValidationError(reason)
        if self.train_ended_at_utc > self.validation_ended_at_utc:
            raise ContractValidationError("train_end_after_validation_end")
        if self.validation_ended_at_utc >= self.locked_test_started_at_utc:
            raise ContractValidationError("validation_end_must_precede_locked_test")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    accepted: bool
    reason: str
    locked_test_rows: int
    selected_rows: int
    baseline_accuracy: float | None
    selected_baseline_accuracy: float | None
    candidate_accuracy: float | None
    coverage: float | None
    calibration_loss: float | None
    candidate_vs_baseline_lift: float | None


def screen_candidate(candidate: CandidateContract, lane: LaneContract) -> CandidateDecision:
    match candidate.selection_scope:
        case "train" | "val":
            pass
        case "test":
            return CandidateDecision(False, "selection_scope_must_be_train_val")
        case unreachable:
            assert_never(unreachable)

    if candidate.generated_at_utc >= lane.tournament_started_at_utc:
        return CandidateDecision(False, "post_start_candidate")
    if candidate.generated_at_utc < lane.tournament_started_at_utc - lane.max_candidate_age:
        return CandidateDecision(False, "stale_candidate")
    if candidate.lane_id != lane.lane_id:
        return CandidateDecision(False, "cross_lane_candidate")
    return CandidateDecision(True, ACCEPTED)


def score_candidate(
    candidate: CandidateContract,
    lane: LaneContract,
    metric: Callable[[CandidateContract], float],
) -> CandidateDecision:
    decision = screen_candidate(candidate, lane)
    if not decision.accepted:
        return decision
    return CandidateDecision(True, ACCEPTED, metric(candidate))


def _accuracy(observations: Sequence[RaceObservation], *, candidate: bool) -> float:
    hits = sum(
        observation.candidate_hit if candidate else observation.baseline_hit
        for observation in observations
    )
    return hits / len(observations)


def _calibration_loss(observations: Sequence[RaceObservation]) -> float:
    return sum(
        (float(observation.candidate_hit) - observation.candidate_probability) ** 2
        for observation in observations
    ) / len(observations)


def evaluate_locked_test_candidate(
    observations: Sequence[RaceObservation],
    window: EvaluationWindow,
    threshold: float,
) -> EvaluationReport:
    if not 0 <= threshold <= 1:
        raise ContractValidationError("threshold_out_of_bounds")

    locked_test_observations = tuple(
        observation
        for observation in observations
        if observation.observed_at_utc >= window.locked_test_started_at_utc
    )
    selected_observations = tuple(
        observation
        for observation in locked_test_observations
        if observation.candidate_probability >= threshold
    )

    if window.selection_cutoff_utc >= window.locked_test_started_at_utc:
        return EvaluationReport(
            accepted=False,
            reason="selection_cutoff_must_precede_locked_test",
            locked_test_rows=len(locked_test_observations),
            selected_rows=len(selected_observations),
            baseline_accuracy=None,
            selected_baseline_accuracy=None,
            candidate_accuracy=None,
            coverage=None,
            calibration_loss=None,
            candidate_vs_baseline_lift=None,
        )
    if len(locked_test_observations) == 0:
        return EvaluationReport(
            accepted=False,
            reason="no_locked_test_rows",
            locked_test_rows=0,
            selected_rows=0,
            baseline_accuracy=None,
            selected_baseline_accuracy=None,
            candidate_accuracy=None,
            coverage=None,
            calibration_loss=None,
            candidate_vs_baseline_lift=None,
        )

    baseline_accuracy = _accuracy(locked_test_observations, candidate=False)
    if len(selected_observations) == 0:
        return EvaluationReport(
            accepted=False,
            reason="no_selected_locked_test_rows",
            locked_test_rows=len(locked_test_observations),
            selected_rows=0,
            baseline_accuracy=baseline_accuracy,
            selected_baseline_accuracy=None,
            candidate_accuracy=None,
            coverage=0.0,
            calibration_loss=None,
            candidate_vs_baseline_lift=None,
        )

    candidate_accuracy = _accuracy(selected_observations, candidate=True)
    selected_baseline_accuracy = _accuracy(selected_observations, candidate=False)
    return EvaluationReport(
        accepted=True,
        reason=ACCEPTED,
        locked_test_rows=len(locked_test_observations),
        selected_rows=len(selected_observations),
        baseline_accuracy=baseline_accuracy,
        selected_baseline_accuracy=selected_baseline_accuracy,
        candidate_accuracy=candidate_accuracy,
        coverage=len(selected_observations) / len(locked_test_observations),
        calibration_loss=_calibration_loss(selected_observations),
        candidate_vs_baseline_lift=candidate_accuracy - selected_baseline_accuracy,
    )
