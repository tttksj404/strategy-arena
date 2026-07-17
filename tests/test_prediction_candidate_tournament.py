from __future__ import annotations

import dataclasses
import unittest
from datetime import UTC, datetime, timedelta, timezone

from tools.prediction_candidate_tournament import (
    CandidateContract,
    ContractValidationError,
    EvaluationWindow,
    LaneContract,
    RaceObservation,
    evaluate_locked_test_candidate,
    screen_candidate,
    score_candidate,
)


TOURNAMENT_STARTED_AT = datetime(2026, 7, 17, 9, 0, tzinfo=UTC)
KST = timezone(timedelta(hours=9))


def _lane() -> LaneContract:
    return LaneContract(
        lane_id="kra_prestart",
        tournament_started_at_utc=TOURNAMENT_STARTED_AT,
        max_candidate_age=timedelta(minutes=30),
    )


def _candidate(**overrides) -> CandidateContract:
    values = {
        "candidate_id": "candidate_a",
        "lane_id": "kra_prestart",
        "selection_scope": "val",
        "generated_at_utc": TOURNAMENT_STARTED_AT - timedelta(minutes=5),
    }
    values.update(overrides)
    return CandidateContract(**values)


def _evaluation_window(**overrides) -> EvaluationWindow:
    values = {
        "train_ended_at_utc": datetime(2026, 7, 17, 7, 0, tzinfo=UTC),
        "validation_ended_at_utc": datetime(2026, 7, 17, 8, 30, tzinfo=UTC),
        "locked_test_started_at_utc": TOURNAMENT_STARTED_AT,
        "selection_cutoff_utc": datetime(2026, 7, 17, 8, 45, tzinfo=UTC),
    }
    values.update(overrides)
    return EvaluationWindow(**values)


def _observation(**overrides) -> RaceObservation:
    values = {
        "observed_at_utc": TOURNAMENT_STARTED_AT,
        "baseline_hit": True,
        "candidate_hit": True,
        "candidate_probability": 0.75,
    }
    values.update(overrides)
    return RaceObservation(**values)


class PredictionCandidateTournamentTestCase(unittest.TestCase):
    def test_candidate_and_lane_contracts_are_immutable(self) -> None:
        candidate = _candidate()
        lane = _lane()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            candidate.selection_scope = "test"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            lane.lane_id = "other_lane"

    def test_rejects_test_selection_scope(self) -> None:
        decision = screen_candidate(_candidate(selection_scope="test"), _lane())

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "selection_scope_must_be_train_val")

    def test_contracts_require_timezone_aware_utc_datetimes(self) -> None:
        invalid_datetimes = (
            (
                "lane_naive_start",
                lambda: LaneContract(
                    lane_id="kra_prestart",
                    tournament_started_at_utc=datetime(2026, 7, 17, 9, 0),
                    max_candidate_age=timedelta(minutes=30),
                ),
            ),
            (
                "lane_non_utc_start",
                lambda: LaneContract(
                    lane_id="kra_prestart",
                    tournament_started_at_utc=datetime(2026, 7, 17, 18, 0, tzinfo=KST),
                    max_candidate_age=timedelta(minutes=30),
                ),
            ),
            (
                "candidate_naive_generated_at",
                lambda: _candidate(generated_at_utc=datetime(2026, 7, 17, 8, 55)),
            ),
            (
                "candidate_non_utc_generated_at",
                lambda: _candidate(
                    generated_at_utc=datetime(2026, 7, 17, 17, 55, tzinfo=KST)
                ),
            ),
        )

        for expected_reason, build_contract in invalid_datetimes:
            with self.subTest(expected_reason=expected_reason):
                with self.assertRaises(ContractValidationError) as raised:
                    build_contract()
                self.assertEqual(raised.exception.reason, expected_reason)

    def test_lane_contract_requires_positive_candidate_age(self) -> None:
        for invalid_age in (timedelta(0), -timedelta(seconds=1)):
            with self.subTest(invalid_age=invalid_age):
                with self.assertRaises(ContractValidationError) as raised:
                    LaneContract(
                        lane_id="kra_prestart",
                        tournament_started_at_utc=TOURNAMENT_STARTED_AT,
                        max_candidate_age=invalid_age,
                    )
                self.assertEqual(raised.exception.reason, "max_candidate_age_must_be_positive")

    def test_rejects_unsafe_candidates_before_metrics_are_computed(self) -> None:
        unsafe_cases = (
            (
                "post_start_candidate",
                _candidate(generated_at_utc=TOURNAMENT_STARTED_AT + timedelta(seconds=1)),
            ),
            (
                "stale_candidate",
                _candidate(generated_at_utc=TOURNAMENT_STARTED_AT - timedelta(minutes=31)),
            ),
            ("cross_lane_candidate", _candidate(lane_id="kcycle_prestart")),
        )

        for expected_reason, candidate in unsafe_cases:
            with self.subTest(expected_reason=expected_reason):
                metric_calls = 0

                def metric(_candidate: CandidateContract) -> float:
                    nonlocal metric_calls
                    metric_calls += 1
                    return 1.0

                decision = score_candidate(candidate, _lane(), metric)

                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, expected_reason)
                self.assertIsNone(decision.score)
                self.assertEqual(metric_calls, 0)

    def test_rejects_candidate_generated_exactly_at_tournament_start(self) -> None:
        decision = screen_candidate(
            _candidate(generated_at_utc=TOURNAMENT_STARTED_AT),
            _lane(),
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "post_start_candidate")

    def test_allows_candidate_on_stale_boundary(self) -> None:
        decision = score_candidate(
            _candidate(generated_at_utc=TOURNAMENT_STARTED_AT - timedelta(minutes=30)),
            _lane(),
            lambda _: 3.0,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "accepted")
        self.assertEqual(decision.score, 3.0)

    def test_scores_safe_train_val_candidate(self) -> None:
        decision = score_candidate(_candidate(selection_scope="train"), _lane(), lambda _: 2.5)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "accepted")
        self.assertEqual(decision.score, 2.5)

    def test_race_observation_contract_is_immutable_and_utc_probability_bounded(self) -> None:
        observation = _observation()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            observation.candidate_probability = 0.25

        invalid_observations = (
            (
                "observation_naive_observed_at",
                lambda: _observation(observed_at_utc=datetime(2026, 7, 17, 9, 0)),
            ),
            (
                "observation_non_utc_observed_at",
                lambda: _observation(
                    observed_at_utc=datetime(2026, 7, 17, 18, 0, tzinfo=KST)
                ),
            ),
            (
                "candidate_probability_out_of_bounds",
                lambda: _observation(candidate_probability=1.01),
            ),
        )

        for expected_reason, build_observation in invalid_observations:
            with self.subTest(expected_reason=expected_reason):
                with self.assertRaises(ContractValidationError) as raised:
                    build_observation()
                self.assertEqual(raised.exception.reason, expected_reason)

    def test_rejects_selection_cutoff_that_reaches_locked_test_start(self) -> None:
        report = evaluate_locked_test_candidate(
            observations=(_observation(),),
            window=_evaluation_window(selection_cutoff_utc=TOURNAMENT_STARTED_AT),
            threshold=0.7,
        )

        self.assertFalse(report.accepted)
        self.assertEqual(report.reason, "selection_cutoff_must_precede_locked_test")
        self.assertIsNone(report.baseline_accuracy)
        self.assertIsNone(report.candidate_accuracy)


class PredictionCandidateChronologicalEvaluatorTestCase(unittest.TestCase):
    def test_reports_baseline_selective_candidate_calibration_coverage_and_lift(self) -> None:
        observations = (
            _observation(
                observed_at_utc=TOURNAMENT_STARTED_AT - timedelta(seconds=1),
                baseline_hit=False,
                candidate_hit=False,
                candidate_probability=0.99,
            ),
            _observation(
                observed_at_utc=TOURNAMENT_STARTED_AT,
                baseline_hit=True,
                candidate_hit=True,
                candidate_probability=0.8,
            ),
            _observation(
                observed_at_utc=TOURNAMENT_STARTED_AT + timedelta(minutes=1),
                baseline_hit=False,
                candidate_hit=True,
                candidate_probability=0.7,
            ),
            _observation(
                observed_at_utc=TOURNAMENT_STARTED_AT + timedelta(minutes=2),
                baseline_hit=True,
                candidate_hit=False,
                candidate_probability=0.2,
            ),
        )

        report = evaluate_locked_test_candidate(
            observations=observations,
            window=_evaluation_window(),
            threshold=0.7,
        )

        self.assertTrue(report.accepted)
        self.assertEqual(report.reason, "accepted")
        self.assertEqual(report.locked_test_rows, 3)
        self.assertEqual(report.selected_rows, 2)
        self.assertAlmostEqual(report.baseline_accuracy, 2 / 3)
        self.assertAlmostEqual(report.selected_baseline_accuracy, 0.5)
        self.assertAlmostEqual(report.candidate_accuracy, 1.0)
        self.assertAlmostEqual(report.coverage, 2 / 3)
        self.assertAlmostEqual(report.calibration_loss, ((1 - 0.8) ** 2 + (1 - 0.7) ** 2) / 2)
        self.assertAlmostEqual(report.candidate_vs_baseline_lift, 0.5)

    def test_rejects_when_threshold_selects_no_locked_test_rows(self) -> None:
        report = evaluate_locked_test_candidate(
            observations=(
                _observation(candidate_probability=0.69),
                _observation(
                    observed_at_utc=TOURNAMENT_STARTED_AT + timedelta(minutes=1),
                    candidate_probability=0.2,
                ),
            ),
            window=_evaluation_window(),
            threshold=0.7,
        )

        self.assertFalse(report.accepted)
        self.assertEqual(report.reason, "no_selected_locked_test_rows")
        self.assertEqual(report.locked_test_rows, 2)
        self.assertEqual(report.selected_rows, 0)
        self.assertAlmostEqual(report.baseline_accuracy, 1.0)
        self.assertIsNone(report.selected_baseline_accuracy)
        self.assertIsNone(report.candidate_accuracy)
