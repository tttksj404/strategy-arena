from __future__ import annotations

from datetime import UTC, datetime, timedelta

from django.test import SimpleTestCase

from .services.long_forecast import Candle
from .services.market_timeframes import CandleGranularity
from .services.trend_continuation_contract import CandidateIdentity, Direction
from .services.trend_continuation_evaluation import (
    HISTORICAL_EXPLORATORY_COST_FLOOR_PCT,
    CandidateDevelopment,
    EvaluationRow,
    FeePolicySnapshot,
    ForwardQuoteSnapshot,
    FundingProof,
    FundingRateEvent,
    LabelRequest,
    ObservationSource,
    build_development_folds,
    build_candidate_development,
    compute_cost_pct,
    evaluate_label_window,
    evaluate_rows,
    forward_gate,
    mean_return_lower_95,
    purge_crosses_boundary,
    select_historical_candidate,
    stream_is_non_overlapping,
    wilson_lower_95,
)


BASE = datetime(2026, 1, 2, 9, tzinfo=UTC)


def _candle(opened_at: datetime, *, close: float = 101, high: float = 102, low: float = 100) -> Candle:
    return Candle(opened_at=opened_at, open=100, high=high, low=low, close=close, quote_volume=1_000)


def _fee_policy() -> FeePolicySnapshot:
    return FeePolicySnapshot(
        version="bitget-public-fee-v1",
        recorded_at=BASE - timedelta(days=2),
        source_url="https://www.bitget.com/api-doc/contract/market/Get-All-Symbols-Contracts",
        raw_sha256="a" * 64,
        effective_from=BASE - timedelta(days=1),
        effective_until=BASE + timedelta(days=1),
        taker_fee_rate=0.0006,
    )


def _funding(*events: FundingRateEvent, end: datetime | None = None) -> FundingProof:
    return FundingProof(
        complete=True,
        events=tuple(events),
        coverage_start=BASE,
        coverage_end=end or BASE + timedelta(days=1),
        provenance_digest="d" * 64,
        page_count=2,
        row_count=len(events),
    )


def _identity(contract: str, direction: Direction = Direction.UP, rule: str = "a") -> CandidateIdentity:
    return CandidateIdentity(
        contract=contract,
        profile="standard_15m_5m",
        trend_direction=direction,
        rule_name=rule,
        candidate_horizon_bars=2,
        strategy_version="trend-v1",
        feature_schema_version="features-v1",
    )


class LabelEvaluationTests(SimpleTestCase):
    def test_invalidation_wins_same_candle_but_return_uses_terminal_close(self):
        # Given: the first future candle breaches an UP invalidation, while both closes finish higher.
        evidence_time = BASE
        request = LabelRequest(
            evidence_time=evidence_time,
            trigger_interval=CandleGranularity.FIVE_MINUTES,
            horizon_bars=2,
            direction=Direction.UP,
            reference_price=100,
            invalidation_price=99,
            source=ObservationSource.HISTORICAL_EXPLORATORY,
            fee_policy=_fee_policy(),
            funding=_funding(end=evidence_time + timedelta(minutes=10)),
        )
        future = (
            _candle(evidence_time, close=103, high=104, low=98.9),
            _candle(evidence_time + timedelta(minutes=5), close=105, high=106, low=102),
        )

        # When: the exact closed label window is evaluated.
        row = evaluate_label_window(request, future, settled_as_of=evidence_time + timedelta(minutes=10))

        # Then: the breach makes the outcome fail, but terminal close and fixed historical cost remain auditable.
        self.assertTrue(row.resolved)
        self.assertFalse(row.outcome_success)
        self.assertEqual(row.terminal_close, 105)
        self.assertAlmostEqual(row.cost_pct or 0, HISTORICAL_EXPLORATORY_COST_FLOOR_PCT)
        self.assertAlmostEqual(row.net_return_pct or 0, 4.8)

    def test_missing_duplicate_or_noncontinuous_future_window_is_unresolved(self):
        # Given: two rows share the same close timestamp, so the required second bar is absent.
        request = LabelRequest(
            evidence_time=BASE,
            trigger_interval=CandleGranularity.FIVE_MINUTES,
            horizon_bars=2,
            direction=Direction.DOWN,
            reference_price=100,
            invalidation_price=103,
            source=ObservationSource.HISTORICAL_EXPLORATORY,
            fee_policy=_fee_policy(),
            funding=_funding(end=BASE + timedelta(minutes=10)),
        )
        duplicate = (_candle(BASE), _candle(BASE))

        # When: the label evaluator checks uniqueness and continuity before price outcomes.
        row = evaluate_label_window(request, duplicate, settled_as_of=BASE + timedelta(minutes=10))

        # Then: no metric-bearing result is manufactured from partial history.
        self.assertFalse(row.resolved)
        self.assertIsNone(row.outcome_success)
        self.assertEqual(row.rejection_code, "incomplete_label_window")
        self.assertEqual(evaluate_rows((row,)).resolved_count, 0)

    def test_exact_zero_net_return_is_a_resolved_failure(self):
        # Given: terminal gain equals the fixed 0.20 percent historical cost exactly.
        request = LabelRequest(
            evidence_time=BASE,
            trigger_interval=CandleGranularity.FIVE_MINUTES,
            horizon_bars=1,
            direction=Direction.UP,
            reference_price=100,
            invalidation_price=99,
            source=ObservationSource.HISTORICAL_EXPLORATORY,
            fee_policy=_fee_policy(),
            funding=_funding(end=BASE + timedelta(minutes=5)),
        )

        # When: the single complete future candle settles at 100.20.
        row = evaluate_label_window(
            request,
            (_candle(BASE, close=100.2, high=100.3, low=100),),
            settled_as_of=BASE + timedelta(minutes=5),
        )

        # Then: floating-point residue cannot turn a strict zero into success.
        self.assertTrue(row.resolved)
        self.assertEqual(row.net_return_pct, 0)
        self.assertFalse(row.outcome_success)

    def test_missing_policy_or_historical_quote_fails_closed(self):
        # Given: missing versioned fee evidence and a live quote on a historical source.
        funding = _funding(end=BASE + timedelta(minutes=10))
        quote = ForwardQuoteSnapshot(BASE, 99.9, 100.1, 0.2, "c" * 64, BASE + timedelta(seconds=1))

        # When: costs are requested across the invalid boundaries.
        missing_policy = compute_cost_pct(
            source=ObservationSource.HISTORICAL_EXPLORATORY,
            direction=Direction.UP,
            evidence_time=BASE,
            label_end_time=BASE + timedelta(minutes=10),
            fee_policy=None,
            funding=funding,
        )
        historical_quote = compute_cost_pct(
            source=ObservationSource.HISTORICAL_EXPLORATORY,
            direction=Direction.UP,
            evidence_time=BASE,
            label_end_time=BASE + timedelta(minutes=10),
            fee_policy=_fee_policy(),
            funding=funding,
            quote=quote,
        )

        # Then: neither path silently substitutes a zero cost.
        self.assertIsNone(missing_policy)
        self.assertIsNone(historical_quote)


class FixedStatisticsAndSplitTests(SimpleTestCase):
    def _row(self, minute: int, *, success: bool = True, value: float = 1.0) -> EvaluationRow:
        evidence = BASE + timedelta(minutes=minute)
        return EvaluationRow(
            evidence_time=evidence,
            label_end_time=evidence + timedelta(minutes=5),
            reference_price=100,
            invalidation_price=99,
            outcome_success=success,
            net_return_pct=value,
            terminal_close=101,
            cost_pct=0.2,
            source=ObservationSource.HISTORICAL_EXPLORATORY,
        )

    def test_fixed_z_vectors_and_exact_forward_counts_fail_closed(self):
        # Given: a known 60-of-80 success vector and nonconstant returns.
        values = tuple(float(value) for value in (1, 2, 3, 4))

        # When: fixed v1 confidence bounds and count gates are calculated.
        wilson = wilson_lower_95(60, 80)
        mean_lower = mean_return_lower_95(values)

        # Then: the exact numerical vectors are stable and only 80/20 are admissible.
        self.assertAlmostEqual(wilson, 64.5153253808, places=9)
        self.assertAlmostEqual(mean_lower or 0, 1.2348486881, places=9)
        passing = evaluate_rows(tuple(self._row(index * 10) for index in range(80)))
        self.assertTrue(forward_gate(passing, exact_count=80))
        self.assertFalse(forward_gate(passing, exact_count=79))
        self.assertFalse(forward_gate(evaluate_rows((self._row(0),)), exact_count=1))
        self.assertFalse(forward_gate(evaluate_rows(tuple(self._row(index * 10) for index in range(79))), exact_count=80))
        self.assertFalse(forward_gate(evaluate_rows(tuple(self._row(index * 10) for index in range(81))), exact_count=80))
        holdout = evaluate_rows(tuple(self._row(index * 10) for index in range(20)))
        self.assertTrue(forward_gate(holdout, exact_count=20))
        self.assertFalse(forward_gate(evaluate_rows(tuple(self._row(index * 10) for index in range(19))), exact_count=20))
        self.assertFalse(forward_gate(evaluate_rows(tuple(self._row(index * 10) for index in range(21))), exact_count=20))

    def test_closed_boundary_purge_embargo_and_stream_non_overlap_are_exact(self):
        # Given: a row crossing B, one ending before B, and intervals sharing an endpoint.
        boundary = BASE + timedelta(hours=1)
        crossing = EvaluationRow(
            evidence_time=boundary - timedelta(minutes=5),
            label_end_time=boundary,
            reference_price=100,
            invalidation_price=99,
            outcome_success=True,
            net_return_pct=1,
            terminal_close=101,
            cost_pct=0.2,
            source=ObservationSource.HISTORICAL_EXPLORATORY,
        )
        non_crossing = self._row(0)
        shared_endpoint = EvaluationRow(
            evidence_time=non_crossing.label_end_time,
            label_end_time=non_crossing.label_end_time + timedelta(minutes=5),
            reference_price=100,
            invalidation_price=99,
            outcome_success=True,
            net_return_pct=1,
            terminal_close=101,
            cost_pct=0.2,
            source=ObservationSource.HISTORICAL_EXPLORATORY,
        )

        # When / Then: the closed predicates reject exactly the shared boundary.
        self.assertTrue(purge_crosses_boundary(crossing, boundary))
        self.assertFalse(purge_crosses_boundary(non_crossing, boundary))
        self.assertFalse(stream_is_non_overlapping((non_crossing, shared_endpoint)))

        # And: standard folds apply the fixed 14-hour embargo and reserve 80-100% as audit-only.
        rows = tuple(self._row(index * 60) for index in range(200))
        folds = build_development_folds(rows, profile="standard_15m_5m")
        self.assertEqual(len(folds), 3)
        self.assertTrue(all(fold.embargo_duration == timedelta(hours=14) for fold in folds))
        self.assertTrue(all(row.evidence_time < rows[160].evidence_time for fold in folds for row in fold.validation))

    def test_historical_selection_uses_fixed_tie_break_and_ignores_audit_payload(self):
        # Given: equally scored candidates whose identity is the final deterministic tie-break.
        metrics = evaluate_rows(tuple(self._row(index * 10) for index in range(80)))
        a = CandidateDevelopment(_identity("AAAUSDT", rule="z"), metrics, (BASE,), BASE + timedelta(days=1))
        b = CandidateDevelopment(_identity("AAAUSDT", rule="a"), metrics, (BASE,), BASE + timedelta(days=1))

        # When: the exploratory selector freezes one candidate.
        selected = select_historical_candidate((a, b))

        # Then: lexicographic candidate identity wins and no historical result is promotable.
        self.assertIsNotNone(selected)
        self.assertEqual(selected.identity.rule_name, "a")
        self.assertTrue(selected.exploratory_only)
        self.assertFalse(selected.promotable)

        # And: changing only the chronological 80-100% audit partition cannot change development metrics.
        rows = tuple(self._row(index * 60) for index in range(500))
        audit_cut = len(rows) * 80 // 100
        mutated_audit = rows[:audit_cut] + tuple(
            self._row(index * 60, success=False, value=-100.0)
            for index in range(audit_cut, len(rows))
        )
        original_development = build_candidate_development(_identity("AAAUSDT"), rows)
        mutated_development = build_candidate_development(_identity("AAAUSDT"), mutated_audit)
        self.assertEqual(original_development.metrics, mutated_development.metrics)
        self.assertEqual(original_development.split_boundaries, mutated_development.split_boundaries)
        self.assertEqual(original_development.historical_audit_start, rows[audit_cut].evidence_time)
