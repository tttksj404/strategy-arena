from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from django.db import close_old_connections
from django.test import TransactionTestCase

from .services.trend_continuation_manifest import (
    ForwardIntervalOverlapError,
    freeze_selection,
    register_forward_consumption,
)
from .test_trend_continuation_manifest import BASE, _identity, _snapshot
from .trend_continuation_models import TrendContinuationForwardConsumption


class ForwardConsumptionRaceTests(TransactionTestCase):
    reset_sequences = True

    def test_overlapping_first_contract_intervals_have_exactly_one_winner(self):
        # Given: two manifests racing to create the initially empty same-contract ledger.
        first = freeze_selection(_snapshot(_identity(), nonce="race-1"), frozen_at=BASE)
        second = freeze_selection(_snapshot(_identity(version="trend-v2"), nonce="race-2"), frozen_at=BASE)
        start = BASE + timedelta(days=1)

        def attempt(digest: str, offset: int) -> str:
            close_old_connections()
            try:
                register_forward_consumption(
                    contract_symbol="TSLAUSDT",
                    interval_start=start + timedelta(minutes=offset),
                    interval_end=start + timedelta(minutes=offset + 5),
                    manifest_digest=digest,
                    profile="standard_15m_5m",
                    trend_direction="UP",
                    rule_name="technical_score_v1",
                    candidate_horizon_bars=1,
                    phase="validation",
                )
            except ForwardIntervalOverlapError:
                return "rejected"
            finally:
                close_old_connections()
            return "created"

        # When: both partially overlapping registrations execute concurrently.
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = tuple(pool.map(lambda args: attempt(*args), ((first.digest, 0), (second.digest, 1))))

        # Then: the immutable ledger serializes the empty-ledger race to exactly one persisted interval.
        self.assertCountEqual(outcomes, ("created", "rejected"))
        self.assertEqual(TrendContinuationForwardConsumption.objects.count(), 1)
