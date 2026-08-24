from __future__ import annotations

from datetime import timedelta

from django.test import SimpleTestCase

from .services.trend_continuation_contract import Direction
from .services.trend_continuation_evaluation import (
    ForwardQuoteSnapshot,
    FundingRateEvent,
    ObservationSource,
    compute_cost_pct,
)
from .test_trend_continuation_evaluation import BASE, _fee_policy, _funding


class DirectionalCostTests(SimpleTestCase):
    def test_costs_use_fee_policy_source_kind_quote_and_directional_funding_boundaries(self):
        label_end = BASE + timedelta(hours=8)
        funding = _funding(
            FundingRateEvent(BASE, 0.9),
            FundingRateEvent(BASE + timedelta(hours=4), 0.001),
            FundingRateEvent(label_end, -0.002),
            FundingRateEvent(label_end + timedelta(microseconds=1), 0.9),
            end=label_end,
        )
        quote = ForwardQuoteSnapshot(
            observed_at=BASE + timedelta(seconds=10),
            bid_price=99.9,
            ask_price=100.1,
            spread_pct=0.2,
            raw_sha256="b" * 64,
            settlement_lookup_started_at=BASE + timedelta(seconds=11),
        )

        historical_down = compute_cost_pct(
            source=ObservationSource.HISTORICAL_EXPLORATORY,
            direction=Direction.DOWN,
            evidence_time=BASE,
            label_end_time=label_end,
            fee_policy=_fee_policy(),
            funding=funding,
        )
        forward_up = compute_cost_pct(
            source=ObservationSource.FORWARD,
            direction=Direction.UP,
            evidence_time=BASE,
            label_end_time=label_end,
            fee_policy=_fee_policy(),
            funding=funding,
            quote=quote,
        )
        forward_down = compute_cost_pct(
            source=ObservationSource.FORWARD,
            direction=Direction.DOWN,
            evidence_time=BASE,
            label_end_time=label_end,
            fee_policy=_fee_policy(),
            funding=funding,
            quote=quote,
        )

        self.assertAlmostEqual(historical_down or 0, 0.4)
        self.assertAlmostEqual(forward_up or 0, 0.42)
        self.assertAlmostEqual(forward_down or 0, 0.52)
