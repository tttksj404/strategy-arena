from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from pulse.services.market_timeframes import CandleGranularity
from pulse.services.trend_continuation_contract import CandidateIdentity, Direction
from pulse.services.trend_continuation_forward_collection import ForwardEvidenceCollection
from pulse.services.trend_continuation_manifest import TrendContinuationSelectionSnapshot
from pulse.services.trend_continuation_research import freeze_selection_idempotently
from pulse.services.trend_continuation_settlement import (
    SettlementIncompleteError,
    settle_observation,
)
from pulse.services.trend_continuation_watcher import run_watcher_cycle
from pulse.test_trend_continuation_manifest import BASE, _identity, _observation, _snapshot
from pulse.trend_continuation_models import (
    TrendContinuationObservation,
    TrendContinuationSettlement,
)


EVIDENCE = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)


def _manifest_snapshot(identity: CandidateIdentity) -> TrendContinuationSelectionSnapshot:
    return TrendContinuationSelectionSnapshot(
        payload={
            "schema": "trend-continuation-selection.v1",
            "exploratory_only": True,
            "candidate": {
                "contract": identity.contract,
                "profile": identity.profile,
                "trend_direction": identity.trend_direction.value,
                "rule_name": identity.rule_name,
                "candidate_horizon_bars": identity.candidate_horizon_bars,
                "strategy_version": identity.strategy_version,
                "feature_schema_version": identity.feature_schema_version,
            },
            "fee_policy": {
                "version": "fee-forward-v1",
                "recorded_at": "2026-08-01T00:00:00+00:00",
                "source_url": "https://www.bitget.com/api-doc/contract/market/Get-All-Symbols-Contracts",
                "raw_sha256": "a" * 64,
                "effective_from": "2026-08-01T00:00:00+00:00",
                "effective_until": "2026-09-01T00:00:00+00:00",
                "taker_fee_rate": 0.0006,
            },
        }
    )


def _candle_body(interval: CandleGranularity) -> str:
    start = EVIDENCE - interval.interval * 200
    rows: list[list[str]] = []
    for index in range(200):
        close = 100 + index * 0.25 + ((index % 4) - 1.5) * 0.35
        low = close - 0.9 - (2.0 if index == 195 else 0.0)
        rows.append(
            [
                str(int((start + interval.interval * index).timestamp() * 1000)),
                str(close - 0.1),
                str(close + 0.9),
                str(low),
                str(close),
                "10",
                str(1000 + (index % 5) * 10),
            ]
        )
    return json.dumps({"code": "00000", "data": rows})


class WatcherCommandTests(TestCase):
    def test_cycles_are_bounded_and_empty_manifest_run_never_fetches(self):
        output = StringIO()
        with patch("pulse.services.bitget_market.BitgetPublicClient.get_text") as fetch:
            call_command(
                "watch_bitget_trend_continuation",
                cycles=2,
                sleep_seconds=0,
                stdout=output,
            )
        payload = json.loads(output.getvalue())
        fetch.assert_not_called()
        self.assertEqual(payload["cycles_completed"], 2)
        self.assertFalse(payload["execution"]["enabled"])
        self.assertNotIn('"action"', output.getvalue())
        self.assertNotIn('"leverage"', output.getvalue())

        for options in ({"cycles": 0}, {"cycles": -1}, {"sleep_seconds": -1}):
            with self.subTest(options=options):
                with self.assertRaises(CommandError):
                    call_command("watch_bitget_trend_continuation", **options)

    def test_incomplete_ready_settlement_makes_command_fail_nonzero(self):
        identity = _identity("MSFTUSDT")
        freeze_selection_idempotently(_snapshot(identity, nonce="watcher-incomplete"), frozen_at=BASE)
        _observation(identity, BASE + timedelta(days=1))
        no_candidate = ForwardEvidenceCollection(
            draft=None,
            evidence_time=BASE + timedelta(days=2),
            rejection_code="technical_gate_failed",
        )
        with (
            patch(
                "pulse.services.trend_continuation_watcher.collect_manifest_forward_evidence",
                return_value=no_candidate,
            ),
            patch(
                "pulse.services.bitget_market.BitgetPublicClient.get_text",
                return_value=json.dumps({"code": "00000", "data": []}),
            ),
            self.assertRaises(CommandError),
        ):
            call_command("watch_bitget_trend_continuation", cycles=1, sleep_seconds=0)

        self.assertEqual(TrendContinuationSettlement.objects.count(), 0)


class WatcherForwardIntegrationTests(TestCase):
    def test_evidence_then_current_funding_then_task2_quote_is_raw_hashed_and_idempotent(self):
        identity = CandidateIdentity(
            "TSLAUSDT",
            "standard_15m_5m",
            Direction.UP,
            "technical_score_v1",
            12,
            "trend-continuation-strategy.v1",
            "trend-continuation-features.v1",
        )
        manifest = freeze_selection_idempotently(
            _manifest_snapshot(identity),
            frozen_at=EVIDENCE - timedelta(minutes=1),
        )
        observed = EVIDENCE + timedelta(seconds=10)
        quote_body = json.dumps(
            {
                "code": "00000",
                "data": [
                    {
                        "symbol": "TSLAUSDT",
                        "lastPr": "149.5",
                        "bidPr": "149.4",
                        "askPr": "149.6",
                        "ts": str(int(observed.timestamp() * 1000)),
                    }
                ],
            }
        )
        funding_body = json.dumps(
            {"code": "00000", "data": [{"symbol": "TSLAUSDT", "fundingRate": "0.0001"}]}
        )
        requests: list[str] = []

        def fetch_text(url: str) -> str:
            path = urlparse(url).path
            requests.append(path)
            if path.endswith("/candles"):
                interval = parse_qs(urlparse(url).query)["granularity"][0]
                return _candle_body(CandleGranularity(interval))
            if path.endswith("/current-fund-rate"):
                return funding_body
            if path.endswith("/ticker"):
                return quote_body
            raise AssertionError(f"unexpected watcher request: {path}")

        first = run_watcher_cycle(fetch_text, collected_at=observed)
        second = run_watcher_cycle(fetch_text, collected_at=observed)

        self.assertTrue(first.successful)
        self.assertTrue(second.successful)
        self.assertEqual(requests[:4], [
            "/api/v2/mix/market/candles",
            "/api/v2/mix/market/candles",
            "/api/v2/mix/market/current-fund-rate",
            "/api/v2/mix/market/ticker",
        ])
        self.assertEqual(requests.count("/api/v2/mix/market/ticker"), 1)
        self.assertEqual(TrendContinuationObservation.objects.count(), 1)
        observation = TrendContinuationObservation.objects.get()
        self.assertEqual(len(observation.quote_snapshot["raw_sha256"]), 64)
        self.assertEqual(observation.funding_snapshot["scope"], "capture_time_current_rate")
        self.assertEqual(len(observation.raw_response_sha256), 64)
        self.assertNotEqual(observation.raw_response_sha256, observation.quote_snapshot["raw_sha256"])
        self.assertEqual(manifest.contract_symbol, observation.contract_symbol)

    def test_incomplete_settlement_provenance_is_non_mutating_failure(self):
        observation = _observation(
            _identity("MSFTUSDT"),
            EVIDENCE - timedelta(hours=1),
        )

        with self.assertRaises(SettlementIncompleteError):
            settle_observation(
                observation,
                lambda url: json.dumps({"code": "00000", "data": []}),
                collected_at=EVIDENCE,
            )

        self.assertFalse(TrendContinuationSettlement.objects.filter(observation=observation).exists())
