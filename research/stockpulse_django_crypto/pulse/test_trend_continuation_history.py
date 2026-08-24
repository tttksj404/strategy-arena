from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase

from pulse.models import Stock
from pulse.services.bitget_market import BitgetContract
from pulse.services.market_timeframes import CandleGranularity
from pulse.services.trend_continuation_contract import Direction
from pulse.services.trend_continuation_funding import (
    DirectionalFundingCost,
    FundingWindow,
    collect_funding_window,
)
from pulse.services.trend_continuation_history import (
    HistoricalCollectionQuery,
    collect_historical_candles,
    required_profile_intervals,
)
from pulse.services.trend_continuation_market import reject_historical_quote_fixture
from pulse.services.trend_continuation_provenance import hash_raw_body


BASE = datetime(2026, 6, 1, tzinfo=UTC)


def _contract() -> BitgetContract:
    return BitgetContract("TSLAUSDT", "TSLA", 20, 8, taker_fee_rate=0.0006)


def _candle_row(opened_at: datetime, close: float = 100.0) -> list[str]:
    return [
        str(int(opened_at.timestamp() * 1000)),
        str(close),
        str(close + 1),
        str(close - 1),
        str(close + 0.25),
        "10",
        "1000",
    ]


class HistoricalPaginationTests(SimpleTestCase):
    def test_profile_requires_every_standard_and_triple_timeframe(self):
        self.assertEqual(
            required_profile_intervals("standard_15m_5m"),
            (CandleGranularity.FIFTEEN_MINUTES, CandleGranularity.FIVE_MINUTES),
        )
        self.assertEqual(
            required_profile_intervals("triple_leveraged_15m_5m_1m"),
            (
                CandleGranularity.FIFTEEN_MINUTES,
                CandleGranularity.FIVE_MINUTES,
                CandleGranularity.ONE_MINUTE,
            ),
        )

    def test_max_pages_is_operational_and_every_raw_page_is_aggregated(self):
        interval = CandleGranularity.FIVE_MINUTES
        rows = [_candle_row(BASE + interval.interval * index, 100 + index) for index in range(201)]

        def run(max_pages: int):
            calls: list[str] = []

            def fetch_text(url: str) -> str:
                calls.append(url)
                return json.dumps({"code": "00000", "data": rows[1:] if len(calls) == 1 else rows[:1]})

            result = collect_historical_candles(
                HistoricalCollectionQuery(_contract(), interval, BASE, BASE + interval.interval * 201, max_pages),
                fetch_text,
            )
            return result, calls

        limited, limited_calls = run(1)
        complete, complete_calls = run(2)

        self.assertFalse(limited.complete)
        self.assertEqual(limited.rejection_code, "history_page_limit")
        self.assertEqual(len(limited_calls), 1)
        self.assertTrue(complete.complete)
        self.assertEqual(complete.terminal_reason, "start_reached")
        self.assertEqual(len(complete_calls), 2)
        self.assertEqual(len(complete.pages), 2)
        self.assertEqual(complete.pages[0].raw_sha256, hash_raw_body(json.dumps({"code": "00000", "data": rows[1:]})))
        self.assertNotEqual(complete.aggregate_raw_sha256, complete.pages[0].raw_sha256)
        self.assertEqual(complete.first_open, BASE)
        self.assertEqual(complete.last_open, BASE + interval.interval * 200)

    def test_continuity_gap_is_reported_without_partial_candles(self):
        interval = CandleGranularity.FIVE_MINUTES
        body = json.dumps({"code": "00000", "data": [_candle_row(BASE), _candle_row(BASE + interval.interval * 2)]})
        result = collect_historical_candles(
            HistoricalCollectionQuery(_contract(), interval, BASE, BASE + interval.interval * 3, 1),
            lambda url: body,
        )

        self.assertFalse(result.complete)
        self.assertEqual(result.candles, ())
        self.assertEqual(result.rejection_code, "missing_candle")
        self.assertEqual(result.continuity_gaps, (BASE + interval.interval,))


class FundingPaginationTests(SimpleTestCase):
    def test_multi_page_funding_and_complete_no_event_proof_are_distinct_from_incomplete(self):
        end = BASE + timedelta(hours=200)
        rows = [
            {"fundingTime": str(int((BASE + timedelta(hours=index + 1)).timestamp() * 1000)), "fundingRate": "0.001"}
            for index in range(171)
        ]

        def fetch_text(url: str) -> str:
            page = int(parse_qs(urlparse(url).query)["pageNo"][0])
            return json.dumps({"code": "00000", "data": rows[:100] if page == 1 else rows[100:]})

        window = FundingWindow(_contract(), BASE, end)
        complete = collect_funding_window(window, fetch_text, collected_at=end + timedelta(seconds=1), max_pages=2)
        empty = collect_funding_window(
            window,
            lambda url: json.dumps({"code": "00000", "data": []}),
            collected_at=end + timedelta(seconds=1),
            max_pages=1,
        )
        incomplete = collect_funding_window(window, fetch_text, collected_at=end + timedelta(seconds=1), max_pages=1)

        self.assertTrue(complete.eligible)
        self.assertEqual(len(complete.pages), 2)
        self.assertEqual(len(complete.events), 171)
        self.assertTrue(empty.eligible)
        self.assertEqual(empty.events, ())
        self.assertTrue(empty.coverage_complete)
        self.assertFalse(incomplete.eligible)
        self.assertEqual(incomplete.rejection_code, "funding_page_limit")
        self.assertFalse(incomplete.coverage_complete)

    def test_funding_utc_boundaries_duplicates_and_directional_cost(self):
        end = BASE + timedelta(hours=8)
        inside = BASE + timedelta(hours=4)
        rows = [
            {"fundingTime": str(int(BASE.timestamp() * 1000)), "fundingRate": "0.001"},
            {"fundingTime": str(int(inside.timestamp() * 1000)), "fundingRate": "0.002"},
            {"fundingTime": str(int(end.timestamp() * 1000)), "fundingRate": "-0.003"},
            {"fundingTime": str(int((end + timedelta(milliseconds=1)).timestamp() * 1000)), "fundingRate": "0.004"},
        ]
        result = collect_funding_window(
            FundingWindow(_contract(), BASE, end),
            lambda url: json.dumps({"code": "00000", "data": rows}),
            collected_at=end + timedelta(seconds=1),
        )
        duplicate = collect_funding_window(
            FundingWindow(_contract(), BASE, end),
            lambda url: json.dumps({"code": "00000", "data": [rows[1], rows[1]]}),
            collected_at=end + timedelta(seconds=1),
        )

        self.assertEqual([event.funding_time for event in result.events], [inside, end])
        self.assertEqual(DirectionalFundingCost.from_events(result.events, Direction.UP).cost_pct, 0.2)
        self.assertEqual(DirectionalFundingCost.from_events(result.events, Direction.DOWN).cost_pct, 0.3)
        self.assertFalse(duplicate.eligible)
        self.assertEqual(duplicate.rejection_code, "duplicate_funding")


class HistoricalFixtureBoundaryTests(SimpleTestCase):
    def test_recursive_quote_or_spread_fixture_is_rejected_before_network_or_evaluation(self):
        payload = {"candles": {"5m": [{"fee": {"spread": 0.2}}]}}
        rejection = reject_historical_quote_fixture(payload)
        self.assertFalse(rejection.eligible)
        self.assertEqual(rejection.blocked_path, "$.candles.5m[0].fee.spread")

        with TemporaryDirectory() as temporary:
            fixture_path = Path(temporary) / "fixture.json"
            fixture_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("pulse.services.bitget_market.BitgetPublicClient.get_text") as network:
                with self.assertRaisesMessage(CommandError, "historical fixture cannot contain bid/ask/spread"):
                    call_command(
                        "evaluate_bitget_trend_continuation",
                        symbol="TSLAUSDT",
                        profile="standard_15m_5m",
                        start="2026-06-01T00:00:00+00:00",
                        end="2026-06-01T01:00:00+00:00",
                        max_pages=1,
                        input_fixture=str(fixture_path),
                    )
            network.assert_not_called()


class HistoricalNetworkPolicyTests(TestCase):
    def test_normal_history_never_calls_ticker_and_fails_closed_without_past_fee_policy(self):
        Stock.objects.create(code="TSLA", name="Tesla", market="US", asset_type="stock")
        end = BASE + timedelta(hours=1)
        contract_body = json.dumps(
            {
                "code": "00000",
                "data": [
                    {
                        "symbol": "TSLAUSDT",
                        "symbolStatus": "normal",
                        "maxLever": "20",
                        "fundInterval": "8",
                        "isRwa": "YES",
                        "takerFeeRate": "0.0006",
                    }
                ],
            }
        )
        requested: list[str] = []

        def fetch_text(url: str) -> str:
            path = urlparse(url).path
            requested.append(path)
            if path.endswith("/contracts"):
                return contract_body
            if path.endswith("/history-candles"):
                interval = CandleGranularity(parse_qs(urlparse(url).query)["granularity"][0])
                count = int((end - BASE) / interval.interval)
                return json.dumps(
                    {"code": "00000", "data": [_candle_row(BASE + interval.interval * index) for index in range(count)]}
                )
            if path.endswith("/history-fund-rate"):
                return json.dumps({"code": "00000", "data": []})
            raise AssertionError(f"unexpected historical URL: {path}")

        output = StringIO()
        with patch("pulse.services.bitget_market.BitgetPublicClient.get_text", side_effect=fetch_text):
            with self.assertRaisesMessage(CommandError, "historical_fee_policy_coverage_unproven"):
                call_command(
                    "evaluate_bitget_trend_continuation",
                    symbol="TSLAUSDT",
                    profile="standard_15m_5m",
                    start=BASE.isoformat(),
                    end=end.isoformat(),
                    max_pages=1,
                    stdout=output,
                )
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["data_contracts"]["fee_policy_complete"])
        self.assertIn("historical_fee_policy_coverage_unproven", payload["data_contracts"]["failure_codes"])
        self.assertNotIn("/api/v2/mix/market/ticker", requested)
