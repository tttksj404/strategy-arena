from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

from django.test import SimpleTestCase

from pulse.services.bitget_market import BitgetContract
from pulse.services.market_timeframes import CandleGranularity
from pulse.services.trend_continuation_contract import PublicMarketRequest
from pulse.services.trend_continuation_market import (
    CandleWindowQuery,
    QuoteObservationQuery,
    collect_completed_candle_window,
    collect_forward_quote,
)
from pulse.services.trend_continuation_provenance import (
    RAW_SHA_256_EMPTY_JSON,
    ProvenanceBlockedSource,
    hash_raw_body,
    normalize_query,
    provenance_request,
    require_v2_public_market_request,
)


def _contract() -> BitgetContract:
    return BitgetContract("TSLAUSDT", "TSLA", 20, 8, taker_fee_rate=0.0006)


def _candle_row(opened_at: datetime) -> list[str]:
    return [
        str(int(opened_at.timestamp() * 1000)),
        "100",
        "101",
        "99",
        "100.5",
        "10",
        "1000",
    ]


class TrendContinuationProvenanceTests(SimpleTestCase):
    def test_normalized_request_and_raw_hash_are_stable_before_parse(self):
        # Given: the same public request with query pairs in different orders.
        body = '{"code":"00000","data":[]}'
        left = provenance_request(
            PublicMarketRequest(
                method="GET",
                path="/api/v2/mix/market/candles",
                query=(("granularity", "5m"), ("symbol", "TSLAUSDT")),
            ),
            contract="TSLAUSDT",
            raw_sha256=hash_raw_body(body),
            collected_at=datetime(2026, 8, 18, 3, 5, tzinfo=UTC),
            row_count=0,
            complete=True,
            continuous=True,
        )
        right = provenance_request(
            PublicMarketRequest(
                method="GET",
                path="/api/v2/mix/market/candles",
                query=(("symbol", "TSLAUSDT"), ("granularity", "5m")),
            ),
            contract="TSLAUSDT",
            raw_sha256=hash_raw_body(body),
            collected_at=datetime(2026, 8, 18, 3, 5, tzinfo=UTC),
            row_count=0,
            complete=True,
            continuous=True,
        )

        # When / Then: the normalized query and raw hash are deterministic.
        self.assertEqual(normalize_query(left.request.query), (("granularity", "5m"), ("symbol", "TSLAUSDT")))
        self.assertEqual(left.normalized_query, right.normalized_query)
        self.assertEqual(left.raw_sha256, hash_raw_body(body))
        self.assertNotEqual(left.raw_sha256, RAW_SHA_256_EMPTY_JSON)

    def test_v2_public_allowlist_blocks_order_url_before_fetch(self):
        # Given: a private/order-like URL.
        with self.assertRaises(ProvenanceBlockedSource):
            # When / Then: the provenance layer rejects it before any fetcher could run.
            require_v2_public_market_request("GET", "/api/v2/mix/order/place-order")

    def test_completed_candle_window_accepts_only_complete_continuous_rows(self):
        # Given: exactly three completed one-minute candles and a raw public response.
        starts_at = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
        ends_at = starts_at + timedelta(minutes=3)
        body = json.dumps({"code": "00000", "data": [_candle_row(starts_at + timedelta(minutes=index)) for index in range(3)]})

        def fetch_text(url: str) -> str:
            query = parse_qs(urlparse(url).query)
            self.assertEqual(query["symbol"], ["TSLAUSDT"])
            self.assertEqual(query["granularity"], ["1m"])
            return body

        # When: the bounded collection wrapper parses the actual Bitget payload shape.
        result = collect_completed_candle_window(
            CandleWindowQuery(
                contract=_contract(),
                granularity=CandleGranularity.ONE_MINUTE,
                starts_at=starts_at,
                ends_at=ends_at,
                endpoint_path="/api/v2/mix/market/history-candles",
            ),
            fetch_text,
            collected_at=ends_at + timedelta(minutes=1),
        )

        # Then: candles and immutable provenance agree on completeness and raw hash.
        self.assertTrue(result.eligible)
        self.assertEqual(result.rejection_code, None)
        self.assertEqual([candle.opened_at for candle in result.candles], [starts_at + timedelta(minutes=index) for index in range(3)])
        self.assertEqual(result.provenance.raw_sha256, hash_raw_body(body))
        self.assertEqual(result.provenance.row_count, 3)
        self.assertTrue(result.provenance.complete)
        self.assertTrue(result.provenance.continuous)

    def test_completed_candle_window_rejects_open_missing_duplicate_and_malformed_rows(self):
        # Given: malformed and incomplete candle windows.
        starts_at = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
        ends_at = starts_at + timedelta(minutes=3)
        cases = {
            "open_candle": {"code": "00000", "data": [_candle_row(starts_at), _candle_row(starts_at + timedelta(minutes=1)), _candle_row(ends_at)]},
            "missing_candle": {"code": "00000", "data": [_candle_row(starts_at), _candle_row(starts_at + timedelta(minutes=2))]},
            "duplicate_candle": {"code": "00000", "data": [_candle_row(starts_at), _candle_row(starts_at), _candle_row(starts_at + timedelta(minutes=1))]},
            "malformed_json": "{",
        }

        for expected_reason, payload in cases.items():
            with self.subTest(expected_reason=expected_reason):
                raw_body = payload if isinstance(payload, str) else json.dumps(payload)

                def fetch_text(url: str) -> str:
                    return raw_body

                # When: collection sees unsafe input.
                result = collect_completed_candle_window(
                    CandleWindowQuery(
                        contract=_contract(),
                        granularity=CandleGranularity.ONE_MINUTE,
                        starts_at=starts_at,
                        ends_at=ends_at,
                        endpoint_path="/api/v2/mix/market/history-candles",
                    ),
                    fetch_text,
                    collected_at=ends_at + timedelta(minutes=1),
                )

                # Then: the output is deterministic unresolved/NO_TRADE material.
                self.assertFalse(result.eligible)
                self.assertEqual(result.rejection_code, expected_reason)
                self.assertEqual(result.candles, ())
                self.assertEqual(result.provenance.raw_sha256, hash_raw_body(raw_body))

    def test_malformed_json_preserves_raw_hash_before_parsing(self):
        # Given: a malformed public candle response that still has raw bytes.
        starts_at = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
        raw_body = "{"

        def fetch_text(url: str) -> str:
            return raw_body

        # When: collection fails during JSON parsing.
        result = collect_completed_candle_window(
            CandleWindowQuery(
                contract=_contract(),
                granularity=CandleGranularity.ONE_MINUTE,
                starts_at=starts_at,
                ends_at=starts_at + timedelta(minutes=1),
                endpoint_path="/api/v2/mix/market/history-candles",
            ),
            fetch_text,
            collected_at=starts_at + timedelta(minutes=2),
        )

        # Then: raw SHA-256 was already preserved before parsing decided unresolved.
        self.assertFalse(result.eligible)
        self.assertEqual(result.rejection_code, "malformed_json")
        self.assertEqual(result.provenance.raw_sha256, hash_raw_body(raw_body))

    def test_historical_candle_collection_never_uses_ticker_endpoint(self):
        # Given: a historical candle fetcher that fails if ticker is requested.
        starts_at = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)
        body = json.dumps({"code": "00000", "data": [_candle_row(starts_at)]})
        requested_urls: list[str] = []

        def fetch_text(url: str) -> str:
            requested_urls.append(url)
            if "/api/v2/mix/market/ticker" in url:
                raise AssertionError("historical candle collection must not call ticker")
            return body

        # When: collecting a historical candle window.
        result = collect_completed_candle_window(
            CandleWindowQuery(
                contract=_contract(),
                granularity=CandleGranularity.ONE_MINUTE,
                starts_at=starts_at,
                ends_at=starts_at + timedelta(minutes=1),
                endpoint_path="/api/v2/mix/market/history-candles",
            ),
            fetch_text,
            collected_at=starts_at + timedelta(minutes=2),
        )

        # Then: only history-candles was requested.
        self.assertTrue(result.eligible)
        self.assertEqual(len(requested_urls), 1)
        self.assertIn("/api/v2/mix/market/history-candles", requested_urls[0])
        self.assertNotIn("/api/v2/mix/market/ticker", requested_urls[0])

    def test_forward_quote_is_watcher_only_and_must_arrive_inside_post_evidence_window(self):
        # Given: a trigger closed at 03:05 and a live ticker observed at the 15-second boundary.
        evidence_time = datetime(2026, 8, 18, 3, 5, tzinfo=UTC)
        raw_body = json.dumps(
            {
                "code": "00000",
                "data": [
                    {
                        "symbol": "TSLAUSDT",
                        "lastPr": "100.5",
                        "bidPr": "100.4",
                        "askPr": "100.6",
                        "ts": str(int((evidence_time + timedelta(seconds=15)).timestamp() * 1000)),
                    }
                ],
            }
        )

        def fetch_text(url: str) -> str:
            self.assertIn("/api/v2/mix/market/ticker", url)
            return raw_body

        # When: the watcher captures a quote before settlement lookup.
        result = collect_forward_quote(
            QuoteObservationQuery(
                contract=_contract(),
                evidence_time=evidence_time,
                watcher_collected_at=evidence_time + timedelta(seconds=15),
                settlement_lookup_started_at=evidence_time + timedelta(seconds=16),
                taker_fee_rate=0.0006,
            ),
            fetch_text,
        )

        # Then: bid/ask, fee, observed time, and raw hash are present.
        self.assertTrue(result.eligible)
        self.assertEqual(result.quote.bid_price, 100.4)
        self.assertEqual(result.quote.ask_price, 100.6)
        self.assertEqual(result.provenance.raw_sha256, hash_raw_body(raw_body))
        self.assertEqual(result.provenance.fee_rate, 0.0006)

        late = collect_forward_quote(
            QuoteObservationQuery(
                contract=_contract(),
                evidence_time=evidence_time,
                watcher_collected_at=evidence_time + timedelta(seconds=16),
                settlement_lookup_started_at=evidence_time + timedelta(seconds=17),
                taker_fee_rate=0.0006,
            ),
            fetch_text,
        )
        self.assertFalse(late.eligible)
        self.assertEqual(late.rejection_code, "late_quote")

        after_settlement = collect_forward_quote(
            QuoteObservationQuery(
                contract=_contract(),
                evidence_time=evidence_time,
                watcher_collected_at=evidence_time + timedelta(seconds=10),
                settlement_lookup_started_at=evidence_time + timedelta(seconds=9),
                taker_fee_rate=0.0006,
            ),
            fetch_text,
        )
        self.assertFalse(after_settlement.eligible)
        self.assertEqual(after_settlement.rejection_code, "quote_after_settlement_lookup")
