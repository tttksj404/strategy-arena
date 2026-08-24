from datetime import UTC, datetime
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase
from rest_framework.test import APIClient

from pulse.bitget_api import (
    BITGET_TREND_CONTINUATION_CACHE_PREFIX,
    _trend_continuation_cache_key,
    _trend_continuation_completed_1m_boundary,
)
from pulse.services.bitget_market import BitgetNetworkError
from pulse.services.trend_continuation_research import TrendContinuationReport


def _report(as_of: datetime, diagnostics=()) -> TrendContinuationReport:
    return TrendContinuationReport(
        as_of=as_of,
        diagnostics=diagnostics,
        rejections=(("forward_watcher_required", 1),),
    )


class BitgetTrendContinuationApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username="trend-api-user", password="strong-pass-123")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def tearDown(self):
        cache.clear()

    def test_unauthenticated_endpoint_cannot_trigger_research(self):
        # Given: an unauthenticated API client.
        client = APIClient()

        # When: the trend-continuation endpoint is requested.
        with patch("pulse.bitget_api.build_trend_continuation_report") as build_report:
            response = client.get("/api/bitget/trend-continuation/")

        # Then: no market/research work is triggered.
        self.assertIn(response.status_code, (401, 403))
        build_report.assert_not_called()

    def test_authenticated_endpoint_returns_research_only_no_trade_contract(self):
        # Given: the current research facade has no forward-validated winner.
        as_of = datetime(2026, 8, 18, 13, 42, tzinfo=UTC)

        # When: the endpoint is requested.
        with patch("pulse.bitget_api.build_trend_continuation_report", return_value=_report(as_of)) as build_report:
            response = self.client.get("/api/bitget/trend-continuation/")

        # Then: the public API is deterministic, research-only, and non-executable.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["mode"], "research_only")
        self.assertEqual(response.data["decision_status"], "NO_TRADE")
        self.assertFalse(response.data["execution"]["enabled"])
        self.assertEqual(response.data["diagnostics"], [])
        self.assertNotIn("action", response.data)
        self.assertNotIn("entry", response.data)
        build_report.assert_called_once()

    def test_public_decision_status_stays_no_trade_for_observe_diagnostics(self):
        # Given: the internal service returns a self-consistent observe diagnostic.
        diagnostic = {
            "decision": "CONTINUATION_OBSERVE",
            "execution": {"enabled": False},
            "contract": "KORUUSDT",
            "profile": "standard",
            "provenance": {"source": "public_bitget_market"},
        }
        as_of = datetime(2026, 8, 18, 13, 42, tzinfo=UTC)

        # When: the public API serializes the report.
        with patch("pulse.bitget_api.build_trend_continuation_report", return_value=_report(as_of, (diagnostic,))):
            response = self.client.get("/api/bitget/trend-continuation/")

        # Then: no promoted observation leaks into the API-level decision.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["decision_status"], "NO_TRADE")
        self.assertFalse(response.data["execution"]["enabled"])
        self.assertNotIn("action", response.data)
        self.assertNotIn("entry", response.data)

    def test_cache_key_uses_latest_completed_one_minute_boundary(self):
        # Given: wall time is inside an in-progress one-minute candle.
        now = datetime(2026, 8, 18, 13, 42, 30, tzinfo=UTC)

        # When: the cache key and response boundary are derived.
        boundary = _trend_continuation_completed_1m_boundary(now)
        cache_key = _trend_continuation_cache_key(now)

        # Then: both point to the latest completed one-minute candle.
        self.assertEqual(boundary, datetime(2026, 8, 18, 13, 41, tzinfo=UTC))
        self.assertTrue(cache_key.endswith("2026-08-18T13:41:00+00:00"))

    def test_cache_is_reused_only_when_boundary_and_payload_identity_match(self):
        # Given: a valid cache wrapper for the current completed one-minute boundary.
        as_of = datetime(2026, 8, 18, 13, 42, tzinfo=UTC)
        with patch("pulse.bitget_api.build_trend_continuation_report", return_value=_report(as_of)) as first_build:
            first = self.client.get("/api/bitget/trend-continuation/")

        # When: the same boundary is requested again.
        with patch("pulse.bitget_api.build_trend_continuation_report", return_value=_report(as_of)) as second_build:
            second = self.client.get("/api/bitget/trend-continuation/")

        # Then: the cached public payload is returned without rebuilding.
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.data["cache"]["boundary"], first.data["cache"]["boundary"])
        first_build.assert_called_once()
        second_build.assert_not_called()

    def test_plain_or_mismatched_cache_payload_is_rejected_before_return(self):
        # Given: a polluted cache entry with the correct prefix but no validated wrapper.
        now = datetime.now(UTC)
        cache.set(
            _trend_continuation_cache_key(now),
            {
                "mode": "research_only",
                "decision_status": "CONTINUATION_OBSERVE",
                "execution": {"enabled": True},
                "action": "ENTER_LONG",
            },
            timeout=60,
        )

        # When: the endpoint is requested.
        as_of = datetime(2026, 8, 18, 13, 42, tzinfo=UTC)
        with patch("pulse.bitget_api.build_trend_continuation_report", return_value=_report(as_of)) as build_report:
            response = self.client.get("/api/bitget/trend-continuation/")

        # Then: the polluted cache is discarded and the service result is returned.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["decision_status"], "NO_TRADE")
        self.assertFalse(response.data["execution"]["enabled"])
        self.assertNotIn("action", response.data)
        build_report.assert_called_once()

    def test_service_market_failure_maps_to_data_unavailable(self):
        # Given: the public market data layer is unavailable.
        error = BitgetNetworkError(url="https://api.bitget.com/api/v2/mix/market/candles", detail="timeout")

        # When: the endpoint is requested.
        with patch("pulse.bitget_api.build_trend_continuation_report", side_effect=error):
            response = self.client.get("/api/bitget/trend-continuation/")

        # Then: the API reports only DATA_UNAVAILABLE.
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["error"]["code"], "DATA_UNAVAILABLE")

    def test_endpoint_does_not_use_legacy_or_direct_ticker_sources(self):
        # Given: legacy and ticker paths would fail if the new route touched them.
        as_of = datetime(2026, 8, 18, 13, 42, tzinfo=UTC)

        # When: trend research is requested.
        with (
            patch("pulse.bitget_api.build_trend_continuation_report", return_value=_report(as_of)),
            patch("pulse.bitget_api.screen_bitget_live_long_alerts") as legacy_screen,
            patch("pulse.services.bitget_ticker.fetch_live_ticker") as ticker_fetch,
        ):
            response = self.client.get("/api/bitget/trend-continuation/")

        # Then: the route stays inside the trend facade boundary.
        self.assertEqual(response.status_code, 200)
        legacy_screen.assert_not_called()
        ticker_fetch.assert_not_called()

    def test_cache_prefix_is_stable_for_ui_integration(self):
        # Given / When / Then: UI polling can rely on the documented cache family.
        self.assertEqual(BITGET_TREND_CONTINUATION_CACHE_PREFIX, "bitget:trend-continuation:v1:")
