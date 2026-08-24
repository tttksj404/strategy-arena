import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from pulse.bitget_api import BITGET_INTRADAY_SCREEN_CACHE_KEY
from pulse.models import Stock
from pulse.services.bitget_market import (
    BitgetContract,
    BitgetNetworkError,
    fetch_completed_4h_candles,
    list_rwa_contracts,
    select_stock_contracts,
)
from pulse.services.bitget_intraday_screener import IntradayScreenReport
from pulse.services.bitget_stock_screener import eligible_local_us_symbols
from pulse.services.live_long_signal_feed import LiveLongSignalFeed
from pulse.services.long_forecast import EvaluationMetrics, SplitEvaluation, promotion_is_reliable


def metrics(signal_count: int, target_hits: int) -> EvaluationMetrics:
    return EvaluationMetrics(
        signal_count=signal_count,
        target_hits=target_hits,
        stop_hits=signal_count - target_hits,
        unresolved_count=0,
        ambiguous_count=0,
        mean_terminal_return_pct=1.2,
    )


class LongPromotionGateTests(SimpleTestCase):
    def test_promotion_requires_validation_and_untouched_holdout_to_clear_fixed_margin(self):
        validation = SplitEvaluation(
            trend=metrics(signal_count=12, target_hits=7),
            baseline=metrics(signal_count=20, target_hits=6),
        )
        held_out = SplitEvaluation(
            trend=metrics(signal_count=11, target_hits=7),
            baseline=metrics(signal_count=20, target_hits=6),
        )

        self.assertTrue(promotion_is_reliable(validation, held_out))

        failed_holdout = SplitEvaluation(
            trend=metrics(signal_count=9, target_hits=7),
            baseline=metrics(signal_count=20, target_hits=6),
        )

        self.assertFalse(promotion_is_reliable(validation, failed_holdout))


class BitgetMarketBoundaryTests(SimpleTestCase):
    def test_only_local_us_stock_or_etf_symbols_survive_rwa_contract_intersection(self):
        payload = json.dumps(
            {
                "code": "00000",
                "data": [
                    {
                        "symbol": "TSLAUSDT",
                        "symbolStatus": "normal",
                        "maxLever": "100",
                        "fundInterval": "8",
                        "isRwa": "YES",
                    },
                    {
                        "symbol": "PAXGUSDT",
                        "symbolStatus": "normal",
                        "maxLever": "50",
                        "fundInterval": "4",
                        "isRwa": "YES",
                    },
                    {
                        "symbol": "NVDAUSDT",
                        "symbolStatus": "maintain",
                        "maxLever": "100",
                        "fundInterval": "8",
                        "isRwa": "YES",
                    },
                ],
            }
        )

        contracts = list_rwa_contracts(lambda _: payload)
        selected = select_stock_contracts(contracts, frozenset({"TSLA", "NVDA"}))

        self.assertEqual([contract.symbol for contract in selected], ["TSLAUSDT"])

    def test_in_progress_candle_is_excluded_before_forecast_features_are_built(self):
        now = datetime(2026, 8, 12, 12, tzinfo=UTC)
        completed = int((now - timedelta(hours=4)).timestamp() * 1000)
        in_progress = int(now.timestamp() * 1000)
        payload = json.dumps(
            {
                "code": "00000",
                "data": [
                    [str(completed), "100", "104", "99", "103", "10", "1000"],
                    [str(in_progress), "103", "110", "102", "109", "10", "1000"],
                ],
            }
        )
        contract = BitgetContract(
            symbol="TSLAUSDT",
            underlying_symbol="TSLA",
            max_leverage=100,
            funding_interval_hours=8,
        )

        candles = fetch_completed_4h_candles(contract, lambda _: payload, now)

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0].close, 103.0)


class BitgetStockScreenApiTests(TestCase):
    def setUp(self):
        cache.delete(BITGET_INTRADAY_SCREEN_CACHE_KEY)

    def tearDown(self):
        cache.delete(BITGET_INTRADAY_SCREEN_CACHE_KEY)

    def test_unauthenticated_endpoint_cannot_trigger_a_market_scan(self):
        client = APIClient()

        with patch("pulse.bitget_api.screen_bitget_live_long_alerts") as screen:
            response = client.get("/api/bitget/stock-long-candidates/")

        self.assertIn(response.status_code, (401, 403))
        screen.assert_not_called()

    def test_authenticated_endpoint_returns_direct_live_signal_feed_without_execution_path(self):
        # Given: an empty but valid live-signal feed from the public-market screen.
        report = IntradayScreenReport(
            as_of=datetime(2026, 8, 12, 12, tzinfo=UTC),
            eligible_contract_count=3,
            scanned_contract_count=3,
            candidates=(),
            rejection_counts=(("no_reliable_winner", 3),),
        )
        feed = LiveLongSignalFeed(base_report=report, signals=(), ticker_rejection_counts=())
        user = User.objects.create_user(username="bitget-screen-user", password="strong-pass-123")
        client = APIClient()
        client.force_authenticate(user=user)

        # When: an authenticated user requests the long-candidate endpoint.
        with patch("pulse.bitget_api.screen_bitget_live_long_alerts", return_value=feed) as screen:
            response = client.get("/api/bitget/stock-long-candidates/")

        # Then: the response contains the direct, read-only live-signal contract and never an order path.
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["execution"]["enabled"])
        self.assertEqual(response.data["mode"], "read_only_live_signal")
        self.assertTrue(response.data["data_policy"]["closed_candles_only"])
        self.assertEqual(response.data["candidates"], [])
        self.assertEqual(response.data["enter_now_signals"], [])
        screen.assert_called_once()

    def test_stale_cached_enter_long_is_discarded_before_returning_a_live_instruction(self):
        report = IntradayScreenReport(
            as_of=datetime(2026, 8, 12, 12, tzinfo=UTC),
            eligible_contract_count=3,
            scanned_contract_count=3,
            candidates=(),
            rejection_counts=(("no_reliable_winner", 3),),
        )
        feed = LiveLongSignalFeed(base_report=report, signals=(), ticker_rejection_counts=())
        stale_observed_at = (datetime.now(UTC) - timedelta(seconds=16)).isoformat()
        cache.set(
            BITGET_INTRADAY_SCREEN_CACHE_KEY,
            {
                "enter_now_signals": [
                    {
                        "action": "ENTER_LONG",
                        "research_status": "forward_evidence_ready",
                        "quote": {"observed_at": stale_observed_at},
                    }
                ]
            },
            timeout=20,
        )
        user = User.objects.create_user(username="bitget-stale-cache-user", password="strong-pass-123")
        client = APIClient()
        client.force_authenticate(user=user)

        with patch("pulse.bitget_api.screen_bitget_live_long_alerts", return_value=feed) as screen:
            response = client.get("/api/bitget/stock-long-candidates/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["enter_now_signals"], [])
        screen.assert_called_once()

    def test_forward_pending_cached_enter_long_is_discarded_before_returning_a_live_instruction(self):
        report = IntradayScreenReport(
            as_of=datetime(2026, 8, 12, 12, tzinfo=UTC),
            eligible_contract_count=3,
            scanned_contract_count=3,
            candidates=(),
            rejection_counts=(("no_reliable_winner", 3),),
        )
        feed = LiveLongSignalFeed(base_report=report, signals=(), ticker_rejection_counts=())
        current_observed_at = datetime.now(UTC).isoformat()
        cache.set(
            BITGET_INTRADAY_SCREEN_CACHE_KEY,
            {
                "enter_now_signals": [
                    {
                        "action": "ENTER_LONG",
                        "research_status": "forward_evidence_pending",
                        "quote": {"observed_at": current_observed_at},
                    }
                ]
            },
            timeout=20,
        )
        user = User.objects.create_user(username="bitget-forward-pending-cache-user", password="strong-pass-123")
        client = APIClient()
        client.force_authenticate(user=user)

        with patch("pulse.bitget_api.screen_bitget_live_long_alerts", return_value=feed) as screen:
            response = client.get("/api/bitget/stock-long-candidates/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["enter_now_signals"], [])
        screen.assert_called_once()

    def test_market_failure_is_reported_as_read_only_data_unavailability(self):
        user = User.objects.create_user(username="bitget-screen-error-user", password="strong-pass-123")
        client = APIClient()
        client.force_authenticate(user=user)

        with patch(
            "pulse.bitget_api.screen_bitget_live_long_alerts",
            side_effect=BitgetNetworkError(url="https://api.bitget.com", detail="timeout"),
        ):
            response = client.get("/api/bitget/stock-long-candidates/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["error"]["code"], "BITGET_MARKET_UNAVAILABLE")


class LocalUniverseBoundaryTests(TestCase):
    def test_only_us_stock_and_etf_rows_are_eligible_for_bitget_contract_intersection(self):
        Stock.objects.create(code="SCREENSTOCK", name="Screen Stock", market="US", asset_type="stock")
        Stock.objects.create(code="SCREENETF", name="Screen ETF", market="US", asset_type="etf")
        Stock.objects.create(code="SCREENOTHER", name="Screen Other", market="US", asset_type="crypto")

        symbols = eligible_local_us_symbols()

        self.assertIn("SCREENSTOCK", symbols)
        self.assertIn("SCREENETF", symbols)
        self.assertNotIn("SCREENOTHER", symbols)
