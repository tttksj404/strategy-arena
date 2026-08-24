import json
from datetime import UTC, datetime, timedelta

from django.test import SimpleTestCase, TestCase

from pulse.intraday_signal_models import IntradayLongSignal, SignalOutcome
from pulse.models import Stock
from pulse.services.bitget_intraday_screener import classify_instrument
from pulse.services.bitget_market import BitgetContract, fetch_completed_candles
from pulse.services.intraday_long_forecast import (
    REQUESTED_LEVERAGE,
    STRATEGY_VERSION,
    IntradayCandleSet,
    IntradayLongPlan,
    InstrumentProfile,
    evaluate_fixed_leverage_feasibility,
    plan_intraday_long,
)
from pulse.services.intraday_signal_ledger import (
    ForwardPerformance,
    promotion_evaluation_is_ready,
    promotion_for,
    promotion_is_ready,
)
from pulse.services.intraday_signal_settlement import _settle
from pulse.services.intraday_signal_settlement import settle_pending_signals
from pulse.services.long_forecast import Candle
from pulse.services.market_timeframes import CandleGranularity


def _rising_candles(granularity: CandleGranularity, count: int) -> tuple[Candle, ...]:
    opened_at = datetime(2026, 8, 12, tzinfo=UTC) - granularity.interval * count
    candles: list[Candle] = []
    for index in range(count):
        open_price = 100 + index * 0.04
        close_price = open_price + 0.03
        candles.append(
            Candle(
                opened_at=opened_at + granularity.interval * index,
                open=open_price,
                high=close_price + 0.04,
                low=open_price - 0.03,
                close=close_price,
                quote_volume=1_000 + index * 3,
            )
        )
    return tuple(candles)


class CompletedIntradayCandleTests(SimpleTestCase):
    def test_fetch_completed_candles_uses_requested_granularity_and_excludes_open_bar(self):
        # Given: one completed one-minute bar and one bar that is still open.
        now = datetime(2026, 8, 12, 12, 0, 30, tzinfo=UTC)
        completed = int((now - timedelta(minutes=1)).timestamp() * 1000)
        open_bar = int(now.replace(second=0, microsecond=0).timestamp() * 1000)
        payload = json.dumps(
            {
                "code": "00000",
                "data": [
                    [str(completed), "100", "101", "99", "100.5", "10", "1000"],
                    [str(open_bar), "100.5", "102", "100", "101.5", "10", "1000"],
                ],
            }
        )
        requested_urls: list[str] = []
        contract = BitgetContract("SOXLUSDT", "SOXL", 50, 8, 0.0006)

        # When: the one-minute completed-candle adapter runs.
        candles = fetch_completed_candles(
            contract,
            CandleGranularity.ONE_MINUTE,
            lambda url: requested_urls.append(url) or payload,
            now,
        )

        # Then: only the closed candle is exposed to the strategy and the API request names 1m.
        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0].close, 100.5)
        self.assertIn("granularity=1m", requested_urls[0])


class FixedLeveragePolicyTests(SimpleTestCase):
    def test_research_risk_policy_is_fixed_at_five_x(self):
        self.assertEqual(REQUESTED_LEVERAGE, 5)


class IntradayProfileTests(SimpleTestCase):
    def test_triple_leveraged_etf_requires_one_minute_trigger_but_standard_profile_does_not(self):
        # Given: complete upward market windows for each relevant timeframe.
        market = IntradayCandleSet(
            context_15m=_rising_candles(CandleGranularity.FIFTEEN_MINUTES, 70),
            confirmation_5m=_rising_candles(CandleGranularity.FIVE_MINUTES, 70),
            trigger_1m=_rising_candles(CandleGranularity.ONE_MINUTE, 70),
        )

        # When: the standard and triple-leveraged profiles create plans from the same closed data.
        standard = plan_intraday_long(InstrumentProfile.STANDARD, market)
        triple = plan_intraday_long(InstrumentProfile.TRIPLE_LEVERAGED_ETF, market)

        # Then: both have a causal plan, but their execution windows respect their profile.
        self.assertIsNotNone(standard)
        self.assertIsNotNone(triple)
        assert standard is not None
        assert triple is not None
        self.assertEqual(standard.expires_at - standard.starts_at, timedelta(minutes=60))
        self.assertEqual(triple.expires_at - triple.starts_at, timedelta(minutes=15))
        self.assertEqual(triple.trigger_granularity, CandleGranularity.ONE_MINUTE)

    def test_triple_leveraged_etf_is_classified_from_its_local_etf_metadata(self):
        # Given: a locally tracked Direxion 3X ETF and a normal common stock.
        triple_etf = Stock(code="SOXL", name="Direxion Daily Semiconductor Bull 3X ETF", market="US", asset_type="etf")
        stock = Stock(code="TSLA", name="Tesla, Inc.", market="US", asset_type="stock")

        # When: the screener chooses its timeframe profile.
        triple_profile = classify_instrument(triple_etf)
        standard_profile = classify_instrument(stock)

        # Then: only the 3X ETF gets the 1m-required profile.
        self.assertEqual(triple_profile, InstrumentProfile.TRIPLE_LEVERAGED_ETF)
        self.assertEqual(standard_profile, InstrumentProfile.STANDARD)


class FixedLeverageFeasibilityTests(SimpleTestCase):
    def test_plan_is_rejected_when_total_five_x_stop_loss_including_costs_exceeds_cap(self):
        # Given: an 8.75 percent modeled stop that becomes 9.6 percent after the fixed public execution cost.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.STANDARD,
            trigger_granularity=CandleGranularity.FIVE_MINUTES,
            entry_price=100,
            target_price=101.4,
            stop_price=98.25,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
        )

        # When: the fixed 5x gate evaluates the all-in downside instead of the chart stop alone.
        feasibility = evaluate_fixed_leverage_feasibility(plan, taker_fee_rate=0.0006)

        # Then: it rejects the trade before an apparently narrow chart stop can understate liquidation risk.
        self.assertFalse(feasibility.passes)
        self.assertIn("total_stop_loss_cap_exceeded", feasibility.reasons)

    def test_plan_is_rejected_when_five_x_stop_loss_exceeds_fixed_cap(self):
        # Given: a technically valid plan with a 1.8 percent price stop.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.STANDARD,
            trigger_granularity=CandleGranularity.FIVE_MINUTES,
            entry_price=100,
            target_price=101.5,
            stop_price=98.2,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
        )

        # When: the fixed 5x feasibility check includes public taker costs.
        feasibility = evaluate_fixed_leverage_feasibility(plan, taker_fee_rate=0.0006)

        # Then: the plan cannot be presented as 5x-eligible.
        self.assertFalse(feasibility.passes)
        self.assertGreater(feasibility.estimated_margin_loss_at_stop_pct_before_fees, 8)

    def test_target_that_only_clears_an_underestimated_execution_cost_is_rejected(self):
        # Given: a technically narrow stop whose target would pass only with a 2bp round-trip slippage assumption.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.STANDARD,
            trigger_granularity=CandleGranularity.FIVE_MINUTES,
            entry_price=100,
            target_price=100.295,
            stop_price=99.8,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
        )

        # When: the 5x feasibility model includes the maximum permitted tight-spread execution buffer.
        feasibility = evaluate_fixed_leverage_feasibility(plan, taker_fee_rate=0.0006)

        # Then: a target that cannot survive realistic public-quote friction is not emitted.
        self.assertFalse(feasibility.passes)
        self.assertIn("post_cost_target_too_small", feasibility.reasons)


class QuotedExecutionCostTests(SimpleTestCase):
    def test_settlement_uses_the_recorded_quote_spread_when_calculating_post_cost_return(self):
        # Given: an entry recorded at a 10bp quote spread, wider than the generic 5bp floor.
        starts_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
        signal = IntradayLongSignal(
            contract_symbol="TSLAUSDT",
            underlying_symbol="TSLA",
            profile=InstrumentProfile.STANDARD.value,
            strategy_version=STRATEGY_VERSION,
            signal_opened_at=starts_at - timedelta(minutes=5),
            starts_at=starts_at,
            expires_at=starts_at + timedelta(minutes=60),
            entry_price="100",
            target_price="101",
            stop_price="99.8",
            taker_fee_rate=0.0006,
            entry_quote_observed_at=starts_at,
            entry_quote_spread_pct=0.10,
            estimated_margin_loss_at_stop_pct=4.0,
        )
        candle = Candle(
            opened_at=starts_at,
            open=100,
            high=101,
            low=99.9,
            close=101,
            quote_volume=1_000,
        )

        # When: the target-first result is settled.
        settlement = _settle(signal, (candle,))

        # Then: two taker fees plus the observed 10bp spread reduce the 5x return by 1.1%.
        self.assertEqual(settlement.outcome, SignalOutcome.TARGET)
        self.assertAlmostEqual(settlement.post_cost_margin_return_pct, 3.9)


class ForwardPromotionTests(TestCase):
    def test_profile_needs_minimum_settled_forward_evidence_before_five_x_promotion(self):
        # Given: a profile with a high hit rate but fewer than the fixed minimum observations.
        performance = ForwardPerformance(
            settled_signal_count=99,
            target_first_count=80,
            stop_first_count=19,
            unresolved_count=0,
            ambiguous_count=0,
            post_cost_expected_margin_return_pct=5.0,
            post_cost_margin_return_lower_bound_pct=3.0,
            worst_margin_loss_at_stop_pct=7.5,
        )

        # When: the forward promotion gate is evaluated.
        ready = promotion_is_ready(performance)

        # Then: insufficient independent forward observations block 5x eligibility.
        self.assertFalse(ready)

    def test_positive_mean_without_positive_lower_confidence_bound_is_not_five_x_eligible(self):
        # Given: 100 settled outcomes whose mean is positive but whose conservative lower bound is not.
        performance = ForwardPerformance(
            settled_signal_count=100,
            target_first_count=80,
            stop_first_count=20,
            unresolved_count=0,
            ambiguous_count=0,
            post_cost_expected_margin_return_pct=0.2,
            post_cost_margin_return_lower_bound_pct=-0.1,
            worst_margin_loss_at_stop_pct=7.0,
        )

        # When: the fixed 5x promotion gate evaluates the sample.
        ready = promotion_is_ready(performance)

        # Then: a noise-level apparent gain is not promoted as a manual entry strategy.
        self.assertFalse(ready)

    def test_promotion_does_not_pool_outcomes_from_a_different_strategy_version(self):
        # Given: a legacy strategy has a settled winner for the same symbol and profile.
        signal = IntradayLongSignal.objects.create(
            contract_symbol="TSLAUSDT",
            underlying_symbol="TSLA",
            profile=InstrumentProfile.STANDARD.value,
            strategy_version="legacy-unquoted-v1",
            signal_opened_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            starts_at=datetime(2026, 8, 12, 12, 5, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, 5, tzinfo=UTC),
            entry_price="100",
            target_price="101",
            stop_price="99.6",
            taker_fee_rate=0.0006,
            estimated_margin_loss_at_stop_pct=8,
            status=SignalOutcome.TARGET,
            terminal_price="101",
            post_cost_margin_return_pct=4.0,
            settled_at=datetime(2026, 8, 12, 13, 5, tzinfo=UTC),
        )

        # When: the current strategy version queries its forward evidence.
        current = promotion_for(InstrumentProfile.STANDARD, signal.contract_symbol, STRATEGY_VERSION)

        # Then: legacy outcomes cannot make the current 5x strategy appear proven.
        self.assertEqual(current.total.settled_signal_count, 0)

    def test_promotion_counts_all_in_stop_loss_not_only_the_chart_stop(self):
        # Given: 100 otherwise successful quoted observations whose 1.5 percent chart stop becomes 8.35 percent all-in.
        starts_at = datetime(2026, 8, 1, tzinfo=UTC)
        IntradayLongSignal.objects.bulk_create(
            [
                IntradayLongSignal(
                    contract_symbol="TSLAUSDT",
                    underlying_symbol="TSLA",
                    profile=InstrumentProfile.STANDARD.value,
                    strategy_version=STRATEGY_VERSION,
                    signal_opened_at=starts_at + timedelta(minutes=5 * index),
                    starts_at=starts_at + timedelta(minutes=5 * index),
                    expires_at=starts_at + timedelta(minutes=5 * index + 60),
                    entry_price="100",
                    target_price="101",
                    stop_price="98.5",
                    taker_fee_rate=0.0006,
                    entry_quote_observed_at=starts_at + timedelta(minutes=5 * index),
                    entry_quote_spread_pct=0.03,
                    estimated_margin_loss_at_stop_pct=7.5,
                    status=SignalOutcome.TARGET,
                    terminal_price="101",
                    post_cost_margin_return_pct=4.0,
                    settled_at=starts_at + timedelta(minutes=5 * index + 61),
                )
                for index in range(100)
            ]
        )

        # When: the versioned forward gate summarizes them.
        evaluation = promotion_for(InstrumentProfile.STANDARD, "TSLAUSDT", STRATEGY_VERSION)

        # Then: modeled execution costs keep the all-in downside above the 5x cap and block promotion.
        self.assertGreater(evaluation.total.worst_margin_loss_at_stop_pct, 8)
        self.assertFalse(promotion_evaluation_is_ready(evaluation))

    def test_promotion_uses_the_recorded_spread_when_checking_the_all_in_stop_cap(self):
        # Given: quoted entries with a 1.4 percent price stop and a 20bp spread that dominates the fixed floor.
        starts_at = datetime(2026, 8, 1, tzinfo=UTC)
        IntradayLongSignal.objects.bulk_create(
            [
                IntradayLongSignal(
                    contract_symbol="RGTIUSDT",
                    underlying_symbol="RGTI",
                    profile=InstrumentProfile.STANDARD.value,
                    strategy_version=STRATEGY_VERSION,
                    signal_opened_at=starts_at + timedelta(minutes=5 * index),
                    starts_at=starts_at + timedelta(minutes=5 * index),
                    expires_at=starts_at + timedelta(minutes=5 * index + 60),
                    entry_price="100",
                    target_price="101",
                    stop_price="98.6",
                    taker_fee_rate=0.0006,
                    entry_quote_observed_at=starts_at + timedelta(minutes=5 * index),
                    entry_quote_spread_pct=0.20,
                    estimated_margin_loss_at_stop_pct=7.0,
                    status=SignalOutcome.TARGET,
                    terminal_price="101",
                    post_cost_margin_return_pct=4.0,
                    settled_at=starts_at + timedelta(minutes=5 * index + 61),
                )
                for index in range(100)
            ]
        )

        # When: the quoted strategy's promotion evidence is summarized.
        evaluation = promotion_for(InstrumentProfile.STANDARD, "RGTIUSDT", STRATEGY_VERSION)

        # Then: actual 20bp execution friction makes the modeled 7% stop an 8.6% all-in loss and blocks promotion.
        self.assertAlmostEqual(evaluation.total.worst_margin_loss_at_stop_pct, 8.6)
        self.assertFalse(promotion_evaluation_is_ready(evaluation))

    def test_expired_persisted_signal_is_settled_from_completed_historical_one_minute_candles(self):
        # Given: a recorded standard-profile plan whose expiry has passed.
        starts_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
        signal = IntradayLongSignal.objects.create(
            contract_symbol="TSLAUSDT",
            underlying_symbol="TSLA",
            profile=InstrumentProfile.STANDARD.value,
            signal_opened_at=starts_at - timedelta(minutes=5),
            starts_at=starts_at,
            expires_at=starts_at + timedelta(minutes=60),
            entry_price="100",
            target_price="101",
            stop_price="99.6",
            taker_fee_rate=0.0006,
            estimated_margin_loss_at_stop_pct=8,
        )
        complete_window_rows = [
            [
                str(int((starts_at + timedelta(minutes=offset)).timestamp() * 1000)),
                "100",
                "101.2" if offset == 0 else "100.8",
                "99.9",
                "100.8",
                "10",
                "1000",
            ]
            for offset in range(60)
        ]
        payload = json.dumps(
            {
                "code": "00000",
                "data": complete_window_rows,
            }
        )

        # When: the forward-only settlement job receives the post-expiry historical candle.
        settled = settle_pending_signals(lambda _: payload, starts_at + timedelta(minutes=61))

        # Then: the immutable signal is marked target-first and receives a post-cost result.
        signal.refresh_from_db()
        self.assertEqual(settled, 1)
        self.assertEqual(signal.status, SignalOutcome.TARGET)
        self.assertIsNotNone(signal.post_cost_margin_return_pct)

    def test_incomplete_historical_window_remains_pending_instead_of_creating_a_false_outcome(self):
        # Given: an expired 60-minute signal but only the first historical one-minute candle is available.
        starts_at = datetime(2026, 8, 12, 12, tzinfo=UTC)
        signal = IntradayLongSignal.objects.create(
            contract_symbol="TSLAUSDT",
            underlying_symbol="TSLA",
            profile=InstrumentProfile.STANDARD.value,
            signal_opened_at=starts_at - timedelta(minutes=5),
            starts_at=starts_at,
            expires_at=starts_at + timedelta(minutes=60),
            entry_price="100",
            target_price="101",
            stop_price="99.6",
            taker_fee_rate=0.0006,
            estimated_margin_loss_at_stop_pct=8,
        )
        first_candle = int(starts_at.timestamp() * 1000)
        partial_payload = json.dumps(
            {
                "code": "00000",
                "data": [[str(first_candle), "100", "101.2", "99.9", "100.8", "10", "1000"]],
            }
        )

        # When: the settlement job sees an incomplete window that happens to touch the target.
        settled = settle_pending_signals(lambda _: partial_payload, starts_at + timedelta(minutes=61))

        # Then: it preserves the pending observation rather than treating partial history as a target-first win.
        signal.refresh_from_db()
        self.assertEqual(settled, 0)
        self.assertEqual(signal.status, SignalOutcome.PENDING)
