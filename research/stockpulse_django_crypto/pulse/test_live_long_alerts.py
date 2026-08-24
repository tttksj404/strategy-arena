import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from django.test import SimpleTestCase

from pulse.services.bitget_intraday_screener import IntradayScreenReport, LocalInstrument, RawCandidate, ScreenCandidate
from pulse.services.bitget_market import BitgetContract, BitgetPayloadError
from pulse.services.bitget_ticker import BitgetTicker, fetch_live_ticker
from pulse.services.intraday_long_forecast import (
    IntradayLongPlan,
    InstrumentProfile,
    evaluate_fixed_leverage_feasibility,
)
from pulse.services.intraday_signal_ledger import ForwardPerformance, ForwardPromotion
from pulse.services.live_long_instruction import LiveEntryState, build_live_long_instruction
from pulse.services.live_long_signal_feed import LiveLongSignal, LiveLongSignalFeed, screen_bitget_live_long_alerts
from pulse.services.live_long_research import collect_research_observations
from pulse.services.market_timeframes import CandleGranularity


def _long_plan() -> IntradayLongPlan:
    return IntradayLongPlan(
        profile=InstrumentProfile.STANDARD,
        trigger_granularity=CandleGranularity.FIVE_MINUTES,
        entry_price=100,
        target_price=100.9,
        stop_price=99.8,
        score=90,
        starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
        expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
    )


def _current_quote(ask_price: float = 100.03) -> BitgetTicker:
    return BitgetTicker(
        symbol="TSLAUSDT",
        last_price=ask_price - 0.01,
        bid_price=ask_price - 0.03,
        ask_price=ask_price,
        observed_at=datetime(2026, 8, 12, 12, 1, tzinfo=UTC),
    )


def _ready_performance(count: int, target_first_count: int) -> ForwardPerformance:
    return ForwardPerformance(
        settled_signal_count=count,
        target_first_count=target_first_count,
        stop_first_count=count - target_first_count,
        unresolved_count=0,
        ambiguous_count=0,
        post_cost_expected_margin_return_pct=1.0,
        post_cost_margin_return_lower_bound_pct=0.5,
        worst_margin_loss_at_stop_pct=7.0,
    )


def _ready_promotion() -> ForwardPromotion:
    return ForwardPromotion(
        total=_ready_performance(100, 88),
        validation=_ready_performance(80, 70),
        held_out=_ready_performance(20, 18),
    )


class LiveLongInstructionTests(SimpleTestCase):
    def test_current_ask_inside_entry_range_emits_exact_enter_now_instruction(self):
        # Given: a still-valid closed-candle long plan and a current public ask inside its no-chase range.
        plan = _long_plan()
        quote = _current_quote()

        # When: the live entry contract evaluates the current quote at 5x.
        instruction = build_live_long_instruction(plan, quote, taker_fee_rate=0.0006, now=quote.observed_at)

        # Then: it tells the user exactly where to enter, protect, partially exit, and fully exit.
        self.assertEqual(instruction.state, LiveEntryState.ENTER_NOW)
        self.assertEqual(instruction.entry_price, 100.03)
        self.assertEqual(instruction.stop_price, 99.8)
        self.assertEqual(instruction.final_take_profit_price, 100.9)
        self.assertGreater(instruction.partial_take_profit_price, instruction.entry_price)
        self.assertLess(instruction.partial_take_profit_price, instruction.final_take_profit_price)

    def test_quote_above_entry_ceiling_never_emits_enter_now(self):
        # Given: the same plan, but a live ask that has already run beyond its no-chase price ceiling.
        plan = _long_plan()
        quote = _current_quote(ask_price=100.2)

        # When: the live entry contract evaluates the chased quote.
        instruction = build_live_long_instruction(plan, quote, taker_fee_rate=0.0006, now=quote.observed_at)

        # Then: no enter-now instruction is produced.
        self.assertEqual(instruction.state, LiveEntryState.SKIP_CHASED)
        self.assertFalse(instruction.is_enter_now)

    def test_wide_bid_ask_spread_never_emits_enter_now_when_actual_cost_breaks_the_five_x_gate(self):
        # Given: a valid chart plan but a current quote with a 13bp bid-ask spread.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.STANDARD,
            trigger_granularity=CandleGranularity.FIVE_MINUTES,
            entry_price=100,
            target_price=102,
            stop_price=98.5,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
        )
        quote = BitgetTicker(
            symbol="TSLAUSDT",
            last_price=99.98,
            bid_price=99.9,
            ask_price=100.03,
            observed_at=datetime(2026, 8, 12, 12, 1, tzinfo=UTC),
        )

        # When: the 5x entry contract evaluates the execution surface.
        instruction = build_live_long_instruction(plan, quote, taker_fee_rate=0.0006, now=quote.observed_at)

        # Then: the candidate is blocked because the actual quote cost pushes the all-in loss above the cap.
        self.assertEqual(instruction.state, LiveEntryState.SKIP_RISK_GATE)
        self.assertFalse(instruction.is_enter_now)

    def test_risk_gate_precedes_waiting_for_a_lower_entry_price(self):
        # Given: a quote below the entry floor whose actual 5x spread cost already breaks the loss cap.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.STANDARD,
            trigger_granularity=CandleGranularity.FIVE_MINUTES,
            entry_price=100,
            target_price=102,
            stop_price=98.5,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
        )
        quote = BitgetTicker(
            symbol="TSLAUSDT",
            last_price=99.65,
            bid_price=99.45,
            ask_price=99.85,
            observed_at=datetime(2026, 8, 12, 12, 1, tzinfo=UTC),
        )

        # When: the price would otherwise be presented as a possible rebound entry.
        instruction = build_live_long_instruction(plan, quote, taker_fee_rate=0.0006, now=quote.observed_at)

        # Then: an infeasible 5x setup is blocked instead of being published as a waiting setup.
        self.assertEqual(instruction.state, LiveEntryState.SKIP_RISK_GATE)
        self.assertIn("total_stop_loss_cap_exceeded", instruction.reasons)

    def test_executable_quote_spread_is_priced_into_the_five_x_gate_instead_of_a_fixed_cutoff(self):
        # Given: a 7bp public spread whose actual all-in 5x loss and net target still clear the fixed envelope.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.STANDARD,
            trigger_granularity=CandleGranularity.FIVE_MINUTES,
            entry_price=100,
            target_price=101.0,
            stop_price=99.85,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
        )
        quote = BitgetTicker(
            symbol="TSLAUSDT",
            last_price=100.0,
            bid_price=99.96,
            ask_price=100.03,
            observed_at=datetime(2026, 8, 12, 12, 1, tzinfo=UTC),
        )

        # When: the quote is evaluated at the actual spread, not an arbitrary 5bp binary rule.
        instruction = build_live_long_instruction(plan, quote, taker_fee_rate=0.0006, now=quote.observed_at)

        # Then: the setup remains eligible and reports the higher real execution cost.
        self.assertEqual(instruction.state, LiveEntryState.ENTER_NOW)
        self.assertGreater(instruction.feasibility.estimated_round_trip_margin_cost_pct, 0.9)
        self.assertLessEqual(instruction.feasibility.estimated_all_in_margin_loss_at_stop_pct, 8.0)

    def test_collapsed_entry_range_is_not_misreported_as_waiting_for_entry(self):
        # Given: actual execution cost that makes the target-based ceiling fall below the no-chase entry floor.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.STANDARD,
            trigger_granularity=CandleGranularity.FIVE_MINUTES,
            entry_price=100,
            target_price=100.1,
            stop_price=99.55,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
        )
        quote = BitgetTicker(
            symbol="TSLAUSDT",
            last_price=99.73,
            bid_price=99.70,
            ask_price=99.75,
            observed_at=datetime(2026, 8, 12, 12, 1, tzinfo=UTC),
        )

        # When: the current quote is evaluated.
        instruction = build_live_long_instruction(plan, quote, taker_fee_rate=0.0006, now=quote.observed_at)

        # Then: a non-overlapping range is rejected rather than displayed as a buy-on-rebound setup.
        self.assertEqual(instruction.state, LiveEntryState.SKIP_RISK_GATE)
        self.assertIn("entry_range_collapsed_by_actual_execution_cost", instruction.reasons)

    def test_triple_leveraged_one_minute_trigger_expires_before_a_delayed_manual_entry(self):
        # Given: a 3x ETF plan whose required one-minute trigger closed more than one minute ago.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.TRIPLE_LEVERAGED_ETF,
            trigger_granularity=CandleGranularity.ONE_MINUTE,
            entry_price=100,
            target_price=100.9,
            stop_price=99.8,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 12, 15, tzinfo=UTC),
        )
        quote = _current_quote()

        # When: a fresh quote arrives after the one-minute trigger window, but before the wider time exit.
        instruction = build_live_long_instruction(
            plan,
            quote,
            taker_fee_rate=0.0006,
            now=plan.starts_at + timedelta(minutes=1, seconds=1),
        )

        # Then: quote freshness cannot revive a stale 1m setup as a 5x entry.
        self.assertEqual(instruction.state, LiveEntryState.SKIP_TRIGGER_STALE)
        self.assertFalse(instruction.is_enter_now)


class BitgetTickerBoundaryTests(SimpleTestCase):
    def test_public_ticker_parses_the_contract_quote_and_timestamp(self):
        # Given: a valid one-contract public ticker response.
        now = datetime(2026, 8, 12, 12, 0, 5, tzinfo=UTC)
        contract = BitgetContract("TSLAUSDT", "TSLA", 50, 8, 0.0006)
        payload = json.dumps(
            {
                "code": "00000",
                "data": [
                    {
                        "symbol": "TSLAUSDT",
                        "lastPr": "100.02",
                        "bidPr": "100.00",
                        "askPr": "100.03",
                        "ts": str(int(now.timestamp() * 1000)),
                    }
                ],
            }
        )
        requested_urls: list[str] = []

        # When: the read-only ticker adapter fetches the contract quote.
        quote = fetch_live_ticker(contract, lambda url: requested_urls.append(url) or payload, now)

        # Then: the exact Bitget contract and quote time are retained for the entry decision.
        self.assertEqual(quote.symbol, "TSLAUSDT")
        self.assertEqual(quote.ask_price, 100.03)
        self.assertEqual(quote.observed_at, now)
        self.assertIn("/api/v2/mix/market/ticker", requested_urls[0])

    def test_stale_public_ticker_cannot_be_used_for_an_entry(self):
        # Given: a syntactically valid quote whose exchange timestamp is older than the live-entry freshness limit.
        now = datetime(2026, 8, 12, 12, 0, 30, tzinfo=UTC)
        contract = BitgetContract("TSLAUSDT", "TSLA", 50, 8, 0.0006)
        payload = json.dumps(
            {
                "code": "00000",
                "data": [
                    {
                        "symbol": "TSLAUSDT",
                        "lastPr": "100.02",
                        "bidPr": "100.00",
                        "askPr": "100.03",
                        "ts": str(int((now.replace(second=0) - timedelta(seconds=16)).timestamp() * 1000)),
                    }
                ],
            }
        )

        # When: the ticker adapter receives the stale response.
        # Then: it rejects it before a manual entry instruction can be made.
        with self.assertRaises(BitgetPayloadError):
            fetch_live_ticker(contract, lambda _: payload, now)


class LiveSignalFeedTests(SimpleTestCase):
    def test_enter_now_feed_has_direct_entry_exit_and_forward_evidence_status(self):
        # Given: an enter-now instruction that has passed quote/risk and independent forward-evidence gates.
        plan = _long_plan()
        quote = _current_quote()
        feasibility = evaluate_fixed_leverage_feasibility(plan, taker_fee_rate=0.0006)
        candidate = ScreenCandidate(
            raw=RawCandidate(
                contract=BitgetContract("TSLAUSDT", "TSLA", 50, 8, 0.0006),
                instrument=LocalInstrument("TSLA", "Tesla, Inc.", InstrumentProfile.STANDARD),
                plan=plan,
                feasibility=feasibility,
            ),
            forward=_ready_promotion(),
        )
        instruction = build_live_long_instruction(plan, quote, taker_fee_rate=0.0006, now=quote.observed_at)
        base_report = IntradayScreenReport(
            as_of=quote.observed_at,
            eligible_contract_count=1,
            scanned_contract_count=1,
            candidates=(candidate,),
            rejection_counts=(),
        )
        feed = LiveLongSignalFeed(
            base_report=base_report,
            signals=(LiveLongSignal(candidate, quote, instruction),),
            ticker_rejection_counts=(),
        )

        # When: the read-only feed serializes the enter-now signal.
        payload = feed.payload()

        # Then: the top-level alert says exactly how to enter and exit only after the forward gate has cleared.
        alert = payload["enter_now_signals"][0]
        self.assertEqual(alert["action"], "ENTER_LONG")
        self.assertEqual(alert["entry"]["price"], 100.03)
        self.assertEqual(alert["exit"]["stop_loss_price"], 99.8)
        self.assertEqual(alert["exit"]["take_profit_2_price"], 100.9)
        self.assertEqual(alert["research_status"], "forward_evidence_ready")
        self.assertEqual(alert["forward_evidence"]["post_cost_margin_return_lower_bound_pct"], 0.5)
        self.assertEqual(alert["estimated_5x"]["estimated_all_in_margin_loss_at_stop_pct"], 2.0)

    def test_unproven_candidate_is_not_published_as_a_live_manual_entry_signal(self):
        # Given: a setup whose current quote fits but whose chronological forward evidence is still empty.
        plan = _long_plan()
        quote = _current_quote()
        feasibility = evaluate_fixed_leverage_feasibility(plan, taker_fee_rate=0.0006)
        zero_performance = ForwardPerformance(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
        candidate = ScreenCandidate(
            raw=RawCandidate(
                contract=BitgetContract("TSLAUSDT", "TSLA", 50, 8, 0.0006),
                instrument=LocalInstrument("TSLA", "Tesla, Inc.", InstrumentProfile.STANDARD),
                plan=plan,
                feasibility=feasibility,
            ),
            forward=ForwardPromotion(zero_performance, zero_performance, zero_performance),
        )
        report = IntradayScreenReport(
            as_of=quote.observed_at,
            eligible_contract_count=1,
            scanned_contract_count=1,
            candidates=(candidate,),
            rejection_counts=(),
        )

        # When: the live feed is constructed.
        with patch("pulse.services.live_long_signal_feed.screen_bitget_intraday_longs", return_value=report), patch(
            "pulse.services.live_long_signal_feed.fetch_live_ticker", return_value=quote
        ) as fetch_ticker:
            payload = screen_bitget_live_long_alerts(now=quote.observed_at).payload()

        # Then: it is visible as a research watch setup, never as a 5x entry instruction.
        self.assertEqual(payload["research_pending_count"], 1)
        self.assertEqual(payload["enter_now_count"], 0)
        self.assertEqual(payload["candidates"], [])
        self.assertEqual(payload["watch_candidate_count"], 1)
        self.assertEqual(payload["watch_candidates"][0]["action"], "WATCH_LONG_SETUP")
        fetch_ticker.assert_called_once()

    def test_current_unproven_setup_is_exposed_as_a_ranked_watch_candidate_not_an_entry(self):
        # Given: a forward-pending setup whose fresh quote meets the actual 5x cost envelope.
        plan = IntradayLongPlan(
            profile=InstrumentProfile.STANDARD,
            trigger_granularity=CandleGranularity.FIVE_MINUTES,
            entry_price=100,
            target_price=101.0,
            stop_price=99.85,
            score=90,
            starts_at=datetime(2026, 8, 12, 12, tzinfo=UTC),
            expires_at=datetime(2026, 8, 12, 13, tzinfo=UTC),
        )
        quote = BitgetTicker(
            symbol="TSLAUSDT",
            last_price=100.0,
            bid_price=99.96,
            ask_price=100.03,
            observed_at=datetime(2026, 8, 12, 12, 1, tzinfo=UTC),
        )
        feasibility = evaluate_fixed_leverage_feasibility(plan, taker_fee_rate=0.0006)
        zero_performance = ForwardPerformance(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
        candidate = ScreenCandidate(
            raw=RawCandidate(
                contract=BitgetContract("TSLAUSDT", "TSLA", 50, 8, 0.0006),
                instrument=LocalInstrument("TSLA", "Tesla, Inc.", InstrumentProfile.STANDARD),
                plan=plan,
                feasibility=feasibility,
            ),
            forward=ForwardPromotion(zero_performance, zero_performance, zero_performance),
        )
        report = IntradayScreenReport(
            as_of=quote.observed_at,
            eligible_contract_count=1,
            scanned_contract_count=1,
            candidates=(candidate,),
            rejection_counts=(),
        )

        # When: the read-only feed is requested.
        with patch("pulse.services.live_long_signal_feed.screen_bitget_intraday_longs", return_value=report), patch(
            "pulse.services.live_long_signal_feed.fetch_live_ticker", return_value=quote
        ):
            payload = screen_bitget_live_long_alerts(now=quote.observed_at).payload()

        # Then: it is visible for ranking and levels, but never promoted to an ENTER_LONG instruction.
        self.assertEqual(payload["enter_now_signals"], [])
        self.assertEqual(payload["watch_candidate_count"], 1)
        self.assertEqual(payload["watch_candidates"][0]["action"], "WATCH_LONG_SETUP")
        self.assertEqual(payload["watch_candidates"][0]["symbol"], "TSLAUSDT")


class ResearchObservationTests(SimpleTestCase):
    def test_research_observation_records_actual_ask_and_quote_spread_without_publishing_an_entry(self):
        # Given: a forward-pending chart setup with a current tight public quote inside its entry zone.
        plan = _long_plan()
        feasibility = evaluate_fixed_leverage_feasibility(plan, taker_fee_rate=0.0006)
        zero_performance = ForwardPerformance(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
        candidate = ScreenCandidate(
            raw=RawCandidate(
                contract=BitgetContract("TSLAUSDT", "TSLA", 50, 8, 0.0006),
                instrument=LocalInstrument("TSLA", "Tesla, Inc.", InstrumentProfile.STANDARD),
                plan=plan,
                feasibility=feasibility,
            ),
            forward=ForwardPromotion(zero_performance, zero_performance, zero_performance),
        )
        report = IntradayScreenReport(
            as_of=datetime(2026, 8, 12, 12, 1, tzinfo=UTC),
            eligible_contract_count=1,
            scanned_contract_count=1,
            candidates=(candidate,),
            rejection_counts=(),
        )
        ticker_payload = json.dumps(
            {
                "code": "00000",
                "data": [
                    {
                        "symbol": "TSLAUSDT",
                        "lastPr": "100.02",
                        "bidPr": "100.00",
                        "askPr": "100.03",
                        "ts": str(int(report.as_of.timestamp() * 1000)),
                    }
                ],
            }
        )

        # When: the research collector snapshots the executable public quote.
        collected = collect_research_observations(report, lambda _: ticker_payload, report.as_of)

        # Then: it records the actual ask and spread for later settlement without making a live alert.
        observation = collected.observations[0]
        self.assertEqual(len(collected.observations), 1)
        self.assertEqual(observation.plan.entry_price, 100.03)
        self.assertAlmostEqual(observation.entry_quote_spread_pct, 0.03, places=4)
