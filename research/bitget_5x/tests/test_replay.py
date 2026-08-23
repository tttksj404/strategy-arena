from datetime import UTC, datetime, timedelta

from unittest import TestCase

from research.bitget_5x.signal_outcomes import SignalOutcome
from research.bitget_5x.intraday_long_forecast import InstrumentProfile, IntradayCandleSet, plan_intraday_long
from research.bitget_5x.intraday_replay import (
    ReplayEntryFilter,
    ReplayRule,
    ReplayTrade,
    evaluate_replay_candidates,
    select_replay_candidate,
)
from research.bitget_5x.intraday_replay_engine import ReplayMarket, replay_market
from research.bitget_5x.long_forecast import Candle
from research.bitget_5x.market_timeframes import CandleGranularity


def _trades(
    rule: ReplayRule,
    target_count: int,
    stop_count: int,
    starts_at: datetime,
) -> tuple[ReplayTrade, ...]:
    outcomes = (SignalOutcome.TARGET,) * target_count + (SignalOutcome.STOP,) * stop_count
    return tuple(
        ReplayTrade(
            rule=rule,
            contract_symbol="TSLAUSDT",
            signal_opened_at=starts_at + timedelta(minutes=index),
            outcome=outcome,
            post_cost_margin_return_pct=8.0 if outcome is SignalOutcome.TARGET else -4.0,
            margin_loss_at_stop_pct=6.0,
        )
        for index, outcome in enumerate(outcomes)
    )


def _rising_candles(
    granularity: CandleGranularity,
    count: int,
    starts_at: datetime,
    increment: float,
) -> tuple[Candle, ...]:
    return tuple(
        Candle(
            opened_at=starts_at + granularity.interval * index,
            open=100 + increment * index,
            high=100 + increment * (index + 1),
            low=100 + increment * index - increment * 0.2,
            close=100 + increment * (index + 0.8),
            quote_volume=100,
        )
        for index in range(count)
    )


def _squeeze_breakout_candles(
    granularity: CandleGranularity,
    count: int,
    starts_at: datetime,
) -> tuple[Candle, ...]:
    candles: list[Candle] = []
    for index in range(count):
        close = 100 + index * 0.05
        if index < count - 40:
            close += 1.5 if index % 2 else -1.5
        half_range = 2.0 if index < count - 40 else 0.08
        if index == count - 1:
            close += 0.4
        candles.append(
            Candle(
                opened_at=starts_at + granularity.interval * index,
                open=close - half_range * 0.2,
                high=close + half_range,
                low=close - half_range,
                close=close,
                quote_volume=150 if index == count - 1 else 100,
            )
        )
    return tuple(candles)


class ReplaySelectionTests(TestCase):
    def test_candidate_diagnostics_show_each_rule_without_changing_validation_only_selection(self):
        starts_at = datetime(2026, 7, 1, tzinfo=UTC)
        candidate_a = ReplayRule("breakout-volume-1.0", 1.0)
        candidate_b = ReplayRule("breakout-volume-1.5", 1.5)
        trades_by_rule = {
            candidate_a.name: _trades(candidate_a, 64, 16, starts_at)
            + _trades(candidate_a, 0, 20, starts_at + timedelta(minutes=80)),
            candidate_b.name: _trades(candidate_b, 60, 20, starts_at)
            + _trades(candidate_b, 20, 0, starts_at + timedelta(minutes=80)),
        }

        diagnostics = evaluate_replay_candidates(trades_by_rule)

        self.assertTrue(diagnostics[candidate_a.name].validation_passes)
        self.assertTrue(diagnostics[candidate_b.name].validation_passes)

    def test_selects_on_validation_only_and_rejects_when_the_selected_rule_fails_untouched_holdout(self):
        # Given: candidate A wins the validation period while candidate B only wins because of its later holdout.
        starts_at = datetime(2026, 7, 1, tzinfo=UTC)
        candidate_a = ReplayRule("breakout-volume-1.5", 1.5)
        candidate_b = ReplayRule("breakout-volume-2.0", 2.0)
        candidate_a_trades = _trades(candidate_a, 64, 16, starts_at) + _trades(
            candidate_a,
            0,
            20,
            starts_at + timedelta(minutes=80),
        )
        candidate_b_trades = _trades(candidate_b, 60, 20, starts_at) + _trades(
            candidate_b,
            20,
            0,
            starts_at + timedelta(minutes=80),
        )

        # When: the fixed replay evaluator selects exactly one candidate before reading a promotion verdict.
        result = select_replay_candidate(
            {
                candidate_a.name: candidate_a_trades,
                candidate_b.name: candidate_b_trades,
            }
        )

        # Then: it picks validation winner A and rejects it instead of swapping to B after seeing holdout results.
        self.assertEqual(result.selected_rule_name, candidate_a.name)
        self.assertFalse(result.is_reliable)
        self.assertEqual(result.held_out.stop_first_count, 20)

    def test_insufficient_validation_sample_returns_no_reliable_winner(self):
        # Given: a seemingly perfect but tiny replay sample.
        rule = ReplayRule("breakout-volume-1.5", 1.5)

        # When: the evaluator sees fewer than the fixed validation minimum.
        result = select_replay_candidate({rule.name: _trades(rule, 19, 0, datetime(2026, 7, 1, tzinfo=UTC))})

        # Then: it does not manufacture a 5x-ready historical winner.
        self.assertIsNone(result.selected_rule_name)
        self.assertFalse(result.is_reliable)


class ReplayEngineTests(TestCase):
    def test_bollinger_squeeze_filter_replays_only_a_causal_low_volatility_breakout(self):
        # Given: a completed uptrend whose final 5m range is narrow relative to the earlier rolling-width regime.
        starts_at = datetime(2026, 7, 1, tzinfo=UTC)
        market = ReplayMarket(
            contract_symbol="TSLAUSDT",
            profile=InstrumentProfile.STANDARD,
            taker_fee_rate=0.0006,
            context_15m=_rising_candles(CandleGranularity.FIFTEEN_MINUTES, 80, starts_at, 0.39),
            confirmation_5m=_squeeze_breakout_candles(CandleGranularity.FIVE_MINUTES, 240, starts_at),
            trigger_1m=(),
            settlement_1m=_rising_candles(CandleGranularity.ONE_MINUTE, 1300, starts_at, 0.026),
        )
        rule = ReplayRule(
            "breakout-bollinger-squeeze-v1-volume-1.0",
            1.0,
            ReplayEntryFilter.BOLLINGER_SQUEEZE,
        )

        # When: the replay combines the existing breakout with a completed-candle Bollinger squeeze filter.
        trades = replay_market(market, rule)

        # Then: only the shared 5x plan's settled research trades are returned.
        self.assertGreater(len(trades), 0)
        self.assertTrue(all(trade.rule.entry_filter is ReplayEntryFilter.BOLLINGER_SQUEEZE for trade in trades))

    def test_technical_score_filter_replays_only_after_the_existing_causal_candidate_confirms(self):
        # Given: a completed multi-timeframe breakout with a previously completed swing-low invalidation.
        starts_at = datetime(2026, 7, 1, tzinfo=UTC)
        confirmation = list(_rising_candles(CandleGranularity.FIVE_MINUTES, 240, starts_at, 0.13))
        pivot = confirmation[-5]
        confirmation[-5] = Candle(
            opened_at=pivot.opened_at,
            open=pivot.open,
            high=pivot.high,
            low=pivot.low - 1.0,
            close=pivot.close,
            quote_volume=pivot.quote_volume,
        )
        market = ReplayMarket(
            contract_symbol="TSLAUSDT",
            profile=InstrumentProfile.STANDARD,
            taker_fee_rate=0.0006,
            context_15m=_rising_candles(CandleGranularity.FIFTEEN_MINUTES, 80, starts_at, 0.39),
            confirmation_5m=tuple(confirmation),
            trigger_1m=(),
            settlement_1m=_rising_candles(CandleGranularity.ONE_MINUTE, 1300, starts_at, 0.026),
        )
        rule = ReplayRule(
            "breakout-technical-score-v1-volume-1.0",
            1.0,
            ReplayEntryFilter.TECHNICAL_SCORE,
        )

        # When: the replay uses the prior technical-score research candidate only as a causal entry filter.
        trades = replay_market(market, rule)

        # Then: it yields settled research trades without changing the shared 5x exit/cost model.
        self.assertGreater(len(trades), 0)
        self.assertTrue(all(trade.rule.entry_filter is ReplayEntryFilter.TECHNICAL_SCORE for trade in trades))

    def test_candidate_volume_ratio_is_applied_by_the_same_plan_builder_as_live_screening(self):
        starts_at = datetime(2026, 7, 1, tzinfo=UTC)
        window = IntradayCandleSet(
            context_15m=_rising_candles(CandleGranularity.FIFTEEN_MINUTES, 80, starts_at, 0.39),
            confirmation_5m=_rising_candles(CandleGranularity.FIVE_MINUTES, 80, starts_at, 0.13),
            trigger_1m=_rising_candles(CandleGranularity.ONE_MINUTE, 80, starts_at, 0.026),
        )

        self.assertIsNotNone(plan_intraday_long(InstrumentProfile.STANDARD, window, minimum_volume_ratio=1.0))
        self.assertIsNone(plan_intraday_long(InstrumentProfile.STANDARD, window, minimum_volume_ratio=1.25))

    def test_stricter_volume_rule_excludes_flat_volume_breakouts_without_using_future_candles(self):
        # Given: causal multi-timeframe uptrend data whose closed trigger volumes only equal their prior average.
        starts_at = datetime(2026, 7, 1, tzinfo=UTC)
        market = ReplayMarket(
            contract_symbol="TSLAUSDT",
            profile=InstrumentProfile.STANDARD,
            taker_fee_rate=0.0006,
            context_15m=_rising_candles(CandleGranularity.FIFTEEN_MINUTES, 80, starts_at, 0.39),
            confirmation_5m=_rising_candles(CandleGranularity.FIVE_MINUTES, 240, starts_at, 0.13),
            trigger_1m=_rising_candles(CandleGranularity.ONE_MINUTE, 1, starts_at, 0.026),
            settlement_1m=_rising_candles(CandleGranularity.ONE_MINUTE, 1260, starts_at, 0.026),
        )

        # When: the baseline and precommitted stricter volume rules replay only completed decision bars.
        baseline = replay_market(market, ReplayRule("breakout-volume-1.0", 1.0))
        strict_volume = replay_market(market, ReplayRule("breakout-volume-1.5", 1.5))

        # Then: the strict rule rejects the low-quality volume surface instead of reusing later outcome candles.
        self.assertGreater(len(baseline), 0)
        self.assertEqual(strict_volume, ())
