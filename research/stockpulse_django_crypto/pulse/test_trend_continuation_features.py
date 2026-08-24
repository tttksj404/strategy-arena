from __future__ import annotations

from datetime import UTC, datetime

from django.test import SimpleTestCase

from pulse.services.long_forecast import Candle
from pulse.services.market_timeframes import CandleGranularity
from pulse.services.trend_continuation_contract import Direction, ResearchDecision
from pulse.services.trend_continuation_features import (
    STANDARD_PROFILE,
    TECHNICAL_SCORE_RULE_NAME,
    TRIPLE_PROFILE,
    FeatureInputError,
    analyze_trend_continuation,
    build_features,
)


EVIDENCE_TIME = datetime(2026, 8, 18, 3, 0, tzinfo=UTC)


def _close(index: int, *, base: float, slope: float) -> float:
    return base + index * slope + ((index % 4) - 1.5) * 0.35


def _candles(
    granularity: CandleGranularity,
    *,
    base: float = 100.0,
    slope: float = 0.25,
    count: int = 60,
    volume: float = 1000.0,
    pivot: str = "low",
) -> tuple[Candle, ...]:
    start = EVIDENCE_TIME - granularity.interval * count
    candles: list[Candle] = []
    for index in range(count):
        close = _close(index, base=base, slope=slope)
        candles.append(
            Candle(
                opened_at=start + granularity.interval * index,
                open=close - 0.1,
                high=close + 0.9,
                low=close - 0.9,
                close=close,
                quote_volume=volume + (index % 5) * 10,
            )
        )
    pivot_index = count - 5
    candle = candles[pivot_index]
    match pivot:
        case "low":
            candles[pivot_index] = Candle(
                candle.opened_at,
                candle.open,
                candle.high,
                candle.low - 2.0,
                candle.close,
                candle.quote_volume,
            )
        case "high":
            candles[pivot_index] = Candle(
                candle.opened_at,
                candle.open,
                candle.high + 2.0,
                candle.low,
                candle.close,
                candle.quote_volume,
            )
        case "none":
            pass
        case _:
            raise AssertionError("unknown pivot fixture")
    return tuple(candles)


class TrendContinuationFeatureVectorTests(SimpleTestCase):
    def test_known_vectors_use_completed_candles_and_correct_macd_signal(self):
        # Given: a fixed 60-bar completed 5m series ending exactly at evidence_time.
        candles = _candles(CandleGranularity.FIVE_MINUTES)

        # When: v1 point-in-time features are built.
        features = build_features(candles, EVIDENCE_TIME, CandleGranularity.FIVE_MINUTES)

        # Then: formulas match known values, including MACD signal over MACD-line history.
        self.assertAlmostEqual(features.sma20, 112.375, places=6)
        self.assertAlmostEqual(features.sma50, 108.639, places=6)
        self.assertAlmostEqual(features.sma20_slope, 0.25, places=6)
        self.assertAlmostEqual(features.ema12, 113.447490, places=6)
        self.assertAlmostEqual(features.ema26, 111.658120, places=6)
        self.assertAlmostEqual(features.macd_line, 1.789371, places=6)
        self.assertAlmostEqual(features.macd_signal, 1.755678, places=6)
        self.assertAlmostEqual(features.macd_hist, 0.033693, places=6)
        self.assertAlmostEqual(features.rsi14, 72.437895, places=6)
        self.assertAlmostEqual(features.roc1, 0.523218, places=6)
        self.assertAlmostEqual(features.bollinger_width20, 5.571515, places=6)
        self.assertAlmostEqual(features.relative_volume20, 1.019608, places=6)
        self.assertAlmostEqual(features.atr14, 1.906209, places=6)
        self.assertAlmostEqual(features.adx14, 35.812713, places=6)
        self.assertNotEqual(round(features.macd_signal, 6), round(features.closes[-9], 6))
        self.assertEqual({record.timeframe for record in features.feature_records}, {"5m"})
        self.assertTrue(all(record.evidence_time == EVIDENCE_TIME for record in features.feature_records))
        self.assertTrue(all(record.complete for record in features.feature_records))

    def test_standard_profile_finds_up_continuation_candidate_with_completed_pivot_invalidation(self):
        # Given: aligned 15m and 5m completed up-regime fixtures.
        context = _candles(CandleGranularity.FIFTEEN_MINUTES)
        confirmation = _candles(CandleGranularity.FIVE_MINUTES)

        # When: the technical-score rule analyzes the standard profile.
        result = analyze_trend_continuation(
            profile=STANDARD_PROFILE,
            context_15m=context,
            confirmation_5m=confirmation,
            evidence_time=EVIDENCE_TIME,
        )

        # Then: it is an observation candidate, not an instruction.
        self.assertEqual(result.decision, ResearchDecision.CONTINUATION_OBSERVE)
        self.assertEqual(result.trend_direction, Direction.UP)
        self.assertEqual(result.rule_name, TECHNICAL_SCORE_RULE_NAME)
        self.assertEqual(result.candidate_horizons, (6, 12, 18))
        self.assertEqual(result.candidate_horizon_bars, 12)
        self.assertGreaterEqual(result.score, 70)
        self.assertIsNotNone(result.reference_price)
        self.assertIsNotNone(result.invalidation_price)
        self.assertLess(result.invalidation_price, result.reference_price)
        self.assertIsNone(result.rejection_code)

    def test_down_profile_is_mirrored_and_uses_swing_high_invalidation(self):
        # Given: aligned downward fixtures with a completed 2-left/2-right high pivot.
        context = _candles(CandleGranularity.FIFTEEN_MINUTES, base=120.0, slope=-0.25, count=61, pivot="high")
        confirmation = _candles(CandleGranularity.FIVE_MINUTES, base=120.0, slope=-0.25, count=61, pivot="high")

        # When: the same rule analyzes the mirrored direction.
        result = analyze_trend_continuation(
            profile=STANDARD_PROFILE,
            context_15m=context,
            confirmation_5m=confirmation,
            evidence_time=EVIDENCE_TIME,
        )

        # Then: direction and invalidation mirror the UP case.
        self.assertEqual(result.decision, ResearchDecision.CONTINUATION_OBSERVE)
        self.assertEqual(result.trend_direction, Direction.DOWN)
        self.assertEqual(result.candidate_horizon_bars, 12)
        self.assertGreaterEqual(result.score, 70)
        self.assertIsNotNone(result.reference_price)
        self.assertIsNotNone(result.invalidation_price)
        self.assertGreater(result.invalidation_price, result.reference_price)

    def test_triple_profile_requires_1m_side_and_rvol_gate(self):
        # Given: aligned 15m/5m fixtures and a confirming 1m trigger.
        context = _candles(CandleGranularity.FIFTEEN_MINUTES)
        confirmation = _candles(CandleGranularity.FIVE_MINUTES)
        trigger = _candles(CandleGranularity.ONE_MINUTE)

        # When / Then: triple profile uses the 1m trigger and default 10-bar horizon.
        passed = analyze_trend_continuation(
            profile=TRIPLE_PROFILE,
            context_15m=context,
            confirmation_5m=confirmation,
            trigger_1m=trigger,
            evidence_time=EVIDENCE_TIME,
        )
        self.assertEqual(passed.decision, ResearchDecision.CONTINUATION_OBSERVE)
        self.assertEqual(passed.candidate_horizons, (5, 10, 15))
        self.assertEqual(passed.candidate_horizon_bars, 10)

        failed = analyze_trend_continuation(
            profile=TRIPLE_PROFILE,
            context_15m=context,
            confirmation_5m=confirmation,
            trigger_1m=_candles(CandleGranularity.ONE_MINUTE, base=120.0, slope=-0.25, pivot="high"),
            evidence_time=EVIDENCE_TIME,
        )
        self.assertEqual(failed.decision, ResearchDecision.NO_TRADE)
        self.assertEqual(failed.trend_direction, Direction.NEUTRAL)

    def test_future_candle_input_fails_closed_instead_of_changing_past_features(self):
        # Given: a valid completed series and the same series with one post-evidence candle appended.
        candles = _candles(CandleGranularity.FIVE_MINUTES)
        future = Candle(
            opened_at=EVIDENCE_TIME,
            open=200.0,
            high=210.0,
            low=190.0,
            close=205.0,
            quote_volume=9999.0,
        )

        # When / Then: completed-only input is stable; future/open input fails closed.
        clean = build_features(candles, EVIDENCE_TIME, CandleGranularity.FIVE_MINUTES)
        with self.assertRaisesMessage(FeatureInputError, "open_or_future_candle"):
            build_features(candles + (future,), EVIDENCE_TIME, CandleGranularity.FIVE_MINUTES)
        self.assertAlmostEqual(clean.sma20, 112.375, places=6)

    def test_malformed_candle_inputs_fail_closed(self):
        # Given: one valid multi-timeframe setup and malformed variants.
        valid = _candles(CandleGranularity.FIVE_MINUTES)
        duplicate = valid[:-1] + (valid[-2],)
        nonascending = valid[:10] + (valid[9],) + valid[11:]
        gap = valid[:30] + valid[31:]
        zero_volume = valid[:-1] + (
            Candle(valid[-1].opened_at, valid[-1].open, valid[-1].high, valid[-1].low, valid[-1].close, 0.0),
        )
        insufficient = valid[:30]

        cases = (
            (duplicate, "duplicate_candle"),
            (nonascending, "duplicate_candle"),
            (gap, "insufficient_lookback"),
            (zero_volume, "zero_or_negative_value"),
            (insufficient, "insufficient_lookback"),
        )
        for candles, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesMessage(FeatureInputError, message):
                    build_features(candles, EVIDENCE_TIME, CandleGranularity.FIVE_MINUTES)

        open_candle = valid + (
            Candle(EVIDENCE_TIME, 1.0, 1.1, 0.9, 1.0, 100.0),
        )
        no_trade = analyze_trend_continuation(
            profile=STANDARD_PROFILE,
            context_15m=_candles(CandleGranularity.FIFTEEN_MINUTES),
            confirmation_5m=open_candle,
            evidence_time=EVIDENCE_TIME,
        )
        self.assertEqual(no_trade.decision, ResearchDecision.NO_TRADE)
        self.assertEqual(no_trade.rejection_code, "5m_open_or_future_candle")
