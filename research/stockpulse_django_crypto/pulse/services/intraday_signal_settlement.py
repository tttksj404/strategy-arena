from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from django.utils import timezone

from pulse.intraday_signal_models import IntradayLongSignal, SignalOutcome

from .bitget_market import BitgetContract, HistoricalCandleQuery, TextFetcher, fetch_historical_completed_candles
from .intraday_long_forecast import REQUESTED_LEVERAGE, round_trip_margin_cost_pct
from .long_forecast import Candle
from .market_timeframes import CandleGranularity


@dataclass(frozen=True, slots=True)
class SignalSettlement:
    """A post-expiry result determined from completed one-minute historical candles."""

    outcome: SignalOutcome
    terminal_price: float
    post_cost_margin_return_pct: float


def _settle(signal: IntradayLongSignal, candles: tuple[Candle, ...]) -> SignalSettlement:
    target_price = float(signal.target_price)
    stop_price = float(signal.stop_price)
    terminal_price = candles[-1].close
    outcome = SignalOutcome.UNRESOLVED
    for candle in candles:
        target_hit = candle.high >= target_price
        stop_hit = candle.low <= stop_price
        if target_hit and stop_hit:
            outcome = SignalOutcome.AMBIGUOUS
            terminal_price = stop_price
            break
        if target_hit:
            outcome = SignalOutcome.TARGET
            terminal_price = target_price
            break
        if stop_hit:
            outcome = SignalOutcome.STOP
            terminal_price = stop_price
            break
    raw_return_pct = (terminal_price / float(signal.entry_price) - 1) * 100
    round_trip_cost_pct = round_trip_margin_cost_pct(signal.taker_fee_rate, signal.entry_quote_spread_pct)
    return SignalSettlement(
        outcome=outcome,
        terminal_price=terminal_price,
        post_cost_margin_return_pct=raw_return_pct * REQUESTED_LEVERAGE - round_trip_cost_pct,
    )


def _contract_for_signal(signal: IntradayLongSignal) -> BitgetContract:
    return BitgetContract(
        symbol=signal.contract_symbol,
        underlying_symbol=signal.underlying_symbol,
        max_leverage=1,
        funding_interval_hours=1,
        taker_fee_rate=signal.taker_fee_rate,
    )


def _has_complete_settlement_window(signal: IntradayLongSignal, candles: tuple[Candle, ...]) -> bool:
    """Require every one-minute bar in the immutable signal window before assigning an outcome."""

    interval = CandleGranularity.ONE_MINUTE.interval
    expected_count = int((signal.expires_at - signal.starts_at) / interval)
    if expected_count <= 0 or len(candles) != expected_count:
        return False
    if candles[0].opened_at != signal.starts_at or candles[-1].opened_at != signal.expires_at - interval:
        return False
    return all(current.opened_at - previous.opened_at == interval for previous, current in zip(candles, candles[1:]))


def settle_pending_signals(fetch_text: TextFetcher, now: datetime | None = None) -> int:
    """Settle expired immutable observations from public history; incomplete or missing windows remain pending."""

    reference_time = now or timezone.now()
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=UTC)
    pending = IntradayLongSignal.objects.filter(status=SignalOutcome.PENDING, expires_at__lte=reference_time).order_by(
        "expires_at"
    )
    settled_count = 0
    for signal in pending:
        candles = fetch_historical_completed_candles(
            HistoricalCandleQuery(
                contract=_contract_for_signal(signal),
                granularity=CandleGranularity.ONE_MINUTE,
                starts_at=signal.starts_at,
                ends_at=signal.expires_at,
            ),
            fetch_text,
        )
        if not _has_complete_settlement_window(signal, candles):
            continue
        settlement = _settle(signal, candles)
        signal.status = settlement.outcome
        signal.terminal_price = str(settlement.terminal_price)
        signal.post_cost_margin_return_pct = settlement.post_cost_margin_return_pct
        signal.settled_at = reference_time
        signal.save(
            update_fields=(
                "status",
                "terminal_price",
                "post_cost_margin_return_pct",
                "settled_at",
                "updated_at",
            )
        )
        settled_count += 1
    return settled_count
