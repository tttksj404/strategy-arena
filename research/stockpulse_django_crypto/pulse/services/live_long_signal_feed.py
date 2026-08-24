from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import assert_never

from .bitget_intraday_screener import IntradayScreenReport, JsonObject, ScreenCandidate, screen_bitget_intraday_longs
from .bitget_market import BitgetError, BitgetPublicClient
from .bitget_ticker import BitgetTicker, fetch_live_ticker
from .intraday_long_forecast import InstrumentProfile
from .live_long_instruction import LiveLongInstruction, build_live_long_instruction
from .market_timeframes import CandleGranularity


class IneligibleLiveSignalError(ValueError):
    """Raised when an unproven candidate reaches the public manual-entry feed."""


@dataclass(frozen=True, slots=True)
class LiveLongSignal:
    """One screened continuation plan paired with the only quote allowed to trigger a manual entry."""

    candidate: ScreenCandidate
    quote: BitgetTicker
    instruction: LiveLongInstruction

    def alert_payload(self) -> JsonObject:
        """Return the concise manual instruction needed by the watcher and HTTP feed."""

        plan = self.candidate.raw.plan
        instruction = self.instruction
        forward = self.candidate.forward
        return {
            "symbol": self.candidate.raw.contract.symbol,
            "underlying_symbol": self.candidate.raw.contract.underlying_symbol,
            "profile": self.candidate.raw.instrument.profile.value,
            "action": instruction.action,
            "state": instruction.state.value,
            "quote": {
                "observed_at": self.quote.observed_at.isoformat(),
                "last_price": round(self.quote.last_price, 6),
                "bid_price": round(self.quote.bid_price, 6),
                "ask_price": round(self.quote.ask_price, 6),
            },
            "entry": {
                "price": round(instruction.entry_price, 6),
                "floor_price": round(instruction.entry_floor_price, 6),
                "ceiling_price": round(instruction.entry_ceiling_price, 6),
                "instruction": "ENTER_LONG 상태에서만 현재 ask_price 이하로 수동 진입; ceiling 초과 추격 금지.",
            },
            "exit": {
                "stop_loss_price": round(instruction.stop_price, 6),
                "take_profit_1_price": round(instruction.partial_take_profit_price, 6),
                "take_profit_2_price": round(instruction.final_take_profit_price, 6),
                "time_exit_at": instruction.expires_at.isoformat(),
            },
            "required_closed_timeframes": [
                granularity.value for granularity in _timeframes_for(self.candidate.raw.instrument.profile)
            ],
            "estimated_5x": {
                "margin_loss_at_stop_pct_before_fees": round(
                    instruction.feasibility.estimated_margin_loss_at_stop_pct_before_fees,
                    2,
                ),
                "estimated_all_in_margin_loss_at_stop_pct": round(
                    instruction.feasibility.estimated_all_in_margin_loss_at_stop_pct,
                    2,
                ),
                "round_trip_margin_cost_pct": round(instruction.feasibility.estimated_round_trip_margin_cost_pct, 2),
                "net_margin_gain_at_final_target_pct": round(
                    instruction.feasibility.estimated_net_margin_gain_at_target_pct,
                    2,
                ),
            },
            "research_status": "forward_evidence_ready" if self.candidate.five_x_eligible else "forward_evidence_pending",
            "forward_evidence": {
                "settled_signal_count": forward.total.settled_signal_count,
                "target_first_rate_pct": round(forward.total.target_first_rate_pct, 2),
                "wilson_lower_bound_pct": round(forward.total.wilson_lower_bound_pct, 2),
                "post_cost_expected_margin_return_pct": round(forward.total.post_cost_expected_margin_return_pct, 2),
                "post_cost_margin_return_lower_bound_pct": round(
                    forward.total.post_cost_margin_return_lower_bound_pct,
                    2,
                ),
                "validation_signal_count": forward.validation.settled_signal_count,
                "held_out_signal_count": forward.held_out.settled_signal_count,
            },
            "signal_started_at": plan.starts_at.isoformat(),
            "reasons": list(instruction.reasons),
            "risk_note": "공개 호가 기준의 수동 진입 계획이며, 실제 체결·펀딩·유지증거금·청산가 및 수익을 보장하지 않습니다.",
        }


@dataclass(frozen=True, slots=True)
class LiveLongWatchCandidate:
    """A quote-ready setup that is ranked for observation but has not earned a 5x entry promotion."""

    candidate: ScreenCandidate
    quote: BitgetTicker
    instruction: LiveLongInstruction

    def alert_payload(self) -> JsonObject:
        payload = LiveLongSignal(self.candidate, self.quote, self.instruction).alert_payload()
        payload["action"] = "WATCH_LONG_SETUP"
        payload["signal_tier"] = "forward_evidence_pending"
        payload["entry"] = {
            **payload["entry"],
            "instruction": "전진 검증 전 관찰 설정입니다. 현재 호가·손절·목표가를 기록하지만 5배 진입 신호로 승격하지 않습니다.",
        }
        payload["risk_note"] = (
            "현재 공개 호가에서 비용·손실 게이트만 통과한 관찰 설정입니다. "
            "전진 검증 전에는 수익·청산 회피·진입 적합성을 보장하지 않습니다."
        )
        return payload


def _timeframes_for(profile: InstrumentProfile) -> tuple[CandleGranularity, ...]:
    """Expose the completed timeframes already enforced by the underlying plan profile."""

    match profile:
        case InstrumentProfile.STANDARD:
            return (CandleGranularity.FIFTEEN_MINUTES, CandleGranularity.FIVE_MINUTES)
        case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
            return (
                CandleGranularity.FIFTEEN_MINUTES,
                CandleGranularity.FIVE_MINUTES,
                CandleGranularity.ONE_MINUTE,
            )
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class LiveLongSignalFeed:
    """A read-only live view over the existing closed-candle research screen."""

    base_report: IntradayScreenReport
    signals: tuple[LiveLongSignal, ...]
    ticker_rejection_counts: tuple[tuple[str, int], ...]
    watch_candidates: tuple[LiveLongWatchCandidate, ...] = ()

    def __post_init__(self) -> None:
        if any(not signal.candidate.five_x_eligible for signal in self.signals):
            raise IneligibleLiveSignalError("forward evidence must clear before a live manual-entry signal can be published")
        if any(
            candidate.candidate.five_x_eligible or not candidate.instruction.is_enter_now
            for candidate in self.watch_candidates
        ):
            raise IneligibleLiveSignalError("watch candidates must be quote-ready setups awaiting forward evidence")

    @property
    def enter_now_signals(self) -> tuple[LiveLongSignal, ...]:
        return tuple(signal for signal in self.signals if signal.instruction.is_enter_now)

    def payload(self) -> JsonObject:
        payload = self.base_report.payload()
        watch_payloads = [candidate.alert_payload() for candidate in self.watch_candidates]
        payload["mode"] = "read_only_live_signal"
        payload["poll_interval_seconds"] = 60
        payload["candidates"] = [signal.alert_payload() for signal in self.signals]
        payload["enter_now_signals"] = [signal.alert_payload() for signal in self.enter_now_signals]
        payload["live_signal_count"] = len(self.signals)
        payload["enter_now_count"] = len(self.enter_now_signals)
        payload["research_pending_count"] = sum(
            not candidate.five_x_eligible for candidate in self.base_report.candidates
        )
        payload["watch_candidates"] = watch_payloads
        payload["watch_candidate_count"] = len(watch_payloads)
        payload["top_watch_candidate"] = watch_payloads[0] if watch_payloads else None
        payload["ticker_rejections"] = [
            {"reason": reason, "count": count} for reason, count in self.ticker_rejection_counts
        ]
        payload["decision_policy"] = (
            "ENTER_LONG은 전진 검증까지 통과한 경우에만 표시됩니다. "
            "WATCH_LONG_SETUP은 현재 공개 ask와 실제 bid/ask 비용으로 5x 손실·목표 게이트를 통과했지만, 전진 검증 전인 관찰 설정입니다."
        )
        return payload


def _live_signal_for(candidate: ScreenCandidate, client: BitgetPublicClient, now: datetime) -> LiveLongSignal:
    quote = fetch_live_ticker(candidate.raw.contract, client.get_text, now)
    instruction = build_live_long_instruction(candidate.raw.plan, quote, candidate.raw.contract.taker_fee_rate, now)
    return LiveLongSignal(candidate=candidate, quote=quote, instruction=instruction)


def screen_bitget_live_long_alerts(now: datetime | None = None) -> LiveLongSignalFeed:
    """Screen completed candles, then emit manual entry instructions only from fresh public quotes."""

    reference_time = now or datetime.now(UTC)
    base_report = screen_bitget_intraday_longs(reference_time)
    client = BitgetPublicClient()
    signals: list[LiveLongSignal] = []
    watch_candidates: list[LiveLongWatchCandidate] = []
    rejection_counts: Counter[str] = Counter()
    # A closed-candle setup is research only until its chronological validation
    # and untouched holdout both clear. Never promote an unproven setup to a
    # user-facing 5x entry instruction merely because its current quote fits.
    for candidate in base_report.candidates:
        try:
            quote_reference_time = now or datetime.now(UTC)
            signal = _live_signal_for(candidate, client, quote_reference_time)
        except BitgetError:
            rejection_counts["ticker_unavailable"] += 1
            continue
        if candidate.five_x_eligible:
            signals.append(signal)
        elif signal.instruction.is_enter_now:
            watch_candidates.append(
                LiveLongWatchCandidate(
                    candidate=candidate,
                    quote=signal.quote,
                    instruction=signal.instruction,
                )
            )
        else:
            rejection_counts[f"research_{signal.instruction.state.value}"] += 1
    watch_candidates.sort(
        key=lambda candidate: (
            -candidate.candidate.raw.plan.score,
            -candidate.instruction.feasibility.estimated_net_margin_gain_at_target_pct,
            candidate.instruction.feasibility.estimated_all_in_margin_loss_at_stop_pct,
            candidate.quote.bid_ask_spread_pct,
            candidate.candidate.raw.contract.symbol,
        )
    )
    return LiveLongSignalFeed(
        base_report=base_report,
        signals=tuple(signals),
        ticker_rejection_counts=tuple(sorted(rejection_counts.items())),
        watch_candidates=tuple(watch_candidates),
    )
