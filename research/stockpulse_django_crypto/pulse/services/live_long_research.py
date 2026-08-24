from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import assert_never

from .bitget_intraday_screener import IntradayScreenReport
from .bitget_market import BitgetError, TextFetcher
from .bitget_ticker import fetch_live_ticker
from .intraday_long_forecast import STRATEGY_VERSION
from .intraday_signal_ledger import SignalObservation
from .live_long_instruction import LiveEntryState, build_live_long_instruction


@dataclass(frozen=True, slots=True)
class ResearchObservationCollection:
    """Executable public quotes retained for forward evaluation without publishing an order instruction."""

    observations: tuple[SignalObservation, ...]
    rejection_counts: tuple[tuple[str, int], ...]


def collect_research_observations(
    report: IntradayScreenReport,
    fetch_text: TextFetcher,
    now: datetime | None = None,
) -> ResearchObservationCollection:
    """Record only fresh quotes that clear the current actual-cost 5x envelope."""

    reference_time = now or datetime.now(UTC)
    observations: list[SignalObservation] = []
    rejection_counts: Counter[str] = Counter()
    for candidate in report.candidates:
        try:
            quote = fetch_live_ticker(candidate.raw.contract, fetch_text, reference_time)
        except BitgetError:
            rejection_counts["ticker_unavailable"] += 1
            continue
        instruction = build_live_long_instruction(
            candidate.raw.plan,
            quote,
            candidate.raw.contract.taker_fee_rate,
            reference_time,
        )
        match instruction.state:
            case LiveEntryState.ENTER_NOW:
                observations.append(
                    SignalObservation(
                        contract_symbol=candidate.raw.contract.symbol,
                        underlying_symbol=candidate.raw.contract.underlying_symbol,
                        profile=candidate.raw.instrument.profile,
                        strategy_version=STRATEGY_VERSION,
                        signal_opened_at=candidate.raw.plan.starts_at - candidate.raw.plan.trigger_granularity.interval,
                        plan=replace(candidate.raw.plan, entry_price=instruction.entry_price),
                        taker_fee_rate=candidate.raw.contract.taker_fee_rate,
                        feasibility=instruction.feasibility,
                        entry_quote_observed_at=quote.observed_at,
                        entry_quote_spread_pct=quote.bid_ask_spread_pct,
                    )
                )
            case (
                LiveEntryState.WAIT_FOR_ENTRY_ZONE
                | LiveEntryState.SKIP_CHASED
                | LiveEntryState.SKIP_TRIGGER_STALE
                | LiveEntryState.SKIP_EXPIRED
                | LiveEntryState.SKIP_INVALIDATED
                | LiveEntryState.SKIP_LIQUIDITY_GATE
                | LiveEntryState.SKIP_RISK_GATE
            ):
                rejection_counts[instruction.state.value] += 1
            case unreachable:
                assert_never(unreachable)
    return ResearchObservationCollection(
        observations=tuple(observations),
        rejection_counts=tuple(sorted(rejection_counts.items())),
    )
