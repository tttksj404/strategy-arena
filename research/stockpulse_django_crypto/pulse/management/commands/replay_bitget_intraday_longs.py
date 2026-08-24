from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TypedDict

from django.core.management.base import BaseCommand, CommandError

from pulse.services.bitget_history import HistoricalRangeQuery, fetch_historical_candle_range
from pulse.services.bitget_intraday_screener import eligible_local_us_instruments
from pulse.services.bitget_market import BitgetError, BitgetPublicClient, list_rwa_contracts, select_stock_contracts
from pulse.services.intraday_long_forecast import (
    MAX_MARGIN_LOSS_AT_STOP_PCT,
    MIN_NET_MARGIN_GAIN_AT_TARGET_PCT,
    REQUESTED_LEVERAGE,
)
from pulse.services.intraday_replay import (
    ReplayEntryFilter,
    ReplayPerformance,
    ReplayRule,
    evaluate_replay_candidates,
    select_replay_candidate,
)
from pulse.services.intraday_replay_engine import ReplayMarket, replay_market
from pulse.services.market_timeframes import CandleGranularity


REPLAY_RULES: tuple[ReplayRule, ...] = (
    ReplayRule("breakout-volume-1.0", 1.0),
    ReplayRule("breakout-volume-1.25", 1.25),
    ReplayRule("breakout-volume-1.5", 1.5),
    ReplayRule("breakout-volume-2.0", 2.0),
    ReplayRule("breakout-technical-score-v1-volume-1.0", 1.0, ReplayEntryFilter.TECHNICAL_SCORE),
    ReplayRule("breakout-bollinger-squeeze-v1-volume-1.0", 1.0, ReplayEntryFilter.BOLLINGER_SQUEEZE),
)


class PerformancePayload(TypedDict):
    settled_signal_count: int
    target_first_count: int
    stop_first_count: int
    unresolved_count: int
    ambiguous_count: int
    target_first_rate_pct: float
    wilson_lower_bound_pct: float
    post_cost_expected_margin_return_pct: float
    post_cost_margin_return_lower_bound_pct: float
    worst_margin_loss_at_stop_pct: float


def _performance_payload(performance: ReplayPerformance) -> PerformancePayload:
    return {
        "settled_signal_count": performance.settled_signal_count,
        "target_first_count": performance.target_first_count,
        "stop_first_count": performance.stop_first_count,
        "unresolved_count": performance.unresolved_count,
        "ambiguous_count": performance.ambiguous_count,
        "target_first_rate_pct": round(performance.target_first_rate_pct, 2),
        "wilson_lower_bound_pct": round(performance.wilson_lower_bound_pct, 2),
        "post_cost_expected_margin_return_pct": round(performance.post_cost_expected_margin_return_pct, 2),
        "post_cost_margin_return_lower_bound_pct": round(performance.post_cost_margin_return_lower_bound_pct, 2),
        "worst_margin_loss_at_stop_pct": round(performance.worst_margin_loss_at_stop_pct, 2),
    }


class Command(BaseCommand):
    help = "Replay fixed Bitget intraday volume filters with chronological validation and one untouched holdout verdict."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default="", help="Comma-separated underlying symbols; defaults to an alphabetic research sample.")
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--max-contracts", type=int, default=4)

    def handle(self, *args, **options):
        days = options["days"]
        max_contracts = options["max_contracts"]
        raw_symbols = options["symbols"]
        if not 1 <= days <= 30:
            raise CommandError("days must be between 1 and 30")
        if max_contracts < 1:
            raise CommandError("max-contracts must be positive")
        requested_symbols = frozenset(symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip())
        client = BitgetPublicClient()
        instruments = {instrument.code: instrument for instrument in eligible_local_us_instruments()}
        try:
            contracts = select_stock_contracts(list_rwa_contracts(client.get_text), frozenset(instruments))
            if requested_symbols:
                contracts = tuple(contract for contract in contracts if contract.underlying_symbol in requested_symbols)
            else:
                contracts = contracts[:max_contracts]
            if not contracts:
                raise CommandError("no normal Bitget RWA contracts matched the requested local US stock/ETF symbols")
            ends_at = datetime.now(UTC).replace(second=0, microsecond=0)
            starts_at = ends_at - timedelta(days=days)
            trades_by_rule = {rule.name: [] for rule in REPLAY_RULES}
            for contract in contracts:
                instrument = instruments[contract.underlying_symbol]
                context_15m = fetch_historical_candle_range(
                    HistoricalRangeQuery(contract, CandleGranularity.FIFTEEN_MINUTES, starts_at, ends_at),
                    client.get_text,
                )
                confirmation_5m = fetch_historical_candle_range(
                    HistoricalRangeQuery(contract, CandleGranularity.FIVE_MINUTES, starts_at, ends_at),
                    client.get_text,
                )
                settlement_1m = fetch_historical_candle_range(
                    HistoricalRangeQuery(contract, CandleGranularity.ONE_MINUTE, starts_at, ends_at),
                    client.get_text,
                )
                market = ReplayMarket(
                    contract_symbol=contract.symbol,
                    profile=instrument.profile,
                    taker_fee_rate=contract.taker_fee_rate,
                    context_15m=context_15m,
                    confirmation_5m=confirmation_5m,
                    trigger_1m=settlement_1m,
                    settlement_1m=settlement_1m,
                )
                for rule in REPLAY_RULES:
                    trades_by_rule[rule.name].extend(replay_market(market, rule))
        except BitgetError as error:
            raise CommandError(str(error)) from error
        frozen_trades_by_rule = {name: tuple(trades) for name, trades in trades_by_rule.items()}
        validation_diagnostics = evaluate_replay_candidates(frozen_trades_by_rule)
        result = select_replay_candidate(frozen_trades_by_rule)
        payload = {
            "mode": "historical_research_only",
            "execution": "disabled",
            "as_of": ends_at.isoformat(),
            "range": {"starts_at": starts_at.isoformat(), "ends_at": ends_at.isoformat(), "days": days},
            "contracts": [contract.symbol for contract in contracts],
            "risk_policy": {
                "fixed_leverage": REQUESTED_LEVERAGE,
                "max_margin_loss_at_stop_pct": MAX_MARGIN_LOSS_AT_STOP_PCT,
                "min_net_margin_gain_at_target_pct": MIN_NET_MARGIN_GAIN_AT_TARGET_PCT,
            },
            "candidate_configurations": {
                rule.name: {
                    "minimum_volume_ratio": rule.minimum_volume_ratio,
                    "entry_filter": rule.entry_filter.value,
                }
                for rule in REPLAY_RULES
            },
            "candidate_trade_counts": {name: len(trades) for name, trades in trades_by_rule.items()},
            "candidate_validation": {
                name: {
                    "passes_all_gates": diagnostic.validation_passes,
                    **_performance_payload(diagnostic.validation),
                }
                for name, diagnostic in validation_diagnostics.items()
            },
            "selection_uses_validation_only": True,
            "selected_on_validation_only": result.selected_rule_name,
            "validation": _performance_payload(result.validation),
            "held_out": _performance_payload(result.held_out),
            "historical_verdict": "RESEARCH_CANDIDATE" if result.is_reliable else "NO_RELIABLE_WINNER",
            "live_promotion": "disabled_until_separate_quoted_forward_evidence_clears",
        }
        self.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
