from __future__ import annotations

from dataclasses import dataclass

from quant_binance.models import DecisionIntent


@dataclass(frozen=True)
class VirtualPerformance:
    starting_equity_usd: float
    ending_equity_usd: float
    total_pnl_usd: float
    total_return_pct: float
    executed_decision_count: int
    win_rate: float
    average_pnl_usd: float
    average_return_bps: float


def virtual_performance_from_decisions(
    *,
    decisions: list[DecisionIntent] | tuple[DecisionIntent, ...],
    starting_equity_usd: float,
) -> VirtualPerformance:
    executed = [decision for decision in decisions if decision.final_mode in {"spot", "futures"}]
    pnls = [
        decision.order_intent_notional_usd * (decision.net_expected_edge_bps / 10000.0)
        for decision in executed
    ]
    total_pnl_usd = round(sum(pnls), 6)
    ending_equity_usd = round(starting_equity_usd + total_pnl_usd, 6)
    executed_count = len(executed)
    wins = sum(1 for pnl in pnls if pnl > 0)
    win_rate = (wins / executed_count) if executed_count else 0.0
    average_pnl_usd = (total_pnl_usd / executed_count) if executed_count else 0.0
    average_return_bps = (
        sum(decision.net_expected_edge_bps for decision in executed) / executed_count
        if executed_count
        else 0.0
    )
    total_return_pct = (
        (total_pnl_usd / starting_equity_usd) * 100.0 if starting_equity_usd > 0 else 0.0
    )
    return VirtualPerformance(
        starting_equity_usd=starting_equity_usd,
        ending_equity_usd=ending_equity_usd,
        total_pnl_usd=total_pnl_usd,
        total_return_pct=round(total_return_pct, 6),
        executed_decision_count=executed_count,
        win_rate=round(win_rate, 6),
        average_pnl_usd=round(average_pnl_usd, 6),
        average_return_bps=round(average_return_bps, 6),
    )
