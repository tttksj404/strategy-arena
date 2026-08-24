from __future__ import annotations

from dataclasses import dataclass

from quant_binance.execution.router import ExecutionRouter
from quant_binance.models import DecisionIntent, MarketSnapshot
from quant_binance.settings import Settings
from quant_binance.strategy.regime import evaluate_snapshot


@dataclass(frozen=True)
class ReplayResult:
    decisions: tuple[DecisionIntent, ...]
    order_count: int


def run_replay(
    *,
    snapshots: list[MarketSnapshot],
    settings: Settings,
    equity_usd: float,
    remaining_portfolio_capacity_usd: float,
    router: ExecutionRouter | None = None,
) -> ReplayResult:
    execution_router = router or ExecutionRouter()
    decisions: list[DecisionIntent] = []
    order_count = 0
    for snapshot in snapshots:
        decision = evaluate_snapshot(
            snapshot,
            settings=settings,
            equity_usd=equity_usd,
            remaining_portfolio_capacity_usd=remaining_portfolio_capacity_usd,
        )
        decisions.append(decision)
        if execution_router.route(decision) is not None:
            order_count += 1
    return ReplayResult(decisions=tuple(decisions), order_count=order_count)
