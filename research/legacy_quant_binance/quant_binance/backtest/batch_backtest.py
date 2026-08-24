"""Batch backtest runner — evaluates strategy on historical time slices."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from quant_binance.backtest.historical_fixture_builder import HistoricalTimeSlice
from quant_binance.service import PaperTradingService
from quant_binance.execution.router import ExecutionRouter
from quant_binance.settings import Settings


@dataclass(frozen=True)
class BacktestTrade:
    symbol: str
    mode: str
    side: str
    entry_time: datetime
    entry_price: float
    predictability_score: float
    notional_usd: float
    gross_return_bps: float
    net_return_bps: float
    cost_bps: float


@dataclass
class BatchBacktestResult:
    trades: list[BacktestTrade]
    total_decisions: int
    cash_decisions: int
    trade_count: int
    gross_pnl_bps: float
    net_pnl_bps: float
    win_rate: float
    win_count: int
    loss_count: int


def run_batch_backtest(
    *,
    slices: list[HistoricalTimeSlice],
    settings: Settings,
    equity_usd: float = 71.0,
    capacity_usd: float = 178.0,
    holding_period: str = "4h",
    cost_bps: float = 16.0,
) -> BatchBacktestResult:
    """Run strategy evaluation on each slice, simulate PnL with forward returns."""
    service = PaperTradingService(settings, router=ExecutionRouter())
    trades: list[BacktestTrade] = []
    cash_count = 0

    for idx, sl in enumerate(slices):
        try:
            decision = service.run_cycle(
                state=sl.state,
                primitive_inputs=sl.primitive_inputs,
                history=sl.history,
                decision_time=sl.decision_time,
                equity_usd=equity_usd,
                remaining_portfolio_capacity_usd=capacity_usd,
            )
        except Exception:
            continue

        if decision.final_mode not in ("spot", "futures"):
            cash_count += 1
            continue

        fwd = sl.forward_return_4h_bps if holding_period == "4h" else sl.forward_return_1h_bps
        gross = fwd if decision.side == "long" else -fwd
        net = gross - cost_bps

        trades.append(BacktestTrade(
            symbol=sl.symbol,
            mode=decision.final_mode,
            side=decision.side,
            entry_time=sl.decision_time,
            entry_price=sl.state.last_trade_price,
            predictability_score=decision.predictability_score,
            notional_usd=decision.order_intent_notional_usd,
            gross_return_bps=round(gross, 4),
            net_return_bps=round(net, 4),
            cost_bps=cost_bps,
        ))

        if idx % 500 == 0 and idx > 0:
            print(f"  [backtest] {idx}/{len(slices)} slices, {len(trades)} trades...", flush=True)

    wins = [t for t in trades if t.net_return_bps > 0]
    losses = [t for t in trades if t.net_return_bps <= 0]

    return BatchBacktestResult(
        trades=trades,
        total_decisions=len(slices),
        cash_decisions=cash_count,
        trade_count=len(trades),
        gross_pnl_bps=round(sum(t.gross_return_bps for t in trades), 4),
        net_pnl_bps=round(sum(t.net_return_bps for t in trades), 4),
        win_rate=round(len(wins) / max(len(trades), 1), 4),
        win_count=len(wins),
        loss_count=len(losses),
    )
