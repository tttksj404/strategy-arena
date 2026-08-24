"""Bootstrap edge_table.json from backtest results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_binance.backtest.batch_backtest import BacktestTrade
from quant_binance.learning import OnlineEdgeLearner


def bootstrap_edge_table(
    *,
    trades: list[BacktestTrade],
    output_path: Path,
    min_observations: int = 5,
) -> dict[str, Any]:
    """Feed backtest trades into edge learner and export edge_table.json.

    Returns diagnostics dict.
    """
    learner = OnlineEdgeLearner(min_observations=min_observations)

    for trade in trades:
        learner.ingest_closed_trade(
            symbol=trade.symbol,
            mode=trade.mode,
            side=trade.side,
            entry_predictability_score=trade.predictability_score,
            realized_return_bps=trade.net_return_bps,
        )

    result = learner.export(output_path)
    return {
        "observation_count": result.observation_count,
        "symbols": result.symbols,
        "output_path": str(result.updated_path),
        "diagnostics": learner.lookup.diagnostics(),
    }
