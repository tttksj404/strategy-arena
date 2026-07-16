"""Metrics and anti-overfitting gates."""

from __future__ import annotations

import numpy as np

from .engine import BacktestResult


def bootstrap_floor(result: BacktestResult, initial: float = 300.0, rounds: int = 1000, seed: int = 17) -> tuple[float, float]:
    """Resample trade returns with replacement; return 5th-percentile terminal equity and ruin probability."""
    if not result.trade_returns:
        return initial, 0.0
    rng = np.random.default_rng(seed)
    returns = np.asarray(result.trade_returns, dtype=float)
    paths = np.ones(rounds) * initial
    ruined = np.zeros(rounds, dtype=bool)
    resampled = returns[rng.integers(0, len(returns), size=(rounds, len(returns)))]
    for column in resampled.T:
        paths *= np.maximum(0.0, 1 + column)
        ruined |= paths <= initial / 2
    return float(np.percentile(paths, 5)), float(ruined.mean())


def gate_result(test: BacktestResult, train: BacktestResult) -> dict[str, bool | float]:
    """Apply the specified test and train-to-test gates."""
    floor, ruin = bootstrap_floor(test)
    return {
        "test_net_positive": test.net_return > 0,
        "mdd_le_25": test.mdd <= 0.25,
        "bootstrap_floor_gt_300": floor > 300,
        "bankruptcy_probability_lt_5": ruin < 0.05,
        "trades_ge_15": test.trades >= 15,
        "sign_preserved": (train.net_return == 0) or (train.net_return * test.net_return > 0),
        "bootstrap_p05_equity": floor,
        "bankruptcy_probability": ruin,
        "all_pass": test.net_return > 0 and test.mdd <= 0.25 and floor > 300 and ruin < 0.05 and test.trades >= 15 and (train.net_return == 0 or train.net_return * test.net_return > 0),
    }
