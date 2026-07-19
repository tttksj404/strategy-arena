"""Gate contract tests."""

import pandas as pd

from src.engine import BacktestResult
from src.gates import gate_result


def result(net: float, trades: int = 20) -> BacktestResult:
    """Build a compact synthetic result for gate checks."""
    index = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    return BacktestResult(pd.Series([300.0, 300.0 * (1 + net)], index=index), net, net, 0.0, 1.0, 1.0, trades, 0.0, 0.0, False, 0.0, 1.0, tuple([net / trades] * trades))


def test_negative_test_return_is_not_a_pass() -> None:
    """A risk-stable but money-losing test result cannot be a practical pass."""
    gates = gate_result(result(-0.02), result(0.03))
    assert not gates["test_net_positive"]
    assert not gates["all_pass"]


def test_bootstrap_resamples_with_replacement() -> None:
    """Distinct trade returns must yield a dispersed terminal distribution (permutation gives one point)."""
    import numpy as np
    from src.gates import bootstrap_floor
    from src.engine import BacktestResult
    import pandas as pd
    r = BacktestResult(pd.Series(dtype=float), 0.1, 0.1, 0.05, 1.0, 0.6, 5, 1.0, 0.0, False, 1.0, 1.0, (0.10, -0.05, 0.02, -0.01, 0.03))
    floor1, _ = bootstrap_floor(r)
    point = 300 * np.prod([1.1, 0.95, 1.02, 0.99, 1.03])
    assert floor1 < point - 1, f"p05 {floor1} not below point {point}: degenerate shuffle"
