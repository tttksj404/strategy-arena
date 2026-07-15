# Computes strategy metrics, Monte Carlo risk, and the 19-gate verdict table.

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

MC_PATHS: Final = 10_000
MC_SEED: Final = 20_260_715


@dataclass(frozen=True, slots=True)
class MetricInput:
    equity: pd.Series
    trade_returns: tuple[float, ...]
    turnover: float = 0.0
    exposure: float = 0.0


@dataclass(frozen=True, slots=True)
class Metrics:
    total_ret: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    recovery_factor: float
    mdd: float
    profit_factor: float
    win_rate: float
    turnover: float
    exposure: float
    n_trades: int


@dataclass(frozen=True, slots=True)
class GateInput:
    metrics: Metrics
    oos_metrics: Metrics
    oos_trade_returns: tuple[float, ...]
    neighbor_is_sharpes: tuple[float, ...]
    stress_oos_return: float
    yearly_returns: dict[int, float]
    regime_returns: dict[str, float]
    equity_btc_correlation: float
    data_valid: bool
    cost_model_valid: bool
    capacity_valid: bool
    capacity_value: str
    factor_exposure_valid: bool
    initial_capital: float = 300.0


@dataclass(frozen=True, slots=True)
class GateRow:
    gate: int
    name: str
    status: str
    value: str


def calculate_metrics(inputs: MetricInput) -> Metrics:
    equity = inputs.equity.dropna().astype(float)
    if len(equity) < 2:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inputs.turnover, inputs.exposure, len(inputs.trade_returns))
    daily = equity.pct_change().dropna()
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    elapsed_days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0, 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (365.0 / elapsed_days) - 1.0)
    volatility = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / volatility * sqrt(365.0)) if volatility > 0.0 else 0.0
    downside = float(daily[daily < 0.0].std(ddof=1))
    sortino = float(daily.mean() / downside * sqrt(365.0)) if downside > 0.0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    mdd = abs(float(drawdown.min()))
    calmar = cagr / mdd if mdd > 0.0 else 0.0
    recovery_factor = total_ret / mdd if mdd > 0.0 else 0.0
    trades = np.asarray(inputs.trade_returns, dtype=float)
    gains = float(trades[trades > 0.0].sum())
    losses = abs(float(trades[trades < 0.0].sum()))
    profit_factor = gains / losses if losses > 0.0 else (float("inf") if gains > 0.0 else 0.0)
    win_rate = float((trades > 0.0).mean()) if trades.size else 0.0
    return Metrics(total_ret, cagr, sharpe, sortino, calmar, recovery_factor, mdd, profit_factor, win_rate, inputs.turnover, inputs.exposure, len(trades))


def yearly_returns(equity: pd.Series) -> dict[int, float]:
    clean = equity.dropna().sort_index()
    results: dict[int, float] = {}
    anchor = 300.0
    for year, values in clean.groupby(clean.index.year):
        results[int(year)] = float(values.iloc[-1] / anchor - 1.0)
        anchor = float(values.iloc[-1])
    return results


def monte_carlo(
    trade_returns: tuple[float, ...],
    initial_capital: float,
) -> tuple[float, float] | None:
    if len(trade_returns) < 20:
        return None
    rng = np.random.default_rng(MC_SEED)
    trades = np.clip(np.asarray(trade_returns, dtype=float), -0.999, None)
    final_capital = np.empty(MC_PATHS, dtype=float)
    for start in range(0, MC_PATHS, 500):
        stop = min(start + 500, MC_PATHS)
        samples = rng.choice(trades, size=(stop - start, len(trades)), replace=True)
        final_capital[start:stop] = initial_capital * np.prod(1.0 + samples, axis=1)
    return float(np.quantile(final_capital, 0.05)), float((final_capital < initial_capital * 0.5).mean())


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def kelly_fraction(trade_returns: tuple[float, ...]) -> float:
    trades = np.asarray(trade_returns, dtype=float)
    variance = float(trades.var(ddof=1)) if trades.size > 1 else 0.0
    return float(trades.mean() / variance) if variance > 0.0 else 0.0


def evaluate_gates(inputs: GateInput) -> tuple[GateRow, ...]:
    neighbors = np.asarray(inputs.neighbor_is_sharpes, dtype=float)
    neighbor_mean = abs(float(neighbors.mean())) if neighbors.size else 0.0
    stable_neighbors = neighbors.size > 1 and bool(np.all(neighbors > 0.0)) and neighbor_mean > 0.0 and float(neighbors.std(ddof=1)) / neighbor_mean < 0.5
    sharpe_base = max(abs(inputs.metrics.sharpe), 1e-12)
    sharpe_divergence = abs(inputs.metrics.sharpe - inputs.oos_metrics.sharpe) / sharpe_base
    positive_year_ratio = float(np.mean(np.asarray(list(inputs.yearly_returns.values())) > 0.0)) if inputs.yearly_returns else 0.0
    simulation = monte_carlo(inputs.oos_trade_returns, inputs.initial_capital)
    mc_status = "UNDETERMINED" if simulation is None else _status(simulation[0] > inputs.initial_capital)
    mc_value = "<20 trades" if simulation is None else f"p05={simulation[0]:.2f}"
    kelly = kelly_fraction(inputs.oos_trade_returns)
    ruin_status = "UNDETERMINED" if simulation is None else _status(simulation[1] < 0.05)
    ruin_value = "<20 trades" if simulation is None else f"p={simulation[1]:.4f}"
    crash_return = inputs.regime_returns.get("2022_bear")
    crash_status = "UNDETERMINED" if crash_return is None else _status(crash_return > -0.25)
    crash_value = "no 2022 data" if crash_return is None else f"return={crash_return:.4f}"
    regime_status = "UNDETERMINED" if not inputs.regime_returns else _status(all(value > 0.0 for value in inputs.regime_returns.values()))
    correlation_known = bool(np.isfinite(inputs.equity_btc_correlation))
    correlation_status = _status(abs(inputs.equity_btc_correlation) < 0.8) if correlation_known else "UNDETERMINED"
    correlation_value = f"corr={inputs.equity_btc_correlation:.4f}" if correlation_known else "insufficient overlap"
    return (
        GateRow(1, "data_validation", _status(inputs.data_valid), str(inputs.data_valid)),
        GateRow(2, "overfit_sensitivity", _status(stable_neighbors and sharpe_divergence < 2.0), f"dispersion={sharpe_divergence:.3f}"),
        GateRow(3, "walk_forward", _status(positive_year_ratio >= 0.6), f"positive_years={positive_year_ratio:.1%}"),
        GateRow(4, "oos_after_cost", _status(inputs.oos_metrics.total_ret > 0.0), f"return={inputs.oos_metrics.total_ret:.4f}"),
        GateRow(5, "trade_bootstrap", mc_status, mc_value),
        GateRow(6, "crash_stress", crash_status, crash_value),
        GateRow(7, "trading_costs", _status(inputs.cost_model_valid and inputs.stress_oos_return > 0.0), f"measured={inputs.cost_model_valid}; double_slippage={inputs.stress_oos_return:.4f}"),
        GateRow(8, "capacity_and_sizing", _status(inputs.capacity_valid), inputs.capacity_value),
        GateRow(9, "kelly", _status(kelly > 0.0), f"f={kelly:.4f}; quarter={0.25 * kelly:.4f}"),
        GateRow(10, "mdd_limit", _status(inputs.oos_metrics.mdd <= 0.25), f"mdd={inputs.oos_metrics.mdd:.4f}"),
        GateRow(11, "sharpe", _status(inputs.oos_metrics.sharpe >= 1.0), f"sharpe={inputs.oos_metrics.sharpe:.4f}"),
        GateRow(12, "sortino", _status(inputs.oos_metrics.sortino >= inputs.oos_metrics.sharpe), f"sortino={inputs.oos_metrics.sortino:.4f}"),
        GateRow(13, "calmar", _status(inputs.oos_metrics.calmar >= 2.0), f"calmar={inputs.oos_metrics.calmar:.4f}"),
        GateRow(14, "profit_factor", _status(inputs.oos_metrics.profit_factor >= 1.5), f"pf={inputs.oos_metrics.profit_factor:.4f}"),
        GateRow(15, "recovery_factor", _status(inputs.oos_metrics.recovery_factor >= 2.0), f"recovery={inputs.oos_metrics.recovery_factor:.4f}"),
        GateRow(16, "bankruptcy", ruin_status, ruin_value),
        GateRow(17, "regime", regime_status, str(inputs.regime_returns)),
        GateRow(18, "btc_correlation", correlation_status, correlation_value),
        GateRow(19, "factor_exposure", _status(inputs.factor_exposure_valid), str(inputs.factor_exposure_valid)),
    )
