# Wave-7 deep validation battery for the combined carry+momentum equity curve.
#
# Reuses research.validation.deep_stats (the same battery used to promote W2c and to
# kill W3d/W3c) for the Monte Carlo bootstrap and the 90-day block shuffle -- this is
# the one explicitly sanctioned cross-wave import for this package (see SPEC.md /
# task contract: "deep_stats.py 재사용 가능하면 임포트"). Everything else (Sharpe/CAGR/
# MDD/Calmar, the dormant-period OOS return, and the correlation with W2c) is a small
# self-contained calculation using the same formulas already established in
# research.wave1.gates.calculate_metrics, reimplemented locally so wave7 never has to
# modify or otherwise depend on code outside research/wave7/.

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import sys
from typing import Final

import pandas as pd  # noqa: PANDAS_OK

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.validation.deep_stats import (
    DeepValidationError,
    TimedValue,
    block_bootstrap,
    trade_bootstrap,
)


OOS_SPLIT: Final = pd.Timestamp("2025-09-30T23:59:59Z")  # 2025-10-01~ dormant-period OOS, matches wave2/gates.py
SEED_BASE: Final = 20_260_715  # matches research.wave1.gates.MC_SEED / research.validation.deep_validate's base seed
BLOCK_DAYS: Final = 90
ANNUAL_DAYS: Final = 365.0

MC_P05_MIN: Final = 300.0
RUIN_MAX: Final = 0.05
BLOCK_MDD_P95_MAX: Final = 0.25


@dataclass(frozen=True, slots=True)
class Wave7ValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def timedvalues_from_series(series: pd.Series) -> tuple[TimedValue, ...]:
    return tuple(TimedValue(pd.Timestamp(idx).to_pydatetime(), float(value)) for idx, value in series.items())


def standard_metrics(equity: pd.Series) -> dict[str, float]:
    """total_ret/cagr/sharpe/mdd/calmar using the same formula as
    research.wave1.gates.calculate_metrics (reimplemented locally; no import)."""
    equity = equity.dropna().astype(float)
    if len(equity) < 2:
        return {"total_ret": 0.0, "cagr": 0.0, "sharpe": 0.0, "mdd": 0.0, "calmar": 0.0}
    daily = equity.pct_change().dropna()
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    elapsed_days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0, 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (ANNUAL_DAYS / elapsed_days) - 1.0)
    volatility = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / volatility * sqrt(ANNUAL_DAYS)) if volatility > 0.0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    mdd = abs(float(drawdown.min()))
    calmar = cagr / mdd if mdd > 0.0 else 0.0
    return {"total_ret": total_ret, "cagr": cagr, "sharpe": sharpe, "mdd": mdd, "calmar": calmar}


def oos_dormant_return(equity: pd.Series, split: pd.Timestamp = OOS_SPLIT) -> float:
    """Compounded return from the last in-sample print through period end."""
    equity = equity.dropna().astype(float)
    is_equity = equity[equity.index <= split]
    oos_equity = equity[equity.index > split]
    if oos_equity.empty:
        return 0.0
    anchor = float(is_equity.iloc[-1]) if not is_equity.empty else float(equity.iloc[0])
    if anchor <= 0.0:
        raise Wave7ValidationError("OOS anchor capital must be positive")
    return float(oos_equity.iloc[-1] / anchor - 1.0)


def correlation_with_carry(combined_returns: pd.Series, carry_returns: pd.Series) -> float:
    aligned = pd.concat(
        [combined_returns.rename("combined"), carry_returns.rename("carry")], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(aligned["combined"].corr(aligned["carry"]))


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def evaluate_candidate(
    candidate_id: str,
    equity: pd.Series,
    combined_returns: pd.Series,
    carry_returns_aligned: pd.Series,
    carry_alone_metrics: dict[str, float],
    carry_alone_oos_return: float,
    seed_index: int,
) -> dict:
    """Run the full SPEC section-3/4 deep validation battery for one candidate.

    ① MC bootstrap 1e4, daily-return resampling (deep_stats.trade_bootstrap.unit) ->
       p05 final capital, ruin probability P(final < 150).
    ② 90-day block shuffle, 1e3 paths (deep_stats.block_bootstrap; REGIME_PATHS is
       already 1,000 and block_days defaults to 90) -> MDD p95.
    ③ Sharpe/CAGR/MDD/Calmar on the combined equity curve.
    ④ Dormant-period (2025-10~) OOS return vs. carry-alone's own OOS return.
    ⑤ Correlation between the combined daily returns and W2c's own daily returns.
    """
    trades_tv = timedvalues_from_series(combined_returns)
    daily_values = tuple(item.value for item in trades_tv)
    try:
        mc = trade_bootstrap(daily_values, SEED_BASE + seed_index * 101)
        blocks = block_bootstrap(trades_tv, SEED_BASE + seed_index * 103, block_days=BLOCK_DAYS)
    except DeepValidationError as error:
        raise Wave7ValidationError(f"{candidate_id}: {error}") from error

    metrics = standard_metrics(equity)
    oos_return = oos_dormant_return(equity)
    correlation = correlation_with_carry(combined_returns, carry_returns_aligned)

    gate_mc_p05 = mc.unit.p05 > MC_P05_MIN
    gate_ruin = mc.unit.ruin_probability < RUIN_MAX
    gate_block_mdd = blocks.mdd_p95 <= BLOCK_MDD_P95_MAX
    gate_oos = oos_return > carry_alone_oos_return
    gate_sharpe = metrics["sharpe"] > carry_alone_metrics["sharpe"]
    overall = gate_mc_p05 and gate_ruin and gate_block_mdd and gate_oos and gate_sharpe

    gates = [
        {
            "gate": 1,
            "name": "mc_bootstrap_p05",
            "status": _status(gate_mc_p05),
            "value": f"p05={mc.unit.p05:.2f} (>{MC_P05_MIN:.0f})",
        },
        {
            "gate": 2,
            "name": "bankruptcy_probability",
            "status": _status(gate_ruin),
            "value": f"ruin={mc.unit.ruin_probability:.4f} (<{RUIN_MAX:.2f})",
        },
        {
            "gate": 3,
            "name": "block_shuffle_mdd_p95",
            "status": _status(gate_block_mdd),
            "value": f"mdd_p95={blocks.mdd_p95:.4f} (<={BLOCK_MDD_P95_MAX:.2f})",
        },
        {
            "gate": 4,
            "name": "dormant_oos_return",
            "status": _status(gate_oos),
            "value": f"oos={oos_return:.4f} (> carry_alone={carry_alone_oos_return:.4f})",
        },
        {
            "gate": 5,
            "name": "sharpe_vs_carry_alone",
            "status": _status(gate_sharpe),
            "value": f"sharpe={metrics['sharpe']:.4f} (> carry_alone={carry_alone_metrics['sharpe']:.4f})",
        },
    ]
    return {
        "candidate_id": candidate_id,
        "metrics": metrics,
        "bootstrap_mc": {
            "p05": mc.unit.p05,
            "ruin_probability": mc.unit.ruin_probability,
            "mean": mc.unit.mean,
            "median": mc.unit.median,
            "paths": mc.paths,
            "trade_count": mc.trade_count,
        },
        "block_shuffle": {
            "block_days": blocks.block_days,
            "block_count": blocks.block_count,
            "paths": blocks.paths,
            "mdd_p05": blocks.mdd_p05,
            "mdd_median": blocks.mdd_median,
            "mdd_p95": blocks.mdd_p95,
            "final_p05": blocks.final_p05,
            "final_median": blocks.final_median,
            "final_p95": blocks.final_p95,
        },
        "dormant_oos_return": oos_return,
        "carry_alone_oos_return": carry_alone_oos_return,
        "carry_alone_sharpe": carry_alone_metrics["sharpe"],
        "correlation_with_carry": correlation,
        "gates": gates,
        "overall": {"status": _status(overall), "passed_gates": sum(g["status"] == "PASS" for g in gates), "total_gates": len(gates)},
    }


__all__ = [
    "BLOCK_DAYS",
    "BLOCK_MDD_P95_MAX",
    "MC_P05_MIN",
    "OOS_SPLIT",
    "RUIN_MAX",
    "SEED_BASE",
    "Wave7ValidationError",
    "correlation_with_carry",
    "evaluate_candidate",
    "oos_dormant_return",
    "standard_metrics",
    "timedvalues_from_series",
]
