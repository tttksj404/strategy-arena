# Wave-9 promotion gates H1-H5 (SPEC.md lines 12-19). Only these five gates are
# implemented: SPEC.md freezes the objective function and gate list alongside the six
# candidates ("H1*H2*H4는 타협 불가"), so no gate beyond H1-H5 is added here.
#
# H5's 90-day block shuffle reuses research.validation.deep_stats.block_bootstrap
# unmodified (the one cross-wave import this module takes, mirroring wave7's own
# "deep_stats.py 재사용 가능하면 임포트" precedent -- see research/wave7/deepval_w7.py).
# Only its `.mdd_p95` field (and other MDD fields) is used: max-drawdown is a scale
# -invariant ratio, so deep_stats' internal INITIAL_CAPITAL=300 constant does not bias
# it. Its `.final_*` dollar fields are NOT used because those *are* scale-dependent
# and wave9 needs a $100 basis; H1/H2's dollar thresholds ($30/$50) are instead
# computed by this module's own mc_bootstrap_trades (same resampling-with-replacement
# methodology as deep_stats.trade_bootstrap, parameterized for $100 starting capital
# and wave9's own bankruptcy line).

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.validation.deep_stats import DeepValidationError, TimedValue, block_bootstrap, deflated_sharpe
from research.wave9_100usd.engine_w9 import MAX_LEVERAGE, MIN_ORDER_USDT, OOS_SPLIT, TOTAL_CAPITAL, series_from_payload


MC_PATHS: Final = 10_000
MC_SEED_BASE: Final = 20_260_721
BLOCK_SEED_BASE: Final = 20_260_722
BLOCK_DAYS: Final = 90
BANKRUPTCY_THRESHOLD: Final = 30.0  # H1: P(final < $30)
BANKRUPTCY_PROBABILITY_MAX: Final = 0.20
P05_FLOOR: Final = 50.0  # H2: MC p05 > $50
BLOCK_MDD_P95_MAX: Final = 0.50  # H5
TOTAL_TRIALS_DISCLOSED: Final = 58  # 52 prior candidates (wave1-8) + this wave's 6


@dataclass(frozen=True, slots=True)
class GateOutcome:
    gate_id: str
    name: str
    status: str  # PASS / FAIL / UNDETERMINED
    detail: str


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def trade_fraction_returns(payload: dict) -> np.ndarray:
    """Per-trade fractional return (pnl_dollars / equity_before) for every completed
    trade recorded by the run stage."""
    trades = payload.get("trades") or []
    values = [float(trade["pnl_dollars"]) / float(trade["equity_before"]) for trade in trades if float(trade["equity_before"]) > 0.0]
    return np.asarray(values, dtype=float)


def mc_bootstrap_trades(trade_returns: np.ndarray, seed: int, paths: int = MC_PATHS, capital: float = TOTAL_CAPITAL) -> dict:
    """SPEC's MC method: bootstrap *trade* returns (not daily returns) with
    replacement, `paths` times, compounding one draw per historical trade per path,
    starting from `capital`. Returns median/p05/P(final<$30)."""
    trade_count = int(trade_returns.size)
    if trade_count == 0:
        return {"median": capital, "p05": capital, "p_bankrupt": 1.0, "paths": paths, "trade_count": 0}
    clipped = np.clip(trade_returns, -0.999999, None)
    rng = np.random.default_rng(seed)
    finals = np.empty(paths, dtype=float)
    for start in range(0, paths, 500):
        stop = min(start + 500, paths)
        samples = rng.choice(clipped, size=(stop - start, trade_count), replace=True)
        finals[start:stop] = capital * np.prod(1.0 + samples, axis=1)
    return {
        "median": float(np.median(finals)),
        "p05": float(np.quantile(finals, 0.05)),
        "p_bankrupt": float(np.mean(finals < BANKRUPTCY_THRESHOLD)),
        "paths": paths,
        "trade_count": trade_count,
    }


def h1_bankruptcy(mc: dict) -> GateOutcome:
    if mc["trade_count"] == 0:
        return GateOutcome("H1", "bankruptcy_probability", "UNDETERMINED", "no trades were executed")
    ok = mc["p_bankrupt"] < BANKRUPTCY_PROBABILITY_MAX
    return GateOutcome(
        "H1", "bankruptcy_probability", _status(ok),
        f"P(final<${BANKRUPTCY_THRESHOLD:.0f})={mc['p_bankrupt']:.4f} (must be <{BANKRUPTCY_PROBABILITY_MAX:.2f}); n_trades={mc['trade_count']}",
    )


def h2_p05_floor(mc: dict) -> GateOutcome:
    if mc["trade_count"] == 0:
        return GateOutcome("H2", "mc_p05_floor", "UNDETERMINED", "no trades were executed")
    ok = mc["p05"] > P05_FLOOR
    return GateOutcome("H2", "mc_p05_floor", _status(ok), f"p05=${mc['p05']:.2f} (must be >${P05_FLOOR:.0f})")


def h3_oos_return(payload: dict) -> GateOutcome:
    trades = payload.get("trades") or []
    oos_trades = [trade for trade in trades if pd.Timestamp(trade["entry_day"]) > OOS_SPLIT]
    if not oos_trades:
        return GateOutcome("H3", "oos_return_positive", "UNDETERMINED", "no OOS trades (entry after 2025-10-01)")
    compounded = 1.0
    for trade in oos_trades:
        equity_before = float(trade["equity_before"])
        if equity_before <= 0.0:
            continue
        compounded *= 1.0 + float(trade["pnl_dollars"]) / equity_before
    oos_return = compounded - 1.0
    return GateOutcome("H3", "oos_return_positive", _status(oos_return > 0.0), f"oos_compounded_return={oos_return:.4f} (n_oos_trades={len(oos_trades)})")


def h4_feasibility(payload: dict, leverage: float) -> GateOutcome:
    meta = payload.get("metadata") or {}
    min_notional = float(meta.get("min_notional_at_start", 0.0))
    gross_fraction = float(meta.get("gross_fraction_at_start", float("inf")))
    single_leg = bool(meta.get("single_leg", False))
    leverage_cap_ok = leverage <= MAX_LEVERAGE
    gross_cap = 0.90 * leverage  # active fraction (0.90) * leverage, per SPEC H4
    ok = single_leg and min_notional >= MIN_ORDER_USDT and gross_fraction <= gross_cap + 1e-9 and leverage_cap_ok
    detail = (
        f"min_notional=${min_notional:.2f} (>=${MIN_ORDER_USDT:.0f}); "
        f"gross={gross_fraction:.3f}x equity (<= active*leverage={gross_cap:.3f}x); "
        f"single_leg={single_leg}; leverage={leverage:g}x (<={MAX_LEVERAGE:g}x); "
        f"infeasible_cycles_during_run={meta.get('infeasible_cycles', 0)}"
    )
    return GateOutcome("H4", "capital_feasibility", _status(ok), detail)


def h5_block_shuffle(payload: dict, seed: int) -> tuple[GateOutcome, dict]:
    trades = payload.get("trades") or []
    timed = tuple(
        TimedValue(pd.Timestamp(trade["exit_day"]).to_pydatetime(), float(trade["pnl_dollars"]) / float(trade["equity_before"]))
        for trade in trades
        if float(trade["equity_before"]) > 0.0
    )
    if not timed:
        return GateOutcome("H5", "block_shuffle_mdd_p95", "UNDETERMINED", "no trades to block-shuffle"), {}
    try:
        blocks = block_bootstrap(timed, seed, block_days=BLOCK_DAYS)
    except DeepValidationError as error:
        return GateOutcome("H5", "block_shuffle_mdd_p95", "UNDETERMINED", str(error)), {}
    ok = blocks.mdd_p95 <= BLOCK_MDD_P95_MAX
    detail = f"mdd_p95={blocks.mdd_p95:.4f} (must be <={BLOCK_MDD_P95_MAX:.2f}); blocks={blocks.block_count}; paths={blocks.paths}"
    payload_out = {
        "block_days": blocks.block_days,
        "block_count": blocks.block_count,
        "paths": blocks.paths,
        "mdd_p05": blocks.mdd_p05,
        "mdd_median": blocks.mdd_median,
        "mdd_p95": blocks.mdd_p95,
        "low_confidence": blocks.block_count < 3,
    }
    return GateOutcome("H5", "block_shuffle_mdd_p95", _status(ok), detail), payload_out


def sharpe_from_trades(trade_returns: np.ndarray, hold_days: int) -> float:
    """Reference-only metric (SPEC: "샤프*Calmar는 참고 지표로만 기록"). Annualized
    using the candidate's own fixed rebalance cadence (365 / hold_days trades/year)
    since trade returns are not daily-spaced."""
    if trade_returns.size < 2:
        return 0.0
    std = float(trade_returns.std(ddof=1))
    if std <= 0.0:
        return 0.0
    periods_per_year = 365.0 / max(hold_days, 1)
    return float(trade_returns.mean() / std * np.sqrt(periods_per_year))


def deflated_sharpe_reference(payload: dict) -> dict | None:
    """Reference-only DSR using the trade-boundary equity curve, corrected for
    TOTAL_TRIALS_DISCLOSED (58 = 52 prior candidates + this wave's 6), per SPEC's
    multiple-testing disclosure requirement."""
    equity = series_from_payload(payload.get("equity") or [])
    if len(equity) < 4:
        return None
    timed = tuple(TimedValue(pd.Timestamp(idx).to_pydatetime(), float(value)) for idx, value in equity.items())
    try:
        result = deflated_sharpe(timed, trials=TOTAL_TRIALS_DISCLOSED)
    except DeepValidationError:
        return None
    return {"score": result.score, "probability": result.probability, "trials": result.trials, "observed_sharpe": result.observed_sharpe}


def evaluate_payload(payload: dict, leverage: float, hold_days: int, seed_index: int) -> dict:
    trade_returns = trade_fraction_returns(payload)
    mc = mc_bootstrap_trades(trade_returns, MC_SEED_BASE + seed_index * 101)
    h1 = h1_bankruptcy(mc)
    h2 = h2_p05_floor(mc)
    h3 = h3_oos_return(payload)
    h4 = h4_feasibility(payload, leverage)
    h5, block_payload = h5_block_shuffle(payload, BLOCK_SEED_BASE + seed_index * 103)
    gates = (h1, h2, h3, h4, h5)

    hard_gate_ids = {"H1", "H2", "H4"}  # SPEC: "H1*H2*H4는 타협 불가"
    hard_pass = all(gate.status == "PASS" for gate in gates if gate.gate_id in hard_gate_ids)
    all_pass = all(gate.status == "PASS" for gate in gates)

    return {
        "candidate_id": payload["candidate_id"],
        "mc_bootstrap": mc,
        "block_shuffle": block_payload,
        "gates": [asdict(gate) for gate in gates],
        "reference_metrics": {
            "sharpe_trade_level": sharpe_from_trades(trade_returns, hold_days),
            "dsr": deflated_sharpe_reference(payload),
            "total_trials_disclosed": TOTAL_TRIALS_DISCLOSED,
        },
        "overall": {
            "status": _status(all_pass),
            "hard_gates_status": _status(hard_pass),
            "passed": sum(gate.status == "PASS" for gate in gates),
            "total": len(gates),
        },
    }


__all__ = [
    "BANKRUPTCY_PROBABILITY_MAX",
    "BANKRUPTCY_THRESHOLD",
    "BLOCK_DAYS",
    "BLOCK_MDD_P95_MAX",
    "MC_PATHS",
    "P05_FLOOR",
    "TOTAL_TRIALS_DISCLOSED",
    "GateOutcome",
    "deflated_sharpe_reference",
    "evaluate_payload",
    "h1_bankruptcy",
    "h2_p05_floor",
    "h3_oos_return",
    "h4_feasibility",
    "h5_block_shuffle",
    "mc_bootstrap_trades",
    "sharpe_from_trades",
    "trade_fraction_returns",
]
