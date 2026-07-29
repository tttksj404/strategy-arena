# Wave-25 promotion gates P1-P5 (SPEC.md "게이트" section). Statistics helpers
# (skewness/top_decile_contribution/bootstrap/MC-ruin/CAGR) are reimplemented locally rather
# than imported from research.wave20_convex.gates20 -- matches that module's own documented
# convention ("reimplemented locally per this repo's own per-wave-gates-module convention")
# and keeps wave25 gate math self-contained and independently auditable. Only
# research.validation.deep_stats.deflated_sharpe (a repo-wide, wave-agnostic utility, already
# reused unmodified by every prior wave's own gates module) is imported directly.
#
# "승격 = P1·P2 필수 + (P3 or P4)" (SPEC.md line 34) -- P5 is disclosed but is NOT part of the
# promotion boolean itself (SPEC.md's own promotion formula names only P1/P2/P3/P4); P5 is
# reported alongside every candidate as an executability diagnostic, matching how gates20 kept
# its own reference-only DSR/OOS sections disclosed-but-non-gating.

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.validation.deep_stats import DeepValidationError, TimedValue, deflated_sharpe
from research.wave25_gamble.configs25 import (
    DSR_CUMULATIVE_TRIALS,
    GAMBLE_CAPITAL,
    IS_OOS_SPLIT,
    P1_BOOTSTRAP_PATHS,
    P1_MIN_TRADES,
    P1_TOP_DECILE_CONTRIBUTION_MIN,
    P1_TOP_DECILE_FRACTION,
    P2_MAX_SINGLE_TRADE_LOSS_USDT,
    P2_MC_PATHS,
    P2_RUIN_FLOOR_USDT,
    P2_RUIN_PROBABILITY_MAX,
    P4_ROLLING_WINDOW_DAYS,
    P4_TOP_QUARTILE_FRACTION,
    P5_MIN_LEG_USDT,
    TOTAL_CAPITAL,
)


@dataclass(frozen=True, slots=True)
class GateOutcome:
    gate_id: str
    name: str
    status: str  # PASS / FAIL / UNDETERMINED
    detail: str


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Shared metric helpers.
# ---------------------------------------------------------------------------


def full_period_annualized(equity: pd.Series) -> float | None:
    clean = equity.dropna()
    if len(clean) < 2:
        return None
    start_value = float(clean.iloc[0])
    end_value = float(clean.iloc[-1])
    if start_value <= 0.0:
        return None
    days = max((pd.Timestamp(clean.index[-1]) - pd.Timestamp(clean.index[0])).total_seconds() / 86_400.0, 1.0)
    growth = end_value / start_value
    if growth <= 0.0:
        return -1.0
    return float(growth ** (365.0 / days) - 1.0)


def calendar_year_return(equity: pd.Series, year: int) -> float | None:
    clean = equity.dropna()
    if clean.empty:
        return None
    in_year = clean[clean.index.year == year]
    if in_year.empty:
        return None
    before = clean[clean.index < in_year.index[0]]
    base = float(before.iloc[-1]) if not before.empty else float(in_year.iloc[0])
    end = float(in_year.iloc[-1])
    if base <= 0.0:
        return None
    return float(end / base - 1.0)


def daily_returns_array(equity: pd.Series) -> np.ndarray:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2:
        return np.asarray([], dtype=float)
    return values[1:] / values[:-1] - 1.0


def skewness(values: np.ndarray) -> float | None:
    """Population moment coefficient of skewness -- same formula as
    research.wave20_convex.gates20.skewness (mean(centered**3)/std**3, population std,
    ddof=0), reimplemented standalone per this module's own docstring."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return None
    mean = float(arr.mean())
    centered = arr - mean
    variance = float(np.mean(centered**2))
    std = float(np.sqrt(variance))
    if std <= 0.0:
        return 0.0
    return float(np.mean(centered**3) / std**3)


def top_decile_contribution(pnls: np.ndarray, fraction: float = P1_TOP_DECILE_FRACTION) -> float | None:
    """SPEC.md P1's second clause: top `fraction` of trades (by count) vs gross profit
    (winning trades only) -- identical definition to gates20.top_decile_contribution."""
    arr = np.asarray(pnls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    gross_profit = float(arr[arr > 0.0].sum())
    if gross_profit <= 0.0:
        return None
    n_top = max(1, int(np.ceil(fraction * arr.size)))
    top_sum = float(np.sort(arr)[::-1][:n_top].sum())
    return top_sum / gross_profit


def bootstrap_skew_diagnostic(pnls: np.ndarray, seed: int, paths: int = P1_BOOTSTRAP_PATHS) -> dict[str, Any] | None:
    """Resamples trade P&Ls WITH replacement `paths` times and recomputes skew every draw --
    SPEC.md P1's THIRD clause ("부트스트랩 왜도 하한 > 0") reads the p05 of this distribution
    directly as part of the hard gate (stricter than wave20's own G3, which only disclosed
    this bootstrap alongside the gate rather than gating on it)."""
    arr = np.asarray(pnls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < P1_MIN_TRADES:
        return None
    rng = np.random.default_rng(seed)
    skews = np.empty(paths, dtype=float)
    for start in range(0, paths, 500):
        stop = min(start + 500, paths)
        samples = rng.choice(arr, size=(stop - start, arr.size), replace=True)
        for row in range(stop - start):
            skews[start + row] = skewness(samples[row]) or 0.0
    return {
        "paths": paths,
        "p05": float(np.quantile(skews, 0.05)),
        "p50": float(np.quantile(skews, 0.50)),
        "p95": float(np.quantile(skews, 0.95)),
        "fraction_positive": float(np.mean(skews > 0.0)),
    }


def best_trade_sensitivity(pnls: np.ndarray, final_equity: float) -> dict[str, Any] | None:
    arr = np.asarray(pnls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    best = float(arr.max())
    return {
        "best_trade_pnl_usdt": best,
        "final_equity_usdt": float(final_equity),
        "final_equity_excluding_best_trade_usdt_approx": float(final_equity) - best,
        "approximation_note": "additive removal, not re-compounded -- diagnostic only",
    }


# ---------------------------------------------------------------------------
# P1 -- convexity: skew>0 AND top-decile trades carry >=50% of gross profit AND bootstrap
# skew lower bound (p05) > 0.
# ---------------------------------------------------------------------------


def gate_p1_convexity(trade_pnls: np.ndarray, final_equity: float, seed: int) -> tuple[GateOutcome, dict[str, Any]]:
    arr = np.asarray(trade_pnls, dtype=float)
    arr = arr[np.isfinite(arr)]
    bootstrap = bootstrap_skew_diagnostic(arr, seed)
    diagnostics: dict[str, Any] = {
        "n_trades": int(arr.size),
        "skew": skewness(arr),
        "top_decile_fraction": P1_TOP_DECILE_FRACTION,
        "top_decile_contribution_of_gross_profit": top_decile_contribution(arr),
        "bootstrap": bootstrap,
        "best_trade_sensitivity": best_trade_sensitivity(arr, final_equity),
    }
    if arr.size < P1_MIN_TRADES:
        return GateOutcome("P1", "convexity", "UNDETERMINED", f"only {arr.size} trades (<{P1_MIN_TRADES}) -- skew/decile/bootstrap unreliable at this sample size"), diagnostics
    skew = diagnostics["skew"]
    decile = diagnostics["top_decile_contribution_of_gross_profit"]
    if skew is None or decile is None or bootstrap is None:
        return GateOutcome("P1", "convexity", "UNDETERMINED", "skew/top-decile/bootstrap not computable (no gross profit or degenerate distribution)"), diagnostics
    bootstrap_ok = bootstrap["p05"] > 0.0
    ok = (skew > 0.0) and (decile >= P1_TOP_DECILE_CONTRIBUTION_MIN) and bootstrap_ok
    detail = (
        f"skew={skew:.4f} (>0), top{int(P1_TOP_DECILE_FRACTION*100)}%_contribution={decile:.4f} (>={P1_TOP_DECILE_CONTRIBUTION_MIN:.2f}), "
        f"bootstrap_skew_p05={bootstrap['p05']:.4f} (>0)"
    )
    return GateOutcome("P1", "convexity", _status(ok), detail), diagnostics


# ---------------------------------------------------------------------------
# P2 -- bankruptcy defense (non-negotiable): system MC ruin probability < 10% AND no single
# realized trade lost more than $25 in absolute dollar terms.
# ---------------------------------------------------------------------------


def _simulate_mc_final(daily_returns: np.ndarray, seed: int, paths: int, capital: float) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    finals = np.empty(paths, dtype=float)
    chunk = 500
    for start in range(0, paths, chunk):
        stop = min(start + chunk, paths)
        samples = rng.choice(daily_returns, size=(stop - start, daily_returns.size), replace=True)
        growth = np.prod(1.0 + np.clip(samples, -0.999999, None), axis=1)
        finals[start:stop] = capital * growth
    return {
        "p05": float(np.quantile(finals, 0.05)),
        "median": float(np.median(finals)),
        "mean": float(np.mean(finals)),
        "ruin_probability": float(np.mean(finals < P2_RUIN_FLOOR_USDT)),
        "paths": paths,
    }


def max_single_trade_loss_usdt(trades_payload: list[dict[str, Any]]) -> float:
    """Largest REALIZED dollar loss on any one trade (0.0 if no losing trades) -- distinct
    from a FRACTIONAL loss cap: once the sleeve has compounded above $25, a trade sized at
    100% of a since-grown sleeve can lose more than $25 in dollar terms even though it can
    never lose more than that trade's own entry_equity_usdt (the engine's own $0 floor only
    bounds the fraction, not the absolute dollar size) -- this is exactly the gap SPEC.md's
    P2 "단일 최대손실 <= $25" closes."""
    losses = [-float(t["pnl_usdt"]) for t in trades_payload if float(t["pnl_usdt"]) < 0.0]
    return max(losses) if losses else 0.0


def gate_p2_bankruptcy(combined_equity: pd.Series, trades_payload: list[dict[str, Any]], seed: int) -> tuple[GateOutcome, dict[str, Any]]:
    returns = daily_returns_array(combined_equity)
    max_loss = max_single_trade_loss_usdt(trades_payload)
    single_loss_ok = max_loss <= P2_MAX_SINGLE_TRADE_LOSS_USDT + 1e-6
    if returns.size < 30:
        detail = f"only {returns.size} daily returns -- MC ruin UNDETERMINED; max_single_trade_loss=${max_loss:.4f} ({'OK' if single_loss_ok else 'VIOLATED'})"
        return GateOutcome("P2", "bankruptcy_defense", "UNDETERMINED", detail), {"max_single_trade_loss_usdt": max_loss}
    mc = _simulate_mc_final(returns, seed, P2_MC_PATHS, TOTAL_CAPITAL)
    ruin_ok = mc["ruin_probability"] < P2_RUIN_PROBABILITY_MAX
    ok = ruin_ok and single_loss_ok
    detail = (
        f"P(final<${P2_RUIN_FLOOR_USDT:.0f})={mc['ruin_probability']:.4f} (<{P2_RUIN_PROBABILITY_MAX:.2f}), "
        f"max_single_trade_loss=${max_loss:.4f} (<=${P2_MAX_SINGLE_TRADE_LOSS_USDT:.0f}), "
        f"p05=${mc['p05']:.2f}, median=${mc['median']:.2f}, paths={mc['paths']}, basis=${TOTAL_CAPITAL:.0f} combined system"
    )
    mc["max_single_trade_loss_usdt"] = max_loss
    return GateOutcome("P2", "bankruptcy_defense", _status(ok), detail), mc


# ---------------------------------------------------------------------------
# P3 -- sleeve final equity beats B0(V1) sleeve final equity.
# ---------------------------------------------------------------------------


def gate_p3_beats_baseline(candidate_final_usdt: float, baseline_final_usdt: float) -> GateOutcome:
    ok = candidate_final_usdt > baseline_final_usdt
    detail = f"candidate_sleeve_final=${candidate_final_usdt:.4f} vs B0_sleeve_final=${baseline_final_usdt:.4f} (must strictly beat it)"
    return GateOutcome("P3", "beats_baseline", _status(ok), detail)


# ---------------------------------------------------------------------------
# P4 -- 30-day rolling-window top-quartile average beats B0's own.
# ---------------------------------------------------------------------------


def rolling_window_return(equity: pd.Series, window_days: int = P4_ROLLING_WINDOW_DAYS) -> pd.Series:
    """Trailing `window_days` return ending at each observation: equity[t]/equity[t-window]-1.
    Assumes `equity` is already a DENSE daily series (one row per calendar day, no gaps) --
    true by construction for every candidate's own `equity_daily`
    (resample("1D").last().ffill() in engine25.run_multi_symbol_convex /
    research.wave20_convex.engine20's own V1-V5), so a plain integer-position shift is exactly
    a `window_days`-calendar-day lag, not an approximation."""
    clean = equity.dropna()
    shifted = clean.shift(window_days)
    return clean / shifted - 1.0


def top_quartile_mean(values: pd.Series, fraction: float = P4_TOP_QUARTILE_FRACTION) -> float | None:
    clean = values.dropna()
    if clean.empty:
        return None
    cutoff = clean.quantile(1.0 - fraction)
    top = clean[clean >= cutoff]
    if top.empty:
        return None
    return float(top.mean())


def gate_p4_short_term_edge(candidate_equity: pd.Series, baseline_equity: pd.Series) -> tuple[GateOutcome, dict[str, Any]]:
    candidate_returns = rolling_window_return(candidate_equity)
    baseline_returns = rolling_window_return(baseline_equity)
    candidate_top = top_quartile_mean(candidate_returns)
    baseline_top = top_quartile_mean(baseline_returns)
    payload = {"candidate_top_quartile_mean_30d_return": candidate_top, "baseline_top_quartile_mean_30d_return": baseline_top}
    if candidate_top is None or baseline_top is None:
        return GateOutcome("P4", "short_term_rolling_edge", "UNDETERMINED", "insufficient rolling-window observations for candidate or baseline"), payload
    ok = candidate_top > baseline_top
    detail = f"candidate_top25%_avg_30d_return={candidate_top:.4f} vs B0_top25%_avg_30d_return={baseline_top:.4f} (must strictly beat it)"
    return GateOutcome("P4", "short_term_rolling_edge", _status(ok), detail), payload


# ---------------------------------------------------------------------------
# P5 -- executability (disclosed, non-gating -- see module docstring): leg size floor, no
# overlapping positions (structural "gross <= sleeve" verification), and sign-preservation
# under 3x measured-slippage stress.
# ---------------------------------------------------------------------------


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def no_overlapping_positions(trades_payload: list[dict[str, Any]]) -> bool:
    """Verifies the single-position-at-a-time invariant empirically from the realized trade
    list (sorted by entry time, each trade's entry must be >= the previous trade's exit) --
    the concrete, checkable form of SPEC.md P5's "gross <= 슬리브" (1x leverage, never more
    than one leg open, so gross exposure can never exceed the sleeve's own current equity)."""
    rows = [(t.get("entry_time"), t.get("exit_time")) for t in trades_payload if t.get("entry_time") is not None]
    rows.sort(key=lambda pair: _parse_ts(pair[0]))
    prev_exit: pd.Timestamp | None = None
    for entry_raw, exit_raw in rows:
        entry_ts = _parse_ts(entry_raw)
        if prev_exit is not None and entry_ts is not None and entry_ts < prev_exit:
            return False
        exit_ts = _parse_ts(exit_raw)
        prev_exit = exit_ts if exit_ts is not None else prev_exit
    return True


def _profit_sign(final_usdt: float, starting_equity: float) -> int:
    diff = final_usdt - starting_equity
    if diff > 1e-9:
        return 1
    if diff < -1e-9:
        return -1
    return 0


def gate_p5_executable(
    trades_payload: list[dict[str, Any]],
    base_final_usdt: float,
    stressed_final_usdt: float,
    starting_equity: float = GAMBLE_CAPITAL,
) -> tuple[GateOutcome, dict[str, Any]]:
    leg_sizes = [float(t["entry_equity_usdt"]) for t in trades_payload if t.get("entry_equity_usdt") is not None]
    min_leg = min(leg_sizes) if leg_sizes else None
    leg_ok = (min_leg is None) or (min_leg >= P5_MIN_LEG_USDT)
    overlap_ok = no_overlapping_positions(trades_payload)
    sign_ok = _profit_sign(base_final_usdt, starting_equity) == _profit_sign(stressed_final_usdt, starting_equity)
    ok = leg_ok and overlap_ok and sign_ok
    detail = (
        f"min_leg_usdt={'N/A' if min_leg is None else f'${min_leg:.4f}'} (>=${P5_MIN_LEG_USDT:.0f}), "
        f"no_overlapping_positions={overlap_ok}, base_final=${base_final_usdt:.4f} stressed_final(3x)=${stressed_final_usdt:.4f} sign_preserved={sign_ok}"
    )
    payload = {"min_leg_usdt": min_leg, "no_overlapping_positions": overlap_ok, "base_final_usdt": base_final_usdt, "stressed_final_usdt": stressed_final_usdt, "sign_preserved": sign_ok}
    return GateOutcome("P5", "executable", _status(ok), detail), payload


# ---------------------------------------------------------------------------
# Reference-only diagnostics (never gate promotion) -- DSR + OOS, same disclosed-only
# convention as research.wave20_convex.gates20.
# ---------------------------------------------------------------------------


def oos_performance(equity: pd.Series, split: pd.Timestamp = IS_OOS_SPLIT) -> dict[str, Any] | None:
    clean = equity.dropna()
    is_part = clean[clean.index <= split]
    oos_part = clean[clean.index > split]
    if is_part.empty or oos_part.empty:
        return None
    anchor = float(is_part.iloc[-1])
    end = float(oos_part.iloc[-1])
    if anchor <= 0.0:
        return None
    total_return = end / anchor - 1.0
    days = max((pd.Timestamp(oos_part.index[-1]) - pd.Timestamp(oos_part.index[0])).total_seconds() / 86_400.0, 1.0)
    growth = 1.0 + total_return
    annualized = float(growth ** (365.0 / days) - 1.0) if growth > 0.0 else -1.0
    return {"split": str(split), "is_anchor_usdt": anchor, "oos_end_usdt": end, "oos_total_return": total_return, "oos_annualized_return": annualized, "oos_days": days}


def deflated_sharpe_reference(equity: pd.Series, trials: int = DSR_CUMULATIVE_TRIALS) -> dict[str, Any] | None:
    clean = equity.dropna()
    if len(clean) < 4:
        return None
    timed = tuple(TimedValue(pd.Timestamp(idx).to_pydatetime(), float(value)) for idx, value in clean.items())
    try:
        result = deflated_sharpe(timed, trials=trials)
    except DeepValidationError:
        return None
    return {"score": result.score, "probability": result.probability, "trials": result.trials, "observed_sharpe": result.observed_sharpe}


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def evaluate_candidate(
    candidate_id: str,
    gamble_equity: pd.Series,
    trades_payload: list[dict[str, Any]],
    combined_equity: pd.Series,
    baseline_gamble_equity: pd.Series,
    baseline_final_usdt: float,
    stressed_final_usdt: float,
    seed_offset: int,
) -> dict[str, Any]:
    trade_pnls = np.asarray([float(t["pnl_usdt"]) for t in trades_payload], dtype=float)
    final_gamble_equity = float(gamble_equity.dropna().iloc[-1]) if len(gamble_equity.dropna()) else GAMBLE_CAPITAL

    p1, p1_diagnostics = gate_p1_convexity(trade_pnls, final_gamble_equity, seed=20_260_729 + seed_offset * 101)
    p2, p2_payload = gate_p2_bankruptcy(combined_equity, trades_payload, seed=20_260_730 + seed_offset * 103)
    p3 = gate_p3_beats_baseline(final_gamble_equity, baseline_final_usdt)
    p4, p4_payload = gate_p4_short_term_edge(gamble_equity, baseline_gamble_equity)
    p5, p5_payload = gate_p5_executable(trades_payload, final_gamble_equity, stressed_final_usdt)

    gates = (p1, p2, p3, p4, p5)
    core_gates = (p1, p2, p3, p4)  # P5 disclosed only, excluded from promoted/any_undetermined-for-promotion logic below
    promoted = p1.status == "PASS" and p2.status == "PASS" and (p3.status == "PASS" or p4.status == "PASS")
    any_undetermined = any(gate.status == "UNDETERMINED" for gate in core_gates)

    return {
        "candidate_id": candidate_id,
        "gates": [asdict(gate) for gate in gates],
        "mc_ruin": p2_payload,
        "convexity_diagnostics": p1_diagnostics,
        "p4_detail": p4_payload,
        "p5_detail": p5_payload,
        "reference_metrics": {
            "dsr_gamble_sleeve": deflated_sharpe_reference(gamble_equity),
            "dsr_combined_system": deflated_sharpe_reference(combined_equity),
            "total_trials_disclosed": DSR_CUMULATIVE_TRIALS,
            "oos_gamble_sleeve": oos_performance(gamble_equity),
            "oos_combined_system": oos_performance(combined_equity),
        },
        "overall": {
            "status": "UNDETERMINED" if (not promoted and any_undetermined) else _status(promoted),
            "passed": sum(gate.status == "PASS" for gate in gates),
            "undetermined": sum(gate.status == "UNDETERMINED" for gate in gates),
            "total": len(gates),
            "promoted": promoted,
        },
    }


__all__ = [
    "GateOutcome",
    "best_trade_sensitivity",
    "bootstrap_skew_diagnostic",
    "calendar_year_return",
    "daily_returns_array",
    "deflated_sharpe_reference",
    "evaluate_candidate",
    "full_period_annualized",
    "gate_p1_convexity",
    "gate_p2_bankruptcy",
    "gate_p3_beats_baseline",
    "gate_p4_short_term_edge",
    "gate_p5_executable",
    "max_single_trade_loss_usdt",
    "no_overlapping_positions",
    "oos_performance",
    "rolling_window_return",
    "skewness",
    "top_decile_contribution",
    "top_quartile_mean",
]
