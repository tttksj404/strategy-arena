# Wave-20 promotion gates G1-G5 (SPEC.md lines 21-27). Only these five are implemented --
# SPEC.md freezes the gate list alongside the five candidates, matching every prior wave's own
# "no gate beyond what SPEC.md names" convention (see e.g. research/wave9_100usd/gates_w9.py's
# own module docstring). "승격 = G1~G5 전부" (all five, no partial credit).
#
# This module also carries the SPEC.md-mandated honesty diagnostics backing G3 ("도박 후보는
# 우연히 대박 하나 나오면 좋아 보이므로 부트스트랩*왜도 필수"): skewness, top-decile trade
# contribution, a trade-resample bootstrap on skew (does convexity survive resampling, or does
# it evaporate the moment the single best trade is left out of a draw), and a first-order
# "CAGR excluding the single best trade" sensitivity check. These are DISCLOSED alongside G3,
# not folded into the PASS/FAIL boolean itself -- G3's own gate condition is exactly
# SPEC.md's two-part test (skew>0 AND top-decile>=50%), nothing softer or stricter.

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
from research.wave20_convex.configs20 import (
    DSR_CUMULATIVE_TRIALS,
    G1_MAX_LOSS_FRACTION,
    G2_MC_PATHS,
    G2_RUIN_FLOOR_USDT,
    G2_RUIN_PROBABILITY_MAX,
    G3_BOOTSTRAP_PATHS,
    G3_MIN_TRADES,
    G3_TOP_DECILE_CONTRIBUTION_MIN,
    G3_TOP_DECILE_FRACTION,
    GAMBLE_CAPITAL,
    IS_OOS_SPLIT,
    TOTAL_CAPITAL,
    WORST_YEARS,
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
# Shared metric helpers (reimplemented locally per this repo's own per-wave-gates-module
# convention -- see research/wave18_idle/gates18.py's full_period_annualized docstring for the
# precedent this follows).
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
    """Simple (not annualized) return over calendar `year`, using the last observation
    BEFORE the year (or the first observation IN the year, if the series starts mid-year) as
    the base, and the last observation in the year as the end. None if the series has no
    coverage in that year at all."""
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
    """Population moment coefficient of skewness -- identical formula to
    research.validation.deep_stats.deflated_sharpe's own internal skew computation
    (mean(centered**3)/std**3, population std, ddof=0), reimplemented standalone here so G3
    does not need a full DSR computation just to read the skew component back out."""
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


def top_decile_contribution(pnls: np.ndarray, fraction: float = G3_TOP_DECILE_FRACTION) -> float | None:
    """SPEC.md G3's second clause: "상위 10% 거래가 총수익의 50% 이상 기여". Contribution is
    measured against GROSS PROFIT (sum of winning trades only) -- the standard way to show
    payoff-shape concentration, and the only definition that stays meaningful when the
    candidate's NET total is negative (a negative net total makes "% of total return"
    undefined/nonsensical, but "the biggest winners still did most of the winning" remains a
    well-posed question). None if there is no gross profit to divide by (either no trades or
    no winning trades at all -- the question of convexity does not even arise)."""
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


def bootstrap_skew_diagnostic(pnls: np.ndarray, seed: int, paths: int = G3_BOOTSTRAP_PATHS) -> dict[str, Any] | None:
    """Resamples trade P&Ls WITH replacement `paths` times (same n each draw) and recomputes
    skew every time -- distinguishes "one lucky trade" (skew collapses/flips sign the moment
    that single draw is diluted or absent from a resample) from a genuinely repeated convex
    payoff shape (skew stays positive across the large majority of resamples)."""
    arr = np.asarray(pnls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < G3_MIN_TRADES:
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
    """First-order (additive, not re-compounded) approximation of the gambling sleeve's own
    final equity with its single best trade removed: final_equity - best_trade_pnl. Disclosed
    as an approximation (a true removal would re-derive every downstream trade's own capital
    base, since position sizing compounds) -- good enough to answer the diagnostic question
    this exists for ("is the headline number mostly one trade?"), not offered as an exact
    counterfactual equity curve."""
    arr = np.asarray(pnls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    best = float(arr.max())
    return {
        "best_trade_pnl_usdt": best,
        "final_equity_usdt": float(final_equity),
        "final_equity_excluding_best_trade_usdt_approx": float(final_equity) - best,
        "approximation_note": "additive removal, not re-compounded -- see best_trade_sensitivity docstring",
    }


# ---------------------------------------------------------------------------
# G1 -- structural loss cap.
# ---------------------------------------------------------------------------


def gate_g1_structural_loss_cap(gamble_equity: pd.Series, trades_payload: list[dict[str, Any]]) -> GateOutcome:
    clean = gamble_equity.dropna()
    min_equity = float(clean.min()) if len(clean) else float("nan")
    max_loss_usdt = GAMBLE_CAPITAL - min_equity if len(clean) else float("nan")
    equity_never_negative = bool((clean >= -1e-9).all()) if len(clean) else False
    worst_trade_fraction = min((float(t["pnl_fraction"]) for t in trades_payload), default=0.0)
    trades_within_cap = worst_trade_fraction >= -G1_MAX_LOSS_FRACTION - 1e-9
    ok = equity_never_negative and max_loss_usdt <= GAMBLE_CAPITAL + 1e-6 and trades_within_cap
    detail = (
        f"min_equity=${min_equity:.4f} (floor=$0), max_loss=${max_loss_usdt:.4f} (<=${GAMBLE_CAPITAL:.0f} allocation), "
        f"worst_trade_fraction={worst_trade_fraction:.4f} (>= -{G1_MAX_LOSS_FRACTION:.0f})"
    )
    return GateOutcome("G1", "structural_loss_cap", _status(ok), detail)


# ---------------------------------------------------------------------------
# G2 -- system-wide Monte Carlo ruin probability.
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
        "ruin_probability": float(np.mean(finals < G2_RUIN_FLOOR_USDT)),
        "paths": paths,
    }


def gate_g2_mc_ruin(combined_equity: pd.Series, seed: int) -> tuple[GateOutcome, dict[str, Any]]:
    returns = daily_returns_array(combined_equity)
    if returns.size < 30:
        return GateOutcome("G2", "mc_ruin_probability", "UNDETERMINED", f"only {returns.size} daily returns -- too few for a stable MC"), {}
    mc = _simulate_mc_final(returns, seed, G2_MC_PATHS, TOTAL_CAPITAL)
    ok = mc["ruin_probability"] < G2_RUIN_PROBABILITY_MAX
    detail = (
        f"P(final<${G2_RUIN_FLOOR_USDT:.0f})={mc['ruin_probability']:.4f} (must be <{G2_RUIN_PROBABILITY_MAX:.2f}); "
        f"p05=${mc['p05']:.2f}, median=${mc['median']:.2f}, paths={mc['paths']}, basis=${TOTAL_CAPITAL:.0f} combined system"
    )
    return GateOutcome("G2", "mc_ruin_probability", _status(ok), detail), mc


# ---------------------------------------------------------------------------
# G3 -- convexity: skew>0 AND top-decile trades carry >=50% of gross profit.
# ---------------------------------------------------------------------------


def gate_g3_convexity(trade_pnls: np.ndarray, final_equity: float, seed: int) -> tuple[GateOutcome, dict[str, Any]]:
    arr = np.asarray(trade_pnls, dtype=float)
    arr = arr[np.isfinite(arr)]
    diagnostics: dict[str, Any] = {
        "n_trades": int(arr.size),
        "skew": skewness(arr),
        "top_decile_fraction": G3_TOP_DECILE_FRACTION,
        "top_decile_contribution_of_gross_profit": top_decile_contribution(arr),
        "bootstrap": bootstrap_skew_diagnostic(arr, seed),
        "best_trade_sensitivity": best_trade_sensitivity(arr, final_equity),
    }
    if arr.size < G3_MIN_TRADES:
        return (
            GateOutcome("G3", "convexity_skew_and_decile", "UNDETERMINED", f"only {arr.size} trades (<{G3_MIN_TRADES}) -- skew/decile-contribution unreliable at this sample size"),
            diagnostics,
        )
    skew = diagnostics["skew"]
    decile = diagnostics["top_decile_contribution_of_gross_profit"]
    if skew is None or decile is None:
        return GateOutcome("G3", "convexity_skew_and_decile", "UNDETERMINED", "skew or top-decile-contribution not computable (no gross profit / degenerate distribution)"), diagnostics
    ok = (skew > 0.0) and (decile >= G3_TOP_DECILE_CONTRIBUTION_MIN)
    detail = f"skew={skew:.4f} (must be >0), top10%_contribution={decile:.4f} (must be >={G3_TOP_DECILE_CONTRIBUTION_MIN:.2f} of gross profit)"
    return GateOutcome("G3", "convexity_skew_and_decile", _status(ok), detail), diagnostics


# ---------------------------------------------------------------------------
# G4 -- full-system CAGR beats I5-solo.
# ---------------------------------------------------------------------------


def gate_g4_beats_stable_solo(combined_equity: pd.Series, stable_solo_cagr: float) -> tuple[GateOutcome, dict[str, Any]]:
    combined_cagr = full_period_annualized(combined_equity)
    if combined_cagr is None:
        return GateOutcome("G4", "system_cagr_beats_i5_solo", "UNDETERMINED", "combined equity series too short"), {}
    ok = combined_cagr > stable_solo_cagr
    detail = f"combined_cagr={combined_cagr:.4f} vs I5_solo_cagr={stable_solo_cagr:.4f} (must beat it)"
    return GateOutcome("G4", "system_cagr_beats_i5_solo", _status(ok), detail), {"combined_full_period_cagr": combined_cagr, "stable_solo_cagr": stable_solo_cagr}


# ---------------------------------------------------------------------------
# G5 -- no worse-year degradation vs I5-solo in 2022/2025.
# ---------------------------------------------------------------------------


def gate_g5_worst_year_defense(combined_equity: pd.Series, stable_solo_equity: pd.Series, years: tuple[int, ...] = WORST_YEARS) -> tuple[GateOutcome, dict[str, Any]]:
    per_year: dict[str, Any] = {}
    determinable = []
    for year in years:
        combined_return = calendar_year_return(combined_equity, year)
        solo_return = calendar_year_return(stable_solo_equity, year)
        row = {"combined_return": combined_return, "i5_solo_return": solo_return}
        if combined_return is not None and solo_return is not None:
            row["no_degradation"] = bool(combined_return >= solo_return - 1e-9)
            determinable.append(row["no_degradation"])
        else:
            row["no_degradation"] = None
        per_year[str(year)] = row
    if not determinable:
        return GateOutcome("G5", "worst_year_defense", "UNDETERMINED", f"no calendar-year coverage for {years}"), per_year
    ok = all(determinable)
    detail = "; ".join(f"{year}: combined={per_year[str(year)]['combined_return']} i5_solo={per_year[str(year)]['i5_solo_return']}" for year in years)
    return GateOutcome("G5", "worst_year_defense", _status(ok), detail), per_year


# ---------------------------------------------------------------------------
# Reference-only DSR (SPEC.md "다중검정: 누적 121회 DSR 보정") -- never gates promotion, matches
# every prior wave's own "reference-only, disclosed, not a hard gate" convention exactly
# (e.g. research/wave19_regime's own SPEC.md line 36 + report language).
# ---------------------------------------------------------------------------


def oos_performance(equity: pd.Series, split: pd.Timestamp = IS_OOS_SPLIT) -> dict[str, Any] | None:
    """SPEC.md's common convention ("IS ~2025-09 / OOS 2025-10~"), applied uniformly to every
    candidate's equity series -- reference-only (not a G1-G5 gate; SPEC.md freezes the gate
    list separately from this convention statement), matching how the rest of this repo
    reports an OOS sub-period return alongside its own gates without making it a gate itself
    (e.g. research.wave1.fam_funding.run_cached's own `stress_total_return`)."""
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
    return {
        "split": str(split),
        "is_anchor_usdt": anchor,
        "oos_end_usdt": end,
        "oos_total_return": total_return,
        "oos_annualized_return": annualized,
        "oos_days": days,
    }


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
    stable_solo_equity: pd.Series,
    stable_solo_cagr: float,
    seed_offset: int,
) -> dict[str, Any]:
    trade_pnls = np.asarray([float(t["pnl_usdt"]) for t in trades_payload], dtype=float)
    final_gamble_equity = float(gamble_equity.dropna().iloc[-1]) if len(gamble_equity.dropna()) else GAMBLE_CAPITAL

    g1 = gate_g1_structural_loss_cap(gamble_equity, trades_payload)
    g2, mc_payload = gate_g2_mc_ruin(combined_equity, seed=20_260_723 + seed_offset * 101)
    g3, g3_diagnostics = gate_g3_convexity(trade_pnls, final_gamble_equity, seed=20_260_724 + seed_offset * 103)
    g4, g4_payload = gate_g4_beats_stable_solo(combined_equity, stable_solo_cagr)
    g5, g5_payload = gate_g5_worst_year_defense(combined_equity, stable_solo_equity)

    gates = (g1, g2, g3, g4, g5)
    all_pass = all(gate.status == "PASS" for gate in gates)
    any_undetermined = any(gate.status == "UNDETERMINED" for gate in gates)

    return {
        "candidate_id": candidate_id,
        "gates": [asdict(gate) for gate in gates],
        "mc_ruin": mc_payload,
        "convexity_diagnostics": g3_diagnostics,
        "g4_detail": g4_payload,
        "g5_detail": g5_payload,
        "reference_metrics": {
            "dsr_gamble_sleeve": deflated_sharpe_reference(gamble_equity),
            "dsr_combined_system": deflated_sharpe_reference(combined_equity),
            "total_trials_disclosed": DSR_CUMULATIVE_TRIALS,
            "oos_gamble_sleeve": oos_performance(gamble_equity),
            "oos_combined_system": oos_performance(combined_equity),
            "oos_stable_solo": oos_performance(stable_solo_equity),
        },
        "overall": {
            "status": "UNDETERMINED" if (not all_pass and any_undetermined) else _status(all_pass),
            "passed": sum(gate.status == "PASS" for gate in gates),
            "undetermined": sum(gate.status == "UNDETERMINED" for gate in gates),
            "total": len(gates),
            "promoted": all_pass,
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
    "gate_g1_structural_loss_cap",
    "gate_g2_mc_ruin",
    "gate_g3_convexity",
    "gate_g4_beats_stable_solo",
    "gate_g5_worst_year_defense",
    "oos_performance",
    "skewness",
    "top_decile_contribution",
]
