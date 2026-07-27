# Wave-21 gate evaluation (SPEC.md "게이트 (승격 조건, 전부 충족)" -- H1-H5).
#
# H4 reuses the EXACT SAME numeric bars and simulation methodology as
# research/wave13_liquidity/gates13.py / research/wave18_idle/gates18.py (ruin P(<$50)<1%,
# MC p05>$100, block MDD p95<=10%, $100 executability, measured-slippage x3 stress: sign
# preserved AND stress block MDD p95<=15%) -- reimplemented locally rather than imported,
# matching every prior wave's own precedent ("이 모듈은 ...의 것을 가져오는 대신 로컬로
# 재구현한다"), parametrized by the FINAL CANDIDATE's own genome (leg_fraction/top_k_pairs)
# rather than a fixed per-wave config, since those two genes now vary.
#
# H1/H2/H3/H5 are NEW to this wave (SPEC.md 오염 차단 장치 3/1/4 + 최악연도 승계):
#   H1 - GA's best fitness must beat the random-search control's best fitness in >= 4 of the
#        5 matched seeds (gates the evolutionary MECHANISM itself, not any one genome).
#   H2 - the final candidate's OOS (2025-10~) CAGR must beat I5's own OOS CAGR (read
#        read-only from research/wave18_idle/results/I5.json -- never recomputed, never
#        modified; this wave does not touch wave18's files).
#   H3 - Deflated Sharpe on the final candidate's FULL equity, trials=7,500 (this wave's own
#        evaluation count: 1,500 x 5 seeds -- SPEC.md's H3 line), must be > 0.
#   H5 - the final candidate's 2022 and 2025 annualized returns must not be worse than I5's
#        own (same two read-only-I5.json years) -- SPEC.md's "최악연도 대비 악화 없음".
#
# Promotion = H1 AND H2 AND H3 AND H4 AND H5, all-or-nothing (SPEC.md: "하나라도 미달이면
# 'GA로도 I5 못 넘음' 정직 보고").

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Final, Sequence

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.validation.deep_stats import DeepValidationError, TimedValue, deflated_sharpe
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, RESERVE_FRACTION, TOTAL_CAPITAL
from research.wave21_ga.genome import Genome

MC_PATHS: Final = 10_000
BLOCK_PATHS: Final = 1_000
BLOCK_DAYS: Final = 90
SEED: Final = 20_260_727  # this wave's own freeze date (2026-07-27), matching wave10-18's freeze-date-as-seed convention

S2_RUIN_THRESHOLD_USDT: Final = 50.0
S2_RUIN_PROBABILITY_MAX: Final = 0.01
S2_P05_FLOOR_USDT: Final = 100.0
H4_BLOCK_MDD_P95_MAX: Final = 0.10
H4_STRESS_BLOCK_MDD_P95_MAX: Final = 0.15

GA_TRIALS: Final = 1_500 * 5  # 7,500 -- SPEC.md H3's own trial count for THIS wave's gate
PRIOR_CUMULATIVE_TRIALS: Final = 121  # SPEC.md line 41's own disclosed prior-wave cumulative candidate count, frozen, not re-derived (same "trust the frozen figure" precedent as gates18.DSR_CUMULATIVE_TRIALS)
CUMULATIVE_TRIALS_WITH_GA: Final = PRIOR_CUMULATIVE_TRIALS + GA_TRIALS  # reference-only figure for the report's multi-testing disclosure section; never gates promotion (same principle as every prior wave's own DSR_CUMULATIVE_TRIALS)

WORST_YEARS: Final[tuple[int, ...]] = (2022, 2025)  # SPEC.md H5: "최악연도(2022·2025)"


def leg_usdt(genome: Genome) -> float:
    return genome.leg_fraction * ACTIVE_CAPITAL


def gross_usdt(genome: Genome) -> float:
    return 2.0 * genome.top_k_pairs * genome.leg_fraction * ACTIVE_CAPITAL


# ---------------------------------------------------------------------------
# H1: GA vs random control.
# ---------------------------------------------------------------------------


def gate_h1_ga_beats_random(ga_best_by_seed: Sequence[float], random_best_by_seed: Sequence[float]) -> dict[str, Any]:
    if len(ga_best_by_seed) != len(random_best_by_seed) or len(ga_best_by_seed) == 0:
        raise ValueError("gate_h1_ga_beats_random: GA and random seed-count lists must be equal length and non-empty")
    wins = [bool(ga_value > random_value) for ga_value, random_value in zip(ga_best_by_seed, random_best_by_seed)]
    n_wins = int(sum(wins))
    threshold = 4  # SPEC.md: "5시드 중 4회 이상"
    ok = n_wins >= threshold
    return {
        "ga_best_by_seed": [float(v) for v in ga_best_by_seed],
        "random_best_by_seed": [float(v) for v in random_best_by_seed],
        "per_seed_ga_wins": wins,
        "n_wins": n_wins,
        "n_seeds": len(wins),
        "threshold": threshold,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# H2: final candidate OOS vs I5 OOS (read-only, research/wave18_idle/results/I5.json).
# ---------------------------------------------------------------------------


def gate_h2_beats_i5_oos(final_oos_cagr: float | None, i5_oos_cagr: float | None) -> dict[str, Any]:
    ok = final_oos_cagr is not None and i5_oos_cagr is not None and final_oos_cagr > i5_oos_cagr
    gap_pp = (final_oos_cagr - i5_oos_cagr) * 100.0 if (final_oos_cagr is not None and i5_oos_cagr is not None) else None
    return {
        "final_oos_cagr": final_oos_cagr,
        "i5_oos_cagr": i5_oos_cagr,
        "gap_pp": gap_pp,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# H3: DSR, trials=7,500.
# ---------------------------------------------------------------------------


def gate_h3_dsr(full_equity: pd.Series, trials: int = GA_TRIALS) -> dict[str, Any]:
    clean = full_equity.dropna()
    if len(clean) < 4:
        return {"status": "FAIL", "reason": "insufficient observations for DSR (<4)"}
    timed = tuple(TimedValue(pd.Timestamp(ts).to_pydatetime(), float(value)) for ts, value in clean.items())
    try:
        dsr = deflated_sharpe(timed, trials=trials)
    except DeepValidationError as error:
        return {"status": "FAIL", "reason": str(error)}
    ok = dsr.score > 0.0
    return {
        "score": dsr.score,
        "probability": dsr.probability,
        "trials": dsr.trials,
        "observed_sharpe": dsr.observed_sharpe,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# H4: inherited S1-S5-equivalent bar (MC principal-preservation, block-shuffle MDD, $ feasibility,
# x3 stress). Local reimplementation of gates13/gates18's own _daily_returns/_simulate_mc/
# _blocks/_mdd/_block_shuffle -- identical formulas, parametrized by genome instead of a fixed config.
# ---------------------------------------------------------------------------


def _daily_returns(equity: pd.Series) -> tuple[tuple[pd.Timestamp, ...], np.ndarray]:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave21_ga equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave21_ga equity returns contain invalid values")
    timestamps = tuple(pd.Timestamp(ts) for ts in clean.index[1:])
    return timestamps, returns


def _simulate_mc(returns: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    finals = np.empty(MC_PATHS, dtype=float)
    chunk_size = 250
    for start in range(0, MC_PATHS, chunk_size):
        stop = min(start + chunk_size, MC_PATHS)
        samples = rng.choice(returns, size=(stop - start, returns.size), replace=True)
        growth = np.prod(1.0 + np.clip(samples, -0.999999, None), axis=1)
        finals[start:stop] = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * growth
    return {
        "p05": float(np.quantile(finals, 0.05)),
        "ruin_probability": float(np.mean(finals < S2_RUIN_THRESHOLD_USDT)),
        "mean": float(np.mean(finals)),
        "median": float(np.median(finals)),
        "paths": MC_PATHS,
    }


def _blocks(timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray) -> tuple[np.ndarray, ...]:
    anchor = timestamps[0]
    grouped: dict[int, list[float]] = {}
    for timestamp, value in zip(timestamps, returns):
        index = (timestamp - anchor).days // BLOCK_DAYS
        grouped.setdefault(index, []).append(float(value))
    return tuple(np.asarray(grouped[key], dtype=float) for key in sorted(grouped))


def _mdd(values: np.ndarray) -> float:
    curve = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * np.cumprod(1.0 + np.clip(values, -0.999999, None))
    peaks = np.maximum.accumulate(np.concatenate(([TOTAL_CAPITAL], curve)))
    return float(np.max(1.0 - curve / peaks[1:]))


def _block_shuffle(timestamps: tuple[pd.Timestamp, ...], returns: np.ndarray, seed: int) -> dict[str, float]:
    blocks = _blocks(timestamps, returns)
    rng = np.random.default_rng(seed)
    mdds = np.empty(BLOCK_PATHS, dtype=float)
    finals = np.empty(BLOCK_PATHS, dtype=float)
    for index in range(BLOCK_PATHS):
        path = np.concatenate([blocks[item] for item in rng.permutation(len(blocks))])
        mdds[index] = _mdd(path)
        finals[index] = TOTAL_CAPITAL * RESERVE_FRACTION + ACTIVE_CAPITAL * np.prod(1.0 + np.clip(path, -0.999999, None))
    return {
        "block_days": BLOCK_DAYS,
        "block_count": len(blocks),
        "paths": BLOCK_PATHS,
        "mdd_p95": float(np.quantile(mdds, 0.95)),
        "final_p05": float(np.quantile(finals, 0.05)),
    }


def gate_h4_inherited(genome: Genome, equity: pd.Series, stress_equity: pd.Series) -> dict[str, Any]:
    """A genome combining a high top_k_pairs with a high leg_fraction can legitimately push
    gross exposure past 1x (SPEC.md's gene table does not itself forbid that combination --
    see genome.py's own module docstring); this gate is precisely what is supposed to catch
    that, not a place that should ever crash. Both the base AND stress equity/MC/block-shuffle
    computations are therefore guarded the same way: `_daily_returns` raises ValueError on a
    non-positive or non-finite equity path (a plausible outcome for a pathological
    high-leverage genome), and that must fail THIS gate, never take down the whole `gates`
    stage on the one genome it most needs to evaluate."""
    leg = leg_usdt(genome)
    gross = gross_usdt(genome)
    min_order_ok = leg >= MIN_ORDER_USDT
    gross_ok = gross <= ACTIVE_CAPITAL + 1e-9  # SPEC.md '1x' -- same 1e-9 float-noise slack gates13/18 use at this exact boundary

    try:
        timestamps, returns = _daily_returns(equity)
        mc = _simulate_mc(returns, SEED)
        p05_ok = mc["p05"] > S2_P05_FLOOR_USDT
        ruin_ok = mc["ruin_probability"] < S2_RUIN_PROBABILITY_MAX
        block = _block_shuffle(timestamps, returns, SEED + 103)
        block_mdd_ok = block["mdd_p95"] <= H4_BLOCK_MDD_P95_MAX
    except ValueError as error:
        mc = {"error": str(error)}
        p05_ok = False
        ruin_ok = False
        block = None
        block_mdd_ok = False

    try:
        stress_timestamps, stress_returns = _daily_returns(stress_equity)
        stress_block = _block_shuffle(stress_timestamps, stress_returns, SEED + 107)
        stress_mdd_ok = stress_block["mdd_p95"] <= H4_STRESS_BLOCK_MDD_P95_MAX
        stress_sign_ok = float(stress_equity.iloc[-1]) > float(stress_equity.iloc[0])
    except ValueError:
        stress_block = None
        stress_mdd_ok = False
        stress_sign_ok = False

    ok = min_order_ok and gross_ok and p05_ok and ruin_ok and block_mdd_ok and stress_mdd_ok and stress_sign_ok
    return {
        "leg_usdt_nominal": leg,
        "gross_usdt_nominal": gross,
        "min_order_usdt": MIN_ORDER_USDT,
        "min_order_feasible": min_order_ok,
        "gross_leverage_1x_ok": gross_ok,
        "mc": mc,
        "mc_p05_ok": p05_ok,
        "mc_ruin_ok": ruin_ok,
        "block_mdd_p95": block["mdd_p95"] if block is not None else None,
        "block_mdd_ok": block_mdd_ok,
        "stress_block_mdd_p95": stress_block["mdd_p95"] if stress_block is not None else None,
        "stress_mdd_ok": stress_mdd_ok,
        "stress_sign_preserved": stress_sign_ok,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# H5: worst-year (2022, 2025) vs I5, read-only from I5's OWN saved equity series.
# ---------------------------------------------------------------------------


def yearly_annualized_returns(equity: pd.Series) -> dict[int, float]:
    """Same anchoring convention as research/wave18_idle/reporting18.py's own _yearly_stats /
    research.wave10_carry100.regime._regime_return: anchor a calendar year's return at the
    last observation ON OR BEFORE that year's Dec-31 boundary (not the year's own first
    observation), so a year's return correctly includes the transition day. Reimplemented
    locally (small, per-wave-local helper) rather than imported, matching this repo's own
    convention of not cross-importing report-shaped helpers between waves."""
    if equity.empty:
        return {}
    years = sorted({int(pd.Timestamp(ts).year) for ts in equity.index})
    out: dict[int, float] = {}
    for year in years:
        boundary_start = pd.Timestamp(f"{year - 1}-12-31T23:59:59Z")
        boundary_end = pd.Timestamp(f"{year}-12-31T23:59:59Z")
        window = equity[(equity.index > boundary_start) & (equity.index <= boundary_end)]
        if window.empty:
            continue
        pre = equity[equity.index <= boundary_start]
        anchor_value = float(pre.iloc[-1]) if len(pre) else float(window.iloc[0])
        anchor_ts = boundary_start if len(pre) else pd.Timestamp(window.index[0])
        end_value = float(window.iloc[-1])
        days = max((pd.Timestamp(window.index[-1]) - anchor_ts).total_seconds() / 86_400.0, 1.0)
        if anchor_value <= 0.0:
            continue
        growth = end_value / anchor_value
        out[year] = float(growth ** (365.0 / days) - 1.0) if growth > 0.0 else -1.0
    return out


def gate_h5_worst_years(final_equity: pd.Series, i5_equity: pd.Series, years: tuple[int, ...] = WORST_YEARS) -> dict[str, Any]:
    final_years = yearly_annualized_returns(final_equity)
    i5_years = yearly_annualized_returns(i5_equity)
    detail: dict[str, Any] = {}
    ok = True
    for year in years:
        final_value = final_years.get(year)
        i5_value = i5_years.get(year)
        if final_value is None or i5_value is None:
            detail[str(year)] = {"final": final_value, "i5": i5_value, "worse": None, "note": "missing data for this year in one series"}
            continue
        worse = final_value < i5_value
        detail[str(year)] = {"final": final_value, "i5": i5_value, "worse": worse}
        if worse:
            ok = False
    return {"years": detail, "status": "PASS" if ok else "FAIL"}


# ---------------------------------------------------------------------------
# Overall promotion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateReport:
    h1: dict[str, Any]
    h2: dict[str, Any]
    h3: dict[str, Any]
    h4: dict[str, Any]
    h5: dict[str, Any]
    overall: str
    promoted: bool
    failure_reasons: tuple[str, ...]


def evaluate_all_gates(h1: dict[str, Any], h2: dict[str, Any], h3: dict[str, Any], h4: dict[str, Any], h5: dict[str, Any]) -> GateReport:
    gates = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5}
    failure_reasons = tuple(name for name, gate in gates.items() if gate.get("status") != "PASS")
    overall = "PASS" if not failure_reasons else "FAIL"
    return GateReport(h1=h1, h2=h2, h3=h3, h4=h4, h5=h5, overall=overall, promoted=overall == "PASS", failure_reasons=failure_reasons)


def gate_report_payload(report: GateReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["failure_reasons"] = list(report.failure_reasons)
    return payload


__all__ = [
    "CUMULATIVE_TRIALS_WITH_GA",
    "GA_TRIALS",
    "PRIOR_CUMULATIVE_TRIALS",
    "WORST_YEARS",
    "GateReport",
    "evaluate_all_gates",
    "gate_h1_ga_beats_random",
    "gate_h2_beats_i5_oos",
    "gate_h3_dsr",
    "gate_h4_inherited",
    "gate_h5_worst_years",
    "gate_report_payload",
    "gross_usdt",
    "leg_usdt",
    "yearly_annualized_returns",
]
