# Wave-23 gate evaluation (task instruction "게이트 K1~K6").
#
# K1 - GA's best fitness must beat the random-search control's best fitness in >= 4 of 5
#      matched seeds (identical mechanism to wave21_ga.gates21.gate_h1_ga_beats_random).
# K2 - final candidate's OOS short-window fitness (SAME formula as fitness23.compute_fitness,
#      applied to the OOS-only equity slice) must beat I5's own OOS short-window fitness (I5's
#      equity read verbatim from research/wave18_idle/results/I5.json -- never recomputed).
# K3 - Deflated Sharpe on the final candidate's OWN full equity curve (task instruction: "DSR은
#      반드시 최종 승격 개체 자신의 equity curve로 계산" -- wave21_ga reported H3's DSR from the
#      SAME genome that ultimately failed its own gross-leverage gate (H4), and the genome that
#      actually got promoted into paper tracking after a manual fix (wave22_overfit's G1) had a
#      DIFFERENT, substantially worse DSR that was never itself gated -- see REGISTRY.md), at
#      CUMULATIVE trials = 121 (wave1-20) + 7,500 (wave21_ga) + this wave's own total eval count
#      (GA 7,500 + random-control 7,500 = 15,000) = 22,621. Must be > 0.
# K4 - ruin defense: Monte-Carlo P(final capital < $50) < 10%, AND single max drawdown <= 30%
#      of the $100 PRINCIPAL (not the $90 active-capital basis the engine's own equity curve is
#      denominated in -- the untouched $10 reserve is never at risk, so a drawdown is measured
#      in principal dollars, matching every prior wave's TOTAL_CAPITAL*RESERVE_FRACTION +
#      ACTIVE_CAPITAL*growth convention).
# K5 - executability: every leg >= MIN_ORDER_USDT ($5), gross <= ACTIVE_CAPITAL (already
#      guaranteed BY CONSTRUCTION via genome23.Genome.normalized_weight -- checked here anyway,
#      empirically, from the realized backtest weights, as defense-in-depth against an engine
#      bug rather than trusting the construction claim alone), and the top-25%-window average
#      return keeps its sign under x3 measured-slippage stress.
# K6 - SPEC-EXECUTION CONSISTENCY: does research/paper/track.py's ACTUAL dispatch/ledger shape
#      support what the final genome asks for? research/paper/track.py's `_signal_and_positions`
#      only has two dispatch shapes (a FundingCandidate-style top-k carry selector, and a
#      pre-registered wave3 CandidateConfig-style momentum selector) and research/paper/ledger.py's
#      `Position` dataclass has NO stop_loss/take_profit/entry-day fields at all -- so a genome
#      using breakout/funding_spike/convex_dual, or ANY stop_loss/take_profit/max_concurrent>1,
#      cannot currently be automatically operated by the live paper tracker, and a genome whose
#      universe_breadth exceeds what research/paper/market_data.py actually fetches live for
#      that family would silently sit in cash for any pick outside that smaller live set --
#      EXACTLY the G1 universe-mismatch incident this wave's SPEC.md was written to never repeat.
#      This gate never modifies research/paper/* (out of this wave's scope); it only verifies.
#
# Promotion = K1 AND K2 AND K3 AND K4 AND K5 AND K6, all-or-nothing.

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

from research.paper.market_data import CARRY_VOLUME_LIMIT, CRYPTO_VOLUME_LIMIT
from research.validation.deep_stats import DeepValidationError, TimedValue, deflated_sharpe
from research.wave10_carry100.engine import ACTIVE_CAPITAL, MIN_ORDER_USDT, RESERVE_FRACTION, TOTAL_CAPITAL
from research.wave23_ga_short import fitness23, genome23
from research.wave23_ga_short.genome23 import Genome

MC_PATHS: Final = 10_000
SEED: Final = 20_260_728  # this wave's own freeze date (2026-07-28)

K4_RUIN_FLOOR_USDT: Final = 50.0  # task instruction "P(최종 < $50)"
K4_RUIN_PROBABILITY_MAX: Final = 0.10  # task instruction "< 10%"
K4_MAX_SINGLE_LOSS_FRACTION_OF_PRINCIPAL: Final = 0.30  # task instruction "단일 최대손실 <= 원금 30%"

GA_TRIALS: Final = 1_500 * 5  # 7,500
RANDOM_TRIALS: Final = 1_500 * 5  # 7,500
THIS_WAVE_TOTAL_TRIALS: Final = GA_TRIALS + RANDOM_TRIALS  # 15,000 -- "이번 총평가수"
PRIOR_CUMULATIVE_TRIALS_BEFORE_WAVE21: Final = 121  # gates21.PRIOR_CUMULATIVE_TRIALS, frozen figure, not re-derived
WAVE21_GA_TRIALS: Final = 7_500  # gates21.GA_TRIALS
CUMULATIVE_TRIALS: Final = PRIOR_CUMULATIVE_TRIALS_BEFORE_WAVE21 + WAVE21_GA_TRIALS + THIS_WAVE_TOTAL_TRIALS  # 121 + 7,500 + 15,000 = 22,621 -- task instruction "누적시행 = 121 + 7500 + 이번 총평가수"

# K6 -- paper tracker's ACTUAL supported shapes (research/paper/track.py, read-only reference; see module docstring).
PAPER_SUPPORTED_KINDS: Final[frozenset[str]] = frozenset({genome23.STRATEGY_KIND_CARRY, genome23.STRATEGY_KIND_MOMENTUM})
PAPER_CARRY_UNIVERSE_CAP: Final = CARRY_VOLUME_LIMIT + 2  # research.paper.market_data.CARRY_VOLUME_LIMIT (40) + BTC/ETH always-unioned majors
PAPER_MOMENTUM_UNIVERSE_CAP: Final = CRYPTO_VOLUME_LIMIT  # research.paper.market_data.CRYPTO_VOLUME_LIMIT (150)


def leg_usdt(genome: Genome) -> float:
    return genome.normalized_weight * ACTIVE_CAPITAL


def gross_usdt(genome: Genome) -> float:
    return genome.normalized_weight * genome.max_concurrent * ACTIVE_CAPITAL


# ---------------------------------------------------------------------------
# K1: GA vs random control.
# ---------------------------------------------------------------------------


def gate_k1_ga_beats_random(ga_best_by_seed: Sequence[float], random_best_by_seed: Sequence[float]) -> dict[str, Any]:
    if len(ga_best_by_seed) != len(random_best_by_seed) or len(ga_best_by_seed) == 0:
        raise ValueError("gate_k1_ga_beats_random: GA and random seed-count lists must be equal length and non-empty")
    wins = [bool(ga_value > random_value) for ga_value, random_value in zip(ga_best_by_seed, random_best_by_seed)]
    n_wins = int(sum(wins))
    threshold = 4  # task instruction "5회 중 4+"
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
# K2: final candidate OOS short-window fitness vs I5 OOS short-window fitness.
# ---------------------------------------------------------------------------


def gate_k2_beats_i5_oos(final_oos_fitness: float | None, i5_oos_fitness: float | None) -> dict[str, Any]:
    ok = final_oos_fitness is not None and i5_oos_fitness is not None and final_oos_fitness > i5_oos_fitness
    gap = (final_oos_fitness - i5_oos_fitness) if (final_oos_fitness is not None and i5_oos_fitness is not None) else None
    return {
        "final_oos_fitness": final_oos_fitness,
        "i5_oos_fitness": i5_oos_fitness,
        "gap": gap,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# K3: DSR at cumulative trials, on the final candidate's OWN full equity curve.
# ---------------------------------------------------------------------------


def gate_k3_dsr(full_equity: pd.Series, trials: int = CUMULATIVE_TRIALS) -> dict[str, Any]:
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
# K4: ruin defense (MC + principal-basis max drawdown).
# ---------------------------------------------------------------------------


def _daily_returns(equity: pd.Series) -> np.ndarray:
    clean = equity.dropna().astype(float)
    values = clean.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("wave23_ga_short equity series must have >=2 finite, positive observations")
    returns = values[1:] / values[:-1] - 1.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("wave23_ga_short equity returns contain invalid values")
    return returns


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
        "ruin_probability_below_50": float(np.mean(finals < K4_RUIN_FLOOR_USDT)),
        "mean": float(np.mean(finals)),
        "median": float(np.median(finals)),
        "paths": MC_PATHS,
    }


def gate_k4_ruin_defense(full_equity: pd.Series) -> dict[str, Any]:
    try:
        returns = _daily_returns(full_equity)
        mc = _simulate_mc(returns, SEED)
        ruin_ok = mc["ruin_probability_below_50"] < K4_RUIN_PROBABILITY_MAX
    except ValueError as error:
        mc = {"error": str(error)}
        ruin_ok = False

    active_basis_mdd = _max_drawdown(full_equity)
    max_dollar_loss = ACTIVE_CAPITAL * active_basis_mdd
    max_loss_fraction_of_principal = max_dollar_loss / TOTAL_CAPITAL
    single_loss_ok = max_loss_fraction_of_principal <= K4_MAX_SINGLE_LOSS_FRACTION_OF_PRINCIPAL

    ok = ruin_ok and single_loss_ok
    return {
        "mc": mc,
        "mc_ruin_ok": ruin_ok,
        "active_basis_mdd": active_basis_mdd,
        "max_dollar_loss_usdt": max_dollar_loss,
        "max_loss_fraction_of_principal": max_loss_fraction_of_principal,
        "single_loss_ok": single_loss_ok,
        "status": "PASS" if ok else "FAIL",
    }


def _max_drawdown(equity: pd.Series) -> float:
    values = equity.to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.nan_to_num(1.0 - values / peaks, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(drawdown))


# ---------------------------------------------------------------------------
# K5: executability (leg size, gross cap, stress-sign preservation).
# ---------------------------------------------------------------------------


def gate_k5_executability(genome: Genome, realized_signed_weights: np.ndarray, full_equity: pd.Series, stress_equity: pd.Series) -> dict[str, Any]:
    leg = leg_usdt(genome)
    gross_by_construction = gross_usdt(genome)
    min_order_ok = leg >= MIN_ORDER_USDT

    realized_gross_max_usdt = float(np.max(np.sum(np.abs(realized_signed_weights), axis=1))) * ACTIVE_CAPITAL if realized_signed_weights.size else 0.0
    gross_ok = gross_by_construction <= ACTIVE_CAPITAL + 1e-9 and realized_gross_max_usdt <= ACTIVE_CAPITAL + 1e-6

    base_top25 = fitness23.compute_fitness(full_equity).top_quantile_mean_return
    stress_top25 = fitness23.compute_fitness(stress_equity).top_quantile_mean_return
    sign_ok = base_top25 > 0.0 and stress_top25 > 0.0

    ok = min_order_ok and gross_ok and sign_ok
    return {
        "leg_usdt": leg,
        "gross_usdt_by_construction": gross_by_construction,
        "realized_gross_max_usdt": realized_gross_max_usdt,
        "min_order_usdt": MIN_ORDER_USDT,
        "min_order_ok": min_order_ok,
        "gross_ok": gross_ok,
        "base_top25_mean_return": base_top25,
        "stress_top25_mean_return": stress_top25,
        "stress_sign_preserved": sign_ok,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# K6: spec-execution consistency with the LIVE paper tracker.
# ---------------------------------------------------------------------------


def gate_k6_paper_reproducibility(genome: Genome) -> dict[str, Any]:
    reasons: list[str] = []

    kind_ok = genome.strategy_kind in PAPER_SUPPORTED_KINDS
    if not kind_ok:
        reasons.append(
            f"strategy_kind={genome.strategy_kind!r}: research/paper/track.py's _signal_and_positions has no dispatch "
            "path for it (only a FundingCandidate-style carry selector and a pre-registered wave3 CandidateConfig "
            "momentum selector exist)"
        )

    cap = PAPER_CARRY_UNIVERSE_CAP if genome.strategy_kind == genome23.STRATEGY_KIND_CARRY else PAPER_MOMENTUM_UNIVERSE_CAP
    breadth_ok = genome.universe_breadth <= cap
    if not breadth_ok:
        reasons.append(
            f"universe_breadth={genome.universe_breadth} > paper tracker's actual live-fetched universe for this "
            f"family (cap={cap}, research/paper/market_data.py) -- same class of mismatch as the G1 permanent-cash incident"
        )

    # NOTE on max_concurrent: NOT gated here, deliberately. Paper's EXISTING _carry_positions/
    # _momentum_positions already open exactly `candidate.top_k` positions and split equity
    # across them (F1e already runs top_k up to 4 live) -- so max_concurrent>1 is faithfully
    # reproducible via an ordinary candidates.py registration (top_k=max_concurrent), unlike
    # the three checks below, which are gaps no registration can paper over: strategy_kind
    # needs signal-computation code that does not exist, universe_breadth is capped by a
    # hardcoded live-fetch constant, and stop_loss/take_profit have no position-model field or
    # exit-check logic anywhere in ledger.py/track.py to attach to.
    exit_mechanics_ok = genome.stop_loss_pct is None and genome.take_profit_pct is None
    if not exit_mechanics_ok:
        reasons.append(
            "research/paper/ledger.py's Position dataclass has no stop_loss/take_profit/entry-day fields, and "
            "track.py's _carry_positions/_momentum_positions never check one -- an automatic stop-loss/take-profit "
            "exit cannot currently be enforced by the live paper tracker as-is (positions there only close via "
            "re-ranking/rotation on the next daily run, never a price-triggered rule)"
        )

    holding_days_ok = genome.holding_days >= 1  # paper's run_once operates on a daily cadence -- always true by gene construction, kept explicit for the record

    ok = kind_ok and breadth_ok and exit_mechanics_ok and holding_days_ok
    return {
        "strategy_kind": genome.strategy_kind,
        "strategy_kind_supported": kind_ok,
        "universe_breadth": genome.universe_breadth,
        "paper_universe_cap": cap,
        "universe_breadth_ok": breadth_ok,
        "exit_mechanics_reproducible": exit_mechanics_ok,
        "max_concurrent": genome.max_concurrent,
        "max_concurrent_note": "not gated -- paper's existing top_k mechanism (F1e runs top_k=4 live) already supports this via ordinary registration",
        "holding_days_ok": holding_days_ok,
        "reasons": reasons,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Overall promotion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateReport:
    k1: dict[str, Any]
    k2: dict[str, Any]
    k3: dict[str, Any]
    k4: dict[str, Any]
    k5: dict[str, Any]
    k6: dict[str, Any]
    overall: str
    promoted: bool
    failure_reasons: tuple[str, ...]


def evaluate_all_gates(k1: dict[str, Any], k2: dict[str, Any], k3: dict[str, Any], k4: dict[str, Any], k5: dict[str, Any], k6: dict[str, Any]) -> GateReport:
    gates = {"K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5, "K6": k6}
    failure_reasons = tuple(name for name, gate in gates.items() if gate.get("status") != "PASS")
    overall = "PASS" if not failure_reasons else "FAIL"
    return GateReport(k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, overall=overall, promoted=overall == "PASS", failure_reasons=failure_reasons)


def gate_report_payload(report: GateReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["failure_reasons"] = list(report.failure_reasons)
    return payload


__all__ = [
    "CUMULATIVE_TRIALS",
    "GA_TRIALS",
    "K4_MAX_SINGLE_LOSS_FRACTION_OF_PRINCIPAL",
    "K4_RUIN_FLOOR_USDT",
    "K4_RUIN_PROBABILITY_MAX",
    "PAPER_CARRY_UNIVERSE_CAP",
    "PAPER_MOMENTUM_UNIVERSE_CAP",
    "PAPER_SUPPORTED_KINDS",
    "PRIOR_CUMULATIVE_TRIALS_BEFORE_WAVE21",
    "RANDOM_TRIALS",
    "THIS_WAVE_TOTAL_TRIALS",
    "WAVE21_GA_TRIALS",
    "GateReport",
    "evaluate_all_gates",
    "gate_k1_ga_beats_random",
    "gate_k2_beats_i5_oos",
    "gate_k3_dsr",
    "gate_k4_ruin_defense",
    "gate_k5_executability",
    "gate_k6_paper_reproducibility",
    "gate_report_payload",
    "gross_usdt",
    "leg_usdt",
]
