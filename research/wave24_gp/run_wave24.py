#!/usr/bin/env python
"""Wave-24 (genetic programming, signal-formula search) pipeline CLI. `--stage evolve` runs the
5 GP seeds, `--stage control` runs the 5 random-tree control seeds, `--stage gates` selects the
final candidate (median of the 5 GP seeds' own best trees -- see select_final_candidate's own
docstring), opens the OOS seal EXACTLY ONCE against it, and evaluates L1-L7, `--stage report`
writes report/wave24_report.md + REGISTRY.md. All stages are cache-only (no network) -- see
research/wave24_gp/SPEC.md.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Final, Sequence, assert_never

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

from research.wave1.gate_reporting import _series
from research.wave24_gp import fitness24, gates24, gp, random_trees
from research.wave24_gp.fitness24 import MarketCache
from research.wave24_gp.reporting24 import write_wave24_report
from research.wave24_gp.tree import Node, from_dict, node_count, ramped_half_and_half, to_dict, to_formula_string

_BENCHMARK_SEED: Final = 999_000_001  # dedicated to the 1-generation timing probe -- independent of gp.SEEDS/random_trees.SEEDS

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
REGISTRY_PATH: Final = BASE_DIR / "REGISTRY.md"
I5_RESULTS_PATH: Final = BASE_DIR.parent / "wave18_idle" / "results" / "I5.json"


class Stage(StrEnum):
    EVOLVE = "evolve"
    CONTROL = "control"
    GATES = "gates"
    REPORT = "report"
    ALL = "all"


class Wave24Error(Exception):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _series_payload(series: pd.Series) -> list[dict[str, Any]]:
    return [{"timestamp": str(timestamp), "value": float(value)} for timestamp, value in series.items()]


def _tree_payload(node: Node) -> dict[str, Any]:
    return {"tree": to_dict(node), "formula": to_formula_string(node), "node_count": node_count(node)}


# ---------------------------------------------------------------------------
# evolve / control.
# ---------------------------------------------------------------------------


def _generation_payload(record: gp.GenerationRecord) -> dict[str, Any]:
    return {
        "generation": record.generation,
        "best_fitness": record.best_fitness,
        "mean_fitness": record.mean_fitness,
        "worst_fitness": record.worst_fitness,
        "mean_node_count": record.mean_node_count,
        "best_tree": _tree_payload(record.best_tree),
    }


def _measure_one_generation(cache: MarketCache) -> dict[str, Any]:
    """Task instruction: '예상 실행시간을 먼저 1세대 측정으로 추정해 리포트에 명시' -- times ONE
    fresh, uncached generation's worth of tree evaluations (POPULATION_SIZE trees, ramped
    half-and-half, the exact same distribution generation 0 of every GP seed starts from) BEFORE
    the real 5-seed evolve run, and extrapolates a generous upper-bound total-wall-clock
    estimate. Upper bound because it ignores fitness-cache hits (elites carried over generations,
    or two individuals landing on the same tree structure), which only ever REDUCE actual
    wall-clock below this estimate, never increase it -- so this is a safe basis for deciding
    whether the run is feasible before committing to it."""
    rng = np.random.default_rng(_BENCHMARK_SEED)
    population = ramped_half_and_half(rng, gp.POPULATION_SIZE)
    fitness_cache: dict[Node, fitness24.FitnessResult] = {}
    started = time.time()
    for individual in population:
        fitness24.evaluate_tree_cached(individual, cache, fitness_cache)
    elapsed = time.time() - started

    seconds_per_eval = elapsed / gp.POPULATION_SIZE
    evolve_total = seconds_per_eval * gp.EVALUATIONS_PER_SEED * len(gp.SEEDS)
    control_total = seconds_per_eval * random_trees.N_EVALUATIONS_PER_SEED * len(random_trees.SEEDS)
    estimate = {
        "population_size": gp.POPULATION_SIZE,
        "one_generation_seconds": elapsed,
        "seconds_per_evaluation": seconds_per_eval,
        "estimated_evolve_seconds_5_seeds": evolve_total,
        "estimated_control_seconds_5_seeds": control_total,
        "estimated_total_seconds": evolve_total + control_total,
        "estimated_total_minutes": (evolve_total + control_total) / 60.0,
        "note": "Generous upper bound: ignores fitness-cache hits (elites/duplicate structures), which only ever REDUCE actual wall-clock below this estimate.",
    }
    _save_json(RESULTS_DIR / "timing_estimate.json", estimate)
    print(
        f"benchmark: 1 generation ({gp.POPULATION_SIZE} fresh trees, no cache warm-up) took {elapsed:.2f}s "
        f"({seconds_per_eval * 1000.0:.1f}ms/eval) -> estimated total (evolve+control, 5 seeds each) "
        f"<= {estimate['estimated_total_minutes']:.1f} min"
    )
    return estimate


def _stage_evolve(cache: MarketCache) -> None:
    _measure_one_generation(cache)
    fitness_cache: dict[Node, fitness24.FitnessResult] = {}
    for seed in gp.SEEDS:
        started = time.time()
        result = gp.run_gp(seed, cache, fitness_cache=fitness_cache, progress=True)
        payload = {
            "seed": result.seed,
            "population_size": gp.POPULATION_SIZE,
            "generations": gp.GENERATIONS,
            "tournament_k": gp.TOURNAMENT_K,
            "crossover_probability": gp.CROSSOVER_PROBABILITY,
            "subtree_mutation_probability": gp.SUBTREE_MUTATION_PROBABILITY,
            "point_mutation_probability": gp.POINT_MUTATION_PROBABILITY,
            "elite_count": gp.ELITE_COUNT,
            "n_evaluations": result.n_evaluations,
            "n_backtests_run": result.n_backtests_run,
            "best_tree": _tree_payload(result.best_tree),
            "best_fitness": result.best_fitness,
            "history": [_generation_payload(record) for record in result.history],
            "wall_seconds": time.time() - started,
        }
        _save_json(RESULTS_DIR / f"gp_seed{seed}.json", payload)
        print(f"evolve: seed={seed} done best_fitness={result.best_fitness:.5f} backtests_run={result.n_backtests_run}/{result.n_evaluations} ({payload['wall_seconds']:.1f}s)")
    print(f"evolve: fitness cache size={len(fitness_cache)} (unique trees ever backtested across all 5 seeds)")


def _stage_control(cache: MarketCache) -> None:
    fitness_cache: dict[Node, fitness24.FitnessResult] = {}
    for seed in random_trees.SEEDS:
        started = time.time()
        result = random_trees.run_random_search(seed, cache, fitness_cache=fitness_cache, progress=True)
        payload = {
            "seed": result.seed,
            "n_evaluations": result.n_evaluations,
            "n_backtests_run": result.n_backtests_run,
            "best_tree": _tree_payload(result.best_tree),
            "best_fitness": result.best_fitness,
            "fitness_history": list(result.fitness_history),
            "wall_seconds": time.time() - started,
        }
        _save_json(RESULTS_DIR / f"random_seed{seed}.json", payload)
        print(f"control: seed={seed} done best_fitness={result.best_fitness:.5f} backtests_run={result.n_backtests_run}/{result.n_evaluations} ({payload['wall_seconds']:.1f}s)")
    print(f"control: fitness cache size={len(fitness_cache)}")


# ---------------------------------------------------------------------------
# gates: final-candidate selection + the ONE-TIME OOS evaluation + L1-L7.
# ---------------------------------------------------------------------------


def select_final_candidate(seeds: Sequence[int], best_trees: Sequence[Node], best_fitnesses: Sequence[float]) -> tuple[Node, int, float]:
    """SPEC.md 과최적화 방어 3: '5시드 재현성: 4/5 이상에서 유사 구조가 나와야 인정(수식이 매번
    완전히 다르면 노이즈 학습)'. Operationalized identically to research.wave21_ga.run_wave21.
    select_final_candidate: pick the MEDIAN (not the max) of the 5 GP seeds' own best-fitness
    trees. A single lucky seed's outlier jackpot cannot become the final candidate this way --
    the median is, by construction, insensitive to any one extreme value. Requires an ODD seed
    count (this wave's own 5) so the middle element is an exact tree, never an interpolation."""
    n = len(seeds)
    if not (len(best_trees) == len(best_fitnesses) == n) or n == 0:
        raise ValueError("select_final_candidate: seeds/trees/fitnesses must be equal-length and non-empty")
    if n % 2 == 0:
        raise ValueError(f"select_final_candidate: requires an odd seed count for an unambiguous median, got {n}")
    order = sorted(range(n), key=lambda index: best_fitnesses[index])
    median_position = order[n // 2]
    return best_trees[median_position], seeds[median_position], best_fitnesses[median_position]


def _load_i5_reference() -> tuple[pd.Series, float | None, float]:
    if not I5_RESULTS_PATH.exists():
        raise Wave24Error(f"{I5_RESULTS_PATH} not found -- research/wave18_idle must have run (I5 is this wave's own baseline, read-only)")
    payload = _load_json(I5_RESULTS_PATH)
    equity = _series(payload["equity"])
    oos_cagr = payload.get("regime_breakdown", {}).get("current_low_funding", {}).get("annualized_return")
    # I5's OWN IS-vs-OOS gap -- context for the report's overfitting-gap section (same rationale
    # as research.wave21_ga.run_wave21._load_i5_reference's own docstring): carry/funding
    # strategies are fundamentally regime-dependent, so even I5 itself shows a large raw IS>OOS
    # gap in the current low-funding OOS window -- a candidate's own raw gap is meaningless
    # without this reference.
    i5_is_cagr = fitness24.cagr(equity[equity.index <= fitness24.OOS_SPLIT])
    return equity, oos_cagr, i5_is_cagr


def _stage_gates(cache: MarketCache) -> None:
    gp_payloads = [_load_json(RESULTS_DIR / f"gp_seed{seed}.json") for seed in gp.SEEDS]
    random_payloads = [_load_json(RESULTS_DIR / f"random_seed{seed}.json") for seed in random_trees.SEEDS]

    gp_best_by_seed = [float(payload["best_fitness"]) for payload in gp_payloads]
    random_best_by_seed = [float(payload["best_fitness"]) for payload in random_payloads]
    l1 = gates24.gate_l1_gp_beats_random(gp_best_by_seed, random_best_by_seed)
    print(f"gates: L1 (GP vs random, {l1['n_wins']}/{l1['n_seeds']} seeds) -> {l1['status']}")

    seeds = [int(payload["seed"]) for payload in gp_payloads]
    best_trees = [from_dict(payload["best_tree"]["tree"]) for payload in gp_payloads]
    final_tree, source_seed, source_fitness = select_final_candidate(seeds, best_trees, gp_best_by_seed)
    print(f"gates: final candidate selected from seed={source_seed} (median of 5, IS fitness={source_fitness:.5f}) -> {to_formula_string(final_tree)}")

    # --- OOS seal opens exactly once, right here. ---
    final = fitness24.final_evaluation(final_tree, cache)

    i5_equity, i5_oos_cagr, i5_is_cagr = _load_i5_reference()
    l2 = gates24.gate_l2_beats_i5_oos(final.oos_cagr_regime_anchored, i5_oos_cagr)
    print(f"gates: L2 (final OOS {final.oos_cagr_regime_anchored} vs I5 OOS {i5_oos_cagr}) -> {l2['status']}")

    l3 = gates24.gate_l3_dsr(final.full_equity)
    print(f"gates: L3 (DSR trials={gates24.CUMULATIVE_TRIALS}, score={l3.get('score')}) -> {l3['status']}")

    l4 = gates24.gate_l4_mc_and_block(final.full_equity)
    print(f"gates: L4 (MC/block MDD) -> {l4['status']}")

    l5 = gates24.gate_l5_executability(final.full_equity, final.stress_equity)
    print(f"gates: L5 (executability/stress sign) -> {l5['status']}")

    l6 = gates24.gate_l6_paper_reproducibility(final_tree)
    print(f"gates: L6 (paper reproducibility) -> {l6['status']}")

    l7 = gates24.gate_l7_simplicity(final_tree)
    print(f"gates: L7 (formula simplicity, nodes={l7['node_count']}, kinds={l7['n_terminal_kinds']}) -> {l7['status']}")

    report = gates24.evaluate_all_gates(l1, l2, l3, l4, l5, l6, l7)
    print(f"gates: overall={report.overall} promoted={report.promoted} failure_reasons={list(report.failure_reasons)}")

    dsr_reference = fitness24.deflated_sharpe_for_trials(final.full_equity, gates24.CUMULATIVE_TRIALS)

    payload = {
        "final_tree": _tree_payload(final_tree),
        "source_seed": source_seed,
        "source_is_fitness": source_fitness,
        "gp_best_by_seed": dict(zip((str(s) for s in seeds), gp_best_by_seed)),
        "random_best_by_seed": dict(zip((str(p["seed"]) for p in random_payloads), random_best_by_seed)),
        "full_equity": _series_payload(final.full_equity),
        "is_equity": _series_payload(final.is_equity),
        "oos_equity": _series_payload(final.oos_equity),
        "stress_equity": _series_payload(final.stress_equity),
        "full_period_cagr": final.full_period_cagr,
        "is_cagr": final.is_cagr,
        "oos_cagr_self_contained": final.oos_cagr_self_contained,
        "oos_cagr_regime_anchored": final.oos_cagr_regime_anchored,
        "mdd_full": final.mdd_full,
        "regime_breakdown": final.regime_breakdown,
        "i5_reference": {"oos_cagr": i5_oos_cagr, "is_cagr": i5_is_cagr, "is_oos_gap_pp": (i5_is_cagr - i5_oos_cagr) * 100.0 if i5_oos_cagr is not None else None},
        "gates": gates24.gate_report_payload(report),
        "dsr_reference_cumulative": dsr_reference,
    }
    _save_json(RESULTS_DIR / "final_candidate.json", payload)
    print(f"gates: wrote {RESULTS_DIR / 'final_candidate.json'}")


def _stage_report() -> None:
    write_wave24_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, I5_RESULTS_PATH)
    print(f"report: wrote {REPORT_DIR / 'wave24_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-24 GP signal-formula search pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.EVOLVE:
                _stage_evolve(fitness24.build_market_cache())
            case Stage.CONTROL:
                _stage_control(fitness24.build_market_cache())
            case Stage.GATES:
                _stage_gates(fitness24.build_market_cache())
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                cache = fitness24.build_market_cache()
                _stage_evolve(cache)
                _stage_control(cache)
                _stage_gates(cache)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave24Error, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
