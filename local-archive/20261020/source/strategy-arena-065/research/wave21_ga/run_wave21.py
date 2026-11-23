#!/usr/bin/env python
"""Wave-21 (GA parameter search, I5-based) pipeline CLI. `--stage evolve` runs the 5 GA seeds,
`--stage control` runs the 5 random-search control seeds, `--stage gates` selects the final
candidate (median of the 5 GA seeds' own best genomes -- see select_final_candidate's own
docstring), opens the OOS seal EXACTLY ONCE against it, and evaluates H1-H5, `--stage report`
writes report/wave21_report.md + REGISTRY.md. All stages are cache-only (no network) -- see
research/wave21_ga/SPEC.md.
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

import pandas as pd  # noqa: PANDAS_OK

from research.wave1.gate_reporting import _series
from research.wave21_ga import fitness, ga, gates21, random_search
from research.wave21_ga.fitness import MarketCache
from research.wave21_ga.genome import Genome, from_dict
from research.wave21_ga.reporting21 import write_wave21_report

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


class Wave21Error(Exception):
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


# ---------------------------------------------------------------------------
# evolve / control.
# ---------------------------------------------------------------------------


def _generation_payload(record: ga.GenerationRecord) -> dict[str, Any]:
    return {
        "generation": record.generation,
        "best_fitness": record.best_fitness,
        "mean_fitness": record.mean_fitness,
        "worst_fitness": record.worst_fitness,
        "best_genome": record.best_genome.to_dict(),
    }


def _stage_evolve(cache: MarketCache) -> None:
    fitness_cache: dict[tuple, fitness.FitnessResult] = {}
    for seed in ga.SEEDS:
        started = time.time()
        result = ga.run_ga(seed, cache, fitness_cache=fitness_cache, progress=True)
        payload = {
            "seed": result.seed,
            "population_size": ga.POPULATION_SIZE,
            "generations": ga.GENERATIONS,
            "tournament_k": ga.TOURNAMENT_K,
            "mutation_probability": ga.MUTATION_PROBABILITY,
            "mutation_sigma_fraction": ga.MUTATION_SIGMA_FRACTION,
            "elite_count": ga.ELITE_COUNT,
            "n_evaluations": result.n_evaluations,
            "n_backtests_run": result.n_backtests_run,
            "best_genome": result.best_genome.to_dict(),
            "best_fitness": result.best_fitness,
            "history": [_generation_payload(record) for record in result.history],
            "wall_seconds": time.time() - started,
        }
        _save_json(RESULTS_DIR / f"ga_seed{seed}.json", payload)
        print(f"evolve: seed={seed} done best_fitness={result.best_fitness:.5f} backtests_run={result.n_backtests_run}/{result.n_evaluations} ({payload['wall_seconds']:.1f}s)")
    print(f"evolve: fitness cache size={len(fitness_cache)} (unique genomes ever backtested across all 5 seeds)")


def _stage_control(cache: MarketCache) -> None:
    fitness_cache: dict[tuple, fitness.FitnessResult] = {}
    for seed in random_search.SEEDS:
        started = time.time()
        result = random_search.run_random_search(seed, cache, fitness_cache=fitness_cache, progress=True)
        payload = {
            "seed": result.seed,
            "n_evaluations": result.n_evaluations,
            "n_backtests_run": result.n_backtests_run,
            "best_genome": result.best_genome.to_dict(),
            "best_fitness": result.best_fitness,
            "fitness_history": list(result.fitness_history),
            "wall_seconds": time.time() - started,
        }
        _save_json(RESULTS_DIR / f"random_seed{seed}.json", payload)
        print(f"control: seed={seed} done best_fitness={result.best_fitness:.5f} backtests_run={result.n_backtests_run}/{result.n_evaluations} ({payload['wall_seconds']:.1f}s)")
    print(f"control: fitness cache size={len(fitness_cache)}")


# ---------------------------------------------------------------------------
# gates: final-candidate selection + the ONE-TIME OOS evaluation + H1-H5.
# ---------------------------------------------------------------------------


def select_final_candidate(seeds: Sequence[int], best_genomes: Sequence[Genome], best_fitnesses: Sequence[float]) -> tuple[Genome, int, float]:
    """SPEC.md: '5회 모두에서 재현되는 개선만 인정(단일 시드 대박 무효)'. Operationalized as
    picking the MEDIAN (not the max) of the 5 GA seeds' own best-fitness genomes: sort the 5
    (seed, genome, fitness) triples by fitness and take the middle one. A single lucky seed's
    outlier jackpot cannot become the final candidate this way -- the median is, by
    construction, insensitive to any one extreme value, and can only be high if the search is
    REPRODUCIBLY finding strong genomes across independent seeds. Requires an ODD seed count
    (SPEC.md's own 5) so the middle element is an exact genome, never an interpolation between
    two."""
    n = len(seeds)
    if not (len(best_genomes) == len(best_fitnesses) == n) or n == 0:
        raise ValueError("select_final_candidate: seeds/genomes/fitnesses must be equal-length and non-empty")
    if n % 2 == 0:
        raise ValueError(f"select_final_candidate: requires an odd seed count for an unambiguous median, got {n}")
    order = sorted(range(n), key=lambda index: best_fitnesses[index])
    median_position = order[n // 2]
    return best_genomes[median_position], seeds[median_position], best_fitnesses[median_position]


def _load_i5_reference() -> tuple[pd.Series, float | None, float]:
    if not I5_RESULTS_PATH.exists():
        raise Wave21Error(f"{I5_RESULTS_PATH} not found -- research/wave18_idle must have run (I5 is this wave's own baseline, read-only)")
    payload = _load_json(I5_RESULTS_PATH)
    equity = _series(payload["equity"])
    oos_cagr = payload.get("regime_breakdown", {}).get("current_low_funding", {}).get("annualized_return")
    # I5's OWN IS-vs-OOS gap -- context for the report's overfitting-gap section: carry/funding
    # strategies are fundamentally regime-dependent (current OOS window happens to be a
    # historically low-funding stretch across the WHOLE crypto market, not specific to any one
    # genome), so even the established, already-promoted I5 baseline shows a large raw IS>OOS
    # gap. Reporting a candidate's own gap WITHOUT this reference risks mislabeling a
    # regime-driven gap as GA-specific overfitting.
    i5_is_cagr = fitness.cagr(equity[equity.index <= fitness.OOS_SPLIT])
    return equity, oos_cagr, i5_is_cagr


def _stage_gates(cache: MarketCache) -> None:
    ga_payloads = [_load_json(RESULTS_DIR / f"ga_seed{seed}.json") for seed in ga.SEEDS]
    random_payloads = [_load_json(RESULTS_DIR / f"random_seed{seed}.json") for seed in random_search.SEEDS]

    ga_best_by_seed = [float(payload["best_fitness"]) for payload in ga_payloads]
    random_best_by_seed = [float(payload["best_fitness"]) for payload in random_payloads]
    h1 = gates21.gate_h1_ga_beats_random(ga_best_by_seed, random_best_by_seed)
    print(f"gates: H1 (GA vs random, {h1['n_wins']}/{h1['n_seeds']} seeds) -> {h1['status']}")

    seeds = [int(payload["seed"]) for payload in ga_payloads]
    best_genomes = [from_dict(payload["best_genome"]) for payload in ga_payloads]
    final_genome, source_seed, source_fitness = select_final_candidate(seeds, best_genomes, ga_best_by_seed)
    print(f"gates: final candidate selected from seed={source_seed} (median of 5, IS fitness={source_fitness:.5f}) -> {final_genome.to_dict()}")

    # --- OOS seal opens exactly once, right here. ---
    final = fitness.final_evaluation(final_genome, cache)

    i5_equity, i5_oos_cagr, i5_is_cagr = _load_i5_reference()
    h2 = gates21.gate_h2_beats_i5_oos(final.oos_cagr_regime_anchored, i5_oos_cagr)
    print(f"gates: H2 (final OOS {final.oos_cagr_regime_anchored} vs I5 OOS {i5_oos_cagr}) -> {h2['status']}")

    h3 = gates21.gate_h3_dsr(final.full_equity)
    print(f"gates: H3 (DSR trials={gates21.GA_TRIALS}, score={h3.get('score')}) -> {h3['status']}")

    h4 = gates21.gate_h4_inherited(final_genome, final.full_equity, final.stress_equity)
    print(f"gates: H4 (inherited MC/block/feasibility/stress) -> {h4['status']}")

    h5 = gates21.gate_h5_worst_years(final.full_equity, i5_equity)
    print(f"gates: H5 (worst years {gates21.WORST_YEARS} vs I5) -> {h5['status']}")

    report = gates21.evaluate_all_gates(h1, h2, h3, h4, h5)
    print(f"gates: overall={report.overall} promoted={report.promoted} failure_reasons={list(report.failure_reasons)}")

    dsr_reference = fitness.deflated_sharpe_for_trials(final.full_equity, gates21.CUMULATIVE_TRIALS_WITH_GA)

    payload = {
        "final_genome": final_genome.to_dict(),
        "source_seed": source_seed,
        "source_is_fitness": source_fitness,
        "ga_best_by_seed": dict(zip((str(s) for s in seeds), ga_best_by_seed)),
        "random_best_by_seed": dict(zip((str(s) for s in (p["seed"] for p in random_payloads)), random_best_by_seed)),
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
        "i5_reference": {"oos_cagr": i5_oos_cagr, "is_cagr": i5_is_cagr, "is_oos_gap_pp": (i5_is_cagr - i5_oos_cagr) * 100.0},
        "gates": gates21.gate_report_payload(report),
        "dsr_reference_cumulative_121_plus_7500": dsr_reference,
    }
    _save_json(RESULTS_DIR / "final_candidate.json", payload)
    print(f"gates: wrote {RESULTS_DIR / 'final_candidate.json'}")


def _stage_report() -> None:
    write_wave21_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH, I5_RESULTS_PATH)
    print(f"report: wrote {REPORT_DIR / 'wave21_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-21 GA parameter search pipeline")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.EVOLVE:
                _stage_evolve(fitness.build_market_cache())
            case Stage.CONTROL:
                _stage_control(fitness.build_market_cache())
            case Stage.GATES:
                _stage_gates(fitness.build_market_cache())
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                cache = fitness.build_market_cache()
                _stage_evolve(cache)
                _stage_control(cache)
                _stage_gates(cache)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave21Error, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
