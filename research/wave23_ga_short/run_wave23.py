#!/usr/bin/env python
"""Wave-23 (GA re-search, short-term profit maximization) pipeline CLI. `--stage evolve` runs
the 5 GA seeds, `--stage control` runs the 5 random-search control seeds, `--stage gates`
selects the final candidate (median of the 5 GA seeds' own best genomes), opens the OOS seal
EXACTLY ONCE against it, and evaluates K1-K6, `--stage report` writes report/wave23_report.md +
REGISTRY.md. All stages are cache-only (no network) -- see research/wave23_ga_short/SPEC.md.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from research.wave23_ga_short import engine23, fitness23, ga23, gates23, random_search23
from research.wave23_ga_short.engine23 import MarketCache
from research.wave23_ga_short.genome23 import Genome, from_dict
from research.wave23_ga_short.reporting23 import write_wave23_report

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


class Wave23Error(Exception):
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


def _series_from_records(records: list[dict[str, Any]]) -> pd.Series:
    if not records:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex([pd.Timestamp(item["timestamp"]) for item in records])
    values = [float(item["value"]) for item in records]
    return pd.Series(values, index=index, dtype=float).sort_index()


# ---------------------------------------------------------------------------
# evolve / control.
# ---------------------------------------------------------------------------


def _generation_payload(record: ga23.GenerationRecord) -> dict[str, Any]:
    return {
        "generation": record.generation,
        "best_fitness": record.best_fitness,
        "mean_fitness": record.mean_fitness,
        "worst_fitness": record.worst_fitness,
        "best_genome": record.best_genome.to_dict(),
    }


def _stage_evolve(cache: MarketCache) -> None:
    fitness_cache: dict[tuple, fitness23.FitnessResult] = {}
    for seed in ga23.SEEDS:
        started = time.time()
        result = ga23.run_ga(seed, cache, fitness_cache=fitness_cache, progress=True)
        payload = {
            "seed": result.seed,
            "population_size": ga23.POPULATION_SIZE,
            "generations": ga23.GENERATIONS,
            "tournament_k": ga23.TOURNAMENT_K,
            "mutation_probability": ga23.MUTATION_PROBABILITY,
            "mutation_sigma_fraction": ga23.MUTATION_SIGMA_FRACTION,
            "elite_count": ga23.ELITE_COUNT,
            "n_evaluations": result.n_evaluations,
            "n_backtests_run": result.n_backtests_run,
            "best_genome": result.best_genome.to_dict(),
            "best_fitness": result.best_fitness,
            "final_population_kind_counts": result.final_population_kind_counts,
            "history": [_generation_payload(record) for record in result.history],
            "wall_seconds": time.time() - started,
        }
        _save_json(RESULTS_DIR / f"ga_seed{seed}.json", payload)
        print(f"evolve: seed={seed} done best_fitness={result.best_fitness:.5f} backtests_run={result.n_backtests_run}/{result.n_evaluations} ({payload['wall_seconds']:.1f}s)")
    print(f"evolve: fitness cache size={len(fitness_cache)} (unique genomes ever backtested across all 5 seeds)")


def _stage_control(cache: MarketCache) -> None:
    fitness_cache: dict[tuple, fitness23.FitnessResult] = {}
    for seed in random_search23.SEEDS:
        started = time.time()
        result = random_search23.run_random_search(seed, cache, fitness_cache=fitness_cache, progress=True)
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
# gates: final-candidate selection + the ONE-TIME OOS evaluation + K1-K6.
# ---------------------------------------------------------------------------


def select_final_candidate(seeds: Sequence[int], best_genomes: Sequence[Genome], best_fitnesses: Sequence[float]) -> tuple[Genome, int, float]:
    """Task instruction: '시드 5개... 5회 중 4+ 재현되는 개선만 인정'. Operationalized exactly
    like research.wave21_ga.run_wave21.select_final_candidate: the MEDIAN (not max) of the 5
    GA seeds' own best-fitness genomes -- a single lucky seed's outlier jackpot cannot become
    the final candidate this way. Requires an ODD seed count (this wave's own 5) for an
    unambiguous median."""
    n = len(seeds)
    if not (len(best_genomes) == len(best_fitnesses) == n) or n == 0:
        raise ValueError("select_final_candidate: seeds/genomes/fitnesses must be equal-length and non-empty")
    if n % 2 == 0:
        raise ValueError(f"select_final_candidate: requires an odd seed count for an unambiguous median, got {n}")
    order = sorted(range(n), key=lambda index: best_fitnesses[index])
    median_position = order[n // 2]
    return best_genomes[median_position], seeds[median_position], best_fitnesses[median_position]


def _load_i5_oos_fitness() -> tuple[float | None, pd.Series]:
    if not I5_RESULTS_PATH.exists():
        raise Wave23Error(f"{I5_RESULTS_PATH} not found -- research/wave18_idle must have run (I5 is this wave's own baseline, read-only)")
    payload = _load_json(I5_RESULTS_PATH)
    equity = _series_from_records(payload["equity"])
    i5_oos_equity = equity[equity.index > engine23.OOS_SPLIT]
    i5_oos_result = fitness23.compute_fitness(i5_oos_equity) if len(i5_oos_equity) > fitness23.ROLLING_WINDOW_DAYS else None
    return (i5_oos_result.fitness if i5_oos_result is not None else None), equity


def _stage_gates(cache: MarketCache) -> None:
    ga_payloads = [_load_json(RESULTS_DIR / f"ga_seed{seed}.json") for seed in ga23.SEEDS]
    random_payloads = [_load_json(RESULTS_DIR / f"random_seed{seed}.json") for seed in random_search23.SEEDS]

    ga_best_by_seed = [float(payload["best_fitness"]) for payload in ga_payloads]
    random_best_by_seed = [float(payload["best_fitness"]) for payload in random_payloads]
    k1 = gates23.gate_k1_ga_beats_random(ga_best_by_seed, random_best_by_seed)
    print(f"gates: K1 (GA vs random, {k1['n_wins']}/{k1['n_seeds']} seeds) -> {k1['status']}")

    seeds = [int(payload["seed"]) for payload in ga_payloads]
    best_genomes = [from_dict(payload["best_genome"]) for payload in ga_payloads]
    final_genome, source_seed, source_fitness = select_final_candidate(seeds, best_genomes, ga_best_by_seed)
    print(f"gates: final candidate selected from seed={source_seed} (median of 5, IS fitness={source_fitness:.5f}) -> {final_genome.to_dict()}")

    # --- OOS seal opens exactly once, right here. ---
    final = fitness23.final_evaluation(final_genome, cache)
    _, signed_full = engine23.run_backtest_with_weights(final_genome, cache, engine23.MODE_OOS_FINAL, stress=False)

    i5_oos_fitness, i5_equity = _load_i5_oos_fitness()
    final_oos_fitness = final.oos_fitness.fitness if final.oos_fitness is not None else None
    k2 = gates23.gate_k2_beats_i5_oos(final_oos_fitness, i5_oos_fitness)
    print(f"gates: K2 (final OOS fitness {final_oos_fitness} vs I5 OOS fitness {i5_oos_fitness}) -> {k2['status']}")

    k3 = gates23.gate_k3_dsr(final.full_equity)
    print(f"gates: K3 (DSR trials={gates23.CUMULATIVE_TRIALS}, score={k3.get('score')}) -> {k3['status']}")

    k4 = gates23.gate_k4_ruin_defense(final.full_equity)
    print(f"gates: K4 (ruin defense) -> {k4['status']}")

    k5 = gates23.gate_k5_executability(final_genome, signed_full, final.full_equity, final.stress_equity)
    print(f"gates: K5 (executability) -> {k5['status']}")

    k6 = gates23.gate_k6_paper_reproducibility(final_genome)
    print(f"gates: K6 (paper reproducibility) -> {k6['status']} reasons={k6['reasons']}")

    report = gates23.evaluate_all_gates(k1, k2, k3, k4, k5, k6)
    print(f"gates: overall={report.overall} promoted={report.promoted} failure_reasons={list(report.failure_reasons)}")

    # Strategy-kind distribution across every GA seed's FINAL population -- "어느 전략을 선호했는지".
    kind_distribution: dict[str, int] = {}
    for payload in ga_payloads:
        for kind, count in payload.get("final_population_kind_counts", {}).items():
            kind_distribution[kind] = kind_distribution.get(kind, 0) + int(count)

    payload = {
        "final_genome": final_genome.to_dict(),
        "source_seed": source_seed,
        "source_is_fitness": source_fitness,
        "ga_best_by_seed": dict(zip((str(s) for s in seeds), ga_best_by_seed)),
        "random_best_by_seed": dict(zip((str(s) for s in (p["seed"] for p in random_payloads)), random_best_by_seed)),
        "strategy_kind_distribution_final_populations": kind_distribution,
        "full_equity": _series_payload(final.full_equity),
        "is_equity": _series_payload(final.is_equity),
        "oos_equity": _series_payload(final.oos_equity),
        "stress_equity": _series_payload(final.stress_equity),
        "full_period_cagr": final.full_period_cagr,
        "mdd_full": final.mdd_full,
        "is_fitness": asdict(final.is_fitness),
        "oos_fitness": asdict(final.oos_fitness) if final.oos_fitness is not None else None,
        "i5_oos_fitness": i5_oos_fitness,
        "gates": gates23.gate_report_payload(report),
    }
    _save_json(RESULTS_DIR / "final_candidate.json", payload)
    print(f"gates: wrote {RESULTS_DIR / 'final_candidate.json'}")


def _stage_report() -> None:
    write_wave23_report(RESULTS_DIR, REPORT_DIR, REGISTRY_PATH)
    print(f"report: wrote {REPORT_DIR / 'wave23_report.md'} and {REGISTRY_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave-23 GA re-search pipeline (short-term profit maximization)")
    parser.add_argument("--stage", required=True, type=Stage, choices=tuple(Stage))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.stage:
            case Stage.EVOLVE:
                _stage_evolve(engine23.build_market_cache())
            case Stage.CONTROL:
                _stage_control(engine23.build_market_cache())
            case Stage.GATES:
                _stage_gates(engine23.build_market_cache())
            case Stage.REPORT:
                _stage_report()
            case Stage.ALL:
                cache = engine23.build_market_cache()
                _stage_evolve(cache)
                _stage_control(cache)
                _stage_gates(cache)
                _stage_report()
            case unreachable:
                assert_never(unreachable)
    except (FileNotFoundError, Wave23Error, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
