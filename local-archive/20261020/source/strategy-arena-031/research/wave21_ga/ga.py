# Wave-21 genetic algorithm (SPEC.md "GA 설정 (동결, 사후 조정 금지)").
#
# Population 60, generations 25 (= 1,500 evaluations/seed), tournament selection (k=3),
# uniform crossover, Gaussian mutation (sigma=10% of each gene's own range, probability=0.15),
# elitism=2. Five independent seeds (2026001-2026005) -- SPEC.md: "5회 모두에서 재현되는
# 개선만 인정(단일 시드 대박 무효)"; run_wave21.py's gates stage implements this by picking
# the MEDIAN (not the max) of the 5 seeds' own best genomes as the final candidate -- see that
# module's docstring for why.
#
# Every evaluation here goes through fitness.evaluate_genome_cached, which internally can only
# ever request mode=MODE_IS (fitness.evaluate_genome takes no mode argument at all) -- this
# module has no code path capable of touching OOS data, structurally.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave21_ga.fitness import FitnessResult, MarketCache, evaluate_genome_cached
from research.wave21_ga.genome import Genome, crossover, mutate, random_genome

POPULATION_SIZE: Final = 60
GENERATIONS: Final = 25
TOURNAMENT_K: Final = 3
MUTATION_PROBABILITY: Final = 0.15
MUTATION_SIGMA_FRACTION: Final = 0.10
ELITE_COUNT: Final = 2
EVALUATIONS_PER_SEED: Final = POPULATION_SIZE * GENERATIONS  # 1,500 -- SPEC.md's own frozen figure
SEEDS: Final[tuple[int, ...]] = (2026001, 2026002, 2026003, 2026004, 2026005)  # SPEC.md verbatim


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_genome: Genome


@dataclass(frozen=True, slots=True)
class GARunResult:
    seed: int
    history: tuple[GenerationRecord, ...]
    best_genome: Genome
    best_fitness: float
    n_evaluations: int  # POPULATION_SIZE * GENERATIONS, the GA's own logical budget (SPEC.md's "1,500 평가")
    n_backtests_run: int  # actual cache misses -- <= n_evaluations, the caching savings this run realized


def _tournament_select(population: list[Genome], fitnesses: list[float], rng: np.random.Generator) -> Genome:
    contender_indices = rng.integers(0, len(population), size=TOURNAMENT_K)
    winner_index = max(contender_indices, key=lambda index: fitnesses[index])
    return population[winner_index]


def run_ga(seed: int, cache: MarketCache, fitness_cache: dict[tuple, FitnessResult] | None = None, progress: bool = True) -> GARunResult:
    rng = np.random.default_rng(seed)
    fitness_cache = {} if fitness_cache is None else fitness_cache
    population = [random_genome(rng) for _ in range(POPULATION_SIZE)]

    history: list[GenerationRecord] = []
    best_genome_overall: Genome | None = None
    best_fitness_overall = float("-inf")
    n_backtests_run = 0

    for generation in range(GENERATIONS):
        results: list[FitnessResult] = []
        for individual in population:
            result, was_cache_hit = evaluate_genome_cached(individual, cache, fitness_cache)
            n_backtests_run += 0 if was_cache_hit else 1
            results.append(result)
        fitnesses = [result.fitness for result in results]
        order = sorted(range(len(population)), key=lambda index: fitnesses[index], reverse=True)
        best_index = order[0]

        if fitnesses[best_index] > best_fitness_overall:
            best_fitness_overall = fitnesses[best_index]
            best_genome_overall = population[best_index]

        history.append(
            GenerationRecord(
                generation=generation,
                best_fitness=float(fitnesses[best_index]),
                mean_fitness=float(np.mean(fitnesses)),
                worst_fitness=float(min(fitnesses)),
                best_genome=population[best_index],
            )
        )
        if progress:
            print(f"ga: seed={seed} gen={generation + 1}/{GENERATIONS} best={fitnesses[best_index]:.5f} mean={float(np.mean(fitnesses)):.5f}")

        elites = [population[index] for index in order[:ELITE_COUNT]]
        next_population: list[Genome] = list(elites)
        while len(next_population) < POPULATION_SIZE:
            parent_a = _tournament_select(population, fitnesses, rng)
            parent_b = _tournament_select(population, fitnesses, rng)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng, MUTATION_SIGMA_FRACTION, MUTATION_PROBABILITY)
            next_population.append(child)
        population = next_population

    assert best_genome_overall is not None  # GENERATIONS >= 1 always, so history is never empty
    return GARunResult(
        seed=seed,
        history=tuple(history),
        best_genome=best_genome_overall,
        best_fitness=best_fitness_overall,
        n_evaluations=EVALUATIONS_PER_SEED,
        n_backtests_run=n_backtests_run,
    )


__all__ = [
    "ELITE_COUNT",
    "EVALUATIONS_PER_SEED",
    "GENERATIONS",
    "MUTATION_PROBABILITY",
    "MUTATION_SIGMA_FRACTION",
    "POPULATION_SIZE",
    "SEEDS",
    "TOURNAMENT_K",
    "GARunResult",
    "GenerationRecord",
    "run_ga",
]
