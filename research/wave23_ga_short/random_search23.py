# Wave-23 random-search control group (오염 차단 장치 2, 필수): the SAME number of evaluations
# as ga23.py (1,500 x 5 seeds = 7,500), drawn i.i.d. from the identical gene distribution
# genome23.random_genome uses to seed the GA's own generation 0. K1 (gates23.gate_k1_ga_beats_random)
# is the whole point of this module existing -- see research.wave21_ga.random_search's own
# module docstring for the full rationale (identical here, kind gene included).
#
# Seeds are DERIVED from, but distinct from, ga23.SEEDS -- both modules draw from
# np.random.default_rng(seed) independently.

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

from research.wave23_ga_short import ga23
from research.wave23_ga_short.engine23 import MarketCache
from research.wave23_ga_short.fitness23 import FitnessResult, evaluate_genome_cached
from research.wave23_ga_short.genome23 import Genome, random_genome

N_EVALUATIONS_PER_SEED: Final = ga23.EVALUATIONS_PER_SEED  # 1,500
SEED_OFFSET: Final = 10_000_000
SEEDS: Final[tuple[int, ...]] = tuple(seed + SEED_OFFSET for seed in ga23.SEEDS)  # (12026101..12026105)


@dataclass(frozen=True, slots=True)
class RandomSearchResult:
    seed: int
    best_genome: Genome
    best_fitness: float
    n_evaluations: int
    n_backtests_run: int
    fitness_history: tuple[float, ...]


def run_random_search(
    seed: int,
    cache: MarketCache,
    n_evaluations: int = N_EVALUATIONS_PER_SEED,
    fitness_cache: dict[tuple, FitnessResult] | None = None,
    progress: bool = True,
) -> RandomSearchResult:
    rng = np.random.default_rng(seed)
    fitness_cache = {} if fitness_cache is None else fitness_cache
    best_genome: Genome | None = None
    best_fitness = float("-inf")
    history: list[float] = []
    n_backtests_run = 0

    for draw_index in range(n_evaluations):
        candidate = random_genome(rng)
        result, was_cache_hit = evaluate_genome_cached(candidate, cache, fitness_cache)
        n_backtests_run += 0 if was_cache_hit else 1
        history.append(result.fitness)
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_genome = candidate
        if progress and (draw_index + 1) % 300 == 0:
            print(f"random23: seed={seed} {draw_index + 1}/{n_evaluations} best={best_fitness:.5f}")

    assert best_genome is not None
    return RandomSearchResult(
        seed=seed,
        best_genome=best_genome,
        best_fitness=best_fitness,
        n_evaluations=n_evaluations,
        n_backtests_run=n_backtests_run,
        fitness_history=tuple(history),
    )


__all__ = ["N_EVALUATIONS_PER_SEED", "SEED_OFFSET", "SEEDS", "RandomSearchResult", "run_random_search"]
