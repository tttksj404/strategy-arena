# Wave-21 random-search control group (SPEC.md 오염 차단 장치 3, 필수): the SAME number of
# evaluations as ga.py (1,500 x 5 seeds = 7,500), drawn i.i.d. from the identical gene
# distribution genome.random_genome uses to seed the GA's own generation 0. H1
# (gates21.gate_h1_ga_beats_random) is the whole point of this module existing: if the GA's
# best fitness does not beat this control's best fitness in at least 4 of 5 matched seeds, the
# evolutionary mechanism itself is not adding anything over chance, and SPEC.md requires that
# be reported as "GA 무의미" rather than quietly discarded.
#
# Seeds are DERIVED from, but distinct from, ga.SEEDS -- both modules draw from
# np.random.default_rng(seed), and giving them the identical integer seed would start the GA's
# generation-0 population and this module's own draws from the same RNG stream (harmless, but
# needlessly conflates "independent" control draws with the GA's own initial population).

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

from research.wave21_ga import ga
from research.wave21_ga.fitness import FitnessResult, MarketCache, evaluate_genome_cached
from research.wave21_ga.genome import Genome, random_genome

N_EVALUATIONS_PER_SEED: Final = ga.EVALUATIONS_PER_SEED  # 1,500 -- SPEC.md: "랜덤 대조군: 동일하게 1,500 평가 x 5시드"
SEED_OFFSET: Final = 10_000_000
SEEDS: Final[tuple[int, ...]] = tuple(seed + SEED_OFFSET for seed in ga.SEEDS)  # (10026001..10026005) -- independent of ga.SEEDS


@dataclass(frozen=True, slots=True)
class RandomSearchResult:
    seed: int
    best_genome: Genome
    best_fitness: float
    n_evaluations: int
    n_backtests_run: int
    fitness_history: tuple[float, ...]  # every draw's own fitness, in draw order


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
            print(f"random: seed={seed} {draw_index + 1}/{n_evaluations} best={best_fitness:.5f}")

    assert best_genome is not None  # n_evaluations >= 1 always in this wave's own usage
    return RandomSearchResult(
        seed=seed,
        best_genome=best_genome,
        best_fitness=best_fitness,
        n_evaluations=n_evaluations,
        n_backtests_run=n_backtests_run,
        fitness_history=tuple(history),
    )


__all__ = ["N_EVALUATIONS_PER_SEED", "SEED_OFFSET", "SEEDS", "RandomSearchResult", "run_random_search"]
