# Wave-24 random-tree control group (SPEC.md 과최적화 방어 2, 필수: "랜덤 트리 대조군: 동일
# 예산의 무작위 트리 생성·평가. GP <= 랜덤이면 'GP 무의미'.").
#
# The SAME number of evaluations as gp.py (200 x 30 x 5 seeds = 6,000/seed, 30,000 total), drawn
# i.i.d. from the IDENTICAL ramped-half-and-half distribution gp.run_gp uses to seed generation 0
# -- both modules draw from np.random.default_rng(seed) and generate trees via
# tree.ramped_half_and_half, so this control isolates exactly one thing: whether SELECTION
# (tournament + elitism, generation over generation) adds anything over drawing the same shapes
# blind. L1 (gates24.gate_l1_gp_beats_random) is the whole point of this module existing: if the
# GP's best fitness does not beat this control's best fitness in at least 4 of 5 matched seeds,
# evolution itself is not doing anything a fixed compute budget of random trees would not also
# do, and SPEC.md requires that be reported as "GP 무의미" rather than quietly discarded.
#
# Seeds are DERIVED from, but distinct from, gp.SEEDS -- giving them the identical integer seed
# would start gp.py's generation-0 population and this module's own draws from the same RNG
# stream (harmless, but needlessly conflates "independent" control draws with the GP's own
# initial population), matching research.wave21_ga.random_search's own precedent.

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

from research.wave24_gp import gp
from research.wave24_gp.fitness24 import FitnessResult, MarketCache, evaluate_tree_cached
from research.wave24_gp.tree import Node, ramped_half_and_half

N_EVALUATIONS_PER_SEED: Final = gp.EVALUATIONS_PER_SEED  # 6,000 -- SPEC.md: "동일 예산의 무작위 트리"
SEED_OFFSET: Final = 10_000_000
SEEDS: Final[tuple[int, ...]] = tuple(seed + SEED_OFFSET for seed in gp.SEEDS)  # (12026201..12026205)


@dataclass(frozen=True, slots=True)
class RandomTreeSearchResult:
    seed: int
    best_tree: Node
    best_fitness: float
    n_evaluations: int
    n_backtests_run: int
    fitness_history: tuple[float, ...]  # every draw's own fitness, in draw order


def run_random_search(
    seed: int,
    cache: MarketCache,
    n_evaluations: int = N_EVALUATIONS_PER_SEED,
    fitness_cache: dict[Node, FitnessResult] | None = None,
    progress: bool = True,
) -> RandomTreeSearchResult:
    rng = np.random.default_rng(seed)
    fitness_cache = {} if fitness_cache is None else fitness_cache
    best_tree: Node | None = None
    best_fitness = float("-inf")
    history: list[float] = []
    n_backtests_run = 0

    # Drawn in POPULATION_SIZE-sized ramped-half-and-half batches (not one gp.POPULATION_SIZE=200
    # draw at a time forever) purely so the ramp (depth 2..5, grow/full split) refreshes on the
    # same period gp.py's own generations do -- a single one-shot ramped_half_and_half(rng, 6000)
    # call would ramp depth across the WHOLE run monotonically instead of representing the full
    # depth/method spread throughout, which would bias this control away from being a fair
    # same-distribution comparison against every one of gp.py's 30 generations.
    batch_size = gp.POPULATION_SIZE
    draw_index = 0
    while draw_index < n_evaluations:
        batch = ramped_half_and_half(rng, min(batch_size, n_evaluations - draw_index))
        for candidate in batch:
            result, was_cache_hit = evaluate_tree_cached(candidate, cache, fitness_cache)
            n_backtests_run += 0 if was_cache_hit else 1
            history.append(result.fitness)
            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best_tree = candidate
            draw_index += 1
            if progress and draw_index % 600 == 0:
                print(f"random_trees: seed={seed} {draw_index}/{n_evaluations} best={best_fitness:.5f}")

    assert best_tree is not None  # n_evaluations >= 1 always in this wave's own usage
    return RandomTreeSearchResult(
        seed=seed,
        best_tree=best_tree,
        best_fitness=best_fitness,
        n_evaluations=n_evaluations,
        n_backtests_run=n_backtests_run,
        fitness_history=tuple(history),
    )


__all__ = ["N_EVALUATIONS_PER_SEED", "SEED_OFFSET", "SEEDS", "RandomTreeSearchResult", "run_random_search"]
