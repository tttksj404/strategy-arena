# Wave-30 search algorithms. Three of them, all sharing one evaluation cache and one budget
# accounting so that P1's "same budget" comparison is exact rather than approximate.
#
# ---------------------------------------------------------------------------------------
# Why NOT another GA (the point of this wave)
# ---------------------------------------------------------------------------------------
# wave21/23/27/28 ran a single-objective GA and wave24 ran GP; wave24 concluded the search
# METHODOLOGY was exhausted. All five collapsed performance into ONE scalar. wave23 documents
# where that leads: given `top-quartile 60d window mean - 3*P(window loss>20%)`, evolution
# converged on all-in high-volatility momentum with 91.5% drawdown. That is not a badly tuned
# penalty, it is what scalarisation does -- a single number cannot express "and also don't
# die", because any finite penalty is purchasable with enough upside.
#
# MAP-Elites removes the scalar. It keeps a separate champion per behaviour cell, so a
# 2x-leverage/8%-drawdown solution is never in competition with a 20x/95%-drawdown one; they
# occupy different cells and both survive to the end. The output is not "the best genome" but
# an ILLUMINATED MAP of what return is reachable at each leverage/drawdown/frequency, which is
# the literal question this wave was asked.
#
# NSGA-II keeps three objectives separate and returns the non-dominated front, making the
# return/ruin/drawdown trade-off explicit instead of hiding it inside a weight.
#
# Neither algorithm has been used anywhere in this repository before.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Callable, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import MarketCache
from research.wave30_qd.fitness30 import GRID_SHAPE, Evaluation, evaluate
from research.wave30_qd.genome30 import Genome, InvalidGenomeError, crossover, mutate, random_genome

MAP_ELITES_INIT: Final = 300
MAP_ELITES_ITERATIONS: Final = 3_700
NSGA_POPULATION: Final = 120
NSGA_GENERATIONS: Final = 30
RANDOM_BUDGET: Final = MAP_ELITES_INIT + MAP_ELITES_ITERATIONS + NSGA_POPULATION * NSGA_GENERATIONS


class Evaluator:
    """Shared, memoised genome -> Evaluation. Counts only DISTINCT evaluations against the
    budget so a cache hit cannot let one algorithm quietly buy extra search."""

    def __init__(self, cache: MarketCache, seed: int, evaluate_fn=None) -> None:
        self.cache = cache
        self.rng = np.random.default_rng(seed)
        self._memo: dict[tuple, Evaluation] = {}
        self.n_evaluations = 0
        # wave31 swaps in its own sprint objective here. The search algorithms themselves are
        # objective-agnostic -- they only read .fitness / .descriptor / .objectives -- so the
        # injection point is deliberately this one function and nothing else.
        self._evaluate_fn = evaluate_fn if evaluate_fn is not None else evaluate

    def __call__(self, genome: Genome) -> Evaluation:
        key = genome.key()
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        result = self._evaluate_fn(self.cache, genome, self.rng)
        self._memo[key] = result
        self.n_evaluations += 1
        return result


@dataclass
class Archive:
    """MAP-Elites archive: at most one elite per behaviour cell, replaced only by a strictly
    better fitness in the SAME cell."""

    cells: dict[tuple[int, int, int], Evaluation] = field(default_factory=dict)

    def consider(self, evaluation: Evaluation) -> bool:
        cell = evaluation.descriptor
        incumbent = self.cells.get(cell)
        if incumbent is None or evaluation.fitness > incumbent.fitness:
            self.cells[cell] = evaluation
            return True
        return False

    @property
    def coverage(self) -> int:
        return len(self.cells)

    @property
    def best(self) -> Evaluation | None:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda item: item.fitness)

    @property
    def qd_score(self) -> float:
        """Sum of positive elite fitness across cells -- the standard quality-diversity
        scalar. Reported alongside coverage because coverage alone rewards filling cells with
        junk, and best-fitness alone ignores the map."""
        return float(sum(max(0.0, item.fitness) for item in self.cells.values()))

    def elites(self) -> list[Evaluation]:
        return sorted(self.cells.values(), key=lambda item: item.fitness, reverse=True)


def _safe_variation(
    parents: list[Genome], rng: np.random.Generator, attempts: int = 25
) -> Genome | None:
    """Produce one feasible child. Infeasible draws are DISCARDED, never repaired, so the
    infeasible region (stop outside the liquidation band) stays genuinely unexplored."""
    for _ in range(attempts):
        try:
            if len(parents) >= 2 and rng.random() < 0.5:
                left, right = rng.choice(len(parents), size=2, replace=False)
                child = crossover(parents[int(left)], parents[int(right)], rng)
                return mutate(child, rng, rate=0.15)
            base = parents[int(rng.integers(len(parents)))]
            return mutate(base, rng, rate=0.25)
        except InvalidGenomeError:
            continue
    return None


def run_map_elites(
    evaluator: Evaluator,
    rng: np.random.Generator,
    n_init: int = MAP_ELITES_INIT,
    n_iterations: int = MAP_ELITES_ITERATIONS,
    progress: Callable[[int, Archive], None] | None = None,
) -> Archive:
    archive = Archive()
    for _ in range(n_init):
        archive.consider(evaluator(random_genome(rng)))

    for step in range(n_iterations):
        elites = list(archive.cells.values())
        if not elites:
            archive.consider(evaluator(random_genome(rng)))
            continue
        # Uniform selection over OCCUPIED CELLS, not over fitness. This is what makes the
        # algorithm illuminate rather than optimise: a lonely elite in a low-leverage cell is
        # sampled as often as the global best, so poor-but-distinct regions keep improving.
        chosen = [elites[int(rng.integers(len(elites)))].genome for _ in range(2)]
        child = _safe_variation(chosen, rng)
        if child is None:
            continue
        archive.consider(evaluator(child))
        if progress is not None and (step + 1) % 500 == 0:
            progress(step + 1, archive)
    return archive


# ---------------------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------------------


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Standard Deb et al. non-dominated sorting. `objectives` is (n, m), all MINIMISED."""
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]
    for i in range(n):
        less_equal = np.all(objectives[i] <= objectives, axis=1)
        strictly_less = np.any(objectives[i] < objectives, axis=1)
        i_dominates = less_equal & strictly_less
        i_dominates[i] = False
        dominated[i] = list(np.flatnonzero(i_dominates))

        others_le = np.all(objectives <= objectives[i], axis=1)
        others_lt = np.any(objectives < objectives[i], axis=1)
        dominates_i = others_le & others_lt
        dominates_i[i] = False
        domination_count[i] = int(dominates_i.sum())
        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        nxt: list[int] = []
        for i in fronts[current]:
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        current += 1
        fronts.append(nxt)
    return [front for front in fronts if front]


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    n, m = objectives.shape
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for axis in range(m):
        order = np.argsort(objectives[:, axis])
        values = objectives[order, axis]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        spread = values[-1] - values[0]
        if spread <= 0:
            continue
        distance[order[1:-1]] += (values[2:] - values[:-2]) / spread
    return distance


def run_nsga2(
    evaluator: Evaluator,
    rng: np.random.Generator,
    population_size: int = NSGA_POPULATION,
    generations: int = NSGA_GENERATIONS,
) -> list[Evaluation]:
    population = [evaluator(random_genome(rng)) for _ in range(population_size)]

    for _ in range(generations):
        objectives = np.array([item.objectives for item in population], dtype=float)
        fronts = fast_non_dominated_sort(objectives)
        rank = np.zeros(len(population), dtype=int)
        for level, front in enumerate(fronts):
            for index in front:
                rank[index] = level
        distance = np.zeros(len(population))
        for front in fronts:
            distance[front] = crowding_distance(objectives[front])

        def binary_tournament() -> Genome:
            a, b = (int(x) for x in rng.integers(0, len(population), size=2))
            if rank[a] != rank[b]:
                return population[a if rank[a] < rank[b] else b].genome
            return population[a if distance[a] > distance[b] else b].genome

        children: list[Evaluation] = []
        while len(children) < population_size:
            parents = [binary_tournament(), binary_tournament()]
            child = _safe_variation(parents, rng)
            if child is None:
                continue
            children.append(evaluator(child))

        combined = population + children
        objectives = np.array([item.objectives for item in combined], dtype=float)
        fronts = fast_non_dominated_sort(objectives)
        survivors: list[int] = []
        for front in fronts:
            if len(survivors) + len(front) <= population_size:
                survivors.extend(front)
                continue
            remaining = population_size - len(survivors)
            front_distance = crowding_distance(objectives[front])
            order = np.argsort(-front_distance)
            survivors.extend([front[int(i)] for i in order[:remaining]])
            break
        population = [combined[i] for i in survivors]

    objectives = np.array([item.objectives for item in population], dtype=float)
    front = fast_non_dominated_sort(objectives)[0]
    return [population[i] for i in front]


def run_random_search(
    evaluator: Evaluator, rng: np.random.Generator, budget: int = RANDOM_BUDGET
) -> Archive:
    """Matched-budget control. Fills its OWN archive so P1 can compare both the best fitness
    and the quality-diversity coverage -- wave21/23/24 only ever compared best fitness, which
    cannot tell whether an algorithm's advantage is optimisation or exploration."""
    archive = Archive()
    for _ in range(budget):
        archive.consider(evaluator(random_genome(rng)))
    return archive
