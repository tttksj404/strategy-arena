# Wave-24 genetic programming loop (SPEC.md "GP 설계": "개체수 200 x 세대 30 x 시드 5.
# 토너먼트 k=3, 서브트리 교차, 서브트리/포인트 변이.").
#
# Population 200, generations 30 (= 6,000 evaluations/seed), tournament selection (k=3),
# subtree crossover, subtree mutation, point mutation, elitism=2. Five independent seeds
# (2026201-2026205, task brief verbatim) -- SPEC.md 과최적화 방어 3: "5시드 재현성: 4/5 이상에서
# 유사 구조가 나와야 인정".
#
# SPEC.md pins the algorithm SHAPE (tournament k=3, subtree crossover, subtree/point mutation,
# elitism, population/generation/seed counts) but -- unlike research.wave21_ga.ga's frozen
# "sigma=10%, probability=0.15" -- leaves the RATE at which each reproduction operator fires
# unstated (a GP individual has no fixed "number of genes" the way a Genome does, so there is no
# single natural per-gene mutation-probability analogue). REPRODUCTION_RATES below therefore
# fixes standard Koza-style defaults (crossover-dominant, small mutation share) ONCE here, stated
# plainly so the choice is auditable rather than silently baked into ga-style per-call defaults.
#
# Every evaluation here goes through fitness24.evaluate_tree_cached, which internally can only
# ever request mode=MODE_IS (fitness24.evaluate_tree takes no mode argument at all) -- this
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

from research.wave24_gp.fitness24 import FitnessResult, MarketCache, evaluate_tree_cached
from research.wave24_gp.tree import (
    CONST_VALUES,
    FUNCTIONS_ARITY,
    FUNCTION_NAMES,
    MA_WINDOWS,
    MAX_DEPTH,
    TERMINAL_VARS,
    Node,
    all_nodes,
    depth_at_index,
    grow,
    node_count,
    ramped_half_and_half,
    random_terminal,
    replace_subtree,
    subtree_depth_budget_ok,
    validate_tree,
)

POPULATION_SIZE: Final = 200
GENERATIONS: Final = 30
TOURNAMENT_K: Final = 3
ELITE_COUNT: Final = 2
CROSSOVER_PROBABILITY: Final = 0.70
SUBTREE_MUTATION_PROBABILITY: Final = 0.20
POINT_MUTATION_PROBABILITY: Final = 0.10  # sums to 1.0 with the two above
CROSSOVER_MAX_ATTEMPTS: Final = 10  # rejection-sampling budget for a depth<=5-respecting splice point (see crossover's own docstring)
EVALUATIONS_PER_SEED: Final = POPULATION_SIZE * GENERATIONS  # 6,000 -- SPEC.md's own "200 x 30"
SEEDS: Final[tuple[int, ...]] = (2026201, 2026202, 2026203, 2026204, 2026205)  # task brief, verbatim

assert abs(CROSSOVER_PROBABILITY + SUBTREE_MUTATION_PROBABILITY + POINT_MUTATION_PROBABILITY - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Reproduction operators (subtree crossover, subtree mutation, point mutation).
# ---------------------------------------------------------------------------


def crossover(parent_a: Node, parent_b: Node, rng: np.random.Generator, max_depth: int = MAX_DEPTH, max_attempts: int = CROSSOVER_MAX_ATTEMPTS) -> Node:
    """Subtree crossover: splice a random subtree of parent_b into a random point of parent_a.
    Depth<=5 is a HARD structural cap (SPEC.md), so this uses rejection sampling over
    (index_a, index_b) pairs -- subtree_depth_budget_ok checks the resulting depth from the
    (ancestor-depth, replacement-depth) pair directly, no candidate tree is ever materialized
    until a valid pair is found. Falls back to returning parent_a UNCHANGED if no valid pair
    turns up within max_attempts (can only happen when every one of parent_b's subtrees is too
    deep for every one of parent_a's non-root splice points -- swapping at the root, index_a=0,
    always trivially fits since parent_b itself is already depth<=max_depth, so this is rare in
    practice and never produces an invalid tree even in the worst case)."""
    nodes_a = all_nodes(parent_a)
    nodes_b = all_nodes(parent_b)
    for _ in range(max_attempts):
        index_a = int(rng.integers(0, len(nodes_a)))
        index_b = int(rng.integers(0, len(nodes_b)))
        replacement = nodes_b[index_b]
        if subtree_depth_budget_ok(parent_a, replacement, index_a, max_depth):
            return replace_subtree(parent_a, index_a, replacement)
    return parent_a


def mutate_subtree(node: Node, rng: np.random.Generator, max_depth: int = MAX_DEPTH) -> Node:
    """Replace a random subtree with a freshly-grown random one, sized (via grow()'s own
    max_depth argument) to exactly fit whatever depth budget remains at that splice point -- no
    rejection sampling needed (a valid `node` already has depth <= max_depth everywhere, so the
    remaining budget at any index is always >= 0 by construction)."""
    nodes = all_nodes(node)
    index = int(rng.integers(0, len(nodes)))
    remaining_budget = max_depth - depth_at_index(node, index)
    replacement = grow(rng, remaining_budget)
    return replace_subtree(node, index, replacement)


def mutate_point(node: Node, rng: np.random.Generator) -> Node:
    """Replace a single node in place with a same-shape random alternative: a terminal swaps for
    ANY other terminal (var or const), a function swaps for another function of the SAME ARITY
    (so its existing children are kept unchanged, no regeneration needed) -- standard GP point
    mutation."""
    nodes = all_nodes(node)
    index = int(rng.integers(0, len(nodes)))
    target = nodes[index]
    if target.is_terminal:
        replacement = random_terminal(rng)
    else:
        same_arity = [name for name in FUNCTION_NAMES if FUNCTIONS_ARITY[name] == FUNCTIONS_ARITY[target.op]]
        new_op = str(rng.choice(same_arity))
        value = int(rng.choice(MA_WINDOWS)) if new_op == "ma" else None
        replacement = Node(op=new_op, children=target.children, value=value)
    return replace_subtree(node, index, replacement)


# ---------------------------------------------------------------------------
# Evolution loop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_tree: Node
    mean_node_count: float


@dataclass(frozen=True, slots=True)
class GPRunResult:
    seed: int
    history: tuple[GenerationRecord, ...]
    best_tree: Node
    best_fitness: float
    n_evaluations: int  # POPULATION_SIZE * GENERATIONS, this run's own logical budget (SPEC.md's "200x30")
    n_backtests_run: int  # actual cache misses -- <= n_evaluations, the caching savings this run realized


def _tournament_select(population: list[Node], fitnesses: list[float], rng: np.random.Generator) -> Node:
    contender_indices = rng.integers(0, len(population), size=TOURNAMENT_K)
    winner_index = max(contender_indices, key=lambda index: fitnesses[index])
    return population[winner_index]


def _reproduce_one(population: list[Node], fitnesses: list[float], rng: np.random.Generator) -> Node:
    draw = rng.uniform()
    if draw < CROSSOVER_PROBABILITY:
        parent_a = _tournament_select(population, fitnesses, rng)
        parent_b = _tournament_select(population, fitnesses, rng)
        child = crossover(parent_a, parent_b, rng)
    elif draw < CROSSOVER_PROBABILITY + SUBTREE_MUTATION_PROBABILITY:
        parent = _tournament_select(population, fitnesses, rng)
        child = mutate_subtree(parent, rng)
    else:
        parent = _tournament_select(population, fitnesses, rng)
        child = mutate_point(parent, rng)
    validate_tree(child)
    return child


def run_gp(seed: int, cache: MarketCache, fitness_cache: dict[Node, FitnessResult] | None = None, progress: bool = True) -> GPRunResult:
    rng = np.random.default_rng(seed)
    fitness_cache = {} if fitness_cache is None else fitness_cache
    population = ramped_half_and_half(rng, POPULATION_SIZE)

    history: list[GenerationRecord] = []
    best_tree_overall: Node | None = None
    best_fitness_overall = float("-inf")
    n_backtests_run = 0

    for generation in range(GENERATIONS):
        results: list[FitnessResult] = []
        for individual in population:
            result, was_cache_hit = evaluate_tree_cached(individual, cache, fitness_cache)
            n_backtests_run += 0 if was_cache_hit else 1
            results.append(result)
        fitnesses = [result.fitness for result in results]
        order = sorted(range(len(population)), key=lambda index: fitnesses[index], reverse=True)
        best_index = order[0]

        if fitnesses[best_index] > best_fitness_overall:
            best_fitness_overall = fitnesses[best_index]
            best_tree_overall = population[best_index]

        mean_nodes = float(np.mean([node_count(individual) for individual in population]))
        history.append(
            GenerationRecord(
                generation=generation,
                best_fitness=float(fitnesses[best_index]),
                mean_fitness=float(np.mean(fitnesses)),
                worst_fitness=float(min(fitnesses)),
                best_tree=population[best_index],
                mean_node_count=mean_nodes,
            )
        )
        if progress:
            print(
                f"gp: seed={seed} gen={generation + 1}/{GENERATIONS} best={fitnesses[best_index]:.5f} "
                f"mean={float(np.mean(fitnesses)):.5f} mean_nodes={mean_nodes:.1f} cache_size={len(fitness_cache)}"
            )

        elites = [population[index] for index in order[:ELITE_COUNT]]
        next_population: list[Node] = list(elites)
        while len(next_population) < POPULATION_SIZE:
            next_population.append(_reproduce_one(population, fitnesses, rng))
        population = next_population

    assert best_tree_overall is not None  # GENERATIONS >= 1 always, so history is never empty
    return GPRunResult(
        seed=seed,
        history=tuple(history),
        best_tree=best_tree_overall,
        best_fitness=best_fitness_overall,
        n_evaluations=EVALUATIONS_PER_SEED,
        n_backtests_run=n_backtests_run,
    )


__all__ = [
    "CONST_VALUES",  # re-exported for reporting24.py convenience
    "CROSSOVER_MAX_ATTEMPTS",
    "CROSSOVER_PROBABILITY",
    "ELITE_COUNT",
    "EVALUATIONS_PER_SEED",
    "GENERATIONS",
    "POINT_MUTATION_PROBABILITY",
    "POPULATION_SIZE",
    "SEEDS",
    "SUBTREE_MUTATION_PROBABILITY",
    "TOURNAMENT_K",
    "GPRunResult",
    "GenerationRecord",
    "crossover",
    "mutate_point",
    "mutate_subtree",
    "run_gp",
]
