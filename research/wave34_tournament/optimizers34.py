# Wave-34 optimisers. Six methods, one interface, one budget counter.
#
# All are hand-written in numpy rather than pulled from cma/pyswarms/optuna, for the same reason
# research/wave13_liquidity/costs_measured.py hand-rolls its isotonic fit: the tournament's whole
# point is that nothing differs between arms except the search rule, and vendored libraries bring
# their own initialisation, boundary handling, and restart heuristics that would silently become
# part of the comparison. Each implementation below is the textbook form, with every deviation
# noted in its docstring.
#
# Shared contract:
#   run_<method>(objective_fn, rng, budget) -> list[Evaluation]
# where objective_fn(vector) -> Evaluation and consumes exactly one budget unit per DISTINCT
# vector (the caller's Budget object enforces this).

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

from research.wave30_qd.fitness30 import Evaluation
from research.wave34_tournament.encoding34 import DIMENSIONS

ObjectiveFn = Callable[[np.ndarray], Evaluation]


class BudgetExhausted(RuntimeError):
    """Raised the moment a method tries to spend more than its allotted evaluations. Methods do
    not police their own budget -- this makes over-spend a crash rather than a silent advantage."""


@dataclass
class Budget:
    """Counts evaluations and memoises repeats.

    A cache hit does NOT consume budget, matching wave30/31/33's Evaluator: otherwise a method
    that happens to revisit points would appear to search more thoroughly than one that does not.
    """

    objective_fn: ObjectiveFn
    limit: int
    spent: int = 0
    history: list[Evaluation] = field(default_factory=list)
    _memo: dict[tuple, Evaluation] = field(default_factory=dict)

    def __call__(self, vector: np.ndarray) -> Evaluation:
        clipped = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
        key = tuple(np.round(clipped, 6))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        if self.spent >= self.limit:
            raise BudgetExhausted(f"budget {self.limit} exhausted")
        result = self.objective_fn(clipped)
        self._memo[key] = result
        self.spent += 1
        self.history.append(result)
        return result

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def best(self) -> Evaluation | None:
        return max(self.history, key=lambda item: item.fitness) if self.history else None


def _safe_loop(body: Callable[[], None]) -> None:
    """Run an optimiser body until its budget runs out. Every method ends by exhausting budget,
    so BudgetExhausted is the normal termination signal, not an error."""
    try:
        body()
    except BudgetExhausted:
        return


# ---------------------------------------------------------------------------------------
# 1. Random search (control)
# ---------------------------------------------------------------------------------------


def run_random(budget: Budget, rng: np.random.Generator) -> None:
    """Uniform over the SAME [0,1]^13 box every other method sees. Sampling the control from a
    different distribution (e.g. the old categorical sampler) would invalidate the comparison."""

    def body() -> None:
        while budget.remaining:
            budget(rng.random(DIMENSIONS))

    _safe_loop(body)


# ---------------------------------------------------------------------------------------
# 2. CMA-ES
# ---------------------------------------------------------------------------------------


def run_cmaes(budget: Budget, rng: np.random.Generator, sigma0: float = 0.30) -> None:
    """Covariance Matrix Adaptation Evolution Strategy, Hansen's (mu/mu_w, lambda) form.

    Standard textbook implementation: weighted recombination, cumulative step-size adaptation
    (p_sigma) and rank-mu + rank-one covariance update (p_c). Two documented deviations:
      * the mean is clipped into the box each generation (the domain is bounded, and reflecting
        or resampling would distort the step-size adaptation more than clipping does);
      * eigen-decomposition is refreshed every generation rather than on a lazy schedule -- at
        13 dimensions that costs nothing and removes a tuning knob from the comparison.
    """

    def body() -> None:
        n = DIMENSIONS
        lam = 4 + int(3 * np.log(n))  # 13 dims -> 11
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        mean = rng.random(n)
        sigma = sigma0
        covariance = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        generation = 0

        while budget.remaining:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            sqrt_c = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            inv_sqrt_c = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

            steps = rng.standard_normal((lam, n))
            offspring = mean + sigma * (steps @ sqrt_c.T)
            scored: list[tuple[float, np.ndarray]] = []
            for candidate in offspring:
                evaluation = budget(candidate)
                scored.append((evaluation.fitness, np.clip(candidate, 0.0, 1.0)))
            scored.sort(key=lambda item: -item[0])

            selected = np.array([vector for _score, vector in scored[:mu]])
            old_mean = mean.copy()
            mean = np.clip(weights @ selected, 0.0, 1.0)

            displacement = (mean - old_mean) / max(sigma, 1e-12)
            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (inv_sqrt_c @ displacement)
            generation += 1
            h_sigma = float(
                np.linalg.norm(p_sigma) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * generation)) / chi_n
                < 1.4 + 2.0 / (n + 1.0)
            )
            p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * displacement

            rank_mu = np.zeros((n, n))
            for weight, vector in zip(weights, selected):
                delta = (vector - old_mean) / max(sigma, 1e-12)
                rank_mu += weight * np.outer(delta, delta)
            correction = (1.0 - h_sigma) * c_c * (2.0 - c_c)
            covariance = (
                (1.0 - c_1 - c_mu) * covariance
                + c_1 * (np.outer(p_c, p_c) + correction * covariance)
                + c_mu * rank_mu
            )
            covariance = np.triu(covariance) + np.triu(covariance, 1).T  # keep it symmetric
            sigma *= float(np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1.0)))
            sigma = float(np.clip(sigma, 1e-4, 1.0))

    _safe_loop(body)


# ---------------------------------------------------------------------------------------
# 3. Particle Swarm Optimisation
# ---------------------------------------------------------------------------------------


def run_pso(budget: Budget, rng: np.random.Generator, n_particles: int = 24) -> None:
    """Canonical inertia-weight PSO (Shi & Eberhart). w decays 0.9 -> 0.4 across the budget,
    c1 = c2 = 1.49445. Velocities are clamped to +-0.25 of the box width and positions are
    reflected at the boundary rather than clipped, because a clipped particle keeps pushing
    outward and effectively dies on the wall."""

    def body() -> None:
        positions = rng.random((n_particles, DIMENSIONS))
        velocities = (rng.random((n_particles, DIMENSIONS)) - 0.5) * 0.2
        personal_best = positions.copy()
        personal_score = np.full(n_particles, -np.inf)
        global_best = positions[0].copy()
        global_score = -np.inf
        c1 = c2 = 1.49445
        v_max = 0.25
        total_iterations = max(1, budget.limit // n_particles)
        iteration = 0

        while budget.remaining:
            for index in range(n_particles):
                score = budget(positions[index]).fitness
                if score > personal_score[index]:
                    personal_score[index] = score
                    personal_best[index] = np.clip(positions[index], 0.0, 1.0)
                if score > global_score:
                    global_score = score
                    global_best = np.clip(positions[index], 0.0, 1.0)

            inertia = 0.9 - 0.5 * min(1.0, iteration / total_iterations)
            r1 = rng.random((n_particles, DIMENSIONS))
            r2 = rng.random((n_particles, DIMENSIONS))
            velocities = (
                inertia * velocities
                + c1 * r1 * (personal_best - positions)
                + c2 * r2 * (global_best - positions)
            )
            velocities = np.clip(velocities, -v_max, v_max)
            positions = positions + velocities
            # Reflect at the walls and reverse the offending velocity component.
            below, above = positions < 0.0, positions > 1.0
            positions = np.where(below, -positions, positions)
            positions = np.where(above, 2.0 - positions, positions)
            velocities = np.where(below | above, -velocities, velocities)
            positions = np.clip(positions, 0.0, 1.0)
            iteration += 1

    _safe_loop(body)


# ---------------------------------------------------------------------------------------
# 4. Simulated Annealing
# ---------------------------------------------------------------------------------------


def run_simulated_annealing(budget: Budget, rng: np.random.Generator) -> None:
    """Metropolis acceptance with a geometric cooling schedule.

    T0 is set from the observed fitness spread of a short random burn-in (5% of budget) instead
    of a hardcoded constant: this objective mixes ordinary log-growth values with a -100 penalty
    cliff, so any fixed temperature would either accept everything or nothing. Proposal is a
    Gaussian perturbation whose width shrinks with temperature, reflected at the walls.
    """

    def body() -> None:
        burn_in = max(10, budget.limit // 20)
        samples = [budget(rng.random(DIMENSIONS)) for _ in range(min(burn_in, budget.remaining))]
        scores = np.array([item.fitness for item in samples])
        temperature0 = float(max(np.std(scores), 1e-3))
        temperature_final = temperature0 * 1e-3

        current = np.clip(np.asarray(max(samples, key=lambda i: i.fitness).extras["vector"]), 0.0, 1.0)
        current_score = max(scores)
        best_score = current_score
        steps = max(1, budget.remaining)
        cooling = (temperature_final / temperature0) ** (1.0 / steps)
        temperature = temperature0

        while budget.remaining:
            width = 0.05 + 0.35 * (temperature / temperature0)
            proposal = current + rng.normal(0.0, width, DIMENSIONS)
            proposal = np.where(proposal < 0.0, -proposal, proposal)
            proposal = np.where(proposal > 1.0, 2.0 - proposal, proposal)
            proposal = np.clip(proposal, 0.0, 1.0)
            score = budget(proposal).fitness
            delta = score - current_score
            if delta >= 0.0 or rng.random() < np.exp(delta / max(temperature, 1e-12)):
                current, current_score = proposal, score
                best_score = max(best_score, score)
            temperature *= cooling

    _safe_loop(body)


# ---------------------------------------------------------------------------------------
# 5. Bayesian optimisation, TPE form
# ---------------------------------------------------------------------------------------


def run_tpe(budget: Budget, rng: np.random.Generator, gamma: float = 0.25, n_candidates: int = 64) -> None:
    """Tree-structured Parzen Estimator (Bergstra et al.), the algorithm behind Optuna/Hyperopt.

    TPE rather than GP regression on purpose: a Gaussian-process surrogate needs a kernel choice
    and a length-scale fit, and those become tuning knobs inside the comparison. TPE has one
    meaningful knob (gamma, the good/bad split quantile) and models each dimension with a
    univariate Parzen mixture, which is exactly what Optuna does by default.

    Per iteration: split observations at the gamma quantile, fit l(x) over the good set and g(x)
    over the bad set as Gaussian mixtures centred on the observations (bandwidth = max(spread/
    sqrt(k), 0.05), the standard heuristic), draw candidates from l, and evaluate the candidate
    maximising log l(x) - log g(x) -- the Expected-Improvement proxy.
    """

    def parzen_logpdf(x: np.ndarray, centres: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        # sum over dimensions of log( mean over centres of N(x; centre, bw) )
        diff = (x[:, None, :] - centres[None, :, :]) / bandwidth[None, None, :]
        component = -0.5 * diff**2 - np.log(bandwidth[None, None, :]) - 0.5 * np.log(2 * np.pi)
        return np.log(np.mean(np.exp(component), axis=1) + 1e-300).sum(axis=1)

    def body() -> None:
        n_startup = max(20, budget.limit // 20)
        observations: list[tuple[np.ndarray, float]] = []
        for _ in range(min(n_startup, budget.remaining)):
            vector = rng.random(DIMENSIONS)
            observations.append((vector, budget(vector).fitness))

        while budget.remaining:
            observations.sort(key=lambda item: -item[1])
            n_good = max(2, int(np.ceil(gamma * len(observations))))
            good = np.array([vector for vector, _score in observations[:n_good]])
            bad = np.array([vector for vector, _score in observations[n_good:]])
            if len(bad) < 2:
                bad = rng.random((2, DIMENSIONS))

            bandwidth_good = np.maximum(good.std(axis=0) / np.sqrt(len(good)), 0.05)
            bandwidth_bad = np.maximum(bad.std(axis=0) / np.sqrt(len(bad)), 0.05)

            # Draw candidates from l(x): pick a good observation, jitter it by its bandwidth.
            picks = good[rng.integers(0, len(good), size=n_candidates)]
            candidates = np.clip(picks + rng.normal(0.0, bandwidth_good, size=(n_candidates, DIMENSIONS)), 0.0, 1.0)
            score = parzen_logpdf(candidates, good, bandwidth_good) - parzen_logpdf(candidates, bad, bandwidth_bad)
            chosen = candidates[int(np.argmax(score))]
            observations.append((chosen, budget(chosen).fitness))

    _safe_loop(body)


# ---------------------------------------------------------------------------------------
# 6. Monte Carlo Tree Search
# ---------------------------------------------------------------------------------------


def run_mcts(budget: Budget, rng: np.random.Generator, branching: int = 4, exploration: float = 0.7) -> None:
    """MCTS over sequential coordinate refinement.

    MCTS is normally described for games, so the formulation matters: here a node is a PREFIX of
    decided coordinates, and a child chooses which of `branching` equal sub-intervals the next
    coordinate falls into. Depth 13 = a fully specified box of width 1/4 per dimension; the
    rollout samples uniformly inside that box and evaluates it. Selection uses UCT with the node
    value being the MEAN of rollouts beneath it, and backpropagation carries that mean upward.
    This is the standard "MCTS for black-box optimisation" reading of the algorithm, and it is
    what makes it a genuinely different search rule from the other five: it partitions the domain
    hierarchically instead of moving points around in it.
    """

    @dataclass
    class Node:
        depth: int
        low: np.ndarray
        high: np.ndarray
        visits: int = 0
        total: float = 0.0
        children: list["Node"] = field(default_factory=list)

        @property
        def mean(self) -> float:
            return self.total / self.visits if self.visits else 0.0

    def expand(node: Node) -> None:
        axis = node.depth
        edges = np.linspace(node.low[axis], node.high[axis], branching + 1)
        for index in range(branching):
            low = node.low.copy()
            high = node.high.copy()
            low[axis] = edges[index]
            high[axis] = edges[index + 1]
            node.children.append(Node(depth=node.depth + 1, low=low, high=high))

    def body() -> None:
        root = Node(depth=0, low=np.zeros(DIMENSIONS), high=np.ones(DIMENSIONS))
        while budget.remaining:
            node = root
            path = [node]
            while node.depth < DIMENSIONS:
                if not node.children:
                    expand(node)
                unvisited = [child for child in node.children if child.visits == 0]
                if unvisited:
                    node = unvisited[int(rng.integers(len(unvisited)))]
                else:
                    log_parent = np.log(max(node.visits, 1))
                    node = max(
                        node.children,
                        key=lambda child: child.mean + exploration * np.sqrt(log_parent / child.visits),
                    )
                path.append(node)
            vector = node.low + rng.random(DIMENSIONS) * (node.high - node.low)
            value = budget(vector).fitness
            for entry in path:
                entry.visits += 1
                entry.total += value

    _safe_loop(body)


METHODS: Final[dict[str, Callable[[Budget, np.random.Generator], None]]] = {
    "random": run_random,
    "cmaes": run_cmaes,
    "pso": run_pso,
    "simulated_annealing": run_simulated_annealing,
    "tpe_bayesian": run_tpe,
    "mcts": run_mcts,
}
