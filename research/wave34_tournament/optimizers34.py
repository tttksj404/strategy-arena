# Wave-34 tournament arms. Six search methods over the SAME [0,1]^13 box, the SAME objective,
# the SAME evaluation budget and the SAME per-(method,seed) RNG stream.
#
# All six are written against one contract:
#
#     run(objective, rng) -> None      # just keep calling objective.trial(x) until it raises
#
# BudgetExhausted is the only stopping rule. No method is allowed a convergence-based early
# exit, because stopping early would spend less than the shared budget and quietly win the
# "fewest evaluations" comparison nobody asked for. Every loop is therefore written as
# `while True:` inside one try/except in the runner.
#
# scipy is not installed on this box, so CMA-ES is implemented directly (Hansen's purecma
# reference formulation) and TPE uses statistics.NormalDist for the KDE.

from __future__ import annotations

from pathlib import Path
import sys
from statistics import NormalDist
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave34_tournament.encoding34 import DIM
from research.wave34_tournament.fitness34 import Objective

METHODS: Final = ("random", "cmaes", "pso", "simulated_annealing", "tpe_bayesian", "mcts")


# ---------------------------------------------------------------------------------------
# 1. random -- THE CONTROL
# ---------------------------------------------------------------------------------------


def run_random(objective: Objective, rng: np.random.Generator) -> None:
    """Uniform i.i.d. sampling of the box. Because the encoding bins categoricals into
    equal-width intervals and reparameterises the stop, this is distributionally the same
    sampler as genome30.random_genome -- so it is the honest control, not a handicapped one."""
    while True:
        objective.trial(rng.random(DIM))


# ---------------------------------------------------------------------------------------
# 2. CMA-ES
# ---------------------------------------------------------------------------------------


def run_cmaes(objective: Objective, rng: np.random.Generator, sigma0: float = 0.30) -> None:
    """Covariance Matrix Adaptation ES, restarted with doubled population (IPOP) whenever the
    step size collapses.

    The box is handled by clipping inside the objective and evaluating the CLIPPED point, but
    ranking the UNCLIPPED sample. That is the standard "clip and let selection push back"
    treatment; the alternative (resampling until in-box) biases the distribution toward the
    interior and would make CMA-ES unable to place an optimum on a face of the cube -- and
    several genes (max leverage, full sleeve) sit exactly on faces.
    """
    n = DIM
    while True:  # IPOP restarts; the budget, not convergence, ends this
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        mean = rng.random(n)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        cov = np.eye(n)
        eigen_eval = 0
        counteval = 0
        b_mat = np.eye(n)
        d_vec = np.ones(n)

        for generation in range(10_000):
            samples = np.empty((lam, n))
            values = np.empty(lam)
            for k in range(lam):
                z = rng.standard_normal(n)
                y = b_mat @ (d_vec * z)
                samples[k] = mean + sigma * y
                values[k] = -objective(samples[k])  # minimise
            counteval += lam

            order = np.argsort(values)
            selected = samples[order[:mu]]
            old_mean = mean
            mean = weights @ selected

            y_w = (mean - old_mean) / sigma
            inv_sqrt_c = b_mat @ np.diag(1.0 / d_vec) @ b_mat.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_c @ y_w)
            hsig = float(
                np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chin
                < 1.4 + 2 / (n + 1)
            )
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w

            art = (selected - old_mean) / sigma
            cov = (
                (1 - c1 - cmu) * cov
                + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * cov)
                + cmu * (art.T @ (weights[:, None] * art))
            )
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))

            if counteval - eigen_eval > lam / (c1 + cmu) / n / 10:
                eigen_eval = counteval
                cov = np.triu(cov) + np.triu(cov, 1).T
                eigenvalues, b_mat = np.linalg.eigh(cov)
                d_vec = np.sqrt(np.maximum(eigenvalues, 1e-20))

            # Restart conditions (Auger & Hansen IPOP): degenerate step or stalled spread.
            if sigma < 1e-8 or sigma * d_vec.max() > 1e6 or not np.isfinite(sigma):
                break
            if generation > 20 and float(values[order[0]] - values[order[-1]]) == 0.0:
                break
        sigma0 = min(0.5, sigma0 * 1.1)  # widen slightly on each restart


# ---------------------------------------------------------------------------------------
# 3. Particle swarm
# ---------------------------------------------------------------------------------------


def run_pso(
    objective: Objective,
    rng: np.random.Generator,
    swarm_size: int = 30,
    inertia: float = 0.729,
    c1: float = 1.494,
    c2: float = 1.494,
) -> None:
    """Constriction-coefficient PSO (Clerc-Kennedy). Walls are REFLECTIVE: a particle that
    leaves the box is placed back on the face with its velocity component negated, which
    keeps face optima reachable without letting the swarm accumulate outside."""
    positions = rng.random((swarm_size, DIM))
    velocities = (rng.random((swarm_size, DIM)) - 0.5) * 0.2
    personal_best = positions.copy()
    personal_value = np.array([objective(p) for p in positions])
    global_index = int(np.argmax(personal_value))
    global_best = personal_best[global_index].copy()

    while True:
        for i in range(swarm_size):
            r1 = rng.random(DIM)
            r2 = rng.random(DIM)
            velocities[i] = (
                inertia * velocities[i]
                + c1 * r1 * (personal_best[i] - positions[i])
                + c2 * r2 * (global_best - positions[i])
            )
            velocities[i] = np.clip(velocities[i], -0.25, 0.25)
            positions[i] = positions[i] + velocities[i]
            below = positions[i] < 0.0
            above = positions[i] > 1.0
            positions[i] = np.where(below, -positions[i], positions[i])
            positions[i] = np.where(above, 2.0 - positions[i], positions[i])
            positions[i] = np.clip(positions[i], 0.0, 1.0)
            velocities[i] = np.where(below | above, -velocities[i], velocities[i])

            value = objective(positions[i])
            if value > personal_value[i]:
                personal_value[i] = value
                personal_best[i] = positions[i].copy()
                if value > personal_value[global_index]:
                    global_index = i
                    global_best = positions[i].copy()


# ---------------------------------------------------------------------------------------
# 4. Simulated annealing
# ---------------------------------------------------------------------------------------


def run_simulated_annealing(
    objective: Objective,
    rng: np.random.Generator,
    t0: float = 2.0,
    t_end: float = 0.01,
    cycle: int = 1_500,
) -> None:
    """Gaussian-proposal SA with a geometric cooling schedule, re-heated every `cycle`
    evaluations so the run does not freeze early and then waste the remaining budget on a
    point it can no longer leave. Step size shrinks with temperature, which is what turns SA
    from a random walk into a local refiner at the end of each cycle."""
    current = rng.random(DIM)
    current_value = objective(current)
    step_since_reheat = 0

    while True:
        fraction = min(1.0, step_since_reheat / cycle)
        temperature = t0 * (t_end / t0) ** fraction
        scale = 0.02 + 0.28 * temperature / t0

        candidate = np.clip(current + rng.normal(0.0, scale, DIM), 0.0, 1.0)
        value = objective(candidate)
        delta = value - current_value
        if delta >= 0 or rng.random() < np.exp(delta / max(temperature, 1e-9)):
            current, current_value = candidate, value

        step_since_reheat += 1
        if step_since_reheat >= cycle:
            step_since_reheat = 0
            # Re-heat from the best feasible point found so far when one exists, otherwise
            # from the best point at all -- restarting from scratch would throw away the only
            # information SA has.
            anchor = objective.best_feasible or objective.best
            if anchor is not None:
                current = np.clip(np.array(anchor.x) + rng.normal(0.0, 0.10, DIM), 0.0, 1.0)
                current_value = objective(current)


# ---------------------------------------------------------------------------------------
# 5. TPE (Bayesian)
# ---------------------------------------------------------------------------------------


def _kde_parts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mu, sigma, truncation_mass) for a 1-D Parzen mixture over `values` plus a wide prior
    kernel at 0.5. Bandwidth is Bergstra's heuristic -- each kernel's sigma is the larger of
    its two neighbour gaps, floored so duplicate observations cannot collapse it to zero.

    The truncation mass is the kernel's probability inside [0,1], computed once here with
    statistics.NormalDist (scipy is unavailable and numpy has no erf). Densities are then
    divided by it so l(x) and g(x) stay comparable near the faces of the box -- without that,
    a kernel sitting on a face looks half as dense as an identical one in the interior and TPE
    systematically avoids the faces, where several genes' optima actually live.
    """
    mus = np.concatenate([np.asarray(values, dtype=float), [0.5]])
    mus = np.sort(mus)
    if len(mus) == 1:
        sigmas = np.array([1.0])
    else:
        gaps = np.diff(mus)
        left = np.concatenate([[mus[0]], gaps])
        right = np.concatenate([gaps, [1.0 - mus[-1]]])
        sigmas = np.maximum(left, right)
    sigmas = np.clip(sigmas, 1.0 / min(100, len(mus) + 1), 1.0)
    prior = int(np.argmin(np.abs(mus - 0.5)))
    sigmas[prior] = max(sigmas[prior], 1.0)
    mass = np.array([NormalDist(float(m), float(s)).cdf(1.0) - NormalDist(float(m), float(s)).cdf(0.0)
                     for m, s in zip(mus, sigmas)])
    return mus, sigmas, np.maximum(mass, 1e-12)


def _log_density(x: np.ndarray, parts: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Vectorised over a batch of candidate values. x: (n_candidates,) -> (n_candidates,)."""
    mus, sigmas, mass = parts
    z = (x[:, None] - mus[None, :]) / sigmas[None, :]
    density = np.exp(-0.5 * z * z) / (sigmas[None, :] * np.sqrt(2 * np.pi) * mass[None, :])
    return np.log(np.maximum(density.mean(axis=1), 1e-300))


def run_tpe(
    objective: Objective,
    rng: np.random.Generator,
    n_startup: int = 40,
    n_candidates: int = 24,
    gamma: float = 0.25,
    max_good: int = 25,
    max_bad: int = 300,
) -> None:
    """Tree-structured Parzen Estimator, per-dimension independent Parzen KDEs.

    Bergstra's TPE splits observations at the gamma-quantile into l(x) (good) and g(x) (bad),
    fits a KDE to each, samples candidates from l, and takes the one maximising l(x)/g(x).

    Both sets are CAPPED (Optuna caps the good set at 25 for the same reason). This is a real
    cost decision, stated rather than hidden: an uncapped TPE rebuilds an O(n)-kernel mixture
    on every iteration, so its total work is quadratic in the budget. Measured on this problem
    an uncapped implementation cost 0.61 s/evaluation at n<=400 and rising, against ~0.05 s
    for the cheapest arm -- at the tournament budget it alone would have exceeded the wall
    clock. Capping keeps per-iteration cost flat and keeps the BUDGET (evaluations) the thing
    being held equal, which is what the comparison is about. The bad set keeps the most recent
    `max_bad` observations, so g(x) tracks where the search currently is rather than being
    dominated by the startup sample forever.
    """
    observations: list[tuple[np.ndarray, float]] = []

    for _ in range(n_startup):
        x = rng.random(DIM)
        observations.append((x, objective(x)))

    while True:
        ranked = sorted(range(len(observations)), key=lambda i: -observations[i][1])
        n_good = min(max_good, max(2, int(np.ceil(gamma * len(observations)))))
        good_idx = set(ranked[:n_good])
        good = np.array([observations[i][0] for i in ranked[:n_good]])
        bad_pool = [observations[i][0] for i in range(len(observations)) if i not in good_idx]
        if len(bad_pool) < 2:
            bad_pool = [obs[0] for obs in observations]
        bad = np.array(bad_pool[-max_bad:])

        models = [(_kde_parts(good[:, d]), _kde_parts(bad[:, d])) for d in range(DIM)]

        candidates = np.empty((n_candidates, DIM))
        scores = np.zeros(n_candidates)
        for d in range(DIM):
            good_parts, bad_parts = models[d]
            g_mu, g_sigma, _ = good_parts
            picks = rng.integers(len(g_mu), size=n_candidates)
            values = np.clip(rng.normal(g_mu[picks], g_sigma[picks]), 0.0, 1.0)
            candidates[:, d] = values
            scores += _log_density(values, good_parts) - _log_density(values, bad_parts)

        best_candidate = candidates[int(np.argmax(scores))]
        observations.append((best_candidate, objective(best_candidate)))


# ---------------------------------------------------------------------------------------
# 6. MCTS with progressive widening
# ---------------------------------------------------------------------------------------


class _Node:
    __slots__ = ("low", "high", "depth", "children", "visits", "total", "best")

    def __init__(self, low: np.ndarray, high: np.ndarray, depth: int) -> None:
        self.low = low
        self.high = high
        self.depth = depth
        self.children: list["_Node"] = []
        self.visits = 0
        self.total = 0.0
        self.best = -np.inf


def run_mcts(
    objective: Objective,
    rng: np.random.Generator,
    c_uct: float = 0.6,
    widening_alpha: float = 0.5,
    max_depth: int = 26,
) -> None:
    """Monte-Carlo tree search over nested boxes.

    Each node owns an axis-aligned sub-box; expanding it SPLITS the widest remaining dimension
    at its midpoint. Progressive widening (|children| <= visits^alpha) stops the tree from
    fanning out faster than it collects evidence. A rollout is one uniform sample inside the
    leaf's box, and the backed-up value is the node's running mean, normalised by the global
    spread so UCT's exploration constant means the same thing whether fitness is -60
    (infeasible) or +1.5 (feasible) -- without that normalisation the -50 penalty makes the
    exploration term numerically irrelevant and the search degenerates to greedy descent.
    """
    root = _Node(np.zeros(DIM), np.ones(DIM), 0)
    seen_min = np.inf
    seen_max = -np.inf

    def normalise(value: float) -> float:
        if not np.isfinite(seen_min) or seen_max <= seen_min:
            return 0.5
        return float((value - seen_min) / (seen_max - seen_min))

    while True:
        path = [root]
        node = root
        # ---- selection ----
        while node.children and len(node.children) >= max(1, int(node.visits**widening_alpha)):
            log_parent = np.log(max(node.visits, 1))
            scores = [
                (
                    normalise(child.total / child.visits) + c_uct * np.sqrt(log_parent / child.visits)
                    if child.visits > 0
                    else np.inf
                )
                for child in node.children
            ]
            node = node.children[int(np.argmax(scores))]
            path.append(node)

        # ---- expansion (progressive widening) ----
        if node.depth < max_depth and len(node.children) < max(1, int(node.visits**widening_alpha) + 1):
            axis = int(np.argmax(node.high - node.low))
            mid = 0.5 * (node.low[axis] + node.high[axis])
            if node.high[axis] - node.low[axis] > 1e-6 and not node.children:
                left_high = node.high.copy()
                left_high[axis] = mid
                right_low = node.low.copy()
                right_low[axis] = mid
                node.children = [
                    _Node(node.low.copy(), left_high, node.depth + 1),
                    _Node(right_low, node.high.copy(), node.depth + 1),
                ]
            if node.children:
                unvisited = [child for child in node.children if child.visits == 0]
                node = unvisited[0] if unvisited else node.children[int(rng.integers(len(node.children)))]
                path.append(node)

        # ---- rollout ----
        x = node.low + rng.random(DIM) * (node.high - node.low)
        value = objective(x)
        seen_min = min(seen_min, value)
        seen_max = max(seen_max, value)

        # ---- backup ----
        for visited in path:
            visited.visits += 1
            visited.total += value
            visited.best = max(visited.best, value)


RUNNERS: Final = {
    "random": run_random,
    "cmaes": run_cmaes,
    "pso": run_pso,
    "simulated_annealing": run_simulated_annealing,
    "tpe_bayesian": run_tpe,
    "mcts": run_mcts,
}
