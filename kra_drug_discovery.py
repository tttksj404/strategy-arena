from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


MARKET_PARITY_TOLERANCE_PP: Final = 0.5
MINIMUM_V4_LIFT_PP: Final = 5.0
MINIMUM_MARKET_ADVANTAGE_PP: Final = 5.0


@dataclass(frozen=True, slots=True)
class Genome:
    weights: tuple[float, ...]
    market_weight: float


@dataclass(frozen=True, slots=True)
class AssayScore:
    genome: Genome
    discovery_market_gaps_pp: tuple[float, ...]
    discovery_v4_lifts_pp: tuple[float, ...]
    confirmation_market_gap_pp: float
    confirmation_v4_lift_pp: float


def _fingerprint(genome: Genome) -> tuple[float, ...]:
    return tuple(round(value, 8) for value in (*genome.weights, genome.market_weight))


def _normalized_genome(
    raw: np.ndarray,
    market_weight: float,
) -> Genome:
    nonnegative = np.clip(raw, 0.0, None)
    total = float(nonnegative.sum())
    normalized = (
        nonnegative / total
        if total > 0.0
        else np.full(len(nonnegative), 1.0 / len(nonnegative))
    )
    scaled = normalized * (1.0 - market_weight)
    return Genome(tuple(float(value) for value in scaled), float(market_weight))


def generate_wide_library(
    base_count: int,
    requested: int,
    seed: int,
    maximum_market_weight: float,
    parents: tuple[Genome, ...],
) -> tuple[Genome, ...]:
    rng = np.random.default_rng(seed)
    library: list[Genome] = []
    seen: set[tuple[float, ...]] = set()

    def add(genome: Genome) -> None:
        fingerprint = _fingerprint(genome)
        if fingerprint not in seen:
            seen.add(fingerprint)
            library.append(genome)

    for index in range(base_count):
        one_hot = np.zeros(base_count, dtype=float)
        one_hot[index] = 1.0
        add(_normalized_genome(one_hot, 0.0))
    add(_normalized_genome(np.ones(base_count, dtype=float), 0.0))
    for parent in parents:
        add(parent)

    target = max(requested, base_count + 1)
    while len(library) < target:
        market_weight = float(rng.uniform(0.0, maximum_market_weight))
        route = int(rng.integers(0, 4))
        match route:
            case 0:
                size = int(rng.integers(1, min(base_count, 6) + 1))
                selected = rng.choice(base_count, size=size, replace=False)
                raw = np.zeros(base_count, dtype=float)
                raw[selected] = rng.dirichlet(np.full(size, 0.7))
            case 1:
                raw = rng.dirichlet(np.full(base_count, 0.35))
            case 2 if parents:
                parent = parents[int(rng.integers(0, len(parents)))]
                raw = np.asarray(parent.weights, dtype=float)
                mutation = rng.normal(0.0, 0.12, base_count)
                mask = rng.random(base_count) < 0.45
                raw = raw + mutation * mask
            case 3 if len(parents) >= 2:
                left = parents[int(rng.integers(0, len(parents)))]
                right = parents[int(rng.integers(0, len(parents)))]
                mix = float(rng.uniform(0.15, 0.85))
                raw = mix * np.asarray(left.weights) + (1.0 - mix) * np.asarray(
                    right.weights
                )
            case _:
                raw = rng.random(base_count)
        add(_normalized_genome(raw, market_weight))
    return tuple(library)


def generate_hybrid_library(
    base_count: int,
    requested: int,
    seed: int,
    maximum_market_weight: float,
    parents: tuple[Genome, ...],
) -> tuple[Genome, ...]:
    rng = np.random.default_rng(seed)
    library: list[Genome] = []
    seen: set[tuple[float, ...]] = set()

    def add(genome: Genome) -> None:
        fingerprint = _fingerprint(genome)
        if fingerprint not in seen and len(library) < max(requested, base_count + 1):
            seen.add(fingerprint)
            library.append(genome)

    for index in range(base_count):
        one_hot = np.zeros(base_count, dtype=float)
        one_hot[index] = 1.0
        add(_normalized_genome(one_hot, 0.0))
    add(_normalized_genome(np.ones(base_count, dtype=float), 0.0))
    for parent in parents:
        add(parent)

    market_grid = (0.0, maximum_market_weight * 0.5, maximum_market_weight)
    mix_grid = (0.1, 0.25, 0.5, 0.75, 0.9)
    for left in range(base_count):
        for right in range(left + 1, base_count):
            for mix in mix_grid:
                raw = np.zeros(base_count, dtype=float)
                raw[left] = mix
                raw[right] = 1.0 - mix
                for market_weight in market_grid:
                    add(_normalized_genome(raw, market_weight))

    for parent in parents:
        raw_parent = np.asarray(parent.weights, dtype=float)
        active = np.flatnonzero(raw_parent > 0.01)
        for index in active[:8]:
            for delta in (-0.10, -0.05, 0.05, 0.10):
                raw = raw_parent.copy()
                raw[index] = max(0.0, raw[index] + delta)
                add(_normalized_genome(raw, parent.market_weight))

    target = max(requested, base_count + 1)
    while len(library) < target:
        market_weight = float(rng.uniform(0.0, maximum_market_weight))
        route = int(rng.integers(0, 4))
        match route:
            case 0:
                size = int(rng.integers(1, min(base_count, 6) + 1))
                selected = rng.choice(base_count, size=size, replace=False)
                raw = np.zeros(base_count, dtype=float)
                raw[selected] = rng.dirichlet(np.full(size, 0.7))
            case 1:
                raw = rng.dirichlet(np.full(base_count, 0.35))
            case 2 if parents:
                parent = parents[int(rng.integers(0, len(parents)))]
                raw = np.asarray(parent.weights, dtype=float)
                mutation = rng.normal(0.0, 0.12, base_count)
                mask = rng.random(base_count) < 0.45
                raw = raw + mutation * mask
            case 3 if len(parents) >= 2:
                left = parents[int(rng.integers(0, len(parents)))]
                right = parents[int(rng.integers(0, len(parents)))]
                mix = float(rng.uniform(0.15, 0.85))
                raw = mix * np.asarray(left.weights) + (1.0 - mix) * np.asarray(
                    right.weights
                )
            case _:
                raw = rng.random(base_count)
        add(_normalized_genome(raw, market_weight))
    return tuple(library)


def select_frontier(
    scores: tuple[AssayScore, ...],
    beam_width: int,
) -> tuple[AssayScore, ...]:
    ranked = sorted(
        scores,
        key=lambda score: (
            min(score.discovery_market_gaps_pp),
            float(np.mean(score.discovery_market_gaps_pp)),
            min(score.discovery_v4_lifts_pp),
            float(np.mean(score.discovery_v4_lifts_pp)),
            -score.genome.market_weight,
        ),
        reverse=True,
    )
    return tuple(ranked[:beam_width])


def market_parity_pass(
    score: AssayScore,
    tolerance_pp: float = MARKET_PARITY_TOLERANCE_PP,
) -> bool:
    market_gaps = (
        *score.discovery_market_gaps_pp,
        score.confirmation_market_gap_pp,
    )
    v4_lifts = (*score.discovery_v4_lifts_pp, score.confirmation_v4_lift_pp)
    required_market_gap = max(MINIMUM_MARKET_ADVANTAGE_PP, -float(tolerance_pp))
    return bool(
        min(market_gaps) >= required_market_gap
        and min(v4_lifts) >= MINIMUM_V4_LIFT_PP
    )
