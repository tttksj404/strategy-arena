# Wave-23 GA gene definitions (SPEC.md "유전자 (단기 특화, 동결)" -- frozen table, 8 axes).
#
# Unlike wave21_ga (which tunes 7 axes of a SINGLE fixed strategy structure -- I5's L4 carry),
# this wave evolves the STRATEGY ITSELF: strategy_kind is a gene. The other 7 axes
# (entry_z, holding_days, position_fraction, stop_loss_pct, take_profit_pct, universe_breadth,
# max_concurrent) apply uniformly to whichever strategy_kind a given individual carries --
# engine23.py's signal builders each reduce their own kind-specific raw signal to a common
# z-scored "how extreme is today's reading vs this symbol's own recent history" unit so a single
# entry_z threshold is meaningful across all 5 kinds (see engine23.py module docstring).
#
# Leverage: SPEC.md "레버리지는 1x 고정" is enforced STRUCTURALLY in engine23.py (every
# genome's per-position weight is normalized so position_fraction * max_concurrent can never
# exceed 1.0 gross, regardless of what values those two genes take) rather than by a post-hoc
# gate -- wave21_ga's own H4 gate caught exactly this failure mode post-hoc (top_k_pairs=3 x
# leg_fraction=0.5 => gross=3x) AFTER the GA had already spent its whole budget searching an
# infeasible corner of the space, and the manual top_k fix applied afterward (wave22_overfit's
# G1) produced a DIFFERENT genome whose own DSR was never actually gated -- see this wave's
# SPEC.md "오염 차단 4" and REGISTRY.md for why that mistake must not repeat. Normalizing DURING
# the search instead means every genome the GA/random-search ever evaluates is ALREADY
# 1x-feasible, so there is no "structurally infeasible but numerically attractive" corner for
# either search to waste budget exploring, and no separate post-hoc genome substitution is ever
# needed.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Gene bounds (SPEC.md table, byte-for-byte -- do not adjust post-hoc).
# ---------------------------------------------------------------------------

STRATEGY_KIND_CARRY: Final = "carry"
STRATEGY_KIND_MOMENTUM: Final = "momentum"
STRATEGY_KIND_BREAKOUT: Final = "breakout"
STRATEGY_KIND_FUNDING_SPIKE: Final = "funding_spike"
STRATEGY_KIND_CONVEX_DUAL: Final = "convex_dual"
STRATEGY_KIND_CHOICES: Final[tuple[str, ...]] = (
    STRATEGY_KIND_CARRY,
    STRATEGY_KIND_MOMENTUM,
    STRATEGY_KIND_BREAKOUT,
    STRATEGY_KIND_FUNDING_SPIKE,
    STRATEGY_KIND_CONVEX_DUAL,
)

# Long-only kinds never take the short side even when their z-score goes negative (matches
# wave1's carry_position convention of a strictly long-the-spread trade, and wave20 V2's
# tail-hunt convention of a strictly long directional bet on an extreme funding reading).
LONG_ONLY_KINDS: Final[frozenset[str]] = frozenset({STRATEGY_KIND_CARRY, STRATEGY_KIND_FUNDING_SPIKE})
DUAL_DIRECTION_KINDS: Final[frozenset[str]] = frozenset({STRATEGY_KIND_MOMENTUM, STRATEGY_KIND_BREAKOUT, STRATEGY_KIND_CONVEX_DUAL})

ENTRY_Z_MIN: Final = 0.5
ENTRY_Z_MAX: Final = 4.0
POSITION_FRACTION_MIN: Final = 0.10
POSITION_FRACTION_MAX: Final = 1.00

HOLDING_DAYS_CHOICES: Final[tuple[int, ...]] = (1, 2, 3, 5, 7, 14)  # SPEC.md "보유기간 <=14일 강제"
STOP_LOSS_PCT_CHOICES: Final[tuple[float | None, ...]] = (None, 0.05, 0.10, 0.20)
TAKE_PROFIT_PCT_CHOICES: Final[tuple[float | None, ...]] = (None, 0.10, 0.25, 0.50)
UNIVERSE_BREADTH_CHOICES: Final[tuple[int, ...]] = (30, 100, 200)
MAX_CONCURRENT_CHOICES: Final[tuple[int, ...]] = (1, 2, 3)

_CONTINUOUS_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "entry_z": (ENTRY_Z_MIN, ENTRY_Z_MAX),
    "position_fraction": (POSITION_FRACTION_MIN, POSITION_FRACTION_MAX),
}
_CATEGORICAL_CHOICES: Final[dict[str, tuple]] = {
    "strategy_kind": STRATEGY_KIND_CHOICES,
    "holding_days": HOLDING_DAYS_CHOICES,
    "stop_loss_pct": STOP_LOSS_PCT_CHOICES,
    "take_profit_pct": TAKE_PROFIT_PCT_CHOICES,
    "universe_breadth": UNIVERSE_BREADTH_CHOICES,
    "max_concurrent": MAX_CONCURRENT_CHOICES,
}
GENE_NAMES: Final[tuple[str, ...]] = (
    "strategy_kind",
    "entry_z",
    "holding_days",
    "position_fraction",
    "stop_loss_pct",
    "take_profit_pct",
    "universe_breadth",
    "max_concurrent",
)


@dataclass(frozen=True, slots=True)
class Genome:
    strategy_kind: str
    entry_z: float
    holding_days: int
    position_fraction: float
    stop_loss_pct: float | None
    take_profit_pct: float | None
    universe_breadth: int
    max_concurrent: int

    def __post_init__(self) -> None:
        # Fail closed on an out-of-registry genome -- same rationale as wave21_ga.genome.Genome:
        # a silently-clamped invalid genome would let a GA bug quietly explore outside the
        # frozen SPEC.md search space instead of surfacing the bug immediately.
        if self.strategy_kind not in STRATEGY_KIND_CHOICES:
            raise ValueError(f"strategy_kind {self.strategy_kind!r} not in {STRATEGY_KIND_CHOICES}")
        lo, hi = _CONTINUOUS_BOUNDS["entry_z"]
        if not (lo <= self.entry_z <= hi):
            raise ValueError(f"entry_z {self.entry_z} outside [{lo}, {hi}]")
        lo, hi = _CONTINUOUS_BOUNDS["position_fraction"]
        if not (lo <= self.position_fraction <= hi):
            raise ValueError(f"position_fraction {self.position_fraction} outside [{lo}, {hi}]")
        if self.holding_days not in HOLDING_DAYS_CHOICES:
            raise ValueError(f"holding_days {self.holding_days} not in {HOLDING_DAYS_CHOICES}")
        if self.stop_loss_pct not in STOP_LOSS_PCT_CHOICES:
            raise ValueError(f"stop_loss_pct {self.stop_loss_pct!r} not in {STOP_LOSS_PCT_CHOICES}")
        if self.take_profit_pct not in TAKE_PROFIT_PCT_CHOICES:
            raise ValueError(f"take_profit_pct {self.take_profit_pct!r} not in {TAKE_PROFIT_PCT_CHOICES}")
        if self.universe_breadth not in UNIVERSE_BREADTH_CHOICES:
            raise ValueError(f"universe_breadth {self.universe_breadth} not in {UNIVERSE_BREADTH_CHOICES}")
        if self.max_concurrent not in MAX_CONCURRENT_CHOICES:
            raise ValueError(f"max_concurrent {self.max_concurrent} not in {MAX_CONCURRENT_CHOICES}")

    @property
    def is_long_only(self) -> bool:
        return self.strategy_kind in LONG_ONLY_KINDS

    @property
    def normalized_weight(self) -> float:
        """The ACTUAL per-position weight the engine uses -- position_fraction clipped so that
        position_fraction * max_concurrent can never exceed 1.0 gross (SPEC.md '레버리지는 1x
        고정', enforced by construction; see module docstring)."""
        raw_gross = self.position_fraction * self.max_concurrent
        if raw_gross <= 1.0:
            return self.position_fraction
        return 1.0 / self.max_concurrent

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {name: getattr(self, name) for name in GENE_NAMES}


def from_dict(payload: dict[str, float | int | str | None]) -> Genome:
    def _opt_float(value: object) -> float | None:
        return None if value is None else float(value)

    return Genome(
        strategy_kind=str(payload["strategy_kind"]),
        entry_z=float(payload["entry_z"]),
        holding_days=int(payload["holding_days"]),
        position_fraction=float(payload["position_fraction"]),
        stop_loss_pct=_opt_float(payload["stop_loss_pct"]),
        take_profit_pct=_opt_float(payload["take_profit_pct"]),
        universe_breadth=int(payload["universe_breadth"]),
        max_concurrent=int(payload["max_concurrent"]),
    )


def random_genome(rng: np.random.Generator) -> Genome:
    """Uniform-random draw over the frozen gene ranges -- shared by ga23.py's generation-0
    population and random_search23.py's control draws (SPEC.md's controlled comparison
    requires both to start from the identical distribution)."""
    stop_choices = STOP_LOSS_PCT_CHOICES
    take_choices = TAKE_PROFIT_PCT_CHOICES
    return Genome(
        strategy_kind=str(rng.choice(STRATEGY_KIND_CHOICES)),
        entry_z=float(rng.uniform(*_CONTINUOUS_BOUNDS["entry_z"])),
        holding_days=int(rng.choice(HOLDING_DAYS_CHOICES)),
        position_fraction=float(rng.uniform(*_CONTINUOUS_BOUNDS["position_fraction"])),
        stop_loss_pct=stop_choices[int(rng.integers(0, len(stop_choices)))],
        take_profit_pct=take_choices[int(rng.integers(0, len(take_choices)))],
        universe_breadth=int(rng.choice(UNIVERSE_BREADTH_CHOICES)),
        max_concurrent=int(rng.choice(MAX_CONCURRENT_CHOICES)),
    )


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def mutate(genome: Genome, rng: np.random.Generator, sigma_fraction: float = 0.10, probability: float = 0.15) -> Genome:
    """Per-gene independent mutation (SPEC.md GA 설정 "변이 p=0.15"): each of the 8 genes
    mutates independently with probability `probability`. A continuous gene gets a Gaussian
    perturbation with sigma = 10% of ITS OWN range, clipped back into bounds (matches
    wave21_ga.genome.mutate's own convention); a categorical gene gets a fresh uniform
    resample from its choice list."""
    values = genome.to_dict()
    for name in GENE_NAMES:
        if rng.uniform() >= probability:
            continue
        if name in _CONTINUOUS_BOUNDS:
            lo, hi = _CONTINUOUS_BOUNDS[name]
            sigma = sigma_fraction * (hi - lo)
            values[name] = _clip(float(values[name]) + float(rng.normal(0.0, sigma)), lo, hi)
        else:
            choices = _CATEGORICAL_CHOICES[name]
            values[name] = choices[int(rng.integers(0, len(choices)))]
    return from_dict(values)


def crossover(parent_a: Genome, parent_b: Genome, rng: np.random.Generator) -> Genome:
    """Uniform crossover (SPEC.md GA 설정 "균등교차"): each gene independently inherited from
    either parent with equal probability."""
    values_a, values_b = parent_a.to_dict(), parent_b.to_dict()
    child = {name: (values_a[name] if rng.uniform() < 0.5 else values_b[name]) for name in GENE_NAMES}
    return from_dict(child)


def genome_key(genome: Genome) -> tuple:
    """Hashable, rounded encoding for fitness-cache lookups (task instruction: '평가 캐싱
    필수'). Continuous genes rounded to 9 significant decimals -- far finer than mutation's own
    sigma, so this only collapses TRUE duplicates, never two genuinely different individuals
    (same rationale as wave21_ga.genome.genome_key)."""
    return (
        genome.strategy_kind,
        round(genome.entry_z, 9),
        genome.holding_days,
        round(genome.position_fraction, 9),
        None if genome.stop_loss_pct is None else round(genome.stop_loss_pct, 9),
        None if genome.take_profit_pct is None else round(genome.take_profit_pct, 9),
        genome.universe_breadth,
        genome.max_concurrent,
    )


__all__ = [
    "DUAL_DIRECTION_KINDS",
    "ENTRY_Z_MAX",
    "ENTRY_Z_MIN",
    "GENE_NAMES",
    "HOLDING_DAYS_CHOICES",
    "LONG_ONLY_KINDS",
    "MAX_CONCURRENT_CHOICES",
    "POSITION_FRACTION_MAX",
    "POSITION_FRACTION_MIN",
    "STOP_LOSS_PCT_CHOICES",
    "STRATEGY_KIND_BREAKOUT",
    "STRATEGY_KIND_CARRY",
    "STRATEGY_KIND_CHOICES",
    "STRATEGY_KIND_CONVEX_DUAL",
    "STRATEGY_KIND_FUNDING_SPIKE",
    "STRATEGY_KIND_MOMENTUM",
    "TAKE_PROFIT_PCT_CHOICES",
    "UNIVERSE_BREADTH_CHOICES",
    "Genome",
    "crossover",
    "from_dict",
    "genome_key",
    "mutate",
    "random_genome",
]
