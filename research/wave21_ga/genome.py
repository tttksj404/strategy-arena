# Wave-21 GA gene definitions (SPEC.md "유전자 (탐색공간, 동결)" -- frozen table, 7 axes).
#
# Base structure being tuned: I5 (research/wave18_idle -- L4 carry + tiered idle-capital
# overlay). Only the 7 genes below ever vary; everything else (delta-neutral 2-leg, 1x
# leverage, measured cost model, $100/$90/$45 capital contract, overlay's OWN fixed
# threshold=8%/window=7/top_k=1, lending rate) is inherited from wave13/wave18 UNCHANGED --
# SPEC.md "그 외 고정 (델타중립·1x·실측비용·체결규약)".
#
# idle_mode's 4 values map 1:1 onto wave18's own I0/I1/I2/I5 (I3 all-universe overlay and I4
# reverse-carry are OUT OF SCOPE -- SPEC.md's enum only lists 4 values, not wave18's full 6):
#   none          -> I0 (nothing on an L4-idle day)
#   usdt_lend     -> I1 (USDT lending fallback only)
#   majors_low_thr-> I2 (BTC/ETH low-threshold carry overlay only, no lending fallback)
#   tiered        -> I5 (majors_low_thr first, usdt_lend fallback -- hierarchical)

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Gene bounds (SPEC.md table, byte-for-byte -- do not adjust post-hoc).
# ---------------------------------------------------------------------------

ENTRY_THRESHOLD_APR_MIN: Final = 0.05
ENTRY_THRESHOLD_APR_MAX: Final = 0.40
EXIT_THRESHOLD_RATIO_MIN: Final = 0.25
EXIT_THRESHOLD_RATIO_MAX: Final = 0.75
LEG_FRACTION_MIN: Final = 0.30
LEG_FRACTION_MAX: Final = 0.50

WINDOW_DAYS_CHOICES: Final[tuple[int, ...]] = (3, 5, 7, 10, 14)
TOP_K_PAIRS_CHOICES: Final[tuple[int, ...]] = (1, 2, 3)
UNIVERSE_BREADTH_CHOICES: Final[tuple[int, ...]] = (30, 100, 200, 300)

IDLE_MODE_NONE: Final = "none"
IDLE_MODE_USDT_LEND: Final = "usdt_lend"
IDLE_MODE_MAJORS_LOW_THR: Final = "majors_low_thr"
IDLE_MODE_TIERED: Final = "tiered"
IDLE_MODE_CHOICES: Final[tuple[str, ...]] = (
    IDLE_MODE_NONE,
    IDLE_MODE_USDT_LEND,
    IDLE_MODE_MAJORS_LOW_THR,
    IDLE_MODE_TIERED,
)

# L4/I5's own frozen values -- SPEC.md table's "비고" column, used as the reference genome
# for tests and for the report's "L4/I5 대비" framing (NOT part of the search itself: the GA
# never seeds an individual with these exact values on purpose, avoiding a "hint" that would
# make the search trivially easy to game / bias the OOS story).
L4_ENTRY_THRESHOLD_APR: Final = 0.15
L4_EXIT_THRESHOLD_RATIO: Final = 0.50
L4_WINDOW_DAYS: Final = 7
L4_TOP_K_PAIRS: Final = 1
L4_LEG_FRACTION: Final = 0.50
L4_UNIVERSE_BREADTH: Final = 200
I5_IDLE_MODE: Final = IDLE_MODE_TIERED

_CONTINUOUS_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "entry_threshold_apr": (ENTRY_THRESHOLD_APR_MIN, ENTRY_THRESHOLD_APR_MAX),
    "exit_threshold_ratio": (EXIT_THRESHOLD_RATIO_MIN, EXIT_THRESHOLD_RATIO_MAX),
    "leg_fraction": (LEG_FRACTION_MIN, LEG_FRACTION_MAX),
}
_CATEGORICAL_CHOICES: Final[dict[str, tuple]] = {
    "window_days": WINDOW_DAYS_CHOICES,
    "top_k_pairs": TOP_K_PAIRS_CHOICES,
    "universe_breadth": UNIVERSE_BREADTH_CHOICES,
    "idle_mode": IDLE_MODE_CHOICES,
}
GENE_NAMES: Final[tuple[str, ...]] = (
    "entry_threshold_apr",
    "exit_threshold_ratio",
    "window_days",
    "top_k_pairs",
    "leg_fraction",
    "universe_breadth",
    "idle_mode",
)


@dataclass(frozen=True, slots=True)
class Genome:
    entry_threshold_apr: float
    exit_threshold_ratio: float
    window_days: int
    top_k_pairs: int
    leg_fraction: float
    universe_breadth: int
    idle_mode: str

    def __post_init__(self) -> None:
        # Fail closed on an out-of-registry genome -- a silently-clamped invalid genome would
        # let a GA bug (e.g. a crossover/mutation typo) quietly explore outside the frozen
        # SPEC.md search space instead of surfacing the bug immediately.
        lo, hi = _CONTINUOUS_BOUNDS["entry_threshold_apr"]
        if not (lo <= self.entry_threshold_apr <= hi):
            raise ValueError(f"entry_threshold_apr {self.entry_threshold_apr} outside [{lo}, {hi}]")
        lo, hi = _CONTINUOUS_BOUNDS["exit_threshold_ratio"]
        if not (lo <= self.exit_threshold_ratio <= hi):
            raise ValueError(f"exit_threshold_ratio {self.exit_threshold_ratio} outside [{lo}, {hi}]")
        lo, hi = _CONTINUOUS_BOUNDS["leg_fraction"]
        if not (lo <= self.leg_fraction <= hi):
            raise ValueError(f"leg_fraction {self.leg_fraction} outside [{lo}, {hi}]")
        if self.window_days not in WINDOW_DAYS_CHOICES:
            raise ValueError(f"window_days {self.window_days} not in {WINDOW_DAYS_CHOICES}")
        if self.top_k_pairs not in TOP_K_PAIRS_CHOICES:
            raise ValueError(f"top_k_pairs {self.top_k_pairs} not in {TOP_K_PAIRS_CHOICES}")
        if self.universe_breadth not in UNIVERSE_BREADTH_CHOICES:
            raise ValueError(f"universe_breadth {self.universe_breadth} not in {UNIVERSE_BREADTH_CHOICES}")
        if self.idle_mode not in IDLE_MODE_CHOICES:
            raise ValueError(f"idle_mode {self.idle_mode!r} not in {IDLE_MODE_CHOICES}")

    @property
    def exit_threshold_apr(self) -> float:
        return self.entry_threshold_apr * self.exit_threshold_ratio

    @property
    def uses_overlay(self) -> bool:
        return self.idle_mode in (IDLE_MODE_MAJORS_LOW_THR, IDLE_MODE_TIERED)

    @property
    def uses_lending(self) -> bool:
        return self.idle_mode in (IDLE_MODE_USDT_LEND, IDLE_MODE_TIERED)

    def to_dict(self) -> dict[str, float | int | str]:
        return {name: getattr(self, name) for name in GENE_NAMES}


L4_BASELINE_GENOME: Final = Genome(
    entry_threshold_apr=L4_ENTRY_THRESHOLD_APR,
    exit_threshold_ratio=L4_EXIT_THRESHOLD_RATIO,
    window_days=L4_WINDOW_DAYS,
    top_k_pairs=L4_TOP_K_PAIRS,
    leg_fraction=L4_LEG_FRACTION,
    universe_breadth=L4_UNIVERSE_BREADTH,
    idle_mode=IDLE_MODE_NONE,
)
I5_BASELINE_GENOME: Final = Genome(
    entry_threshold_apr=L4_ENTRY_THRESHOLD_APR,
    exit_threshold_ratio=L4_EXIT_THRESHOLD_RATIO,
    window_days=L4_WINDOW_DAYS,
    top_k_pairs=L4_TOP_K_PAIRS,
    leg_fraction=L4_LEG_FRACTION,
    universe_breadth=L4_UNIVERSE_BREADTH,
    idle_mode=I5_IDLE_MODE,
)


def from_dict(payload: dict[str, float | int | str]) -> Genome:
    return Genome(
        entry_threshold_apr=float(payload["entry_threshold_apr"]),
        exit_threshold_ratio=float(payload["exit_threshold_ratio"]),
        window_days=int(payload["window_days"]),
        top_k_pairs=int(payload["top_k_pairs"]),
        leg_fraction=float(payload["leg_fraction"]),
        universe_breadth=int(payload["universe_breadth"]),
        idle_mode=str(payload["idle_mode"]),
    )


def random_genome(rng: np.random.Generator) -> Genome:
    """Uniform-random draw over the frozen gene ranges -- the shared initialization/sampling
    distribution for BOTH ga.py's generation-0 population and random_search.py's control
    draws (SPEC.md's controlled comparison requires the two searches to start from the same
    distribution; only what they DO with subsequent draws differs)."""
    return Genome(
        entry_threshold_apr=float(rng.uniform(*_CONTINUOUS_BOUNDS["entry_threshold_apr"])),
        exit_threshold_ratio=float(rng.uniform(*_CONTINUOUS_BOUNDS["exit_threshold_ratio"])),
        window_days=int(rng.choice(WINDOW_DAYS_CHOICES)),
        top_k_pairs=int(rng.choice(TOP_K_PAIRS_CHOICES)),
        leg_fraction=float(rng.uniform(*_CONTINUOUS_BOUNDS["leg_fraction"])),
        universe_breadth=int(rng.choice(UNIVERSE_BREADTH_CHOICES)),
        idle_mode=str(rng.choice(IDLE_MODE_CHOICES)),
    )


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def mutate(genome: Genome, rng: np.random.Generator, sigma_fraction: float = 0.10, probability: float = 0.15) -> Genome:
    """Per-gene independent mutation (SPEC.md "가우시안 변이(σ=범위의 10%, 확률 0.15)"):
    each of the 7 genes is mutated independently with probability `probability`. A continuous
    gene's mutation is a Gaussian perturbation with sigma = 10% of that gene's OWN range,
    clipped back into bounds; a categorical gene's mutation is a fresh uniform resample from
    its choice list (there is no continuous "direction" to perturb a category in)."""
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
    """Uniform crossover (SPEC.md "균등교차"): each gene independently inherited from either
    parent with equal probability."""
    values_a, values_b = parent_a.to_dict(), parent_b.to_dict()
    child = {name: (values_a[name] if rng.uniform() < 0.5 else values_b[name]) for name in GENE_NAMES}
    return from_dict(child)


def genome_key(genome: Genome) -> tuple:
    """Hashable, rounded encoding for fitness-cache lookups (task instruction: '평가 캐싱(동일
    유전자 -> 결과 재사용) 필수'). Continuous genes are rounded to 9 significant decimal
    places -- far finer than mutation's own sigma (>=3% of a gene's range even after several
    generations of decay-free Gaussian noise), so this only collapses TRUE duplicates
    (elite carry-over, or two independent draws landing on the exact same float by chance),
    never two genuinely-different individuals."""
    return (
        round(genome.entry_threshold_apr, 9),
        round(genome.exit_threshold_ratio, 9),
        genome.window_days,
        genome.top_k_pairs,
        round(genome.leg_fraction, 9),
        genome.universe_breadth,
        genome.idle_mode,
    )


__all__ = [
    "ENTRY_THRESHOLD_APR_MAX",
    "ENTRY_THRESHOLD_APR_MIN",
    "EXIT_THRESHOLD_RATIO_MAX",
    "EXIT_THRESHOLD_RATIO_MIN",
    "GENE_NAMES",
    "IDLE_MODE_CHOICES",
    "IDLE_MODE_MAJORS_LOW_THR",
    "IDLE_MODE_NONE",
    "IDLE_MODE_TIERED",
    "IDLE_MODE_USDT_LEND",
    "I5_BASELINE_GENOME",
    "LEG_FRACTION_MAX",
    "LEG_FRACTION_MIN",
    "L4_BASELINE_GENOME",
    "TOP_K_PAIRS_CHOICES",
    "UNIVERSE_BREADTH_CHOICES",
    "WINDOW_DAYS_CHOICES",
    "Genome",
    "crossover",
    "from_dict",
    "genome_key",
    "mutate",
    "random_genome",
]
