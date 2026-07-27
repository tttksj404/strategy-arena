# Wave-22 gene-perturbation utilities (validation #1, "파라미터 안정성 지형"). Pure genome ->
# genome transforms, no market cache / backtest dependency -- kept separate from
# sensitivity.py so this logic is unit-testable without the real disk cache (see
# tests/test_wave22.py).
#
# ±10%/±20% only has a direct numeric meaning for the 3 CONTINUOUS genes (entry_threshold_apr,
# exit_threshold_ratio, leg_fraction): tier -2/-1/+1/+2 <-> -20%/-10%/+10%/+20% of the gene's
# OWN current value (not of its registered range), clipped back into genome.py's frozen bounds.
# The 3 ORDERED-CHOICE genes (window_days, top_k_pairs, universe_breadth) have no continuous
# "%"; a step in the sorted choice list is used instead (tier -2/-1/+1/+2 <-> two/one choice(s)
# down/up from G1's own position in genome.py's frozen WINDOW_DAYS_CHOICES /
# TOP_K_PAIRS_CHOICES / UNIVERSE_BREADTH_CHOICES tuples). idle_mode has no order at all (4
# unordered categories); its "neighborhood" is simply the 3 other categories (tier 1/2/3), each
# a single categorical flip.
#
# A tier is UNAVAILABLE (available=False, genome=None) when either (a) it steps outside the
# gene's registered range/choice list, or (b) for a continuous gene, clipping back into bounds
# collapses the perturbed value onto the baseline itself (no real perturbation happened). Both
# cases are left in the output with a `note` explaining why, rather than silently dropped --
# G1 sitting at a hard boundary on a gene (this happens for window_days=14 [top of range] and
# top_k_pairs=1 [bottom of range]) is itself a fact worth surfacing, not noise to hide.

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave21_ga.genome import (
    ENTRY_THRESHOLD_APR_MAX,
    ENTRY_THRESHOLD_APR_MIN,
    EXIT_THRESHOLD_RATIO_MAX,
    EXIT_THRESHOLD_RATIO_MIN,
    GENE_NAMES,
    IDLE_MODE_CHOICES,
    LEG_FRACTION_MAX,
    LEG_FRACTION_MIN,
    TOP_K_PAIRS_CHOICES,
    UNIVERSE_BREADTH_CHOICES,
    WINDOW_DAYS_CHOICES,
    Genome,
    from_dict,
)

CONTINUOUS_AXES: Final[tuple[str, ...]] = ("entry_threshold_apr", "exit_threshold_ratio", "leg_fraction")
ORDERED_CATEGORICAL_AXES: Final[tuple[str, ...]] = ("window_days", "top_k_pairs", "universe_breadth")
UNORDERED_CATEGORICAL_AXES: Final[tuple[str, ...]] = ("idle_mode",)
assert set(CONTINUOUS_AXES) | set(ORDERED_CATEGORICAL_AXES) | set(UNORDERED_CATEGORICAL_AXES) == set(GENE_NAMES)

_CONTINUOUS_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "entry_threshold_apr": (ENTRY_THRESHOLD_APR_MIN, ENTRY_THRESHOLD_APR_MAX),
    "exit_threshold_ratio": (EXIT_THRESHOLD_RATIO_MIN, EXIT_THRESHOLD_RATIO_MAX),
    "leg_fraction": (LEG_FRACTION_MIN, LEG_FRACTION_MAX),
}
_ORDERED_CHOICES: Final[dict[str, tuple]] = {
    "window_days": WINDOW_DAYS_CHOICES,
    "top_k_pairs": TOP_K_PAIRS_CHOICES,
    "universe_breadth": UNIVERSE_BREADTH_CHOICES,
}
# tier -> relative delta of the gene's OWN current value (continuous genes only).
_CONTINUOUS_TIER_DELTA: Final[dict[int, float]] = {-2: -0.20, -1: -0.10, 1: 0.10, 2: 0.20}
SINGLE_AXIS_TIERS: Final[tuple[int, ...]] = (-2, -1, 1, 2)
IDLE_MODE_TIERS: Final[tuple[int, ...]] = (1, 2, 3)


def axis_kind(axis: str) -> str:
    if axis in CONTINUOUS_AXES:
        return "continuous"
    if axis in ORDERED_CATEGORICAL_AXES:
        return "ordered_categorical"
    if axis in UNORDERED_CATEGORICAL_AXES:
        return "unordered_categorical"
    raise ValueError(f"unknown axis {axis!r}, expected one of {GENE_NAMES}")


def tier_label(axis: str, tier: int) -> str:
    if tier == 0:
        return "baseline"
    if axis in CONTINUOUS_AXES:
        delta = _CONTINUOUS_TIER_DELTA.get(tier)
        return f"{delta:+.0%}" if delta is not None else f"tier{tier:+d}(undefined)"
    if axis in ORDERED_CATEGORICAL_AXES:
        return f"{tier:+d}step"
    if axis in UNORDERED_CATEGORICAL_AXES:
        return f"alt{tier}"
    raise ValueError(f"unknown axis {axis!r}")


def _value_at_tier(base: Genome, axis: str, tier: int) -> tuple[object | None, bool, str]:
    """Returns (value, available, note) for `axis` at signed integer `tier` (0 == baseline,
    always available -- used by grid_perturbations to hold one axis fixed)."""
    if tier == 0:
        return getattr(base, axis), True, ""
    kind = axis_kind(axis)
    if kind == "continuous":
        lo, hi = _CONTINUOUS_BOUNDS[axis]
        delta = _CONTINUOUS_TIER_DELTA.get(tier)
        if delta is None:
            return None, False, f"tier {tier} undefined for a continuous axis (expected one of {sorted(_CONTINUOUS_TIER_DELTA)})"
        base_value = float(getattr(base, axis))
        raw = base_value * (1.0 + delta)
        clipped = min(max(raw, lo), hi)
        if math.isclose(clipped, base_value, rel_tol=1e-9, abs_tol=1e-12):
            return None, False, f"{delta:+.0%} of {base_value} clips to registry bound [{lo}, {hi}] and collapses onto baseline -- G1 sits at/near this bound"
        note = "" if math.isclose(clipped, raw, rel_tol=1e-9) else f"clipped from {raw:.6f} to registry bound [{lo}, {hi}]"
        return clipped, True, note
    if kind == "ordered_categorical":
        choices = _ORDERED_CHOICES[axis]
        idx = choices.index(getattr(base, axis))
        new_idx = idx + tier
        if not (0 <= new_idx < len(choices)):
            return None, False, f"index {idx}{tier:+d}={new_idx} outside choice range [0, {len(choices) - 1}] {choices} -- G1 is at the range boundary on this axis"
        return choices[new_idx], True, ""
    if kind == "unordered_categorical":
        alternatives = [c for c in IDLE_MODE_CHOICES if c != base.idle_mode]
        if not (1 <= tier <= len(alternatives)):
            return None, False, f"tier {tier} undefined ({len(alternatives)} non-baseline alternatives exist)"
        return alternatives[tier - 1], True, ""
    raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class Perturbation:
    axis: str
    tier: int
    tier_label: str
    available: bool
    gene_value: object | None
    genome: Genome | None
    note: str


def perturb_axis(base: Genome, axis: str, tier: int) -> Perturbation:
    value, available, note = _value_at_tier(base, axis, tier)
    genome = None
    if available:
        values = base.to_dict()
        values[axis] = value
        genome = from_dict(values)
    return Perturbation(axis=axis, tier=tier, tier_label=tier_label(axis, tier), available=available, gene_value=value, genome=genome, note=note)


def axis_perturbations(base: Genome, axis: str) -> list[Perturbation]:
    tiers = IDLE_MODE_TIERS if axis in UNORDERED_CATEGORICAL_AXES else SINGLE_AXIS_TIERS
    return [perturb_axis(base, axis, tier) for tier in tiers]


def all_single_axis_perturbations(base: Genome) -> dict[str, list[Perturbation]]:
    return {axis: axis_perturbations(base, axis) for axis in GENE_NAMES}


@dataclass(frozen=True, slots=True)
class GridCell:
    axis_a: str
    tier_a: int
    axis_b: str
    tier_b: int
    available: bool
    genome: Genome | None
    note: str


def grid_perturbations(base: Genome, axis_a: str, axis_b: str, tiers: tuple[int, ...] = (-2, -1, 0, 1, 2)) -> list[GridCell]:
    """Cartesian product of both axes' tiers (baseline tier 0 included so the grid's center
    cell is G1 itself). Neither axis may be idle_mode (unordered -- a 2-axis numeric "landscape"
    grid is not a meaningful concept for it; see sensitivity.py's own methodology note)."""
    if axis_a in UNORDERED_CATEGORICAL_AXES or axis_b in UNORDERED_CATEGORICAL_AXES:
        raise ValueError("grid_perturbations: idle_mode (unordered categorical) is not supported in a 2-axis grid")
    if axis_a == axis_b:
        raise ValueError("grid_perturbations: axis_a and axis_b must differ")
    cells: list[GridCell] = []
    for tier_a in tiers:
        for tier_b in tiers:
            value_a, avail_a, note_a = _value_at_tier(base, axis_a, tier_a)
            value_b, avail_b, note_b = _value_at_tier(base, axis_b, tier_b)
            if not (avail_a and avail_b):
                combined_note = "; ".join(n for n in (note_a, note_b) if n)
                cells.append(GridCell(axis_a, tier_a, axis_b, tier_b, False, None, combined_note))
                continue
            values = base.to_dict()
            values[axis_a] = value_a
            values[axis_b] = value_b
            cells.append(GridCell(axis_a, tier_a, axis_b, tier_b, True, from_dict(values), ""))
    return cells


__all__ = [
    "CONTINUOUS_AXES",
    "IDLE_MODE_TIERS",
    "ORDERED_CATEGORICAL_AXES",
    "SINGLE_AXIS_TIERS",
    "UNORDERED_CATEGORICAL_AXES",
    "GridCell",
    "Perturbation",
    "all_single_axis_perturbations",
    "axis_kind",
    "axis_perturbations",
    "grid_perturbations",
    "perturb_axis",
    "tier_label",
]
