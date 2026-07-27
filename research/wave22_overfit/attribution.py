# Wave-22 validation #5 -- gene contribution decomposition. From I5, change G1's 7 genes ONE AT
# A TIME to G1's own value (task: "I5에서 G1으로 7축을 하나씩 바꿔가며(one-at-a-time)") and measure
# each axis's standalone full-period-CAGR contribution. If the I5->G1 improvement concentrates in
# a single axis, that axis's "goodness" is more likely a lucky coincidence the GA's own multiple
# -testing surfaced; if it spreads across several axes, that is more consistent with a genuine,
# structural parameter-combination effect (harder to hit by chance).
#
# Two independent one-at-a-time decompositions are computed, since they can disagree when genes
# interact non-additively:
#   - forward:  I5 with ONLY axis X set to G1's value (measures axis X's contribution starting
#               FROM I5)
#   - backward: G1 with ONLY axis X reverted to I5's value (measures axis X's contribution
#               starting FROM G1, i.e. "leave this one out")
# Their sum vs the total I5->G1 gap is also reported (a large residual = strong interaction
# between genes, which this one-at-a-time method cannot attribute to any single axis).
#
# Concentration rule (pre-registered here, before results are read): among axes with a POSITIVE
# forward contribution (an axis that helps when added to I5 in isolation), let `share` be that
# axis's contribution divided by the sum of all positive contributions. "Concentrated in a single
# axis" = the top axis's share > CONCENTRATION_THRESHOLD (0.60). "Spread across >=2 axes" = the
# top axis's share <= CONCENTRATION_THRESHOLD AND at least 2 axes each have share >=
# SPREAD_MIN_SHARE (0.15). These thresholds are arbitrary-but-fixed judgment calls, disclosed
# here rather than tuned after seeing the numbers.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave21_ga import fitness
from research.wave21_ga.genome import GENE_NAMES, Genome, from_dict
from research.wave22_overfit.evaluate import MetricsCache
from research.wave22_overfit.genomes import G1_GENOME, I5_GENOME

CONCENTRATION_THRESHOLD: Final = 0.60
SPREAD_MIN_SHARE: Final = 0.15


def _one_axis_swap(source: Genome, axis: str, value: Any) -> Genome:
    values = source.to_dict()
    values[axis] = value
    return from_dict(values)


def concentration_from_contributions(contributions: dict[str, float]) -> dict[str, Any]:
    """Pure function, independent of genomes/cache -- see module docstring's 'Concentration
    rule' for the pre-registered thresholds. Exposed at module level (not nested inside run())
    specifically so it is directly unit-testable against hand-built contribution dicts
    (tests/test_wave22.py) without needing a market cache."""
    positive = {axis: value for axis, value in contributions.items() if value > 0.0}
    positive_sum = sum(positive.values())
    if positive_sum <= 0.0 or not positive:
        return {"positive_axes": [], "shares": {}, "top_axis": None, "top_share": None, "concentrated_single_axis": None, "spread_across_2plus": None, "n_axes_with_meaningful_share": 0}
    shares = {axis: value / positive_sum for axis, value in positive.items()}
    top_axis = max(shares, key=lambda axis: shares[axis])
    top_share = shares[top_axis]
    n_meaningful = sum(1 for share in shares.values() if share >= SPREAD_MIN_SHARE)
    return {
        "positive_axes": sorted(positive, key=lambda axis: -positive[axis]),
        "shares": shares,
        "top_axis": top_axis,
        "top_share": top_share,
        "concentrated_single_axis": bool(top_share > CONCENTRATION_THRESHOLD),
        "spread_across_2plus": bool(top_share <= CONCENTRATION_THRESHOLD and n_meaningful >= 2),
        "n_axes_with_meaningful_share": n_meaningful,
    }


def run(cache: fitness.MarketCache, metrics_cache: MetricsCache | None = None, i5: Genome = I5_GENOME, g1: Genome = G1_GENOME) -> dict[str, Any]:
    metrics_cache = metrics_cache if metrics_cache is not None else MetricsCache()
    i5_metrics = metrics_cache.get(i5, cache)
    g1_metrics = metrics_cache.get(g1, cache)

    total_gap_full = g1_metrics.full_cagr - i5_metrics.full_cagr
    total_gap_oos = g1_metrics.oos_cagr_self_contained - i5_metrics.oos_cagr_self_contained

    rows: dict[str, dict[str, Any]] = {}
    for axis in GENE_NAMES:
        i5_value, g1_value = getattr(i5, axis), getattr(g1, axis)
        unchanged = i5_value == g1_value
        if unchanged:
            rows[axis] = {
                "i5_value": i5_value, "g1_value": g1_value, "unchanged": True,
                "forward_full_cagr": i5_metrics.full_cagr, "forward_contribution_pp": 0.0, "forward_oos_contribution_pp": 0.0,
                "backward_full_cagr": g1_metrics.full_cagr, "backward_contribution_pp": 0.0, "backward_oos_contribution_pp": 0.0,
            }
            continue
        forward_genome = _one_axis_swap(i5, axis, g1_value)
        backward_genome = _one_axis_swap(g1, axis, i5_value)
        forward_metrics = metrics_cache.get(forward_genome, cache)
        backward_metrics = metrics_cache.get(backward_genome, cache)
        rows[axis] = {
            "i5_value": i5_value,
            "g1_value": g1_value,
            "unchanged": False,
            "forward_full_cagr": forward_metrics.full_cagr,
            "forward_contribution_pp": (forward_metrics.full_cagr - i5_metrics.full_cagr) * 100.0,
            "forward_oos_contribution_pp": (forward_metrics.oos_cagr_self_contained - i5_metrics.oos_cagr_self_contained) * 100.0,
            "backward_full_cagr": backward_metrics.full_cagr,
            "backward_contribution_pp": (g1_metrics.full_cagr - backward_metrics.full_cagr) * 100.0,
            "backward_oos_contribution_pp": (g1_metrics.oos_cagr_self_contained - backward_metrics.oos_cagr_self_contained) * 100.0,
        }

    changed_axes = [axis for axis in GENE_NAMES if not rows[axis]["unchanged"]]
    unchanged_axes = [axis for axis in GENE_NAMES if rows[axis]["unchanged"]]

    forward_concentration = concentration_from_contributions({axis: rows[axis]["forward_contribution_pp"] for axis in changed_axes})
    backward_concentration = concentration_from_contributions({axis: rows[axis]["backward_contribution_pp"] for axis in changed_axes})

    sum_forward = sum(rows[axis]["forward_contribution_pp"] for axis in changed_axes) / 100.0
    residual_full = total_gap_full - sum_forward

    forward_and_backward_agree = (
        forward_concentration["concentrated_single_axis"] == backward_concentration["concentrated_single_axis"]
        if forward_concentration["concentrated_single_axis"] is not None and backward_concentration["concentrated_single_axis"] is not None
        else None
    )

    return {
        "methodology": {
            "definition": "one-at-a-time (I5->G1, forward) and leave-one-out (G1->I5, backward) single-axis swaps, task instruction: 'I5에서 G1으로 7축을 하나씩'",
            "concentration_threshold": CONCENTRATION_THRESHOLD,
            "spread_min_share": SPREAD_MIN_SHARE,
            "concentration_rule": f"concentrated_single_axis = top contributing axis's share of the summed POSITIVE forward contributions > {CONCENTRATION_THRESHOLD:.0%}; spread_across_2plus = top share <= {CONCENTRATION_THRESHOLD:.0%} AND >=2 axes each hold >= {SPREAD_MIN_SHARE:.0%} share",
        },
        "i5_full_cagr": i5_metrics.full_cagr,
        "g1_full_cagr": g1_metrics.full_cagr,
        "total_gap_full_cagr_pp": total_gap_full * 100.0,
        "total_gap_oos_cagr_pp": total_gap_oos * 100.0,
        "changed_axes": changed_axes,
        "unchanged_axes": unchanged_axes,
        "per_axis": rows,
        "forward_concentration": forward_concentration,
        "backward_concentration": backward_concentration,
        "forward_and_backward_agree_on_concentration": forward_and_backward_agree,
        "sum_of_forward_contributions_pp": sum_forward * 100.0,
        "interaction_residual_pp": residual_full * 100.0,
        "limitations": [
            f"{len(unchanged_axes)} of 7 genes ({', '.join(unchanged_axes) if unchanged_axes else 'none'}) are IDENTICAL between I5 and G1, so only {len(changed_axes)} axes can possibly contribute to the gap -- 'spread across >=2 axes' is evaluated against this smaller active set, not all 7",
            "one-at-a-time attribution ignores higher-order interactions between genes; the interaction_residual_pp figure quantifies how much of the total gap the linear one-at-a-time sum does NOT explain -- a large residual means the genes interact and no single-axis story (concentrated or spread) fully describes the result",
            "forward and backward decompositions can disagree when interactions are strong; both are reported rather than only the more favorable one",
        ],
    }


__all__ = ["CONCENTRATION_THRESHOLD", "SPREAD_MIN_SHARE", "concentration_from_contributions", "run"]
