# Wave-22 validation #1 -- parameter stability landscape (task's "가장 중요" validation).
# Shakes each of G1's 7 genes +-10%/+-20% (see perturb.py for exactly what that means per gene
# kind) and re-evaluates with the SAME engine used to certify G1 itself
# (research/wave21_ga/fitness.py). An edge should look like a plateau (neighbors perform close
# to G1); overfitting should look like a spike (neighbors perform much worse). Also runs a 2x
# axis grid for >=2 axis pairs (task requirement), to catch interaction effects a single-axis
# sweep would miss.
#
# Stability statistic (task: "'이웃 평균 성과 / G1 성과' 비율 산출(0.8 이상이면 안정)"), computed
# PER AXIS from that axis's available neighbor points' full-period CAGR. The task's pre-
# registered verdict table then needs ONE aggregate number ("안정성 비율 >=0.8"); this module
# reports both a mean-of-axis-ratios and a min-of-axis-ratios aggregate, and uses the MIN as the
# primary/decision figure -- deliberately the stricter of the two (a single sharp cliff on one
# gene is a real overfitting signal even if the other six genes are flat), matching the task's
# explicit "판정을 유리하게 쓰지 말 것" instruction. The mean is reported alongside for context,
# never as the decision figure.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave21_ga import fitness
from research.wave21_ga.genome import GENE_NAMES, Genome
from research.wave22_overfit import perturb
from research.wave22_overfit.evaluate import MetricsCache
from research.wave22_overfit.genomes import G1_GENOME

STABLE_RATIO_THRESHOLD: Final = 0.8
GRID_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    # Core signal shape: the entry/exit thresholds jointly define the hysteresis band.
    ("entry_threshold_apr", "exit_threshold_ratio"),
    # Sizing/concentration: top_k_pairs x leg_fraction is EXACTLY the pair whose product sets
    # gross exposure -- the mechanism that forced G1's own top_k_pairs 3->1 adjustment
    # (genomes.py's provenance note). Most substantively relevant pair to probe for wave22.
    ("top_k_pairs", "leg_fraction"),
)
GRID_TIERS: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)


def _point_payload(perturbation: perturb.Perturbation, metrics_cache: MetricsCache, cache: fitness.MarketCache, g1_full_cagr: float, g1_oos_cagr: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tier": perturbation.tier,
        "tier_label": perturbation.tier_label,
        "available": perturbation.available,
        "gene_value": perturbation.gene_value,
        "note": perturbation.note,
    }
    if not perturbation.available:
        payload.update(full_cagr=None, oos_cagr=None, full_cagr_ratio=None, oos_cagr_ratio=None, gross_feasible_1x=None)
        return payload
    metrics = metrics_cache.get(perturbation.genome, cache)
    payload.update(
        full_cagr=metrics.full_cagr,
        oos_cagr=metrics.oos_cagr_self_contained,
        full_cagr_ratio=(metrics.full_cagr / g1_full_cagr) if g1_full_cagr != 0.0 else None,
        oos_cagr_ratio=(metrics.oos_cagr_self_contained / g1_oos_cagr) if g1_oos_cagr != 0.0 else None,
        gross_feasible_1x=metrics.gross_feasible_1x,
    )
    return payload


def _axis_summary(axis: str, base: Genome, metrics_cache: MetricsCache, cache: fitness.MarketCache, g1_full_cagr: float, g1_oos_cagr: float) -> dict[str, Any]:
    perturbations = perturb.axis_perturbations(base, axis)
    points = [_point_payload(p, metrics_cache, cache, g1_full_cagr, g1_oos_cagr) for p in perturbations]
    available_full = [pt["full_cagr"] for pt in points if pt["available"]]
    available_oos = [pt["oos_cagr"] for pt in points if pt["available"]]

    neighbor_avg_full = (sum(available_full) / len(available_full)) if available_full else None
    neighbor_avg_oos = (sum(available_oos) / len(available_oos)) if available_oos else None
    neighbor_avg_ratio = (neighbor_avg_full / g1_full_cagr) if (neighbor_avg_full is not None and g1_full_cagr != 0.0) else None
    neighbor_avg_oos_ratio = (neighbor_avg_oos / g1_oos_cagr) if (neighbor_avg_oos is not None and g1_oos_cagr != 0.0) else None
    ratios = [pt["full_cagr_ratio"] for pt in points if pt["available"] and pt["full_cagr_ratio"] is not None]
    neighbor_min_ratio = min(ratios) if ratios else None

    return {
        "kind": perturb.axis_kind(axis),
        "baseline_value": getattr(base, axis),
        "points": points,
        "n_available": len(available_full),
        "n_total_tiers": len(points),
        "at_range_boundary": len(available_full) < len(points),
        "neighbor_avg_full_cagr": neighbor_avg_full,
        "neighbor_avg_full_cagr_ratio": neighbor_avg_ratio,
        "neighbor_min_full_cagr_ratio": neighbor_min_ratio,
        "neighbor_avg_oos_cagr_ratio": neighbor_avg_oos_ratio,
        "stable_mean_ratio_ge_0_8": (neighbor_avg_ratio >= STABLE_RATIO_THRESHOLD) if neighbor_avg_ratio is not None else None,
    }


def _grid_pair(axis_a: str, axis_b: str, base: Genome, metrics_cache: MetricsCache, cache: fitness.MarketCache, g1_full_cagr: float) -> dict[str, Any]:
    cells = perturb.grid_perturbations(base, axis_a, axis_b, GRID_TIERS)
    cell_payloads = []
    ratios: list[float] = []
    for cell in cells:
        if not cell.available:
            cell_payloads.append({"tier_a": cell.tier_a, "tier_b": cell.tier_b, "available": False, "note": cell.note, "full_cagr": None, "full_cagr_ratio": None, "gross_feasible_1x": None})
            continue
        metrics = metrics_cache.get(cell.genome, cache)
        ratio = (metrics.full_cagr / g1_full_cagr) if g1_full_cagr != 0.0 else None
        if ratio is not None:
            ratios.append(ratio)
        cell_payloads.append({
            "tier_a": cell.tier_a,
            "tier_b": cell.tier_b,
            "available": True,
            "note": "",
            "full_cagr": metrics.full_cagr,
            "full_cagr_ratio": ratio,
            "gross_feasible_1x": metrics.gross_feasible_1x,
        })
    n_infeasible = sum(1 for pt in cell_payloads if pt.get("gross_feasible_1x") is False)
    return {
        "axis_a": axis_a,
        "axis_b": axis_b,
        "tiers": list(GRID_TIERS),
        "cells": cell_payloads,
        "n_available_cells": sum(1 for pt in cell_payloads if pt["available"]),
        "n_gross_infeasible_cells": n_infeasible,
        "min_full_cagr_ratio": min(ratios) if ratios else None,
        "mean_full_cagr_ratio": (sum(ratios) / len(ratios)) if ratios else None,
    }


def run(cache: fitness.MarketCache, base: Genome = G1_GENOME, metrics_cache: MetricsCache | None = None) -> dict[str, Any]:
    metrics_cache = metrics_cache if metrics_cache is not None else MetricsCache()
    base_metrics = metrics_cache.get(base, cache)
    g1_full_cagr = base_metrics.full_cagr
    g1_oos_cagr = base_metrics.oos_cagr_self_contained

    per_axis = {axis: _axis_summary(axis, base, metrics_cache, cache, g1_full_cagr, g1_oos_cagr) for axis in GENE_NAMES}

    axis_mean_ratios = {axis: summary["neighbor_avg_full_cagr_ratio"] for axis, summary in per_axis.items() if summary["neighbor_avg_full_cagr_ratio"] is not None}
    if not axis_mean_ratios:
        raise RuntimeError("sensitivity.run: no axis produced any available neighbor -- cannot compute a stability ratio")
    worst_axis = min(axis_mean_ratios, key=lambda axis: axis_mean_ratios[axis])
    overall_min = axis_mean_ratios[worst_axis]
    overall_mean = sum(axis_mean_ratios.values()) / len(axis_mean_ratios)
    n_below_0_8 = sum(1 for r in axis_mean_ratios.values() if r < STABLE_RATIO_THRESHOLD)
    n_below_0_6 = sum(1 for r in axis_mean_ratios.values() if r < 0.6)

    grid = {"pairs": [_grid_pair(axis_a, axis_b, base, metrics_cache, cache, g1_full_cagr) for axis_a, axis_b in GRID_PAIRS]}

    return {
        "methodology": {
            "perturbation_definition": "continuous genes: +-10%/+-20% of G1's own value, clipped to genome.py bounds; ordered-choice genes: 1/2 steps in the frozen choice tuple; idle_mode: the 3 other categories (unordered, no direction)",
            "stability_statistic": "neighbor_avg_full_period_cagr / g1_full_period_cagr, per axis",
            "stable_threshold": STABLE_RATIO_THRESHOLD,
            "primary_aggregate": "min across the 7 per-axis neighbor-avg ratios (conservative: every axis must be stable, not just on average)",
            "secondary_aggregate": "mean across the 7 per-axis neighbor-avg ratios (reported for context only, never decides the verdict)",
            "grid_pairs_rationale": "entry/exit thresholds (hysteresis band shape); top_k_pairs/leg_fraction (gross-exposure sizing pair, the exact mechanism behind G1's own top_k_pairs 3->1 adjustment)",
        },
        "g1_full_cagr": g1_full_cagr,
        "g1_oos_cagr_self_contained": g1_oos_cagr,
        "per_axis": per_axis,
        "overall": {
            "stability_ratio_min_of_axis_means": overall_min,
            "stability_ratio_mean_of_axis_means": overall_mean,
            "worst_axis": worst_axis,
            "n_axes_below_0_8": n_below_0_8,
            "n_axes_below_0_6": n_below_0_6,
            "primary_metric": "stability_ratio_min_of_axis_means",
            "primary_value": overall_min,
            "stable": bool(overall_min >= STABLE_RATIO_THRESHOLD),
            "fail_threshold_0_6_breached": bool(overall_min < 0.6),
        },
        "grid": grid,
    }


__all__ = ["GRID_PAIRS", "GRID_TIERS", "STABLE_RATIO_THRESHOLD", "run"]
