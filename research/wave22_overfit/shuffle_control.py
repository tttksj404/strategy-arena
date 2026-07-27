# Wave-22 validation #6 -- false discovery control. Draws 30 random genomes from the SAME
# distribution the wave-21 GA/random-search itself samples from (genome.random_genome: uniform
# over each gene's own registered range/choice list -- task: "G1 유전자를 무작위로 섞은(shuffle)
# 유전자 30개"; a genuine per-gene value permutation is not well-defined across genes of
# different types/units, so "shuffle" is operationalized as "redraw every gene independently at
# random", i.e. exactly genome.random_genome's own distribution, matching wave21's own H1
# random-search control's sampling convention) and evaluates each with the identical engine.
# G1's percentile rank within that null distribution answers: could a genome THIS good have
# plausibly come from undirected random search, or does G1 stand out?
#
# Sizing constraint (methodology decision, disclosed): the 30 draws are rejection-sampled to
# satisfy the SAME gross<=1x constraint G1 itself must satisfy (gates21.gross_usdt <=
# ACTIVE_CAPITAL) -- otherwise a random genome could win purely by taking on more leverage
# (top_k_pairs=2 or 3 combined with any leg_fraction in its registered [0.30, 0.50] floor
# already pushes gross > 1x; only top_k_pairs=1 is EVER gross-feasible at this capital, given
# leg_fraction's own registered floor of 0.30), which would make the comparison about leverage
# instead of about parameter-choice skill. A direct consequence: every gross-feasible genome
# (G1 included) has top_k_pairs=1 -- that gene is effectively FORCED, not a free draw, in this
# control. The remaining 6 genes stay free draws. This is disclosed in the output's
# `methodology` block, not hidden.

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Final

import numpy as np

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from research.wave10_carry100.engine import ACTIVE_CAPITAL
from research.wave21_ga import fitness, gates21
from research.wave21_ga.genome import Genome, random_genome
from research.wave22_overfit.evaluate import MetricsCache
from research.wave22_overfit.genomes import G1_GENOME

N_SHUFFLES: Final = 30
SEED: Final = 20_260_728  # wave22's own freeze-date-style seed (one day after wave21_ga.gates21.SEED=20260727)
TOP_PERCENTILE_PASS: Final = 5.0
MAX_DRAW_ATTEMPTS_PER_ACCEPT: Final = 500  # generous safety cap; measured acceptance rate is ~1/3 (see module docstring)


def draw_gross_feasible_genomes(n: int, seed: int) -> tuple[list[Genome], int]:
    rng = np.random.default_rng(seed)
    accepted: list[Genome] = []
    attempts = 0
    max_attempts = n * MAX_DRAW_ATTEMPTS_PER_ACCEPT
    while len(accepted) < n and attempts < max_attempts:
        attempts += 1
        candidate = random_genome(rng)
        if gates21.gross_usdt(candidate) <= ACTIVE_CAPITAL + 1e-9:
            accepted.append(candidate)
    if len(accepted) < n:
        raise RuntimeError(f"draw_gross_feasible_genomes: only {len(accepted)}/{n} accepted after {attempts} draws (max {max_attempts}) -- acceptance rate lower than expected")
    return accepted, attempts


def _percentile_rank(g1_value: float, pool: list[float]) -> dict[str, Any]:
    n_below = sum(1 for value in pool if value < g1_value)
    n_equal = sum(1 for value in pool if value == g1_value)
    beats_fraction_of_pool_pct = (n_below + 0.5 * n_equal) / len(pool) * 100.0 if pool else None
    pooled_with_g1 = sorted(pool + [g1_value], reverse=True)
    rank_best_is_1 = pooled_with_g1.index(g1_value) + 1
    top_pct_of_pooled = rank_best_is_1 / len(pooled_with_g1) * 100.0
    return {
        "n_pool": len(pool),
        "n_random_below_g1": n_below,
        "g1_beats_pct_of_random_pool": beats_fraction_of_pool_pct,
        "g1_rank_within_pooled_31": rank_best_is_1,
        "g1_top_pct_of_pooled_31": top_pct_of_pooled,
        "g1_in_top_5pct": bool(top_pct_of_pooled <= TOP_PERCENTILE_PASS),
    }


def run(cache: fitness.MarketCache, metrics_cache: MetricsCache | None = None, g1: Genome = G1_GENOME, n: int = N_SHUFFLES, seed: int = SEED) -> dict[str, Any]:
    metrics_cache = metrics_cache if metrics_cache is not None else MetricsCache()
    g1_metrics = metrics_cache.get(g1, cache)

    genomes, attempts = draw_gross_feasible_genomes(n, seed)
    rows: list[dict[str, Any]] = []
    for index, genome in enumerate(genomes):
        metrics = metrics_cache.get(genome, cache)
        rows.append({
            "index": index,
            "genome": genome.to_dict(),
            "full_cagr": metrics.full_cagr,
            "oos_cagr_self_contained": metrics.oos_cagr_self_contained,
            "mdd_full": metrics.mdd_full,
            "gross_usdt": metrics.gross_usdt,
        })

    full_cagrs = [row["full_cagr"] for row in rows]
    oos_cagrs = [row["oos_cagr_self_contained"] for row in rows]
    rank_full = _percentile_rank(g1_metrics.full_cagr, full_cagrs)
    rank_oos = _percentile_rank(g1_metrics.oos_cagr_self_contained, oos_cagrs)

    return {
        "methodology": {
            "sampling_distribution": "research.wave21_ga.genome.random_genome (uniform per-gene draw over registered bounds/choices) -- identical to wave21's own GA-generation-0/random-search-control distribution",
            "n_draws_requested": n,
            "n_draws_attempted": attempts,
            "seed": seed,
            "sizing_constraint": "rejection-sampled to gross_usdt <= ACTIVE_CAPITAL (same 1x constraint G1 itself satisfies)",
            "forced_axis_note": "leg_fraction's own registered floor (0.30) already makes top_k_pairs in {2,3} gross-infeasible at any leg_fraction, so EVERY accepted draw (G1 included) has top_k_pairs=1 -- that gene is effectively forced, not a free draw, under this constraint; the other 6 genes remain free",
            "primary_ranking_metric": "full-period CAGR (matches G1's own headline metric)",
            "secondary_ranking_metric": "OOS(self-contained) CAGR -- a genome could rank well on full-period but poorly OOS, which is itself informative",
            "pass_criterion": f"G1 in the top {TOP_PERCENTILE_PASS:.0f}% of the pooled (30 random + G1) distribution",
        },
        "g1_full_cagr": g1_metrics.full_cagr,
        "g1_oos_cagr_self_contained": g1_metrics.oos_cagr_self_contained,
        "draws": rows,
        "rank_by_full_cagr": rank_full,
        "rank_by_oos_cagr": rank_oos,
        "g1_in_top_5pct_full_cagr": rank_full["g1_in_top_5pct"],
        "g1_in_top_5pct_oos_cagr": rank_oos["g1_in_top_5pct"],
        "limitations": [
            f"n={n} random draws is a small null sample -- percentile estimates have coarse resolution (1 draw = {100.0/n:.1f} percentage points); a 'top 5%' claim from n=30 rests on roughly the single best 1-2 draws' position",
            "the sizing constraint forces top_k_pairs=1 on every draw (see forced_axis_note) -- this control tests whether G1's OTHER 6 gene choices beat random chance, not whether its sizing choice does (sizing feasibility already settles that separately)",
            "an unconstrained variant (no gross filter, comparing on a leverage-normalized basis) was not run -- out of this validation's scope as specified",
        ],
    }


__all__ = ["MAX_DRAW_ATTEMPTS_PER_ACCEPT", "N_SHUFFLES", "SEED", "TOP_PERCENTILE_PASS", "draw_gross_feasible_genomes", "run"]
