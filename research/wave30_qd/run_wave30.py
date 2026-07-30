# Wave-30 orchestration. Phase 1 (search) runs the five seeds in parallel processes, IS only.
# Phase 2 (judgement) unseals OOS exactly once, on the median-seed candidate that SPEC.md's
# frozen selection rule picks, and runs gates P1-P7.
#
# Usage:
#   python research/wave30_qd/run_wave30.py search      # phase 1, writes results/seed_*.json
#   python research/wave30_qd/run_wave30.py judge       # phase 2, writes results/final.json
#   python research/wave30_qd/run_wave30.py all

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd import gates30
from research.wave30_qd.dataio30 import build_market_cache, i5_baseline_total_curve
from research.wave30_qd.fitness30 import Evaluation, GRID_SHAPE, baseline_reference, evaluate, final_evaluation
from research.wave30_qd.genome30 import Genome
from research.wave30_qd.search30 import (
    MAP_ELITES_INIT,
    MAP_ELITES_ITERATIONS,
    NSGA_GENERATIONS,
    NSGA_POPULATION,
    RANDOM_BUDGET,
    Evaluator,
    run_map_elites,
    run_nsga2,
    run_random_search,
)

SEEDS: Final = (2026301, 2026302, 2026303, 2026304, 2026305)
RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"


def _evaluation_json(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "genome": evaluation.genome.to_dict(),
        "fitness": evaluation.fitness,
        "fold_cagrs": list(evaluation.fold_cagrs),
        "is_total_cagr": evaluation.is_total_cagr,
        "is_total_final_usdt": evaluation.is_total_final,
        "sleeve_mdd": evaluation.sleeve_mdd,
        "total_mdd": evaluation.total_mdd,
        "trades_per_year": evaluation.trades_per_year,
        "n_trades": evaluation.n_trades,
        "n_liquidations": evaluation.n_liquidations,
        "wipe_probability": evaluation.wipe_probability,
        "descriptor": list(evaluation.descriptor),
        "mean_leverage": evaluation.mean_leverage,
        "min_notional_usdt": evaluation.min_notional_usdt,
        "sleeve_survived": evaluation.sleeve_survived,
    }


def run_one_seed(seed: int) -> dict[str, Any]:
    """One seed's full IS budget: MAP-Elites, NSGA-II, and the matched random control.

    All three share ONE Evaluator so the memo cache is shared and the budget accounting is
    exact; the random control is given RANDOM_BUDGET = the sum of the other two budgets, as
    SPEC.md freezes.
    """
    started = time.time()
    cache = build_market_cache()

    evolved_evaluator = Evaluator(cache, seed=seed)
    archive = run_map_elites(
        evolved_evaluator,
        np.random.default_rng(seed),
        n_init=MAP_ELITES_INIT,
        n_iterations=MAP_ELITES_ITERATIONS,
    )
    map_elites_evaluations = evolved_evaluator.n_evaluations

    pareto = run_nsga2(
        evolved_evaluator,
        np.random.default_rng(seed + 500_000),
        population_size=NSGA_POPULATION,
        generations=NSGA_GENERATIONS,
    )
    # Cells the Pareto search discovered also belong on the map -- MAP-Elites and NSGA-II are
    # two lenses on ONE evolved archive, and P1 compares that archive against random search.
    for evaluation in pareto:
        archive.consider(evaluation)

    control_evaluator = Evaluator(cache, seed=seed + 900_000)
    random_archive = run_random_search(
        control_evaluator, np.random.default_rng(seed + 900_000), budget=RANDOM_BUDGET
    )

    best = archive.best
    random_best = random_archive.best
    elites = archive.elites()
    return {
        "seed": seed,
        "runtime_seconds": time.time() - started,
        "evaluations_evolved": evolved_evaluator.n_evaluations,
        "evaluations_map_elites": map_elites_evaluations,
        "evaluations_nsga2": evolved_evaluator.n_evaluations - map_elites_evaluations,
        "evaluations_random": control_evaluator.n_evaluations,
        "archive_coverage": archive.coverage,
        "random_coverage": random_archive.coverage,
        "qd_score": archive.qd_score,
        "random_qd_score": random_archive.qd_score,
        "evolved_best_fitness": best.fitness if best else float("-inf"),
        "random_best_fitness": random_best.fitness if random_best else float("-inf"),
        "best": _evaluation_json(best) if best else None,
        "random_best": _evaluation_json(random_best) if random_best else None,
        "archive": [_evaluation_json(item) for item in elites],
        "pareto_front": [_evaluation_json(item) for item in pareto],
        "grid_shape": list(GRID_SHAPE),
    }


def phase_search() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        for row in pool.map(run_one_seed, SEEDS):
            (RESULTS_DIR / f"seed_{row['seed']}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            print(
                f"seed {row['seed']}: evolved best {row['evolved_best_fitness']:.6f} "
                f"vs random {row['random_best_fitness']:.6f} | coverage {row['archive_coverage']}"
                f" vs {row['random_coverage']} | QD {row['qd_score']:.3f} vs {row['random_qd_score']:.3f}"
                f" | {row['runtime_seconds']/60:.1f} min",
                flush=True,
            )
            rows.append(row)
    return rows


def _genome_from_dict(payload: dict[str, Any]) -> Genome:
    return Genome(
        signal_family=payload["signal_family"],
        lookback_bars=int(payload["lookback_bars"]),
        entry_threshold=float(payload["entry_threshold"]),
        stop_pct=float(payload["stop_pct"]),
        target_r=float(payload["target_r"]),
        trail_enabled=bool(payload["trail_enabled"]),
        risk_frac=float(payload["risk_frac"]),
        max_hold_bars=int(payload["max_hold_bars"]),
        allow_short=bool(payload["allow_short"]),
        symbols=tuple(payload["symbols"]),
        max_concurrent=int(payload["max_concurrent"]),
        cooldown_bars_after_loss=int(payload["cooldown_bars_after_loss"]),
        sleeve_fraction=float(payload["sleeve_fraction"]),
    ).validate()


def select_median_seed_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """SPEC.md frozen selection rule: per-seed archive best, then the MEDIAN of those five by
    fitness. Never the best seed -- that is how wave21 talked itself into G1."""
    ordered = sorted(rows, key=lambda row: row["evolved_best_fitness"])
    return ordered[len(ordered) // 2]


def phase_judge(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        rows = [
            json.loads((RESULTS_DIR / f"seed_{seed}.json").read_text(encoding="utf-8"))
            for seed in SEEDS
        ]
    chosen_row = select_median_seed_candidate(rows)
    genome = _genome_from_dict(chosen_row["best"]["genome"])

    cache = build_market_cache()
    rng = np.random.default_rng(30_000)
    # ---- THE single OOS unsealing for this wave ----
    final = final_evaluation(cache, genome, rng)
    result = final.pop("_result")
    final.pop("_summary")

    total_curve = result.total_equity_daily
    baseline_curve = i5_baseline_total_curve(cache)
    verdict = gates30.evaluate_all_gates(
        seed_rows=rows,
        final=final,
        genome=genome,
        total_curve=total_curve,
        baseline_curve=baseline_curve,
        daily_index=cache.daily_index,
        trade_returns=result.trade_returns,
        seed=30_000,
    )

    union_best = max(
        (item for row in rows for item in row["archive"]),
        key=lambda item: item["fitness"],
    )
    payload = {
        "wave": "wave30_qd",
        "method": "MAP-Elites (quality-diversity) + NSGA-II (multi-objective Pareto)",
        "selection_rule": "median-seed archive best (SPEC.md frozen)",
        "chosen_seed": chosen_row["seed"],
        "candidate": chosen_row["best"],
        "baseline_reference_i5": baseline_reference(cache),
        "final_evaluation": final,
        **verdict,
        "union_best_reference_only": union_best,
        "seed_summary": [
            {
                key: row[key]
                for key in (
                    "seed",
                    "evolved_best_fitness",
                    "random_best_fitness",
                    "archive_coverage",
                    "random_coverage",
                    "qd_score",
                    "random_qd_score",
                    "evaluations_evolved",
                    "evaluations_random",
                    "runtime_seconds",
                )
            }
            for row in rows
        ],
        "equity_curve_daily": [
            {"timestamp": str(cache.daily_index[i]), "total_usdt": float(total_curve[i])}
            for i in range(0, len(total_curve), 7)
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "all"
    rows: list[dict[str, Any]] | None = None
    if command in {"search", "all"}:
        rows = phase_search()
    if command in {"judge", "all"}:
        payload = phase_judge(rows)
        print(json.dumps({k: v for k, v in payload.items() if k not in {"equity_curve_daily", "archive"}}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
