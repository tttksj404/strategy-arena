# Wave-31 orchestration. Phase 1 searches the sprint objective on IS across five seeds in
# parallel; phase 2 unseals OOS exactly once on the median-seed candidate and runs Q1-Q7.
#
#   python research/wave31_sprint/run_wave31.py search
#   python research/wave31_sprint/run_wave31.py judge
#   python research/wave31_sprint/run_wave31.py all

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
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

from research.wave30_qd.dataio30 import OOS_SPLIT, build_market_cache, i5_baseline_total_curve
from research.wave30_qd.engine30 import annualized_return, max_drawdown, run_genome
from research.wave30_qd.fitness30 import Evaluation
from research.wave30_qd.run_wave30 import _genome_from_dict
from research.wave30_qd.search30 import (
    Evaluator,
    run_map_elites,
    run_nsga2,
    run_random_search,
)
from research.wave31_sprint import gates31
from research.wave31_sprint.fitness31 import (
    FITNESS_WINDOW,
    GRID_SHAPE,
    WINDOWS,
    baseline_sprint_profile,
    evaluate_sprint,
    sprint_profile,
    summarise_sprint,
)

SEEDS: Final = (2026311, 2026312, 2026313, 2026314, 2026315)
MAP_ELITES_INIT: Final = 300
MAP_ELITES_ITERATIONS: Final = 2_700
NSGA_POPULATION: Final = 100
NSGA_GENERATIONS: Final = 12
RANDOM_BUDGET: Final = MAP_ELITES_INIT + MAP_ELITES_ITERATIONS + NSGA_POPULATION * NSGA_GENERATIONS
RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"


def _evaluation_json(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "genome": evaluation.genome.to_dict(),
        "fitness": evaluation.fitness,
        "is_total_cagr": evaluation.is_total_cagr,
        "is_total_final_usdt": evaluation.is_total_final,
        "sleeve_mdd": evaluation.sleeve_mdd,
        "total_mdd": evaluation.total_mdd,
        "trades_per_year": evaluation.trades_per_year,
        "n_trades": evaluation.n_trades,
        "n_liquidations": evaluation.n_liquidations,
        "wipe_probability": evaluation.wipe_probability,
        "prob_halving_30d": evaluation.extras["prob_halving_30d"],
        "descriptor": list(evaluation.descriptor),
        "mean_leverage": evaluation.mean_leverage,
        "min_notional_usdt": evaluation.min_notional_usdt,
        "sleeve_survived": evaluation.sleeve_survived,
        "sprint": evaluation.extras["sprint"],
    }


def run_one_seed(seed: int) -> dict[str, Any]:
    started = time.time()
    cache = build_market_cache()

    evolved = Evaluator(cache, seed=seed, evaluate_fn=evaluate_sprint)
    archive = run_map_elites(
        evolved, np.random.default_rng(seed), n_init=MAP_ELITES_INIT, n_iterations=MAP_ELITES_ITERATIONS
    )
    map_elites_evaluations = evolved.n_evaluations
    pareto = run_nsga2(
        evolved,
        np.random.default_rng(seed + 500_000),
        population_size=NSGA_POPULATION,
        generations=NSGA_GENERATIONS,
    )
    for evaluation in pareto:
        archive.consider(evaluation)

    control = Evaluator(cache, seed=seed + 900_000, evaluate_fn=evaluate_sprint)
    random_archive = run_random_search(
        control, np.random.default_rng(seed + 900_000), budget=RANDOM_BUDGET
    )

    best = archive.best
    random_best = random_archive.best
    return {
        "seed": seed,
        "runtime_seconds": time.time() - started,
        "evaluations_evolved": evolved.n_evaluations,
        "evaluations_map_elites": map_elites_evaluations,
        "evaluations_nsga2": evolved.n_evaluations - map_elites_evaluations,
        "evaluations_random": control.n_evaluations,
        "archive_coverage": archive.coverage,
        "random_coverage": random_archive.coverage,
        "qd_score": archive.qd_score,
        "random_qd_score": random_archive.qd_score,
        "evolved_best_fitness": best.fitness if best else float("-inf"),
        "random_best_fitness": random_best.fitness if random_best else float("-inf"),
        "best": _evaluation_json(best) if best else None,
        "random_best": _evaluation_json(random_best) if random_best else None,
        "archive": [_evaluation_json(item) for item in archive.elites()],
        "pareto_front": [_evaluation_json(item) for item in pareto],
        "grid_shape": list(GRID_SHAPE),
    }


def phase_search() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        for row in pool.map(run_one_seed, SEEDS):
            (RESULTS_DIR / f"seed_{row['seed']}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(
                f"seed {row['seed']}: evolved median-30d {row['evolved_best_fitness']*100:+.2f}% "
                f"vs random {row['random_best_fitness']*100:+.2f}% | coverage {row['archive_coverage']}"
                f" vs {row['random_coverage']} | QD {row['qd_score']:.3f} vs {row['random_qd_score']:.3f}"
                f" | {row['runtime_seconds']/60:.1f} min",
                flush=True,
            )
            rows.append(row)
    return rows


def select_median_seed_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["evolved_best_fitness"])
    return ordered[len(ordered) // 2]


def phase_judge(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        rows = [json.loads((RESULTS_DIR / f"seed_{s}.json").read_text(encoding="utf-8")) for s in SEEDS]
    chosen = select_median_seed_candidate(rows)
    genome = _genome_from_dict(chosen["best"]["genome"])

    cache = build_market_cache()
    rng = np.random.default_rng(31_000)
    # ---- THE single OOS unsealing for this wave ----
    result = run_genome(cache, genome, mode="full")
    summary = summarise_sprint(cache, genome, result, rng)

    total = result.total_equity_daily
    baseline = i5_baseline_total_curve(cache)
    oos_start = int(cache.daily_index.searchsorted(OOS_SPLIT, side="right"))
    is_end = max(0, oos_start - 1)

    candidate_profiles = {
        "is": sprint_profile(total[: is_end + 1]),
        "oos": sprint_profile(total[oos_start:]),
        "full": sprint_profile(total),
    }
    baseline_profiles = baseline_sprint_profile(cache)

    def window(curve: np.ndarray, start: int, end: int) -> dict:
        segment = curve[start : end + 1]
        span = float(end - start)
        return {
            "start_usdt": float(segment[0]),
            "end_usdt": float(segment[-1]),
            "days": span,
            "annualized": annualized_return(segment, span),
            "mdd": float(abs(max_drawdown(segment))),
        }

    last = len(total) - 1
    verdict = gates31.evaluate_all_gates(
        seed_rows=rows,
        genome=genome,
        total_curve=total,
        daily_index=cache.daily_index,
        trade_returns=result.trade_returns,
        candidate_oos_profile=candidate_profiles["oos"],
        baseline_oos_profile=baseline_profiles["oos"],
        min_notional_usdt=summary.min_notional_usdt,
        n_trades=summary.n_trades,
        seed=31_000,
    )

    payload = {
        "wave": "wave31_sprint",
        "objective": f"median rolling {FITNESS_WINDOW}-day total-system return (IS only)",
        "method": "MAP-Elites + NSGA-II (wave30 engine, unmodified)",
        "selection_rule": "median-seed archive best (SPEC.md frozen)",
        "chosen_seed": chosen["seed"],
        "candidate": chosen["best"],
        "candidate_sprint_profiles": candidate_profiles,
        "baseline_sprint_profiles": baseline_profiles,
        "equity_windows": {
            "is": window(total, 0, is_end),
            "oos": window(total, oos_start, last),
            "full": window(total, 0, last),
            "baseline_is": window(baseline, 0, is_end),
            "baseline_oos": window(baseline, oos_start, last),
            "baseline_full": window(baseline, 0, last),
        },
        "candidate_trade_stats": {
            "n_trades_full": summary.n_trades,
            "n_liquidations_full": summary.n_liquidations,
            "mean_leverage": summary.mean_leverage,
            "min_notional_usdt": summary.min_notional_usdt,
            "max_notional_usdt": float(max((t.notional_usdt for t in result.trades), default=float("nan"))),
            "sleeve_survived": summary.sleeve_survived,
            "max_mae": float(max((t.mae for t in result.trades), default=float("nan"))),
            "liquidation_band": genome.liquidation_band,
        },
        **verdict,
        "seed_summary": [
            {
                k: row[k]
                for k in (
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
        "windows_frozen": list(WINDOWS),
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
        print(f"\nchosen seed {payload['chosen_seed']} | overall {payload['overall']} | "
              f"failures {payload['failure_reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
