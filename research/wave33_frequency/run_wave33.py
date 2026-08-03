# Wave-33 orchestration: search the frequency-constrained per-entry objective on IS, then
# unseal OOS once on the median-seed candidate and rule on F1-F6.
#
#   python research/wave33_frequency/run_wave33.py search
#   python research/wave33_frequency/run_wave33.py judge
#   python research/wave33_frequency/run_wave33.py all

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

from research.wave30_qd.dataio30 import build_market_cache
from research.wave30_qd.fitness30 import Evaluation
from research.wave30_qd.gates30 import GateOutcome, gate_p6_executability
from research.wave30_qd.genome30 import LEV_CAP, STOP_BAND_MARGIN, Genome
from research.wave30_qd.run_wave30 import _genome_from_dict
from research.wave30_qd.search30 import Evaluator, run_map_elites, run_nsga2, run_random_search
from research.wave33_frequency.fitness33 import (
    GRID_SHAPE,
    MIN_TRADES_PER_ACTIVE_DAY,
    TARGET_PER_ENTRY_USDT,
    evaluate_frequency,
    oos_entry_profile,
)

SEEDS: Final = (2026331, 2026332, 2026333, 2026334, 2026335)
MAP_ELITES_INIT: Final = 300
MAP_ELITES_ITERATIONS: Final = 2_200
NSGA_POPULATION: Final = 120
NSGA_GENERATIONS: Final = 12
RANDOM_BUDGET: Final = MAP_ELITES_INIT + MAP_ELITES_ITERATIONS + NSGA_POPULATION * NSGA_GENERATIONS
CUMULATIVE_TRIALS: Final = 295_081
F5_MAX_RUIN: Final = 0.05
MC_PATHS: Final = 10_000
RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"


def _evaluation_json(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "genome": evaluation.genome.to_dict(),
        "fitness": evaluation.fitness,
        "descriptor": list(evaluation.descriptor),
        "mean_leverage": evaluation.mean_leverage,
        "n_trades": evaluation.n_trades,
        "n_liquidations": evaluation.n_liquidations,
        "min_notional_usdt": evaluation.min_notional_usdt,
        "account_mdd": evaluation.sleeve_mdd,
        "entry_profile": evaluation.extras["entry_profile"],
        "infeasible_reasons": evaluation.extras["infeasible_reasons"],
    }


def run_one_seed(seed: int) -> dict[str, Any]:
    started = time.time()
    cache = build_market_cache()

    evolved = Evaluator(cache, seed=seed, evaluate_fn=evaluate_frequency)
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

    control = Evaluator(cache, seed=seed + 900_000, evaluate_fn=evaluate_frequency)
    random_archive = run_random_search(control, np.random.default_rng(seed + 900_000), budget=RANDOM_BUDGET)

    best = archive.best
    random_best = random_archive.best
    # Best genome that actually satisfies the request's frequency+survival constraint.
    feasible = [item for item in archive.cells.values() if not item.extras["infeasible_reasons"]]
    best_feasible = max(feasible, key=lambda item: item.fitness) if feasible else None
    return {
        "seed": seed,
        "runtime_seconds": time.time() - started,
        "evaluations_evolved": evolved.n_evaluations,
        "evaluations_map_elites": map_elites_evaluations,
        "evaluations_random": control.n_evaluations,
        "archive_coverage": archive.coverage,
        "random_coverage": random_archive.coverage,
        "qd_score": archive.qd_score,
        "random_qd_score": random_archive.qd_score,
        "evolved_best_fitness": best.fitness if best else float("-inf"),
        "random_best_fitness": random_best.fitness if random_best else float("-inf"),
        "feasible_count": len(feasible),
        "best": _evaluation_json(best) if best else None,
        "best_feasible": _evaluation_json(best_feasible) if best_feasible else None,
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
            feasible = row["best_feasible"]
            text = (
                f"진입당 중앙값 ${feasible['entry_profile']['median_usdt']:+.3f} "
                f"({feasible['entry_profile']['trades_per_active_day']:.2f}회/활동일)"
                if feasible else "제약 만족 개체 없음"
            )
            print(
                f"seed {row['seed']}: 진화최고 {row['evolved_best_fitness']:+.3f} vs 랜덤 "
                f"{row['random_best_fitness']:+.3f} | 커버리지 {row['archive_coverage']}/{row['random_coverage']} "
                f"| 제약충족 {row['feasible_count']} | {text} | {row['runtime_seconds']/60:.1f}분",
                flush=True,
            )
            rows.append(row)
    return rows


def select_median_seed_candidate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """SPEC.md rule: per-seed archive best, then the median of the five by fitness.

    Preference is given to each seed's best FEASIBLE genome (one satisfying the request's
    frequency and survival conditions). If no seed produced a feasible genome at all, the rule
    falls back to the unconstrained best and the report must say so -- silently judging an
    infeasible genome against F2/F3 would misrepresent what was actually searched.
    """
    feasible_rows = [row for row in rows if row.get("best_feasible")]
    if feasible_rows:
        ordered = sorted(feasible_rows, key=lambda row: row["best_feasible"]["fitness"])
        return ordered[len(ordered) // 2], "best_feasible"
    ordered = sorted(rows, key=lambda row: row["evolved_best_fitness"])
    return ordered[len(ordered) // 2], "best"


def gate_f1(rows: list[dict[str, Any]]) -> GateOutcome:
    fitness_wins = sum(1 for r in rows if r["evolved_best_fitness"] > r["random_best_fitness"])
    coverage_wins = sum(1 for r in rows if r["archive_coverage"] > r["random_coverage"])
    ok = fitness_wins >= 4 and coverage_wins >= 4
    return GateOutcome(
        "F1_method_validity",
        "PASS" if ok else "FAIL",
        {
            "fitness_wins": fitness_wins,
            "coverage_wins": coverage_wins,
            "required": 4,
            "per_seed": [
                {
                    "seed": r["seed"],
                    "evolved": r["evolved_best_fitness"],
                    "random": r["random_best_fitness"],
                    "coverage": r["archive_coverage"],
                    "random_coverage": r["random_coverage"],
                    "feasible_count": r["feasible_count"],
                }
                for r in rows
            ],
        },
    )


def gate_f2(profile: dict) -> GateOutcome:
    reasons = []
    if profile["trades_per_active_day"] < MIN_TRADES_PER_ACTIVE_DAY:
        reasons.append(f"{profile['trades_per_active_day']:.3f} entries/active-day < {MIN_TRADES_PER_ACTIVE_DAY}")
    if not profile["survived_full_span"]:
        reasons.append("account died before the IS span ended")
    return GateOutcome(
        "F2_frequency_and_survival",
        "PASS" if not reasons else "FAIL",
        {
            "trades_per_active_day": profile["trades_per_active_day"],
            "minimum": MIN_TRADES_PER_ACTIVE_DAY,
            "n_trades": profile["n_trades"],
            "active_days": profile["active_days"],
            "survived_full_span": profile["survived_full_span"],
            "reasons": reasons,
        },
    )


def gate_f3(profile: dict) -> GateOutcome:
    median = profile["median_usdt"]
    return GateOutcome(
        "F3_per_entry_target",
        "PASS" if median >= TARGET_PER_ENTRY_USDT else "FAIL",
        {
            "median_usdt_per_entry": median,
            "target_usdt": TARGET_PER_ENTRY_USDT,
            "shortfall_usdt": TARGET_PER_ENTRY_USDT - median,
            "mean_usdt_per_entry": profile["mean_usdt"],
            "p95_usdt_per_entry": profile["p95_usdt"],
            "best_usdt": profile["best_usdt"],
            "share_ge_target": profile["share_ge_target"],
            "share_le_negative_target": profile["share_le_negative_target"],
            "capital_for_target_ev": profile["capital_for_target_ev"],
        },
    )


def gate_f4(oos: dict) -> GateOutcome:
    reasons = []
    if oos["n_trades"] == 0:
        reasons.append("no OOS trades")
    else:
        if oos["mean_usdt"] <= 0.0:
            reasons.append(f"OOS mean per entry {oos['mean_usdt']:+.4f} <= 0")
        if oos["median_usdt"] <= 0.0:
            reasons.append(f"OOS median per entry {oos['median_usdt']:+.4f} <= 0")
    return GateOutcome("F4_oos_per_entry", "PASS" if not reasons else "FAIL", {**oos, "reasons": reasons})


def gate_f5(trade_returns: list[float], base_usdt: float, seed: int) -> GateOutcome:
    """Fixed-size ruin: resample the realised per-entry returns, apply them to a CONSTANT base,
    and count paths whose account drops below $50. Constant base is the point -- under fixed
    sizing a bad run does not shrink its bets, so ruin arrives faster than under compounding."""
    returns = np.asarray(trade_returns, dtype=float)
    if len(returns) == 0:
        return GateOutcome("F5_ruin", "FAIL", {"reason": "no trades"})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(returns), size=(MC_PATHS, len(returns)), dtype=np.int32)
    pnl = returns[draws] * base_usdt
    equity = 100.0 + np.cumsum(pnl, axis=1)
    worst = np.minimum.accumulate(equity, axis=1)[:, -1]
    ruin = float((worst < 50.0).mean())
    return GateOutcome(
        "F5_ruin",
        "PASS" if ruin < F5_MAX_RUIN else "FAIL",
        {
            "paths": MC_PATHS,
            "ruin_probability": ruin,
            "max_ruin_probability": F5_MAX_RUIN,
            "p05_final_usdt": float(np.percentile(equity[:, -1], 5)),
            "median_final_usdt": float(np.median(equity[:, -1])),
            "base_usdt": base_usdt,
        },
    )


def phase_judge(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        rows = [json.loads((RESULTS_DIR / f"seed_{s}.json").read_text(encoding="utf-8")) for s in SEEDS]
    chosen_row, source = select_median_seed_candidate(rows)
    candidate = chosen_row[source]
    genome = _genome_from_dict(candidate["genome"])

    cache = build_market_cache()
    profiles, meta = oos_entry_profile(cache, genome)  # THE single OOS unsealing

    outcomes = [
        gate_f1(rows),
        gate_f2(candidate["entry_profile"]),
        gate_f3(candidate["entry_profile"]),
        gate_f4(profiles["oos"]),
        gate_f5(meta["trade_returns"], meta["base_usdt"], 33_000),
        gate_p6_executability(genome, meta["min_notional_usdt"], profiles["full"]["n_trades"]),
    ]
    gates: dict[str, Any] = {}
    failures: list[str] = []
    for outcome in outcomes:
        name = "F6_executability" if outcome.name == "P6_executability" else outcome.name
        detail = dict(outcome.detail)
        status = outcome.status
        if name == "F6_executability" and meta["n_liquidations"] != 0:
            status = "FAIL"
            detail["liquidations"] = meta["n_liquidations"]
        gates[name] = {"status": status, **detail}
        if status != "PASS":
            failures.append(name)

    payload = {
        "wave": "wave33_frequency",
        "request": "at least $10 net per entry, at least one entry per day, $100 account, <=20x",
        "objective": "median per-entry USDT P&L at fixed $100 sizing (IS only)",
        "candidate_source": source,
        "chosen_seed": chosen_row["seed"],
        "candidate": candidate,
        "entry_profiles": profiles,
        "candidate_meta": {k: v for k, v in meta.items() if k != "trade_returns"},
        "cumulative_trials": CUMULATIVE_TRIALS,
        "oos_note": "third use of the 2025-10-01~ window (wave31 candidate C, wave32 L1, wave33 this)",
        "gates": gates,
        "overall": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "promoted": not failures,
        "seed_summary": [
            {k: r[k] for k in ("seed", "evolved_best_fitness", "random_best_fitness", "archive_coverage",
                               "random_coverage", "qd_score", "random_qd_score", "feasible_count",
                               "runtime_seconds")}
            for r in rows
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\ncandidate source: {source} | seed {chosen_row['seed']}")
    ip = candidate["entry_profile"]
    print(f"IS  진입당 중앙값 ${ip['median_usdt']:+.3f} | 평균 ${ip['mean_usdt']:+.3f} | "
          f"{ip['trades_per_active_day']:.2f}회/활동일 | {ip['n_trades']}거래 | "
          f"P(≥+$10) {ip['share_ge_target']:.1%} | P(≤−$10) {ip['share_le_negative_target']:.1%}")
    o = profiles["oos"]
    print(f"OOS 진입당 중앙값 ${o['median_usdt']:+.3f} | 평균 ${o['mean_usdt']:+.3f} | {o['n_trades']}거래")
    print(f"진입당 기대값 $10에 필요한 자본: ${ip['capital_for_target_ev']:,.0f}" if np.isfinite(ip["capital_for_target_ev"])
          else "진입당 기대값 $10에 필요한 자본: 도달 불가 (기대값 ≤ 0 — 자본을 키워도 손실만 커진다)")
    for name, body in gates.items():
        print(f"[{body['status']}] {name}")
    print(f"\nOVERALL {payload['overall']} | failures {failures}")
    return payload


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "all"
    rows: list[dict[str, Any]] | None = None
    if command in {"search", "all"}:
        rows = phase_search()
    if command in {"judge", "all"}:
        phase_judge(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
