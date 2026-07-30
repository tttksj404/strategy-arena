# Wave-34 orchestration.
#
#   python research/wave34_tournament/run_wave34.py probe   [--budget 120]
#   python research/wave34_tournament/run_wave34.py run     [--method cmaes] [--seed 2026341]
#   python research/wave34_tournament/run_wave34.py rl      [--seed 2026341]
#   python research/wave34_tournament/run_wave34.py judge
#
# `run` is RESUMABLE by construction: it writes results/seed_<seed>_<method>.json the instant
# a (method, seed) pair finishes and SKIPS any pair whose file already exists. Re-running the
# bare command after an interruption therefore fills only the holes. Nothing is accumulated in
# memory across pairs -- the judge phase re-reads the files from disk.

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
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
from research.wave30_qd.run_wave30 import _genome_from_dict
from research.wave33_frequency.fitness33 import (
    MIN_TRADES_PER_ACTIVE_DAY,
    TARGET_PER_ENTRY_USDT,
    oos_entry_profile,
)
from research.wave34_tournament.encoding34 import DIM, decode, feasibility_probe
from research.wave34_tournament.fitness34 import BudgetExhausted, Objective
from research.wave34_tournament.optimizers34 import METHODS, RUNNERS
from research.wave34_tournament import rl34

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
REPORT_DIR: Final = Path(__file__).resolve().parent / "report"
SEEDS: Final = (2026341, 2026342, 2026343)
BUDGET_FILE: Final = RESULTS_DIR / "budget.json"
PROBE_FILE: Final = RESULTS_DIR / "probe.json"
RL_EPISODES: Final = 25
MC_PATHS: Final = 10_000
G5_MAX_RUIN: Final = 0.05
G4_OOS_FLOOR_RATIO: Final = 0.0  # preregistered in SPEC.md: OOS total P&L must stay > 0


# ---------------------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------------------


def _probe_one(argument: tuple[str, int]) -> dict[str, Any]:
    method, budget = argument
    cache = build_market_cache()
    objective = Objective(cache, budget=budget)
    started = time.time()
    try:
        RUNNERS[method](objective, np.random.default_rng(99_000))
    except BudgetExhausted:
        pass
    elapsed = time.time() - started
    return {
        "method": method,
        "evaluations": objective.n_evaluations,
        "seconds": elapsed,
        "seconds_per_eval": elapsed / max(1, objective.n_evaluations),
        "best_fitness": objective.best.fitness if objective.best else None,
        "best_feasible_fitness": objective.best_feasible.fitness if objective.best_feasible else None,
        "feasible_found": objective.best_feasible is not None,
    }


def phase_probe(budget: int, workers: int) -> dict[str, Any]:
    """Measure seconds/evaluation per method BEFORE committing to a budget.

    The cost of one evaluation is not a property of the method, it is a property of the
    genomes the method proposes: a genome that survives runs thousands of trades and takes
    ~20x longer than one that dies immediately. So a method that is good at finding survivors
    is genuinely slower per evaluation, and the only way to size the run is to measure it.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe_one, (method, budget)): method for method in METHODS}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"  probe {row['method']:>20}  {row['seconds_per_eval']:.4f} s/eval  "
                  f"({row['evaluations']} evals in {row['seconds']:.1f}s)  feasible={row['feasible_found']}")
    rows.sort(key=lambda r: -r["seconds_per_eval"])
    payload = {
        "probe_budget_per_method": budget,
        "workers": workers,
        "rows": rows,
        "slowest_method": rows[0]["method"],
        "slowest_seconds_per_eval": rows[0]["seconds_per_eval"],
        "sum_seconds_per_eval": sum(r["seconds_per_eval"] for r in rows),
        "encoding_feasibility": feasibility_probe(50_000),
    }
    PROBE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def recommend_budget(probe: dict[str, Any], wall_clock_minutes: float, workers: int, n_seeds: int) -> dict[str, Any]:
    """Choose ONE budget shared by all six methods so the whole tournament fits the ceiling.

    Every (method, seed) pair is one process. When the pool can hold all of them at once the
    wall clock is simply the SLOWEST pair -- budget * slowest_seconds_per_eval -- and MCTS
    sets that number alone. When it cannot, the bound is total serial work over workers. The
    chosen budget is whatever satisfies both.

    A 2.0x safety factor is applied to the probe's measured per-eval cost. This is not padding
    for its own sake: probe evaluations are the FIRST evaluations each method makes, and early
    samples die early and therefore run cheap. Cost per evaluation rises as a method starts
    finding survivors, which is exactly what the good methods do.
    """
    per_eval_sum = probe["sum_seconds_per_eval"]
    slowest = probe["slowest_seconds_per_eval"]
    drift_factor = 2.0
    ceiling = wall_clock_minutes * 60.0

    n_pairs = len(METHODS) * n_seeds
    concurrent = max(1, min(workers, n_pairs))
    waves = int(np.ceil(n_pairs / concurrent))

    bound_slowest_chain = ceiling / (slowest * drift_factor * waves)
    bound_throughput = ceiling * concurrent / (per_eval_sum * n_seeds * drift_factor)
    budget = int(min(bound_slowest_chain, bound_throughput))
    budget = max(200, int(round(budget / 100.0) * 100))
    return {
        "wall_clock_minutes": wall_clock_minutes,
        "drift_factor": drift_factor,
        "workers": workers,
        "n_pairs": n_pairs,
        "concurrent_pairs": concurrent,
        "waves": waves,
        "n_seeds": n_seeds,
        "sum_seconds_per_eval": per_eval_sum,
        "slowest_seconds_per_eval": slowest,
        "budget_bound_slowest_chain": bound_slowest_chain,
        "budget_bound_throughput": bound_throughput,
        "chosen_budget": budget,
        "predicted_wall_clock_seconds": budget * slowest * drift_factor * waves,
    }


# ---------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------


def result_path(seed: int, method: str) -> Path:
    return RESULTS_DIR / f"seed_{seed}_{method}.json"


def _run_pair(argument: tuple[str, int, int, float]) -> dict[str, Any]:
    method, seed, budget, deadline = argument
    path = result_path(seed, method)
    if path.exists():
        return {"method": method, "seed": seed, "skipped": True}

    cache = build_market_cache()
    objective = Objective(cache, budget=budget, deadline_seconds=deadline)
    # Every arm gets the same stream identity: one generator seeded by (seed, method index),
    # so arm A's luck is not arm B's luck but neither is drawn from a privileged seed.
    rng = np.random.default_rng([seed, METHODS.index(method)])
    started = time.time()
    try:
        RUNNERS[method](objective, rng)
    except BudgetExhausted:
        pass
    elapsed = time.time() - started

    payload = {
        "method": method,
        "seed": seed,
        "budget": budget,
        "evaluations": objective.n_evaluations,
        "truncated": bool(objective.truncated),
        "runtime_seconds": elapsed,
        "seconds_per_eval": elapsed / max(1, objective.n_evaluations),
        "n_feasible_found": sum(1 for h in objective.history if np.isfinite(h)),
        "n_distinct_feasible": sum(1 for t in objective._memo.values() if t.feasible),
        "best_fitness": objective.best.fitness if objective.best else None,
        "best_feasible_fitness": objective.best_feasible.fitness if objective.best_feasible else None,
        "best": objective.best.as_dict() if objective.best else None,
        "best_feasible": objective.best_feasible.as_dict() if objective.best_feasible else None,
        "history_stride": 10,
        "history": [float(v) if np.isfinite(v) else None for v in objective.history[::10]],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "method": method,
        "seed": seed,
        "skipped": False,
        "best_feasible_fitness": payload["best_feasible_fitness"],
        "runtime_seconds": elapsed,
        "evaluations": payload["evaluations"],
    }


def phase_run(methods: list[str], seeds: list[int], budget: int, workers: int, deadline: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [(m, s, budget, deadline) for m in methods for s in seeds if not result_path(s, m).exists()]
    print(f"[run] {len(pairs)} pair(s) outstanding, budget={budget}, workers={workers}", flush=True)
    if not pairs:
        return
    # Slowest methods first so the long tail starts early and the pool drains evenly.
    order = {"mcts": 0, "simulated_annealing": 1, "tpe_bayesian": 2, "cmaes": 3, "pso": 4, "random": 5}
    pairs.sort(key=lambda p: order.get(p[0], 9))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in as_completed([pool.submit(_run_pair, p) for p in pairs]):
            done = row.result()
            print(
                f"[done] {done['method']:>20} seed={done['seed']} "
                f"best_feasible={done.get('best_feasible_fitness')} "
                f"evals={done.get('evaluations')} {done.get('runtime_seconds', 0):.0f}s",
                flush=True,
            )


def phase_rl(seeds: list[int], episodes: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = build_market_cache()
    for seed in seeds:
        path = result_path(seed, "rl_qlearning")
        if path.exists():
            continue
        started = time.time()
        q, features = rl34.train_q_learning(cache, seed=seed, episodes=episodes)
        result = rl34.replay_greedy(cache, q, features, is_only=True)
        payload = {
            "method": "rl_qlearning",
            "seed": seed,
            "episodes": episodes,
            "runtime_seconds": time.time() - started,
            "leverage": rl34.RL_LEVERAGE,
            "symbol": rl34.RL_SYMBOL,
            "n_states": rl34.N_STATES,
            "q_nonzero_states": int((np.abs(q).sum(axis=1) > 0).sum()),
            **result.as_dict(),
        }
        payload["episodes"] = episodes  # as_dict() carries the dataclass default; the real
        # episode count lives on the runner, so it is re-stamped after the spread.
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[rl] seed={seed} final=${payload['final_usdt']:.2f} "
              f"changes={payload['n_position_changes']} /day={payload['trades_per_active_day']:.2f} "
              f"{payload['runtime_seconds']:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------------------


def _median_seed_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The seed whose best-feasible fitness is the MEDIAN of that method's seeds. Taking the
    max seed would crown the luckiest run of three and is the single most common way a
    tournament reports a winner that does not reproduce."""
    scored = sorted(rows, key=lambda r: (r["best_feasible_fitness"] is None, r["best_feasible_fitness"] or -1e18))
    return scored[len(scored) // 2]


def _ruin_probability(trade_returns: list[float], base_usdt: float, seed: int) -> dict[str, Any]:
    """Same construction as wave33's F5: resample realised per-entry returns onto a CONSTANT
    base and count paths whose running minimum equity falls below $50."""
    returns = np.asarray(trade_returns, dtype=float)
    if len(returns) == 0:
        return {"ruin_probability": 1.0, "reason": "no trades"}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(returns), size=(MC_PATHS, len(returns)), dtype=np.int32)
    equity = 100.0 + np.cumsum(returns[draws] * base_usdt, axis=1)
    worst = np.minimum.accumulate(equity, axis=1)[:, -1]
    return {
        "paths": MC_PATHS,
        "ruin_probability": float((worst < 50.0).mean()),
        "p05_final_usdt": float(np.percentile(equity[:, -1], 5)),
        "median_final_usdt": float(np.median(equity[:, -1])),
        "base_usdt": base_usdt,
    }


def _concentration(cache, genome) -> dict[str, Any]:
    """How few trades carry the headline number.

    A fixed-base account's final value is a plain SUM of per-trade dollars, so "the strategy
    made $5,000" can mean 2,800 trades each contributing $1.8, or 40 trades contributing
    everything while the other 2,760 bled. Those are completely different objects to trade
    live, and every wave in this campaign that reported a big number without this table had
    the second one. Reported per split.
    """
    from research.wave30_qd.dataio30 import OOS_SPLIT
    from research.wave30_qd.engine30 import run_genome
    from research.wave33_frequency.fitness33 import FIXED_SIZING

    result = run_genome(cache, genome, mode="full", sizing=FIXED_SIZING)
    if not result.trades:
        return {"n_trades": 0}
    pnl = np.array([t.net_return_on_base * t.base_usdt for t in result.trades], dtype=float)
    exit_days = np.array([int(cache.day_of_bar[t.exit_bar]) for t in result.trades])
    oos_day = int(cache.daily_index.searchsorted(OOS_SPLIT, side="right"))

    out: dict[str, Any] = {}
    for label, mask in (("is", exit_days < oos_day), ("oos", exit_days >= oos_day), ("full", np.ones(len(pnl), bool))):
        values = pnl[mask]
        if len(values) == 0:
            out[label] = {"n_trades": 0}
            continue
        total = float(values.sum())
        ordered = np.sort(values)[::-1]
        cumulative = np.cumsum(ordered)
        if total > 0:
            n_half = int(np.searchsorted(cumulative, total * 0.5) + 1)
            n_all = int(np.searchsorted(cumulative, total) + 1)
        else:
            n_half = n_all = -1
        out[label] = {
            "n_trades": int(len(values)),
            "total_usdt": total,
            "top1_usdt": float(ordered[0]),
            "top10_usdt": float(ordered[:10].sum()),
            "top10_share_of_total": float(ordered[:10].sum() / total) if total > 0 else None,
            "n_trades_for_half_of_profit": n_half,
            "n_trades_carrying_all_profit": n_all,
            "share_of_trades_carrying_all_profit": (n_all / len(values)) if n_all > 0 else None,
            "losing_trade_share": float((values < 0).mean()),
        }
    return out


def phase_judge() -> dict[str, Any]:
    per_method: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        rows = []
        for seed in SEEDS:
            path = result_path(seed, method)
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        if rows:
            per_method[method] = rows

    missing = [(m, s) for m in METHODS for s in SEEDS if not result_path(s, m).exists()]

    summary = []
    for method, rows in per_method.items():
        values = [r["best_feasible_fitness"] for r in rows]
        finite = [v for v in values if v is not None]
        median_row = _median_seed_row(rows)
        summary.append({
            "method": method,
            "seeds": [r["seed"] for r in rows],
            "per_seed_best_feasible": values,
            "median_best_feasible": float(np.median(finite)) if finite else None,
            "min_best_feasible": min(finite) if finite else None,
            "max_best_feasible": max(finite) if finite else None,
            "median_seed": median_row["seed"],
            "n_feasible_seeds": len(finite),
            "mean_seconds_per_eval": float(np.mean([r["seconds_per_eval"] for r in rows])),
            "total_runtime_seconds": float(np.sum([r["runtime_seconds"] for r in rows])),
            "evaluations": int(np.sum([r["evaluations"] for r in rows])),
            "any_truncated": any(r.get("truncated") for r in rows),
            "distinct_feasible_found": int(np.sum([r.get("n_distinct_feasible", 0) for r in rows])),
            "median_seed_final_usdt": (median_row.get("best_feasible") or {}).get("final_usdt"),
        })

    # Secondary, always-defined scale: the penalised scalar. It exists so G1 can still be
    # ruled on if NO method (control included) ever reaches the feasible region -- in which
    # case "which method got closest to feasibility" is the only question left, and silently
    # reporting "no result" would hide a real measurement.
    for row in summary:
        rows = per_method[row["method"]]
        penalised = [r["best_fitness"] for r in rows if r["best_fitness"] is not None]
        row["per_seed_best_penalised"] = penalised
        row["median_best_penalised"] = float(np.median(penalised)) if penalised else None
        row["min_best_penalised"] = min(penalised) if penalised else None
        row["max_best_penalised"] = max(penalised) if penalised else None

    control = next((s for s in summary if s["method"] == "random"), None)
    feasible_ranked = sorted(
        [s for s in summary if s["median_best_feasible"] is not None],
        key=lambda s: -s["median_best_feasible"],
    )
    any_feasible = bool(feasible_ranked)
    scale = "best_feasible_fitness" if any_feasible else "best_fitness (penalised; NO arm reached feasibility)"
    ranked = feasible_ranked if any_feasible else sorted(
        [s for s in summary if s["median_best_penalised"] is not None],
        key=lambda s: -s["median_best_penalised"],
    )
    key_median = "median_best_feasible" if any_feasible else "median_best_penalised"
    key_min = "min_best_feasible" if any_feasible else "min_best_penalised"
    key_max = "max_best_feasible" if any_feasible else "max_best_penalised"

    # ---- G1: does method choice matter beyond seed noise? ----
    g1: dict[str, Any] = {"gate": "G1_method_validity", "scale": scale}
    if control is None or control[key_median] is None or not ranked:
        g1.update({"verdict": "FAIL", "reason": "no control result at all"})
    else:
        lo, hi = control[key_min], control[key_max]
        beats = [s["method"] for s in ranked if s["method"] != "random" and s[key_median] > hi]
        g1.update({
            "random_median": control[key_median],
            "random_seed_range": [lo, hi],
            "per_method_median": {s["method"]: s[key_median] for s in ranked},
            "methods_above_random_seed_range": beats,
            "verdict": "PASS" if beats else "FAIL",
            "interpretation": (
                f"{', '.join(beats)} clears the random control's seed-to-seed range"
                if beats
                else "every method's median sits inside the random control's own seed-to-seed "
                     "spread -- method choice does not matter on this problem"
            ),
        })

    # ---- pick the single judged candidate ----
    winner = ranked[0] if (ranked and any_feasible) else None
    verdicts: list[dict[str, Any]] = [g1]
    candidate_payload: dict[str, Any] | None = None

    if not any_feasible:
        # OOS stays sealed. There is nothing to judge: no arm produced a genome that trades
        # >=1x/active-day AND survives the IS span, so G2 fails on its own terms and G3-G5
        # have no candidate to be measured on. Reporting them as FAIL-with-no-candidate is the
        # honest form; inventing a candidate from the least-bad infeasible point would put an
        # account that already died in front of the OOS split.
        for gate in ("G2_frequency_and_survival", "G3_profit_is", "G4_oos", "G5_ruin"):
            verdicts.append({
                "gate": gate,
                "verdict": "FAIL",
                "reason": "no arm found a feasible genome; OOS was not unsealed",
            })

    if winner is not None:
        winner_rows = per_method[winner["method"]]
        median_row = next(r for r in winner_rows if r["seed"] == winner["median_seed"])
        best = median_row["best_feasible"]
        genome = _genome_from_dict(best["genome"])

        cache = build_market_cache()
        # THE single OOS unsealing of this wave.
        profiles, meta = oos_entry_profile(cache, genome)
        ruin = _ruin_probability(meta["trade_returns"], meta["base_usdt"], seed=winner["median_seed"])

        is_profile = profiles["is"]
        oos_profile = profiles["oos"]
        full = profiles["full"]
        years = full["active_days"] / 365.25 if full["active_days"] else float("nan")
        final = best["final_usdt"]
        cagr = (final / 100.0) ** (1.0 / years) - 1.0 if years and years > 0 and final > 0 else float("nan")

        verdicts.append({
            "gate": "G2_frequency_and_survival",
            "verdict": "PASS" if (best["trades_per_active_day"] >= MIN_TRADES_PER_ACTIVE_DAY and best["survived"]) else "FAIL",
            "trades_per_active_day": best["trades_per_active_day"],
            "survived_is": best["survived"],
            "n_trades": best["n_trades"],
        })
        verdicts.append({
            "gate": "G3_profit_is",
            "verdict": "PASS" if final > 100.0 else "FAIL",
            "final_usdt_is": final,
        })
        oos_ok = oos_profile["n_trades"] > 0 and oos_profile["total_usdt"] > G4_OOS_FLOOR_RATIO
        verdicts.append({
            "gate": "G4_oos",
            "verdict": "PASS" if oos_ok else "FAIL",
            "oos": oos_profile,
            "threshold": "OOS total P&L > $0 with at least one OOS trade (preregistered)",
        })
        verdicts.append({
            "gate": "G5_ruin",
            "verdict": "PASS" if ruin["ruin_probability"] < G5_MAX_RUIN else "FAIL",
            **ruin,
            "max_ruin_probability": G5_MAX_RUIN,
        })

        candidate_payload = {
            "method": winner["method"],
            "seed": winner["median_seed"],
            "genome": best["genome"],
            "is": {
                "final_usdt": final,
                "cagr": cagr,
                "years": years,
                "trades_per_active_day": best["trades_per_active_day"],
                "n_trades": best["n_trades"],
                "account_mdd": best["account_mdd"],
                "mean_usdt": best["mean_usdt"],
                "median_usdt": best["median_usdt"],
                "share_ge_target": best["share_ge_target"],
                "leverage": best["leverage"],
                "is_partition": is_profile,
            },
            "oos": oos_profile,
            "full_span": full,
            "meta": {k: v for k, v in meta.items() if k != "trade_returns"},
            "ruin": ruin,
            "concentration": _concentration(cache, genome),
        }

    rl_rows = []
    for seed in SEEDS:
        path = result_path(seed, "rl_qlearning")
        if path.exists():
            rl_rows.append(json.loads(path.read_text(encoding="utf-8")))

    payload = {
        "seeds": list(SEEDS),
        "budget": json.loads(BUDGET_FILE.read_text(encoding="utf-8")) if BUDGET_FILE.exists() else None,
        "probe": json.loads(PROBE_FILE.read_text(encoding="utf-8")) if PROBE_FILE.exists() else None,
        "missing_pairs": [{"method": m, "seed": s} for m, s in missing],
        "summary": summary,
        "ranked": [s["method"] for s in ranked],
        "gates": verdicts,
        "candidate": candidate_payload,
        "rl_arm": rl_rows,
        "overall": "PASS" if all(v.get("verdict") == "PASS" for v in verdicts) else "FAIL",
    }
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall": payload["overall"],
        "ranked": payload["ranked"],
        "g1": g1.get("verdict"),
        "gates": [(v["gate"], v.get("verdict")) for v in verdicts],
    }, indent=2))
    return payload


# ---------------------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="wave-34 search-method tournament")
    parser.add_argument("phase", choices=["probe", "budget", "run", "rl", "judge"])
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--probe-budget", type=int, default=120)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--minutes", type=float, default=90.0)
    parser.add_argument("--episodes", type=int, default=RL_EPISODES)
    parser.add_argument("--deadline", type=float, default=4500.0,
                        help="per-(method,seed) wall-clock backstop in seconds; a pair that hits "
                             "it is recorded as truncated rather than being allowed to stall the wave")
    args = parser.parse_args(argv)

    methods = args.method or list(METHODS)
    seeds = args.seed or list(SEEDS)

    if args.phase == "probe":
        probe = phase_probe(args.probe_budget, workers=min(args.workers, len(METHODS)))
        recommendation = recommend_budget(probe, args.minutes, args.workers, len(seeds))
        print(json.dumps(recommendation, indent=2))
        return 0

    if args.phase == "budget":
        probe = json.loads(PROBE_FILE.read_text(encoding="utf-8"))
        recommendation = recommend_budget(probe, args.minutes, args.workers, len(seeds))
        if args.budget:
            recommendation["chosen_budget"] = args.budget
            recommendation["override_reason"] = "operator-chosen, written to SPEC.md before the run"
        BUDGET_FILE.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
        print(json.dumps(recommendation, indent=2))
        return 0

    if args.phase == "run":
        budget = args.budget
        if budget is None:
            budget = int(json.loads(BUDGET_FILE.read_text(encoding="utf-8"))["chosen_budget"])
        phase_run(methods, seeds, budget, args.workers, args.deadline)
        return 0

    if args.phase == "rl":
        phase_rl(seeds, args.episodes)
        return 0

    phase_judge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
