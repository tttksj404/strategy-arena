# Wave-34 orchestration: run the six-method tournament and the RL arm on IS, then unseal OOS once
# on the winning method's median-seed candidate and rule on gates G1-G6.
#
#   python research/wave34_tournament/run_wave34.py tournament
#   python research/wave34_tournament/run_wave34.py rl
#   python research/wave34_tournament/run_wave34.py judge
#   python research/wave34_tournament/run_wave34.py all

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
from research.wave30_qd.engine30 import COMPOUNDING, annualized_return, max_drawdown, run_genome
from research.wave30_qd.gates30 import GateOutcome, _daily_returns, gate_p6_executability
from research.wave30_qd.run_wave30 import _genome_from_dict
from research.wave33_frequency.fitness33 import entry_profile
from research.wave34_tournament.encoding34 import (
    CAPACITY_LIMIT_USDT,
    MIN_TRADES_PER_ACTIVE_DAY,
    objective,
)
from research.wave34_tournament.optimizers34 import METHODS, Budget

SEEDS: Final = (2026341, 2026342, 2026343, 2026344, 2026345)
BUDGET_PER_METHOD: Final = 3_000
CUMULATIVE_TRIALS: Final = 385_081
G4_MAX_RUIN: Final = 0.05
MC_PATHS: Final = 10_000
RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"


def _evaluation_json(evaluation) -> dict[str, Any]:
    return {
        "genome": evaluation.genome.to_dict(),
        "vector": evaluation.extras["vector"],
        "fitness": evaluation.fitness,
        "log_growth": evaluation.extras["log_growth"],
        "is_total_final_usdt": evaluation.is_total_final,
        "account_mdd": evaluation.sleeve_mdd,
        "n_trades": evaluation.n_trades,
        "n_liquidations": evaluation.n_liquidations,
        "mean_leverage": evaluation.mean_leverage,
        "min_notional_usdt": evaluation.min_notional_usdt,
        "max_notional_usdt": evaluation.extras["max_notional_usdt"],
        "infeasible_reasons": evaluation.extras["infeasible_reasons"],
        "entry_profile": evaluation.extras["entry_profile"],
    }


def run_one_seed(task: tuple[int, tuple[str, ...]]) -> dict[str, Any]:
    """Run the requested methods on one seed and MERGE into that seed's file if it exists.

    Methods are fully independent given (seed, budget, objective), so which ones run in which OS
    process changes nothing about the experiment -- only wall-clock scheduling. Splitting is
    necessary in practice because MCTS costs ~20x per evaluation what CMA-ES does: not from tree
    overhead but because MCTS actually finds FEASIBLE genomes, and a feasible genome trades
    thousands of times and is therefore expensive to simulate. Budget stays at the frozen 3,000
    evaluations per method regardless of how the run is sliced.
    """
    seed, requested = task
    cache = build_market_cache()
    started = time.time()
    existing_path = RESULTS_DIR / f"seed_{seed}.json"
    out: dict[str, Any] = (
        json.loads(existing_path.read_text(encoding="utf-8"))
        if existing_path.exists()
        else {"seed": seed, "methods": {}}
    )
    for name in requested:
        method = METHODS[name]
        budget = Budget(lambda vector: objective(cache, vector), limit=BUDGET_PER_METHOD)
        method_started = time.time()
        method(budget, np.random.default_rng(seed))
        best = budget.best()
        feasible = [item for item in budget.history if not item.extras["infeasible_reasons"]]
        best_feasible = max(feasible, key=lambda item: item.fitness) if feasible else None
        out["methods"][name] = {
            "spent": budget.spent,
            "runtime_seconds": time.time() - method_started,
            "best_fitness": best.fitness if best else float("-inf"),
            "best": _evaluation_json(best) if best else None,
            "feasible_count": len(feasible),
            "best_feasible_fitness": best_feasible.fitness if best_feasible else float("-inf"),
            "best_feasible": _evaluation_json(best_feasible) if best_feasible else None,
        }
    out["runtime_seconds"] = time.time() - started
    return out


def load_seed_rows() -> list[dict[str, Any]]:
    """Read whatever seeds are already on disk. Lets the tournament be executed in separate
    invocations (one or two seeds at a time) without changing the frozen budget: each seed's run
    is completely independent, so splitting execution changes nothing about the experiment."""
    rows = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"seed_{seed}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def phase_tournament(
    seeds: tuple[int, ...] = SEEDS, methods: tuple[str, ...] = tuple(METHODS)
) -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    tasks = [(seed, methods) for seed in seeds]
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        for row in pool.map(run_one_seed, tasks):
            (RESULTS_DIR / f"seed_{row['seed']}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            text = " | ".join(
                f"{name[:6]} {row['methods'][name]['best_fitness']:+.3f}"
                f"({row['methods'][name]['feasible_count']})"
                for name in methods
            )
            print(f"seed {row['seed']} [{row['runtime_seconds']/60:.1f}분] {text}", flush=True)
            rows.append(row)
    return rows


def missing_work() -> dict[int, list[str]]:
    """Which (seed, method) pairs still have no result. Lets execution resume after an
    interruption without redoing finished work."""
    outstanding: dict[int, list[str]] = {}
    for seed in SEEDS:
        path = RESULTS_DIR / f"seed_{seed}.json"
        done = set(json.loads(path.read_text(encoding="utf-8"))["methods"]) if path.exists() else set()
        remaining = [name for name in METHODS if name not in done]
        if remaining:
            outstanding[seed] = remaining
    return outstanding


def phase_rl() -> dict[str, Any]:
    from research.wave34_tournament.rl34 import run_rl_arm

    cache = build_market_cache()
    started = time.time()
    payload = run_rl_arm(cache, SEEDS)
    payload["runtime_seconds"] = time.time() - started
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rl_arm.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRL arm ({payload['runtime_seconds']/60:.1f}분, {payload['n_states']} states x {payload['n_actions']} actions)")
    print(f"{'lev':>5} {'IS 배수(중앙)':>14} {'IS 최고':>10} {'OOS 배수(중앙)':>15} {'OOS 최고':>10} "
          f"{'IS 진입/일':>11} {'OOS 진입/일':>12} {'청산':>6}")
    for label, row in payload["summary"].items():
        print(f"{label:>5} {row['is_growth_median']:13.3f} {row['is_growth_best']:9.3f} "
              f"{row['oos_growth_median']:14.3f} {row['oos_growth_best']:9.3f} "
              f"{row['is_entries_per_day_median']:10.2f} {row['oos_entries_per_day_median']:11.2f} "
              f"{row['liquidations_total']:6d}")
    return payload


def select_winner(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, float]]:
    """SPEC.md rule: per method, take the MEDIAN across seeds of that seed's best fitness; the
    method with the highest median wins; its median-seed genome is the judged candidate.

    Ranking uses each seed's best FEASIBLE fitness where one exists, because an infeasible best
    (fitness carrying the -100 penalty) is not a candidate for anything -- ranking on it would
    let a method win by finding a slightly-less-terrible violation.
    """
    medians: dict[str, float] = {}
    for name in METHODS:
        scores = [row["methods"][name]["best_feasible_fitness"] for row in rows]
        medians[name] = float(np.median(scores))
    winner = max(medians, key=lambda name: medians[name])
    ordered = sorted(rows, key=lambda row: row["methods"][winner]["best_feasible_fitness"])
    return winner, ordered[len(ordered) // 2], medians


def gate_g1(rows: list[dict[str, Any]]) -> GateOutcome:
    detail: dict[str, Any] = {"per_method": {}}
    for name in METHODS:
        if name == "random":
            continue
        wins = sum(
            1
            for row in rows
            if row["methods"][name]["best_feasible_fitness"] > row["methods"]["random"]["best_feasible_fitness"]
        )
        detail["per_method"][name] = {
            "seed_wins_over_random": wins,
            "required": 4,
            "beats_random": wins >= 4,
            "median_best_feasible": float(np.median([r["methods"][name]["best_feasible_fitness"] for r in rows])),
        }
    detail["random_median_best_feasible"] = float(
        np.median([r["methods"]["random"]["best_feasible_fitness"] for r in rows])
    )
    any_beat = any(item["beats_random"] for item in detail["per_method"].values())
    # G1 is a MEASUREMENT, not a promotion condition (SPEC.md), so its status records what was
    # observed rather than blocking the wave.
    return GateOutcome("G1_method_superiority", "PASS" if any_beat else "FAIL", detail)


def gate_g2(profile: dict) -> GateOutcome:
    reasons = []
    if profile["trades_per_active_day"] < MIN_TRADES_PER_ACTIVE_DAY:
        reasons.append(f"{profile['trades_per_active_day']:.3f} entries/active-day < {MIN_TRADES_PER_ACTIVE_DAY}")
    if not profile["survived_full_span"]:
        reasons.append("account died before the span ended")
    return GateOutcome(
        "G2_frequency_and_survival",
        "PASS" if not reasons else "FAIL",
        {
            "trades_per_active_day": profile["trades_per_active_day"],
            "n_trades": profile["n_trades"],
            "active_days": profile["active_days"],
            "survived_full_span": profile["survived_full_span"],
            "reasons": reasons,
        },
    )


def gate_g3(candidate_oos: dict, baseline_oos: dict) -> GateOutcome:
    reasons = []
    if candidate_oos["total_return"] <= 0.0:
        reasons.append(f"OOS total return {candidate_oos['total_return']:+.4f} <= 0")
    if candidate_oos["annualized"] <= baseline_oos["annualized"]:
        reasons.append(
            f"OOS annualised {candidate_oos['annualized']:+.4f} <= I5 baseline {baseline_oos['annualized']:+.4f}"
        )
    return GateOutcome(
        "G3_oos",
        "PASS" if not reasons else "FAIL",
        {"candidate_oos": candidate_oos, "baseline_oos": baseline_oos, "reasons": reasons},
    )


def gate_g4(total_curve: np.ndarray, seed: int) -> GateOutcome:
    rng = np.random.default_rng(seed)
    daily = _daily_returns(total_curve)
    draws = rng.integers(0, len(daily), size=(MC_PATHS, len(daily)))
    finals = (100.0 * np.cumprod(1.0 + daily[draws], axis=1))[:, -1]
    ruin = float((finals < 50.0).mean())
    return GateOutcome(
        "G4_ruin",
        "PASS" if ruin < G4_MAX_RUIN else "FAIL",
        {
            "paths": MC_PATHS,
            "ruin_probability": ruin,
            "max_ruin_probability": G4_MAX_RUIN,
            "p05_usdt": float(np.percentile(finals, 5)),
            "median_usdt": float(np.median(finals)),
        },
    )


def gate_g6_capacity(max_notional: float) -> GateOutcome:
    return GateOutcome(
        "G6_capacity",
        "PASS" if max_notional <= CAPACITY_LIMIT_USDT else "FAIL",
        {
            "max_notional_usdt": max_notional,
            "limit_usdt": CAPACITY_LIMIT_USDT,
            "cost_model_fitted_order_usdt": 45.0,
            "extrapolation_multiple": max_notional / 45.0 if max_notional else 0.0,
        },
    )


def phase_judge(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        rows = [json.loads((RESULTS_DIR / f"seed_{s}.json").read_text(encoding="utf-8")) for s in SEEDS]
    winner, chosen_row, medians = select_winner(rows)
    candidate = chosen_row["methods"][winner]["best_feasible"]
    if candidate is None:
        payload = {
            "wave": "wave34_tournament",
            "winner_method": winner,
            "method_medians": medians,
            "overall": "FAIL",
            "failure_reasons": ["no method produced a feasible candidate on any seed"],
            "promoted": False,
            "gates": {"G1_method_superiority": {**gate_g1(rows).detail, "status": gate_g1(rows).status}},
        }
        (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("\n어떤 기법도 제약을 만족하는 후보를 내지 못했다 -> 기각")
        return payload

    genome = _genome_from_dict(candidate["genome"])
    cache = build_market_cache()
    result = run_genome(cache, genome, mode="full", sizing=COMPOUNDING)  # THE single OOS unsealing
    total = result.total_equity_daily
    baseline = i5_baseline_total_curve(cache)
    oos_start = int(cache.daily_index.searchsorted(OOS_SPLIT, side="right"))
    last = len(total) - 1

    def window(curve: np.ndarray, start: int, end: int) -> dict:
        segment = curve[start : end + 1]
        span = float(end - start)
        return {
            "start_usdt": float(segment[0]),
            "end_usdt": float(segment[-1]),
            "days": span,
            "total_return": float(segment[-1] / segment[0] - 1.0),
            "annualized": annualized_return(segment, span),
            "mdd": float(abs(max_drawdown(segment))),
        }

    full_profile = entry_profile(cache, result)
    max_notional = float(max((t.notional_usdt for t in result.trades), default=0.0))

    outcomes = [
        gate_g1(rows),
        gate_g2(candidate["entry_profile"]),
        gate_g3(window(total, oos_start, last), window(baseline, oos_start, last)),
        gate_g4(total, 34_000),
        gate_p6_executability(
            genome,
            result.min_notional_usdt if full_profile.n_trades else float("nan"),
            full_profile.n_trades,
        ),
        gate_g6_capacity(max_notional),
    ]
    gates: dict[str, Any] = {}
    failures: list[str] = []
    for outcome in outcomes:
        name = "G5_executability" if outcome.name == "P6_executability" else outcome.name
        detail = dict(outcome.detail)
        status = outcome.status
        if name == "G5_executability" and result.n_liquidations != 0:
            status = "FAIL"
            detail["liquidations"] = result.n_liquidations
        gates[name] = {"status": status, **detail}
        # G1 is a measurement, never a promotion blocker (SPEC.md).
        if status != "PASS" and name != "G1_method_superiority":
            failures.append(name)

    payload = {
        "wave": "wave34_tournament",
        "request": "$100 fixed, >=1 entry per day, maximise profit",
        "objective": "log(IS final total capital / 100) with a -100 penalty for violating frequency/survival",
        "winner_method": winner,
        "method_medians": medians,
        "chosen_seed": chosen_row["seed"],
        "candidate": candidate,
        "equity_windows": {
            "is": window(total, 0, max(0, oos_start - 1)),
            "oos": window(total, oos_start, last),
            "full": window(total, 0, last),
            "baseline_oos": window(baseline, oos_start, last),
            "baseline_full": window(baseline, 0, last),
        },
        "full_span_entry_profile": full_profile.as_dict(),
        "max_notional_usdt": max_notional,
        "cumulative_trials": CUMULATIVE_TRIALS,
        "oos_note": "fourth use of the 2025-10-01~ window (wave31 C, wave32 L1, wave33 freq, wave34 this)",
        "gates": gates,
        "overall": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "promoted": not failures,
        "method_table": [
            {
                "method": name,
                "median_best_feasible": medians[name],
                "per_seed_best_feasible": [row["methods"][name]["best_feasible_fitness"] for row in rows],
                "per_seed_feasible_count": [row["methods"][name]["feasible_count"] for row in rows],
                "mean_runtime_seconds": float(np.mean([row["methods"][name]["runtime_seconds"] for row in rows])),
            }
            for name in METHODS
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n우승 기법: {winner} (기법별 중앙값: " +
          ", ".join(f"{k} {v:+.3f}" for k, v in sorted(medians.items(), key=lambda kv: -kv[1])) + ")")
    print(f"판정 후보 시드 {chosen_row['seed']}")
    eq = payload["equity_windows"]
    print(f"IS  $100 -> ${eq['is']['end_usdt']:,.2f} (연 {eq['is']['annualized']*100:+.2f}%, MDD {eq['is']['mdd']*100:.1f}%)")
    print(f"OOS ${eq['oos']['start_usdt']:,.2f} -> ${eq['oos']['end_usdt']:,.2f} "
          f"(연 {eq['oos']['annualized']*100:+.2f}% vs I5 {eq['baseline_oos']['annualized']*100:+.2f}%)")
    print(f"진입 {candidate['entry_profile']['trades_per_active_day']:.2f}회/활동일 | 최대 notional ${max_notional:,.0f}")
    for name, body in gates.items():
        print(f"[{body['status']}] {name}")
    print(f"\nOVERALL {payload['overall']} | failures {failures}")
    return payload


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "all"
    explicit = tuple(int(value) for value in argv[2:] if value.isdigit())
    named = tuple(name for value in argv[2:] for name in value.split(",") if name in METHODS)
    if command == "status":
        outstanding = missing_work()
        print(f"완료 시드 {len(SEEDS) - len(outstanding)}/{len(SEEDS)}")
        for seed, remaining in outstanding.items():
            print(f"  seed {seed}: 미완 {remaining}")
        return 0
    if command in {"tournament", "all"}:
        phase_tournament(explicit or SEEDS, named or tuple(METHODS))
    if command in {"rl", "all"}:
        phase_rl()
    if command in {"judge", "all"}:
        rows = load_seed_rows()
        if len(rows) != len(SEEDS):
            print(f"judge 보류: 시드 결과 {len(rows)}/{len(SEEDS)}개만 존재한다", file=sys.stderr)
            return 1
        phase_judge(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
