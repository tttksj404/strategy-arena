# Wave-35 search on the widened universe (20 symbols, 2022-01-01 ~ 2026-07-14).
#
# Two arms only: Simulated Annealing -- the sole method that beat matched random search in wave34's
# controlled tournament (5/5 seeds) -- and a matched-budget random control. The other four methods
# are not re-run: wave34 already measured them on an identical objective, and re-running known
# losers here would add multiple-testing without adding information.
#
# Objective, encoding and constraint are inherited from wave34 unchanged so the two waves are
# directly comparable; the ONLY difference is the data underneath and the breadth gene.

from __future__ import annotations

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

from research.wave30_qd.engine30 import COMPOUNDING, run_genome
from research.wave30_qd.fitness30 import Evaluation
from research.wave30_qd.genome30 import (
    COOLDOWN_CHOICES,
    ENTRY_THRESHOLD_RANGE,
    LOOKBACK_CHOICES,
    MAX_HOLD_CHOICES,
    RISK_FRAC_RANGE,
    SIGNAL_FAMILIES,
    SLEEVE_FRACTION_CHOICES,
    TARGET_R_RANGE,
    CONCURRENCY_CHOICES,
    _max_stop_for_risk,
    _min_stop_for_risk,
)
from research.wave34_tournament.optimizers34 import Budget, run_random, run_simulated_annealing
from research.wave33_frequency.fitness33 import entry_profile
from research.wave35_universe.dataio35 import build_wide_cache
from research.wave35_universe.genome35 import BREADTH_CHOICES, Genome35

DIMENSIONS: Final = 13
MIN_TRADES_PER_ACTIVE_DAY: Final = 1.0
INFEASIBLE_PENALTY: Final = 100.0
LOG_FLOOR: Final = -10.0
CAPACITY_LIMIT_USDT: Final = 50_000.0
RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"


def _pick(u: float, choices: tuple) -> Any:
    return choices[int(np.clip(int(u * len(choices)), 0, len(choices) - 1))]


def _lerp(u: float, low: float, high: float) -> float:
    return float(low + np.clip(u, 0.0, 1.0) * (high - low))


def decode(vector: np.ndarray, ranked_symbols: tuple[str, ...]) -> Genome35:
    """[0,1]^13 -> Genome35, always feasible.

    stop_pct is reparameterised as a position INSIDE its feasible interval for the decoded
    risk_frac (wave34's device, carried over): the infeasible region is not representable, so no
    genome is ever silently repaired into validity.
    """
    u = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
    risk_frac = _lerp(u[6], *RISK_FRAC_RANGE)
    low, high = _min_stop_for_risk(risk_frac), _max_stop_for_risk(risk_frac)
    stop_pct = _lerp(u[3], low, high) if high > low else low
    breadth = int(_pick(u[9], BREADTH_CHOICES))
    breadth = min(breadth, len(ranked_symbols))
    return Genome35(
        signal_family=str(_pick(u[0], SIGNAL_FAMILIES)),
        lookback_bars=int(_pick(u[1], LOOKBACK_CHOICES)),
        entry_threshold=_lerp(u[2], *ENTRY_THRESHOLD_RANGE),
        stop_pct=stop_pct,
        target_r=_lerp(u[4], *TARGET_R_RANGE),
        trail_enabled=bool(u[5] >= 0.5),
        risk_frac=risk_frac,
        max_hold_bars=int(_pick(u[7], MAX_HOLD_CHOICES)),
        allow_short=bool(u[8] >= 0.5),
        symbols=tuple(ranked_symbols[:breadth]),
        max_concurrent=int(_pick(u[10], CONCURRENCY_CHOICES)),
        cooldown_bars_after_loss=int(_pick(u[11], COOLDOWN_CHOICES)),
        sleeve_fraction=float(_pick(u[12], SLEEVE_FRACTION_CHOICES)),
    ).validate()


def objective(cache, ranked_symbols: tuple[str, ...], vector: np.ndarray) -> Evaluation:
    """fitness = log(IS final total capital / 100), minus 100 if the request's frequency/survival
    condition is violated. Identical to wave34's objective."""
    genome = decode(vector, ranked_symbols)
    result = run_genome(cache, genome, mode="is", sizing=COMPOUNDING)
    profile = entry_profile(cache, result)
    final_total = float(result.total_equity_daily[result.daily_valid][-1])
    log_growth = float(np.clip(np.log(max(final_total, 1e-9) / 100.0), LOG_FLOOR, None))

    reasons: list[str] = []
    if profile.trades_per_active_day < MIN_TRADES_PER_ACTIVE_DAY:
        reasons.append(f"{profile.trades_per_active_day:.3f} entries/active-day < 1.0")
    if not profile.survived_full_span:
        reasons.append("account died before the IS span ended")

    return Evaluation(
        genome=genome,
        fitness=float(log_growth - (0.0 if not reasons else INFEASIBLE_PENALTY)),
        fold_cagrs=(),
        is_total_cagr=0.0,
        is_total_final=final_total,
        sleeve_mdd=profile.account_mdd,
        total_mdd=profile.account_mdd,
        trades_per_year=profile.trades_per_active_day * 365.0,
        n_trades=profile.n_trades,
        n_liquidations=result.n_liquidations,
        wipe_probability=0.0,
        descriptor=(0, 0, 0),
        mean_leverage=float(result.mean_realized_leverage),
        min_notional_usdt=float(result.min_notional_usdt),
        sleeve_survived=profile.survived_full_span,
        extras={
            "vector": [float(x) for x in np.clip(vector, 0.0, 1.0)],
            "log_growth": log_growth,
            "infeasible_reasons": reasons,
            "entry_profile": profile.as_dict(),
            "max_notional_usdt": float(max((t.notional_usdt for t in result.trades), default=0.0)),
        },
    )


def _evaluation_json(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "genome": evaluation.genome.to_dict(),
        "vector": evaluation.extras["vector"],
        "fitness": evaluation.fitness,
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


ARMS: Final = {"simulated_annealing": run_simulated_annealing, "random": run_random}


def run_seed(seed: int, budget_per_arm: int, arms: tuple[str, ...] = ()) -> dict[str, Any]:
    cache, ranked = build_wide_cache()
    started = time.time()
    path = RESULTS_DIR / f"seed_{seed}.json"
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "seed": seed,
        "universe": list(ranked),
        "arms": {},
    }
    requested = arms or tuple(ARMS)
    for name in requested:
        method = ARMS[name]
        if name in out["arms"]:
            continue
        budget = Budget(lambda vector: objective(cache, ranked, vector), limit=budget_per_arm)
        arm_started = time.time()
        method(budget, np.random.default_rng(seed))
        best = budget.best()
        feasible = [item for item in budget.history if not item.extras["infeasible_reasons"]]
        best_feasible = max(feasible, key=lambda item: item.fitness) if feasible else None
        out["arms"][name] = {
            "spent": budget.spent,
            "runtime_seconds": time.time() - arm_started,
            "best_fitness": best.fitness if best else float("-inf"),
            "feasible_count": len(feasible),
            "best_feasible_fitness": best_feasible.fitness if best_feasible else float("-inf"),
            "best_feasible": _evaluation_json(best_feasible) if best_feasible else None,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(
            f"  seed {seed} {name:20s} best(제약충족) "
            f"{out['arms'][name]['best_feasible_fitness']:+8.3f} "
            f"(충족 {len(feasible)}, {time.time()-arm_started:.0f}s)",
            flush=True,
        )
    out["runtime_seconds"] = time.time() - started
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out



SEEDS: Final = (2026351, 2026352, 2026353, 2026354, 2026355)
BUDGET_PER_ARM: Final = 3_000  # identical to wave34, so the two waves are directly comparable


def status() -> None:
    for seed in SEEDS:
        path = RESULTS_DIR / f"seed_{seed}.json"
        done = sorted(json.loads(path.read_text(encoding="utf-8"))["arms"]) if path.exists() else []
        print(f"  seed {seed}: 완료 arm {done or '없음'}")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "status":
        status()
        return 0
    if command == "run":
        seeds = tuple(int(v) for v in argv[2:] if v.isdigit()) or SEEDS
        arms = tuple(v for v in argv[2:] if v in ARMS)
        for seed in seeds:
            run_seed(seed, BUDGET_PER_ARM, arms)
        return 0
    print(f"unknown command {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
