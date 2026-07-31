#!/usr/bin/env python3
# Wave-50: wave49 named the cause, so this tests the fix.
#
# wave49's causal walk-forward reached +125.30%/yr and failed two gates: path minimum $4.60 after a 97.56%
# drawdown (R4), and 105.39x compounding collapsing to 1.05x once three of twenty-three windows are
# removed (R8). The cause was identifiable rather than mysterious: the objective is log(final/start), which
# rewards leverage without bound, so the search pushed to 19.70x and the curve became a lottery.
#
# Two things could fix that, and they are tested together because they act on the same mechanism from
# different sides:
#   - a hard leverage ceiling, which caps how much a single window can risk
#   - a drawdown penalty in the TRAINING objective, so a genome that survives its own training window
#     badly is not selected even if it ends high
#
# The penalty is applied to training only. Penalising the applied window would be scoring a decision with
# information from after it was made, which is the error the whole walk-forward protocol exists to avoid.
#
# What makes this falsifiable: wave49's median window return was +25.53% with 16 of 23 positive, so if the
# edge is real and only the sizing was reckless, some ceiling should preserve a decent return while fixing
# the path. If instead every ceiling either keeps the lottery or erases the return, then the family's
# apparent edge WAS the leverage, and it closes.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import build_market_cache
from research.wave30_qd.engine30 import COMPOUNDING, max_drawdown, run_genome
from research.wave34_tournament.encoding34 import DIMENSIONS, decode
from research.wave34_tournament.optimizers34 import Budget, run_simulated_annealing
from research.wave49_causal_sprint.run_wave49 import _Scored
from research.wave49_causal_sprint.window30 import window_cache

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

BARS_PER_DAY: Final = 24
TRAIN_DAYS: Final = 365
APPLY_DAYS: Final = 90
SEARCH_BUDGET = 400  # overridable via --budget; wave50's seed test showed 400 cannot select stably
START_CAPITAL: Final = 100.0
I5_CORRECTED_CAGR: Final = 0.0828
RUIN_FLOOR: Final = 50.0
LOG_FLOOR: Final = -10.0
INFEASIBLE_PENALTY: Final = 100.0
MIN_TRADES_PER_WINDOW: Final = 5

LEVERAGE_CAPS: Final = (2.0, 3.0, 5.0, 8.0, 20.0)
_SEED_OFFSET = 0  # shifted by walk_forward_seeded for the robustness test; 0 for the headline runs
MDD_KNEE: Final = 0.30  # drawdown beyond this is penalised in training
MDD_WEIGHT: Final = 3.0


def train_objective(train_cache, vector: np.ndarray, leverage_cap: float, mdd_weight: float) -> float:
    """Log growth on the training window, minus a drawdown penalty, with leverage hard-capped.

    The cap is enforced by penalty rather than by clipping the decoded genome: clipping would let the
    search believe it had chosen 19.7x while actually trading 8x, which is the exact failure mode
    genome30.leverage's own docstring warns about ("a quietly false leverage would corrupt the whole map").
    """
    genome = decode(vector)
    if genome.leverage > leverage_cap:
        return -INFEASIBLE_PENALTY - float(genome.leverage)  # ordered penalty guides SA back inside
    result = run_genome(train_cache, genome, mode="full", sizing=COMPOUNDING)
    equity = result.total_equity_daily[result.daily_valid]
    if len(equity) == 0:
        return -INFEASIBLE_PENALTY
    final = float(equity[-1])
    if final <= 0.0:
        return -INFEASIBLE_PENALTY
    log_growth = float(np.clip(np.log(max(final, 1e-9) / START_CAPITAL), LOG_FLOOR, None))
    if len(result.trades) < MIN_TRADES_PER_WINDOW:
        return log_growth - INFEASIBLE_PENALTY
    drawdown = float(max_drawdown(equity))
    return log_growth - mdd_weight * max(0.0, drawdown - MDD_KNEE)


def search_window(train_cache, seed: int, leverage_cap: float, mdd_weight: float):
    best_vector: list[np.ndarray] = []
    best_score = [-np.inf]

    def objective_fn(vector: np.ndarray):
        score = train_objective(train_cache, vector, leverage_cap, mdd_weight)
        if score > best_score[0]:
            best_score[0] = score
            best_vector.clear()
            best_vector.append(np.asarray(vector, dtype=float).copy())
        return _Scored(score, np.asarray(vector, dtype=float))

    budget = Budget(objective_fn=objective_fn, limit=SEARCH_BUDGET)
    run_simulated_annealing(budget, np.random.default_rng(seed))
    return (best_vector[0] if best_vector else np.full(DIMENSIONS, 0.5)), best_score[0]


def walk_forward(cache, leverage_cap: float, mdd_weight: float, verbose: bool = False) -> dict:
    train_bars = TRAIN_DAYS * BARS_PER_DAY
    apply_bars = APPLY_DAYS * BARS_PER_DAY
    capital = START_CAPITAL
    equity_curve: list[float] = []
    equity_days: list[int] = []
    window_returns: list[float] = []
    leverages: list[float] = []
    families: list[str] = []
    trade_returns: list[float] = []
    total_trades = total_liquidations = 0

    start = train_bars
    index = 0
    while start + apply_bars <= cache.n_bars:
        train_cache = window_cache(cache, start - train_bars, start)
        vector, _ = search_window(train_cache, 20260731 + _SEED_OFFSET + index, leverage_cap, mdd_weight)
        genome = decode(vector)
        apply_cache = window_cache(cache, start, start + apply_bars)
        applied = run_genome(apply_cache, genome, mode="full", sizing=COMPOUNDING)
        equity = applied.total_equity_daily[applied.daily_valid]
        if len(equity) == 0:
            start += apply_bars
            index += 1
            continue
        factor = float(equity[-1]) / START_CAPITAL
        scaled = capital * (equity / START_CAPITAL)
        equity_curve.extend(scaled.tolist())
        first_day = int(cache.day_of_bar[start])
        equity_days.extend(range(first_day, first_day + len(scaled)))
        capital *= factor
        window_returns.append(factor - 1.0)
        leverages.append(float(genome.leverage))
        families.append(genome.signal_family)
        trade_returns.extend(float(x) for x in applied.trade_returns)
        total_trades += len(applied.trades)
        total_liquidations += applied.n_liquidations
        if verbose:
            print(f"    {apply_cache.index[0].date()} {genome.signal_family:17s} "
                  f"{genome.leverage:5.2f}x {factor-1.0:+9.2%} -> ${capital:,.2f}", flush=True)
        start += apply_bars
        index += 1
        if capital <= 0.01:
            break

    equity = np.asarray(equity_curve, dtype=float)
    days = len(equity)
    returns = np.asarray(window_returns, dtype=float)
    order = np.argsort(-returns)
    trimmed = {}
    for k in (1, 2, 3):
        keep = np.ones(len(returns), dtype=bool)
        keep[order[:k]] = False
        trimmed[k] = float(np.prod(1.0 + returns[keep])) if len(returns) > k else float("nan")
    per_trade = np.asarray(trade_returns, dtype=float)
    peak = np.maximum.accumulate(equity) if days else np.array([START_CAPITAL])
    return {
        "leverage_cap": leverage_cap,
        "mdd_weight": mdd_weight,
        "final_usdt": float(equity[-1]) if days else START_CAPITAL,
        "annualised": float((equity[-1] / START_CAPITAL) ** (365.0 / days) - 1.0) if days and equity[-1] > 0 else -1.0,
        "mdd": float(np.max(1.0 - equity / peak)) if days else 0.0,
        "min_equity": float(equity.min()) if days else START_CAPITAL,
        "share_below_floor": float((equity < RUIN_FLOOR).mean()) if days else 0.0,
        "days": days,
        "windows": len(returns),
        "window_median": float(np.median(returns)) if len(returns) else float("nan"),
        "window_positive_share": float((returns > 0).mean()) if len(returns) else float("nan"),
        "full_multiple": float(np.prod(1.0 + returns)) if len(returns) else float("nan"),
        "trimmed_multiple_3": trimmed[3],
        "trimmed": trimmed,
        "n_trades": total_trades,
        "n_liquidations": total_liquidations,
        "realised_leverage_median": float(np.median(leverages)) if leverages else float("nan"),
        "realised_leverage_max": float(np.max(leverages)) if leverages else float("nan"),
        "families": {f: families.count(f) for f in sorted(set(families))},
        "share_entries_over_30usd": float((per_trade * START_CAPITAL >= 30.0).mean()) if len(per_trade) else float("nan"),
        "equity": equity.tolist(),
        "equity_days": equity_days,
        "window_returns": returns.tolist(),
    }


def walk_forward_seeded(cache, leverage_cap: float, mdd_weight: float, seed_offset: int) -> dict:
    """walk_forward with the per-window search seeds shifted, so only the search path changes."""
    global _SEED_OFFSET
    previous = _SEED_OFFSET
    _SEED_OFFSET = seed_offset
    try:
        return walk_forward(cache, leverage_cap, mdd_weight)
    finally:
        _SEED_OFFSET = previous


def gates_for(row: dict) -> dict:
    return {
        "R3_beats_i5": row["annualised"] > I5_CORRECTED_CAGR,
        "R4_survives": row["min_equity"] >= RUIN_FLOOR,
        "R5_no_liquidation": row["n_liquidations"] == 0,
        "R7_window_consistency": row["window_positive_share"] > 0.5,
        "R8_not_tail_dependent": row["trimmed_multiple_3"] > 1.5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wave50: leverage ceiling x drawdown penalty")
    parser.add_argument("--only", type=float, help="single leverage cap")
    parser.add_argument("--seed-test", type=float,
                        help="re-run one cap under several seed offsets to separate signal from search luck")
    parser.add_argument("--budget", type=int, help="evaluations per reselection (default 400)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="single seed offset, for running one budget-heavy seed per invocation")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    global SEARCH_BUDGET
    if args.budget:
        SEARCH_BUDGET = args.budget

    if args.budget and args.seed_test:
        # Does more search remove the seed dependence? If 400 evaluations cannot select stably but 2000
        # can, the instability was undersampling and the family is still open. If the spread survives a
        # 5x budget, the training window genuinely does not predict the applied one and the family closes.
        # One seed per invocation, because a heavier budget puts a full sweep past the shell timeout.
        cache_local = build_market_cache()
        path_budget = RESULTS_DIR / f"budget_{args.budget}_cap_{args.seed_test:.1f}.json"
        store = json.loads(path_budget.read_text(encoding="utf-8")) if path_budget.exists() else {"rows": {}}
        offset = args.seed_offset
        print(f"=== 예산 {args.budget}회 · 상한 {args.seed_test:.1f}x · 시드오프셋 {offset} ===")
        row = walk_forward_seeded(cache_local, args.seed_test, MDD_WEIGHT, offset)
        g = gates_for(row)
        store["rows"][str(offset)] = {k: v for k, v in row.items()
                                      if k not in ("equity", "equity_days", "window_returns")}
        path_budget.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")
        print(f"  연 {row['annualised']:+8.2%} | MDD {row['mdd']:6.2%} | 최저 ${row['min_equity']:,.2f} "
              f"| 창중앙 {row['window_median']:+7.2%} | 상위3제거 {row['trimmed_multiple_3']:.2f}x "
              f"| 게이트 {sum(g.values())}/5")
        done = sorted(store["rows"], key=lambda k: int(k))
        print(f"\n  누적 시드 {done}")
        if len(done) >= 3:
            annualised = [store["rows"][k]["annualised"] for k in done]
            medians = [store["rows"][k]["window_median"] for k in done]
            passes = [sum(gates_for(store["rows"][k]).values()) for k in done]
            spread = max(annualised) - min(annualised)
            print(f"  연환산 {min(annualised):+.2%} ~ {max(annualised):+.2%} (폭 {spread:.2%}p)")
            print(f"  창중앙 {min(medians):+.2%} ~ {max(medians):+.2%} · 5/5 통과 {sum(1 for p in passes if p==5)}/{len(done)}")
            print(f"  예산 400 기준: 폭 67.59%p · 5/5 통과 0/5")
            if spread < 0.30 and sum(1 for p in passes if p == 5) >= len(done) - 1:
                print("  => 예산 증가가 시드 의존을 제거했다. 계열이 다시 열린다.")
            else:
                print("  => 예산을 5배로 올려도 시드 의존이 남는다. 훈련창이 적용창을 예측하지 못한다 -> 계열 닫힘.")
        else:
            print(f"  (판정에는 시드 3개 이상 필요 — --seed-offset 으로 이어서 실행)")
        return 0

    if args.seed_test:
        # Caps 2.0x and 3.0x produced wildly different outcomes (+225.82% at 5/5 gates versus +24.34% at
        # 3/5). A ceiling one step apart should not flip a result that hard, so the likely explanation is
        # that 400 evaluations of a stochastic search land in different regions per cap. Repeating one cap
        # across seed offsets separates "this ceiling works" from "that run got lucky", and it is the only
        # check that can tell them apart.
        cache_local = build_market_cache()
        print(f"=== 시드 견고성 검정: 상한 {args.seed_test:.1f}x ===")
        print("같은 상한, 탐색 시드만 변경. 흩어지면 결과는 탐색운이다.\n")
        print(f"{'시드오프셋':>9} {'연환산':>9} {'MDD':>7} {'최저$':>9} {'창중앙':>8} {'상위3제거':>9} {'게이트':>6}")
        rows = []
        for offset in (0, 1000, 2000, 3000, 4000):
            row = walk_forward_seeded(cache_local, args.seed_test, MDD_WEIGHT, offset)
            g = gates_for(row)
            rows.append(row)
            print(f"{offset:9d} {row['annualised']:+8.2%} {row['mdd']:6.2%} ${row['min_equity']:8,.2f} "
                  f"{row['window_median']:+7.2%} {row['trimmed_multiple_3']:8.2f}x {sum(g.values()):5d}/5",
                  flush=True)
        annualised = [r["annualised"] for r in rows]
        passes = [sum(gates_for(r).values()) for r in rows]
        print(f"\n  연환산 범위 {min(annualised):+.2%} ~ {max(annualised):+.2%} "
              f"(중앙 {float(np.median(annualised)):+.2%})")
        print(f"  5/5 통과 시드 {sum(1 for p in passes if p == 5)}/{len(passes)}")
        if max(annualised) - min(annualised) > 1.0:
            print("  => 시드에 따라 결과가 100%p 이상 흔들린다. 이것은 탐색운이며 배포 근거가 아니다.")
        elif sum(1 for p in passes if p == 5) >= 4:
            print("  => 시드 전반에서 통과한다. 상한 자체의 성질로 볼 근거가 있다.")
        else:
            print("  => 일부 시드만 통과한다. 견고하지 않다.")
        (RESULTS_DIR / f"seed_test_{args.seed_test:.1f}.json").write_text(
            json.dumps({"cap": args.seed_test, "rows": [
                {k: v for k, v in r.items() if k not in ("equity", "equity_days", "window_returns")}
                for r in rows]}, indent=2, default=str),
            encoding="utf-8")
        return 0

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "final.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rows": {}}
    cache = build_market_cache()

    print("=== wave50: 레버리지 상한 x 낙폭 벌점 (훈련창에만) ===")
    print(f"낙폭 벌점: 무릎 {MDD_KNEE:.0%} 초과분 x {MDD_WEIGHT} · 상한 위반은 순서화된 벌점으로 배제")
    print(f"wave49 기준(상한 없음): 연 +125.30% · MDD 97.56% · 최저 $4.60 · 상위3창제거 1.05x\n")

    caps = [args.only] if args.only else list(LEVERAGE_CAPS)
    for cap in caps:
        key = f"{cap:.1f}"
        if key in payload["rows"] and not args.only:
            print(f"  상한 {cap:4.1f}x (캐시)")
            continue
        row = walk_forward(cache, cap, MDD_WEIGHT)
        payload["rows"][key] = row
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        g = gates_for(row)
        print(f"  상한 {cap:4.1f}x | 연 {row['annualised']:+8.2%} | MDD {row['mdd']:6.2%} | "
              f"최저 ${row['min_equity']:8,.2f} | 창중앙 {row['window_median']:+7.2%} | "
              f"상위3제거 {row['trimmed_multiple_3']:7.2f}x | 게이트 {sum(g.values())}/5", flush=True)

    if args.only:
        print(f"\n{time.time()-started:.0f}s · 나머지는 --only 없이 실행")
        return 0

    print(f"\n=== 상세 ===")
    print(f"{'상한':>6} {'연환산':>9} {'MDD':>7} {'최저$':>9} {'$50미달':>8} {'실현lev중앙':>11} "
          f"{'창양수':>7} {'전체배수':>9} {'상위3제거':>9} {'$30+':>6} {'게이트':>6}")
    best = None
    for cap in LEVERAGE_CAPS:
        row = payload["rows"].get(f"{cap:.1f}")
        if not row:
            continue
        g = gates_for(row)
        passed = sum(g.values())
        print(f"{cap:5.1f}x {row['annualised']:+8.2%} {row['mdd']:6.2%} ${row['min_equity']:8,.2f} "
              f"{row['share_below_floor']:7.1%} {row['realised_leverage_median']:10.2f}x "
              f"{row['window_positive_share']:6.1%} {row['full_multiple']:8.2f}x "
              f"{row['trimmed_multiple_3']:8.2f}x {row['share_entries_over_30usd']:5.1%} {passed:5d}/5")
        if passed == 5 and (best is None or row["annualised"] > best["annualised"]):
            best = row

    print("\n=== 판정 ===")
    if best:
        print(f"  R3~R8 전부 통과하는 상한이 존재한다: {best['leverage_cap']:.1f}x")
        print(f"  연 {best['annualised']:+.2%} · MDD {best['mdd']:.2%} · 최저 ${best['min_equity']:,.2f} "
              f"· 상위3제거 {best['trimmed_multiple_3']:.2f}x")
        print(f"  진입당 $30 이상 {best['share_entries_over_30usd']:.1%} · 계열 {best['families']}")
    else:
        print("  어떤 상한에서도 R3~R8 전부를 통과하지 못한다.")
        print("  => wave49의 겉보기 edge는 레버리지 자체였고, 계열이 닫힌다.")
    payload["best_cap"] = best["leverage_cap"] if best else None
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n{time.time()-started:.0f}s · results/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
