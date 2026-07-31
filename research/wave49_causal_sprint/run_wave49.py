#!/usr/bin/env python3
# Wave-49: re-test wave31's high-risk sprint family under a CAUSAL walk-forward.
#
# wave31's candidate failed only on ruin (22.52%), which the stated risk tolerance now accepts. But two
# limitations remained, and they are the ones that actually matter: it was chosen by searching 255,621
# genomes and opening a single IS/OOS split, and wave37 demonstrated that post-hoc selection and causal
# selection can disagree in SIGN -- cross-sectional funding showed +33.44% under an opened holdout and
# -22.32% under a walk-forward on better data.
#
# So the question is not "is wave31's genome good" but "does a search over this family, run with only
# past data at every decision point, produce something that works forward". Those are different claims and
# only the second one is deployable.
#
# Protocol, identical in shape to waves 37/38/39/42: train on the trailing 365 days, search, commit to one
# genome, trade the next 90 days, repeat. The applied window never participates in its own selection, so
# every point on the chained curve was out-of-sample when produced. Simulated annealing does the searching
# because wave34's controlled tournament found it beat random 5/5 -- the one place in this campaign where
# method choice was shown to matter.
#
# What would make this fail honestly: the chained curve underperforming a buy-and-hold or the corrected I5
# bar, or ruin arriving far more often than wave31's 22.52% suggested. Both are recorded either way.

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
from research.wave30_qd.engine30 import COMPOUNDING, run_genome
from research.wave34_tournament.encoding34 import DIMENSIONS, decode
from research.wave34_tournament.optimizers34 import Budget, run_simulated_annealing
from research.wave49_causal_sprint.window30 import window_cache

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"

BARS_PER_DAY: Final = 24
TRAIN_DAYS: Final = 365
APPLY_DAYS: Final = 90
SEARCH_BUDGET: Final = 400  # evaluations per reselection
START_CAPITAL: Final = 100.0
I5_CORRECTED_CAGR: Final = 0.0828  # wave45/46's corrected benchmark, not the published 10.27%
RUIN_FLOOR: Final = 50.0
LOG_FLOOR: Final = -10.0
INFEASIBLE_PENALTY: Final = 100.0
MIN_TRADES_PER_WINDOW: Final = 5  # a genome that barely trades in 365 days cannot be judged


def train_objective(train_cache, vector: np.ndarray) -> float:
    """Log growth over the training window, penalised for not trading or dying.

    Deliberately the same SHAPE as encoding34.objective (log growth, hard penalty for infeasibility) so
    this wave is not quietly optimising a different thing than wave31 did. The difference is only that it
    is evaluated on a trailing window rather than the whole in-sample span.
    """
    genome = decode(vector)
    result = run_genome(train_cache, genome, mode="full", sizing=COMPOUNDING)
    equity = result.total_equity_daily[result.daily_valid]
    if len(equity) == 0:
        return -INFEASIBLE_PENALTY
    final = float(equity[-1])
    log_growth = float(np.clip(np.log(max(final, 1e-9) / START_CAPITAL), LOG_FLOOR, None))
    if len(result.trades) < MIN_TRADES_PER_WINDOW:
        return log_growth - INFEASIBLE_PENALTY
    if final <= 0.0:
        return -INFEASIBLE_PENALTY
    return log_growth


def search_window(train_cache, seed: int) -> tuple[np.ndarray, float]:
    best_vector: list[np.ndarray] = []
    best_score: list[float] = [-np.inf]

    def objective_fn(vector: np.ndarray):
        score = train_objective(train_cache, vector)
        if score > best_score[0]:
            best_score[0] = score
            best_vector.clear()
            best_vector.append(np.asarray(vector, dtype=float).copy())
        # optimizers34's Budget expects an object it can compare by .fitness
        return _Scored(score, np.asarray(vector, dtype=float))

    budget = Budget(objective_fn=objective_fn, limit=SEARCH_BUDGET)
    rng = np.random.default_rng(seed)
    run_simulated_annealing(budget, rng)
    if not best_vector:
        return np.full(DIMENSIONS, 0.5), -np.inf
    return best_vector[0], best_score[0]


class _Scored:
    """Minimal Evaluation stand-in for optimizers34's Budget.

    The optimisers read `.fitness` to compare candidates and `.extras["vector"]` to recover the point that
    produced a result (simulated annealing restarts from its best burn-in sample that way), so both are
    provided. Constructing a full encoding34.Evaluation here would require inventing the dozen descriptor
    and diagnostic fields it carries, none of which this wave's objective computes.
    """

    __slots__ = ("fitness", "extras")

    def __init__(self, fitness: float, vector: np.ndarray) -> None:
        self.fitness = fitness
        self.extras = {"vector": [float(x) for x in vector]}


def walk_forward(cache, verbose: bool = True) -> dict:
    train_bars = TRAIN_DAYS * BARS_PER_DAY
    apply_bars = APPLY_DAYS * BARS_PER_DAY

    capital = START_CAPITAL
    equity_curve: list[float] = []
    equity_days: list[int] = []
    selections: list[dict] = []
    total_trades = 0
    total_liquidations = 0
    per_trade_returns: list[float] = []
    evaluations = 0

    start = train_bars
    window_index = 0
    while start + apply_bars <= cache.n_bars:
        train_cache = window_cache(cache, start - train_bars, start)
        vector, score = search_window(train_cache, seed=20260731 + window_index)
        evaluations += SEARCH_BUDGET
        genome = decode(vector)

        apply_cache = window_cache(cache, start, start + apply_bars)
        applied = run_genome(apply_cache, genome, mode="full", sizing=COMPOUNDING)
        equity = applied.total_equity_daily[applied.daily_valid]
        if len(equity) == 0:
            start += apply_bars
            window_index += 1
            continue

        # run_genome always starts from $100, so the window's growth factor is what carries forward.
        factor = float(equity[-1]) / START_CAPITAL
        scaled = capital * (equity / START_CAPITAL)
        equity_curve.extend(scaled.tolist())
        # window_cache re-bases day_of_bar to 0, so the applied cache's own value cannot identify the
        # calendar day. The GLOBAL index is taken from the unsliced cache at the same bar; using the
        # re-based value silently collapsed every window onto 2019 in the first run of this wave.
        first_day = int(cache.day_of_bar[start])
        equity_days.extend(range(first_day, first_day + len(scaled)))
        capital *= factor
        total_trades += len(applied.trades)
        total_liquidations += applied.n_liquidations
        per_trade_returns.extend(float(x) for x in applied.trade_returns)

        selections.append({
            "apply_from": str(apply_cache.index[0].date()),
            "apply_to": str(apply_cache.index[-1].date()),
            "train_score": score,
            "signal_family": genome.signal_family,
            "lookback_bars": genome.lookback_bars,
            "leverage": float(genome.leverage),
            "stop_pct": float(genome.stop_pct),
            "target_r": float(genome.target_r),
            "allow_short": bool(genome.allow_short),
            "applied_return": factor - 1.0,
            "n_trades": len(applied.trades),
            "capital_after": capital,
        })
        if verbose:
            print(
                f"  {apply_cache.index[0].date()} ~ {apply_cache.index[-1].date()} | "
                f"{genome.signal_family:9s} lb{genome.lookback_bars:<3d} {genome.leverage:5.2f}x "
                f"| 적용 {factor-1.0:+8.2%} | ${capital:10,.2f} | 거래 {len(applied.trades):3d}",
                flush=True,
            )
        start += apply_bars
        window_index += 1
        if capital <= 0.01:
            print("  자본 소진 — 중단")
            break

    equity = np.asarray(equity_curve, dtype=float)
    days = len(equity)
    peak = np.maximum.accumulate(equity) if days else np.array([START_CAPITAL])
    returns = np.asarray(per_trade_returns, dtype=float)
    return {
        "final_usdt": float(equity[-1]) if days else START_CAPITAL,
        "annualised": float((equity[-1] / START_CAPITAL) ** (365.0 / days) - 1.0) if days and equity[-1] > 0 else -1.0,
        "mdd": float(np.max(1.0 - equity / peak)) if days else 0.0,
        "days": days,
        "reselections": len(selections),
        "evaluations": evaluations,
        "n_trades": total_trades,
        "n_liquidations": total_liquidations,
        "trades_per_day": total_trades / days if days else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else float("nan"),
        "per_trade_mean": float(returns.mean()) if len(returns) else float("nan"),
        "per_trade_p95": float(np.percentile(returns, 95)) if len(returns) else float("nan"),
        "per_trade_min": float(returns.min()) if len(returns) else float("nan"),
        "share_entries_over_30usd": float((returns * START_CAPITAL >= 30.0).mean()) if len(returns) else float("nan"),
        "selections": selections,
        "equity": equity.tolist(),
        "equity_days": equity_days,
        "min_equity": float(equity.min()) if days else START_CAPITAL,
        "share_below_ruin_floor": float((equity < RUIN_FLOOR).mean()) if days else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="wave49 causal walk-forward over the sprint family")
    parser.add_argument("--stage", choices=("run", "judge"), required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "final.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cache = build_market_cache()

    if args.stage == "run":
        print("=== wave49: 스프린트 계열 인과적 워크포워드 ===")
        print(f"훈련 {TRAIN_DAYS}일 -> 적용 {APPLY_DAYS}일 · 창당 탐색 {SEARCH_BUDGET}회 (SA)")
        print(f"봉 {cache.n_bars:,} · 종목 {cache.symbols}\n")
        payload["base"] = walk_forward(cache)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        base = payload["base"]
        print(f"\n$100 -> ${base['final_usdt']:,.2f} | 연 {base['annualised']:+.2%} | MDD {base['mdd']:.2%}")
        print(f"재선정 {base['reselections']}회 · 평가 {base['evaluations']:,}회 · {time.time()-started:.0f}s")
        print("다음: --stage judge")
        return 0

    if "base" not in payload:
        print("먼저 --stage run")
        return 1
    base = payload["base"]

    import pandas as pd

    equity = np.asarray(base["equity"], dtype=float)
    years = np.array([cache.daily_index[i].year for i in base["equity_days"]])
    yearly: dict[str, float] = {}
    previous = START_CAPITAL
    for year in sorted(set(years.tolist())):
        segment = equity[years == year]
        if len(segment) < 2:
            continue
        yearly[str(year)] = float(segment[-1] / previous - 1.0)
        previous = segment[-1]
    recent = [v for k, v in yearly.items() if int(k) >= 2025]

    print(f"\n=== 인과적 곡선 (적용 {base['days']}일) ===")
    print(f"  $100 -> ${base['final_usdt']:,.2f} | 연 {base['annualised']:+.2%} | MDD {base['mdd']:.2%}")
    print(f"  최저 자산 ${base['min_equity']:,.2f} · ${RUIN_FLOOR:.0f} 미달 일수 비중 {base['share_below_ruin_floor']:.1%}")
    print(f"  거래 {base['n_trades']}회 ({base['trades_per_day']:.3f}/일) · 강제청산 {base['n_liquidations']}건")
    print(f"  승률 {base['win_rate']:.1%} · 진입당 평균 {base['per_trade_mean']:+.3%} · p95 {base['per_trade_p95']:+.2%}")
    print(f"  진입당 $30 이상 비중 {base['share_entries_over_30usd']:.1%} ($100 고정 환산)")

    print("\n=== 연도별 ===")
    for year, change in sorted(yearly.items()):
        print(f"  {year}: {change:+9.2%}")
    if recent:
        print(f"  2025년 이후 평균 {float(np.mean(recent)):+.2%}")

    from collections import Counter
    families = Counter(s["signal_family"] for s in base["selections"])
    print(f"\n=== 선정 분포 ===\n  계열 {dict(families)}")
    print(f"  레버리지 중앙 {np.median([s['leverage'] for s in base['selections']]):.2f}x "
          f"· 범위 {min(s['leverage'] for s in base['selections']):.2f}~{max(s['leverage'] for s in base['selections']):.2f}x")
    positive = sum(1 for s in base["selections"] if s["applied_return"] > 0)
    print(f"  적용창 양수 {positive}/{len(base['selections'])} = {positive/len(base['selections']):.1%}")

    window_returns = np.array([s["applied_return"] for s in base["selections"]], dtype=float)
    full_multiple = float(np.prod(1.0 + window_returns))
    order = np.argsort(-window_returns)
    trimmed_multiples = {}
    for k in (1, 2, 3):
        keep = np.ones(len(window_returns), dtype=bool)
        keep[order[:k]] = False
        trimmed_multiples[k] = float(np.prod(1.0 + window_returns[keep]))
    trimmed_multiple = trimmed_multiples[3]
    print(f"\n=== 꼬리 집중도 ===")
    print(f"  전체 복리 배수 {full_multiple:,.2f}x · 창 수익률 중앙 {np.median(window_returns):+.2%}")
    for k in (1, 2, 3):
        print(f"  상위 {k}창 제거 후 {trimmed_multiples[k]:,.2f}x")

    gates = {
        "R1_causality": {"status": "PASS" if base["reselections"] >= 15 else "FAIL",
                          "reselections": base["reselections"],
                          "detail": "적용창은 선정에 미사용; 훈련창은 직전 365일만"},
        "R2_slicer_verified": {"status": "PASS",
                                "detail": "전 구간 슬라이스가 run_genome(full)과 거래수·수익률·최종값 완전 일치 (window30.py)"},
        "R3_beats_i5": {"status": "PASS" if base["annualised"] > I5_CORRECTED_CAGR else "FAIL",
                         "annualised": base["annualised"], "bar": I5_CORRECTED_CAGR,
                         "detail": "정정된 I5 기준선(8.28%) 초과 여부"},
        # The first version of this gate compared FINAL equity to the ruin floor and passed a path whose
        # minimum was $4.60 after a 97.56% drawdown. A survival gate that only looks at the endpoint is
        # not measuring survival, so it now tests the worst point on the path.
        "R4_survives": {"status": "PASS" if base["min_equity"] >= RUIN_FLOOR else "FAIL",
                         "final_usdt": base["final_usdt"], "min_equity": base["min_equity"],
                         "floor": RUIN_FLOOR, "share_days_below_floor": base["share_below_ruin_floor"],
                         "detail": "경로 최저 자산이 파산선을 넘었는가 (최종값이 아니라 최저값)"},
        "R5_no_liquidation": {"status": "PASS" if base["n_liquidations"] == 0 else "FAIL",
                               "n_liquidations": base["n_liquidations"]},
        "R6_recent_regime": {"status": "PASS" if recent and float(np.mean(recent)) > 0 else "FAIL",
                              "mean_since_2025": float(np.mean(recent)) if recent else None,
                              "detail": "2025~2026(BTC 하락 구간)에서 양수인가 — wave48이 지적한 베타 의존 검정"},
        "R7_window_consistency": {"status": "PASS" if positive / len(base["selections"]) > 0.5 else "FAIL",
                                   "positive_share": positive / len(base["selections"])},
        # Tail concentration. A curve carried by a handful of windows has far fewer effective observations
        # than its window count suggests, so its confidence interval is much wider than the headline
        # implies. wave18/38/44 all found headline figures leaning on one period; this makes that test a
        # gate instead of a footnote.
        "R8_not_tail_dependent": {
            "status": "PASS" if trimmed_multiple > 1.5 else "FAIL",
            "full_multiple": full_multiple,
            "multiple_without_top_1": trimmed_multiples[1],
            "multiple_without_top_2": trimmed_multiples[2],
            "multiple_without_top_3": trimmed_multiples[3],
            "detail": "상위 3개 적용창을 제거해도 자본이 1.5배 이상 남는가",
        },
    }
    failures = [k for k, v in gates.items() if v["status"] == "FAIL"]
    print()
    for name, gate in gates.items():
        print(f"[{gate['status']}] {name}")
    print(f"\nOVERALL {'PASS' if not failures else 'FAIL'} | failures {failures}")

    payload.update({"yearly": yearly, "recent_mean": float(np.mean(recent)) if recent else None,
                    "gates": gates, "failures": failures,
                    "overall": "PASS" if not failures else "FAIL"})
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("results/final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
