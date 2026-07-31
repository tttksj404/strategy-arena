#!/usr/bin/env python3
# Wave-35 judgement: the single OOS unsealing for the widened universe.
#
# Selection rule mirrors every prior wave: per-seed best FEASIBLE genome, then the MEDIAN seed by
# fitness -- never the best seed. Only that one genome sees OOS, and if it fails the wave ends
# rather than moving on to the next-best (which would turn the holdout into a search).
#
# Why this OOS is not simply the fifth reuse of the same holdout: 17 of the 20 symbols here have
# never appeared in any prior wave, and the price series come from a venue (Bitget) whose data was
# never used before wave35. The 2025-10..2026-07 window is calendar-shared with earlier waves, so
# the market regime is not independent, but the instruments and the data source largely are. That
# is a partial refresh, not a clean one, and the report says so.

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np

from research.wave30_qd.dataio30 import i5_baseline_total_curve
from research.wave30_qd.engine30 import COMPOUNDING, annualized_return, max_drawdown, run_genome
from research.wave30_qd.gates30 import _daily_returns
from research.wave33_frequency.fitness33 import entry_profile
from research.wave35_universe.dataio35 import OOS_SPLIT, build_wide_cache
from research.wave35_universe.genome35 import Genome35
from research.wave35_universe.search35 import ARMS, CAPACITY_LIMIT_USDT, RESULTS_DIR, SEEDS

MC_PATHS: Final = 10_000
MAX_RUIN: Final = 0.05
CUMULATIVE_TRIALS: Final = 415_081  # 385,081 (through wave34) + 30,000 (wave35: 2 arms x 3,000 x 5)


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"seed_{seed}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def genome_from_dict(payload: dict[str, Any]) -> Genome35:
    return Genome35(
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


def main() -> int:
    rows = load_rows()
    if len(rows) != len(SEEDS):
        print(f"시드 결과 {len(rows)}/{len(SEEDS)}개만 존재 — 판정 보류", file=sys.stderr)
        return 1

    scores = {
        arm: [row["arms"][arm]["best_feasible_fitness"] for row in rows] for arm in ARMS
    }
    sa_wins = sum(
        1
        for row in rows
        if row["arms"]["simulated_annealing"]["best_feasible_fitness"]
        > row["arms"]["random"]["best_feasible_fitness"]
    )
    print("=== arm별 성적 (적합도 = log(IS 최종/100)) ===")
    for arm, values in scores.items():
        print(f"  {arm:20s} 중앙 {np.median(values):+7.3f} | " + " ".join(f"{v:+7.3f}" for v in values))
    print(f"  SA가 랜덤을 이긴 시드: {sa_wins}/5")

    ordered = sorted(rows, key=lambda row: row["arms"]["simulated_annealing"]["best_feasible_fitness"])
    chosen = ordered[len(ordered) // 2]
    candidate = chosen["arms"]["simulated_annealing"]["best_feasible"]
    genome = genome_from_dict(candidate["genome"])
    print(f"\n판정 후보: median-seed {chosen['seed']} (SA)")

    cache, _ranked = build_wide_cache()
    result = run_genome(cache, genome, mode="full", sizing=COMPOUNDING)  # THE OOS unsealing
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

    windows = {
        "is": window(total, 0, max(0, oos_start - 1)),
        "oos": window(total, oos_start, last),
        "full": window(total, 0, last),
        "baseline_oos": window(baseline, oos_start, last),
        "baseline_full": window(baseline, 0, last),
    }
    profile = entry_profile(cache, result)
    max_notional = float(max((t.notional_usdt for t in result.trades), default=0.0))

    rng = np.random.default_rng(35_000)
    daily = _daily_returns(total)
    draws = rng.integers(0, len(daily), size=(MC_PATHS, len(daily)))
    finals = (100.0 * np.cumprod(1.0 + daily[draws], axis=1))[:, -1]
    ruin = float((finals < 50.0).mean())

    gates: dict[str, dict] = {}
    gates["W1_method_validity"] = {
        "status": "PASS" if sa_wins >= 4 else "FAIL",
        "sa_wins_over_random": sa_wins,
        "required": 4,
        "sa_median": float(np.median(scores["simulated_annealing"])),
        "random_median": float(np.median(scores["random"])),
    }
    freq_ok = profile.trades_per_active_day >= 1.0 and profile.survived_full_span
    gates["W2_frequency_and_survival"] = {
        "status": "PASS" if freq_ok else "FAIL",
        "trades_per_active_day": profile.trades_per_active_day,
        "n_trades": profile.n_trades,
        "survived_full_span": profile.survived_full_span,
    }
    oos_ok = windows["oos"]["total_return"] > 0.0 and windows["oos"]["annualized"] > windows["baseline_oos"]["annualized"]
    gates["W3_oos"] = {
        "status": "PASS" if oos_ok else "FAIL",
        "candidate_oos": windows["oos"],
        "baseline_oos": windows["baseline_oos"],
    }
    gates["W4_ruin"] = {
        "status": "PASS" if ruin < MAX_RUIN else "FAIL",
        "ruin_probability": ruin,
        "max_ruin_probability": MAX_RUIN,
        "p05_usdt": float(np.percentile(finals, 5)),
        "median_usdt": float(np.median(finals)),
    }
    exec_reasons = []
    if not np.isfinite(result.min_notional_usdt) or result.min_notional_usdt < 5.0:
        exec_reasons.append(f"min notional {result.min_notional_usdt}")
    if genome.leverage > 20.0 + 1e-9:
        exec_reasons.append("leverage over cap")
    if result.n_liquidations != 0:
        exec_reasons.append(f"{result.n_liquidations} liquidations")
    gates["W5_executability"] = {
        "status": "PASS" if not exec_reasons else "FAIL",
        "min_notional_usdt": float(result.min_notional_usdt),
        "leverage": genome.leverage,
        "n_liquidations": result.n_liquidations,
        "reasons": exec_reasons,
    }
    gates["W6_capacity"] = {
        "status": "PASS" if max_notional <= CAPACITY_LIMIT_USDT else "FAIL",
        "max_notional_usdt": max_notional,
        "limit_usdt": CAPACITY_LIMIT_USDT,
    }

    failures = [name for name, body in gates.items() if body["status"] != "PASS"]
    payload = {
        "wave": "wave35_universe",
        "universe": list(genome.symbols),
        "universe_available": len(chosen["universe"]),
        "span": f"{cache.index[0].date()} ~ {cache.index[-1].date()}",
        "chosen_seed": chosen["seed"],
        "candidate": candidate,
        "equity_windows": windows,
        "full_span_entry_profile": profile.as_dict(),
        "max_notional_usdt": max_notional,
        "arm_scores": scores,
        "sa_wins_over_random": sa_wins,
        "cumulative_trials": CUMULATIVE_TRIALS,
        "gates": gates,
        "overall": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "promoted": not failures,
    }
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n유니버스: {len(genome.symbols)}종목 (가용 {len(chosen['universe'])}) | 폭 유전자 선택 = {len(genome.symbols)}")
    print(f"유전자: {genome.signal_family} lb{genome.lookback_bars} stop{genome.stop_pct*100:.3f}% "
          f"R{genome.target_r:.2f} lev{genome.leverage:.3f}x hold{genome.max_hold_bars} "
          f"conc{genome.max_concurrent} sleeve{genome.sleeve_fraction*100:.0f}% "
          f"{'trail' if genome.trail_enabled else 'no-trail'} {'both' if genome.allow_short else 'long-only'}")
    for name, body in windows.items():
        print(f"  {name:14s} ${body['start_usdt']:9,.2f} → ${body['end_usdt']:10,.2f} "
              f"{body['days']:5.0f}일 연 {body['annualized']*100:+9.2f}% MDD {body['mdd']*100:5.1f}%")
    print(f"  진입 {profile.trades_per_active_day:.2f}회/활동일 ({profile.n_trades}거래) | "
          f"청산 {result.n_liquidations}건 | 최대 notional ${max_notional:,.0f}")
    print()
    for name, body in gates.items():
        print(f"[{body['status']}] {name}")
    print(f"\nOVERALL {payload['overall']} | failures {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
