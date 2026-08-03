# Wave-32: unseal OOS for wave31's L1 candidate, stress it, and rule on gates D1-D8.
#
# This module performs the SECOND use of the same OOS window (wave31 spent the first on its own
# candidate C). SPEC.md registers that fact and forbids continuing to L2..L5 if L1 fails --
# repeatedly opening OOS until something passes converts the holdout from a test into a search.
#
# No searching happens here. The genome is read from wave31's saved artefact and asserted
# unchanged against SPEC.md's frozen table before anything runs.

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

from research.wave30_qd.dataio30 import OOS_SPLIT, build_market_cache, i5_baseline_total_curve
from research.wave30_qd.engine30 import (
    ExecutionStress,
    annualized_return,
    max_drawdown,
    run_genome,
)
from research.wave30_qd.fitness30 import bootstrap_wipe_probability
from research.wave30_qd.gates30 import (
    GateOutcome,
    _daily_returns,
    gate_p5_deflated_sharpe,
    gate_p6_executability,
)
from research.wave30_qd.genome30 import Genome
from research.wave30_qd.run_wave30 import _genome_from_dict
from research.wave31_sprint.fitness31 import FITNESS_WINDOW, sprint_profile, window_statistics

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
L1_SOURCE: Final = (
    Path(__file__).resolve().parents[1] / "wave31_sprint" / "results" / "low_risk_alternative.json"
)

# SPEC.md frozen values. Asserted, not re-derived -- if wave31's artefact ever drifts from the
# pre-registration this run must refuse rather than silently judge a different strategy.
L1_EXPECTED: Final = {
    "signal_family": "breakout",
    "lookback_bars": 168,
    "stop_pct": 0.01461576,
    "target_r": 4.838777,
    "trail_enabled": False,
    "risk_frac": 0.04506704,
    "max_hold_bars": 48,
    "allow_short": True,
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "max_concurrent": 3,
    "cooldown_bars_after_loss": 24,
    "sleeve_fraction": 1.0,
}

STRESS_SCENARIOS: Final = {
    "S0_baseline": ExecutionStress(cost_multiplier=1.0, stop_slippage=0.0),
    "S1_cost_x3": ExecutionStress(cost_multiplier=3.0, stop_slippage=0.0),
    "S2_stop_slip_10bp": ExecutionStress(cost_multiplier=1.0, stop_slippage=0.0010),
    "S3_combined_worst": ExecutionStress(cost_multiplier=3.0, stop_slippage=0.0020),
}
WORST_SCENARIO: Final = "S3_combined_worst"

D3_MAX_RUIN: Final = 0.05
D4_MAX_HALVING: Final = 0.10
D4_MAX_DECIMATION: Final = 0.01
D4_MAX_WIPE: Final = 0.05
D5_CUMULATIVE_TRIALS: Final = 255_681
D7_WORST_YEAR_FLOOR: Final = -0.20
MC_PATHS: Final = 10_000


def load_l1_genome() -> Genome:
    payload = json.loads(L1_SOURCE.read_text(encoding="utf-8"))["L1_genome"]
    for key, expected in L1_EXPECTED.items():
        actual = payload[key]
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-9:
                raise SystemExit(f"L1 genome drift on {key}: {actual} != {expected}")
        elif list(actual) != list(expected) if isinstance(expected, list) else actual != expected:
            raise SystemExit(f"L1 genome drift on {key}: {actual!r} != {expected!r}")
    genome = _genome_from_dict(payload)
    if abs(genome.leverage - 3.083455) > 1e-5:
        raise SystemExit(f"L1 derived leverage drift: {genome.leverage}")
    return genome


def _spans(cache) -> tuple[int, int, int]:
    oos_start = int(cache.daily_index.searchsorted(OOS_SPLIT, side="right"))
    return 0, max(0, oos_start - 1), oos_start


def _curve_window(curve: np.ndarray, start: int, end: int) -> dict:
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


def run_scenario(cache, genome: Genome, stress: ExecutionStress) -> dict[str, Any]:
    result = run_genome(cache, genome, mode="full", stress=stress)
    total = result.total_equity_daily
    _, is_end, oos_start = _spans(cache)
    last = len(total) - 1
    returns = result.trade_returns
    return {
        "is": _curve_window(total, 0, is_end),
        "oos": _curve_window(total, oos_start, last),
        "full": _curve_window(total, 0, last),
        "profile_full": sprint_profile(total),
        "profile_oos": sprint_profile(total[oos_start:]),
        "n_trades": len(result.trades),
        "n_liquidations": result.n_liquidations,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "mean_trade_return": float(returns.mean()) if len(returns) else 0.0,
        "min_notional_usdt": result.min_notional_usdt,
        "max_notional_usdt": float(max((t.notional_usdt for t in result.trades), default=float("nan"))),
        "max_mae": float(max((t.mae for t in result.trades), default=float("nan"))),
        "_result": result,
    }


def gate_d1_oos(candidate_oos: dict, baseline_oos: dict) -> GateOutcome:
    c = candidate_oos["windows"][str(FITNESS_WINDOW)]["p50"]
    b = baseline_oos["windows"][str(FITNESS_WINDOW)]["p50"]
    return GateOutcome(
        "D1_oos_beats_i5",
        "PASS" if c > b else "FAIL",
        {
            "window_days": FITNESS_WINDOW,
            "candidate_oos_median": c,
            "baseline_oos_median": b,
            "gap_pp": (c - b) * 100.0,
            "candidate_oos_p95": candidate_oos["windows"][str(FITNESS_WINDOW)]["p95"],
            "candidate_oos_p05": candidate_oos["windows"][str(FITNESS_WINDOW)]["p05"],
        },
    )


def gate_d2_oos_consistency(candidate_oos: dict) -> GateOutcome:
    stats = candidate_oos["windows"][str(FITNESS_WINDOW)]
    return GateOutcome(
        "D2_oos_consistency",
        "PASS" if stats["positive_share"] > 0.50 else "FAIL",
        {
            "oos_positive_share": stats["positive_share"],
            "minimum": 0.50,
            "oos_windows": stats["n_windows"],
        },
    )


def gate_d3_ruin(total_curve: np.ndarray, seed: int) -> GateOutcome:
    rng = np.random.default_rng(seed)
    daily = _daily_returns(total_curve)
    draws = rng.integers(0, len(daily), size=(MC_PATHS, len(daily)))
    finals = (100.0 * np.cumprod(1.0 + daily[draws], axis=1))[:, -1]
    ruin = float((finals < 50.0).mean())
    return GateOutcome(
        "D3_ruin",
        "PASS" if ruin < D3_MAX_RUIN else "FAIL",
        {
            "paths": MC_PATHS,
            "ruin_probability": ruin,
            "max_ruin_probability": D3_MAX_RUIN,
            "p05_usdt": float(np.percentile(finals, 5)),
            "median_usdt": float(np.median(finals)),
        },
    )


def gate_d4_survival(total_curve: np.ndarray, trade_returns: np.ndarray, seed: int) -> GateOutcome:
    stats = window_statistics(total_curve, FITNESS_WINDOW)
    wipe = bootstrap_wipe_probability(np.asarray(trade_returns, dtype=float), np.random.default_rng(seed), paths=MC_PATHS)
    reasons = []
    if stats["prob_loss_over_50"] >= D4_MAX_HALVING:
        reasons.append(f"P(30d -50%) {stats['prob_loss_over_50']:.4f} >= {D4_MAX_HALVING}")
    if stats["prob_loss_over_90"] >= D4_MAX_DECIMATION:
        reasons.append(f"P(30d -90%) {stats['prob_loss_over_90']:.4f} >= {D4_MAX_DECIMATION}")
    if wipe >= D4_MAX_WIPE:
        reasons.append(f"wipe {wipe:.4f} >= {D4_MAX_WIPE}")
    return GateOutcome(
        "D4_sprint_survival",
        "PASS" if not reasons else "FAIL",
        {
            "prob_loss_over_50": stats["prob_loss_over_50"],
            "prob_loss_over_90": stats["prob_loss_over_90"],
            "sleeve_wipe_probability": wipe,
            "reasons": reasons,
        },
    )


def gate_d7_worst_year(total_curve: np.ndarray, daily_index, baseline: np.ndarray) -> GateOutcome:
    import pandas as pd  # noqa: PANDAS_OK

    frame = pd.DataFrame({"c": total_curve, "b": baseline}, index=pd.DatetimeIndex(daily_index))
    rows = []
    for year, group in frame.groupby(frame.index.year):
        if len(group) < 2:
            continue
        rows.append(
            {
                "year": int(year),
                "candidate_return": float(group["c"].iloc[-1] / group["c"].iloc[0] - 1.0),
                "baseline_return": float(group["b"].iloc[-1] / group["b"].iloc[0] - 1.0),
            }
        )
    worst = min(rows, key=lambda r: r["candidate_return"])
    return GateOutcome(
        "D7_worst_year",
        "PASS" if worst["candidate_return"] >= D7_WORST_YEAR_FLOOR else "FAIL",
        {
            "worst_year": worst["year"],
            "worst_year_return": worst["candidate_return"],
            "floor": D7_WORST_YEAR_FLOOR,
            "baseline_same_year": worst["baseline_return"],
            "per_year": rows,
        },
    )


def gate_d8_stress(scenarios: dict[str, dict]) -> GateOutcome:
    worst = scenarios[WORST_SCENARIO]
    full_return = worst["full"]["total_return"]
    oos_median = worst["profile_oos"]["windows"][str(FITNESS_WINDOW)]["p50"]
    liquidations = worst["n_liquidations"]
    reasons = []
    if full_return <= 0.0:
        reasons.append(f"S3 full-period total return {full_return:.4f} <= 0")
    if oos_median <= 0.0:
        reasons.append(f"S3 OOS 30d median {oos_median:.4f} <= 0")
    if liquidations != 0:
        reasons.append(f"S3 liquidations {liquidations} != 0")
    return GateOutcome(
        "D8_stress_durability",
        "PASS" if not reasons else "FAIL",
        {
            "scenario": WORST_SCENARIO,
            "full_total_return": full_return,
            "full_annualized": worst["full"]["annualized"],
            "oos_30d_median": oos_median,
            "n_liquidations": liquidations,
            "n_trades": worst["n_trades"],
            "win_rate": worst["win_rate"],
            "mean_trade_return": worst["mean_trade_return"],
            "reasons": reasons,
            "all_scenarios": {
                name: {
                    "full_total_return": data["full"]["total_return"],
                    "full_annualized": data["full"]["annualized"],
                    "full_mdd": data["full"]["mdd"],
                    "oos_annualized": data["oos"]["annualized"],
                    "oos_30d_median": data["profile_oos"]["windows"][str(FITNESS_WINDOW)]["p50"],
                    "n_trades": data["n_trades"],
                    "n_liquidations": data["n_liquidations"],
                    "win_rate": data["win_rate"],
                    "mean_trade_return": data["mean_trade_return"],
                }
                for name, data in scenarios.items()
            },
        },
    )


def main() -> int:
    cache = build_market_cache()
    genome = load_l1_genome()
    print(f"L1 genome verified against SPEC.md: {genome.signal_family} "
          f"lev {genome.leverage:.6f}x band {genome.liquidation_band:.6f} symbols {genome.symbols}")

    scenarios = {name: run_scenario(cache, genome, stress) for name, stress in STRESS_SCENARIOS.items()}
    base = scenarios["S0_baseline"]
    result = base.pop("_result")
    for data in scenarios.values():
        data.pop("_result", None)

    total = result.total_equity_daily
    baseline_curve = i5_baseline_total_curve(cache)
    _, is_end, oos_start = _spans(cache)
    baseline_oos_profile = sprint_profile(baseline_curve[oos_start:])

    outcomes = [
        gate_d1_oos(base["profile_oos"], baseline_oos_profile),
        gate_d2_oos_consistency(base["profile_oos"]),
        gate_d3_ruin(total, 32_000),
        gate_d4_survival(total, result.trade_returns, 32_001),
        gate_p5_deflated_sharpe(total, cache.daily_index, trials=D5_CUMULATIVE_TRIALS),
        gate_p6_executability(genome, base["min_notional_usdt"], base["n_trades"]),
        gate_d7_worst_year(total, cache.daily_index, baseline_curve),
        gate_d8_stress(scenarios),
    ]
    renamed = {"P5_deflated_sharpe": "D5_deflated_sharpe", "P6_executability": "D6_executability"}
    gates: dict[str, Any] = {}
    failures: list[str] = []
    for outcome in outcomes:
        name = renamed.get(outcome.name, outcome.name)
        detail = dict(outcome.detail)
        if name == "D6_executability" and base["n_liquidations"] != 0:
            outcome = GateOutcome(name, "FAIL", {**detail, "liquidations": base["n_liquidations"]})
            detail = outcome.detail
        gates[name] = {"status": outcome.status, **detail}
        if outcome.status != "PASS":
            failures.append(name)

    payload = {
        "wave": "wave32_deploy",
        "candidate": "L1",
        "genome": genome.to_dict(),
        "oos_unsealing": "second use of the 2025-10-01~ window (wave31 spent the first); trials 255,681",
        "baseline_oos_30d_median": baseline_oos_profile["windows"][str(FITNESS_WINDOW)]["p50"],
        "scenarios": scenarios,
        "gates": gates,
        "overall": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "promoted": not failures,
        "paper_authorised": not failures,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"{'scenario':>20} {'full ret':>10} {'full ann':>10} {'MDD':>8} {'OOS ann':>10} "
          f"{'OOS 30d p50':>12} {'trades':>7} {'liq':>4} {'win%':>6} {'mean/trade':>11}")
    for name, data in scenarios.items():
        print(f"{name:>20} {data['full']['total_return']*100:9.1f}% {data['full']['annualized']*100:9.2f}% "
              f"{data['full']['mdd']*100:7.1f}% {data['oos']['annualized']*100:9.2f}% "
              f"{data['profile_oos']['windows'][str(FITNESS_WINDOW)]['p50']*100:11.2f}% "
              f"{data['n_trades']:7d} {data['n_liquidations']:4d} {data['win_rate']*100:5.1f}% "
              f"{data['mean_trade_return']*100:+10.3f}%")
    print()
    for name, body in gates.items():
        print(f"[{body['status']}] {name}")
    print(f"\nOVERALL {payload['overall']} | failures {failures}")
    print(f"paper 전진검증 승인: {payload['paper_authorised']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
