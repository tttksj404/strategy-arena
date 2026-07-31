#!/usr/bin/env python3
# Wave-36 judgement: selection from the full grid, then THE single OOS unsealing, then gates X1-X7.
#
# X2 (market neutrality) is the gate that decides whether this wave's entire premise is true. The
# engine smoke test already showed price P&L exceeding funding P&L, which means the book is NOT a
# pure funding harvester -- shorting the highest-funding perpetual is also implicitly shorting
# crowded longs. That can be a real effect, but if the result turns out to be market BETA in
# disguise then the strategy is just leveraged direction wearing a costume, and every earlier wave
# already showed where that ends.

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
import pandas as pd  # noqa: PANDAS_OK

from research.wave35_universe.dataio35 import build_wide_cache
from research.wave36_crosssection.engine36 import (
    CAPACITY_LIMIT_USDT,
    LEV_CAP,
    MIN_LEG_USDT,
    Config36,
    build_stamp_panel,
    run_config,
)

MC_PATHS: Final = 10_000
X2_MAX_ABS_CORRELATION: Final = 0.30
X4_MAX_RUIN: Final = 0.05
X5_MAX_MDD: Final = 0.25
CUMULATIVE_TRIALS: Final = 424_801  # 415,081 through wave35 + 9,720 grid combinations
RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"


def _annualised(curve: np.ndarray, stamps: pd.DatetimeIndex, start: int, end: int) -> float:
    if end <= start or curve[start] <= 0:
        return 0.0
    years = (stamps[end] - stamps[start]).days / 365.25
    ratio = curve[end] / curve[start]
    return float(ratio ** (1.0 / years) - 1.0) if ratio > 0 and years > 0 else -1.0


def _mdd(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(abs(np.min((curve - peak) / np.maximum(peak, 1e-12))))


def judge() -> dict[str, Any]:
    from research.wave36_crosssection.run_wave36 import LOOKBACKS, TOTAL_COMBINATIONS, load_all

    rows = load_all()
    if len(rows) != TOTAL_COMBINATIONS:
        print(f"그리드 미완: {len(rows)}/{TOTAL_COMBINATIONS}", file=sys.stderr)
        return {}
    feasible = [r for r in rows if not r["infeasible_reasons"]]
    chosen = max(feasible, key=lambda r: r["fitness"])
    config = Config36(**chosen["config"]).validate()
    print(f"전수 {len(rows):,}조합 · 제약충족 {len(feasible):,} · 선정 적합도 {chosen['fitness']:+.4f}")
    print(f"선정 설정: {config.to_dict()}\n")

    cache, symbols = build_wide_cache()
    panel = build_stamp_panel(cache)
    n_is = int(panel.is_mask.sum())
    last = len(panel.stamps) - 1

    full = run_config(panel, config, is_only=False)          # THE OOS unsealing
    stressed = run_config(panel, config, cost_multiplier=3.0, is_only=False)

    baseline = 100.0 * panel.stable_per_dollar[panel.day_of_stamp]
    windows = {}
    for label, (start, end) in {
        "is": (0, n_is - 1),
        "oos": (n_is - 1, last),
        "full": (0, last),
    }.items():
        windows[label] = {
            "start_usdt": float(full.total_equity[start]),
            "end_usdt": float(full.total_equity[end]),
            "annualised": _annualised(full.total_equity, panel.stamps, start, end),
            "mdd": _mdd(full.total_equity[start : end + 1]),
        }
    windows["baseline_oos"] = {
        "start_usdt": float(baseline[n_is - 1]),
        "end_usdt": float(baseline[last]),
        "annualised": _annualised(baseline, panel.stamps, n_is - 1, last),
        "mdd": _mdd(baseline[n_is - 1 : last + 1]),
    }
    windows["baseline_full"] = {
        "start_usdt": float(baseline[0]),
        "end_usdt": float(baseline[last]),
        "annualised": _annualised(baseline, panel.stamps, 0, last),
        "mdd": _mdd(baseline),
    }

    # ---- X2: is it actually market neutral? ----
    equity = pd.Series(full.total_equity, index=panel.stamps)
    strategy_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    btc_price = pd.Series(panel.price[:, list(symbols).index("BTCUSDT")], index=panel.stamps)
    btc_returns = btc_price.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    common = strategy_returns.index.intersection(btc_returns.index)
    correlation = float(np.corrcoef(strategy_returns.loc[common], btc_returns.loc[common])[0, 1])
    beta = float(
        np.polyfit(btc_returns.loc[common], strategy_returns.loc[common], 1)[0]
    )

    # ---- X4: ruin ----
    rng = np.random.default_rng(36_000)
    draws = rng.integers(0, len(strategy_returns), size=(MC_PATHS, len(strategy_returns)))
    finals = (100.0 * np.cumprod(1.0 + strategy_returns.to_numpy()[draws], axis=1))[:, -1]
    ruin = float((finals < 50.0).mean())

    funding_share = (
        full.funding_collected_usdt / (abs(full.funding_collected_usdt) + abs(full.price_pnl_usdt))
        if (abs(full.funding_collected_usdt) + abs(full.price_pnl_usdt)) > 0
        else 0.0
    )
    gates: dict[str, dict[str, Any]] = {}
    gates["X1_probe_reproduction"] = {
        "status": "PASS" if full.funding_collected_usdt > 0 else "FAIL",
        "funding_usdt": full.funding_collected_usdt,
        "price_pnl_usdt": full.price_pnl_usdt,
        "cost_usdt": full.cost_paid_usdt,
        "funding_share_of_gross_pnl": funding_share,
        "note": "프로브는 펀딩만 계산했다. 엔진의 펀딩 부호가 양수여야 프로브와 정합.",
    }
    gates["X2_market_neutrality"] = {
        "status": "PASS" if abs(correlation) < X2_MAX_ABS_CORRELATION else "FAIL",
        "correlation_with_btc": correlation,
        "beta_to_btc": beta,
        "max_abs_correlation": X2_MAX_ABS_CORRELATION,
    }
    gates["X3_oos"] = {
        "status": "PASS" if windows["oos"]["annualised"] > windows["baseline_oos"]["annualised"] else "FAIL",
        "candidate_oos_annualised": windows["oos"]["annualised"],
        "baseline_oos_annualised": windows["baseline_oos"]["annualised"],
        "oos_start_usdt": windows["oos"]["start_usdt"],
        "oos_end_usdt": windows["oos"]["end_usdt"],
    }
    gates["X4_ruin"] = {
        "status": "PASS" if ruin < X4_MAX_RUIN and not full.margin_call else "FAIL",
        "ruin_probability": ruin,
        "max_ruin_probability": X4_MAX_RUIN,
        "margin_call": full.margin_call,
        "p05_usdt": float(np.percentile(finals, 5)),
        "median_usdt": float(np.median(finals)),
    }
    gates["X5_drawdown"] = {
        "status": "PASS" if windows["full"]["mdd"] <= X5_MAX_MDD else "FAIL",
        "full_mdd": windows["full"]["mdd"],
        "max_mdd": X5_MAX_MDD,
        "is_mdd": windows["is"]["mdd"],
        "oos_mdd": windows["oos"]["mdd"],
    }
    exec_reasons = []
    if not np.isfinite(full.min_leg_notional) or full.min_leg_notional < MIN_LEG_USDT:
        exec_reasons.append(f"min leg ${full.min_leg_notional}")
    if config.leverage > LEV_CAP:
        exec_reasons.append("leverage over cap")
    if full.max_leg_notional > CAPACITY_LIMIT_USDT:
        exec_reasons.append(f"max leg ${full.max_leg_notional:,.0f} over capacity")
    gates["X6_executability"] = {
        "status": "PASS" if not exec_reasons else "FAIL",
        "min_leg_notional": full.min_leg_notional,
        "max_leg_notional": full.max_leg_notional,
        "gross_leverage": config.leverage,
        "reasons": exec_reasons,
    }
    stress_full = _annualised(stressed.total_equity, panel.stamps, 0, last)
    stress_oos = _annualised(stressed.total_equity, panel.stamps, n_is - 1, last)
    gates["X7_cost_stress"] = {
        "status": "PASS" if stress_full > 0 and stress_oos > 0 else "FAIL",
        "stress_full_annualised": stress_full,
        "stress_oos_annualised": stress_oos,
        "stress_margin_call": stressed.margin_call,
        "stress_cost_usdt": stressed.cost_paid_usdt,
    }

    failures = [name for name, body in gates.items() if body["status"] != "PASS"]
    payload = {
        "wave": "wave36_crosssection",
        "premise": "cross-sectional funding dispersion, dollar-neutral, no directional prediction",
        "grid_combinations": len(rows),
        "feasible_combinations": len(feasible),
        "cumulative_trials": CUMULATIVE_TRIALS,
        "selected": chosen,
        "config": config.to_dict(),
        "windows": windows,
        "pnl_decomposition": {
            "funding_usdt": full.funding_collected_usdt,
            "price_pnl_usdt": full.price_pnl_usdt,
            "cost_usdt": full.cost_paid_usdt,
            "rotations": full.n_rotations,
            "stamps_in_market": full.stamps_in_market,
        },
        "gates": gates,
        "overall": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "promoted": not failures,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for label in ("is", "oos", "full", "baseline_oos", "baseline_full"):
        w = windows[label]
        print(f"  {label:14s} ${w['start_usdt']:9,.2f} → ${w['end_usdt']:11,.2f} "
              f"연 {w['annualised']*100:+9.2f}%  MDD {w['mdd']*100:6.2f}%")
    print(f"\n  손익 분해: 펀딩 ${full.funding_collected_usdt:+,.2f} · 가격 ${full.price_pnl_usdt:+,.2f} "
          f"· 비용 ${full.cost_paid_usdt:,.2f} | 펀딩 비중 {funding_share:.1%}")
    print(f"  BTC 상관 {correlation:+.4f} · 베타 {beta:+.4f}")
    print(f"  레그 노셔널 ${full.min_leg_notional:,.2f} ~ ${full.max_leg_notional:,.2f} | 마진콜 {full.margin_call}")
    print()
    for name, body in gates.items():
        print(f"[{body['status']}] {name}")
    print(f"\nOVERALL {payload['overall']} | failures {failures}")
    return payload


if __name__ == "__main__":
    judge()
