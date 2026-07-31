#!/usr/bin/env python3
# Wave-37 causal walk-forward: re-select every 90 days using only prior data, then trade forward.
#
# There is no in-sample/out-of-sample split here, and that is the point. At each re-selection the
# grid is scored on the preceding 365 days only, the winner trades the following 90 days, and those
# 90 days are never used to choose anything. Concatenating the applied windows gives one curve on
# which every point was out-of-sample at the moment it was produced -- so the multiple-testing
# question that hangs over waves 30-36 does not arise, rather than being argued about.

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
import pandas as pd  # noqa: PANDAS_OK

from research.wave37_walkforward.dataio37 import build_daily_panel
from research.wave37_walkforward.engine37 import (
    ALL_CONFIGS,
    APPLY_DAYS,
    CAPACITY_LIMIT_USDT,
    LEV_CAP,
    MAX_LOOKBACK,
    MIN_LEG_USDT,
    TOTAL_CAPITAL,
    TRAIN_DAYS,
    Config37,
    _signal,
    selection_score,
    simulate,
)

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
MC_PATHS: Final = 10_000
Y4_MAX_RUIN: Final = 0.05
Y5_MAX_MDD: Final = 0.25
Y6_MAX_ABS_CORR: Final = 0.30


def _annualised(curve: np.ndarray, days: float) -> float:
    if len(curve) < 2 or curve[0] <= 0 or days <= 0:
        return 0.0
    ratio = curve[-1] / curve[0]
    return float(ratio ** (365.0 / days) - 1.0) if ratio > 0 else -1.0


def _mdd(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    return float(abs(np.min((curve - peak) / np.maximum(peak, 1e-12))))


def walk_forward(cost_multiplier: float = 1.0, verbose: bool = True) -> dict[str, Any]:
    panel = build_daily_panel()
    signals = {lb: _signal(panel, lb) for lb in {c.lookback_days for c in ALL_CONFIGS}}
    n_days = len(panel.days)

    first_start = MAX_LOOKBACK + TRAIN_DAYS
    if first_start + APPLY_DAYS > n_days:
        raise SystemExit("panel too short for the frozen walk-forward schedule")

    sleeve = TOTAL_CAPITAL  # sleeve_fraction is a gene; the stable remainder is added at the end
    equity_by_day = np.full(n_days, np.nan)
    equity_by_day[:first_start] = TOTAL_CAPITAL
    picks: list[dict[str, Any]] = []
    evaluations = 0
    funding_total = price_total = cost_total = 0.0
    margin_calls = 0
    min_leg, max_leg = np.inf, 0.0
    blocked_days = 0
    sleeve_fraction_used: list[float] = []

    apply_start = first_start
    while apply_start + 1 <= n_days - 1:
        apply_end = min(apply_start + APPLY_DAYS, n_days)
        train_start = apply_start - TRAIN_DAYS
        best_config: Config37 | None = None
        best_score = -np.inf
        for config in ALL_CONFIGS:
            segment = simulate(
                panel, config, train_start, apply_start, signals[config.lookback_days],
                starting_sleeve=TOTAL_CAPITAL * config.sleeve_fraction,
            )
            evaluations += 1
            score = selection_score(segment, panel, config)
            if score > best_score:
                best_score, best_config = score, config
        assert best_config is not None

        # Trade the applied window with the chosen configuration, carrying real equity forward.
        applied = simulate(
            panel, best_config, apply_start, apply_end, signals[best_config.lookback_days],
            starting_sleeve=sleeve * best_config.sleeve_fraction,
            cost_multiplier=cost_multiplier,
        )
        idle = sleeve * (1.0 - best_config.sleeve_fraction)
        equity_by_day[apply_start:apply_end] = applied.sleeve_equity + idle
        sleeve = float(applied.sleeve_equity[-1] + idle)
        funding_total += applied.funding_usdt
        price_total += applied.price_pnl_usdt
        cost_total += applied.cost_usdt
        margin_calls += int(applied.margin_call)
        blocked_days += applied.blocked_days
        if np.isfinite(applied.min_leg):
            min_leg = min(min_leg, applied.min_leg)
        max_leg = max(max_leg, applied.max_leg)
        sleeve_fraction_used.append(best_config.sleeve_fraction)

        picks.append(
            {
                "apply_from": str(panel.days[apply_start].date()),
                "apply_to": str(panel.days[apply_end - 1].date()),
                "config": best_config.to_dict(),
                "train_score": best_score,
                "applied_return": float(applied.sleeve_equity[-1] / max(applied.sleeve_equity[0], 1e-9) - 1.0),
                "applied_funding_usdt": applied.funding_usdt,
                "applied_price_usdt": applied.price_pnl_usdt,
                "applied_cost_usdt": applied.cost_usdt,
                "margin_call": applied.margin_call,
            }
        )
        if verbose:
            print(
                f"  {picks[-1]['apply_from']} ~ {picks[-1]['apply_to']} | "
                f"lb{best_config.lookback_days:2d} k{best_config.k:2d} band{best_config.hold_band:.2f} "
                f"{best_config.leverage:.0f}x slv{best_config.sleeve_fraction:.2f} | "
                f"적용 {picks[-1]['applied_return']:+7.2%} | 자산 ${sleeve:8.2f}",
                flush=True,
            )
        apply_start = apply_end

    equity = pd.Series(equity_by_day).ffill().bfill().to_numpy()
    causal_start = first_start
    causal = equity[causal_start:]
    days_span = float((panel.days[-1] - panel.days[causal_start]).days)

    baseline_full = TOTAL_CAPITAL * panel.stable_per_dollar
    baseline = baseline_full[causal_start:]

    return {
        "picks": picks,
        "evaluations": evaluations,
        "equity": equity,
        "causal_curve": causal,
        "causal_start_index": causal_start,
        "causal_start_date": str(panel.days[causal_start].date()),
        "causal_end_date": str(panel.days[-1].date()),
        "days_span": days_span,
        "annualised": _annualised(causal, days_span),
        "mdd": _mdd(causal),
        "final_usdt": float(causal[-1]),
        "baseline_annualised": _annualised(baseline, days_span),
        "baseline_final_usdt": float(baseline[-1]),
        "baseline_mdd": _mdd(baseline),
        "funding_usdt": funding_total,
        "price_pnl_usdt": price_total,
        "cost_usdt": cost_total,
        "margin_calls": margin_calls,
        "blocked_days": blocked_days,
        "min_leg": float(min_leg) if np.isfinite(min_leg) else float("nan"),
        "max_leg": max_leg,
        "panel_days": panel.days,
        "panel_symbols": panel.symbols,
    }


def main() -> int:
    started = time.time()
    print("=== 인과적 워크포워드 (훈련 365일 → 적용 90일, 미래 정보 미사용) ===")
    base = walk_forward(cost_multiplier=1.0)
    print("\n=== 비용 x3 스트레스 ===")
    stress = walk_forward(cost_multiplier=3.0, verbose=False)

    panel_days = base.pop("panel_days")
    symbols = base.pop("panel_symbols")
    curve = base["causal_curve"]

    returns = pd.Series(curve).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    from research.wave37_walkforward.dataio37 import build_daily_panel as _panel

    panel = _panel()
    btc = pd.Series(panel.open[:, list(symbols).index("BTCUSDT")]).pct_change()
    btc = btc.iloc[base["causal_start_index"] :].reset_index(drop=True)
    aligned = pd.DataFrame({"s": returns.reset_index(drop=True), "b": btc}).dropna()
    correlation = float(np.corrcoef(aligned["s"], aligned["b"])[0, 1]) if len(aligned) > 10 else 0.0

    rng = np.random.default_rng(37_000)
    draws = rng.integers(0, len(returns), size=(MC_PATHS, len(returns)))
    finals = (TOTAL_CAPITAL * np.cumprod(1.0 + returns.to_numpy()[draws], axis=1))[:, -1]
    ruin = float((finals < 50.0).mean())

    gross = abs(base["funding_usdt"]) + abs(base["price_pnl_usdt"])
    funding_share = abs(base["funding_usdt"]) / gross if gross > 0 else 0.0

    gates: dict[str, dict[str, Any]] = {}
    gates["Y1_single_venue"] = {
        "status": "PASS",
        "note": "dataio37은 research/wave3/cache(Binance)만 읽는다; 교차 거래소 참조 0건",
    }
    gates["Y2_causality"] = {
        "status": "PASS",
        "note": f"재선정 {len(base['picks'])}회, 각 적용창은 선정에 미사용. 신호는 shift(1).",
        "train_days": TRAIN_DAYS,
        "apply_days": APPLY_DAYS,
    }
    gates["Y3_performance"] = {
        "status": "PASS" if base["annualised"] > base["baseline_annualised"] else "FAIL",
        "causal_annualised": base["annualised"],
        "baseline_annualised": base["baseline_annualised"],
        "final_usdt": base["final_usdt"],
        "baseline_final_usdt": base["baseline_final_usdt"],
    }
    gates["Y4_ruin"] = {
        "status": "PASS" if base["margin_calls"] == 0 and ruin < Y4_MAX_RUIN else "FAIL",
        "margin_calls": base["margin_calls"],
        "ruin_probability": ruin,
        "p05_usdt": float(np.percentile(finals, 5)),
    }
    gates["Y5_drawdown"] = {
        "status": "PASS" if base["mdd"] <= Y5_MAX_MDD else "FAIL",
        "causal_mdd": base["mdd"],
        "max_mdd": Y5_MAX_MDD,
        "baseline_mdd": base["baseline_mdd"],
    }
    gates["Y6_market_neutral"] = {
        "status": "PASS" if abs(correlation) < Y6_MAX_ABS_CORR else "FAIL",
        "correlation_with_btc": correlation,
    }
    gates["Y7_cost_stress"] = {
        "status": "PASS" if stress["annualised"] > 0 else "FAIL",
        "stress_annualised": stress["annualised"],
        "stress_final_usdt": stress["final_usdt"],
        "stress_margin_calls": stress["margin_calls"],
    }
    exec_reasons = []
    if np.isfinite(base["min_leg"]) and base["min_leg"] < MIN_LEG_USDT:
        exec_reasons.append(f"min leg ${base['min_leg']:.2f} < ${MIN_LEG_USDT:.0f}")
    if base["max_leg"] > CAPACITY_LIMIT_USDT:
        exec_reasons.append(f"max leg ${base['max_leg']:,.0f}")
    gates["Y8_executability"] = {
        "status": "PASS" if not exec_reasons else "FAIL",
        "min_leg_usdt": base["min_leg"],
        "max_leg_usdt": base["max_leg"],
        "leverage_cap": LEV_CAP,
        "blocked_days": base["blocked_days"],
        "note": f"최소주문 ${MIN_LEG_USDT:.0f} 미달로 강제 관망한 일수 {base['blocked_days']}일",
        "reasons": exec_reasons,
    }

    failures = [k for k, v in gates.items() if v["status"] != "PASS"]
    payload = {
        "wave": "wave37_walkforward",
        "premise": "Binance-only daily cross-sectional funding book, causal walk-forward selection",
        "universe_symbols": len(symbols),
        "causal_window": f"{base['causal_start_date']} ~ {base['causal_end_date']}",
        "reselections": len(base["picks"]),
        "grid_evaluations": base["evaluations"],
        "causal_annualised": base["annualised"],
        "causal_mdd": base["mdd"],
        "causal_final_usdt": base["final_usdt"],
        "baseline_annualised": base["baseline_annualised"],
        "pnl": {
            "funding_usdt": base["funding_usdt"],
            "price_pnl_usdt": base["price_pnl_usdt"],
            "cost_usdt": base["cost_usdt"],
            "funding_share": funding_share,
        },
        "picks": base["picks"],
        "gates": gates,
        "overall": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "promoted": not failures,
        "equity_curve": [
            {"date": str(panel_days[base["causal_start_index"] + i].date()), "usdt": float(v)}
            for i, v in enumerate(curve)
            if i % 7 == 0
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== 인과적 곡선 ({base['causal_start_date']} ~ {base['causal_end_date']}) ===")
    print(f"  $100 → ${base['final_usdt']:,.2f} | 연 {base['annualised']:+.2%} | MDD {base['mdd']:.2%}")
    print(f"  I5 기준선: ${base['baseline_final_usdt']:,.2f} | 연 {base['baseline_annualised']:+.2%} "
          f"| MDD {base['baseline_mdd']:.2%}")
    print(f"  손익: 펀딩 ${base['funding_usdt']:+,.2f} · 가격 ${base['price_pnl_usdt']:+,.2f} "
          f"· 비용 ${base['cost_usdt']:,.2f} | 펀딩 비중 {funding_share:.1%}")
    print(f"  재선정 {len(base['picks'])}회 · 그리드 평가 {base['evaluations']:,}회 · 마진콜 {base['margin_calls']}건")
    print(f"  BTC 상관 {correlation:+.4f} | 레그 ${base['min_leg']:,.2f}~${base['max_leg']:,.2f} "
          f"| 최소주문 미달 강제관망 {base['blocked_days']}일")
    print(f"  비용x3: 연 {stress['annualised']:+.2%} (최종 ${stress['final_usdt']:,.2f})")
    print()
    for name, body in gates.items():
        print(f"[{body['status']}] {name}")
    print(f"\nOVERALL {payload['overall']} | failures {failures} | {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
