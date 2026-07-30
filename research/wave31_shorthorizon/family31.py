# Wave-31 part 2: re-score the whole wave-20 candidate family under the user's OWN objective.
#
# wave-20 ranked V1..V5 by total multiple on a $25 sleeve and promoted V1. But the user's stated
# principle is different: shortest horizon, full $100 deployed, and dollars per entry. V3 (first
# 7 days of a new listing) and V4 (liquidation-cascade bounce on 1H bars) are structurally SHORT
# and have never been scored that way. No new parameters are fitted here -- these are the frozen
# wave-20 engines, so this adds no overfitting burden.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave20_convex import engine20

OUT = Path(__file__).resolve().parent / "results"
CAPITAL = 100.0
IS_END = pd.Timestamp("2025-09-30 23:59:59", tz="UTC")


def score(name: str, res) -> dict:
    # engine20's Trade names this pnl_fraction (sim30's T30 calls the same thing roi).
    rois = np.asarray([t.pnl_fraction for t in res.trades], dtype=float)
    days = np.asarray([(t.exit_time - t.entry_time).total_seconds() / 86400.0 for t in res.trades])
    eq = res.equity.to_numpy(dtype=float)
    eq = eq[np.isfinite(eq)]
    peak = np.maximum.accumulate(eq)
    span = max((res.equity.index[-1] - res.equity.index[0]).days / 365.25, 1e-9)
    mult = float(eq[-1]) / float(eq[0]) if len(eq) and eq[0] > 0 else 0.0
    big = rois >= 0.30
    # Bootstrap the per-trade ROIs to get a wipe probability on a FULL $100 account.
    wipe = float("nan")
    if len(rois) >= 5:
        rng = np.random.default_rng(20260731)
        draw = rng.choice(rois, size=(10000, len(rois)), replace=True)
        curve = CAPITAL * np.cumprod(np.maximum(0.0, 1.0 + draw), axis=1)
        wipe = float((curve.min(axis=1) <= 0.005).mean())
    return {
        "candidate": name, "multiple": mult,
        "cagr": (mult ** (1.0 / span) - 1.0) if mult > 0 else -1.0,
        "trades": len(rois), "win_rate": float((rois > 0).mean()) if len(rois) else 0.0,
        "median_hold_days": float(np.median(days)) if len(days) else 0.0,
        "avg_usd_per_trade": float((rois * CAPITAL).mean()) if len(rois) else 0.0,
        "best_usd": float(rois.max() * CAPITAL) if len(rois) else 0.0,
        "worst_usd": float(rois.min() * CAPITAL) if len(rois) else 0.0,
        "n_ge_30pct": int(big.sum()), "hold_of_big": float(np.median(days[big])) if big.any() else 0.0,
        "mdd": float(((eq - peak) / np.maximum(peak, 1e-9)).min()) if len(eq) else 0.0,
        "wipe_prob": wipe,
        "exposure_share": float(days.sum() / max((res.equity.index[-1] - res.equity.index[0]).days, 1)),
    }


def main() -> int:
    runners = {"V1 양방향돌파": engine20.run_v1, "V2 꼬리사냥": engine20.run_v2,
               "V3 신규상장7일": engine20.run_v3, "V4 청산캐스케이드": engine20.run_v4,
               "V5 복권바스켓": engine20.run_v5}
    rows = []
    print(f"{'후보':>16} {'배수':>7} {'CAGR':>8} {'거래':>5} {'승률':>6} {'중앙보유':>8} "
          f"{'평균$/거래':>10} {'최고$':>8} {'최악$':>8} {'+30%':>5} {'MDD':>8} {'전멸%':>7}", flush=True)
    for name, fn in runners.items():
        try:
            m = score(name, fn())
        except Exception as exc:  # a candidate whose data is unavailable must not kill the sweep
            print(f"{name:>16}  실행 실패: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(m)
        print(f"{name:>16} {m['multiple']:6.2f}x {m['cagr']*100:7.1f}% {m['trades']:5d} "
              f"{m['win_rate']*100:5.1f}% {m['median_hold_days']:7.2f}일 {m['avg_usd_per_trade']:9.2f}$ "
              f"{m['best_usd']:7.2f}$ {m['worst_usd']:7.2f}$ {m['n_ge_30pct']:5d} "
              f"{m['mdd']*100:7.1f}% {m['wipe_prob']*100:6.2f}%", flush=True)

    print("\n(모든 금액은 자본 $100 전액 투입 기준 — ROI 10% = $10)", flush=True)
    short = [r for r in rows if r["median_hold_days"] <= 7.0]
    if short:
        best = max(short, key=lambda r: r["cagr"])
        print(f"\n중앙보유 ≤7일 후보 {len(short)}개 중 최고: {best['candidate']} "
              f"CAGR {best['cagr']*100:.1f}%, 평균 {best['avg_usd_per_trade']:.2f}$/거래, "
              f"전멸확률 {best['wipe_prob']*100:.2f}%", flush=True)
    else:
        print("\n중앙보유 ≤7일인 후보가 하나도 없다.", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
