# Wave-30 sweep: leverage x hard-stop, selected on IS only, OOS opened once on the winner.
# Answers two questions the user asked together:
#   (1) with wipe probability driven to ~0, what is the most profit available?
#   (2) how often does ONE entry return +30% (their own historical benchmark)?

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave30_riskcap.sim30 import R30, fidelity_check, simulate, v1_inputs

OUT = Path(__file__).resolve().parent / "results"
IS_END = pd.Timestamp("2025-09-30 23:59:59", tz="UTC")
SLEEVE, STABLE = 25.0, 75.0

# Frozen in SPEC.md before running. No post-hoc additions.
LEVERAGES = [1, 2, 3, 4, 5, 6, 8, 10]
STOPS: list[float | None] = [None, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]

WIPE_CAP = 0.01        # R1
WORST_TRADE_CAP = -0.50  # R2


def monte_carlo(rois: np.ndarray, paths: int = 10000, seed: int = 20260730) -> dict:
    """Bootstrap the realised per-trade ROIs. Vectorised: a zero in the cumprod stays zero."""
    if len(rois) < 5:
        return {"wipe_prob": float("nan"), "p05": float("nan"), "median": float("nan"), "ruin_prob": float("nan")}
    rng = np.random.default_rng(seed)
    draw = rng.choice(rois, size=(paths, len(rois)), replace=True)
    curve = SLEEVE * np.cumprod(np.maximum(0.0, 1.0 + draw), axis=1)
    final = curve[:, -1]
    return {
        "wipe_prob": float((curve.min(axis=1) <= 0.005).mean()),
        "p05": float(np.percentile(final, 5)),
        "median": float(np.median(final)),
        "ruin_prob": float(((final + STABLE) < 50.0).mean()),   # total capital halved
    }


def metrics(res: R30) -> dict:
    rois = np.asarray([t.roi for t in res.trades], dtype=float)
    eq = res.equity.to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    mc = monte_carlo(rois)
    n_big = int((rois >= 0.30).sum())   # the user's benchmark: one entry, +30%
    return {
        "leverage": res.leverage, "stop": res.stop, "final": res.final, "multiple": res.final / SLEEVE,
        "trades": len(rois), "liquidations": sum(1 for t in res.trades if t.reason == "liquidated"),
        "stopped": sum(1 for t in res.trades if t.reason in {"stopped", "stop_gap"}),
        "mdd": float(((eq - peak) / np.maximum(peak, 1e-9)).min()),
        "win_rate": float((rois > 0).mean()) if len(rois) else 0.0,
        "roi_med": float(np.median(rois)) if len(rois) else 0.0,
        "roi_p75": float(np.percentile(rois, 75)) if len(rois) else 0.0,
        "roi_p90": float(np.percentile(rois, 90)) if len(rois) else 0.0,
        "roi_max": float(rois.max()) if len(rois) else 0.0,
        "roi_worst": float(rois.min()) if len(rois) else 0.0,
        "n_ge_30pct": n_big, "share_ge_30pct": n_big / len(rois) if len(rois) else 0.0,
        **{f"mc_{k}": v for k, v in mc.items()},
    }


def passes(m: dict) -> bool:
    return (m["trades"] >= 20 and m["liquidations"] == 0
            and m["mc_wipe_prob"] <= WIPE_CAP and m["roi_worst"] >= WORST_TRADE_CAP)


def main() -> int:
    ok, msg = fidelity_check()
    print(f"[Gate 0 fidelity] {'PASS' if ok else 'FAIL'} — {msg}", flush=True)
    if not ok:
        return 1

    inp = v1_inputs()
    rows = []
    print(f"\n{'lev':>4} {'손절':>6} {'손실상한':>8} {'IS배수':>8} {'거래':>5} {'손절수':>6} {'청산':>5} "
          f"{'MDD':>8} {'최악1회':>8} {'전멸%':>7} {'+30%건수':>8}", flush=True)
    for lev in LEVERAGES:
        for stop in STOPS:
            res = simulate(inp, leverage=float(lev), stop=stop, model="fixed", end_at=IS_END)
            m = metrics(res)
            m["gate_pass"] = passes(m)
            rows.append(m)
            cap = "—" if stop is None else f"{min(stop, 1/lev - 0.005) * lev * 100:5.0f}%"
            mark = "OK" if m["gate_pass"] else "  "
            print(f"{lev:3d}x {'없음' if stop is None else f'{stop*100:4.0f}%':>6} {cap:>8} "
                  f"{m['multiple']:7.2f}x {m['trades']:5d} {m['stopped']:6d} {m['liquidations']:5d} "
                  f"{m['mdd']*100:7.1f}% {m['roi_worst']*100:7.1f}% {m['mc_wipe_prob']*100:6.2f}% "
                  f"{m['n_ge_30pct']:8d}  {mark}", flush=True)

    survivors = [r for r in rows if r["gate_pass"]]
    baseline = next(r for r in rows if r["leverage"] == 1 and r["stop"] is None)
    print(f"\nR1+R2 통과: {len(survivors)}/{len(rows)}셀 | IS 기준선(1x·손절없음) = {baseline['multiple']:.3f}x", flush=True)
    if not survivors:
        (OUT / "sweep.json").write_text(json.dumps({"rows": rows, "winner": None}, indent=2), encoding="utf-8")
        print("승격 없음: 전멸확률 제약을 만족하는 셀이 없다.")
        return 0

    winner = max(survivors, key=lambda r: (r["multiple"], -abs(r["mdd"])))
    print(f"선정: {winner['leverage']:.0f}x / 손절 {winner['stop']} → IS {winner['multiple']:.3f}x", flush=True)

    # --- OOS opened exactly once, on the selected cell only. ---
    oos = metrics(simulate(inp, leverage=winner["leverage"], stop=winner["stop"], model="fixed", start_at=IS_END))
    full = metrics(simulate(inp, leverage=winner["leverage"], stop=winner["stop"], model="fixed"))
    base_full = metrics(simulate(inp, leverage=1.0, stop=None, model="fixed"))
    print(f"OOS  {oos['multiple']:.3f}x | 거래 {oos['trades']} | 최악1회 {oos['roi_worst']*100:.1f}% | +30% {oos['n_ge_30pct']}건")
    print(f"전기간 {full['multiple']:.3f}x  (1x·손절없음 = {base_full['multiple']:.3f}x)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(
        {"rows": rows, "winner_is": winner, "winner_oos": oos, "winner_full": full,
         "baseline_full": base_full, "gates": {"wipe_cap": WIPE_CAP, "worst_trade_cap": WORST_TRADE_CAP}},
        indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
