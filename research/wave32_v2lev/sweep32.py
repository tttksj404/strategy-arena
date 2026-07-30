# Wave-32 sweep: V2 leverage 1 -> 5 under the fixed-notional model with intrabar liquidation.
# Gate 0 runs first and hard-stops the sweep on FAIL (wave-30 caught a real modelling error here).
# Metric conventions are wave-31's family31.score verbatim so the rows are comparable across waves.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave32_v2lev import sim32
from research.wave32_v2lev.sim32 import CAPITAL, R32

OUT = Path(__file__).resolve().parent / "results"
LEVERAGES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
MC_PATHS = 10_000
MC_SEED = 20260731          # same seed family as wave-31 so the bootstrap is comparable
WIPE_FLOOR = 0.005          # "equity ever <= $0.005"


def score(res: R32) -> dict:
    rois = np.asarray([t.roi for t in res.trades], dtype=float)
    days = np.asarray([(t.exit_time - t.entry_time).total_seconds() / 86400.0 for t in res.trades])
    eq = res.equity.to_numpy(dtype=float)
    eq = eq[np.isfinite(eq)]
    peak = np.maximum.accumulate(eq)
    span = max((res.equity.index[-1] - res.equity.index[0]).days / 365.25, 1e-9)
    mult = float(eq[-1]) / float(eq[0]) if len(eq) and eq[0] > 0 else 0.0
    big = rois >= 0.30

    wipe = float("nan")
    if len(rois) >= 5:
        rng = np.random.default_rng(MC_SEED)
        draw = rng.choice(rois, size=(MC_PATHS, len(rois)), replace=True)
        curve = CAPITAL * np.cumprod(np.maximum(0.0, 1.0 + draw), axis=1)
        wipe = float((curve.min(axis=1) <= WIPE_FLOOR).mean())

    return {
        "leverage": res.leverage, "model": res.model,
        "final_usd": float(eq[-1]) if len(eq) else 0.0,
        "multiple": mult,
        "cagr": (mult ** (1.0 / span) - 1.0) if mult > 0 else -1.0,
        "trades": len(rois),
        "liquidations": res.liquidations,
        "mdd": float(((eq - peak) / np.maximum(peak, 1e-9)).min()) if len(eq) else 0.0,
        "win_rate": float((rois > 0).mean()) if len(rois) else 0.0,
        "median_hold_days": float(np.median(days)) if len(days) else 0.0,
        # dollars per entry on wave-31's convention: roi x $100 ("한 번 들어갈 때 $100 기준")
        "avg_usd_per_trade": float((rois * CAPITAL).mean()) if len(rois) else 0.0,
        "best_usd": float(rois.max() * CAPITAL) if len(rois) else 0.0,
        "worst_usd": float(rois.min() * CAPITAL) if len(rois) else 0.0,
        "worst_roi": float(rois.min()) if len(rois) else 0.0,
        "n_ge_30pct": int(big.sum()),
        "hold_of_big": float(np.median(days[big])) if big.any() else 0.0,
        "wipe_prob": wipe,
        # honesty instrumentation: how concentrated is the headline?
        "top1_roi_share_of_sum": float(rois.max() / rois.sum()) if len(rois) and rois.sum() > 0 else float("nan"),
        "n_roi_le_m50pct": int((rois <= -0.50).sum()),
    }


def gates(m: dict, base_cagr: float) -> dict:
    r1 = bool(m["wipe_prob"] <= 0.01 and m["liquidations"] == 0)
    r2 = bool(m["worst_roi"] >= -0.50)
    r3 = bool(m["cagr"] > base_cagr)
    return {"R1": r1, "R2": r2, "R3": r3, "R1R2": r1 and r2, "all": r1 and r2 and r3}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Gate 0 (fidelity) — must PASS before any leverage number is reported ===", flush=True)
    inp = sim32.v2_inputs()
    g0 = sim32.gate0_report(inp)
    print(f"  trades ref={g0['ref_trades']} sim={g0['sim_trades']} | final |err|={g0['final_abs_err']:.3e}"
          f" | equity |err|={g0['equity_max_abs_err']:.3e}", flush=True)
    print(f"  GATE 0: {'PASS' if g0['pass'] else 'FAIL'}", flush=True)
    (OUT / "gate0.json").write_text(json.dumps(g0, indent=2), encoding="utf-8")
    if not g0["pass"]:
        print("  Gate 0 FAILED — stopping. No leverage numbers reported.", flush=True)
        return 1
    print(f"  funding-convention gap at L=1 (fixed vs rebalanced): "
          f"{g0['fixed_vs_rebalanced_L1_rel_gap']*100:+.4f}%  "
          f"(${g0['fixed_L1_final_per_100']:.2f} vs ${g0['rebalanced_L1_final_per_100']:.2f})", flush=True)

    rows: list[dict] = []
    for lev in LEVERAGES:
        res = sim32.simulate(inp, leverage=lev, model="fixed", starting_equity=CAPITAL)
        m = score(res)
        rows.append(m)
        tag = f"L{lev:g}".replace(".", "p")
        (OUT / f"v2_{tag}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"  L={lev:<4g} final=${m['final_usd']:>10.2f}  trades={m['trades']:>3d} "
              f"liq={m['liquidations']:>2d}  MDD={m['mdd']*100:>7.1f}%  "
              f"worst={m['worst_usd']:>8.2f}$  wipe={m['wipe_prob']*100:>6.2f}%", flush=True)

    base = next(r for r in rows if r["leverage"] == 1.0)
    base_cagr = base["cagr"]
    for m in rows:
        m["gates"] = gates(m, base_cagr)
        tag = f"L{m['leverage']:g}".replace(".", "p")
        (OUT / f"v2_{tag}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    print("\n=== sweep table (자본 $100 전액, fixed-notional, 장중 청산 검사) ===", flush=True)
    hdr = (f"{'L':>4} {'최종$':>10} {'배수':>7} {'CAGR':>8} {'거래':>5} {'청산':>4} {'MDD':>8} "
           f"{'승률':>6} {'중위일':>6} {'평균$':>8} {'최고$':>9} {'최악$':>9} {'≥30%':>5} {'전멸%':>7} "
           f"{'R1':>3} {'R2':>3} {'R3':>3}")
    print(hdr, flush=True)
    for m in rows:
        g = m["gates"]
        print(f"{m['leverage']:>4g} {m['final_usd']:>10.2f} {m['multiple']:>6.2f}x "
              f"{m['cagr']*100:>7.1f}% {m['trades']:>5d} {m['liquidations']:>4d} "
              f"{m['mdd']*100:>7.1f}% {m['win_rate']*100:>5.1f}% {m['median_hold_days']:>6.2f} "
              f"{m['avg_usd_per_trade']:>8.2f} {m['best_usd']:>9.2f} {m['worst_usd']:>9.2f} "
              f"{m['n_ge_30pct']:>5d} {m['wipe_prob']*100:>6.2f}% "
              f"{'O' if g['R1'] else 'X':>3} {'O' if g['R2'] else 'X':>3} {'O' if g['R3'] else 'X':>3}",
              flush=True)

    passing = [m for m in rows if m["gates"]["R1R2"]]
    best = max(passing, key=lambda m: m["leverage"]) if passing else None
    print("\n=== 판정 ===", flush=True)
    if best is None:
        print("  R1+R2를 통과하는 레버리지가 없다 (1x 포함).", flush=True)
    else:
        print(f"  R1+R2 통과 최고 레버리지 = {best['leverage']:g}x "
              f"(CAGR {best['cagr']*100:.1f}%, 청산 {best['liquidations']}회, "
              f"전멸확률 {best['wipe_prob']*100:.2f}%, 최악 {best['worst_usd']:.2f}$)", flush=True)
        print(f"  R3(1x CAGR {base_cagr*100:.1f}% 초과) 동시 통과: "
              f"{'예' if best['gates']['R3'] else '아니오 — 즉 레버리지를 올려서 얻은 게 없다'}", flush=True)
    print(f"\n  집중도 점검: L=1 최고 1회가 전체 ROI 합의 {base['top1_roi_share_of_sum']*100:.1f}%, "
          f"+30% 이상 진입은 {base['n_ge_30pct']}회 / {base['trades']}회", flush=True)

    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {OUT / 'sweep.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
