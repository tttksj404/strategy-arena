# Wave-31: shortest-horizon maximum profit, on the FULL $100 (no sleeve split).
#
# Two user principles drive this wave:
#   1. "최단기간 최대수익이 원칙"  -> objective is CAGR, not total multiple, so earning the same
#      money faster scores higher automatically.
#   2. "ROI 10%면 $100일 때 $10을 벌어야지" -> the whole $100 goes into the position, so a trade's
#      ROI maps one-to-one onto dollars. The cost of that is real and stated: there is no stable
#      sleeve left to absorb a loss, so "wipe" now means the account itself.
#
# wave-30 measured that V1's profit lives in its LONG holds (21-60d: +16.81% avg, 88.9% win;
# 2-7d: -4.40% avg, 13.3% win). So capping holding time alone just amputates the winners. The
# speed knob that can actually work is the ATR multiplier: a tighter band triggers sooner and
# reverses sooner, which is what genuinely shortens the cycle.

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
CAPITAL = 100.0                       # full deployment, per the user's principle

ATR_MULTS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
HOLD_DAYS: list[int | None] = [3, 5, 7, 10, 14, 21, None]
LEVERAGES = [1.0, 2.0, 3.0]

WIPE_CAP, WORST_TRADE_CAP = 0.01, -0.50


def monte_carlo(rois: np.ndarray, paths: int = 10000, seed: int = 20260731) -> dict:
    if len(rois) < 5:
        return {"wipe_prob": float("nan"), "p05": float("nan"), "half_prob": float("nan")}
    rng = np.random.default_rng(seed)
    draw = rng.choice(rois, size=(paths, len(rois)), replace=True)
    curve = CAPITAL * np.cumprod(np.maximum(0.0, 1.0 + draw), axis=1)
    return {
        "wipe_prob": float((curve.min(axis=1) <= 0.005).mean()),
        "p05": float(np.percentile(curve[:, -1], 5)),
        "half_prob": float((curve[:, -1] < CAPITAL / 2).mean()),
    }


def metrics(res: R30, atr_mult: float, hold: int | None) -> dict:
    rois = np.asarray([t.roi for t in res.trades], dtype=float)
    days = np.asarray([(t.exit_time - t.entry_time).total_seconds() / 86400.0 for t in res.trades])
    eq = res.equity.to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    span_years = max((res.equity.index[-1] - res.equity.index[0]).days / 365.25, 1e-9)
    mult = res.final / CAPITAL
    cagr = mult ** (1.0 / span_years) - 1.0 if mult > 0 else -1.0
    exposure = float(days.sum()) if len(days) else 0.0
    big = rois >= 0.30
    return {
        "atr_mult": atr_mult, "hold_days": hold, "leverage": res.leverage,
        "final": res.final, "multiple": mult, "cagr": cagr,
        "trades": len(rois), "liquidations": sum(1 for t in res.trades if t.reason == "liquidated"),
        "mdd": float(((eq - peak) / np.maximum(peak, 1e-9)).min()),
        "win_rate": float((rois > 0).mean()) if len(rois) else 0.0,
        # The user's headline metric: dollars per entry, on the full $100.
        "avg_usd_per_trade": float((rois * CAPITAL).mean()) if len(rois) else 0.0,
        "median_hold_days": float(np.median(days)) if len(days) else 0.0,
        "usd_per_exposure_day": float((res.final - CAPITAL) / exposure) if exposure > 0 else 0.0,
        "roi_med": float(np.median(rois)) if len(rois) else 0.0,
        "roi_max": float(rois.max()) if len(rois) else 0.0,
        "roi_worst": float(rois.min()) if len(rois) else 0.0,
        "n_ge_30pct": int(big.sum()),
        "hold_of_big": float(np.median(days[big])) if big.any() else 0.0,
        **{f"mc_{k}": v for k, v in monte_carlo(rois).items()},
    }


def passes(m: dict) -> bool:
    return (m["trades"] >= 20 and m["liquidations"] == 0
            and m["mc_wipe_prob"] <= WIPE_CAP and m["roi_worst"] >= WORST_TRADE_CAP)


def run(inp: dict, atr_mult: float, hold: int | None, lev: float, **kw) -> dict:
    res = simulate(inp, leverage=lev, stop=None, model="fixed", starting_equity=CAPITAL,
                   atr_multiplier=atr_mult, max_hold_bars=None if hold is None else hold * 24, **kw)
    return metrics(res, atr_mult, hold)


def main() -> int:
    ok, msg = fidelity_check()
    print(f"[Gate 0] {'PASS' if ok else 'FAIL'} - {msg}", flush=True)
    if not ok:
        return 1
    inp = v1_inputs()

    rows = []
    print(f"\n{'ATR':>5} {'상한':>5} {'lev':>4} {'CAGR':>8} {'배수':>7} {'거래':>5} {'중앙보유':>8} "
          f"{'평균$/거래':>10} {'$/노출일':>9} {'최악1회':>8} {'전멸%':>7} {'+30%':>5}", flush=True)
    for am in ATR_MULTS:
        for hd in HOLD_DAYS:
            for lev in LEVERAGES:
                m = run(inp, am, hd, lev, end_at=IS_END)
                m["gate_pass"] = passes(m)
                rows.append(m)
                if m["trades"] >= 20:
                    print(f"{am:5.2f} {str(hd) if hd else '없음':>5} {lev:3.0f}x {m['cagr']*100:7.1f}% "
                          f"{m['multiple']:6.2f}x {m['trades']:5d} {m['median_hold_days']:7.1f}일 "
                          f"{m['avg_usd_per_trade']:9.2f}$ {m['usd_per_exposure_day']:8.3f}$ "
                          f"{m['roi_worst']*100:7.1f}% {m['mc_wipe_prob']*100:6.2f}% {m['n_ge_30pct']:5d}"
                          + ("  OK" if m["gate_pass"] else ""), flush=True)

    base = next(r for r in rows if r["atr_mult"] == 2.0 and r["hold_days"] is None and r["leverage"] == 1.0)
    survivors = [r for r in rows if r["gate_pass"]]
    print(f"\nR1+R2 통과 {len(survivors)}/{len(rows)}셀 | IS 기준선(V1 1x 전액) CAGR {base['cagr']*100:.1f}%", flush=True)
    if not survivors:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "search.json").write_text(json.dumps({"rows": rows, "winner": None}, indent=2, default=str), encoding="utf-8")
        print("승격 없음.")
        return 0

    winner = max(survivors, key=lambda r: (r["cagr"], -r["median_hold_days"]))
    fast = [r for r in survivors if r["median_hold_days"] <= 7.0]
    fastest = max(fast, key=lambda r: r["cagr"]) if fast else None

    print(f"CAGR 최대: ATR {winner['atr_mult']} / 상한 {winner['hold_days']} / {winner['leverage']:.0f}x "
          f"-> IS CAGR {winner['cagr']*100:.1f}%, 중앙보유 {winner['median_hold_days']:.1f}일", flush=True)
    if fastest:
        print(f"단기제약(중앙 ≤7일) 최고: ATR {fastest['atr_mult']} / 상한 {fastest['hold_days']} / "
              f"{fastest['leverage']:.0f}x -> IS CAGR {fastest['cagr']*100:.1f}%, "
              f"중앙보유 {fastest['median_hold_days']:.1f}일, 평균 {fastest['avg_usd_per_trade']:.2f}$/거래", flush=True)
    else:
        print("단기제약(중앙 ≤7일)을 만족하면서 게이트를 통과하는 셀이 없다.", flush=True)

    # --- OOS opened exactly once, on the selected cell(s) only. ---
    out = {"rows": rows, "baseline_is": base, "winner_is": winner,
           "winner_oos": run(inp, winner["atr_mult"], winner["hold_days"], winner["leverage"], start_at=IS_END),
           "baseline_oos": run(inp, 2.0, None, 1.0, start_at=IS_END)}
    if fastest:
        out["fastest_is"] = fastest
        out["fastest_oos"] = run(inp, fastest["atr_mult"], fastest["hold_days"], fastest["leverage"], start_at=IS_END)
    print(f"\nOOS 승자 {out['winner_oos']['multiple']:.3f}x | 기준선 {out['baseline_oos']['multiple']:.3f}x"
          + (f" | 단기셀 {out['fastest_oos']['multiple']:.3f}x" if fastest else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "search.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
