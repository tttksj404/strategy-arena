# Wave-31 verification: is the winner a PLATEAU or a SPIKE?
#
# The IS search picked ATR 3.0 / 21-day cap / 2x at 49.92x. Its immediate neighbours in the same
# grid are 11.00x (14-day cap) and 4.06x (no cap) -- a 4.5x jump between adjacent cells, and the
# ATR value sits on the edge of the searched range. Both are classic overfitting tells, and this
# campaign has already produced two of them (wave-21's top_k=3, wave-28's pyramid overlay).
#
# A real effect is a plateau: neighbours should be comparable. A fitted artefact is a spike that
# collapses one grid step away. This refines the grid around the winner to tell them apart, and
# counts the OOS trades, because an OOS "pass" carried by 3 trades is noise, not evidence.

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave30_riskcap.sim30 import simulate, v1_inputs
from research.wave31_shorthorizon.search31 import CAPITAL, IS_END, metrics

OUT = Path(__file__).resolve().parent / "results"
FINE_ATR = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]
FINE_HOLD = [16, 18, 19, 20, 21, 22, 23, 25, 28]


def cell(inp: dict, am: float, hd: int | None, lev: float, **kw) -> dict:
    res = simulate(inp, leverage=lev, stop=None, model="fixed", starting_equity=CAPITAL,
                   atr_multiplier=am, max_hold_bars=None if hd is None else hd * 24, **kw)
    return metrics(res, am, hd)


def main() -> int:
    inp = v1_inputs()

    print("=== IS 배수: ATR(행) x 보유상한(열), 2x ===", flush=True)
    print(f"{'ATR':>5} " + " ".join(f"{h:>7}" for h in FINE_HOLD), flush=True)
    grid = {}
    for am in FINE_ATR:
        vals = []
        for hd in FINE_HOLD:
            m = cell(inp, am, hd, 2.0, end_at=IS_END)
            grid[(am, hd)] = m
            vals.append(m["multiple"])
        print(f"{am:5.2f} " + " ".join(f"{v:6.2f}x" for v in vals), flush=True)

    win = grid[(3.0, 21)]
    neigh = [grid[k]["multiple"] for k in grid if k != (3.0, 21)
             and abs(k[0] - 3.0) <= 0.25 and abs(k[1] - 21) <= 2]
    ratio = win["multiple"] / max(np.median(neigh), 1e-9)
    print(f"\n승자 {win['multiple']:.2f}x | 인접 8셀 중앙값 {np.median(neigh):.2f}x | 비율 {ratio:.2f}x", flush=True)
    print(f"인접셀 범위 {min(neigh):.2f}x ~ {max(neigh):.2f}x", flush=True)
    verdict = "SPIKE(과최적화)" if ratio > 2.0 else "PLATEAU(안정)"
    print(f"판정: {verdict}  (기준: 인접 중앙값의 2배 초과면 스파이크)", flush=True)

    # OOS trade counts -- an OOS pass carried by a handful of trades is not evidence.
    print("\n=== OOS 거래수 확인 ===", flush=True)
    oos = {}
    for label, (am, hd, lev) in {"승자 3.0/21/2x": (3.0, 21, 2.0),
                                 "단기셀 3.0/3/3x": (3.0, 3, 3.0),
                                 "기준선 2.0/none/1x": (2.0, None, 1.0)}.items():
        m = cell(inp, am, hd, lev, start_at=IS_END)
        oos[label] = m
        print(f"  {label:>20}: {m['multiple']:.3f}x, 거래 {m['trades']}건, "
              f"승률 {m['win_rate']*100:.0f}%, 중앙보유 {m['median_hold_days']:.1f}일, "
              f"최악 {m['roi_worst']*100:.1f}%", flush=True)

    # Deflated Sharpe on the winner, charged the full accumulated trial count.
    win_full = cell(inp, 3.0, 21, 2.0)
    rois = np.asarray([t.roi for t in simulate(
        inp, leverage=2.0, stop=None, model="fixed", starting_equity=CAPITAL,
        atr_multiplier=3.0, max_hold_bars=21 * 24).trades], dtype=float)
    n_trials = 137 + 80 + 126 + len(FINE_ATR) * len(FINE_HOLD)   # cumulative campaign trials
    sr = float(rois.mean() / rois.std(ddof=1)) if rois.std(ddof=1) > 0 else 0.0
    nd = NormalDist()   # stdlib; scipy is not installed in this environment
    e_max = (nd.inv_cdf(1 - 1 / n_trials) * (1 - np.euler_gamma)
             + nd.inv_cdf(1 - 1 / (n_trials * np.e)) * np.euler_gamma)
    n = len(rois)
    c = rois - rois.mean()
    s = rois.std(ddof=1)
    g3 = float((c ** 3).mean() / s ** 3)          # skew
    g4 = float((c ** 4).mean() / s ** 4)          # kurtosis (non-excess)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    dsr = float(nd.cdf((sr - e_max) * np.sqrt(n - 1) / denom))
    print(f"\n=== DSR (누적 시행 {n_trials}회 보정) ===", flush=True)
    print(f"  거래당 SR {sr:.4f} | 기대최대SR {e_max:.4f} | 거래 {n}건 | 왜도 {g3:.2f}", flush=True)
    print(f"  **DSR = {dsr:.4f}**  ({'PASS' if dsr > 0.95 else 'FAIL'}, 기준 >0.95)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stability.json").write_text(json.dumps(
        {"grid": {f"{k[0]}_{k[1]}": v for k, v in grid.items()}, "winner_full": win_full,
         "neighbour_median": float(np.median(neigh)), "spike_ratio": float(ratio), "verdict": verdict,
         "oos": oos, "dsr": dsr, "sr_per_trade": sr, "n_trials": n_trials},
        indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
