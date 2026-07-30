# Wave-30 part 2: is "pick the best (leverage, stop)" a STABLE RULE or an in-sample artifact?
#
# The IS sweep's winner (2x, no stop) returned 18.97x in-sample and 0.63x out-of-sample. But the
# 1x baseline also lost out-of-sample, so a single IS/OOS split cannot tell "this cell was
# overfit" apart from "V1's whole family is in a losing regime right now".
#
# Walk-forward does separate them: every year, re-select the sizing using ONLY data available
# before that year, then trade it forward blind. If the selection rule has real content, the
# chained walk-forward equity beats fixed 1x. If it does not, 1x is the answer and the sweep's
# 18.97x was a number we fitted, not a number we can earn.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.wave30_riskcap.sim30 import simulate, v1_inputs
from research.wave30_riskcap.sweep30 import LEVERAGES, STOPS, metrics, passes

OUT = Path(__file__).resolve().parent / "results"
SLEEVE = 25.0
TRAIN_YEARS = 2


def year_bounds(inp: dict) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    idx = inp["index"]
    first, last = idx[0], idx[-1]
    years = range(int(first.year) + TRAIN_YEARS, int(last.year) + 1)
    out = []
    for y in years:
        start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), last)
        if start < last:
            out.append((start, end))
    return out


def select(inp: dict, train_start: pd.Timestamp, train_end: pd.Timestamp) -> tuple[float, float | None, float]:
    """Best (leverage, stop) on the trailing window, under the same R1/R2 gates. No lookahead."""
    best = (1.0, None, -1.0)
    for lev in LEVERAGES:
        for stop in STOPS:
            res = simulate(inp, leverage=float(lev), stop=stop, model="fixed",
                           start_at=train_start, end_at=train_end)
            if len(res.trades) < 8:
                continue
            m = metrics(res)
            # R2 only + no liquidation: the MC wipe gate needs more trades than a 2y window gives,
            # so the loss cap does the safety work here and the full gate is re-applied at the end.
            if m["liquidations"] or m["roi_worst"] < -0.50:
                continue
            if m["multiple"] > best[2]:
                best = (float(lev), stop, m["multiple"])
    return best


def main() -> int:
    inp = v1_inputs()
    windows = year_bounds(inp)
    print(f"walk-forward windows: {len(windows)} ({windows[0][0].date()} -> {windows[-1][1].date()})", flush=True)

    wf_equity, fixed_equity = SLEEVE, SLEEVE
    rows, wf_rois, fixed_rois = [], [], []
    for start, end in windows:
        train_start = start - pd.DateOffset(years=TRAIN_YEARS)
        lev, stop, train_mult = select(inp, train_start, start)
        fwd = simulate(inp, leverage=lev, stop=stop, model="fixed",
                       start_at=start, end_at=end, starting_equity=wf_equity)
        base = simulate(inp, leverage=1.0, stop=None, model="fixed",
                        start_at=start, end_at=end, starting_equity=fixed_equity)
        wf_rois += [t.roi for t in fwd.trades]
        fixed_rois += [t.roi for t in base.trades]
        step_wf = fwd.final / max(wf_equity, 1e-9)
        step_fx = base.final / max(fixed_equity, 1e-9)
        wf_equity, fixed_equity = fwd.final, base.final
        rows.append({"year": int(start.year), "picked_leverage": lev, "picked_stop": stop,
                     "train_multiple": train_mult, "fwd_multiple": step_wf, "fixed_multiple": step_fx,
                     "wf_equity": wf_equity, "fixed_equity": fixed_equity, "trades": len(fwd.trades)})
        print(f"  {start.year}: 학습최적={lev:.0f}x/{stop} (학습 {train_mult:5.2f}x) -> "
              f"실전 {step_wf:5.2f}x  누적 ${wf_equity:8.2f} | 고정1x {step_fx:5.2f}x 누적 ${fixed_equity:8.2f}", flush=True)

    wf_arr, fx_arr = np.asarray(wf_rois), np.asarray(fixed_rois)
    summary = {
        "windows": rows,
        "walkforward_final": wf_equity, "walkforward_multiple": wf_equity / SLEEVE,
        "fixed1x_final": fixed_equity, "fixed1x_multiple": fixed_equity / SLEEVE,
        "wf_trades": len(wf_rois), "fixed_trades": len(fixed_rois),
        "wf_n_ge_30pct": int((wf_arr >= 0.30).sum()) if len(wf_arr) else 0,
        "fixed_n_ge_30pct": int((fx_arr >= 0.30).sum()) if len(fx_arr) else 0,
        "wf_worst_trade": float(wf_arr.min()) if len(wf_arr) else 0.0,
        "fixed_worst_trade": float(fx_arr.min()) if len(fx_arr) else 0.0,
        "years_wf_beat_fixed": sum(1 for r in rows if r["fwd_multiple"] > r["fixed_multiple"]),
    }
    print(f"\n워크포워드 최종 ${wf_equity:.2f} ({wf_equity/SLEEVE:.2f}x) vs 고정1x ${fixed_equity:.2f} ({fixed_equity/SLEEVE:.2f}x)")
    print(f"연도별 승: 워크포워드 {summary['years_wf_beat_fixed']}/{len(rows)}")
    print(f"+30% 진입: 워크포워드 {summary['wf_n_ge_30pct']}건 / 고정1x {summary['fixed_n_ge_30pct']}건")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "walkforward.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
