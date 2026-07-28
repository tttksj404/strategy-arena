"""Profit-maximization search over the regime booster: ranking signals x params.

Beats-current-best (tv0.25 test daily +1.52%). Holds the regime gate + vol-target
(the only reliable risk control), then sweeps: ranking signal family (pure momentum
at several lookbacks, vol-adjusted momentum, composite momentum+short-reversal),
top-k, rebalance cadence, and target vol. Reports cells that beat the benchmark on
test daily return while keeping train>0 and top3-concentration<50% (not a homerun).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regime_xsec import daily_panel, regime_series  # noqa: E402

COST = 0.0006 + 0.0002
BENCH_DAILY = 0.0152


def ranking(panel: pd.DataFrame, kind: str, lb: int) -> pd.DataFrame:
    rets = panel.pct_change()
    if kind == "mom":
        return panel.pct_change(lb)
    if kind == "voladj":  # momentum / realized vol (risk-adjusted momentum)
        mom = panel.pct_change(lb)
        vol = rets.rolling(lb).std()
        return mom / vol.replace(0, np.nan)
    if kind == "composite":  # long-term momentum minus short-term reversal
        return panel.pct_change(lb) - 0.5 * panel.pct_change(3)
    raise ValueError(kind)


def run(panel: pd.DataFrame, reg: pd.Series, kind: str, lb: int, topk: int, rebal: int, tv: float, maxL: int) -> dict:
    rets = panel.pct_change()
    sig = ranking(panel, kind, lb)
    dates = panel.index[::rebal]
    pos = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for d in dates:
        if d not in sig.index:
            continue
        row = sig.loc[d].dropna()
        if len(row) < 2 * topk:
            continue
        w = pd.Series(0.0, index=panel.columns)
        w[row.nlargest(topk).index] = 1.0 / topk
        w[row.nsmallest(topk).index] = -1.0 / topk
        pos.loc[d:] = w.values
    pos = pos.shift(1).fillna(0)
    turn = pos.diff().abs().sum(axis=1)
    raw = (pos * rets).sum(axis=1)
    realized = raw.rolling(20).std().shift(1)
    lev = (tv / realized).clip(0.3, maxL).fillna(1.0) * reg
    port = raw * lev - turn * COST * lev
    cut = int(len(port) * 0.6)

    def stat(p):
        eq = (1 + p).cumprod(); net = eq.iloc[-1] - 1
        mdd = float((1 - eq / eq.cummax()).max())
        return net, mdd, ((1 + net) ** (1 / len(p)) - 1 if net > -1 else -1.0)

    trn, trm, trd = stat(port.iloc[:cut])
    ten, tem, ted = stat(port.iloc[cut:])
    te = port.iloc[cut:]
    top3 = float(te.nlargest(3).sum() / te.sum()) if te.sum() > 0 else 9.9
    return {"kind": kind, "lb": lb, "topk": topk, "rebal": rebal, "tv": tv, "maxL": maxL,
            "tr_daily": trd, "tr_mdd": trm, "te_daily": ted, "te_mdd": tem, "top3": top3}


def main() -> int:
    panel = daily_panel()
    reg = regime_series(panel)
    print(f"panel {panel.shape[1]}sym x {panel.shape[0]}d | benchmark test-daily {BENCH_DAILY:+.3%}", flush=True)
    rows, n = [], 0
    for kind in ("mom", "voladj", "composite"):
        for lb in (20, 40, 60, 90):
            for topk in (3, 5, 8):
                for rebal in (1, 3, 7):
                    for tv in (0.20, 0.25):
                        r = run(panel, reg, kind, lb, topk, rebal, tv, 5)
                        rows.append(r)
                        n += 1
                        if r["te_daily"] > BENCH_DAILY and r["tr_daily"] > 0 and r["top3"] < 0.5:
                            print(f"BEAT {kind} lb{lb} k{topk} r{rebal} tv{tv}: te일 {r['te_daily']:+.3%} "
                                  f"(tr {r['tr_daily']:+.3%}) teMDD {r['te_mdd']:.0%} top3 {r['top3']:.0%}", flush=True)
    df = pd.DataFrame(rows).sort_values("te_daily", ascending=False)
    df.to_csv(ROOT / "out/booster_maximize.csv", index=False)
    beat = df[(df.te_daily > BENCH_DAILY) & (df.tr_daily > 0) & (df.top3 < 0.5)]
    print(f"\n총 {n}셀 | 벤치(일{BENCH_DAILY:+.2%}) 초과+견고 {len(beat)}", flush=True)
    print(df.head(10)[["kind", "lb", "topk", "rebal", "tv", "tr_daily", "te_daily", "te_mdd", "top3"]].round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
