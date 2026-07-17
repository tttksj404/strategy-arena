"""Regime-switched cross-sectional momentum — alt-season auto-firing booster.

Detects alt-season from price alone (equal-weight index 20d trend > 0 AND breadth
>= 0.6), and only then runs a delta-neutral cross-sectional momentum book
(lb40 rank, top5 long / bottom5 short) sized to a target daily vol. Off-regime =
cash. Verdict: test daily +0.64% (tv0.08) to +0.90% (tv0.12), MDD 19~27%, but
train daily is thin (+0.10%) with train MDD ~41% — a conditional booster to pair
with G053, not a standalone. Circuit breakers worsen it (daily-xsec drawdowns are
too abrupt), so risk control is the regime gate + vol target only.

Run: python3 -m src.regime_xsec  → prints current regime state + target leverage.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
COST = 0.0006 + 0.0002
LOOKBACK, TOPK = 40, 5
TREND_D, BREADTH_TH = 20, 0.6


def daily_panel(min_vol: float = 10_000_000) -> pd.DataFrame:
    mani = {m["symbol"]: m for m in json.loads((ROOT / "out/data_manifest_crypto.json").read_text())}
    uni = [s for s in mani if mani[s].get("tier") == "A" and mani[s].get("usdtVolume", 0) >= min_vol]
    closes = {}
    for s in uni:
        c = pd.read_parquet(ROOT / "data/candles_1h" / f"{s}.parquet").set_index("ts")["close"].resample("1D").last()
        if len(c) >= 120:
            closes[s] = c
    return pd.DataFrame(closes).sort_index()


def regime_series(panel: pd.DataFrame) -> pd.Series:
    rets = panel.pct_change()
    idx = (1 + rets.mean(axis=1)).cumprod()
    trend = idx / idx.shift(TREND_D) - 1
    breadth = (rets.rolling(20).mean() > 0).mean(axis=1)
    return ((trend > 0) & (breadth >= BREADTH_TH)).shift(1).fillna(False).astype(float)


def portfolio(panel: pd.DataFrame, target_vol: float = 0.08, max_lev: int = 5) -> tuple[pd.Series, pd.Series, pd.Series]:
    rets = panel.pct_change()
    reg = regime_series(panel)
    sig = panel.pct_change(LOOKBACK)
    pos = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for d in panel.index:
        row = sig.loc[d].dropna()
        if len(row) < 2 * TOPK:
            continue
        w = pd.Series(0.0, index=panel.columns)
        w[row.nlargest(TOPK).index] = 1.0 / TOPK
        w[row.nsmallest(TOPK).index] = -1.0 / TOPK
        pos.loc[d] = w.values
    pos = pos.shift(1).fillna(0)
    turn = pos.diff().abs().sum(axis=1)
    raw = (pos * rets).sum(axis=1)
    realized = raw.rolling(20).std().shift(1)
    lev = (target_vol / realized).clip(0.3, max_lev).fillna(1.0) * reg
    port = raw * lev - turn * COST * lev
    return port, reg, lev


def main() -> int:
    panel = daily_panel()
    for tv, ml in ((0.08, 5), (0.12, 5)):
        port, reg, lev = portfolio(panel, tv, ml)
        cut = int(len(port) * 0.6)
        def stat(p):
            eq = (1 + p).cumprod(); net = eq.iloc[-1] - 1
            return net, float((1 - eq / eq.cummax()).max()), (1 + net) ** (1 / len(p)) - 1
        trn, trm, trd = stat(port.iloc[:cut]); ten, tem, ted = stat(port.iloc[cut:])
        print(f"tv{tv} L{ml}: train 일{trd:+.3%}/MDD{trm:.0%} | test 일{ted:+.3%}/MDD{tem:.0%}")
    port, reg, lev = portfolio(panel)
    state = "ON (알트시즌 — 발동)" if reg.iloc[-1] else "OFF (현금)"
    print(f"\n현재 국면: {state} | 목표레버: {lev.iloc[-1]:.2f} | 패널 {panel.shape[1]}심볼")
    print(f"최근 10일 국면: {reg.iloc[-10:].astype(int).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
