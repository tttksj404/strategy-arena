"""Funding-family sweep: extreme-fade and flip strategies over all collected symbols.

F_fade: when the funding rate is at a trailing extreme, take the receiving side
until it normalizes. F_flip: after K same-sign settlements the sign flips —
follow the new sign for a fixed horizon. Signals are event-driven, forward-filled
onto 1H bars, and run through the same paper-fidelity engine and gates.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import Costs, run_backtest
from .gates import gate_result

ROOT = Path(__file__).resolve().parents[1]
LEVERS = (2, 3, 5)


def fade_signal(candles: pd.DataFrame, fund: pd.DataFrame) -> pd.Series:
    rates = fund.set_index("ts")["rate"]
    hi = rates.rolling(90, min_periods=30).quantile(0.95)
    lo = rates.rolling(90, min_periods=30).quantile(0.05)
    mid_hi = rates.rolling(90, min_periods=30).quantile(0.65)
    mid_lo = rates.rolling(90, min_periods=30).quantile(0.35)
    state = 0.0
    out = []
    for ts, rate in rates.items():
        upper, lower, mh, ml = hi.get(ts, np.nan), lo.get(ts, np.nan), mid_hi.get(ts, np.nan), mid_lo.get(ts, np.nan)
        if np.isnan(upper) or np.isnan(lower):
            out.append(0.0)
            continue
        if state == 0:
            if rate >= upper and rate > 0:
                state = -1.0
            elif rate <= lower and rate < 0:
                state = 1.0
        elif state == -1 and rate <= mh:
            state = 0.0
        elif state == 1 and rate >= ml:
            state = 0.0
        out.append(state)
    events = pd.Series(out, index=rates.index)
    return events.reindex(candles["ts"], method="ffill").fillna(0.0).reset_index(drop=True)


def flip_signal(candles: pd.DataFrame, fund: pd.DataFrame, streak: int = 9, horizon_events: int = 6) -> pd.Series:
    rates = fund.set_index("ts")["rate"]
    signs = np.sign(rates.to_numpy())
    out = np.zeros(len(signs))
    run = 0
    prev = 0.0
    hold = 0
    hold_dir = 0.0
    for i, sign in enumerate(signs):
        if hold > 0:
            out[i] = hold_dir
            hold -= 1
        if sign != 0 and sign == prev:
            run += 1
        else:
            if prev != 0 and run >= streak and sign != 0:
                hold, hold_dir = horizon_events, sign
                out[i] = sign
            run = 1
        prev = sign if sign != 0 else prev
    events = pd.Series(out, index=rates.index)
    return events.reindex(candles["ts"], method="ffill").fillna(0.0).reset_index(drop=True)


def main() -> int:
    manifests = {}
    for name in ("out/data_manifest.json", "out/data_manifest_crypto.json"):
        for row in json.loads((ROOT / name).read_text(encoding="utf-8")):
            manifests[row["symbol"]] = row
    rows = []
    done = 0
    for path in sorted(glob.glob(str(ROOT / "data/funding/*.parquet"))):
        symbol = os.path.basename(path)[:-8]
        meta = manifests.get(symbol)
        if not meta or meta.get("usdtVolume", 0) < 1_000_000 or meta.get("tier") == "C":
            continue
        fund = pd.read_parquet(path)
        if len(fund) < 120:
            continue
        candles = pd.read_parquet(ROOT / "data/candles_1h" / f"{symbol}.parquet")
        cut = max(1, int(len(candles) * 0.6))
        train_c, test_c = candles.iloc[:cut].reset_index(drop=True), candles.iloc[cut:].reset_index(drop=True)
        costs = Costs(float(meta.get("half_spread_bp", 1.0)))
        for strat, builder in (("F_fade", fade_signal), ("F_flip", flip_signal)):
            signal = builder(candles, fund)
            train_s, test_s = signal.iloc[:cut].reset_index(drop=True), signal.iloc[cut:].reset_index(drop=True)
            for lever in LEVERS:
                train_fund = fund[(fund.ts >= train_c.ts.iloc[0]) & (fund.ts <= train_c.ts.iloc[-1])]
                test_fund = fund[(fund.ts >= test_c.ts.iloc[0]) & (fund.ts <= test_c.ts.iloc[-1])]
                tr = run_backtest(train_c, train_s, train_fund, lever, costs)
                te = run_backtest(test_c, test_s, test_fund, lever, costs)
                gates = gate_result(te, tr)
                rows.append({"symbol": symbol, "strategy": strat, "L": lever,
                             "train_net": tr.net_return, "test_net": te.net_return,
                             "train_trades": tr.trades, "test_trades": te.trades,
                             "test_MDD": te.mdd, "liquidated": te.liquidated,
                             "funding_paid": te.funding_paid, "all_pass": bool(gates["all_pass"]),
                             "vol": meta.get("usdtVolume", 0)})
        done += 1
        if done % 50 == 0:
            print(f"progress {done}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "out/leaderboard_funding.csv", index=False)
    print(f"rows={len(frame)} symbols={done}")
    print("all_pass:", int(frame.all_pass.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
