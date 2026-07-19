"""Cross-exchange delta-neutral funding arbitrage: Bitget vs Binance.

When the 8h funding spread |Binance - Bitget| exceeds an entry threshold, go short
the higher-funding leg and long the lower-funding leg (delta-neutral, no price
risk). Collect the net funding spread each settlement; exit when the spread
compresses below an exit threshold, reverses sign, or max-hold is hit. Charge
round-trip taker on both legs. Train60/test40 per symbol; report which entry
threshold clears the fee hurdle.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BN_FEE = 0.0005   # Binance USDT-M taker
BG_FEE = 0.0006   # Bitget taker
ROUND_TRIP = 2 * (BN_FEE + BG_FEE)  # enter+exit on both legs ~0.0022


def bn_funding(symbol: str) -> pd.DataFrame | None:
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1000"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=15))
    except Exception:
        return None
    if not data:
        return None
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df.fundingTime, unit="ms", utc=True)
    df["bn"] = df.fundingRate.astype(float)
    return df[["ts", "bn"]].sort_values("ts")


def aligned_spread(symbol: str) -> pd.DataFrame | None:
    bnf = bn_funding(symbol)
    if bnf is None:
        return None
    bg_path = ROOT / "data/funding" / f"{symbol}.parquet"
    if not bg_path.exists():
        return None
    bgf = pd.read_parquet(bg_path).rename(columns={"rate": "bg"}).sort_values("ts")
    m = pd.merge_asof(bnf, bgf, on="ts", tolerance=pd.Timedelta("4h"), direction="nearest").dropna()
    if len(m) < 60:
        return None
    m["spread"] = m["bn"] - m["bg"]  # short the higher leg → receive |spread| per settlement
    return m.reset_index(drop=True)


def simulate(spread: pd.Series, entry_bps: float, exit_bps: float, max_hold: int) -> dict:
    """Delta-neutral: enter when |spread|>=entry, collect |spread| each step, exit on compress/flip/max."""
    s = spread.to_numpy()
    entry, exit_t = entry_bps / 10000, exit_bps / 10000
    pnl = 0.0
    pos = 0        # +1 = short-Binance/long-Bitget receiving when bn>bg; sign tracks receive direction
    held = 0
    trades = 0
    returns = []
    acc = 0.0
    for x in s:
        if pos == 0:
            if abs(x) >= entry:
                pos = 1 if x > 0 else -1
                acc = -ROUND_TRIP  # entry cost both legs
                held = 0
                trades += 1
        else:
            acc += pos * x  # receive net funding (pos aligned with spread sign)
            held += 1
            flipped = (pos > 0 and x < 0) or (pos < 0 and x > 0)
            if abs(x) <= exit_t or flipped or held >= max_hold:
                acc -= ROUND_TRIP  # exit cost both legs
                pnl += acc
                returns.append(acc)
                pos = 0
    r = np.array(returns)
    return {"trades": trades, "pnl_pct": pnl * 100, "win": float((r > 0).mean()) if len(r) else 0.0,
            "avg_bps": float(r.mean() * 10000) if len(r) else 0.0}


def main() -> int:
    mani = {m["symbol"]: m for m in json.loads((ROOT / "out/data_manifest_crypto.json").read_text())}
    universe = sorted([s for s in mani if mani[s].get("usdtVolume", 0) >= 10_000_000])
    frames = {}
    for symbol in universe:
        sp = aligned_spread(symbol)
        time.sleep(0.1)
        if sp is not None:
            frames[symbol] = sp
    print(f"aligned symbols: {len(frames)}", flush=True)
    rows = []
    for symbol, sp in frames.items():
        cut = int(len(sp) * 0.6)
        tr, te = sp.spread.iloc[:cut], sp.spread.iloc[cut:]
        for entry in (3, 5, 8):
            rt = simulate(tr, entry, exit_bps=1, max_hold=42)   # 42 settlements = 14 days
            rte = simulate(te, entry, exit_bps=1, max_hold=42)
            if rte["trades"] >= 3 and rt["pnl_pct"] > 0:
                rows.append({"sym": symbol, "entry": entry, "tr_pnl%": round(rt["pnl_pct"], 2),
                             "te_pnl%": round(rte["pnl_pct"], 2), "te_trades": rte["trades"],
                             "te_win": round(rte["win"], 2), "te_avg_bps": round(rte["avg_bps"], 1)})
    df = pd.DataFrame(rows).sort_values("te_pnl%", ascending=False)
    df.to_csv(ROOT / "out/funding_arb_results.csv", index=False)
    print(f"train-positive cells: {len(df)}", flush=True)
    if not df.empty:
        print(df.head(15).to_string(index=False), flush=True)
        pos = df[(df["te_pnl%"] > 0) & (df["te_win"] >= 0.6)]
        print(f"\ntest-positive & win>=60%: {len(pos)}", flush=True)
        if not pos.empty:
            print(pos.head(10).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
