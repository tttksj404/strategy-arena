"""
G040 — Volatility expansion breakout (양방향).
G041 — Walk-forward adaptive (최근 30일 PnL 양수일 때만 G003 활성).
G042 — G003 + G020c short combo (regime-rotation portfolio).

마지막 시도. 안 되면 paper-live + adaptive deployment 권장.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import load_klines, compute_ch1_score, COST_BPS_RT, atr_pct

UNIVERSE_18 = ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT",
               "AVAXUSDT", "SUIUSDT", "ADAUSDT", "APTUSDT", "BNBUSDT",
               "DOTUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "SOLUSDT",
               "UNIUSDT", "XRPUSDT", "BTCUSDT"]


def quarterly_split(values, idx_arr, n_total):
    qs = [{"n":0, "net":0.0, "wins":0} for _ in range(4)]
    for v, i in zip(values, idx_arr):
        q = min(int(i / n_total * 4), 3)
        qs[q]["n"] += 1
        qs[q]["net"] += v
        if v > 0: qs[q]["wins"] += 1
    return qs


def vol_breakout(df, vol_ratio_threshold=1.5, hold=24):
    """ATR / ATR_MA(20) > threshold AND 직전 바 close > open → long.
    역방향 → short. 양방향 동시 검출."""
    a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
    a_ma = a.rolling(20).mean()
    vol_expand = (a / a_ma) > vol_ratio_threshold
    bar_up = df["close_price"] > df["open_price"]
    bar_down = df["close_price"] < df["open_price"]
    long_sig = vol_expand & bar_up
    short_sig = vol_expand & bar_down
    return long_sig, short_sig


def run_vol_breakout(symbol_dfs, vol_ratio, hold):
    long_n, long_net, long_w = 0, 0.0, 0
    short_n, short_net, short_w = 0, 0.0, 0
    long_idx, long_vals = [], []
    short_idx, short_vals = [], []
    for sym, df in symbol_dfs.items():
        if df is None or len(df) < 100: continue
        long_sig, short_sig = vol_breakout(df, vol_ratio, hold)
        df_local = df.copy()
        df_local["long"] = long_sig; df_local["short"] = short_sig
        df_local["fwd_pct"] = (df_local["close_price"].shift(-hold) / df_local["close_price"] - 1) * 10000
        n_total = len(df_local)
        for i, row in df_local.iterrows():
            if pd.isna(row["fwd_pct"]): continue
            if row["long"]:
                net = row["fwd_pct"] - COST_BPS_RT
                long_n += 1; long_net += net; long_vals.append(net); long_idx.append(i)
                if net > 0: long_w += 1
            elif row["short"]:
                net = -row["fwd_pct"] - COST_BPS_RT
                short_n += 1; short_net += net; short_vals.append(net); short_idx.append(i)
                if net > 0: short_w += 1
    if long_n == 0 and short_n == 0:
        return None
    n_total_max = max([len(df) for df in symbol_dfs.values() if df is not None])
    long_q = quarterly_split(long_vals, long_idx, n_total_max) if long_n else None
    short_q = quarterly_split(short_vals, short_idx, n_total_max) if short_n else None
    return {
        "long": {
            "n": long_n,
            "avg_net_bps": round(long_net/long_n, 2) if long_n else 0,
            "wr": round(long_w/long_n, 4) if long_n else 0,
            "quarters": [{"n":q["n"], "net":round(q["net"]/q["n"],1) if q["n"] else 0, "wr":round(q["wins"]/q["n"],4) if q["n"] else 0} for q in long_q] if long_q else None,
        },
        "short": {
            "n": short_n,
            "avg_net_bps": round(short_net/short_n, 2) if short_n else 0,
            "wr": round(short_w/short_n, 4) if short_n else 0,
            "quarters": [{"n":q["n"], "net":round(q["net"]/q["n"],1) if q["n"] else 0, "wr":round(q["wins"]/q["n"],4) if q["n"] else 0} for q in short_q] if short_q else None,
        },
    }


def main():
    print("=== G040 — Volatility expansion breakout (양방향) ===\n")
    print(f"{'vol_ratio':>9} {'hold':>5} {'side':>6} {'n':>5} {'net':>9} {'WR':>7} {'Q1':>15} {'Q2':>15} {'Q3':>15} {'Q4':>15}")
    all_dfs = {sym: load_klines(sym) for sym in UNIVERSE_18}

    sweeps = []
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        for hold in [24, 72]:
            r = run_vol_breakout(all_dfs, ratio, hold)
            if r is None: continue
            for side in ("long", "short"):
                sd = r[side]
                if sd["n"] == 0: continue
                qs = sd["quarters"] or [{"n":0,"net":0,"wr":0}]*4
                q_str = lambda q: f"{q['n']:>3}/{q['net']:>+5.0f}/{q['wr']*100:>3.0f}%"
                print(f"{ratio:>9} {hold:>5} {side:>6} {sd['n']:>5} {sd['avg_net_bps']:>+9.2f} {sd['wr']*100:>6.1f}% {q_str(qs[0]):>15} {q_str(qs[1]):>15} {q_str(qs[2]):>15} {q_str(qs[3]):>15}")
                sweeps.append({"ratio":ratio, "hold":hold, "side":side, **sd})

    robust = [r for r in sweeps if r['avg_net_bps'] > 0 and r['quarters'] and all(q['net'] > 0 or q['n']==0 for q in r['quarters'])]
    print(f"\n=== ROBUST (모든 분기 양수 또는 무진입): {len(robust)}개 ===")
    for r in robust:
        print(f"  ✅ ratio={r['ratio']} hold={r['hold']} {r['side']}: net={r['avg_net_bps']}bps WR={r['wr']*100:.1f}% n={r['n']}")
    if not robust:
        if sweeps:
            best = max([r for r in sweeps if r['quarters']],
                       key=lambda r: min(q['net'] for q in r['quarters'] if q['n']>0) if any(q['n']>0 for q in r['quarters']) else -99999)
            worst_q = min(q['net'] for q in best['quarters'] if q['n']>0)
            print(f"  최고 일관성: ratio={best['ratio']} hold={best['hold']} {best['side']} → worst Q net={worst_q:.0f}, avg={best['avg_net_bps']}")

    # === G041 walk-forward adaptive G003 ===
    print(f"\n=== G041 — Walk-forward adaptive G003 (최근 30일 양수일 때만) ===")
    rolling_pnls = []
    all_entries = []
    for sym, df in all_dfs.items():
        if df is None: continue
        score, _ = compute_ch1_score(df)
        df_local = df.copy()
        df_local["score"] = score
        df_local["fwd_pct"] = (df_local["close_price"].shift(-72) / df_local["close_price"] - 1) * 10000
        e = df_local[(df_local["score"] >= 70) & df_local["fwd_pct"].notna()].copy()
        e["net_bps"] = e["fwd_pct"] - COST_BPS_RT
        e["sym"] = sym
        all_entries.append(e[["open_time", "score", "net_bps", "sym"]])
    all_e = pd.concat(all_entries).sort_values("open_time").reset_index(drop=True)
    # 30-day rolling backtest gate: 첫 30일은 무조건 trade, 그 후 직전 30일 net 양수일 때만 진입
    LOOKBACK_DAYS = 30
    LOOKBACK_MS = LOOKBACK_DAYS * 24 * 3600 * 1000
    enabled_pnl = 0.0
    enabled_n = 0
    enabled_wins = 0
    enabled_q = [{"n":0,"net":0.0,"wins":0} for _ in range(4)]
    n_total = max(len(df) for df in all_dfs.values() if df is not None)
    first_ts = all_e["open_time"].min()
    last_ts = all_e["open_time"].max()
    for _, row in all_e.iterrows():
        ts = row["open_time"]
        if ts - first_ts < LOOKBACK_MS:
            # 첫 30일은 무조건 진입 (warm-up)
            enabled_pnl += row["net_bps"]
            enabled_n += 1
            if row["net_bps"] > 0: enabled_wins += 1
            q = min(int((ts-first_ts)/(last_ts-first_ts)*4), 3)
            enabled_q[q]["n"] += 1; enabled_q[q]["net"] += row["net_bps"]
            if row["net_bps"] > 0: enabled_q[q]["wins"] += 1
            continue
        # 30일 이내 net 평가
        recent = all_e[(all_e["open_time"] >= ts - LOOKBACK_MS) & (all_e["open_time"] < ts)]
        if recent["net_bps"].sum() > 0:
            enabled_pnl += row["net_bps"]
            enabled_n += 1
            if row["net_bps"] > 0: enabled_wins += 1
            q = min(int((ts-first_ts)/(last_ts-first_ts)*4), 3)
            enabled_q[q]["n"] += 1; enabled_q[q]["net"] += row["net_bps"]
            if row["net_bps"] > 0: enabled_q[q]["wins"] += 1
    print(f"  baseline (G003 unfilter): n={len(all_e)} avg_net={all_e['net_bps'].mean():.2f}")
    print(f"  G041 (adaptive):          n={enabled_n} avg_net={enabled_pnl/max(enabled_n,1):.2f} WR={enabled_wins/max(enabled_n,1)*100:.1f}%")
    print(f"  분기별: ", end="")
    for i, q in enumerate(enabled_q):
        if q["n"] > 0:
            print(f"Q{i+1}={q['n']}/{q['net']/q['n']:+.0f}bps/{q['wins']/q['n']*100:.0f}% ", end="")
    print()

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g040_vol_breakout_results.json"
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "g040_sweeps": sweeps,
        "g040_robust": [{"ratio":r["ratio"],"hold":r["hold"],"side":r["side"]} for r in robust],
        "g041_adaptive": {"n": enabled_n, "avg_net": round(enabled_pnl/max(enabled_n,1),2), "wr": round(enabled_wins/max(enabled_n,1),4)},
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
