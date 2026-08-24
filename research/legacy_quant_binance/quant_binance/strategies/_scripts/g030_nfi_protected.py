"""
G030/G031/G032 — NFI #1 (Local dip + oversold) 단순 포팅 + protection layer.

가설: G003 의 regime 의존성은 protection layer 부재 때문. NFI 는 5+년 라이브에서
      ema_fast>ema_200, sma_200 rising, BTC not downtrend 등 trend filter 를 항상 적용.

G030 — NFI #1 단순 포팅 (oversold + protection)
G031 — PB001 CH1 + NFI protection layer (CH1 신호 + trend 필터)
G032 — G031 + lottery 변형 (CH1 score 80)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import (
    load_klines, compute_ch1_score, COST_BPS_RT, rsi, mfi, cci,
)

UNIVERSE_18 = ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT",
               "AVAXUSDT", "SUIUSDT", "ADAUSDT", "APTUSDT", "BNBUSDT",
               "DOTUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "SOLUSDT",
               "UNIUSDT", "XRPUSDT", "BTCUSDT"]


def add_nfi_indicators(df, btc_df):
    """NFI 핵심 인디케이터 + protection layer 시리즈."""
    c = df["close_price"]
    df = df.copy()
    df["rsi_14"] = rsi(c, 14)
    df["mfi_14"] = mfi(df["high_price"], df["low_price"], c, df["base_volume"], 14)
    df["cci_20"] = cci(df["high_price"], df["low_price"], c, 20)
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()
    df["sma_200"] = c.rolling(200).mean()
    df["sma_200_dec_24"] = df["sma_200"] < df["sma_200"].shift(24)  # downtrend flag
    # local dip: (close - min(open[12])) / min  (12-bar lookback on 1h = 12h)
    open_min_12 = df["open_price"].rolling(12).min()
    df["dip_pct"] = (c - open_min_12) / open_min_12
    # BTC protection
    btc_ema50 = btc_df["close_price"].ewm(span=50, adjust=False).mean()
    btc_ema200 = btc_df["close_price"].ewm(span=200, adjust=False).mean()
    btc_uptrend = (btc_ema50 > btc_ema200) & (btc_df["close_price"] > btc_ema200)
    btc_pass_dict = dict(zip(btc_df["open_time"], btc_uptrend))
    df["btc_uptrend"] = df["open_time"].map(btc_pass_dict).fillna(False)
    return df


# ─── 시그널 ──────────────────────────────────────────────────────────────

def sig_nfi_1(df):
    """NFI #1 단순화: dip + oversold + uptrend + BTC protection."""
    return (
        (df["dip_pct"] > 0.015)        # 1h 12-bar dip 1.5%
        & (df["rsi_14"] < 40)
        & (df["mfi_14"] < 40)
        & (df["cci_20"] < -100)
        & (df["close_price"] > df["ema_200"])  # 알트 자체 uptrend
        & (df["ema_50"] > df["ema_200"])       # ema cross uptrend
        & (~df["sma_200_dec_24"])              # sma200 not declining
        & (df["btc_uptrend"])                  # BTC regime
    )


def sig_pb001_protected(df, threshold=70):
    """PB001 CH1 score + NFI protection layer."""
    score, _ = compute_ch1_score(df)
    df["score"] = score
    return (
        (df["score"] >= threshold)
        & (df["close_price"] > df["ema_200"])
        & (df["ema_50"] > df["ema_200"])
        & (~df["sma_200_dec_24"])
        & (df["btc_uptrend"])
    )


def backtest(symbol_dfs, signal_fn_factory, hold, label):
    total_n = 0
    total_net = 0.0
    wins = lottery5 = lottery10 = lottery20 = 0
    quarter_stats = [{"n":0, "net":0.0, "wins":0} for _ in range(4)]
    for sym, df in symbol_dfs.items():
        if df is None: continue
        if sym == "BTCUSDT":
            btc = df
        else:
            btc = symbol_dfs.get("BTCUSDT")
        if btc is None or len(btc) < 200: continue
        df_local = add_nfi_indicators(df, btc)
        entries = signal_fn_factory(df_local)
        df_local["fwd_pct"] = (df_local["close_price"].shift(-hold) / df_local["close_price"] - 1) * 10000
        e = df_local[entries & df_local["fwd_pct"].notna()]
        if len(e) == 0: continue
        net = e["fwd_pct"] - COST_BPS_RT
        total_n += len(e)
        total_net += float(net.sum())
        wins += int((net > 0).sum())
        lottery5 += int((net > 500).sum())
        lottery10 += int((net > 1000).sum())
        lottery20 += int((net > 2000).sum())
        # 분기별
        n_total = len(df_local)
        for q in range(4):
            a, b = int(n_total*q/4), int(n_total*(q+1)/4)
            sub_idx = [i for i in e.index if a <= i < b]
            if not sub_idx: continue
            sub = e.loc[sub_idx]
            sub_net = sub["fwd_pct"] - COST_BPS_RT
            quarter_stats[q]["n"] += len(sub)
            quarter_stats[q]["net"] += float(sub_net.sum())
            quarter_stats[q]["wins"] += int((sub_net > 0).sum())
    if total_n == 0:
        return None
    qs = []
    for q in quarter_stats:
        if q["n"] == 0:
            qs.append({"n":0, "net":0, "wr":0})
        else:
            qs.append({"n":q["n"], "net":round(q["net"]/q["n"],1), "wr":round(q["wins"]/q["n"],4)})
    return {
        "label": label,
        "n": total_n,
        "avg_net_bps": round(total_net / total_n, 2),
        "win_rate": round(wins / total_n, 4),
        "lottery5": lottery5, "lottery10": lottery10, "lottery20": lottery20,
        "quarters": qs,
    }


def main():
    print("=== G030/G031/G032 — NFI 영향 + PB001 protection ===\n")
    print(f"{'label':<45} {'n':>5} {'net':>9} {'WR':>7} {'L10%':>5} {'Q1':>15} {'Q2':>15} {'Q3':>15} {'Q4':>15}")
    all_dfs = {sym: load_klines(sym) for sym in UNIVERSE_18}

    variants = [
        ("G030a NFI #1 (dip+oversold+uptrend+BTC) hold24",  lambda d: sig_nfi_1(d), 24),
        ("G030b NFI #1 hold72",                              lambda d: sig_nfi_1(d), 72),
        ("G031a PB001 CH1≥70 + NFI protection hold24",       lambda d: sig_pb001_protected(d, 70), 24),
        ("G031b PB001 CH1≥70 + NFI protection hold72",       lambda d: sig_pb001_protected(d, 70), 72),
        ("G032a PB001 CH1≥80 + NFI protection hold24",       lambda d: sig_pb001_protected(d, 80), 24),
        ("G032b PB001 CH1≥80 + NFI protection hold72",       lambda d: sig_pb001_protected(d, 80), 72),
    ]
    results = []
    for label, fn, hold in variants:
        r = backtest(all_dfs, fn, hold, label)
        if r is None:
            print(f"{label:<45} no entries")
            continue
        results.append(r)
        qs = r["quarters"]
        q_str = lambda q: f"{q['n']:>3}/{q['net']:>+5.0f}/{q['wr']*100:>3.0f}%"
        print(f"{label:<45} {r['n']:>5} {r['avg_net_bps']:>+9.2f} {r['win_rate']*100:>6.1f}% {r['lottery10']:>5} {q_str(qs[0]):>15} {q_str(qs[1]):>15} {q_str(qs[2]):>15} {q_str(qs[3]):>15}")

    # robust check
    robust = [r for r in results if r['avg_net_bps'] > 0 and all(q['net'] > 0 for q in r['quarters'] if q['n']>0)]
    print(f"\n=== ROBUST (모든 분기 net>0): {len(robust)}개 ===")
    for r in robust:
        print(f"  ✅ {r['label']}: avg net={r['avg_net_bps']}bps WR={r['win_rate']*100:.1f}% n={r['n']} lottery10%={r['lottery10']}")
    if not robust:
        # 가장 일관된 것 - 분기별 net 의 min 이 가장 큰 것
        if results:
            best = max([r for r in results if any(q['n']>0 for q in r['quarters'])],
                       key=lambda r: min(q['net'] for q in r['quarters'] if q['n']>0))
            worst_q = min(q['net'] for q in best['quarters'] if q['n']>0)
            print(f"  최고 일관성: {best['label']} → worst Q net = {worst_q:.0f}, avg = {best['avg_net_bps']}")

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g030_nfi_protected_results.json"
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
        "robust": [r["label"] for r in robust],
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
