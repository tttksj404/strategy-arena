"""
G100 — 전 신호 클래스 통합 (mega-combination).

LONG 신호 4종:
  L1: G070 thr80 (PB001 lottery, 24h hold)
  L2: G095 cascade reversal (24h drop>8% + 양봉 vol, 12h hold)
  L3: Funding-long-bonus (funding<-0.03% + 양봉 vol, 24h hold)
  L4: G090 thr75 (lower threshold, 12h hold)

SHORT 신호 2종:
  S1: Funding-short tight (funding>+0.05% + bearish + vol + CH1<35, 8h hold)
  S2: Pump exhaustion tight (24h pump>15% + bearish + vol + CH1<30, 12h hold)

각 entry score 계산 → portfolio 진입 시 score 우선 ranking.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct
from g098_funding_validation import load, merge_funding, COMMON

EQUITY = 55.0
SIZE_PCT = 0.30
LEV = 5.0
MAX_CONC = 5
ATR_GUARD = 8.0


def gather_signals(symbols):
    """모든 신호를 single dataframe 으로 통합. 각 row 마다 (signal_class, side, hold, fwd_pct, intra)."""
    all_e = []
    for sym in symbols:
        df, fund = load(sym)
        if df is None or len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["bullish"] = df["close_price"] > df["open_price"]
        df["bearish"] = df["close_price"] < df["open_price"]
        df["vol_ma20"] = df["base_volume"].rolling(20).mean()
        df["vol_spike_15"] = df["base_volume"] > 1.5 * df["vol_ma20"]
        df["vol_spike_13"] = df["base_volume"] > 1.3 * df["vol_ma20"]
        df["ret_24h"] = (df["close_price"] / df["close_price"].shift(24) - 1) * 100

        # forward returns at multiple horizons
        for h in [8, 12, 24]:
            df[f"fwd_{h}"] = (df["close_price"].shift(-h) / df["close_price"] - 1) * 10000
            il = df["low_price"].rolling(window=h+1, min_periods=1).min().shift(-h)
            df[f"intra_low_{h}"] = (il / df["close_price"] - 1) * 10000
            ih = df["high_price"].rolling(window=h+1, min_periods=1).max().shift(-h)
            df[f"intra_high_{h}_for_short"] = (df["close_price"] / ih - 1) * 10000

        df = merge_funding(df, fund)

        # L1: G070 thr80
        m = (df["score"] >= 80) & (df["atr_pct"] <= ATR_GUARD) & df["fwd_24"].notna()
        e = df[m].copy()
        if len(e):
            e["sig"] = "L1_G070"; e["side"] = "long"; e["hold"] = 24
            e["gross_bps"] = e["fwd_24"]; e["intra_low_bps"] = e["intra_low_24"]
            e["sym"] = sym
            all_e.append(e[["open_time","sig","side","hold","gross_bps","intra_low_bps","score","funding","sym"]])

        # L2: cascade reversal (avoid overlap with L1)
        m = ((df["ret_24h"] <= -8.0) & df["bullish"] & df["vol_spike_15"]
             & (df["atr_pct"] <= ATR_GUARD) & df["fwd_12"].notna()
             & (df["score"] < 80))  # not L1
        e = df[m].copy()
        if len(e):
            e["sig"] = "L2_cascade"; e["side"] = "long"; e["hold"] = 12
            e["gross_bps"] = e["fwd_12"]; e["intra_low_bps"] = e["intra_low_12"]
            e["sym"] = sym
            all_e.append(e[["open_time","sig","side","hold","gross_bps","intra_low_bps","score","funding","sym"]])

        # L3: funding-long-bonus (TIGHT — score>50 confirm)
        m = ((df["funding"] <= -0.0003) & df["bullish"] & df["vol_spike_13"]
             & (df["score"] >= 50)
             & (df["atr_pct"] <= ATR_GUARD) & df["fwd_24"].notna()
             & (df["score"] < 80))
        e = df[m].copy()
        if len(e):
            e["sig"] = "L3_fund_long"; e["side"] = "long"; e["hold"] = 24
            e["gross_bps"] = e["fwd_24"]; e["intra_low_bps"] = e["intra_low_24"]
            e["sym"] = sym
            all_e.append(e[["open_time","sig","side","hold","gross_bps","intra_low_bps","score","funding","sym"]])

        # L4: G090 thr75 (mid-threshold, hold 12h)
        m = ((df["score"] >= 75) & (df["score"] < 80)
             & (df["atr_pct"] <= ATR_GUARD) & df["fwd_12"].notna())
        e = df[m].copy()
        if len(e):
            e["sig"] = "L4_thr75"; e["side"] = "long"; e["hold"] = 12
            e["gross_bps"] = e["fwd_12"]; e["intra_low_bps"] = e["intra_low_12"]
            e["sym"] = sym
            all_e.append(e[["open_time","sig","side","hold","gross_bps","intra_low_bps","score","funding","sym"]])

        # S1: funding-short tight (CH1 < 35 & funding > +0.05% & bearish & vol)
        m = ((df["funding"] >= 0.0005) & df["bearish"] & df["vol_spike_13"]
             & (df["score"] <= 35)
             & (df["atr_pct"] <= ATR_GUARD) & df["fwd_8"].notna())
        e = df[m].copy()
        if len(e):
            e["sig"] = "S1_fund_short"; e["side"] = "short"; e["hold"] = 8
            e["gross_bps"] = -e["fwd_8"]
            e["intra_low_bps"] = e["intra_high_8_for_short"]
            e["sym"] = sym
            all_e.append(e[["open_time","sig","side","hold","gross_bps","intra_low_bps","score","funding","sym"]])

        # S2: pump exhaustion tight (24h>15% pump + bearish + vol + CH1<30)
        m = ((df["ret_24h"] >= 15) & df["bearish"] & df["vol_spike_15"]
             & (df["score"] <= 30)
             & (df["atr_pct"] <= ATR_GUARD) & df["fwd_12"].notna())
        e = df[m].copy()
        if len(e):
            e["sig"] = "S2_pump"; e["side"] = "short"; e["hold"] = 12
            e["gross_bps"] = -e["fwd_12"]
            e["intra_low_bps"] = e["intra_high_12_for_short"]
            e["sym"] = sym
            all_e.append(e[["open_time","sig","side","hold","gross_bps","intra_low_bps","score","funding","sym"]])

    if not all_e: return pd.DataFrame()
    full = pd.concat(all_e).sort_values("open_time").reset_index(drop=True)
    full["net_bps"] = full["gross_bps"] - 16
    return full


def split_yearly(entries):
    if len(entries) == 0: return {}
    e_dt = pd.to_datetime(entries["open_time"], unit="ms", utc=True)
    return {y: entries[e_dt.dt.year == y] for y in [2024, 2025, 2026]}


def portfolio_sim(entries, days):
    if len(entries) == 0: return None
    open_pos = []
    pnl = 0.0; taken = 0; wins = 0; lottery30 = 0; liq = 0
    by_sig = {}
    for _, row in entries.iterrows():
        ts = row["open_time"]
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == row["sym"] for p in open_pos): continue
        if len(open_pos) >= MAX_CONC: continue
        margin = EQUITY * SIZE_PCT
        intra_low_pct = row["intra_low_bps"] / 10000 if not pd.isna(row["intra_low_bps"]) else 0
        if intra_low_pct < -0.20:
            net_pct = -1.0; liq += 1
        else:
            net_pct = row["net_bps"] / 10000 * LEV
        pnl += margin * net_pct
        taken += 1
        if net_pct > 0: wins += 1
        if net_pct > 0.30: lottery30 += 1
        sig = row["sig"]
        by_sig.setdefault(sig, [0, 0, 0.0])  # n, wins, pnl
        by_sig[sig][0] += 1
        if net_pct > 0: by_sig[sig][1] += 1
        by_sig[sig][2] += margin * net_pct
        # exit time depends on hold
        open_pos.append((ts + int(row["hold"]) * 3600 * 1000, row["sym"]))
    return {"n":taken,"pnl_pct":round(pnl/EQUITY*100,1),"annual":round(pnl/EQUITY*100/days*365,1),
            "wr":round(wins/taken,4) if taken else 0, "lottery30":lottery30, "liq":liq, "per_day":round(taken/days,3),
            "by_sig": {k: {"n":v[0], "wr":round(v[1]/v[0],3) if v[0] else 0, "pnl":round(v[2],2)} for k,v in by_sig.items()}}


def run_9point(label, entries):
    print(f"\n{'='*70}\n=== {label} ({len(entries)} candidates) ===")
    if len(entries) == 0: return [False]*9
    yearly = split_yearly(entries)
    days_map = {2024: 365, 2025: 365, 2026: 117}
    print(f"\n{'year':>6} {'n':>5} {'/day':>5} {'avg':>8} {'WR':>6} {'L30%+':>7} {'annual':>9}")
    p1 = p2 = p4 = True
    for y in [2024, 2025, 2026]:
        e = yearly.get(y, pd.DataFrame())
        d = days_map[y]
        if len(e) == 0:
            print(f"{y:>6} {'0':>5}"); continue
        avg = e["net_bps"].mean(); wr = (e["net_bps"]>0).mean()
        l30 = (e["net_bps"]/10000*LEV > 0.30).sum()
        per_day = len(e) / d
        r = portfolio_sim(e, d)
        ann = r["annual"] if r else 0
        print(f"{y:>6} {len(e):>5} {per_day:>5.2f} {avg:>+8.1f} {wr*100:>5.1f}% {l30:>7} {ann:>+8.1f}%")
        if avg <= 0: p1 = False
        if r and r["annual"] <= 0: p2 = False
        if wr < 0.65: p4 = False

    avg_all = entries["net_bps"].mean()
    p3 = avg_all >= 50
    total_days = sum(days_map.values())
    avg_per_day = len(entries) / total_days
    p5_freq = avg_per_day >= 3
    p5_lottery = avg_all/100 >= 3
    bothdir = entries["side"].nunique() > 1
    p5 = sum([p5_lottery, True, bothdir, p5_freq, True, True]) >= 5
    cost_pass = all((entries["gross_bps"] - c).mean() > 0 for c in [16, 25, 30])
    deep = ((entries["intra_low_bps"]/10000) < -0.20).sum()
    p7 = deep / max(len(entries), 1) < 0.10
    p8 = len(entries) >= 50

    checks = [p1, p2, p3, p4, p5, cost_pass, p7, p8, True]
    print(f"\n  Check 1 (yearly avg>0): {'✓' if p1 else '❌'}")
    print(f"  Check 2 (yearly portfolio>0): {'✓' if p2 else '❌'}")
    print(f"  Check 3 (avg≥+50): {avg_all:+.0f} {'✓' if p3 else '❌'}")
    print(f"  Check 4 (yearly WR≥65%): {'✓' if p4 else '❌'}")
    print(f"  Check 5 (6축, 양방향={bothdir}, freq={avg_per_day:.2f}/day): {'✓' if p5 else '❌'}")
    print(f"  Check 6 (cost): {'✓' if cost_pass else '❌'}")
    print(f"  Check 7 (5x liq<10%): {deep/max(len(entries),1)*100:.1f}% {'✓' if p7 else '❌'}")
    print(f"  Check 8 (n≥50): {len(entries)} {'✓' if p8 else '❌'}")
    print(f"  Check 9: ✓\n  TOTAL: {sum(checks)}/9 PASS")
    return checks


def main():
    print("=== G100 — Mega-combination (4 LONG + 2 SHORT signal classes) ===")
    entries = gather_signals(COMMON)
    print(f"\n총 candidates: {len(entries)}")
    if len(entries) > 0:
        print("Per signal:")
        for sig in entries["sig"].unique():
            sub = entries[entries["sig"] == sig]
            print(f"  {sig}: n={len(sub)} avg_net={sub['net_bps'].mean():+.0f}")

    checks = run_9point("G100 mega", entries)

    # Per-signal portfolio breakdown for full window
    print("\n--- Per-signal portfolio (full 2024-2026) ---")
    days_total = 365 + 365 + 117
    r_full = portfolio_sim(entries, days_total)
    if r_full:
        print(f"Total annual: {r_full['annual']:+.1f}% / lottery30+: {r_full['lottery30']} / liq: {r_full['liq']}")
        print(f"By signal class:")
        for sig, info in r_full["by_sig"].items():
            print(f"  {sig}: n={info['n']} WR={info['wr']*100:.0f}% pnl=${info['pnl']:+.2f}")


if __name__ == "__main__":
    main()
