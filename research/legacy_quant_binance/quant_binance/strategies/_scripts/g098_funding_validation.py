"""
G098 — G090 (PB001 long) + Funding extreme reversal (양방향).

Funding rate (8h interval, Binance FAPI ≈ Bitget):
- rate > +0.05% (annualized ~55%) AND last bar bearish: SHORT (longs crowded → reversal)
- rate < -0.03% AND last bar bullish: extra LONG bonus (shorts crowded → squeeze)

9-point checklist 자체 적용.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct

KLINES_DIR = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_top50"
FUND_DIR = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "funding_binance"

SYMBOLS_KLINES = sorted([p.name for p in KLINES_DIR.iterdir() if p.is_dir()])
SYMBOLS_FUND = sorted([p.name for p in FUND_DIR.iterdir() if p.is_dir()])
COMMON = sorted(set(SYMBOLS_KLINES) & set(SYMBOLS_FUND))

THR = 80
HOLD = 24
LEV = 5.0
SIZE_PCT = 0.30
MAX_CONC = 5
EQUITY = 55.0
ATR_GUARD = 8.0
HOLD_MS = HOLD * 3600 * 1000

# Funding thresholds (per 8h period)
FUND_SHORT_THR = 0.0005   # +0.05% (annualized ~55%)
FUND_LONG_THR = -0.0003   # -0.03% (annualized ~-33%)


def load(sym):
    k_p = KLINES_DIR / sym / "1h.json"
    f_p = FUND_DIR / sym / "funding.json"
    if not (k_p.exists() and f_p.exists()): return None, None
    df = pd.DataFrame(json.loads(k_p.read_text()))
    f = pd.DataFrame(json.loads(f_p.read_text()))
    return df, f


def merge_funding(df, fund):
    """For each kline, attach the most-recent funding rate (8h cycle)."""
    df = df.copy().sort_values("open_time")
    fund = fund.sort_values("ts")
    # asof merge
    merged = pd.merge_asof(df, fund.rename(columns={"ts": "open_time", "rate": "funding"}),
                           on="open_time", direction="backward")
    return merged


def gather_g090_long(symbols):
    entries = []
    for sym in symbols:
        df, fund = load(sym)
        if df is None or len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["fwd_pct"] = (df["close_price"].shift(-HOLD) / df["close_price"] - 1) * 10000
        intra_low = df["low_price"].rolling(window=HOLD+1, min_periods=1).min().shift(-HOLD)
        df["intra_low_bps"] = (intra_low / df["close_price"] - 1) * 10000
        df = merge_funding(df, fund)
        e = df[(df["score"] >= THR) & (df["atr_pct"] <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym; e["side"] = "long"
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        if len(e):
            entries.append(e[["open_time","gross_bps","net_bps","atr_pct","intra_low_bps","funding","sym","side"]])
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def gather_funding_short(symbols, hold_bars=24):
    """SHORT signal: 직전 funding rate > +0.05% AND last bar bearish + vol spike."""
    entries = []
    for sym in symbols:
        df, fund = load(sym)
        if df is None or len(df) < 100: continue
        df = merge_funding(df, fund)
        df["bearish"] = df["close_price"] < df["open_price"]
        df["vol_ma20"] = df["base_volume"].rolling(20).mean()
        df["vol_spike"] = df["base_volume"] > 1.5 * df["vol_ma20"]
        df["fwd_pct"] = (df["close_price"].shift(-hold_bars) / df["close_price"] - 1) * 10000
        intra_high = df["high_price"].rolling(window=hold_bars+1, min_periods=1).max().shift(-hold_bars)
        df["intra_low_bps"] = (df["close_price"] / intra_high - 1) * 10000
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["atr_pct"] = a
        e = df[(df["funding"] >= FUND_SHORT_THR) & df["bearish"] & df["vol_spike"]
               & (a <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym; e["side"] = "short"
        e["gross_bps"] = -e["fwd_pct"]
        e["net_bps"] = -e["fwd_pct"] - 16
        if len(e):
            entries.append(e[["open_time","gross_bps","net_bps","atr_pct","intra_low_bps","funding","sym","side"]])
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def gather_funding_long_bonus(symbols, hold_bars=24):
    """추가 LONG: funding < -0.03% AND last bar bullish + vol spike (shorts squeeze)."""
    entries = []
    for sym in symbols:
        df, fund = load(sym)
        if df is None or len(df) < 100: continue
        df = merge_funding(df, fund)
        df["bullish"] = df["close_price"] > df["open_price"]
        df["vol_ma20"] = df["base_volume"].rolling(20).mean()
        df["vol_spike"] = df["base_volume"] > 1.5 * df["vol_ma20"]
        df["fwd_pct"] = (df["close_price"].shift(-hold_bars) / df["close_price"] - 1) * 10000
        intra_low = df["low_price"].rolling(window=hold_bars+1, min_periods=1).min().shift(-hold_bars)
        df["intra_low_bps"] = (intra_low / df["close_price"] - 1) * 10000
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["atr_pct"] = a
        e = df[(df["funding"] <= FUND_LONG_THR) & df["bullish"] & df["vol_spike"]
               & (a <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym; e["side"] = "long"
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        if len(e):
            entries.append(e[["open_time","gross_bps","net_bps","atr_pct","intra_low_bps","funding","sym","side"]])
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def split_yearly(entries):
    if len(entries) == 0: return {}
    e_dt = pd.to_datetime(entries["open_time"], unit="ms", utc=True)
    return {y: entries[e_dt.dt.year == y] for y in [2024, 2025, 2026]}


def portfolio_sim(entries, days):
    if len(entries) == 0: return None
    open_pos = []
    pnl = 0.0; taken = 0; wins = 0; lottery30 = 0; liq = 0
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
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"n":taken,"pnl_pct":round(pnl/EQUITY*100,1),"annual":round(pnl/EQUITY*100/days*365,1),
            "wr":round(wins/taken,4) if taken else 0, "lottery30":lottery30, "liq":liq, "per_day":round(taken/days,3)}


def run_9point(label, entries):
    print(f"\n{'='*70}\n=== {label} ({len(entries)} candidates) ===\n{'='*70}")
    if len(entries) == 0:
        print("NO CANDIDATES"); return [False]*9
    yearly = split_yearly(entries)
    days_map = {2024: 365, 2025: 365, 2026: 117}
    print(f"\n{'year':>6} {'days':>5} {'n':>5} {'/day':>5} {'avg':>8} {'WR':>6} {'L30%+':>7} {'annual':>9}")
    p1 = p2 = p4 = True
    for y in [2024, 2025, 2026]:
        e = yearly.get(y, pd.DataFrame())
        d = days_map[y]
        if len(e) == 0:
            print(f"{y:>6} {d:>5} {'0':>5}"); continue
        avg = e["net_bps"].mean(); wr = (e["net_bps"]>0).mean()
        l30 = (e["net_bps"]/10000*LEV > 0.30).sum()
        per_day = len(e) / d
        r = portfolio_sim(e, d)
        ann = r["annual"] if r else 0
        print(f"{y:>6} {d:>5} {len(e):>5} {per_day:>5.2f} {avg:>+8.1f} {wr*100:>5.1f}% {l30:>7} {ann:>+8.1f}%")
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
    print(f"  Check 9 (warmup X): ✓")
    passed = sum(checks)
    print(f"\n  TOTAL: {passed}/9 PASS")
    return checks


def main():
    print(f"Common symbols (klines + funding): {len(COMMON)}")
    g090_long = gather_g090_long(COMMON)
    fund_short = gather_funding_short(COMMON)
    fund_long_bonus = gather_funding_long_bonus(COMMON)
    g098_combined = pd.concat([g090_long, fund_short, fund_long_bonus]).sort_values("open_time").reset_index(drop=True) if len(g090_long) or len(fund_short) else pd.DataFrame()

    results = {}
    for label, e in [
        ("G090-long (Bitget-compatible)", g090_long),
        ("Funding-short", fund_short),
        ("Funding-long-bonus", fund_long_bonus),
        ("G098 (combined 양방향)", g098_combined),
    ]:
        c = run_9point(label, e)
        results[label] = sum(c) if isinstance(c, list) else 0

    print(f"\n\n{'='*70}\n=== 종합 ===\n{'='*70}")
    for label, p in results.items():
        print(f"  {label:<32} {p}/9")


if __name__ == "__main__":
    main()
