"""
G090/G095/G096 — 9-point 사전 검증.
G090: G070 logic on top 50 universe (빈도 ↑ 시도)
G095: Liquidation cascade reversal LONG (24h drop>8% + 양봉 volume spike)
G096: Pump exhaustion SHORT (24h pump>12% + 음봉 volume spike)
G097: G090 + G095 + G096 결합 (양방향 portfolio)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct

DIR = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_top50"
SYMBOLS = sorted([p.name for p in DIR.iterdir() if p.is_dir()])

THR = 80
HOLD = 24
LEV = 5.0
SIZE_PCT = 0.30
MAX_CONC = 5
EQUITY = 55.0
ATR_GUARD = 8.0
HOLD_MS = HOLD * 3600 * 1000


def load(sym):
    p = DIR / sym / "1h.json"
    return pd.DataFrame(json.loads(p.read_text())) if p.exists() else None


def gather_g090(symbols):
    """G070 logic on wide universe."""
    entries = []
    for sym in symbols:
        df = load(sym)
        if df is None or len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["fwd_pct"] = (df["close_price"].shift(-HOLD) / df["close_price"] - 1) * 10000
        # intra-bar low (5x lev liquidation)
        intra_low = df["low_price"].rolling(window=HOLD+1, min_periods=1).min().shift(-HOLD)
        df["intra_low_bps"] = (intra_low / df["close_price"] - 1) * 10000
        e = df[(df["score"] >= THR) & (df["atr_pct"] <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        e["side"] = "long"
        if len(e):
            entries.append(e[["open_time","gross_bps","net_bps","atr_pct","intra_low_bps","sym","side"]])
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def gather_g095(symbols, hold=12):
    """Liquidation cascade reversal LONG.
    조건: 직전 24h 누적 -8% 이하 AND 마지막 봉 양봉 + 거래량 > 1.5× 20봉 평균."""
    entries = []
    HOLD_BARS = hold
    for sym in symbols:
        df = load(sym)
        if df is None or len(df) < 100: continue
        df["ret_24h"] = (df["close_price"] / df["close_price"].shift(24) - 1) * 100
        df["bullish"] = df["close_price"] > df["open_price"]
        df["vol_ma20"] = df["base_volume"].rolling(20).mean()
        df["vol_spike"] = df["base_volume"] > 1.5 * df["vol_ma20"]
        df["fwd_pct"] = (df["close_price"].shift(-HOLD_BARS) / df["close_price"] - 1) * 10000
        intra_low = df["low_price"].rolling(window=HOLD_BARS+1, min_periods=1).min().shift(-HOLD_BARS)
        df["intra_low_bps"] = (intra_low / df["close_price"] - 1) * 10000
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["atr_pct"] = a
        e = df[(df["ret_24h"] <= -8.0) & df["bullish"] & df["vol_spike"] & (a <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym; e["side"] = "long"
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        if len(e):
            entries.append(e[["open_time","gross_bps","net_bps","atr_pct","intra_low_bps","sym","side"]])
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def gather_g096(symbols, hold=12):
    """Pump exhaustion SHORT.
    조건: 직전 24h 누적 +12% 이상 AND 마지막 봉 음봉 + 거래량 > 1.5× 20봉 평균."""
    entries = []
    HOLD_BARS = hold
    for sym in symbols:
        df = load(sym)
        if df is None or len(df) < 100: continue
        df["ret_24h"] = (df["close_price"] / df["close_price"].shift(24) - 1) * 100
        df["bearish"] = df["close_price"] < df["open_price"]
        df["vol_ma20"] = df["base_volume"].rolling(20).mean()
        df["vol_spike"] = df["base_volume"] > 1.5 * df["vol_ma20"]
        df["fwd_pct"] = (df["close_price"].shift(-HOLD_BARS) / df["close_price"] - 1) * 10000
        # short: gross = -fwd
        intra_high = df["high_price"].rolling(window=HOLD_BARS+1, min_periods=1).max().shift(-HOLD_BARS)
        df["intra_low_bps"] = (df["close_price"] / intra_high - 1) * 10000  # short 관점 unfavorable
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["atr_pct"] = a
        e = df[(df["ret_24h"] >= 12.0) & df["bearish"] & df["vol_spike"] & (a <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym; e["side"] = "short"
        e["gross_bps"] = -e["fwd_pct"]  # short
        e["net_bps"] = -e["fwd_pct"] - 16
        if len(e):
            entries.append(e[["open_time","gross_bps","net_bps","atr_pct","intra_low_bps","sym","side"]])
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def split_yearly(entries):
    if len(entries) == 0: return {}
    e_dt = pd.to_datetime(entries["open_time"], unit="ms", utc=True)
    return {y: entries[e_dt.dt.year == y] for y in [2024, 2025, 2026]}


def portfolio_sim(entries, lev=LEV, days=None):
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
            net_pct = row["net_bps"] / 10000 * lev
        pnl += margin * net_pct
        taken += 1
        if net_pct > 0: wins += 1
        if net_pct > 0.30: lottery30 += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"n":taken,"pnl_pct":round(pnl/EQUITY*100,1),
            "annual":round(pnl/EQUITY*100/days*365,1) if days else 0,
            "wr":round(wins/taken,4) if taken else 0, "lottery30":lottery30, "liq":liq,
            "per_day":round(taken/days,3) if days else 0}


def run_9point(label, entries):
    print(f"\n{'='*70}\n=== {label} ({len(entries)} candidates) ===\n{'='*70}")
    if len(entries) == 0:
        print("NO CANDIDATES"); return 0, []
    yearly = split_yearly(entries)
    days_map = {2024: 365, 2025: 365, 2026: 117}
    print(f"\n{'year':>6} {'days':>5} {'n':>5} {'/day':>5} {'avg_net':>9} {'WR':>6} {'L30%+':>6}")
    p1 = p2 = p4 = True
    for y in [2024, 2025, 2026]:
        e = yearly.get(y, pd.DataFrame())
        d = days_map[y]
        if len(e) == 0:
            print(f"{y:>6} {d:>5} {'0':>5}"); continue
        avg = e["net_bps"].mean(); wr = (e["net_bps"]>0).mean()
        l30 = (e["net_bps"]/10000*LEV > 0.30).sum()
        per_day = len(e) / d
        print(f"{y:>6} {d:>5} {len(e):>5} {per_day:>5.2f} {avg:>+9.1f} {wr*100:>5.1f}% {l30:>6}")
        r = portfolio_sim(e, days=d)
        if r and r["annual"] <= 0: p2 = False
        if avg <= 0: p1 = False
        if wr < 0.65: p4 = False

    avg_all = entries["net_bps"].mean()
    p3 = avg_all >= 50

    total_days = sum(days_map.values())
    avg_per_day_total = len(entries) / total_days
    p5_freq = avg_per_day_total >= 3
    p5_lottery = avg_all/100 >= 3

    print(f"\n  Check 1 (yearly avg>0): {'✓' if p1 else '❌'}")
    print(f"  Check 2 (yearly portfolio>0): {'✓' if p2 else '❌'}")
    print(f"  Check 3 (avg≥+50bps): {avg_all:+.0f} {'✓' if p3 else '❌'}")
    print(f"  Check 4 (yearly WR≥65%): {'✓' if p4 else '❌'}")
    print(f"  Check 5 (6축): lottery={'✓' if p5_lottery else '❌'} | 5x✓ | 양방향=? | freq{avg_per_day_total:.2f}/day {'✓' if p5_freq else '❌'} | 단기✓ | 빠검✓")

    cost_pass = True
    for c in [16, 25, 30]:
        net = (entries["gross_bps"] - c).mean()
        if net <= 0: cost_pass = False
    p6 = cost_pass
    print(f"  Check 6 (cost): {'✓' if p6 else '❌'}")

    deep = ((entries["intra_low_bps"]/10000) < -0.20).sum()
    p7 = deep / max(len(entries), 1) < 0.10
    print(f"  Check 7 (5x liq<10%): {deep}/{len(entries)} = {deep/max(len(entries),1)*100:.1f}% {'✓' if p7 else '❌'}")

    p8 = len(entries) >= 50
    print(f"  Check 8 (n≥50): {len(entries)} {'✓' if p8 else '❌'}")
    print(f"  Check 9 (warmup X): ✓")
    return [p1, p2, p3, p4, p5_freq and p5_lottery, p6, p7, p8, True], (avg_all, avg_per_day_total)


def main():
    print(f"Universe: top 50 alts (memes + majors)\n")
    g090 = gather_g090(SYMBOLS)
    g095 = gather_g095(SYMBOLS, hold=12)
    g096 = gather_g096(SYMBOLS, hold=12)
    g097 = pd.concat([g090, g095, g096]).sort_values("open_time").reset_index(drop=True) if len(g090) or len(g095) or len(g096) else pd.DataFrame()

    results = {}
    for label, e in [
        ("G090 (G070 logic / 50 alts)", g090),
        ("G095 (cascade reversal LONG)", g095),
        ("G096 (pump exhaustion SHORT)", g096),
        ("G097 (G090+G095+G096 결합)", g097),
    ]:
        checks, info = run_9point(label, e)
        passed = sum(checks) if isinstance(checks, list) else 0
        results[label] = (passed, info)

    print(f"\n\n{'='*70}\n=== 종합 ===\n{'='*70}")
    print(f"{'전략':<32} {'PASS':>6} {'avg_net':>9} {'/day':>6}")
    for label, (passed, info) in results.items():
        if isinstance(info, tuple):
            avg_net, per_day = info
            print(f"{label:<32} {passed}/9    {avg_net:>+9.0f}  {per_day:>5.2f}")
        else:
            print(f"{label:<32} {passed}/9")


if __name__ == "__main__":
    main()
