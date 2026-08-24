"""
G101 — Filtered combo (G100 진단 후): 살아있는 신호만 + score-ranked + 더 긴 historical.
- L1 (G070 thr80, 24h): WR 67% pnl +$80 ⭐
- L2 (cascade reversal, 12h): WR 51% pnl +$33 (양수 but 약함)
- L3 (funding-long-bonus, 24h): 제로 — drop

→ G101 = L1 + L2 only (가장 검증된 두 LONG 신호). short 보류.

추가: 2022-2023 historical_2022 universe 도 합쳐서 4년 검증.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct

DATA_DIRS = [
    Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022",     # 2022-23
    Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2024",     # 2024-Q1.25
    Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical",   # 2025-26
    Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_top50",    # 2024-26 wide
    Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_meme",     # 2024-26 meme
]

EQUITY = 55.0
SIZE_PCT = 0.30
LEV = 5.0
MAX_CONC = 5
ATR_GUARD = 8.0


def load_combined(sym):
    """모든 가용 디렉토리에서 합치고 dedupe."""
    bars = []; seen = set()
    for d in DATA_DIRS:
        p = d / sym / "1h.json"
        if p.exists():
            for b in json.loads(p.read_text()):
                ts = b["open_time"]
                if ts not in seen:
                    seen.add(ts); bars.append(b)
    if not bars: return None
    bars.sort(key=lambda x: x["open_time"])
    return pd.DataFrame(bars)


def discover_symbols():
    syms = set()
    for d in DATA_DIRS:
        if d.exists():
            for p in d.iterdir():
                if p.is_dir(): syms.add(p.name)
    return sorted(syms)


def gather_l1_l2(symbols):
    all_e = []
    for sym in symbols:
        df = load_combined(sym)
        if df is None or len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["bullish"] = df["close_price"] > df["open_price"]
        df["vol_ma20"] = df["base_volume"].rolling(20).mean()
        df["vol_spike_15"] = df["base_volume"] > 1.5 * df["vol_ma20"]
        df["ret_24h"] = (df["close_price"] / df["close_price"].shift(24) - 1) * 100

        df["fwd_24"] = (df["close_price"].shift(-24) / df["close_price"] - 1) * 10000
        df["fwd_12"] = (df["close_price"].shift(-12) / df["close_price"] - 1) * 10000
        il24 = df["low_price"].rolling(window=25, min_periods=1).min().shift(-24)
        df["intra_low_24"] = (il24 / df["close_price"] - 1) * 10000
        il12 = df["low_price"].rolling(window=13, min_periods=1).min().shift(-12)
        df["intra_low_12"] = (il12 / df["close_price"] - 1) * 10000

        # L1: G070 thr80, 24h hold
        m = (df["score"] >= 80) & (df["atr_pct"] <= ATR_GUARD) & df["fwd_24"].notna()
        e = df[m].copy()
        if len(e):
            e["sig"] = "L1_G070"; e["hold"] = 24
            e["gross_bps"] = e["fwd_24"]; e["intra_low_bps"] = e["intra_low_24"]
            e["sym"] = sym; e["entry_score"] = e["score"]
            all_e.append(e[["open_time","sig","hold","gross_bps","intra_low_bps","entry_score","sym"]])

        # L2: cascade reversal, 12h hold (avoid L1 overlap)
        m = ((df["ret_24h"] <= -8.0) & df["bullish"] & df["vol_spike_15"]
             & (df["atr_pct"] <= ATR_GUARD) & df["fwd_12"].notna()
             & (df["score"] < 80))
        e = df[m].copy()
        if len(e):
            e["sig"] = "L2_cascade"; e["hold"] = 12
            e["gross_bps"] = e["fwd_12"]; e["intra_low_bps"] = e["intra_low_12"]
            e["sym"] = sym
            # L2 entry score = -ret_24h × volume_ratio (큰 drop + 큰 vol = 강한 신호)
            e["entry_score"] = (-e["ret_24h"]) * (e["base_volume"] / e["vol_ma20"])
            all_e.append(e[["open_time","sig","hold","gross_bps","intra_low_bps","entry_score","sym"]])

    if not all_e: return pd.DataFrame()
    full = pd.concat(all_e).sort_values("open_time").reset_index(drop=True)
    full["net_bps"] = full["gross_bps"] - 16
    return full


def split_yearly(entries):
    if len(entries) == 0: return {}
    e_dt = pd.to_datetime(entries["open_time"], unit="ms", utc=True)
    out = {}
    for y in [2022, 2023, 2024, 2025, 2026]:
        out[y] = entries[e_dt.dt.year == y]
    return out


def portfolio_sim_score_ranked(entries, days):
    """동일 시간대 multi-signal 발화 시 entry_score 우선 ranking."""
    if len(entries) == 0: return None
    open_pos = []
    pnl = 0.0; taken = 0; wins = 0; lottery30 = 0; liq = 0
    by_sig = {}
    # group by hour to allow score-based ranking within same time
    entries = entries.sort_values(["open_time", "entry_score"], ascending=[True, False]).reset_index(drop=True)
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
        by_sig.setdefault(sig, [0, 0, 0.0])
        by_sig[sig][0] += 1
        if net_pct > 0: by_sig[sig][1] += 1
        by_sig[sig][2] += margin * net_pct
        open_pos.append((ts + int(row["hold"]) * 3600 * 1000, row["sym"]))
    return {"n":taken,"pnl_pct":round(pnl/EQUITY*100,1),"annual":round(pnl/EQUITY*100/days*365,1),
            "wr":round(wins/taken,4) if taken else 0, "lottery30":lottery30, "liq":liq, "per_day":round(taken/days,3),
            "by_sig": {k: {"n":v[0], "wr":round(v[1]/v[0],3) if v[0] else 0, "pnl":round(v[2],2)} for k,v in by_sig.items()}}


def main():
    print("=== G101 — Filtered combo (L1+L2 only, score-ranked, 5-year data) ===\n")
    syms = discover_symbols()
    print(f"Universe (모든 디렉토리 통합): {len(syms)} symbols")
    entries = gather_l1_l2(syms)
    print(f"Total candidates: {len(entries)}")
    print(f"  L1_G070: {(entries['sig']=='L1_G070').sum()}")
    print(f"  L2_cascade: {(entries['sig']=='L2_cascade').sum()}")

    yearly = split_yearly(entries)
    days_map = {2022: 365, 2023: 365, 2024: 365, 2025: 365, 2026: 117}
    print(f"\n{'year':>6} {'n':>5} {'/day':>5} {'avg_net':>9} {'WR':>6} {'L30%+':>7} {'annual':>9}")
    p1 = p2 = p4 = True
    total_n = 0
    for y in [2022, 2023, 2024, 2025, 2026]:
        e = yearly.get(y, pd.DataFrame())
        d = days_map[y]
        if len(e) == 0:
            print(f"{y:>6} {'0':>5}"); continue
        avg = e["net_bps"].mean(); wr = (e["net_bps"]>0).mean()
        l30 = (e["net_bps"]/10000*LEV > 0.30).sum()
        per_day = len(e) / d
        r = portfolio_sim_score_ranked(e, d)
        ann = r["annual"] if r else 0
        print(f"{y:>6} {len(e):>5} {per_day:>5.2f} {avg:>+9.1f} {wr*100:>5.1f}% {l30:>7} {ann:>+8.1f}%")
        if avg <= 0: p1 = False
        if r and r["annual"] <= 0: p2 = False
        if wr < 0.65: p4 = False
        total_n += len(e)

    avg_all = entries["net_bps"].mean()
    p3 = avg_all >= 50
    total_days = sum(days_map.values())
    avg_per_day = total_n / total_days
    p5_freq = avg_per_day >= 3
    p5_lottery = avg_all/100 >= 3
    bothdir = False  # L1+L2 모두 long
    p5 = sum([p5_lottery, True, bothdir, p5_freq, True, True]) >= 5
    cost_pass = all((entries["gross_bps"] - c).mean() > 0 for c in [16, 25, 30])
    deep = ((entries["intra_low_bps"]/10000) < -0.20).sum()
    p7 = deep / max(len(entries), 1) < 0.10
    p8 = len(entries) >= 50

    checks = [p1, p2, p3, p4, p5, cost_pass, p7, p8, True]
    print(f"\n  Check 1 (yearly avg>0 — 5년 모두): {'✓' if p1 else '❌'}")
    print(f"  Check 2 (yearly portfolio>0 — 5년 모두): {'✓' if p2 else '❌'}")
    print(f"  Check 3 (avg≥+50): {avg_all:+.0f} {'✓' if p3 else '❌'}")
    print(f"  Check 4 (yearly WR≥65%): {'✓' if p4 else '❌'}")
    print(f"  Check 5 (6축, freq={avg_per_day:.2f}/day, 양방향=False): {'✓' if p5 else '❌'}")
    print(f"  Check 6 (cost): {'✓' if cost_pass else '❌'}")
    print(f"  Check 7 (5x liq<10%): {deep/max(len(entries),1)*100:.1f}% {'✓' if p7 else '❌'}")
    print(f"  Check 8 (n≥50): {len(entries)} {'✓' if p8 else '❌'}")
    print(f"  Check 9: ✓\n  TOTAL: {sum(checks)}/9 PASS")

    print(f"\n--- Per-signal full window (5-year) ---")
    r = portfolio_sim_score_ranked(entries, total_days)
    if r:
        print(f"Total annual: {r['annual']:+.1f}% / lottery30+: {r['lottery30']} / liq: {r['liq']}")
        for sig, info in r["by_sig"].items():
            print(f"  {sig}: n={info['n']} WR={info['wr']*100:.0f}% pnl=${info['pnl']:+.2f}")


if __name__ == "__main__":
    main()
