"""G080 — G070 logic on memecoin-only universe. 9-point checklist 사전 검증."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct

MEME_DIR = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_meme"
DATA_25 = Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical"  # PEPE/WIF/DOGE 일부 fallback

# Available memecoins
MEMES = ["SHIBUSDT","FLOKIUSDT","BONKUSDT","MEMEUSDT","PNUTUSDT","BOMEUSDT","NOTUSDT","TURBOUSDT","NEIROUSDT","DOGEUSDT","PEPEUSDT","WIFUSDT"]

THR = 80
HOLD = 24
LEV = 5.0
SIZE_PCT = 0.30
MAX_CONC = 5
EQUITY = 55.0
ATR_GUARD = 8.0
HOLD_MS = HOLD * 3600 * 1000


def load(sym):
    for base in (MEME_DIR, DATA_25):
        p = base / sym / "1h.json"
        if p.exists():
            return pd.DataFrame(json.loads(p.read_text()))
    return None


def gather(symbols, threshold=THR, hold=HOLD):
    entries = []
    for sym in symbols:
        df = load(sym)
        if df is None or len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["fwd_pct"] = (df["close_price"].shift(-hold) / df["close_price"] - 1) * 10000
        # intra-bar low for liquidation check
        intra_low = []
        for i in range(len(df)):
            if i + hold >= len(df): intra_low.append(np.nan); continue
            window = df.iloc[i:i+hold+1]
            entry = window.iloc[0]["close_price"]
            intra_low.append((window["low_price"].min() / entry - 1) * 10000)
        df["intra_low_bps"] = intra_low
        e = df[(df["score"] >= threshold) & (df["atr_pct"] <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        if len(e):
            entries.append(e[["open_time","score","gross_bps","net_bps","atr_pct","intra_low_bps","sym"]])
    if not entries: return pd.DataFrame()
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True)


def split_by_year(entries):
    """24~26 데이터를 연도별 분할."""
    if len(entries) == 0: return {}
    out = {}
    e_dt = pd.to_datetime(entries["open_time"], unit="ms", utc=True)
    for year in [2024, 2025, 2026]:
        mask = (e_dt.dt.year == year)
        out[year] = entries[mask]
    return out


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
    return {
        "n": taken, "pnl_usd": round(pnl, 2),
        "pnl_pct": round(pnl/EQUITY*100, 1),
        "annual": round(pnl/EQUITY*100/days*365, 1) if days else 0,
        "wr": round(wins/taken, 4) if taken else 0,
        "lottery30": lottery30, "liquidations": liq,
        "trades_per_day": round(taken/days, 3) if days else 0,
    }


def main():
    print("=== G080 — G070 logic on MEMECOIN universe (사전 9-point 검증) ===\n")
    sym_status = {}
    for sym in MEMES:
        df = load(sym)
        sym_status[sym] = len(df) if df is not None else 0
    print("Universe data:", {k: v for k, v in sym_status.items() if v > 0})

    all_e = gather(MEMES)
    print(f"\nTotal candidates (thr80, atr<8%, hold24): {len(all_e)}")
    if len(all_e) == 0:
        print("NO CANDIDATES — abort"); return

    yearly = split_by_year(all_e)
    print()
    print(f"{'period':>7} {'days':>5} {'n':>5} {'avg_net':>9} {'WR':>6} {'lottery30':>10}")
    yearly_results = {}
    days_map = {2024: 365, 2025: 365, 2026: 117}  # 2026 jan~apr
    for year, e in yearly.items():
        days = days_map[year]
        if len(e) == 0:
            print(f"{year:>7} {days:>5} {'0':>5}  no entries")
            yearly_results[year] = None
            continue
        avg = e["net_bps"].mean()
        wr = (e["net_bps"] > 0).mean()
        l30 = (e["net_bps"]/10000 * LEV > 0.30).sum()
        print(f"{year:>7} {days:>5} {len(e):>5} {avg:>+9.1f} {wr*100:>5.1f}% {l30:>10}")
        yearly_results[year] = (len(e), avg, wr, l30)

    print("\n=== 9-point Checklist ===\n")
    checks = []

    # Check 1: yearly trade-level avg net > 0
    print("--- Check 1: 연도별 trade-level avg net > 0 ---")
    p1 = True
    for year, e in yearly.items():
        if len(e) == 0: continue
        avg = e["net_bps"].mean()
        ok = "✓" if avg > 0 else "❌"
        print(f"  {year}: {avg:+.0f} {ok}")
        if avg <= 0: p1 = False
    checks.append(p1)
    print(f"  Check 1: {'PASS ✓' if p1 else 'FAIL ❌'}\n")

    # Check 2: portfolio sim yearly annual > 0
    print("--- Check 2: Portfolio sim 연도별 annual > 0 ---")
    p2 = True
    for year, e in yearly.items():
        if len(e) == 0: continue
        r = portfolio_sim(e, days=days_map[year])
        if r is None: continue
        ok = "✓" if r["annual"] > 0 else "❌"
        print(f"  {year}: n={r['n']} annual={r['annual']:+.1f}% liq={r['liquidations']} {ok}")
        if r["annual"] <= 0: p2 = False
    checks.append(p2)
    print(f"  Check 2: {'PASS ✓' if p2 else 'FAIL ❌'}\n")

    # Check 3: avg net ≥ +50 bps
    print("--- Check 3: trade-level avg net ≥ +50 bps ---")
    avg_all = all_e["net_bps"].mean()
    p3 = avg_all >= 50
    print(f"  overall: {avg_all:+.1f} {'✓' if p3 else '❌'}")
    checks.append(p3)
    print()

    # Check 4: WR ≥ 65% per period (low-freq lottery)
    print("--- Check 4: WR ≥ 65% per year ---")
    p4 = True
    for year, e in yearly.items():
        if len(e) == 0: continue
        wr = (e["net_bps"] > 0).mean()
        ok = "✓" if wr >= 0.65 else "❌"
        print(f"  {year}: WR={wr*100:.1f}% {ok}")
        if wr < 0.65: p4 = False
    checks.append(p4)
    print(f"  Check 4: {'PASS ✓' if p4 else 'FAIL ❌'}\n")

    # Check 5: 6축 정합성
    print("--- Check 5: 사용자 6축 ---")
    total_days = sum(days_map.values())
    avg_per_day = len(all_e) / total_days
    axis = {
        "lottery": avg_all/100 >= 3,
        "5-10x lev": True,
        "양방향": False,  # long-only
        "≥3건/일": avg_per_day >= 3,
        "단기 (≤24h)": True,
        "빠른 검증": True,
    }
    fit = sum(axis.values())
    for k, v in axis.items():
        print(f"  {k}: {'✓' if v else '❌'}")
    p5 = fit >= 5
    print(f"  Check 5: {fit}/6 {'PASS ✓' if p5 else 'FAIL ❌'}\n")
    checks.append(p5)

    # Check 6: cost sensitivity
    print("--- Check 6: Cost (16→30bps) ---")
    p6 = True
    for cost in [16, 25, 30]:
        net = (all_e["gross_bps"] - cost).mean()
        print(f"  {cost}bps: {net:+.1f}")
        if net <= 0: p6 = False
    checks.append(p6)
    print(f"  Check 6: {'PASS ✓' if p6 else 'FAIL ❌'}\n")

    # Check 7: leverage 안전성 (intra -20% 빈도 < 10%)
    print("--- Check 7: 5x lev 안전성 (intra-bar -20% < 10%) ---")
    deep = ((all_e["intra_low_bps"]/10000) < -0.20).sum()
    rate = deep / max(len(all_e), 1)
    p7 = rate < 0.10
    print(f"  -20% 거래 {deep}/{len(all_e)} ({rate*100:.1f}%) {'✓' if p7 else '❌'}\n")
    checks.append(p7)

    # Check 8: sample n ≥ 50
    print("--- Check 8: Sample n ≥ 50 ---")
    p8 = len(all_e) >= 50
    print(f"  n={len(all_e)} {'✓' if p8 else '❌'}\n")
    checks.append(p8)

    # Check 9: lengthy warmup X
    print("--- Check 9: 백테스트 only ---  ✓\n")
    checks.append(True)

    # SUMMARY
    passed = sum(checks)
    print(f"{'='*60}")
    print(f"=== G080 (memecoin universe) {passed}/9 PASS ===")
    print(f"{'='*60}")
    if passed == 9:
        print("→ ✅ Production candidate 자격 부여")
    else:
        failed = [i+1 for i, c in enumerate(checks) if not c]
        print(f"→ ❌ DRAFT — 미통과: Check {failed}")


if __name__ == "__main__":
    main()
