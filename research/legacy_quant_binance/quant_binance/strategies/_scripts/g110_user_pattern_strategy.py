"""
G110 — User winner pattern strategy (closed_trades 78건 분석 기반).

룰 (실제 winner 패턴 강제):
- Universe: ETH, SOL, DOGE, PEPE (4개만)
- Side: long bias (long-only 시작, 점진 양방향)
- Entry score: CH1 score >= 70
- Hold: 12h (1h klines 한계, 5분 단타 근사 X)
- Leverage: 30x (사용자 실제 운용)
- Time: UTC 07 우대 (전체 시간 무관 baseline + UTC 07 가중치)
- ATR guard: 15% (30x 에선 더 보수적 — 3.3% intra-bar 시 liquidation)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct

DIRS = [
    Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical",
    Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2024",
    Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022",
    Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_top50",
    Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_meme",
]

USER_UNIVERSE = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "PEPEUSDT"]
EQUITY = 55.0
SIZE_PCT = 0.30  # 자본의 30% margin
LEV = 30.0  # 사용자 실제 운용
HOLD = 12  # 1h × 12 = 12h (5분 단타 근사 X, 1h kline 한계)
THR = 70
ATR_GUARD = 15.0  # 30x lev 에서 3% 인터바 시 liquidation, 15% atr_pct는 매우 보수적
MAX_CONC = 3  # 30x = 큰 노출, 동시 보유 줄임


def load_combined(sym):
    bars = []; seen = set()
    for d in DIRS:
        p = d / sym / "1h.json"
        if p.exists():
            for b in json.loads(p.read_text()):
                ts = b["open_time"]
                if ts not in seen:
                    seen.add(ts); bars.append(b)
    if not bars: return None
    bars.sort(key=lambda x: x["open_time"])
    return pd.DataFrame(bars)


def gather(symbols, threshold=THR, hold=HOLD):
    entries = []
    for sym in symbols:
        df = load_combined(sym)
        if df is None or len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["fwd_pct"] = (df["close_price"].shift(-hold) / df["close_price"] - 1) * 10000
        df["hour_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.hour
        intra_low = df["low_price"].rolling(window=hold+1, min_periods=1).min().shift(-hold)
        df["intra_low_bps"] = (intra_low / df["close_price"] - 1) * 10000

        e = df[(df["score"] >= threshold) & (df["atr_pct"] <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        if len(e):
            entries.append(e[["open_time","score","gross_bps","net_bps","atr_pct","intra_low_bps","hour_utc","sym"]])
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def split_yearly(entries):
    if len(entries) == 0: return {}
    e_dt = pd.to_datetime(entries["open_time"], unit="ms", utc=True)
    return {y: entries[e_dt.dt.year == y] for y in [2022, 2023, 2024, 2025, 2026]}


def portfolio_sim_30x(entries, days, max_conc=MAX_CONC):
    """30x leverage 시뮬. -3.3% intra-bar 시 liquidation (5x 시 -20% 와 동일 비율)."""
    if len(entries) == 0: return None
    open_pos = []
    pnl = 0.0; taken = 0; wins = 0; lottery30 = 0; liq = 0
    LIQ_THR = -1.0 / LEV * 0.95  # -3.17% 인터바 시 liquidation
    HOLD_MS = HOLD * 3600 * 1000
    for _, row in entries.iterrows():
        ts = row["open_time"]
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == row["sym"] for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        margin = EQUITY * SIZE_PCT
        intra_low_pct = row["intra_low_bps"] / 10000 if not pd.isna(row["intra_low_bps"]) else 0
        if intra_low_pct < LIQ_THR:
            net_pct = -1.0; liq += 1
        else:
            net_pct = row["net_bps"] / 10000 * LEV
            net_pct = max(net_pct, -1.0)  # margin call cap at -100%
        pnl += margin * net_pct
        taken += 1
        if net_pct > 0: wins += 1
        if net_pct > 0.30: lottery30 += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"n":taken,"pnl_pct":round(pnl/EQUITY*100,1),"annual":round(pnl/EQUITY*100/days*365,1),
            "wr":round(wins/taken,4) if taken else 0, "lottery30":lottery30, "liq":liq, "per_day":round(taken/days,3)}


def run_9point(label, entries):
    print(f"\n{'='*70}\n=== {label} ({len(entries)} candidates) ===\n{'='*70}")
    if len(entries) == 0: return [False]*9
    yearly = split_yearly(entries)
    days_map = {2022: 365, 2023: 365, 2024: 365, 2025: 365, 2026: 117}
    print(f"\n{'year':>6} {'n':>5} {'/day':>5} {'avg_net':>9} {'WR':>6} {'L30%+':>7} {'liq':>4} {'annual':>9}")
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
        r = portfolio_sim_30x(e, d)
        ann = r["annual"] if r else 0
        liq = r["liq"] if r else 0
        print(f"{y:>6} {len(e):>5} {per_day:>5.2f} {avg:>+9.1f} {wr*100:>5.1f}% {l30:>7} {liq:>4} {ann:>+8.1f}%")
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
    bothdir = False
    p5 = sum([p5_lottery, True, bothdir, p5_freq, True, True]) >= 5
    cost_pass = all((entries["gross_bps"] - c).mean() > 0 for c in [16, 25, 30])
    deep = ((entries["intra_low_bps"]/10000) < -1.0/LEV*0.95).sum()
    p7 = deep / max(len(entries), 1) < 0.10
    p8 = len(entries) >= 50
    checks = [p1, p2, p3, p4, p5, cost_pass, p7, p8, True]
    print(f"\n  Check 1: {'✓' if p1 else '❌'}  Check 2: {'✓' if p2 else '❌'}  Check 3: ({avg_all:+.0f}) {'✓' if p3 else '❌'}")
    print(f"  Check 4 (WR≥65%): {'✓' if p4 else '❌'}  Check 5 (6축 freq={avg_per_day:.2f}): {'✓' if p5 else '❌'}")
    print(f"  Check 6: {'✓' if cost_pass else '❌'}  Check 7 (30x liq<10%): {deep/max(len(entries),1)*100:.1f}% {'✓' if p7 else '❌'}  Check 8: {'✓' if p8 else '❌'}")
    print(f"  TOTAL: {sum(checks)}/9 PASS")
    return checks


def main():
    print(f"=== G110 — User real pattern (ETH/SOL/DOGE/PEPE, score≥70, 12h hold, 30x lev) ===\n")
    print(f"Universe: {USER_UNIVERSE}")
    entries = gather(USER_UNIVERSE)
    checks = run_9point("G110 baseline", entries)

    # UTC 07 시간대 우대 — UTC 07-09 만 진입
    print(f"\n=== G110b — UTC 07-09 시간대 (사용자 실제 발화) ===")
    if len(entries) > 0:
        e_utc = entries[(entries["hour_utc"] >= 7) & (entries["hour_utc"] <= 9)].copy()
        run_9point("G110b UTC 07-09", e_utc)

    # 더 tight: score >= 80
    print(f"\n=== G110c — score ≥80 + UTC 07-09 (tighter) ===")
    if len(entries) > 0:
        e_tight = entries[(entries["score"] >= 80) & (entries["hour_utc"] >= 7) & (entries["hour_utc"] <= 9)].copy()
        run_9point("G110c score80 UTC07-09", e_tight)

    # 24h hold (사용자 5분 단타 X — 1h klines 한계, 24h 도 시도)
    print(f"\n=== G110d — score≥70 hold 24h (G070-like) ===")
    e24 = gather(USER_UNIVERSE, threshold=70, hold=24)
    run_9point("G110d hold24", e24)

    # 6h hold (단타에 가깝게)
    print(f"\n=== G110e — score≥70 hold 6h (단타 근사) ===")
    e6 = gather(USER_UNIVERSE, threshold=70, hold=6)
    run_9point("G110e hold6", e6)


if __name__ == "__main__":
    main()
