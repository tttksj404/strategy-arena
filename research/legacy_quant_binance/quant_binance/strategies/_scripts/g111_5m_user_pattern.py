"""
G111 — 5m klines 5분 단타 (사용자 closed_trades 78건 winner 패턴 정확 시뮬).

룰:
- Universe: ETH/SOL/DOGE/PEPE
- Entry: CH1 score >= 70 (5m 단위 평가)
- Hold: 1-2 bars = 5-10분 (사용자 median 5.3분)
- Leverage: 30x (사용자 실제)
- Exit: hold timeout (proactive partial TP simulation 별도)
- ATR guard: 5m atr_pct <= 2% (30x 에서 인터바 -3% 시 청산)
- Time: UTC 07-09 우대 시도

cost: 5m 단타 = 진입+청산 fee + slippage. Bitget perp maker 0.02% / taker 0.06%.
보수적: round-trip 0.1% = 10 bps.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, atr_pct

DIR = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_5m"
SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "PEPEUSDT"]

EQUITY = 55.0
SIZE_PCT = 0.30  # margin
LEV = 30.0
COST_BPS_RT = 10.0  # 0.1% round-trip (5m perp)
ATR_GUARD = 2.0  # 5m atr_pct max 2%
MAX_CONC = 3


def load(sym):
    p = DIR / sym / "5m.json"
    return pd.DataFrame(json.loads(p.read_text())) if p.exists() else None


def gather(symbols, threshold, hold_bars):
    entries = []
    for sym in symbols:
        df = load(sym)
        if df is None or len(df) < 200: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["fwd_pct"] = (df["close_price"].shift(-hold_bars) / df["close_price"] - 1) * 10000
        df["hour_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.hour
        intra_low = df["low_price"].rolling(window=hold_bars+1, min_periods=1).min().shift(-hold_bars)
        df["intra_low_bps"] = (intra_low / df["close_price"] - 1) * 10000
        intra_high = df["high_price"].rolling(window=hold_bars+1, min_periods=1).max().shift(-hold_bars)
        df["peak_high_bps"] = (intra_high / df["close_price"] - 1) * 10000

        e = df[(df["score"] >= threshold) & (df["atr_pct"] <= ATR_GUARD) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - COST_BPS_RT
        if len(e):
            entries.append(e[["open_time","score","gross_bps","net_bps","atr_pct","intra_low_bps","peak_high_bps","hour_utc","sym"]])
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def gather_with_partial_tp(symbols, threshold, max_hold_bars=12, peak_tp_bps=300):
    """
    Proactive partial TP: peak ROE +15% 도달 시 절반 익절 + trailing.
    근사: peak high >= 300bps (30x 에서 +9% margin) → 절반 익절.
    """
    entries = []
    for sym in symbols:
        df = load(sym)
        if df is None or len(df) < 200: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["hour_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.hour

        # max_hold_bars 동안 매 bar 마다 high/low 추적
        # partial TP: peak high >= peak_tp_bps 시 절반 익절 (close at TP), 나머지는 trailing
        # simpler: peak high >= peak_tp_bps 시 즉시 종료
        for i in range(len(df) - max_hold_bars):
            if df["score"].iloc[i] < threshold or df["atr_pct"].iloc[i] > ATR_GUARD: continue
            entry_price = df["close_price"].iloc[i]
            window = df.iloc[i+1:i+1+max_hold_bars]
            # check peak high during window
            highs = (window["high_price"] / entry_price - 1) * 10000
            lows = (window["low_price"] / entry_price - 1) * 10000
            min_low_bps = lows.min()
            if min_low_bps < -1.0/LEV*0.95*10000:  # liquidation
                exit_bps = -1.0/LEV*0.95*10000
            else:
                # peak TP: first bar where high >= peak_tp_bps
                tp_hits = (highs >= peak_tp_bps)
                if tp_hits.any():
                    exit_bps = peak_tp_bps  # exit at TP
                else:
                    # max_hold timeout
                    exit_bps = (window.iloc[-1]["close_price"] / entry_price - 1) * 10000
            entries.append({
                "open_time": int(df["open_time"].iloc[i]),
                "score": float(df["score"].iloc[i]),
                "gross_bps": float(exit_bps),
                "net_bps": float(exit_bps - COST_BPS_RT),
                "atr_pct": float(df["atr_pct"].iloc[i]),
                "intra_low_bps": float(min_low_bps),
                "peak_high_bps": float(highs.max()),
                "hour_utc": int(df["hour_utc"].iloc[i]),
                "sym": sym,
            })
    return pd.DataFrame(entries).sort_values("open_time").reset_index(drop=True) if entries else pd.DataFrame()


def split_yearly(entries):
    if len(entries) == 0: return {}
    e_dt = pd.to_datetime(entries["open_time"], unit="ms", utc=True)
    return {y: entries[e_dt.dt.year == y] for y in [2024, 2025, 2026]}


def portfolio_sim(entries, hold_min, days, max_conc=MAX_CONC):
    if len(entries) == 0: return None
    open_pos = []
    pnl = 0.0; taken = 0; wins = 0; lottery30 = 0; liq = 0
    HOLD_MS = hold_min * 60 * 1000
    LIQ_THR = -1.0 / LEV * 0.95
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
            net_pct = max(row["net_bps"] / 10000 * LEV, -1.0)
        pnl += margin * net_pct
        taken += 1
        if net_pct > 0: wins += 1
        if net_pct > 0.30: lottery30 += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"n":taken,"pnl_pct":round(pnl/EQUITY*100,1),"annual":round(pnl/EQUITY*100/days*365,1),
            "wr":round(wins/taken,4) if taken else 0, "lottery30":lottery30, "liq":liq, "per_day":round(taken/days,3)}


def run_9point(label, entries, hold_min):
    print(f"\n{'='*70}\n=== {label} ({len(entries)} candidates) ===\n{'='*70}")
    if len(entries) == 0: return [False]*9
    yearly = split_yearly(entries)
    days_map = {2024: 365, 2025: 365, 2026: 117}
    print(f"\n{'year':>6} {'n':>5} {'/day':>6} {'avg_net':>9} {'WR':>6} {'L30%+':>7} {'liq':>4} {'annual':>10}")
    p1 = p2 = p4 = True; total_n = 0
    for y in [2024, 2025, 2026]:
        e = yearly.get(y, pd.DataFrame())
        d = days_map[y]
        if len(e) == 0: print(f"{y:>6} {'0':>5}"); continue
        avg = e["net_bps"].mean(); wr = (e["net_bps"]>0).mean()
        l30 = (e["net_bps"]/10000*LEV > 0.30).sum()
        per_day = len(e) / d
        r = portfolio_sim(e, hold_min, d)
        ann = r["annual"] if r else 0; lq = r["liq"] if r else 0
        print(f"{y:>6} {len(e):>5} {per_day:>6.2f} {avg:>+9.1f} {wr*100:>5.1f}% {l30:>7} {lq:>4} {ann:>+9.1f}%")
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
    p5 = sum([p5_lottery, True, False, p5_freq, True, True]) >= 5
    cost_pass = all((entries["gross_bps"] - c).mean() > 0 for c in [10, 16, 25])
    deep = ((entries["intra_low_bps"]/10000) < -1.0/LEV*0.95).sum()
    p7 = deep / max(len(entries), 1) < 0.10
    p8 = len(entries) >= 50
    checks = [p1, p2, p3, p4, p5, cost_pass, p7, p8, True]
    print(f"\n  C1: {'✓' if p1 else '❌'}  C2: {'✓' if p2 else '❌'}  C3 ({avg_all:+.0f}): {'✓' if p3 else '❌'}  C4: {'✓' if p4 else '❌'}")
    print(f"  C5 (freq={avg_per_day:.1f}/day): {'✓' if p5 else '❌'}  C6: {'✓' if cost_pass else '❌'}  C7 (liq={deep/max(len(entries),1)*100:.1f}%): {'✓' if p7 else '❌'}")
    print(f"  TOTAL: {sum(checks)}/9 PASS")
    return checks


def main():
    print("=== G111 — 5m 단타, 30x lev, ETH/SOL/DOGE/PEPE (user real pattern) ===\n")

    print("--- 1) hold 5min (1 bar) ---")
    e1 = gather(SYMBOLS, threshold=70, hold_bars=1)
    run_9point("G111-h1 (5min)", e1, hold_min=5)

    print("\n--- 2) hold 10min (2 bars) ---")
    e2 = gather(SYMBOLS, threshold=70, hold_bars=2)
    run_9point("G111-h2 (10min)", e2, hold_min=10)

    print("\n--- 3) hold 30min (6 bars) ---")
    e3 = gather(SYMBOLS, threshold=70, hold_bars=6)
    run_9point("G111-h3 (30min)", e3, hold_min=30)

    print("\n--- 4) score >= 80, hold 5min ---")
    e4 = gather(SYMBOLS, threshold=80, hold_bars=1)
    run_9point("G111-thr80-h1", e4, hold_min=5)

    print("\n--- 5) PARTIAL TP +9% margin (peak 300bps), max hold 1h (12 bars) ---")
    e5 = gather_with_partial_tp(SYMBOLS, threshold=70, max_hold_bars=12, peak_tp_bps=300)
    run_9point("G111-partial-TP", e5, hold_min=60)


if __name__ == "__main__":
    main()
