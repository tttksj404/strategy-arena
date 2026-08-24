"""
G055 (dynamic size) + G057 (vol overlay) + G058 (dynamic concurrency) — variable-1 each.

기준: G050 max5/30% baseline (3-period validated +74%/년)

G055: G050 + dynamic size (Kelly-style, recent net 양 비례)
G057: G050 + G040 vol-burst overlay (ATR ratio ≥2.5 시 추가 size up)
G058: G050 + dynamic concurrency (recent positive streak 길수록 max_conc 증가)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g050_v2_hypothetical import gather
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct

DATA_22 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022"
DATA_24 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2024"
DATA_25 = Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical"
UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

EQUITY = 55.0
HOLD_MS = 72 * 3600 * 1000


def gate14d(history, ts):
    LB = 14 * 86400 * 1000
    if not history: return True
    first = history[0][0]
    if ts - first < LB: return True
    recent = [n for t, n in history if ts - LB <= t < ts]
    if not recent: return True
    return sum(recent) > 0


def gather_with_atr(data_dir, universe):
    """G050 candidates + ATR ratio (vol burst flag)."""
    entries = []
    for sym in universe:
        p = Path(data_dir) / sym / "1h.json"
        if not p.exists(): continue
        df = pd.DataFrame(json.loads(p.read_text()))
        if len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        a_ma = a.rolling(20).mean()
        df["score"] = score
        df["atr_ratio"] = a / a_ma
        df["vol_burst"] = (df["atr_ratio"] >= 2.5) & (df["close_price"] > df["open_price"])
        df["fwd_pct"] = (df["close_price"].shift(-72) / df["close_price"] - 1) * 10000
        e = df[(df["score"] >= 70) & df["fwd_pct"].notna()].copy()
        e["sym"] = sym
        e["gross_bps"] = e["fwd_pct"]
        e["net_bps"] = e["fwd_pct"] - 16
        entries.append(e[["open_time","score","gross_bps","net_bps","atr_ratio","vol_burst","sym"]])
    if not entries: return pd.DataFrame()
    return pd.concat(entries).sort_values("open_time").reset_index(drop=True)


# ─── 4 strategies ─────────────────────────────────────────────────

def sim_g050_baseline(all_e, max_conc=5, base_size_pct=0.30):
    """Baseline G050: gate14d, fixed size."""
    open_pos = []; history = []; pnl = 0.0; taken = 0; big_wins = 0
    for _, row in all_e.iterrows():
        ts = row["open_time"]; net = row["net_bps"]
        open_pos = [p for p in open_pos if p[0] > ts]
        active = gate14d(history, ts)
        history.append((ts, net))
        if not active or any(p[1] == row["sym"] for p in open_pos) or len(open_pos) >= max_conc: continue
        size = EQUITY * base_size_pct
        pnl += size * net / 10000
        taken += 1
        if net > 1000: big_wins += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"taken":taken, "pnl_usd":round(pnl,2), "pnl_pct":round(pnl/EQUITY*100,1), "big_wins":big_wins}


def sim_g055_dynamic_size(all_e, max_conc=5):
    """G055: 직전 14일 누적 net 양수 강도에 비례하여 size 조정.
    base 30%, recent strong (>+5000bps) → 50%, weak (<+1000) → 20%."""
    LB = 14 * 86400 * 1000
    open_pos = []; history = []; pnl = 0.0; taken = 0; big_wins = 0
    for _, row in all_e.iterrows():
        ts = row["open_time"]; net = row["net_bps"]
        open_pos = [p for p in open_pos if p[0] > ts]
        # gate
        if not history: active = True
        else:
            first = history[0][0]
            if ts - first < LB: active = True
            else:
                recent = [n for t, n in history if ts - LB <= t < ts]
                active = sum(recent) > 0 if recent else True
        # dynamic size
        if not history or ts - history[0][0] < LB:
            size_pct = 0.30
        else:
            recent_sum = sum(n for t, n in history if ts - LB <= t < ts)
            if recent_sum > 5000: size_pct = 0.50
            elif recent_sum > 2000: size_pct = 0.40
            elif recent_sum > 0: size_pct = 0.30
            else: size_pct = 0.20  # gate 도 false 일 거지만 safety
        history.append((ts, net))
        if not active or any(p[1] == row["sym"] for p in open_pos) or len(open_pos) >= max_conc: continue
        size = EQUITY * size_pct
        pnl += size * net / 10000
        taken += 1
        if net > 1000: big_wins += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"taken":taken, "pnl_usd":round(pnl,2), "pnl_pct":round(pnl/EQUITY*100,1), "big_wins":big_wins}


def sim_g057_vol_overlay(all_e, max_conc=5):
    """G057: G050 + vol_burst 시 size 50% (보통 30%). burst 안 떠도 30% 정상 운용."""
    open_pos = []; history = []; pnl = 0.0; taken = 0; big_wins = 0; vol_taken = 0
    for _, row in all_e.iterrows():
        ts = row["open_time"]; net = row["net_bps"]
        open_pos = [p for p in open_pos if p[0] > ts]
        active = gate14d(history, ts)
        history.append((ts, net))
        if not active or any(p[1] == row["sym"] for p in open_pos) or len(open_pos) >= max_conc: continue
        if row["vol_burst"]:
            size_pct = 0.50; vol_taken += 1
        else:
            size_pct = 0.30
        size = EQUITY * size_pct
        pnl += size * net / 10000
        taken += 1
        if net > 1000: big_wins += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"taken":taken, "pnl_usd":round(pnl,2), "pnl_pct":round(pnl/EQUITY*100,1), "big_wins":big_wins, "vol_burst_taken":vol_taken}


def sim_g058_dynamic_conc(all_e, base_conc=5, base_size=0.30):
    """G058: 직전 7일 net 양수면 max_conc=8, 음수면 3."""
    LB = 14 * 86400 * 1000
    SHORT_LB = 7 * 86400 * 1000
    open_pos = []; history = []; pnl = 0.0; taken = 0; big_wins = 0
    for _, row in all_e.iterrows():
        ts = row["open_time"]; net = row["net_bps"]
        open_pos = [p for p in open_pos if p[0] > ts]
        # gate14d
        if not history: active = True
        else:
            first = history[0][0]
            if ts - first < LB: active = True
            else:
                recent = [n for t, n in history if ts - LB <= t < ts]
                active = sum(recent) > 0 if recent else True
        # dynamic conc
        if not history or ts - history[0][0] < SHORT_LB:
            conc = base_conc
        else:
            short_recent = sum(n for t, n in history if ts - SHORT_LB <= t < ts)
            if short_recent > 2000: conc = 8
            elif short_recent > 0: conc = base_conc
            else: conc = 3
        history.append((ts, net))
        if not active or any(p[1] == row["sym"] for p in open_pos) or len(open_pos) >= conc: continue
        size = EQUITY * base_size
        pnl += size * net / 10000
        taken += 1
        if net > 1000: big_wins += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {"taken":taken, "pnl_usd":round(pnl,2), "pnl_pct":round(pnl/EQUITY*100,1), "big_wins":big_wins}


def main():
    print("=== G055/G057/G058 — G050 변형 3종 ($55, max5 base) ===\n")

    e_22 = gather_with_atr(DATA_22, UNIV_22)
    e_24 = gather_with_atr(DATA_24, UNIV_24)
    e_25 = gather_with_atr(DATA_25, UNIV_25)
    print(f"Candidates: 22-23={len(e_22)} / 24-Q1.25={len(e_24)} / 25-26={len(e_25)}\n")

    print(f"{'strategy':<28} {'period':>10} {'taken':>5} {'PnL$':>8} {'PnL%':>7} {'annual':>9} {'big':>4}")
    print("-" * 80)
    periods = [("OOS22-23", e_22, 730), ("OOS24-Q1.25", e_24, 456), ("IS25-26", e_25, 374)]
    strategies = [
        ("G050 baseline (max5/30%)", sim_g050_baseline, {}),
        ("G055 dynamic size (Kelly)", sim_g055_dynamic_size, {}),
        ("G057 vol overlay (burst→50%)", sim_g057_vol_overlay, {}),
        ("G058 dynamic conc (3~8)", sim_g058_dynamic_conc, {}),
    ]
    summary = {s[0]: [] for s in strategies}
    for label_p, e, days in periods:
        for sname, fn, kwargs in strategies:
            r = fn(e, **kwargs)
            annual = round(r["pnl_pct"] / days * 365, 1)
            extra = f" vol={r.get('vol_burst_taken','')}" if 'vol_burst_taken' in r else ""
            print(f"{sname:<28} {label_p:>10} {r['taken']:>5} ${r['pnl_usd']:>+7.2f} {r['pnl_pct']:>+6.1f}% {annual:>+8.1f}% {r['big_wins']:>4}{extra}")
            summary[sname].append((label_p, r["pnl_pct"], annual, days))
        print()

    # 가중 평균 (days)
    print("=== Summary: weighted avg annual (1560 days OOS+IS) ===")
    print(f"{'strategy':<28} {'OOS22':>8} {'OOS24':>8} {'IS25':>8} {'wavg':>9}")
    for sname, results in summary.items():
        per = {p[0]:p[1:] for p in results}
        avg_pct = sum(p[1] * p[2] for p in [(per["OOS22-23"], 730), (per["OOS24-Q1.25"], 456), (per["IS25-26"], 374)]) / 1560
        print(f"{sname:<28} {per['OOS22-23'][1]:>+7.1f}% {per['OOS24-Q1.25'][1]:>+7.1f}% {per['IS25-26'][1]:>+7.1f}% {avg_pct:>+8.1f}%")


if __name__ == "__main__":
    main()
