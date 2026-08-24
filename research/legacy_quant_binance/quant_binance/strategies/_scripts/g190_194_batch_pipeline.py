"""G190-G194 batch generator + thorough validator with cost stress test.

Candidates:
  G190: size=0.45 + lev=6.0 (combined winner G186+G187, 2-variable exception)
  G191: G187 + max_concurrent 5 -> 8
  G192: G187 + atr_guard 8% -> 6%
  G193: G187 + threshold 80 -> 78
  G194: G187 + holding 24h -> 16h

Validation per candidate:
  1) Standard walk-forward (cost 16 bps)
  2) Cost stress test (cost 24 bps — Bitget realistic upper)
  3) Decision: PASS only if BOTH meet criteria
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
SCRIPTS = ROOT / "quant_binance" / "strategies" / "_scripts"
sys.path.insert(0, str(SCRIPTS))
from g002_mingogogo_ch1_backtest import compute_ch1_score, atr_pct  # type: ignore

DATA_22 = ROOT / "quant_runtime" / "historical_2022"
DATA_24 = ROOT / "quant_runtime" / "historical_2024"
DATA_25 = ROOT / "quant_runtime" / "historical_is25"

UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

EQUITY = 100.0
DEFAULT_ATR_GUARD = 8.0
COST_STANDARD = 16.0
COST_STRESS = 24.0

CANDIDATES = [
    {"id":"G190","folder":"G190_size45_lev6_100usd",       "var":"size+lev (combined)",  "params":{"size":0.45,"lev":6.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}},
    {"id":"G191","folder":"G191_lev6_conc8_100usd",        "var":"max_concurrent 5->8",  "params":{"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}},
    {"id":"G192","folder":"G192_lev6_atr6_100usd",         "var":"atr_guard 8->6%",      "params":{"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":5,"atr":6.0}},
    {"id":"G193","folder":"G193_lev6_thr78_100usd",        "var":"threshold 80->78",     "params":{"size":0.40,"lev":6.0,"thr":78,"hold":24,"max_conc":5,"atr":8.0}},
    {"id":"G194","folder":"G194_lev6_hold16_100usd",       "var":"holding 24->16h",      "params":{"size":0.40,"lev":6.0,"thr":80,"hold":16,"max_conc":5,"atr":8.0}},
]

DECISION_CRITERIA = {
    "min_wr": 0.70,
    "all_periods_positive": True,
    "min_annual_pnl_usd": 300,
    "min_trades_total": 30,
}


def load_period_dfs(data_dir, universe):
    out = {}
    for sym in universe:
        p = Path(data_dir) / sym / "1h.json"
        if not p.exists(): continue
        data = json.loads(p.read_text())
        if isinstance(data, list):
            data.sort(key=lambda b: b["open_time"]); df = pd.DataFrame(data)
        else: df = pd.DataFrame(data)
        for c in ("open_price","high_price","low_price","close_price","base_volume","quote_volume"):
            if c in df.columns: df[c] = df[c].astype(float)
        if len(df) >= 100: out[sym] = df
    return out


def gather_long_entries(dfs, threshold, hold_bars, atr_guard_pct):
    rows = []
    for sym, df in dfs.items():
        score, _ = compute_ch1_score(df)
        df = df.copy()
        df["score"] = score
        df["atr_pct"] = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["fwd_pct"] = (df["close_price"].shift(-hold_bars) / df["close_price"] - 1) * 10000
        mask = (df["score"] >= threshold) & (df["atr_pct"] <= atr_guard_pct) & df["fwd_pct"].notna()
        e = df[mask].copy()
        if len(e) == 0: continue
        e["sym"] = sym
        rows.append(e[["open_time","score","atr_pct","fwd_pct","sym"]])
    return pd.concat(rows).sort_values("open_time").reset_index(drop=True) if rows else pd.DataFrame()


def portfolio_sim(entries, equity, size_pct, leverage, hold_bars, max_conc, days, cost_bps):
    if len(entries) == 0: return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos, pnl_usd, taken, wins = [], 0.0, 0, 0
    big_wins, big_losses, peak_concurrent = 0, 0, 0
    win_pnl, loss_pnl = [], []
    for _, row in entries.iterrows():
        ts = row["open_time"]
        net_bps = row["fwd_pct"] - cost_bps
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == row["sym"] for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        margin = equity * size_pct
        net_pct = net_bps / 10000 * leverage
        if net_pct < -0.90: net_pct = -0.90
        trade_pnl = margin * net_pct
        pnl_usd += trade_pnl; taken += 1
        if net_pct > 0: wins += 1; win_pnl.append(trade_pnl)
        else: loss_pnl.append(trade_pnl)
        if net_pct > 0.30: big_wins += 1
        if net_pct < -0.20: big_losses += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
        peak_concurrent = max(peak_concurrent, len(open_pos))
    return {
        "n_taken": taken,
        "win_rate": round(wins/taken, 4) if taken else 0,
        "big_wins_30pct": big_wins, "big_losses_20pct": big_losses,
        "pnl_usd": round(pnl_usd, 2),
        "annual_pct": round(pnl_usd/equity*100/days*365, 1),
        "annual_pnl_usd": round(pnl_usd/days*365, 2),
        "avg_winner_pnl_usd": round(np.mean(win_pnl), 2) if win_pnl else 0,
        "avg_loser_pnl_usd":  round(np.mean(loss_pnl), 2) if loss_pnl else 0,
        "peak_concurrent": peak_concurrent,
        "peak_notional_x_equity": round(peak_concurrent * size_pct * leverage, 2),
    }


def evaluate(params, cost_bps):
    periods = [
        ("OOS22-23", load_period_dfs(DATA_22, UNIV_22), 730),
        ("OOS24-Q1", load_period_dfs(DATA_24, UNIV_24), 456),
        ("IS25-26",  load_period_dfs(DATA_25, UNIV_25), 374),
    ]
    res = {}
    for plabel, dfs, days in periods:
        ent = gather_long_entries(dfs, params["thr"], params["hold"], params["atr"])
        r = portfolio_sim(ent, EQUITY, params["size"], params["lev"], params["hold"], params["max_conc"], days, cost_bps)
        res[plabel] = r if r else {"days": days, "n_taken": 0}
        if r: res[plabel]["days"] = days
    valid = [r for r in res.values() if r.get("n_taken", 0) > 0]
    total_days = sum(r.get("days", 0) for r in valid)
    total_taken = sum(r.get("n_taken", 0) for r in valid)
    total_pnl = sum(r.get("pnl_usd", 0) for r in valid)
    weighted_wr = sum(r["win_rate"] * r["n_taken"] for r in valid) / max(total_taken, 1)
    annual_pnl = total_pnl / total_days * 365 if total_days else 0
    all_pos = all(r.get("pnl_usd", 0) > 0 for r in valid) and len(valid) == 3
    peak_notional = max((r.get("peak_notional_x_equity", 0) for r in valid), default=0)
    return {
        "periods": res,
        "weighted": {
            "n_taken": total_taken,
            "win_rate": round(weighted_wr, 4),
            "pnl_usd": round(total_pnl, 2),
            "annual_pnl_usd": round(annual_pnl, 2),
            "monthly_pnl_usd": round(annual_pnl/12, 2),
            "all_periods_positive": all_pos,
            "peak_notional_x_equity": peak_notional,
        }
    }


def decide(weighted, label):
    c = DECISION_CRITERIA
    checks = {
        "wr_>=_70%":          weighted["win_rate"] >= c["min_wr"],
        "all_periods_pos":    weighted["all_periods_positive"],
        "annual_pnl_>=_$300": weighted["annual_pnl_usd"] >= c["min_annual_pnl_usd"],
        "trades_>=_30":       weighted["n_taken"] >= c["min_trades_total"],
    }
    return {"label": label, "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def main():
    print("=" * 130)
    print(f"{'ID':<6} {'change':<24} {'cost':>5} {'trades':>6} {'WR':>6} {'PnL$':>9} {'annPnL$':>10} {'monthly$':>10} {'allPos':>7} {'peakLev':>8} {'VERDICT':>9}")
    print("-" * 130)
    summary = {}
    for cand in CANDIDATES:
        results = {}
        for cost_label, cost in [("std16", COST_STANDARD), ("stress24", COST_STRESS)]:
            ev = evaluate(cand["params"], cost)
            d = decide(ev["weighted"], cost_label)
            w = ev["weighted"]
            print(f"{cand['id']:<6} {cand['var']:<24} {cost:>5.0f} {w['n_taken']:>6} {w['win_rate']*100:>5.1f}% {w['pnl_usd']:>+8.2f} {w['annual_pnl_usd']:>+9.2f} {w['monthly_pnl_usd']:>+9.2f} {('Y' if w['all_periods_positive'] else 'N'):>7} {w['peak_notional_x_equity']:>7.1f}x {d['verdict']:>9}")
            results[cost_label] = {"evaluation": ev, "decision": d}
        # PASS only if BOTH std16 + stress24 pass (robustness)
        robust = results["std16"]["decision"]["verdict"] == "PASS" and results["stress24"]["decision"]["verdict"] == "PASS"
        results["robust_PASS"] = robust
        summary[cand["id"]] = {**cand, "results": results}
        print()

    # G185-G187 cost stress reference (from prior validation)
    print("\n--- prior winners reference (cost 16 bps standard) ---")
    print(f"{'G185':<6} {'baseline':<24} {16:>5.0f} {66:>6} {84.9:>5.1f}% {1464.50:>+8.2f} {342.66:>+9.2f} {28.55:>+9.2f} {'Y':>7} {10.0:>7.1f}x {'PASS':>9}")
    print(f"{'G186':<6} {'size 0.45':<24} {16:>5.0f} {66:>6} {84.9:>5.1f}% {1647.57:>+8.2f} {385.49:>+9.2f} {32.12:>+9.2f} {'Y':>7} {11.25:>7.1f}x {'PASS':>9}")
    print(f"{'G187':<6} {'lev 6x':<24} {16:>5.0f} {66:>6} {84.9:>5.1f}% {1757.41:>+8.2f} {411.19:>+9.2f} {34.27:>+9.2f} {'Y':>7} {12.0:>7.1f}x {'PASS':>9}")

    # Robust passers
    robust_passers = [(sid, s) for sid, s in summary.items() if s["results"]["robust_PASS"]]
    print()
    print("=" * 60)
    print(f"ROBUST PASS (std16 + stress24 모두 PASS): {len(robust_passers)} / {len(CANDIDATES)}")
    print("=" * 60)
    if robust_passers:
        robust_passers.sort(key=lambda x: x[1]["results"]["std16"]["evaluation"]["weighted"]["annual_pnl_usd"], reverse=True)
        for sid, s in robust_passers:
            w = s["results"]["std16"]["evaluation"]["weighted"]
            ws = s["results"]["stress24"]["evaluation"]["weighted"]
            print(f"  🏆 {sid} ({s['var']}): std=${w['annual_pnl_usd']:.0f}/yr WR{w['win_rate']*100:.1f}% / stress=${ws['annual_pnl_usd']:.0f}/yr (WR{ws['win_rate']*100:.1f}%) peak {w['peak_notional_x_equity']:.0f}x")
    else:
        print("  (none robust)")

    # Save
    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g190_194_batch_summary.json"
    out.write_text(json.dumps({
        "criteria": DECISION_CRITERIA,
        "cost_bps": {"standard": COST_STANDARD, "stress": COST_STRESS},
        "candidates": summary,
        "robust_passers": [sid for sid, _ in robust_passers],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
