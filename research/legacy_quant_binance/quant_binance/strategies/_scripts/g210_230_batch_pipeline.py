"""G210/G220/G230 batch — universe filter on G191 (TOP).

G210 = G191 + drop dead weight (WIF, LTC, BTC)        → 15 syms
G220 = G191 + universe TOP 10 only                    → 10 syms
G230 = G191 + universe TOP 5 only                     → 5 syms

Compare against G191 baseline. Cost stress test included.
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

# Base universes per period (before filter)
UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

DEAD_WEIGHT = {"WIFUSDT", "LTCUSDT", "BTCUSDT"}
TOP_10 = {"DOGEUSDT","PEPEUSDT","SOLUSDT","ARBUSDT","ADAUSDT","LINKUSDT","DOTUSDT","NEARUSDT","AVAXUSDT","UNIUSDT"}
TOP_5  = {"DOGEUSDT","PEPEUSDT","SOLUSDT","ARBUSDT","ADAUSDT"}


def filter_universe(base, mode):
    if mode == "drop_dead":
        return [s for s in base if s not in DEAD_WEIGHT]
    if mode == "top10":
        return [s for s in base if s in TOP_10]
    if mode == "top5":
        return [s for s in base if s in TOP_5]
    return base  # baseline (full)


CANDIDATES = [
    {"id":"G191_ref", "filter":"baseline",  "params":{"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "deploy":False},
    {"id":"G210",     "filter":"drop_dead", "params":{"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "deploy":True, "folder":"G210_g191_drop_dead"},
    {"id":"G220",     "filter":"top10",     "params":{"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "deploy":True, "folder":"G220_g191_top10"},
    {"id":"G230",     "filter":"top5",      "params":{"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "deploy":True, "folder":"G230_g191_top5"},
]

EQUITY = 100.0
COST_STANDARD = 16.0
COST_STRESS = 24.0

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
    big_wins, big_losses = 0, 0
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
    return {
        "n_taken": taken,
        "win_rate": round(wins/taken, 4) if taken else 0,
        "big_wins_30pct": big_wins, "big_losses_20pct": big_losses,
        "pnl_usd": round(pnl_usd, 2),
        "annual_pct": round(pnl_usd/equity*100/days*365, 1),
        "annual_pnl_usd": round(pnl_usd/days*365, 2),
        "avg_winner_pnl_usd": round(np.mean(win_pnl), 2) if win_pnl else 0,
        "avg_loser_pnl_usd":  round(np.mean(loss_pnl), 2) if loss_pnl else 0,
    }


def evaluate(params, filter_mode, cost_bps):
    periods = [
        ("OOS22-23", filter_universe(UNIV_22, filter_mode), DATA_22, 730),
        ("OOS24-Q1", filter_universe(UNIV_24, filter_mode), DATA_24, 456),
        ("IS25-26",  filter_universe(UNIV_25, filter_mode), DATA_25, 374),
    ]
    res = {}
    for plabel, universe, dpath, days in periods:
        dfs = load_period_dfs(dpath, universe)
        ent = gather_long_entries(dfs, params["thr"], params["hold"], params["atr"])
        r = portfolio_sim(ent, EQUITY, params["size"], params["lev"], params["hold"], params["max_conc"], days, cost_bps)
        if r is None:
            res[plabel] = {"days": days, "n_sym": len(universe), "n_taken": 0}
        else:
            r["days"] = days; r["n_sym"] = len(universe); res[plabel] = r
    valid = [r for r in res.values() if r.get("n_taken", 0) > 0]
    total_days = sum(r.get("days", 0) for r in valid)
    total_taken = sum(r.get("n_taken", 0) for r in valid)
    total_pnl = sum(r.get("pnl_usd", 0) for r in valid)
    weighted_wr = sum(r["win_rate"] * r["n_taken"] for r in valid) / max(total_taken, 1)
    annual_pnl = total_pnl / total_days * 365 if total_days else 0
    all_pos = all(r.get("pnl_usd", 0) > 0 for r in valid) and len(valid) == 3
    return {
        "periods": res,
        "weighted": {
            "n_taken": total_taken,
            "win_rate": round(weighted_wr, 4),
            "pnl_usd": round(total_pnl, 2),
            "annual_pnl_usd": round(annual_pnl, 2),
            "monthly_pnl_usd": round(annual_pnl/12, 2),
            "all_periods_positive": all_pos,
        }
    }


def decide(weighted):
    c = DECISION_CRITERIA
    checks = {
        "wr_>=_70%":          weighted["win_rate"] >= c["min_wr"],
        "all_periods_pos":    weighted["all_periods_positive"],
        "annual_pnl_>=_$300": weighted["annual_pnl_usd"] >= c["min_annual_pnl_usd"],
        "trades_>=_30":       weighted["n_taken"] >= c["min_trades_total"],
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def main():
    print("=" * 130)
    print(f"{'ID':<10} {'filter':<14} {'cost':>5} {'sym22/24/25':<13} {'trades':>6} {'WR':>6} {'PnL$':>9} {'annPnL$':>10} {'monthly$':>10} {'allPos':>7} {'VERDICT':>9}")
    print("-" * 130)
    summary = {}
    for cand in CANDIDATES:
        results = {}
        for cost_label, cost in [("std16", COST_STANDARD), ("stress24", COST_STRESS)]:
            ev = evaluate(cand["params"], cand["filter"], cost)
            d = decide(ev["weighted"])
            w = ev["weighted"]
            sym_str = "/".join(str(ev["periods"][p].get("n_sym", 0)) for p in ["OOS22-23","OOS24-Q1","IS25-26"])
            print(f"{cand['id']:<10} {cand['filter']:<14} {cost:>5.0f} {sym_str:<13} {w['n_taken']:>6} {w['win_rate']*100:>5.1f}% {w['pnl_usd']:>+8.2f} {w['annual_pnl_usd']:>+9.2f} {w['monthly_pnl_usd']:>+9.2f} {('Y' if w['all_periods_positive'] else 'N'):>7} {d['verdict']:>9}")
            results[cost_label] = {"evaluation": ev, "decision": d}
        robust = results["std16"]["decision"]["verdict"] == "PASS" and results["stress24"]["decision"]["verdict"] == "PASS"
        results["robust_PASS"] = robust
        summary[cand["id"]] = {**cand, "results": results}
        print()

    g191_std = summary["G191_ref"]["results"]["std16"]["evaluation"]["weighted"]
    print()
    print("=" * 80)
    print(f"vs G191 baseline (std16): WR {g191_std['win_rate']*100:.1f}% / annual ${g191_std['annual_pnl_usd']:.2f} / monthly ${g191_std['monthly_pnl_usd']:.2f}")
    print("=" * 80)

    deploy_targets = []
    for sid, s in summary.items():
        if not s.get("deploy"): continue
        if not s["results"]["robust_PASS"]: continue
        std_w = s["results"]["std16"]["evaluation"]["weighted"]
        # Improvement vs G191
        delta_pnl = std_w["annual_pnl_usd"] - g191_std["annual_pnl_usd"]
        delta_wr = (std_w["win_rate"] - g191_std["win_rate"]) * 100
        s["delta_vs_g191"] = {"annual_pnl_usd": round(delta_pnl, 2), "wr_pp": round(delta_wr, 2)}
        sign = "+" if delta_pnl >= 0 else ""
        print(f"  🏆 {sid}: WR {std_w['win_rate']*100:.1f}% (Δ{sign}{delta_wr:+.1f}pp) / annual ${std_w['annual_pnl_usd']:.2f} (Δ{sign}{delta_pnl:+.2f})")
        deploy_targets.append(sid)

    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g210_230_batch_summary.json"
    out.write_text(json.dumps({
        "criteria": DECISION_CRITERIA,
        "candidates": summary,
        "deploy_targets": deploy_targets,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")
    print(f"\nDeploy targets (robust PASS): {deploy_targets}")


if __name__ == "__main__":
    main()
