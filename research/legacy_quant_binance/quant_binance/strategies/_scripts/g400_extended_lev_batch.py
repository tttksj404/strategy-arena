"""G400 series — extended leverage range 10x-20x peak ($100 capital).

User context updated: peak leverage 5-10x → expanded to 5-20x.

New tracking: liquidation_near_misses (trades where net_pct hit -90% floor)
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

DEAD_WEIGHT = {"WIFUSDT", "LTCUSDT", "BTCUSDT"}
TOP_10 = {"DOGEUSDT","PEPEUSDT","SOLUSDT","ARBUSDT","ADAUSDT","LINKUSDT","DOTUSDT","NEARUSDT","AVAXUSDT","UNIUSDT"}

EQUITY = 100.0
COST_STANDARD = 16.0
COST_STRESS = 24.0

CANDIDATES = [
    {"id":"G400", "params":{"size":0.20,"lev":10.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "univ_filter":"drop_dead", "desc":"10x ref (G305 reprise)"},
    {"id":"G401", "params":{"size":0.20,"lev":12.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "univ_filter":"drop_dead", "desc":"12x lev"},
    {"id":"G402", "params":{"size":0.20,"lev":15.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "univ_filter":"drop_dead", "desc":"15x lev"},
    {"id":"G403", "params":{"size":0.20,"lev":20.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "univ_filter":"drop_dead", "desc":"20x MAX edge"},
    {"id":"G404", "params":{"size":0.20,"lev":15.0,"thr":80,"hold":24,"max_conc":5,"atr":6.0}, "univ_filter":"drop_dead", "desc":"15x + atr 6% (safer)"},
    {"id":"G405", "params":{"size":0.20,"lev":20.0,"thr":80,"hold":24,"max_conc":5,"atr":6.0}, "univ_filter":"drop_dead", "desc":"20x + atr 6%"},
    {"id":"G406", "params":{"size":0.15,"lev":15.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "univ_filter":"drop_dead", "desc":"15x + conc 8 (peak 18x)"},
    {"id":"G407", "params":{"size":0.10,"lev":20.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "univ_filter":"drop_dead", "desc":"20x + conc 8 small (16x)"},
    {"id":"G408", "params":{"size":0.20,"lev":10.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "univ_filter":"drop_dead", "desc":"10x + conc 8 (16x)"},
]

DECISION_CRITERIA = {
    "min_wr": 0.70,
    "all_periods_positive": True,
    "min_annual_pnl_usd": 30,
    "min_trades_total": 30,
    "max_peak_leverage": 20.0,
    "max_liquidation_rate": 0.05,  # ≤5% trades hit -90% floor
}


def filter_universe(base, mode):
    if mode == "drop_dead":
        return [s for s in base if s not in DEAD_WEIGHT]
    if mode == "top10":
        return [s for s in base if s in TOP_10]
    return base


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
    liquidation_hits = 0
    for _, row in entries.iterrows():
        ts = row["open_time"]
        net_bps = row["fwd_pct"] - cost_bps
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == row["sym"] for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        margin = equity * size_pct
        net_pct = net_bps / 10000 * leverage
        if net_pct < -0.90:
            net_pct = -0.90
            liquidation_hits += 1
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
        "pnl_usd": round(pnl_usd, 2),
        "annual_pnl_usd": round(pnl_usd/days*365, 2),
        "monthly_pnl_usd": round(pnl_usd/days*30.4, 2),
        "avg_winner_pnl_usd": round(np.mean(win_pnl), 2) if win_pnl else 0,
        "avg_loser_pnl_usd":  round(np.mean(loss_pnl), 2) if loss_pnl else 0,
        "liquidation_hits": liquidation_hits,
        "liquidation_rate": round(liquidation_hits / taken, 4) if taken else 0,
        "big_wins_30pct": big_wins, "big_losses_20pct": big_losses,
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
    total_liq = sum(r.get("liquidation_hits", 0) for r in valid)
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
            "liquidation_hits": total_liq,
            "liquidation_rate": round(total_liq / max(total_taken, 1), 4),
        }
    }


def decide(weighted, theoretical_peak):
    c = DECISION_CRITERIA
    checks = {
        "wr_>=_70%":             weighted["win_rate"] >= c["min_wr"],
        "all_periods_pos":       weighted["all_periods_positive"],
        "annual_pnl_>=_$30":     weighted["annual_pnl_usd"] >= c["min_annual_pnl_usd"],
        "trades_>=_30":          weighted["n_taken"] >= c["min_trades_total"],
        "peak_lev_<=_20x":       theoretical_peak <= c["max_peak_leverage"],
        "liq_rate_<=_5%":        weighted["liquidation_rate"] <= c["max_liquidation_rate"],
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def main():
    print("=" * 165)
    print(f"{'ID':<5} {'desc':<28} {'sz':>4} {'lev':>4} {'cn':>3} {'atr':>4} {'cost':>4} {'trades':>6} {'WR':>6} {'PnL$':>9} {'annPnL$':>9} {'mo$':>6} {'pkLv':>5} {'liq':>4} {'liq%':>5} {'allPos':>7} {'verdict':>8}")
    print("-" * 165)
    summary = {}
    for cand in CANDIDATES:
        p = cand["params"]
        peak = p["max_conc"] * p["size"] * p["lev"]
        results = {}
        for cost_label, cost in [("std16", COST_STANDARD), ("stress24", COST_STRESS)]:
            ev = evaluate(p, cand["univ_filter"], cost)
            d = decide(ev["weighted"], peak)
            w = ev["weighted"]
            print(f"{cand['id']:<5} {cand['desc']:<28} {p['size']:>4.2f} {p['lev']:>4.1f} {p['max_conc']:>3} {p['atr']:>4.1f} {cost:>4.0f} {w['n_taken']:>6} {w['win_rate']*100:>5.1f}% {w['pnl_usd']:>+8.2f} {w['annual_pnl_usd']:>+8.2f} {w['monthly_pnl_usd']:>+5.2f} {peak:>4.1f}x {w['liquidation_hits']:>4} {w['liquidation_rate']*100:>4.1f}% {('Y' if w['all_periods_positive'] else 'N'):>7} {d['verdict']:>8}")
            results[cost_label] = {"evaluation": ev, "decision": d}
        robust = results["std16"]["decision"]["verdict"] == "PASS" and results["stress24"]["decision"]["verdict"] == "PASS"
        results["robust_PASS"] = robust
        results["theoretical_peak_leverage"] = peak
        summary[cand["id"]] = {**cand, "results": results}
        print()

    robust_passers = [(sid, s) for sid, s in summary.items() if s["results"]["robust_PASS"]]
    print()
    print("=" * 90)
    print(f"ROBUST PASS (std16+stress24, peak ≤ 20x, liq ≤ 5%): {len(robust_passers)} / {len(CANDIDATES)}")
    print("=" * 90)
    if robust_passers:
        robust_passers.sort(key=lambda x: x[1]["results"]["std16"]["evaluation"]["weighted"]["annual_pnl_usd"], reverse=True)
        print(f"{'rank':<5} {'ID':<5} {'desc':<28} {'WR':>6} {'annPnL$':>9} {'mo$':>6} {'peakLv':>7} {'liq%':>5}")
        print("-" * 80)
        for i, (sid, s) in enumerate(robust_passers, 1):
            w = s["results"]["std16"]["evaluation"]["weighted"]
            peak = s["results"]["theoretical_peak_leverage"]
            print(f"{i:<5} {sid:<5} {s['desc']:<28} {w['win_rate']*100:>5.1f}% {w['annual_pnl_usd']:>+8.2f} {w['monthly_pnl_usd']:>+5.2f} {peak:>6.1f}x {w['liquidation_rate']*100:>4.1f}%")

    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g400_batch_summary.json"
    out.write_text(json.dumps({
        "criteria": DECISION_CRITERIA,
        "candidates": summary,
        "robust_passers": [sid for sid, _ in robust_passers],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
