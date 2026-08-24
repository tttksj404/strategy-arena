"""G300 series — REALISTIC $100 capital constraint.

제약:
  - peak leverage = max_conc × size_pct × leverage ≤ 10x (user 5-10x 컨텍스트)
  - margin total ≤ 100% of equity (no over-allocation)
  - universe = drop WIF/LTC/BTC (G210 lesson)
  - cross margin assumed (Bitget default)

Decision criteria (downsized realistic):
  - WR ≥ 0.70
  - All 3 periods positive
  - Annual PnL ≥ $30 (user's original $100 goal)
  - Trades ≥ 30
  - Peak leverage ≤ 10x

Cost stress: 16bps standard + 24bps stress.
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

# Candidates spanning peak leverage 5x-10x range
CANDIDATES = [
    # peak 5x family
    {"id":"G300", "params":{"size":0.20,"lev":5.0, "thr":80,"hold":24,"max_conc":5,"atr":8.0}, "univ_filter":"drop_dead", "desc":"safe 5x baseline"},
    # peak 6x
    {"id":"G301", "params":{"size":0.20,"lev":6.0, "thr":80,"hold":24,"max_conc":5,"atr":8.0}, "univ_filter":"drop_dead", "desc":"6x lev"},
    # peak 7.2x (smaller size, more concurrent)
    {"id":"G302", "params":{"size":0.15,"lev":6.0, "thr":80,"hold":24,"max_conc":8,"atr":8.0}, "univ_filter":"drop_dead", "desc":"conc8 + smaller size"},
    # peak 8x
    {"id":"G303", "params":{"size":0.20,"lev":8.0, "thr":80,"hold":24,"max_conc":5,"atr":8.0}, "univ_filter":"drop_dead", "desc":"8x lev"},
    # peak 9.6x
    {"id":"G304", "params":{"size":0.15,"lev":8.0, "thr":80,"hold":24,"max_conc":8,"atr":8.0}, "univ_filter":"drop_dead", "desc":"high lev + conc8"},
    # peak 10x edge
    {"id":"G305", "params":{"size":0.20,"lev":10.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "univ_filter":"drop_dead", "desc":"max edge 10x"},
    # peak 8x with TOP 10 only
    {"id":"G306", "params":{"size":0.15,"lev":6.0, "thr":80,"hold":24,"max_conc":8,"atr":8.0}, "univ_filter":"top10",    "desc":"top10 + conc8"},
    # peak 6x with tighter ATR
    {"id":"G307", "params":{"size":0.20,"lev":6.0, "thr":80,"hold":24,"max_conc":5,"atr":6.0}, "univ_filter":"drop_dead", "desc":"6x + atr 6%"},
]

DECISION_CRITERIA = {
    "min_wr": 0.70,
    "all_periods_positive": True,
    "min_annual_pnl_usd": 30,  # user's original target on $100
    "min_trades_total": 30,
    "max_peak_leverage": 10.0,
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
    peak_concurrent = 0
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
        "pnl_usd": round(pnl_usd, 2),
        "annual_pnl_usd": round(pnl_usd/days*365, 2),
        "monthly_pnl_usd": round(pnl_usd/days*30.4, 2),
        "avg_winner_pnl_usd": round(np.mean(win_pnl), 2) if win_pnl else 0,
        "avg_loser_pnl_usd":  round(np.mean(loss_pnl), 2) if loss_pnl else 0,
        "peak_concurrent": peak_concurrent,
        "peak_notional_x_equity": round(peak_concurrent * size_pct * leverage, 2),
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


def decide(weighted, theoretical_peak):
    c = DECISION_CRITERIA
    checks = {
        "wr_>=_70%":          weighted["win_rate"] >= c["min_wr"],
        "all_periods_pos":    weighted["all_periods_positive"],
        "annual_pnl_>=_$30":  weighted["annual_pnl_usd"] >= c["min_annual_pnl_usd"],
        "trades_>=_30":       weighted["n_taken"] >= c["min_trades_total"],
        "peak_lev_<=_10x":    theoretical_peak <= c["max_peak_leverage"],
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def main():
    print("=" * 145)
    print(f"{'ID':<6} {'desc':<26} {'size':>5} {'lev':>4} {'conc':>5} {'atr':>4} {'univ':<10} {'cost':>5} {'trades':>6} {'WR':>6} {'PnL$':>8} {'annPnL$':>8} {'mo$':>6} {'peakLv':>7} {'allPos':>7} {'verdict':>8}")
    print("-" * 145)
    summary = {}
    for cand in CANDIDATES:
        p = cand["params"]
        theoretical_peak = p["max_conc"] * p["size"] * p["lev"]
        results = {}
        for cost_label, cost in [("std16", COST_STANDARD), ("stress24", COST_STRESS)]:
            ev = evaluate(p, cand["univ_filter"], cost)
            d = decide(ev["weighted"], theoretical_peak)
            w = ev["weighted"]
            print(f"{cand['id']:<6} {cand['desc']:<26} {p['size']:>5.2f} {p['lev']:>4.1f} {p['max_conc']:>5} {p['atr']:>4.1f} {cand['univ_filter']:<10} {cost:>5.0f} {w['n_taken']:>6} {w['win_rate']*100:>5.1f}% {w['pnl_usd']:>+7.2f} {w['annual_pnl_usd']:>+7.2f} {w['monthly_pnl_usd']:>+5.2f} {theoretical_peak:>6.1f}x {('Y' if w['all_periods_positive'] else 'N'):>7} {d['verdict']:>8}")
            results[cost_label] = {"evaluation": ev, "decision": d}
        robust = results["std16"]["decision"]["verdict"] == "PASS" and results["stress24"]["decision"]["verdict"] == "PASS"
        results["robust_PASS"] = robust
        results["theoretical_peak_leverage"] = theoretical_peak
        summary[cand["id"]] = {**cand, "results": results}
        print()

    robust_passers = [(sid, s) for sid, s in summary.items() if s["results"]["robust_PASS"]]
    print()
    print("=" * 80)
    print(f"ROBUST PASS (std16+stress24, peak ≤ 10x): {len(robust_passers)} / {len(CANDIDATES)}")
    print("=" * 80)
    if robust_passers:
        # rank by annual PnL
        robust_passers.sort(key=lambda x: x[1]["results"]["std16"]["evaluation"]["weighted"]["annual_pnl_usd"], reverse=True)
        print(f"{'rank':<5} {'ID':<6} {'desc':<28} {'WR':>6} {'annPnL$':>8} {'mo$':>6} {'peakLv':>7}")
        print("-" * 75)
        for i, (sid, s) in enumerate(robust_passers, 1):
            w = s["results"]["std16"]["evaluation"]["weighted"]
            peak = s["results"]["theoretical_peak_leverage"]
            print(f"{i:<5} {sid:<6} {s['desc']:<28} {w['win_rate']*100:>5.1f}% {w['annual_pnl_usd']:>+7.2f} {w['monthly_pnl_usd']:>+5.2f} {peak:>6.1f}x")

    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g300_batch_summary.json"
    out.write_text(json.dumps({
        "criteria": DECISION_CRITERIA,
        "candidates": summary,
        "robust_passers": [sid for sid, _ in robust_passers],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
