"""G186-G189 batch generator + validator + winner selector.

Variable-1 changes from G185:
  G186: size_pct_per_trade  0.40 -> 0.45
  G187: leverage            5.0  -> 6.0
  G188: holding_period_bars 24   -> 48
  G189: entry_threshold     80   -> 85

Pipeline:
  1) write each strategy's overrides.json + card.md skeleton
  2) walk-forward 3-period backtest (reuses g185 logic)
  3) decision: WR >=0.70 + all 3 periods positive + annual PnL >=$300 + trades >=30
  4) pick winners + write batch_summary.json + recommend deploy targets
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
SCRIPTS = ROOT / "quant_binance" / "strategies" / "_scripts"
sys.path.insert(0, str(SCRIPTS))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct  # type: ignore

DATA_22 = ROOT / "quant_runtime" / "historical_2022"
DATA_24 = ROOT / "quant_runtime" / "historical_2024"
DATA_25 = ROOT / "quant_runtime" / "historical_is25"

UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

EQUITY = 100.0
MAX_CONC = 5
ATR_GUARD = 8.0

CANDIDATES = [
    {"id":"G186","folder":"G186_size45_100usd",     "var":"size_pct_per_trade",   "from":0.40,"to":0.45,"params":{"size":0.45,"lev":5.0,"thr":80,"hold":24}},
    {"id":"G187","folder":"G187_lev6_100usd",        "var":"leverage",             "from":5.0, "to":6.0, "params":{"size":0.40,"lev":6.0,"thr":80,"hold":24}},
    {"id":"G188","folder":"G188_hold48_100usd",      "var":"holding_period_bars",  "from":24,  "to":48,  "params":{"size":0.40,"lev":5.0,"thr":80,"hold":48}},
    {"id":"G189","folder":"G189_thr85_100usd",       "var":"entry_threshold",      "from":80,  "to":85,  "params":{"size":0.40,"lev":5.0,"thr":85,"hold":24}},
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
            data.sort(key=lambda b: b["open_time"])
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        for c in ("open_price","high_price","low_price","close_price","base_volume","quote_volume"):
            if c in df.columns:
                df[c] = df[c].astype(float)
        if len(df) >= 100:
            out[sym] = df
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
        if len(e) == 0:
            continue
        e["sym"] = sym
        rows.append(e[["open_time","score","atr_pct","fwd_pct","sym"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).sort_values("open_time").reset_index(drop=True)


def portfolio_sim(entries, equity, size_pct, leverage, hold_bars, max_conc, days):
    if len(entries) == 0: return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos, pnl_usd, taken, wins = [], 0.0, 0, 0
    big_wins, big_losses = 0, 0
    win_pnl, loss_pnl = [], []
    for _, row in entries.iterrows():
        ts = row["open_time"]
        net_bps = row["fwd_pct"] - COST_BPS_RT
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


def write_overrides(folder_name, sid, var, params):
    out_dir = ROOT / "quant_binance" / "strategies" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {
        "_strategy_id": sid,
        "_parent_id": "G185",
        "_playbook": "PB001",
        "_setup": f"G185 + {var} change",
        "_changed_keys_vs_parent": [var],
        "_capital_context_usd": 100,
        "score_engine": {"$ref": "G004"},
        "entry_threshold": params["thr"],
        "holding_period_bars": params["hold"],
        "timeframe": "1h",
        "universe": UNIV_25,
        "side": "long",
        "leverage": params["lev"],
        "size_pct_per_trade": params["size"],
        "max_concurrent": MAX_CONC,
        "cost_bps_round_trip": COST_BPS_RT,
        "atr_volatility_guard": {"enabled": True, "max_atr_pct": ATR_GUARD,
                                 "rule": f"atr_pct(14) > {ATR_GUARD}% skip"},
    }
    (out_dir / "overrides.json").write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def evaluate(cand):
    p = cand["params"]
    periods = [
        ("OOS22-23", load_period_dfs(DATA_22, UNIV_22), 730),
        ("OOS24-Q1", load_period_dfs(DATA_24, UNIV_24), 456),
        ("IS25-26",  load_period_dfs(DATA_25, UNIV_25), 374),
    ]
    res = {}
    for plabel, dfs, days in periods:
        ent = gather_long_entries(dfs, p["thr"], p["hold"], ATR_GUARD)
        r = portfolio_sim(ent, EQUITY, p["size"], p["lev"], p["hold"], MAX_CONC, days)
        if r is None:
            res[plabel] = {"days": days, "n_taken": 0, "note": "no entries"}
        else:
            r["days"] = days; res[plabel] = r
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
        "all_periods_positive": weighted["all_periods_positive"],
        "annual_pnl_>=_$300": weighted["annual_pnl_usd"] >= c["min_annual_pnl_usd"],
        "trades_>=_30":       weighted["n_taken"] >= c["min_trades_total"],
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def main():
    summary = {}
    print("=" * 110)
    print(f"{'ID':<6} {'var change':<25} {'trades':>6} {'WR':>6} {'PnL$':>9} {'annPnL$':>10} {'monthly$':>10} {'allPos':>7} {'VERDICT':>8}")
    print("-" * 110)
    for cand in CANDIDATES:
        out_dir = write_overrides(cand["folder"], cand["id"], cand["var"], cand["params"])
        ev = evaluate(cand)
        d = decide(ev["weighted"])
        var_str = f"{cand['var']} {cand['from']}->{cand['to']}"
        w = ev["weighted"]
        print(f"{cand['id']:<6} {var_str:<25} {w['n_taken']:>6} {w['win_rate']*100:>5.1f}% {w['pnl_usd']:>+8.2f} {w['annual_pnl_usd']:>+9.2f} {w['monthly_pnl_usd']:>+9.2f} {('Y' if w['all_periods_positive'] else 'N'):>7} {d['verdict']:>8}")
        # save runs
        runs = out_dir / "runs"
        runs.mkdir(exist_ok=True)
        (runs / "validation_3period.json").write_text(json.dumps({
            "strategy": cand["id"], "parent": "G185", "var": cand["var"], "from": cand["from"], "to": cand["to"],
            "params": cand["params"], "evaluation": ev, "decision": d,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        summary[cand["id"]] = {**cand, "evaluation": ev, "decision": d}

    # G185 reference (already validated)
    g185_ref = {"weighted": {"n_taken": 66, "win_rate": 0.849, "pnl_usd": 1464.50,
                              "annual_pnl_usd": 342.66, "monthly_pnl_usd": 28.55, "all_periods_positive": True}}
    g185_dec = decide(g185_ref["weighted"])
    print(f"{'G185':<6} {'(parent baseline)':<25} {66:>6} {84.9:>5.1f}% {1464.50:>+8.2f} {342.66:>+9.2f} {28.55:>+9.2f} {'Y':>7} {g185_dec['verdict']:>8}")
    print("-" * 110)

    # Pick winners
    passers = [(sid, s) for sid, s in summary.items() if s["decision"]["verdict"] == "PASS"]
    print()
    print(f"=== PASS count: {len(passers)} / {len(CANDIDATES)} ===")
    if passers:
        # rank by annual_pnl_usd desc
        passers.sort(key=lambda x: x[1]["evaluation"]["weighted"]["annual_pnl_usd"], reverse=True)
        print("\nWinners (sorted by annual PnL desc):")
        for sid, s in passers:
            w = s["evaluation"]["weighted"]
            print(f"  {sid}: WR {w['win_rate']*100:.1f}% / annual ${w['annual_pnl_usd']:.2f} / monthly ${w['monthly_pnl_usd']:.2f}")
    else:
        print("No candidate passed. Stay with G185.")

    out_summary = ROOT / "quant_binance" / "strategies" / "_scripts" / "g186_189_batch_summary.json"
    out_summary.write_text(json.dumps({
        "criteria": DECISION_CRITERIA,
        "candidates": summary,
        "g185_reference": {"weighted": g185_ref["weighted"], "decision": g185_dec},
        "passers": [sid for sid, _ in passers],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_summary}")


if __name__ == "__main__":
    main()
