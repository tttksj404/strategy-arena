"""Per-symbol breakdown for deployed strategies — alpha driver vs dead weight."""
import json, sys
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

STRATEGIES = {
    "G185": {"size":0.40,"lev":5.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0},
    "G186": {"size":0.45,"lev":5.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0},
    "G187": {"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0},
    "G190": {"size":0.45,"lev":6.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0},
    "G191": {"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0},
    "G192": {"size":0.40,"lev":6.0,"thr":80,"hold":24,"max_conc":5,"atr":6.0},
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


def gather_with_sym(dfs, threshold, hold_bars, atr_guard_pct):
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


def portfolio_sim_per_sym(entries, params):
    if len(entries) == 0: return {}
    HOLD_MS = params["hold"] * 3600 * 1000
    open_pos = []
    per_sym = {}  # sym -> {n, wins, pnl_usd, trades_pct}
    for _, row in entries.iterrows():
        ts = row["open_time"]
        sym = row["sym"]
        net_bps = row["fwd_pct"] - COST_BPS_RT
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == sym for p in open_pos): continue
        if len(open_pos) >= params["max_conc"]: continue
        margin = EQUITY * params["size"]
        net_pct = net_bps / 10000 * params["lev"]
        if net_pct < -0.90: net_pct = -0.90
        trade_pnl = margin * net_pct
        d = per_sym.setdefault(sym, {"n":0, "wins":0, "pnl_usd":0.0, "pnl_pcts":[]})
        d["n"] += 1
        d["pnl_usd"] += trade_pnl
        d["pnl_pcts"].append(net_pct * 100)
        if trade_pnl > 0: d["wins"] += 1
        open_pos.append((ts + HOLD_MS, sym))
    for sym in per_sym:
        d = per_sym[sym]
        d["wr"] = round(d["wins"]/d["n"], 3) if d["n"] else 0
        d["pnl_usd"] = round(d["pnl_usd"], 2)
        d["avg_pct"] = round(np.mean(d["pnl_pcts"]), 2)
        d["max_pct"] = round(max(d["pnl_pcts"]), 2)
        d["min_pct"] = round(min(d["pnl_pcts"]), 2)
        del d["pnl_pcts"]
    return per_sym


def per_strategy(sid, params):
    periods = [
        ("OOS22-23", load_period_dfs(DATA_22, UNIV_22), 730),
        ("OOS24-Q1", load_period_dfs(DATA_24, UNIV_24), 456),
        ("IS25-26",  load_period_dfs(DATA_25, UNIV_25), 374),
    ]
    sym_total = {}  # aggregate over all 3 periods
    for plabel, dfs, days in periods:
        ent = gather_with_sym(dfs, params["thr"], params["hold"], params["atr"])
        ps = portfolio_sim_per_sym(ent, params)
        for sym, d in ps.items():
            t = sym_total.setdefault(sym, {"n":0, "wins":0, "pnl_usd":0.0, "periods":[]})
            t["n"] += d["n"]; t["wins"] += d["wins"]; t["pnl_usd"] += d["pnl_usd"]
            t["periods"].append(plabel)
    for sym in sym_total:
        t = sym_total[sym]
        t["wr"] = round(t["wins"]/t["n"], 3) if t["n"] else 0
        t["pnl_usd"] = round(t["pnl_usd"], 2)
        t["periods"] = ",".join(t["periods"])
    return sym_total


def main():
    print("=" * 100)
    print("Per-symbol breakdown (전체 universe = 19 syms across 3 periods)")
    print("=" * 100)

    all_data = {}
    for sid, params in STRATEGIES.items():
        all_data[sid] = per_strategy(sid, params)

    # Strategy-specific per-symbol breakdown
    for sid in STRATEGIES.keys():
        data = all_data[sid]
        rows = sorted(data.items(), key=lambda x: x[1]["pnl_usd"], reverse=True)
        total_pnl = sum(d["pnl_usd"] for _, d in rows)
        total_n = sum(d["n"] for _, d in rows)
        total_wins = sum(d["wins"] for _, d in rows)
        print(f"\n--- {sid} per-symbol contribution (total ${total_pnl:.2f}, {total_n} trades, {total_wins/max(total_n,1)*100:.1f}% WR) ---")
        print(f"  {'sym':<10} {'n':>4} {'wins':>5} {'WR%':>6} {'PnL$':>9} {'%total':>7} {'periods':<25}")
        for sym, d in rows:
            pct_total = d["pnl_usd"] / total_pnl * 100 if total_pnl else 0
            print(f"  {sym:<10} {d['n']:>4} {d['wins']:>5} {d['wr']*100:>5.1f}% {d['pnl_usd']:>+8.2f} {pct_total:>+6.1f}% {d['periods']:<25}")
        # Symbols with no entries
        all_syms = set(UNIV_22 + UNIV_24 + UNIV_25)
        no_entry = sorted(all_syms - set(data.keys()))
        if no_entry:
            print(f"  no-entry ({len(no_entry)}): {','.join(no_entry)}")

    # Cross-strategy summary: which symbols are alpha drivers (top by pooled PnL across strategies)
    print("\n" + "=" * 100)
    print("Cross-strategy: 코인별 누적 기여 (6 strategies 합산)")
    print("=" * 100)
    pool = {}
    for sid in STRATEGIES.keys():
        for sym, d in all_data[sid].items():
            p = pool.setdefault(sym, {"n":0, "wins":0, "pnl_usd":0.0, "appears_in":0})
            p["n"] += d["n"]; p["wins"] += d["wins"]; p["pnl_usd"] += d["pnl_usd"]
            if d["n"] > 0: p["appears_in"] += 1
    for sym in pool:
        p = pool[sym]
        p["wr"] = round(p["wins"]/p["n"], 3) if p["n"] else 0
        p["pnl_usd"] = round(p["pnl_usd"], 2)
    rows = sorted(pool.items(), key=lambda x: x[1]["pnl_usd"], reverse=True)
    print(f"\n  {'sym':<10} {'pooled_n':>9} {'WR%':>6} {'PnL$':>10} {'in_strat':>9}")
    for sym, p in rows:
        print(f"  {sym:<10} {p['n']:>9} {p['wr']*100:>5.1f}% {p['pnl_usd']:>+9.2f} {p['appears_in']:>4}/{len(STRATEGIES)}")

    # Save full data
    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "per_symbol_breakdown.json"
    out.write_text(json.dumps(all_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
