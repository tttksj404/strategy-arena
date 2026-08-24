"""G400 secondary validation — 4 stress tests.

PASS only if ALL 4:
  1. Sub-period robustness: split each 3 OOS periods into halves (6 halves), all positive
  2. Cost stress 40bps: still PASS basic criteria at extreme cost
  3. Max drawdown <= -30% of equity ($30 on $100)
  4. Symbol concentration: drop top 3 contributors, still PASS
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

EQUITY = 100.0

CANDIDATES = [
    {"id":"G400", "params":{"size":0.20,"lev":10.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "filter":"drop_dead"},
    {"id":"G401", "params":{"size":0.20,"lev":12.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "filter":"drop_dead"},
    {"id":"G402", "params":{"size":0.20,"lev":15.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "filter":"drop_dead"},
    {"id":"G403", "params":{"size":0.20,"lev":20.0,"thr":80,"hold":24,"max_conc":5,"atr":8.0}, "filter":"drop_dead"},
    {"id":"G405", "params":{"size":0.20,"lev":20.0,"thr":80,"hold":24,"max_conc":5,"atr":6.0}, "filter":"drop_dead"},
    {"id":"G406", "params":{"size":0.15,"lev":15.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "filter":"drop_dead"},
    {"id":"G408", "params":{"size":0.20,"lev":10.0,"thr":80,"hold":24,"max_conc":8,"atr":8.0}, "filter":"drop_dead"},
]


def filter_universe(base, mode):
    if mode == "drop_dead":
        return [s for s in base if s not in DEAD_WEIGHT]
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


def portfolio_sim_full(entries, equity, size_pct, leverage, hold_bars, max_conc, cost_bps, exclude_syms=None):
    """Returns trades list with equity curve + per-symbol PnL."""
    if exclude_syms is None: exclude_syms = set()
    if len(entries) == 0: return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos = []
    trades = []  # (ts, sym, pnl_usd, cum_equity)
    cum = equity
    peak = equity
    max_dd_pct = 0.0
    per_sym = {}
    for _, row in entries.iterrows():
        ts = row["open_time"]
        sym = row["sym"]
        if sym in exclude_syms: continue
        net_bps = row["fwd_pct"] - cost_bps
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == sym for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        margin = equity * size_pct
        net_pct = net_bps / 10000 * leverage
        if net_pct < -0.90: net_pct = -0.90
        trade_pnl = margin * net_pct
        cum += trade_pnl
        peak = max(peak, cum)
        dd = (peak - cum) / peak * 100
        max_dd_pct = max(max_dd_pct, dd)
        trades.append({"ts": int(ts), "sym": sym, "pnl_usd": trade_pnl, "cum_equity": cum})
        d = per_sym.setdefault(sym, {"n":0, "wins":0, "pnl_usd":0.0})
        d["n"] += 1
        d["pnl_usd"] += trade_pnl
        if trade_pnl > 0: d["wins"] += 1
        open_pos.append((ts + HOLD_MS, sym))
    if not trades: return None
    return {
        "trades": trades,
        "n": len(trades),
        "wins": sum(1 for t in trades if t["pnl_usd"] > 0),
        "pnl_usd": round(sum(t["pnl_usd"] for t in trades), 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "per_sym": per_sym,
    }


def check_sub_period(params, filter_mode, cost_bps=16):
    """Split each 3 OOS periods into halves -> 6 sub-periods. All positive?"""
    periods = [
        ("OOS22-23", filter_universe(UNIV_22, filter_mode), DATA_22, 730),
        ("OOS24-Q1", filter_universe(UNIV_24, filter_mode), DATA_24, 456),
        ("IS25-26",  filter_universe(UNIV_25, filter_mode), DATA_25, 374),
    ]
    sub_results = []
    for plabel, universe, dpath, days in periods:
        dfs = load_period_dfs(dpath, universe)
        ent = gather_long_entries(dfs, params["thr"], params["hold"], params["atr"])
        if len(ent) == 0:
            sub_results.append((f"{plabel}_H1", 0)); sub_results.append((f"{plabel}_H2", 0)); continue
        # split chronologically
        mid_ts = ent["open_time"].iloc[len(ent) // 2]
        h1 = ent[ent["open_time"] < mid_ts]
        h2 = ent[ent["open_time"] >= mid_ts]
        for sublabel, sub_ent in [(f"{plabel}_H1", h1), (f"{plabel}_H2", h2)]:
            r = portfolio_sim_full(sub_ent, EQUITY, params["size"], params["lev"], params["hold"], params["max_conc"], cost_bps)
            sub_results.append((sublabel, r["pnl_usd"] if r else 0))
    all_pos = all(pnl > 0 for _, pnl in sub_results)
    n_pos = sum(1 for _, pnl in sub_results if pnl > 0)
    return {"sub_results": sub_results, "all_positive": all_pos, "n_positive": n_pos, "n_total": 6}


def check_high_cost(params, filter_mode, cost_bps=40):
    """40 bps cost (worst-case Bitget alt slippage + funding)."""
    periods = [
        ("OOS22-23", filter_universe(UNIV_22, filter_mode), DATA_22, 730),
        ("OOS24-Q1", filter_universe(UNIV_24, filter_mode), DATA_24, 456),
        ("IS25-26",  filter_universe(UNIV_25, filter_mode), DATA_25, 374),
    ]
    res = []
    for plabel, universe, dpath, days in periods:
        dfs = load_period_dfs(dpath, universe)
        ent = gather_long_entries(dfs, params["thr"], params["hold"], params["atr"])
        r = portfolio_sim_full(ent, EQUITY, params["size"], params["lev"], params["hold"], params["max_conc"], cost_bps)
        res.append({"period": plabel, "days": days, "result": r})
    total_pnl = sum(r["result"]["pnl_usd"] for r in res if r["result"])
    total_n = sum(r["result"]["n"] for r in res if r["result"])
    total_wins = sum(r["result"]["wins"] for r in res if r["result"])
    total_days = sum(r["days"] for r in res if r["result"])
    wr = round(total_wins/max(total_n,1), 4)
    annual_pnl = total_pnl/total_days*365 if total_days else 0
    all_pos = all(r["result"] and r["result"]["pnl_usd"] > 0 for r in res)
    return {
        "annual_pnl_usd": round(annual_pnl, 2),
        "win_rate": wr,
        "n": total_n,
        "all_periods_positive": all_pos,
        "passes_basic": (wr >= 0.70 and all_pos and annual_pnl >= 30 and total_n >= 30),
    }


def check_drawdown(params, filter_mode, cost_bps=16, max_dd_threshold_pct=30):
    """Concatenate all 3 periods chronologically -> single equity curve -> max DD."""
    periods = [
        ("OOS22-23", filter_universe(UNIV_22, filter_mode), DATA_22),
        ("OOS24-Q1", filter_universe(UNIV_24, filter_mode), DATA_24),
        ("IS25-26",  filter_universe(UNIV_25, filter_mode), DATA_25),
    ]
    all_trades = []
    cum = EQUITY
    peak = EQUITY
    max_dd_pct = 0.0
    for plabel, universe, dpath in periods:
        dfs = load_period_dfs(dpath, universe)
        ent = gather_long_entries(dfs, params["thr"], params["hold"], params["atr"])
        r = portfolio_sim_full(ent, EQUITY, params["size"], params["lev"], params["hold"], params["max_conc"], cost_bps)
        if r is None: continue
        for t in r["trades"]:
            cum += t["pnl_usd"]
            peak = max(peak, cum)
            dd = (peak - cum) / peak * 100
            max_dd_pct = max(max_dd_pct, dd)
            all_trades.append({**t, "period": plabel, "cum_global": cum})
    return {
        "max_dd_pct": round(max_dd_pct, 2),
        "max_dd_usd": round(max_dd_pct/100 * peak, 2),
        "passes": max_dd_pct <= max_dd_threshold_pct,
        "n_trades": len(all_trades),
    }


def check_symbol_concentration(params, filter_mode, cost_bps=16, drop_top=3):
    """Identify top-N contributors, drop them, see if strategy still passes basic criteria."""
    periods = [
        ("OOS22-23", filter_universe(UNIV_22, filter_mode), DATA_22, 730),
        ("OOS24-Q1", filter_universe(UNIV_24, filter_mode), DATA_24, 456),
        ("IS25-26",  filter_universe(UNIV_25, filter_mode), DATA_25, 374),
    ]
    # First pass: compute per-symbol PnL across all periods
    all_per_sym = {}
    for plabel, universe, dpath, days in periods:
        dfs = load_period_dfs(dpath, universe)
        ent = gather_long_entries(dfs, params["thr"], params["hold"], params["atr"])
        r = portfolio_sim_full(ent, EQUITY, params["size"], params["lev"], params["hold"], params["max_conc"], cost_bps)
        if r is None: continue
        for sym, d in r["per_sym"].items():
            t = all_per_sym.setdefault(sym, 0.0)
            all_per_sym[sym] = t + d["pnl_usd"]
    # Sort by PnL desc
    top_syms = sorted(all_per_sym.items(), key=lambda x: x[1], reverse=True)[:drop_top]
    drop = {sym for sym, _ in top_syms}
    # Re-run excluding top contributors
    total_pnl = 0.0
    total_n = 0
    total_wins = 0
    total_days = 0
    all_pos = True
    period_pnls = []
    for plabel, universe, dpath, days in periods:
        dfs = load_period_dfs(dpath, universe)
        ent = gather_long_entries(dfs, params["thr"], params["hold"], params["atr"])
        r = portfolio_sim_full(ent, EQUITY, params["size"], params["lev"], params["hold"], params["max_conc"], cost_bps, exclude_syms=drop)
        if r is None:
            period_pnls.append((plabel, 0)); all_pos = False; continue
        period_pnls.append((plabel, r["pnl_usd"]))
        total_pnl += r["pnl_usd"]; total_n += r["n"]; total_wins += r["wins"]; total_days += days
        if r["pnl_usd"] <= 0: all_pos = False
    wr = round(total_wins/max(total_n,1), 4)
    annual_pnl = total_pnl/max(total_days,1)*365
    return {
        "dropped": list(drop),
        "dropped_pnl_total": round(sum(v for _, v in top_syms), 2),
        "remaining_annual_pnl_usd": round(annual_pnl, 2),
        "remaining_wr": wr,
        "remaining_n": total_n,
        "all_periods_positive": all_pos,
        "period_pnls": period_pnls,
        "passes_basic": (wr >= 0.70 and all_pos and annual_pnl >= 30 and total_n >= 30),
    }


def main():
    print("=" * 130)
    print("G400 SECONDARY VALIDATION — 4 stress tests")
    print("=" * 130)
    results = {}
    for cand in CANDIDATES:
        sid = cand["id"]
        p = cand["params"]
        print(f"\n--- {sid} (size={p['size']} lev={p['lev']}x conc={p['max_conc']} atr={p['atr']}%) ---")
        s = {}
        s["sub_period"] = check_sub_period(p, cand["filter"])
        print(f"  [1] sub-period (6 halves all positive): {s['sub_period']['n_positive']}/{s['sub_period']['n_total']} positive — {'PASS' if s['sub_period']['all_positive'] else 'FAIL'}")
        for label, pnl in s["sub_period"]["sub_results"]:
            mark = "+" if pnl > 0 else ""
            print(f"      {label}: ${pnl:>+8.2f}")

        s["high_cost"] = check_high_cost(p, cand["filter"])
        print(f"  [2] cost 40bps stress: ann ${s['high_cost']['annual_pnl_usd']:.2f}, WR {s['high_cost']['win_rate']*100:.1f}%, n={s['high_cost']['n']}, allPos={s['high_cost']['all_periods_positive']} — {'PASS' if s['high_cost']['passes_basic'] else 'FAIL'}")

        s["drawdown"] = check_drawdown(p, cand["filter"])
        print(f"  [3] drawdown ≤30% (chronological): max DD {s['drawdown']['max_dd_pct']:.2f}% (${s['drawdown']['max_dd_usd']:.2f}) — {'PASS' if s['drawdown']['passes'] else 'FAIL'}")

        s["concentration"] = check_symbol_concentration(p, cand["filter"])
        c = s["concentration"]
        print(f"  [4] symbol concentration (drop top 3 = {','.join(c['dropped'])}, ${c['dropped_pnl_total']:.2f}): remaining ann ${c['remaining_annual_pnl_usd']:.2f}, WR {c['remaining_wr']*100:.1f}%, n={c['remaining_n']}, allPos={c['all_periods_positive']} — {'PASS' if c['passes_basic'] else 'FAIL'}")

        all_pass = (s["sub_period"]["all_positive"] and s["high_cost"]["passes_basic"] and s["drawdown"]["passes"] and s["concentration"]["passes_basic"])
        s["all_pass"] = all_pass
        s["verdict"] = "ROBUST_PASS" if all_pass else "FAIL"
        print(f"  ===> VERDICT: {s['verdict']}")
        results[sid] = s

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)
    print(f"{'ID':<6} {'sub-period':>11} {'cost40bps':>11} {'max DD%':>10} {'concentration':>14} {'verdict':>15}")
    survivors = []
    for sid, s in results.items():
        sp = "PASS" if s["sub_period"]["all_positive"] else f"FAIL({s['sub_period']['n_positive']}/6)"
        hc = "PASS" if s["high_cost"]["passes_basic"] else "FAIL"
        dd = f"{s['drawdown']['max_dd_pct']:.1f}% {'PASS' if s['drawdown']['passes'] else 'FAIL'}"
        cc = "PASS" if s["concentration"]["passes_basic"] else "FAIL"
        print(f"{sid:<6} {sp:>11} {hc:>11} {dd:>10} {cc:>14} {s['verdict']:>15}")
        if s["all_pass"]: survivors.append(sid)
    print(f"\nROBUST_PASS survivors: {len(survivors)}/{len(CANDIDATES)} → {survivors}")

    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g400_secondary_validation.json"
    out.write_text(json.dumps({"results": results, "survivors": survivors}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
