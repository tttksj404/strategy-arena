"""G800 — exhaustive grid search across NEW dimensions.

Dimensions explored (full cross product):
  threshold:    [78, 80, 82]
  hold:         [16, 24, 36]
  atr_max:      [6, 8, 10]
  atr_min:      [0, 3, 5]              # NEW: require minimum volatility
  consecutive:  [1, 2]                 # NEW: require N consecutive bars at threshold
  universe:     [no_dead, top10, meme] # universe variants
  lev:          [10, 15, 20]           # within user 5-20x range
  size_pct:     0.20 (fixed sweet spot)
  max_conc:     5 (fixed)

Total: 3*3*3*3*2*3*3 = 486 combinations × 3 cost levels.

Find top 5 by composite score, identify 2 most novel + strong.
"""
import json, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
SCRIPTS = ROOT / "quant_binance" / "strategies" / "_scripts"
sys.path.insert(0, str(SCRIPTS))
from g002_mingogogo_ch1_backtest import compute_ch1_score, atr_pct, COST_BPS_RT  # type: ignore

DATA_22 = ROOT / "quant_runtime" / "historical_2022"
DATA_24 = ROOT / "quant_runtime" / "historical_2024"
DATA_25 = ROOT / "quant_runtime" / "historical_is25"

UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"]
DEAD = {"WIFUSDT","LTCUSDT","BTCUSDT"}
TOP_10 = {"DOGEUSDT","PEPEUSDT","SOLUSDT","ARBUSDT","ADAUSDT","LINKUSDT","DOTUSDT","NEARUSDT","AVAXUSDT","UNIUSDT"}
MEMECOIN = {"DOGEUSDT","PEPEUSDT","WIFUSDT"}

EQUITY = 100.0

# Existing G400 family params (used to compute novelty distance)
EXISTING_PARAMS = [
    {"thr":80,"hold":24,"atr_max":8,"atr_min":0,"consec":1,"univ":"no_dead","lev":15,"size":0.20,"max_conc":5},  # G402
    {"thr":80,"hold":24,"atr_max":8,"atr_min":0,"consec":1,"univ":"no_dead","lev":20,"size":0.20,"max_conc":5},  # G403
    {"thr":80,"hold":24,"atr_max":6,"atr_min":0,"consec":1,"univ":"no_dead","lev":20,"size":0.20,"max_conc":5},  # G405
    {"thr":80,"hold":24,"atr_max":8,"atr_min":0,"consec":1,"univ":"no_dead","lev":15,"size":0.15,"max_conc":8},  # G406
    {"thr":80,"hold":24,"atr_max":8,"atr_min":0,"consec":1,"univ":"no_dead","lev":10,"size":0.20,"max_conc":8},  # G408
    {"thr":80,"hold":24,"atr_max":8,"atr_min":0,"consec":1,"univ":"meme","lev":20,"size":1.00,"max_conc":1},     # G710
    {"thr":80,"hold":24,"atr_max":8,"atr_min":0,"consec":1,"univ":"meme","lev":30,"size":1.00,"max_conc":1},     # G711
]


def filter_universe(base, mode):
    if mode == "no_dead": return [s for s in base if s not in DEAD]
    if mode == "top10":   return [s for s in base if s in TOP_10]
    if mode == "meme":    return [s for s in base if s in MEMECOIN]
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


def precompute_indicators(dfs, holds):
    """Pre-compute score, atr_pct, and fwd_pct for each hold value. Massive speedup."""
    cache = {}
    for sym, df in dfs.items():
        score, _ = compute_ch1_score(df)
        atrp = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        ts = df["open_time"].astype(int).values
        cl = df["close_price"].astype(float).values
        fwd_pct_by_hold = {}
        for h in holds:
            shifted = np.roll(cl, -h).astype(float)
            shifted[-h:] = np.nan
            fwd = (shifted / cl - 1) * 10000
            fwd_pct_by_hold[h] = fwd
        cache[sym] = {"ts": ts, "score": score.values, "atr": atrp.values, "fwd": fwd_pct_by_hold}
    return cache


def gather_entries(cache, thr, hold, atr_min, atr_max, consec):
    rows = []
    for sym, c in cache.items():
        score, atr_arr, fwd = c["score"], c["atr"], c["fwd"][hold]
        n = len(score)
        if n < 50: continue
        # consecutive: require last `consec` bars all >= thr
        if consec > 1:
            consec_ok = np.ones(n, dtype=bool)
            for k in range(consec):
                shifted = np.roll(score, k)
                shifted[:k] = -np.inf
                consec_ok = consec_ok & (shifted >= thr)
        else:
            consec_ok = score >= thr
        atr_ok = (atr_arr <= atr_max) & (atr_arr >= atr_min)
        fwd_ok = ~np.isnan(fwd)
        mask = consec_ok & atr_ok & fwd_ok
        idxs = np.where(mask)[0]
        if len(idxs) == 0: continue
        for i in idxs:
            rows.append((int(c["ts"][i]), sym, float(fwd[i])))
    if not rows: return []
    rows.sort(key=lambda x: x[0])
    return rows


def portfolio_sim(entries, equity, size_pct, leverage, hold_bars, max_conc, days, cost_bps):
    if not entries: return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos = []
    pnl_usd, taken, wins = 0.0, 0, 0
    big_wins, liquidations = 0, 0
    win_pnl, loss_pnl = [], []
    for ts, sym, gross_bps in entries:
        net_bps = gross_bps - cost_bps
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == sym for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        margin = equity * size_pct
        net_pct = net_bps / 10000 * leverage
        if net_pct < -0.90:
            net_pct = -0.90; liquidations += 1
        trade_pnl = margin * net_pct
        pnl_usd += trade_pnl; taken += 1
        if net_pct > 0: wins += 1; win_pnl.append(trade_pnl)
        else: loss_pnl.append(trade_pnl)
        if net_pct > 0.30: big_wins += 1
        open_pos.append((ts + HOLD_MS, sym))
    return {
        "n": taken,
        "wr": wins / taken if taken else 0,
        "pnl_usd": pnl_usd,
        "annual_pnl_usd": pnl_usd / days * 365 if days else 0,
        "liquidations": liquidations,
        "big_wins": big_wins,
    }


def evaluate(cache_22, cache_24, cache_25, thr, hold, atr_min, atr_max, consec, lev, size, max_conc, cost_bps):
    e22 = gather_entries(cache_22, thr, hold, atr_min, atr_max, consec)
    e24 = gather_entries(cache_24, thr, hold, atr_min, atr_max, consec)
    e25 = gather_entries(cache_25, thr, hold, atr_min, atr_max, consec)
    r22 = portfolio_sim(e22, EQUITY, size, lev, hold, max_conc, 730, cost_bps)
    r24 = portfolio_sim(e24, EQUITY, size, lev, hold, max_conc, 456, cost_bps)
    r25 = portfolio_sim(e25, EQUITY, size, lev, hold, max_conc, 374, cost_bps)
    valid = [r for r in (r22, r24, r25) if r and r["n"] > 0]
    if not valid: return None
    total_n = sum(r["n"] for r in valid)
    total_pnl = sum(r["pnl_usd"] for r in valid)
    total_days = sum(d for r, d in zip((r22, r24, r25), (730, 456, 374)) if r and r["n"] > 0)
    weighted_wr = sum(r["wr"] * r["n"] for r in valid) / max(total_n, 1)
    annual_pnl = total_pnl / total_days * 365 if total_days else 0
    all_pos = all(r and r["pnl_usd"] > 0 for r in (r22, r24, r25)) and len(valid) == 3
    total_liq = sum(r["liquidations"] for r in valid)
    return {
        "n": total_n, "wr": weighted_wr, "annual_pnl": annual_pnl, "monthly_pnl": annual_pnl/12,
        "all_pos": all_pos, "liq_rate": total_liq/total_n if total_n else 0,
    }


def novelty_distance(p):
    """Count how many dims differ from CLOSEST existing strategy. Higher = more novel."""
    min_dist = 99
    for ex in EXISTING_PARAMS:
        d = 0
        for k in ["thr","hold","atr_max","atr_min","consec","univ","lev","size","max_conc"]:
            if p.get(k) != ex.get(k): d += 1
        min_dist = min(min_dist, d)
    return min_dist


def main():
    GRID = {
        "thr":     [78, 80, 82],
        "hold":    [16, 24, 36],
        "atr_max": [6, 8, 10],
        "atr_min": [0, 3, 5],
        "consec":  [1, 2],
        "univ":    ["no_dead", "top10", "meme"],
        "lev":     [10, 15, 20],
    }
    SIZE_FIXED = 0.20
    CONC_FIXED = 5

    keys = list(GRID.keys())
    combos = list(product(*GRID.values()))
    total = len(combos)
    print(f"Total combinations: {total}")
    holds_unique = sorted(set(GRID["hold"]))

    # Pre-load all data once
    print("Loading + precomputing indicators...")
    t0 = time.time()
    cache_full_22 = precompute_indicators(load_period_dfs(DATA_22, UNIV_22), holds_unique)
    cache_full_24 = precompute_indicators(load_period_dfs(DATA_24, UNIV_24), holds_unique)
    cache_full_25 = precompute_indicators(load_period_dfs(DATA_25, UNIV_25), holds_unique)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Pre-filter caches per universe
    universe_caches = {}
    for univ_mode in GRID["univ"]:
        u22 = set(filter_universe(UNIV_22, univ_mode))
        u24 = set(filter_universe(UNIV_24, univ_mode))
        u25 = set(filter_universe(UNIV_25, univ_mode))
        universe_caches[univ_mode] = (
            {s: c for s, c in cache_full_22.items() if s in u22},
            {s: c for s, c in cache_full_24.items() if s in u24},
            {s: c for s, c in cache_full_25.items() if s in u25},
        )

    print(f"Running {total} combinations (cost std=16bps + stress=24bps)...")
    results = []
    t0 = time.time()
    for i, combo in enumerate(combos):
        p = dict(zip(keys, combo))
        c22, c24, c25 = universe_caches[p["univ"]]
        ev_std = evaluate(c22, c24, c25, p["thr"], p["hold"], p["atr_min"], p["atr_max"], p["consec"],
                          p["lev"], SIZE_FIXED, CONC_FIXED, 16.0)
        ev_stress = evaluate(c22, c24, c25, p["thr"], p["hold"], p["atr_min"], p["atr_max"], p["consec"],
                             p["lev"], SIZE_FIXED, CONC_FIXED, 24.0)
        if ev_std is None: continue
        p["size"] = SIZE_FIXED; p["max_conc"] = CONC_FIXED
        nov = novelty_distance(p)
        results.append({
            "params": p, "novelty": nov,
            "std": ev_std, "stress": ev_stress,
        })
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{total} ({(i+1)/total*100:.0f}%) — elapsed {time.time()-t0:.0f}s")

    print(f"Done in {time.time()-t0:.1f}s. {len(results)} valid results.")

    # Filter: PASS basic (WR ≥0.70, all pos, ann ≥$300, n ≥30, both cost levels)
    def passes(r):
        for ev in (r["std"], r["stress"]):
            if not (ev["wr"] >= 0.70 and ev["all_pos"] and ev["annual_pnl"] >= 300 and ev["n"] >= 30):
                return False
        return True

    passers = [r for r in results if passes(r)]
    print(f"\nROBUST PASS (both cost levels): {len(passers)}/{len(results)}")

    # Sort by composite score: annual PnL × WR × novelty^0.5
    for r in passers:
        ev = r["std"]
        r["score"] = ev["annual_pnl"] * ev["wr"] * (1 + r["novelty"] * 0.15)

    passers.sort(key=lambda r: r["score"], reverse=True)

    # Show top 15 robust passers
    print("\n=== TOP 15 ROBUST PASS (sorted by composite annual×WR×novelty) ===")
    print(f"{'rank':<5} {'thr':>4} {'hld':>4} {'atrMn':>5} {'atrMx':>5} {'cnsc':>4} {'univ':<9} {'lev':>4} {'WR%':>5} {'annPnL$':>8} {'mo$':>6} {'liq%':>5} {'nov':>4}")
    for i, r in enumerate(passers[:15], 1):
        p = r["params"]; ev = r["std"]
        print(f"{i:<5} {p['thr']:>4} {p['hold']:>4} {p['atr_min']:>5} {p['atr_max']:>5} {p['consec']:>4} {p['univ']:<9} {p['lev']:>4} {ev['wr']*100:>4.1f}% ${ev['annual_pnl']:>+6.2f} ${ev['monthly_pnl']:>+4.2f} {ev['liq_rate']*100:>4.1f}% {r['novelty']:>4}")

    # Pick 2 BEST NOVEL — top score that have novelty >= 2 (differ from existing in 2+ dims)
    novel_passers = [r for r in passers if r["novelty"] >= 2]
    novel_passers.sort(key=lambda r: r["std"]["annual_pnl"], reverse=True)
    print(f"\n=== TOP 5 NOVEL (novelty >= 2 from existing G400/G710 family) ===")
    for i, r in enumerate(novel_passers[:5], 1):
        p = r["params"]; ev = r["std"]
        print(f"{i:<5} thr={p['thr']} hold={p['hold']} atr_min={p['atr_min']} atr_max={p['atr_max']} consec={p['consec']} univ={p['univ']} lev={p['lev']} | WR {ev['wr']*100:.1f}% ann ${ev['annual_pnl']:.2f} mo ${ev['monthly_pnl']:.2f} liq {ev['liq_rate']*100:.1f}% novelty={r['novelty']}")

    # Save full results
    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g800_grid_results.json"
    out.write_text(json.dumps({
        "n_total": total, "n_valid": len(results), "n_robust_pass": len(passers),
        "top_overall_by_score": passers[:20],
        "top_novel_by_pnl": novel_passers[:10],
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")
    print(f"\nPick suggestion: top 2 novel candidates above for deploy.")


if __name__ == "__main__":
    main()
