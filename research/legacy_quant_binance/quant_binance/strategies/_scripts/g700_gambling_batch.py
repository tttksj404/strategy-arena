"""G700 series — PURE GAMBLING mode, $50 capital.

극단 high-risk high-reward. WR 낮아도 OK, 청산 risk 감수.

Candidates (7종):
  G700: CH1 score >= 90 + lev 20x + size 0.50 (peak 50x notional, super rare)
  G701: CH1 score >= 95 + lev 25x + size 0.30 (peak 37.5x)
  G702: ATR > 10% + score >= 80 + lev 20x + size 0.30 (only volatile)
  G703: Memecoin only (DOGE/PEPE/WIF) + score >= 80 + lev 25x + size 0.30 (peak 37.5x)
  G704: Score >= 80 + size 1.0 + max_conc 1 + lev 10x (all-in single position)
  G705: Score >= 85 + lev 30x + size 0.20 + max_conc 5 (peak 30x)
  G706: Memecoin + score >= 85 + lev 50x + size 0.20 (peak 50x EXTREME)

Relaxed criteria:
  - WR >= 0.30 (lottery OK)
  - annual PnL >= $50 ($50 capital, +100%/yr to justify gambling)
  - min n >= 5
  - liquidation rate accepted (gambling = some -100% loss expected)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

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
MEMECOIN = {"DOGEUSDT","PEPEUSDT","WIFUSDT"}

EQUITY = 50.0
HOLD_BARS = 24

CANDIDATES = [
    {"id":"G700", "desc":"score>=90 + lev20 size0.5",     "params":{"thr":90,"lev":20,"size":0.50,"max_conc":5,"atr":99,"hold":24,"univ_filter":"all"}},
    {"id":"G701", "desc":"score>=95 + lev25 size0.3",     "params":{"thr":95,"lev":25,"size":0.30,"max_conc":5,"atr":99,"hold":24,"univ_filter":"all"}},
    {"id":"G702", "desc":"ATR>=10% + score>=80 + lev20",  "params":{"thr":80,"lev":20,"size":0.30,"max_conc":5,"atr":99,"min_atr":10.0,"hold":24,"univ_filter":"all"}},
    {"id":"G703", "desc":"meme + score>=80 + lev25",      "params":{"thr":80,"lev":25,"size":0.30,"max_conc":5,"atr":99,"hold":24,"univ_filter":"meme"}},
    {"id":"G704", "desc":"score>=80 ALL-IN size1 lev10",  "params":{"thr":80,"lev":10,"size":1.00,"max_conc":1,"atr":99,"hold":24,"univ_filter":"all"}},
    {"id":"G705", "desc":"score>=85 + lev30 size0.2",     "params":{"thr":85,"lev":30,"size":0.20,"max_conc":5,"atr":99,"hold":24,"univ_filter":"all"}},
    {"id":"G706", "desc":"meme + score>=85 + lev50 EXT",  "params":{"thr":85,"lev":50,"size":0.20,"max_conc":3,"atr":99,"hold":24,"univ_filter":"meme"}},
]


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


def filter_universe(base, mode):
    if mode == "meme":
        return [s for s in base if s in MEMECOIN]
    return base


def gather_signals(dfs, params):
    rows = []
    for sym, df in dfs.items():
        score, _ = compute_ch1_score(df)
        df = df.copy()
        df["score"] = score
        df["atr_pct"] = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["fwd_pct"] = (df["close_price"].shift(-params["hold"]) / df["close_price"] - 1) * 10000
        mask = (df["score"] >= params["thr"]) & (df["atr_pct"] <= params["atr"]) & df["fwd_pct"].notna()
        if "min_atr" in params:
            mask = mask & (df["atr_pct"] >= params["min_atr"])
        e = df[mask].copy()
        if len(e) == 0: continue
        e["sym"] = sym
        rows.append(e[["open_time","score","atr_pct","fwd_pct","sym"]])
    return pd.concat(rows).sort_values("open_time").reset_index(drop=True) if rows else pd.DataFrame()


def portfolio_sim(entries, equity, params, days, cost_bps):
    if len(entries) == 0: return None
    HOLD_MS = params["hold"] * 3600 * 1000
    open_pos, pnl_usd, taken, wins = [], 0.0, 0, 0
    big_wins, liquidations = 0, 0
    win_pnl, loss_pnl = [], []
    max_single_loss = 0.0
    max_single_win = 0.0
    for _, row in entries.iterrows():
        ts = row["open_time"]
        sym = row["sym"]
        net_bps = row["fwd_pct"] - cost_bps
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == sym for p in open_pos): continue
        if len(open_pos) >= params["max_conc"]: continue
        margin = equity * params["size"]
        net_pct = net_bps / 10000 * params["lev"]
        if net_pct < -0.90:
            net_pct = -0.90
            liquidations += 1
        trade_pnl = margin * net_pct
        pnl_usd += trade_pnl; taken += 1
        if net_pct > 0: wins += 1; win_pnl.append(trade_pnl)
        else: loss_pnl.append(trade_pnl)
        if net_pct > 0.30: big_wins += 1
        max_single_loss = min(max_single_loss, trade_pnl)
        max_single_win = max(max_single_win, trade_pnl)
        open_pos.append((ts + HOLD_MS, sym))
    return {
        "n_taken": taken,
        "win_rate": round(wins/taken, 4) if taken else 0,
        "pnl_usd": round(pnl_usd, 2),
        "annual_pnl_usd": round(pnl_usd/days*365, 2),
        "monthly_pnl_usd": round(pnl_usd/days*30.4, 2),
        "max_single_win_usd": round(max_single_win, 2),
        "max_single_loss_usd": round(max_single_loss, 2),
        "big_wins_30pct": big_wins,
        "liquidations": liquidations,
        "liquidation_rate": round(liquidations/taken, 4) if taken else 0,
        "avg_winner_usd": round(np.mean(win_pnl), 2) if win_pnl else 0,
        "avg_loser_usd": round(np.mean(loss_pnl), 2) if loss_pnl else 0,
    }


def evaluate(params, cost_bps):
    periods = [
        ("OOS22-23", filter_universe(UNIV_22, params["univ_filter"]), DATA_22, 730),
        ("OOS24-Q1", filter_universe(UNIV_24, params["univ_filter"]), DATA_24, 456),
        ("IS25-26",  filter_universe(UNIV_25, params["univ_filter"]), DATA_25, 374),
    ]
    res = {}
    for plabel, univ, dpath, days in periods:
        if not univ:
            res[plabel] = {"days": days, "n_taken": 0, "note":"empty universe (memecoin)"}; continue
        dfs = load_period_dfs(dpath, univ)
        ent = gather_signals(dfs, params)
        r = portfolio_sim(ent, EQUITY, params, days, cost_bps)
        if r is None:
            res[plabel] = {"days": days, "n_taken": 0}
        else:
            r["days"] = days; res[plabel] = r
    valid = [r for r in res.values() if r.get("n_taken", 0) > 0]
    total_days = sum(r.get("days", 0) for r in valid)
    total_taken = sum(r.get("n_taken", 0) for r in valid)
    total_pnl = sum(r.get("pnl_usd", 0) for r in valid)
    total_liq = sum(r.get("liquidations", 0) for r in valid)
    total_big_wins = sum(r.get("big_wins_30pct", 0) for r in valid)
    weighted_wr = sum(r["win_rate"] * r["n_taken"] for r in valid) / max(total_taken, 1)
    annual_pnl = total_pnl / total_days * 365 if total_days else 0
    all_pos = all(r.get("pnl_usd", 0) > 0 for r in valid) and len(valid) == 3
    max_single_loss = min((r.get("max_single_loss_usd", 0) for r in valid), default=0)
    max_single_win = max((r.get("max_single_win_usd", 0) for r in valid), default=0)
    return {
        "periods": res,
        "weighted": {
            "n_taken": total_taken,
            "win_rate": round(weighted_wr, 4),
            "pnl_usd": round(total_pnl, 2),
            "annual_pnl_usd": round(annual_pnl, 2),
            "monthly_pnl_usd": round(annual_pnl/12, 2),
            "all_periods_positive": all_pos,
            "liquidations": total_liq,
            "liquidation_rate": round(total_liq/max(total_taken,1), 4),
            "big_wins_30pct": total_big_wins,
            "max_single_win_usd": round(max_single_win, 2),
            "max_single_loss_usd": round(max_single_loss, 2),
        }
    }


def decide(weighted):
    """Gambling-mode criteria: relaxed."""
    return {"verdict": "PASS" if (
        weighted["win_rate"] >= 0.30 and
        weighted["annual_pnl_usd"] >= 50 and  # $50 capital, +100%/yr min
        weighted["n_taken"] >= 5
    ) else "FAIL"}


def main():
    print("=" * 175)
    print(f"$50 capital | GAMBLING MODE | hold {HOLD_BARS}h")
    print("=" * 175)
    print(f"{'ID':<5} {'desc':<32} {'cost':>4} {'trades':>6} {'WR':>6} {'PnL$':>9} {'annPnL$':>9} {'mo$':>6} {'big30%':>6} {'maxWin$':>8} {'maxLoss$':>9} {'liq':>4} {'liq%':>5} {'allPos':>7} {'verdict':>8}")
    print("-" * 175)
    summary = {}
    for cand in CANDIDATES:
        results = {}
        for cost_label, cost in [("std16", 16.0), ("stress24", 24.0)]:
            ev = evaluate(cand["params"], cost)
            d = decide(ev["weighted"])
            w = ev["weighted"]
            print(f"{cand['id']:<5} {cand['desc']:<32} {cost:>4.0f} {w['n_taken']:>6} {w['win_rate']*100:>5.1f}% {w['pnl_usd']:>+8.2f} {w['annual_pnl_usd']:>+8.2f} {w['monthly_pnl_usd']:>+5.2f} {w['big_wins_30pct']:>6} {w['max_single_win_usd']:>+7.2f} {w['max_single_loss_usd']:>+8.2f} {w['liquidations']:>4} {w['liquidation_rate']*100:>4.1f}% {('Y' if w['all_periods_positive'] else 'N'):>7} {d['verdict']:>8}")
            results[cost_label] = {"evaluation": ev, "decision": d}
        robust = results["std16"]["decision"]["verdict"] == "PASS" and results["stress24"]["decision"]["verdict"] == "PASS"
        results["robust_PASS"] = robust
        summary[cand["id"]] = {**cand, "results": results}
        print()

    # Sort by annual PnL
    print("\n" + "=" * 90)
    print("RANK by annual PnL (gambling = relaxed criteria)")
    print("=" * 90)
    ranked = sorted(summary.items(), key=lambda x: x[1]["results"]["std16"]["evaluation"]["weighted"]["annual_pnl_usd"], reverse=True)
    for sid, s in ranked:
        w = s["results"]["std16"]["evaluation"]["weighted"]
        ws = s["results"]["stress24"]["evaluation"]["weighted"]
        v = s["results"]["std16"]["decision"]["verdict"]
        v2 = s["results"]["stress24"]["decision"]["verdict"]
        mark = "🎰" if (v == "PASS" and v2 == "PASS") else "⚠️" if v == "PASS" else "❌"
        print(f"  {mark} {sid} ({s['desc']}): std=${w['annual_pnl_usd']:+.2f}/yr WR{w['win_rate']*100:.1f}% n={w['n_taken']} liq{w['liquidation_rate']*100:.1f}% maxWin${w['max_single_win_usd']:+.2f} maxLoss${w['max_single_loss_usd']:+.2f} | std={v} stress={v2}")

    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g700_gambling_summary.json"
    out.write_text(json.dumps({
        "candidates": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
