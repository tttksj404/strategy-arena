"""G710 series — TRUE MAX GAMBLE: size 1.0 + max_conc 1 + extreme leverage.

각 진입 = 자본 $50 통째로 1 position 에 투입. 새 시그널은 보유 중이면 skip.

Candidates (7종) — leverage scan 20x to Bitget max 75x:
  G710: meme + score>=80 + lev 20x + size 1.0 + conc 1 (peak 20x = $1000)
  G711: meme + score>=80 + lev 30x + size 1.0 + conc 1 (peak 30x = $1500)
  G712: meme + score>=80 + lev 50x + size 1.0 + conc 1 (peak 50x = $2500, liq at -2%)
  G713: meme + score>=80 + lev 75x + size 1.0 + conc 1 (peak 75x = $3750, BITGET MAX)
  G714: ALL alts + score>=80 + lev 25x + size 1.0 + conc 1 (broader uni)
  G715: ALL alts + score>=80 + lev 50x + size 1.0 + conc 1
  G716: ALL alts + score>=80 + lev 75x + size 1.0 + conc 1

Liquidation 위협 큼. Track maxLoss (1번 trade 최대 손실).
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
SIZE = 1.0
MAX_CONC = 1
ATR_GUARD = 8.0
THR = 80

CANDIDATES = [
    {"id":"G710", "desc":"meme + lev20 size1.0 conc1",   "lev":20, "univ":"meme"},
    {"id":"G711", "desc":"meme + lev30 size1.0 conc1",   "lev":30, "univ":"meme"},
    {"id":"G712", "desc":"meme + lev50 size1.0 conc1",   "lev":50, "univ":"meme"},
    {"id":"G713", "desc":"meme + lev75 size1.0 conc1",   "lev":75, "univ":"meme"},
    {"id":"G714", "desc":"all + lev25 size1.0 conc1",    "lev":25, "univ":"all"},
    {"id":"G715", "desc":"all + lev50 size1.0 conc1",    "lev":50, "univ":"all"},
    {"id":"G716", "desc":"all + lev75 size1.0 conc1",    "lev":75, "univ":"all"},
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
    if mode == "meme": return [s for s in base if s in MEMECOIN]
    return base


def gather_signals(dfs, thr, hold_bars, atr_guard):
    rows = []
    for sym, df in dfs.items():
        score, _ = compute_ch1_score(df)
        df = df.copy()
        df["score"] = score
        df["atr_pct"] = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["fwd_pct"] = (df["close_price"].shift(-hold_bars) / df["close_price"] - 1) * 10000
        mask = (df["score"] >= thr) & (df["atr_pct"] <= atr_guard) & df["fwd_pct"].notna()
        e = df[mask].copy()
        if len(e) == 0: continue
        e["sym"] = sym
        rows.append(e[["open_time","score","atr_pct","fwd_pct","sym"]])
    return pd.concat(rows).sort_values("open_time").reset_index(drop=True) if rows else pd.DataFrame()


def portfolio_sim(entries, equity, leverage, hold_bars, days, cost_bps):
    if len(entries) == 0: return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos = None  # max_conc 1 = single position at a time
    pnl_usd, taken, wins, liquidations, big_wins = 0.0, 0, 0, 0, 0
    win_pnl, loss_pnl = [], []
    max_single_loss, max_single_win = 0.0, 0.0
    skipped_due_to_open = 0
    for _, row in entries.iterrows():
        ts = row["open_time"]
        sym = row["sym"]
        if open_pos and open_pos[0] > ts:
            skipped_due_to_open += 1
            continue
        net_bps = row["fwd_pct"] - cost_bps
        margin = equity * SIZE  # = $50
        net_pct = net_bps / 10000 * leverage
        if net_pct < -0.99:
            net_pct = -0.99  # near-total wipeout (margin call eats all)
            liquidations += 1
        elif net_pct < -0.50:
            liquidations += 1  # half+ loss = effectively liquidated
        trade_pnl = margin * net_pct
        pnl_usd += trade_pnl; taken += 1
        if net_pct > 0: wins += 1; win_pnl.append(trade_pnl)
        else: loss_pnl.append(trade_pnl)
        if net_pct > 0.30: big_wins += 1
        max_single_loss = min(max_single_loss, trade_pnl)
        max_single_win = max(max_single_win, trade_pnl)
        open_pos = (ts + HOLD_MS, sym)
    return {
        "n_taken": taken,
        "n_skipped": skipped_due_to_open,
        "win_rate": round(wins/taken, 4) if taken else 0,
        "pnl_usd": round(pnl_usd, 2),
        "annual_pnl_usd": round(pnl_usd/days*365, 2),
        "monthly_pnl_usd": round(pnl_usd/days*30.4, 2),
        "max_single_win_usd": round(max_single_win, 2),
        "max_single_loss_usd": round(max_single_loss, 2),
        "big_wins_30pct": big_wins,
        "liquidations": liquidations,
        "liquidation_rate": round(liquidations/taken, 4) if taken else 0,
    }


def evaluate(cand, cost_bps):
    periods = [
        ("OOS22-23", filter_universe(UNIV_22, cand["univ"]), DATA_22, 730),
        ("OOS24-Q1", filter_universe(UNIV_24, cand["univ"]), DATA_24, 456),
        ("IS25-26",  filter_universe(UNIV_25, cand["univ"]), DATA_25, 374),
    ]
    res = {}
    for plabel, univ, dpath, days in periods:
        if not univ:
            res[plabel] = {"days": days, "n_taken": 0}; continue
        dfs = load_period_dfs(dpath, univ)
        ent = gather_signals(dfs, THR, HOLD_BARS, ATR_GUARD)
        r = portfolio_sim(ent, EQUITY, cand["lev"], HOLD_BARS, days, cost_bps)
        if r is None:
            res[plabel] = {"days": days, "n_taken": 0}
        else:
            r["days"] = days; res[plabel] = r
    valid = [r for r in res.values() if r.get("n_taken", 0) > 0]
    total_days = sum(r.get("days", 0) for r in valid)
    total_taken = sum(r.get("n_taken", 0) for r in valid)
    total_skipped = sum(r.get("n_skipped", 0) for r in valid)
    total_pnl = sum(r.get("pnl_usd", 0) for r in valid)
    total_liq = sum(r.get("liquidations", 0) for r in valid)
    total_big = sum(r.get("big_wins_30pct", 0) for r in valid)
    weighted_wr = sum(r["win_rate"] * r["n_taken"] for r in valid) / max(total_taken, 1)
    annual_pnl = total_pnl / total_days * 365 if total_days else 0
    all_pos = all(r.get("pnl_usd", 0) > 0 for r in valid) and len(valid) == 3
    max_single_loss = min((r.get("max_single_loss_usd", 0) for r in valid), default=0)
    max_single_win = max((r.get("max_single_win_usd", 0) for r in valid), default=0)
    return {
        "periods": res,
        "weighted": {
            "n_taken": total_taken,
            "n_skipped": total_skipped,
            "win_rate": round(weighted_wr, 4),
            "pnl_usd": round(total_pnl, 2),
            "annual_pnl_usd": round(annual_pnl, 2),
            "monthly_pnl_usd": round(annual_pnl/12, 2),
            "all_periods_positive": all_pos,
            "liquidations": total_liq,
            "liquidation_rate": round(total_liq/max(total_taken,1), 4),
            "big_wins_30pct": total_big,
            "max_single_win_usd": round(max_single_win, 2),
            "max_single_loss_usd": round(max_single_loss, 2),
        }
    }


def decide(weighted):
    return {"verdict": "PASS" if (
        weighted["win_rate"] >= 0.30 and
        weighted["annual_pnl_usd"] >= 50 and
        weighted["n_taken"] >= 5
    ) else "FAIL"}


def main():
    print("=" * 175)
    print(f"$50 capital | TRUE MAX GAMBLE | size 1.0 | max_conc 1 | hold 24h | thr 80 | atr<=8%")
    print("=" * 175)
    print(f"{'ID':<5} {'desc':<32} {'cost':>4} {'taken':>5} {'skip':>5} {'WR':>6} {'PnL$':>9} {'annPnL$':>9} {'mo$':>6} {'big30%':>6} {'maxWin$':>9} {'maxLoss$':>10} {'liq':>4} {'liq%':>5} {'allPos':>7} {'verdict':>8}")
    print("-" * 175)
    summary = {}
    for cand in CANDIDATES:
        results = {}
        for cost_label, cost in [("std16", 16.0), ("stress24", 24.0)]:
            ev = evaluate(cand, cost)
            d = decide(ev["weighted"])
            w = ev["weighted"]
            print(f"{cand['id']:<5} {cand['desc']:<32} {cost:>4.0f} {w['n_taken']:>5} {w['n_skipped']:>5} {w['win_rate']*100:>5.1f}% {w['pnl_usd']:>+8.2f} {w['annual_pnl_usd']:>+8.2f} {w['monthly_pnl_usd']:>+5.2f} {w['big_wins_30pct']:>6} {w['max_single_win_usd']:>+8.2f} {w['max_single_loss_usd']:>+9.2f} {w['liquidations']:>4} {w['liquidation_rate']*100:>4.1f}% {('Y' if w['all_periods_positive'] else 'N'):>7} {d['verdict']:>8}")
            results[cost_label] = {"evaluation": ev, "decision": d}
        robust = results["std16"]["decision"]["verdict"] == "PASS" and results["stress24"]["decision"]["verdict"] == "PASS"
        results["robust_PASS"] = robust
        summary[cand["id"]] = {**cand, "results": results}
        print()

    # Rank
    print("\n" + "=" * 90)
    print("🎰 RANK (by annual PnL, gambling criteria)")
    print("=" * 90)
    ranked = sorted(summary.items(), key=lambda x: x[1]["results"]["std16"]["evaluation"]["weighted"]["annual_pnl_usd"], reverse=True)
    for sid, s in ranked:
        w = s["results"]["std16"]["evaluation"]["weighted"]
        v = s["results"]["std16"]["decision"]["verdict"]
        v2 = s["results"]["stress24"]["decision"]["verdict"]
        mark = "🎰" if (v == "PASS" and v2 == "PASS") else "⚠️" if v == "PASS" else "❌"
        print(f"  {mark} {sid} ({s['desc']}): ann ${w['annual_pnl_usd']:+.2f} mo ${w['monthly_pnl_usd']:+.2f} WR{w['win_rate']*100:.1f}% n={w['n_taken']} liq{w['liquidation_rate']*100:.1f}% maxWin${w['max_single_win_usd']:+.2f} maxLoss${w['max_single_loss_usd']:+.2f}")

    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g710_max_gamble_summary.json"
    out.write_text(json.dumps({
        "candidates": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
