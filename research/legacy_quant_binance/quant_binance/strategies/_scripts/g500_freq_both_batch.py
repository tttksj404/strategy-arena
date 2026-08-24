"""G500 series — HIGH FREQUENCY + LONG/SHORT (both directions), $50 capital.

새 paradigm:
  - 자본 $50
  - 빈도 ↑ (수수료 극복 시도)
  - long + short 둘 다
  - 거래당 expectancy 작아도 OK (volume 으로 누적)

Candidates (양방향 신호 7종):
  G500: BB %B 극값 (BB%B<5% long, BB%B>95% short)
  G510: RSI 극값 (<25 long, >75 short)
  G520: BB + RSI 동시 (확신도 ↑, n ↓)
  G530: Stochastic K 극값 (<10 long, >90 short)
  G540: MACD hist cross (positive cross long, negative cross short)
  G550: Z-score MA20 reversion (|z|>2)
  G560: Donchian breakout (HH20 break long, LL20 break short, trend-follow)

각 cost 16bps + 24bps stress.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
SCRIPTS = ROOT / "quant_binance" / "strategies" / "_scripts"
sys.path.insert(0, str(SCRIPTS))
from g002_mingogogo_ch1_backtest import (
    rsi, mfi, stoch_k, bbands_pct, macd_hist, atr_pct, COST_BPS_RT
)  # type: ignore

DATA_22 = ROOT / "quant_runtime" / "historical_2022"
DATA_24 = ROOT / "quant_runtime" / "historical_2024"
DATA_25 = ROOT / "quant_runtime" / "historical_is25"

UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"]
# Note: drop WIF/LTC/BTC from active 25 universe (G210 lesson: dead weight)

EQUITY = 50.0
SIZE_PCT = 0.10  # $5 margin per trade
LEVERAGE = 5.0
MAX_CONC = 5
HOLD_BARS = 6

CANDIDATES = [
    {"id":"G500", "desc":"BB%B 5/95",       "signal":"bb_extreme",     "thr_low":0.05, "thr_high":0.95, "hold":6},
    {"id":"G510", "desc":"RSI 25/75",       "signal":"rsi_extreme",    "thr_low":25,    "thr_high":75,    "hold":6},
    {"id":"G520", "desc":"BB+RSI combo",    "signal":"bb_rsi_combo",   "thr_low":0.05,  "thr_high":0.95,  "hold":6},
    {"id":"G530", "desc":"Stoch K 10/90",   "signal":"stoch_extreme",  "thr_low":10,    "thr_high":90,    "hold":4},
    {"id":"G540", "desc":"MACD hist cross", "signal":"macd_cross",     "thr_low":0,     "thr_high":0,     "hold":6},
    {"id":"G550", "desc":"Z-score |z|>2",   "signal":"zscore",         "thr_low":-2.0,  "thr_high":2.0,   "hold":4},
    {"id":"G560", "desc":"Donchian 20",     "signal":"donchian",       "thr_low":0,     "thr_high":0,     "hold":12},
]

DECISION_CRITERIA = {
    "min_wr":              0.50,
    "all_periods_positive": True,
    "min_annual_pnl_usd":  10,    # $50 capital, +20%/yr OK
    "min_trades_total":    100,   # frequency requirement
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


def gen_signals(df, signal_type, thr_low, thr_high, hold_bars):
    """Returns DataFrame with columns: open_time, sym (set later), side (long/short), gross_bps."""
    n = len(df)
    if n < 50: return pd.DataFrame()

    h = df["high_price"]; l = df["low_price"]; c = df["close_price"]; v = df["base_volume"]
    fwd_close = c.shift(-hold_bars)
    fwd_pct = (fwd_close / c - 1) * 10000

    long_mask = pd.Series(False, index=df.index)
    short_mask = pd.Series(False, index=df.index)

    if signal_type == "bb_extreme":
        bbp = bbands_pct(c)
        long_mask = bbp <= thr_low
        short_mask = bbp >= thr_high
    elif signal_type == "rsi_extreme":
        r = rsi(c)
        long_mask = r <= thr_low
        short_mask = r >= thr_high
    elif signal_type == "bb_rsi_combo":
        bbp = bbands_pct(c); r = rsi(c)
        long_mask = (bbp <= thr_low) & (r <= 30)
        short_mask = (bbp >= thr_high) & (r >= 70)
    elif signal_type == "stoch_extreme":
        sk = stoch_k(h, l, c)
        long_mask = sk <= thr_low
        short_mask = sk >= thr_high
    elif signal_type == "macd_cross":
        m = macd_hist(c)
        prev = m.shift(1)
        long_mask = (m > 0) & (prev <= 0)   # hist crosses up
        short_mask = (m < 0) & (prev >= 0)  # hist crosses down
    elif signal_type == "zscore":
        ma = c.rolling(20).mean()
        sd = c.rolling(20).std()
        z = (c - ma) / sd.replace(0, np.nan)
        long_mask = z <= thr_low
        short_mask = z >= thr_high
    elif signal_type == "donchian":
        hh = h.rolling(20).max().shift(1)
        ll = l.rolling(20).min().shift(1)
        long_mask = c > hh    # break above 20-bar high
        short_mask = c < ll   # break below 20-bar low

    rows = []
    for i in range(n):
        if pd.isna(fwd_pct.iloc[i]): continue
        if long_mask.iloc[i]:
            rows.append({"open_time": int(df["open_time"].iloc[i]), "side":"long",  "gross_bps": float(fwd_pct.iloc[i])})
        elif short_mask.iloc[i]:
            # short profits when price goes down
            rows.append({"open_time": int(df["open_time"].iloc[i]), "side":"short", "gross_bps": -float(fwd_pct.iloc[i])})
    return pd.DataFrame(rows)


def gather_all_signals(dfs, signal_type, thr_low, thr_high, hold_bars):
    all_rows = []
    for sym, df in dfs.items():
        sig = gen_signals(df, signal_type, thr_low, thr_high, hold_bars)
        if len(sig) == 0: continue
        sig["sym"] = sym
        all_rows.append(sig)
    if not all_rows: return pd.DataFrame()
    return pd.concat(all_rows).sort_values("open_time").reset_index(drop=True)


def portfolio_sim(entries, equity, size_pct, leverage, hold_bars, max_conc, days, cost_bps):
    if len(entries) == 0: return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos, pnl_usd, taken, wins = [], 0.0, 0, 0
    win_pnl, loss_pnl = [], []
    long_n, short_n, long_pnl, short_pnl = 0, 0, 0.0, 0.0
    for _, row in entries.iterrows():
        ts = row["open_time"]
        sym = row["sym"]
        side = row["side"]
        net_bps = row["gross_bps"] - cost_bps
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == sym for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        margin = equity * size_pct
        net_pct = net_bps / 10000 * leverage
        if net_pct < -0.90: net_pct = -0.90
        trade_pnl = margin * net_pct
        pnl_usd += trade_pnl; taken += 1
        if side == "long": long_n += 1; long_pnl += trade_pnl
        else: short_n += 1; short_pnl += trade_pnl
        if net_pct > 0: wins += 1; win_pnl.append(trade_pnl)
        else: loss_pnl.append(trade_pnl)
        open_pos.append((ts + HOLD_MS, sym))
    return {
        "n_taken": taken,
        "win_rate": round(wins/taken, 4) if taken else 0,
        "pnl_usd": round(pnl_usd, 2),
        "annual_pnl_usd": round(pnl_usd/days*365, 2),
        "monthly_pnl_usd": round(pnl_usd/days*30.4, 2),
        "avg_per_trade_usd": round(pnl_usd/max(taken,1), 4),
        "long_n": long_n, "short_n": short_n,
        "long_pnl_usd": round(long_pnl, 2),
        "short_pnl_usd": round(short_pnl, 2),
    }


def evaluate(cand, cost_bps):
    periods = [
        ("OOS22-23", UNIV_22, DATA_22, 730),
        ("OOS24-Q1", UNIV_24, DATA_24, 456),
        ("IS25-26",  UNIV_25, DATA_25, 374),
    ]
    res = {}
    for plabel, univ, dpath, days in periods:
        dfs = load_period_dfs(dpath, univ)
        ent = gather_all_signals(dfs, cand["signal"], cand["thr_low"], cand["thr_high"], cand["hold"])
        r = portfolio_sim(ent, EQUITY, SIZE_PCT, LEVERAGE, cand["hold"], MAX_CONC, days, cost_bps)
        if r is None:
            res[plabel] = {"days": days, "n_taken": 0}
        else:
            r["days"] = days; res[plabel] = r
    valid = [r for r in res.values() if r.get("n_taken", 0) > 0]
    total_days = sum(r.get("days", 0) for r in valid)
    total_taken = sum(r.get("n_taken", 0) for r in valid)
    total_pnl = sum(r.get("pnl_usd", 0) for r in valid)
    total_long = sum(r.get("long_n", 0) for r in valid)
    total_short = sum(r.get("short_n", 0) for r in valid)
    total_long_pnl = sum(r.get("long_pnl_usd", 0) for r in valid)
    total_short_pnl = sum(r.get("short_pnl_usd", 0) for r in valid)
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
            "long_n": total_long, "short_n": total_short,
            "long_pnl_usd": round(total_long_pnl, 2),
            "short_pnl_usd": round(total_short_pnl, 2),
        }
    }


def decide(weighted):
    c = DECISION_CRITERIA
    checks = {
        "wr_>=_50%":          weighted["win_rate"] >= c["min_wr"],
        "all_periods_pos":    weighted["all_periods_positive"],
        "annual_pnl_>=_$10":  weighted["annual_pnl_usd"] >= c["min_annual_pnl_usd"],
        "trades_>=_100":      weighted["n_taken"] >= c["min_trades_total"],
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def main():
    print("=" * 160)
    print(f"$50 capital | size {SIZE_PCT} | lev {LEVERAGE}x | max_conc {MAX_CONC} | both long+short")
    print("=" * 160)
    print(f"{'ID':<5} {'desc':<22} {'cost':>5} {'trades':>6} {'long/short':>11} {'WR':>6} {'long$':>8} {'short$':>8} {'PnL$':>9} {'annPnL$':>9} {'mo$':>6} {'avg/tr':>7} {'allPos':>7} {'verdict':>8}")
    print("-" * 160)
    summary = {}
    for cand in CANDIDATES:
        results = {}
        for cost_label, cost in [("std16", 16.0), ("stress24", 24.0)]:
            ev = evaluate(cand, cost)
            d = decide(ev["weighted"])
            w = ev["weighted"]
            ls = f"{w['long_n']}/{w['short_n']}"
            print(f"{cand['id']:<5} {cand['desc']:<22} {cost:>5.0f} {w['n_taken']:>6} {ls:>11} {w['win_rate']*100:>5.1f}% {w['long_pnl_usd']:>+7.2f} {w['short_pnl_usd']:>+7.2f} {w['pnl_usd']:>+8.2f} {w['annual_pnl_usd']:>+8.2f} {w['monthly_pnl_usd']:>+5.2f} ${(w['pnl_usd']/max(w['n_taken'],1)):>+5.3f} {('Y' if w['all_periods_positive'] else 'N'):>7} {d['verdict']:>8}")
            results[cost_label] = {"evaluation": ev, "decision": d}
        robust = results["std16"]["decision"]["verdict"] == "PASS" and results["stress24"]["decision"]["verdict"] == "PASS"
        results["robust_PASS"] = robust
        summary[cand["id"]] = {**cand, "results": results}
        print()

    robust_passers = [(sid, s) for sid, s in summary.items() if s["results"]["robust_PASS"]]
    print()
    print("=" * 90)
    print(f"ROBUST PASS: {len(robust_passers)} / {len(CANDIDATES)}")
    print("=" * 90)
    if robust_passers:
        robust_passers.sort(key=lambda x: x[1]["results"]["std16"]["evaluation"]["weighted"]["annual_pnl_usd"], reverse=True)
        for sid, s in robust_passers:
            w = s["results"]["std16"]["evaluation"]["weighted"]
            print(f"  🏆 {sid} ({s['desc']}): WR {w['win_rate']*100:.1f}% / ann ${w['annual_pnl_usd']:.2f} / mo ${w['monthly_pnl_usd']:.2f} / n={w['n_taken']} (long {w['long_n']} ${w['long_pnl_usd']:+.2f} | short {w['short_n']} ${w['short_pnl_usd']:+.2f})")

    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g500_freq_both_summary.json"
    out.write_text(json.dumps({
        "criteria": DECISION_CRITERIA,
        "params": {"equity":EQUITY,"size":SIZE_PCT,"lev":LEVERAGE,"max_conc":MAX_CONC},
        "candidates": summary,
        "robust_passers": [sid for sid, _ in robust_passers],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
