"""G600 series — SHORT ONLY with filters, $50 capital.

목표: 적당한 빈도 + 양수 PnL + short only.

기본 가설: 단순 RSI/BB 거꾸로 = 모두 fail (G500 입증).
시도: 필터 추가 (변동성, regime, hold 시간, 펀딩 컨텍스트).

Candidates (7종):
  G600: BB upper touch + ATR>5% (high vol short)
  G610: RSI>75 + hold 24h (long hold for trend follow)
  G620: BB+RSI+bearish candle (3-condition combo)
  G630: BB upper + Stoch overbought (double extreme)
  G640: BTC 7d down + alt RSI>75 (regime gate)
  G650: CH1 score ≤30 (inverse PB001, with hold 24h)
  G660: Strong rally + RSI extreme (squeeze sell — 4h price > +5% AND RSI>80)

각 cost 16/24/32 stress (short = funding cost 추가 의식).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
SCRIPTS = ROOT / "quant_binance" / "strategies" / "_scripts"
sys.path.insert(0, str(SCRIPTS))
from g002_mingogogo_ch1_backtest import (
    rsi, mfi, stoch_k, bbands_pct, macd_hist, atr_pct, compute_ch1_score, COST_BPS_RT
)  # type: ignore

DATA_22 = ROOT / "quant_runtime" / "historical_2022"
DATA_24 = ROOT / "quant_runtime" / "historical_2024"
DATA_25 = ROOT / "quant_runtime" / "historical_is25"

UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT"]

EQUITY = 50.0
SIZE_PCT = 0.10
LEVERAGE = 5.0
MAX_CONC = 5

CANDIDATES = [
    {"id":"G600", "desc":"BB upper + ATR>5%",        "signal":"bb_atr",       "hold":6,  "filters":{"min_atr":5.0}},
    {"id":"G610", "desc":"RSI>75 + 24h hold",         "signal":"rsi75_24h",    "hold":24, "filters":{}},
    {"id":"G620", "desc":"BB+RSI+bearish candle",     "signal":"bb_rsi_bear",  "hold":6,  "filters":{}},
    {"id":"G630", "desc":"BB upper + Stoch>90",       "signal":"bb_stoch",     "hold":6,  "filters":{}},
    {"id":"G640", "desc":"BTC 7d down + alt RSI>75",  "signal":"regime_short", "hold":12, "filters":{}},
    {"id":"G650", "desc":"CH1 score<=30 + 24h",       "signal":"ch1_inverse",  "hold":24, "filters":{}},
    {"id":"G660", "desc":"Rally squeeze sell",        "signal":"squeeze",      "hold":6,  "filters":{}},
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


def get_btc_trend(data_dir):
    """Return DataFrame with BTC price + 7d momentum (for regime gate)."""
    p = data_dir / "BTCUSDT" / "1h.json"
    if not p.exists(): return None
    data = json.loads(p.read_text())
    if isinstance(data, list):
        data.sort(key=lambda b: b["open_time"]); df = pd.DataFrame(data)
    else: df = pd.DataFrame(data)
    for c in ("open_price","high_price","low_price","close_price","base_volume","quote_volume"):
        if c in df.columns: df[c] = df[c].astype(float)
    df["btc_7d_mom"] = (df["close_price"] / df["close_price"].shift(7*24) - 1) * 100
    return df[["open_time","btc_7d_mom"]].set_index("open_time")["btc_7d_mom"]


def gen_short_signals(df, signal_type, hold_bars, filters, btc_mom=None):
    n = len(df)
    if n < 100: return pd.DataFrame()
    h = df["high_price"]; l = df["low_price"]; c = df["close_price"]; o = df["open_price"]
    fwd_close = c.shift(-hold_bars)
    fwd_pct = (fwd_close / c - 1) * 10000
    short_mask = pd.Series(False, index=df.index)

    if signal_type == "bb_atr":
        bbp = bbands_pct(c)
        atr = atr_pct(h, l, c, 14)
        short_mask = (bbp >= 0.95) & (atr >= filters.get("min_atr", 0))
    elif signal_type == "rsi75_24h":
        r = rsi(c)
        short_mask = (r >= 75)
    elif signal_type == "bb_rsi_bear":
        bbp = bbands_pct(c); r = rsi(c)
        bear_candle = c < o  # red candle
        short_mask = (bbp >= 0.95) & (r >= 75) & bear_candle
    elif signal_type == "bb_stoch":
        bbp = bbands_pct(c); sk = stoch_k(h, l, c)
        short_mask = (bbp >= 0.95) & (sk >= 90)
    elif signal_type == "regime_short":
        r = rsi(c)
        if btc_mom is None: return pd.DataFrame()
        # need BTC 7d down at this timestamp
        bm = df["open_time"].map(btc_mom).fillna(0)
        short_mask = (r >= 75) & (bm < 0)
    elif signal_type == "ch1_inverse":
        score, _ = compute_ch1_score(df)
        short_mask = (score <= 30)
    elif signal_type == "squeeze":
        # 4h ago price comparison + RSI extreme
        ret_4h = (c / c.shift(4) - 1) * 100  # past 4h return
        r = rsi(c)
        short_mask = (ret_4h >= 5.0) & (r >= 80)

    rows = []
    for i in range(n):
        if pd.isna(fwd_pct.iloc[i]): continue
        if short_mask.iloc[i]:
            rows.append({"open_time": int(df["open_time"].iloc[i]), "gross_bps": -float(fwd_pct.iloc[i])})
    return pd.DataFrame(rows)


def gather_all_signals(dfs, signal_type, hold_bars, filters, btc_mom=None):
    all_rows = []
    for sym, df in dfs.items():
        sig = gen_short_signals(df, signal_type, hold_bars, filters, btc_mom)
        if len(sig) == 0: continue
        sig["sym"] = sym
        all_rows.append(sig)
    if not all_rows: return pd.DataFrame()
    return pd.concat(all_rows).sort_values("open_time").reset_index(drop=True)


def portfolio_sim(entries, equity, size_pct, leverage, hold_bars, max_conc, days, cost_bps):
    if len(entries) == 0: return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos, pnl_usd, taken, wins = [], 0.0, 0, 0
    for _, row in entries.iterrows():
        ts = row["open_time"]
        sym = row["sym"]
        net_bps = row["gross_bps"] - cost_bps
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == sym for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        margin = equity * size_pct
        net_pct = net_bps / 10000 * leverage
        if net_pct < -0.90: net_pct = -0.90
        trade_pnl = margin * net_pct
        pnl_usd += trade_pnl; taken += 1
        if net_pct > 0: wins += 1
        open_pos.append((ts + HOLD_MS, sym))
    return {
        "n_taken": taken,
        "win_rate": round(wins/taken, 4) if taken else 0,
        "pnl_usd": round(pnl_usd, 2),
        "annual_pnl_usd": round(pnl_usd/days*365, 2),
        "monthly_pnl_usd": round(pnl_usd/days*30.4, 2),
        "avg_per_trade_usd": round(pnl_usd/max(taken,1), 4),
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
        btc_mom = get_btc_trend(dpath)
        ent = gather_all_signals(dfs, cand["signal"], cand["hold"], cand["filters"], btc_mom)
        r = portfolio_sim(ent, EQUITY, SIZE_PCT, LEVERAGE, cand["hold"], MAX_CONC, days, cost_bps)
        if r is None:
            res[plabel] = {"days": days, "n_taken": 0}
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
            "avg_per_trade_usd": round(total_pnl/max(total_taken,1), 4),
        }
    }


def decide(weighted):
    return {"verdict": "PASS" if (
        weighted["win_rate"] >= 0.50 and
        weighted["all_periods_positive"] and
        weighted["annual_pnl_usd"] >= 5 and  # any positive (relaxed for short)
        weighted["n_taken"] >= 50
    ) else "FAIL"}


def main():
    print("=" * 145)
    print(f"$50 capital | size {SIZE_PCT} | lev {LEVERAGE}x | max_conc {MAX_CONC} | SHORT ONLY")
    print("=" * 145)
    print(f"{'ID':<5} {'desc':<32} {'cost':>5} {'hold':>5} {'trades':>6} {'WR':>6} {'PnL$':>8} {'annPnL$':>8} {'mo$':>6} {'avg/tr$':>8} {'allPos':>7} {'verdict':>8}")
    print("-" * 145)
    summary = {}
    for cand in CANDIDATES:
        results = {}
        for cost_label, cost in [("std16", 16.0), ("stress24", 24.0), ("funding32", 32.0)]:
            ev = evaluate(cand, cost)
            d = decide(ev["weighted"])
            w = ev["weighted"]
            print(f"{cand['id']:<5} {cand['desc']:<32} {cost:>5.0f} {cand['hold']:>5} {w['n_taken']:>6} {w['win_rate']*100:>5.1f}% {w['pnl_usd']:>+7.2f} {w['annual_pnl_usd']:>+7.2f} {w['monthly_pnl_usd']:>+5.2f} {w['avg_per_trade_usd']:>+7.3f} {('Y' if w['all_periods_positive'] else 'N'):>7} {d['verdict']:>8}")
            results[cost_label] = {"evaluation": ev, "decision": d}
        # passed only if all 3 cost levels pass (very robust)
        robust = all(r["decision"]["verdict"] == "PASS" for r in [results["std16"], results["stress24"], results["funding32"]])
        results["robust_PASS"] = robust
        summary[cand["id"]] = {**cand, "results": results}
        print()

    robust_passers = [(sid, s) for sid, s in summary.items() if s["results"]["robust_PASS"]]
    std_passers = [(sid, s) for sid, s in summary.items() if s["results"]["std16"]["decision"]["verdict"] == "PASS"]
    print()
    print("=" * 90)
    print(f"std16 PASS: {len(std_passers)}/{len(CANDIDATES)} | ROBUST (3 cost levels): {len(robust_passers)}/{len(CANDIDATES)}")
    print("=" * 90)
    if std_passers:
        std_passers.sort(key=lambda x: x[1]["results"]["std16"]["evaluation"]["weighted"]["annual_pnl_usd"], reverse=True)
        print("\nstd16 PASS candidates:")
        for sid, s in std_passers:
            w = s["results"]["std16"]["evaluation"]["weighted"]
            ws = s["results"]["stress24"]["evaluation"]["weighted"]
            wf = s["results"]["funding32"]["evaluation"]["weighted"]
            mark = "🏆" if s["results"]["robust_PASS"] else "⚠️"
            print(f"  {mark} {sid} ({s['desc']}): std=${w['annual_pnl_usd']:.2f}/yr WR{w['win_rate']*100:.1f}% n={w['n_taken']} | stress24=${ws['annual_pnl_usd']:.2f} | fund32=${wf['annual_pnl_usd']:.2f}")

    out = ROOT / "quant_binance" / "strategies" / "_scripts" / "g600_short_only_summary.json"
    out.write_text(json.dumps({
        "candidates": summary,
        "robust_passers": [sid for sid, _ in robust_passers],
        "std_passers": [sid for sid, _ in std_passers],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
