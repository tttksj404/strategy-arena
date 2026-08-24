"""G002 CH1 hyperparameter sweep — threshold × holding period."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import (
    load_klines, compute_ch1_score, SYMBOLS, COST_BPS_RT
)

THRESHOLDS = [70, 80, 90]
HOLDS = [4, 12, 24, 72]  # bars on 1h TF = 4h / 12h / 24h / 72h (PB001 3-day)

OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g002_sweep_result.json"

import pandas as pd

def run(threshold, hold):
    total_trades = 0
    total_net = 0.0
    total_gross = 0.0
    wins = 0
    lottery5 = 0
    lottery10 = 0
    per_sym = {}
    for sym in SYMBOLS:
        df = load_klines(sym)
        if df is None or len(df) < 100:
            continue
        score, _ = compute_ch1_score(df)
        df["score"] = score
        df["fwd_pct"] = (df["close_price"].shift(-hold) / df["close_price"] - 1) * 10000
        e = df[df["score"] >= threshold].dropna(subset=["fwd_pct"])
        if len(e) == 0:
            continue
        net = e["fwd_pct"] - COST_BPS_RT
        total_trades += len(e)
        total_net += net.sum()
        total_gross += e["fwd_pct"].sum()
        wins += (net > 0).sum()
        lottery5 += int((net > 500).sum())
        lottery10 += int((net > 1000).sum())
        per_sym[sym] = {"n": len(e), "net": round(net.mean(), 2), "wr": round((net > 0).mean(), 4)}
    if total_trades == 0:
        return None
    return {
        "threshold": threshold,
        "hold_bars": hold,
        "trades": int(total_trades),
        "avg_net_bps": round(total_net / total_trades, 2),
        "avg_gross_bps": round(total_gross / total_trades, 2),
        "win_rate": round(wins / total_trades, 4),
        "lottery5": int(lottery5),
        "lottery10": int(lottery10),
        "per_symbol_summary": per_sym,
    }


def main():
    results = []
    print(f"{'thr':>4} {'hold':>5} {'n':>6} {'gross':>8} {'net':>8} {'WR':>6} {'L5%':>5} {'L10%':>5}")
    for t in THRESHOLDS:
        for h in HOLDS:
            r = run(t, h)
            if r is None:
                continue
            results.append(r)
            print(f"{t:>4} {h:>5} {r['trades']:>6} {r['avg_gross_bps']:>+8.2f} {r['avg_net_bps']:>+8.2f} {r['win_rate']*100:>5.1f}% {r['lottery5']:>5} {r['lottery10']:>5}")
    # best by avg_net_bps
    best = max(results, key=lambda x: x["avg_net_bps"])
    print(f"\nBEST by net: threshold={best['threshold']} hold={best['hold_bars']}h n={best['trades']} net={best['avg_net_bps']}bps WR={best['win_rate']*100:.1f}%")
    OUT.write_text(json.dumps({"sweep": results, "best": best}, indent=2, ensure_ascii=False))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
