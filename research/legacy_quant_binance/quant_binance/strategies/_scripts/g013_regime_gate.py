"""
G013 — G003 + BTC regime gate (variable-1: regime_filter 추가)

가설: Q2/Q4 음수 분기는 BTC 약세장. BTC 7-day 모멘텀이 음수면 진입 스킵.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import load_klines, compute_ch1_score, COST_BPS_RT

UNIVERSE_18 = ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT",
               "AVAXUSDT", "SUIUSDT", "ADAUSDT", "APTUSDT", "BNBUSDT",
               "DOTUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "SOLUSDT",
               "UNIUSDT", "XRPUSDT", "BTCUSDT"]


def btc_regime_filter(threshold_pct):
    """BTC 1h df 로드 → 7일(168bar) 수익률 시리즈. open_time → 통과 여부 dict."""
    df = load_klines("BTCUSDT")
    df["btc_7d_pct"] = (df["close_price"] / df["close_price"].shift(168) - 1) * 100
    return dict(zip(df["open_time"], df["btc_7d_pct"] >= threshold_pct))


def run(threshold_long, regime_threshold_pct, hold=72):
    btc_pass = btc_regime_filter(regime_threshold_pct)
    total_n = 0
    total_net = 0.0
    wins = 0
    lottery5 = lottery10 = lottery20 = 0
    blocked = 0
    for sym in UNIVERSE_18:
        df = load_klines(sym)
        if df is None: continue
        score, _ = compute_ch1_score(df)
        df["score"] = score
        df["fwd_pct"] = (df["close_price"].shift(-hold) / df["close_price"] - 1) * 10000
        df["btc_pass"] = df["open_time"].map(btc_pass).fillna(False)
        cand = df[df["score"] >= threshold_long].dropna(subset=["fwd_pct"])
        e = cand[cand["btc_pass"]]
        blocked += len(cand) - len(e)
        if len(e) == 0: continue
        net = e["fwd_pct"] - COST_BPS_RT
        total_n += len(e)
        total_net += float(net.sum())
        wins += int((net > 0).sum())
        lottery5 += int((net > 500).sum())
        lottery10 += int((net > 1000).sum())
        lottery20 += int((net > 2000).sum())
    return {
        "regime_threshold_pct": regime_threshold_pct,
        "n": total_n,
        "blocked_by_regime": blocked,
        "avg_net_bps": round(total_net / max(total_n,1), 2),
        "win_rate": round(wins / max(total_n,1), 4),
        "lottery5": lottery5, "lottery10": lottery10, "lottery20": lottery20,
    }


def quarterly_with_regime(regime_thresh):
    """분기별 G013 성과."""
    btc_pass = btc_regime_filter(regime_thresh)
    out = []
    all_dfs = {sym: load_klines(sym) for sym in UNIVERSE_18}
    for q in range(4):
        total_n, total_net, wins = 0, 0.0, 0
        for sym, df in all_dfs.items():
            if df is None: continue
            score, _ = compute_ch1_score(df)
            df_q = df.copy()
            df_q["score"] = score
            df_q["fwd_pct"] = (df_q["close_price"].shift(-72) / df_q["close_price"] - 1) * 10000
            df_q["btc_pass"] = df_q["open_time"].map(btc_pass).fillna(False)
            n = len(df_q)
            a, b = int(n*q/4), int(n*(q+1)/4)
            sub = df_q.iloc[a:b]
            e = sub[(sub["score"] >= 70) & sub["btc_pass"]].dropna(subset=["fwd_pct"])
            if len(e) == 0: continue
            net = e["fwd_pct"] - COST_BPS_RT
            total_n += len(e); total_net += float(net.sum()); wins += int((net>0).sum())
        out.append({"q": q+1, "n": total_n, "net": round(total_net/max(total_n,1),1), "wr": round(wins/max(total_n,1),4)})
    return out


def main():
    print("=== G013: BTC 7-day 모멘텀 regime gate ===\n")
    print(f"{'regime':>8} {'n':>6} {'blocked':>8} {'net':>9} {'WR':>7} {'L5%':>5} {'L10%':>5} {'L20%':>5}")
    candidates = []
    for thr in [-10, -5, -2, 0, 2, 5]:
        r = run(70, thr)
        candidates.append(r)
        print(f"{thr:>+5}%  {r['n']:>6} {r['blocked_by_regime']:>8} {r['avg_net_bps']:>+9.2f} {r['win_rate']*100:>6.1f}% {r['lottery5']:>5} {r['lottery10']:>5} {r['lottery20']:>5}")

    # 최고 EV 후보 (n × net 가중 — 실제 누적 PnL 추정)
    best_by_total = max(candidates, key=lambda r: r['n'] * r['avg_net_bps'])
    best_by_avg = max(candidates, key=lambda r: r['avg_net_bps'])
    print(f"\n  최고 누적 PnL: regime≥{best_by_total['regime_threshold_pct']:+}% → n={best_by_total['n']}, net={best_by_total['avg_net_bps']}bps, total≈{best_by_total['n']*best_by_total['avg_net_bps']/100:.0f}%")
    print(f"  최고 거래당 EV: regime≥{best_by_avg['regime_threshold_pct']:+}% → net={best_by_avg['avg_net_bps']}bps WR {best_by_avg['win_rate']*100:.1f}%")

    print("\n=== 최고 후보 분기별 일관성 ===")
    print(f"{'regime':>8} {'Q1':>16} {'Q2':>16} {'Q3':>16} {'Q4':>16}")
    for thr in [-10, -5, 0]:
        qs = quarterly_with_regime(thr)
        row = f"{thr:>+5}%  "
        for q in qs:
            row += f"  {q['n']:>4}/{q['net']:>+5.0f}/{q['wr']*100:>4.0f}%"
        print(row)

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g013_regime_results.json"
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "regime_sweep": candidates,
        "best_by_total": best_by_total,
        "best_by_avg": best_by_avg,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
