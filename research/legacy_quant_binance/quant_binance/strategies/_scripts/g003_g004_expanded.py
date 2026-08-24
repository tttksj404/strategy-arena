"""
G003 (universe 확장) + G004 (threshold 80 lottery) — 변수 1개 룰 준수.

G002 baseline: 8 alt / threshold 70 / hold 72h → net +221 bps WR 58.3% n=2670 (374d 윈도우)

G003: G002 + universe 18종 (1 변수 변경: universe 8→18)
G004: G002 + threshold 80 (1 변수 변경: threshold 70→80)
G005: G002 + hold 24h (1 변수 변경: hold 72→24)

추가 G006 = G003 universe + G004 threshold combo (최적 lottery)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import load_klines, compute_ch1_score, COST_BPS_RT

UNIVERSE_8 = ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT", "AVAXUSDT", "SUIUSDT", "ADAUSDT"]
UNIVERSE_18 = UNIVERSE_8 + [
    "APTUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
    "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT", "BTCUSDT",
]
# MATIC 제외 (시간 범위 다름), ETH 추가 안함 (BTC 1개로 majors 대표)


def run(universe, threshold, hold, label):
    total_n = 0
    total_net = 0.0
    total_gross = 0.0
    wins = 0
    lottery5 = 0
    lottery10 = 0
    lottery20 = 0
    per_sym = {}
    big_winners_sym = {}  # 종목별 최대 수익
    for sym in universe:
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
        total_n += len(e)
        total_net += net.sum()
        total_gross += e["fwd_pct"].sum()
        wins += int((net > 0).sum())
        lottery5 += int((net > 500).sum())
        lottery10 += int((net > 1000).sum())
        lottery20 += int((net > 2000).sum())
        per_sym[sym] = {
            "n": int(len(e)),
            "net_bps": round(float(net.mean()), 2),
            "wr": round(float((net > 0).mean()), 4),
            "best_bps": round(float(net.max()), 2),
        }
        big_winners_sym[sym] = round(float(net.max()), 2)
    if total_n == 0:
        return None
    top_winners = sorted(big_winners_sym.items(), key=lambda x: -x[1])[:3]
    return {
        "label": label,
        "universe_size": len(universe),
        "threshold": threshold,
        "hold_bars": hold,
        "n": int(total_n),
        "avg_gross_bps": round(total_gross / total_n, 2),
        "avg_net_bps": round(total_net / total_n, 2),
        "win_rate": round(wins / total_n, 4),
        "lottery_5pct": int(lottery5),
        "lottery_10pct": int(lottery10),
        "lottery_20pct": int(lottery20),
        "lottery_5pct_per_day": round(lottery5 / 374, 3),
        "lottery_10pct_per_day": round(lottery10 / 374, 3),
        "top_3_winners": top_winners,
        "per_symbol": per_sym,
    }


def main():
    variants = [
        ("G002_baseline",  UNIVERSE_8,  70, 72),
        ("G003_universe",  UNIVERSE_18, 70, 72),
        ("G004_threshold", UNIVERSE_8,  80, 72),
        ("G005_hold24",    UNIVERSE_8,  70, 24),
        ("G006_combo",     UNIVERSE_18, 80, 72),  # 최적 lottery (univ + thr 동시 — variable-2)
    ]
    print(f"{'label':<18} {'univ':>4} {'thr':>4} {'hold':>5} {'n':>6} {'gross':>9} {'net':>9} {'WR':>7} {'L5%':>5} {'L10%':>5} {'L20%':>5}")
    results = []
    for label, univ, t, h in variants:
        r = run(univ, t, h, label)
        if r is None:
            print(f"{label:<18} (no entries)")
            continue
        results.append(r)
        print(f"{label:<18} {r['universe_size']:>4} {t:>4} {h:>5} {r['n']:>6} {r['avg_gross_bps']:>+9.2f} {r['avg_net_bps']:>+9.2f} {r['win_rate']*100:>6.1f}% {r['lottery_5pct']:>5} {r['lottery_10pct']:>5} {r['lottery_20pct']:>5}")

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g003_g006_results.json"
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_days": 374,
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")

    # 최고 성과 (avg_net_bps × n 가중) - 거래 수도 고려
    best_ev = max(results, key=lambda r: r['avg_net_bps'] * (r['n'] / max(r['n'], 1)) ** 0.5)
    best_lottery = max(results, key=lambda r: r['lottery_10pct'])
    best_total_pnl = max(results, key=lambda r: r['avg_net_bps'] * r['n'])
    print(f"\n=== 우승자 ===")
    print(f"  최고 EV (sample-adjusted): {best_ev['label']} → net +{best_ev['avg_net_bps']}bps WR {best_ev['win_rate']*100:.1f}% n={best_ev['n']}")
    print(f"  최고 lottery 10%+ 빈도:   {best_lottery['label']} → {best_lottery['lottery_10pct']}건 ({best_lottery['lottery_10pct_per_day']}/day)")
    print(f"  최고 누적 PnL (가상):     {best_total_pnl['label']} → net*n = {best_total_pnl['avg_net_bps'] * best_total_pnl['n']:.0f} bps-trades")


if __name__ == "__main__":
    main()
