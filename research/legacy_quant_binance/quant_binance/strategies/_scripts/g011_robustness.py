"""
G011 robustness — G003 의 alpha 가 시간 regime 에 의존하는지 검증.

374-day window 을 4 분기로 나눠 각각 G003 (thr 70, hold 72h, 18 alts) 실행.
+ G003 + G004 병합 portfolio 시뮬레이션 (자본 60/40 배분, capacity 제한).
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import load_klines, compute_ch1_score, COST_BPS_RT

UNIVERSE_18 = ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT",
               "AVAXUSDT", "SUIUSDT", "ADAUSDT", "APTUSDT", "BNBUSDT",
               "DOTUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "SOLUSDT",
               "UNIUSDT", "XRPUSDT", "BTCUSDT"]


def split_quarter(df, q):
    """0~3 분기로 분할. 각 분기는 374/4 ≈ 94일."""
    n = len(df)
    a = int(n * q / 4)
    b = int(n * (q + 1) / 4)
    return df.iloc[a:b].copy()


def run_g003_on_subset(symbol_dfs, threshold, hold):
    total_n = 0
    total_net = 0.0
    wins = 0
    lottery5 = 0
    lottery10 = 0
    lottery20 = 0
    for df in symbol_dfs.values():
        if len(df) < 100:
            continue
        score, _ = compute_ch1_score(df)
        df = df.copy()
        df["score"] = score
        df["fwd_pct"] = (df["close_price"].shift(-hold) / df["close_price"] - 1) * 10000
        e = df[df["score"] >= threshold].dropna(subset=["fwd_pct"])
        if len(e) == 0:
            continue
        net = e["fwd_pct"] - COST_BPS_RT
        total_n += len(e)
        total_net += float(net.sum())
        wins += int((net > 0).sum())
        lottery5 += int((net > 500).sum())
        lottery10 += int((net > 1000).sum())
        lottery20 += int((net > 2000).sum())
    if total_n == 0:
        return None
    return {
        "n": total_n,
        "avg_net_bps": round(total_net / total_n, 2),
        "win_rate": round(wins / total_n, 4),
        "lottery5": lottery5, "lottery10": lottery10, "lottery20": lottery20,
    }


def quarterly_robustness():
    print("=== G011: G003 quarterly robustness ===")
    print(f"{'period':<10} {'days':>5} {'n':>6} {'net':>9} {'WR':>7} {'L5%':>5} {'L10%':>5} {'L20%':>5}")
    all_dfs = {}
    for sym in UNIVERSE_18:
        df = load_klines(sym)
        if df is not None:
            all_dfs[sym] = df
    quarters = []
    full = run_g003_on_subset(all_dfs, 70, 72)
    if full:
        print(f"{'FULL':<10} {'374':>5} {full['n']:>6} {full['avg_net_bps']:>+9.2f} {full['win_rate']*100:>6.1f}% {full['lottery5']:>5} {full['lottery10']:>5} {full['lottery20']:>5}")
    for q in range(4):
        sub = {sym: split_quarter(df, q) for sym, df in all_dfs.items()}
        r = run_g003_on_subset(sub, 70, 72)
        if r is None:
            print(f"Q{q}: no entries")
            continue
        r["quarter"] = q
        quarters.append(r)
        print(f"Q{q+1} (~94d) {'':>2} {r['n']:>6} {r['avg_net_bps']:>+9.2f} {r['win_rate']*100:>6.1f}% {r['lottery5']:>5} {r['lottery10']:>5} {r['lottery20']:>5}")

    # 일관성 메트릭
    nets = [q['avg_net_bps'] for q in quarters]
    wrs = [q['win_rate'] for q in quarters]
    print(f"\n  net 분기별: min={min(nets):.1f} max={max(nets):.1f} stdev={pd.Series(nets).std():.1f}")
    print(f"  WR 분기별:  min={min(wrs)*100:.1f}% max={max(wrs)*100:.1f}% stdev={pd.Series(wrs).std()*100:.1f}pp")
    consistent = all(n > 0 for n in nets) and min(wrs) >= 0.50
    print(f"  ROBUSTNESS: {'PASS ✓' if consistent else 'FAIL ✗'} (모든 분기 net>0 + WR≥50%)")
    return quarters, full, consistent


def portfolio_simulation():
    """G003 + G004 병합. 자본 $50, G003 60% / G004 40%, max concurrent positions = 5."""
    print("\n=== G012: G003 + G004 병합 portfolio 시뮬 ===")
    all_dfs = {}
    for sym in UNIVERSE_18:
        df = load_klines(sym)
        if df is not None:
            all_dfs[sym] = df
    # 모든 진입 후보 수집
    all_entries = []
    for sym, df in all_dfs.items():
        score, _ = compute_ch1_score(df)
        df = df.copy()
        df["score"] = score
        df["fwd_pct"] = (df["close_price"].shift(-72) / df["close_price"] - 1) * 10000
        for idx, row in df.iterrows():
            if pd.isna(row["score"]) or pd.isna(row["fwd_pct"]):
                continue
            if row["score"] >= 80:
                all_entries.append((row.name, sym, "G004", float(row["score"]), float(row["fwd_pct"]), idx))
            elif row["score"] >= 70:
                all_entries.append((row.name, sym, "G003", float(row["score"]), float(row["fwd_pct"]), idx))
    # 시간순 정렬 (open_time)
    # 진입 capacity: 자본 $50, G003 max $3.5/trade (10%), G004 max $7/trade (14%)
    # max concurrent positions = 5 (alt별 1개)
    all_entries.sort(key=lambda x: x[5])  # idx 기준
    EQUITY = 50.0
    G003_SIZE = 3.5  # USD per trade
    G004_SIZE = 7.0  # USD per trade
    MAX_CONCURRENT = 5
    HOLD_BARS = 72
    open_pos = []  # list of (exit_idx, sym, strat, size, fwd_pct)
    realized_pnl = 0.0
    g003_taken = 0
    g004_taken = 0
    g003_skipped = 0
    g004_skipped = 0
    big_winners = []
    for ts, sym, strat, score, fwd_pct, idx in all_entries:
        # 종료된 포지션 정리
        open_pos = [p for p in open_pos if p[0] > idx]
        if any(p[1] == sym for p in open_pos):
            # 이미 동일 심볼 포지션 보유 → skip
            if strat == "G003": g003_skipped += 1
            else: g004_skipped += 1
            continue
        if len(open_pos) >= MAX_CONCURRENT:
            if strat == "G003": g003_skipped += 1
            else: g004_skipped += 1
            continue
        size = G004_SIZE if strat == "G004" else G003_SIZE
        net_pct = (fwd_pct - COST_BPS_RT) / 10000
        pnl = size * net_pct
        realized_pnl += pnl
        if strat == "G003": g003_taken += 1
        else: g004_taken += 1
        if net_pct > 0.10:
            big_winners.append((sym, strat, round(net_pct*100, 1)))
        open_pos.append((idx + HOLD_BARS, sym, strat, size, fwd_pct))
    print(f"  Equity 시작: ${EQUITY}")
    print(f"  G003 진입: {g003_taken} (skip {g003_skipped} = capacity full)")
    print(f"  G004 진입: {g004_taken} (skip {g004_skipped} = capacity full)")
    print(f"  실현 PnL: ${realized_pnl:.2f}  ({realized_pnl/EQUITY*100:+.1f}% on $50)")
    print(f"  big winner (>10% trade): {len(big_winners)}건")
    if big_winners[:5]:
        print(f"  Top 5: {big_winners[:5]}")
    return {
        "equity_start": EQUITY,
        "g003_taken": g003_taken, "g003_skipped": g003_skipped,
        "g004_taken": g004_taken, "g004_skipped": g004_skipped,
        "realized_pnl_usd": round(realized_pnl, 2),
        "realized_pnl_pct": round(realized_pnl/EQUITY*100, 1),
        "big_winners_count": len(big_winners),
    }


def main():
    quarters, full, consistent = quarterly_robustness()
    portfolio = portfolio_simulation()

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g011_g012_robustness.json"
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "g011_quarterly_robustness": {
            "full": full,
            "by_quarter": quarters,
            "consistent": consistent,
        },
        "g012_portfolio_simulation": portfolio,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
