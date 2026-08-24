"""
G020/G021/G022 — PB001 CH1 실패에 대한 대안 시그널 클래스 검증.

가설: G003 (mean-reversion) 가 trending 시장에서 실패 → trend-following 이 그 시기 winner.
       두 클래스 결합 = regime-robust portfolio.

G020 — Donchian 20-period breakout (classic trend-following)
G021 — PB001 CH7 (Squeeze Momentum 양수 breakout) 단독
G022 — PB001 CH8 (급락경고) inverse + 단순 모멘텀 (대안 short 후보)
G023 — Volume surge + RSI cross (전형적 breakout pattern)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import (
    load_klines, COST_BPS_RT,
    rsi, atr_pct, macd_hist, bbands_pct,
)

UNIVERSE_18 = ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT",
               "AVAXUSDT", "SUIUSDT", "ADAUSDT", "APTUSDT", "BNBUSDT",
               "DOTUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "SOLUSDT",
               "UNIUSDT", "XRPUSDT", "BTCUSDT"]


def kc_bands(close, length=20, mult=1.5):
    """Keltner Channel: EMA ± mult × ATR."""
    ema = close.ewm(span=length, adjust=False).mean()
    a = atr_pct(close, close, close, length) * close / 100  # rough atr in price
    return ema - mult * a, ema, ema + mult * a


def squeeze_momentum(high, low, close, length=20):
    """LazyBear Squeeze Momentum: linreg of (close - midline) over length.
    Returns (momentum, squeeze_on)
    squeeze_on: BB inside KC (True = squeezed)."""
    bb_lower = close.rolling(length).mean() - 2 * close.rolling(length).std()
    bb_upper = close.rolling(length).mean() + 2 * close.rolling(length).std()
    kc_l, kc_m, kc_u = kc_bands(close, length, 1.5)
    squeeze = (bb_lower > kc_l) & (bb_upper < kc_u)
    # momentum: close - midpoint of (highest, lowest, sma)
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    sma = close.rolling(length).mean()
    val = close - ((hh + ll) / 2 + sma) / 2
    # linreg approx: rolling mean of difference
    mom = val.rolling(length).apply(lambda x: x.iloc[-1] if len(x) else 0, raw=False)
    return mom, squeeze


def backtest_signal(symbol_dfs, signal_fn, hold, label, side="long"):
    """signal_fn(df) → boolean Series (entry where True). Forward-bar simulation."""
    total_n = 0
    total_net = 0.0
    wins = lottery5 = lottery10 = lottery20 = 0
    quarter_stats = [{"n":0, "net":0.0, "wins":0} for _ in range(4)]
    for sym, df in symbol_dfs.items():
        if df is None or len(df) < 100: continue
        entries = signal_fn(df)
        df_local = df.copy()
        df_local["entry"] = entries
        df_local["fwd_pct"] = (df_local["close_price"].shift(-hold) / df_local["close_price"] - 1) * 10000
        if side == "short":
            df_local["fwd_pct"] = -df_local["fwd_pct"]
        e = df_local[df_local["entry"] & df_local["fwd_pct"].notna()]
        if len(e) == 0: continue
        net = e["fwd_pct"] - COST_BPS_RT
        total_n += len(e)
        total_net += float(net.sum())
        wins += int((net > 0).sum())
        lottery5 += int((net > 500).sum())
        lottery10 += int((net > 1000).sum())
        lottery20 += int((net > 2000).sum())
        # 분기별 분류
        n_total = len(df_local)
        for q in range(4):
            a, b = int(n_total*q/4), int(n_total*(q+1)/4)
            sub = e[(e.index >= e.index.min()) & ([(i >= a and i < b) for i in e.index])]  # crude
            sub_idx = [i for i in e.index if a <= i < b]
            if not sub_idx: continue
            sub_e = e.loc[sub_idx]
            sub_net = sub_e["fwd_pct"] - COST_BPS_RT
            quarter_stats[q]["n"] += len(sub_e)
            quarter_stats[q]["net"] += float(sub_net.sum())
            quarter_stats[q]["wins"] += int((sub_net > 0).sum())
    if total_n == 0:
        return None
    qs = []
    for q in quarter_stats:
        if q["n"] == 0:
            qs.append({"n":0, "net":0, "wr":0})
        else:
            qs.append({"n":q["n"], "net":round(q["net"]/q["n"],1), "wr":round(q["wins"]/q["n"],4)})
    return {
        "label": label,
        "side": side,
        "n": total_n,
        "avg_net_bps": round(total_net / total_n, 2),
        "win_rate": round(wins / total_n, 4),
        "lottery5": lottery5, "lottery10": lottery10, "lottery20": lottery20,
        "quarters": qs,
    }


# ─── 시그널 정의 ─────────────────────────────────────────────────────────

def sig_donchian_breakout(df, lookback=20):
    """Donchian: close > 직전 20봉 high → trend-following long."""
    rolling_high = df["high_price"].rolling(lookback).max().shift(1)
    return df["close_price"] > rolling_high

def sig_donchian_breakdown(df, lookback=20):
    """반대: close < 직전 20봉 low → trend-following short."""
    rolling_low = df["low_price"].rolling(lookback).min().shift(1)
    return df["close_price"] < rolling_low

def sig_squeeze_release_long(df, length=20):
    """Squeeze Momentum: squeeze 가 OFF 로 풀리는 순간 + momentum 양수 → long breakout."""
    mom, sq = squeeze_momentum(df["high_price"], df["low_price"], df["close_price"], length)
    sq_off_now = ~sq & sq.shift(1).fillna(False)  # 직전 squeeze ON, 현재 OFF = release
    return sq_off_now & (mom > 0)

def sig_volume_rsi_break(df, vol_mult=2.0, rsi_cross=50):
    """volume > 2× 20봉 평균 + RSI 50 상향 cross → 모멘텀 진입."""
    vol_ma = df["base_volume"].rolling(20).mean()
    vol_surge = df["base_volume"] > vol_ma * vol_mult
    r = rsi(df["close_price"])
    rsi_cross_up = (r > rsi_cross) & (r.shift(1) <= rsi_cross)
    return vol_surge & rsi_cross_up

def sig_pure_momentum_7d(df, period=168, threshold_pct=10):
    """7일(168bar) 수익률 > +10% → 트렌드 지속 가정 long."""
    ret = (df["close_price"] / df["close_price"].shift(period) - 1) * 100
    return ret > threshold_pct


def main():
    print("=== G020 alternatives — 대안 시그널 클래스 ===\n")
    print(f"{'label':<35} {'hold':>4} {'n':>5} {'net':>9} {'WR':>7} {'Q1':>15} {'Q2':>15} {'Q3':>15} {'Q4':>15}")
    all_dfs = {sym: load_klines(sym) for sym in UNIVERSE_18}

    variants = [
        # (label, signal_fn, hold, side)
        ("G020a Donchian-20 breakout L72h",       sig_donchian_breakout,        72, "long"),
        ("G020b Donchian-20 breakout L24h",       sig_donchian_breakout,        24, "long"),
        ("G020c Donchian-20 breakdown S72h",      sig_donchian_breakdown,       72, "short"),
        ("G021a Squeeze release L72h",            sig_squeeze_release_long,     72, "long"),
        ("G021b Squeeze release L24h",            sig_squeeze_release_long,     24, "long"),
        ("G023a Volume+RSI break L72h",           sig_volume_rsi_break,         72, "long"),
        ("G023b Volume+RSI break L24h",           sig_volume_rsi_break,         24, "long"),
        ("G024a Pure 7d momentum L72h",           sig_pure_momentum_7d,         72, "long"),
        ("G024b Pure 7d momentum L24h",           sig_pure_momentum_7d,         24, "long"),
    ]
    results = []
    for label, fn, hold, side in variants:
        r = backtest_signal(all_dfs, fn, hold, label, side)
        if r is None:
            print(f"{label:<35} no entries")
            continue
        results.append(r)
        qs = r["quarters"]
        q_str = lambda q: f"{q['n']:>3}/{q['net']:>+5.0f}/{q['wr']*100:>3.0f}%"
        print(f"{label:<35} {hold:>4} {r['n']:>5} {r['avg_net_bps']:>+9.2f} {r['win_rate']*100:>6.1f}% {q_str(qs[0]):>15} {q_str(qs[1]):>15} {q_str(qs[2]):>15} {q_str(qs[3]):>15}")

    # robustness 우승자: 모든 분기 net>0 AND avg_net_bps>0
    robust = [r for r in results if all(q['net'] > 0 for q in r['quarters'] if q['n']>0) and r['avg_net_bps'] > 0]
    print(f"\n=== Robust (모든 분기 net>0): {len(robust)}개 ===")
    for r in robust:
        print(f"  {r['label']}: avg net={r['avg_net_bps']}bps WR={r['win_rate']*100:.1f}% n={r['n']}")
    if not robust:
        print("  (없음 — 모든 시그널이 적어도 1개 분기에서 음수)")
        # 가장 일관된 후보
        best = min(results, key=lambda r: max([q['n'] for q in r['quarters'] if q['n']>0]+[1]) and -min(q['net'] for q in r['quarters'] if q['n']>0))
        print(f"  가장 손실 작은 후보: {best['label']} (worst Q net = {min(q['net'] for q in best['quarters'] if q['n']>0):.0f})")

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g020_alternatives_results.json"
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
        "robust_candidates": [r["label"] for r in robust],
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
