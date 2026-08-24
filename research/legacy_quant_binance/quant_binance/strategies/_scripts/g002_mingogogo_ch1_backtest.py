"""
G002 Mingogogo CH1 (10-indicator weighted score) — MVP 백테스트

PB001 mining 결과 (claimed_performance.md): 17 post / 41 cherry-picked → 신뢰도 1.5/5.
이 스크립트는 cherry-pick 영향 없이 우리 데이터로 CH1 가설 진위 객관 측정.

CH1 가중치 (PB001 rules.md 식별):
  RSI 15 + MFI 12 + Stoch 12 + CCI 10 + W%R 10 + BB%B 10 + MACD 8 + ADX 8 + OBV 8 + ATR 7 = 100

각 인디케이터는 0~100 으로 정규화 (oversold/buy bias 일수록 높은 점수).
가중 합산 점수 >= 70 = "강력매수" 진입 (PB001 임계).

Universe: alt USDT-perp (사용자 lottery 컨텍스트 핏)
Holding: 4h (S001 baseline 과 비교 가능)
Cost: 16 bps round-trip (S001 동일)

Forward-bar 시뮬레이션 — TP/SL intra-bar 정밀 측정 X. 4시간 후 close 기준.
"""
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from datetime import datetime, timezone

import pandas as pd
import numpy as np

ARCHIVE = Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical"
LOCAL = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical"
OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g002_backtest_result.json"

# Alt-only universe ($50 lottery context)
SYMBOLS = ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT", "AVAXUSDT", "SUIUSDT", "ADAUSDT"]

TF = "1h"
HOLD_BARS = 4   # 4h holding (S001 비교)
COST_BPS_RT = 16.0
SCORE_THRESHOLD = 70  # PB001 CH1 강력매수 임계

WEIGHTS = {
    "rsi": 15, "mfi": 12, "stoch": 12, "cci": 10, "wr": 10,
    "bbpct": 10, "macd": 8, "adx": 8, "obv": 8, "atr": 7,
}


def load_klines(symbol):
    for base in (ARCHIVE, LOCAL):
        p = base / symbol / f"{TF}.json"
        if p.exists():
            data = json.loads(p.read_text())
            data.sort(key=lambda b: b["open_time"])
            df = pd.DataFrame(data)
            for c in ("open_price", "high_price", "low_price", "close_price", "base_volume", "quote_volume"):
                df[c] = df[c].astype(float)
            return df
    return None


def rsi(close, length=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def mfi(high, low, close, vol, length=14):
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = rmf.where(tp > tp.shift(1), 0).rolling(length).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(length).sum()
    mr = pos / neg.replace(0, np.nan)
    out = 100 - (100 / (1 + mr))
    return out.fillna(50)


def stoch_k(high, low, close, k=14, smooth=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    raw = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return raw.rolling(smooth).mean().fillna(50)


def cci(high, low, close, length=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(length).mean()
    md = (tp - sma).abs().rolling(length).mean()
    out = (tp - sma) / (0.015 * md).replace(0, np.nan)
    return out.fillna(0)


def williams_r(high, low, close, length=14):
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    out = -100 * (hh - close) / (hh - ll).replace(0, np.nan)
    return out.fillna(-50)


def bbands_pct(close, length=20, std=2):
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = ma + std * sd
    lower = ma - std * sd
    out = (close - lower) / (upper - lower).replace(0, np.nan)
    return out.fillna(0.5)


def macd_hist(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def adx(high, low, close, length=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_v = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_v.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_v.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/length, adjust=False).mean().fillna(20)


def obv_slope(close, vol, length=20):
    sign = np.sign(close.diff().fillna(0))
    obv = (sign * vol).cumsum()
    return obv.diff(length).fillna(0)


def atr_pct(high, low, close, length=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/length, adjust=False).mean()
    return (a / close * 100).fillna(0)


def normalize(series, low, high, invert=False):
    """Map series to 0~100. invert=True 이면 작을수록 100 (oversold = bullish)."""
    s = series.clip(low, high)
    if invert:
        s = high - (s - low)
    return ((s - low) / (high - low) * 100).clip(0, 100)


def compute_ch1_score(df):
    """10-indicator weighted score (PB001 CH1). 0~100 scale, 70+ = 강력매수."""
    h, l, c, v = df["high_price"], df["low_price"], df["close_price"], df["base_volume"]

    # 각 인디케이터 → 0~100 점수 (높을수록 long bias)
    s = {}
    s["rsi"]   = normalize(rsi(c), 20, 80, invert=True)            # RSI 작을수록 oversold = bullish
    s["mfi"]   = normalize(mfi(h, l, c, v), 20, 80, invert=True)   # MFI 동일
    s["stoch"] = normalize(stoch_k(h, l, c), 20, 80, invert=True)  # Stoch 동일
    s["cci"]   = normalize(cci(h, l, c), -200, 200, invert=True)   # CCI -100 이하 oversold
    s["wr"]    = normalize(williams_r(h, l, c), -100, 0, invert=True)  # W%R -80 이하 oversold
    s["bbpct"] = normalize(bbands_pct(c), 0, 1, invert=True)       # BB%B 0 근처 lower band touch
    s["macd"]  = normalize(macd_hist(c), -c.std()*0.01, c.std()*0.01, invert=False)  # MACD hist 양수 = momentum
    s["adx"]   = normalize(adx(h, l, c), 20, 50, invert=False)     # ADX 25+ trending
    s["obv"]   = normalize(obv_slope(c, v), -v.mean()*5, v.mean()*5, invert=False)   # OBV 상승 = 매집
    s["atr"]   = normalize(atr_pct(h, l, c), 1, 5, invert=False)   # ATR% 1~5 = 적정 변동성

    score = sum(s[k] * WEIGHTS[k] / 100 for k in WEIGHTS)
    return score, s


def backtest_symbol(symbol):
    df = load_klines(symbol)
    if df is None or len(df) < 100:
        return None
    score, _ = compute_ch1_score(df)
    df["score"] = score
    df["fwd_pct"] = (df["close_price"].shift(-HOLD_BARS) / df["close_price"] - 1) * 10000  # bps

    # 진입: score >= threshold (직전 봉 X, 현 봉 close 기준)
    entries = df[df["score"] >= SCORE_THRESHOLD].copy()
    entries = entries.dropna(subset=["fwd_pct"])

    if len(entries) == 0:
        return {"symbol": symbol, "trades": 0}

    entries["net_bps"] = entries["fwd_pct"] - COST_BPS_RT

    return {
        "symbol": symbol,
        "trades": len(entries),
        "mean_gross_bps": round(entries["fwd_pct"].mean(), 2),
        "mean_net_bps": round(entries["net_bps"].mean(), 2),
        "median_net_bps": round(entries["net_bps"].median(), 2),
        "win_rate_net": round((entries["net_bps"] > 0).mean(), 4),
        "win_rate_gross": round((entries["fwd_pct"] > 0).mean(), 4),
        "best_bps": round(entries["net_bps"].max(), 2),
        "worst_bps": round(entries["net_bps"].min(), 2),
        "score_dist": {
            "mean": round(entries["score"].mean(), 1),
            "max": round(entries["score"].max(), 1),
        },
        "lottery_count_gt_500bps": int((entries["net_bps"] > 500).sum()),  # +5%
        "lottery_count_gt_1000bps": int((entries["net_bps"] > 1000).sum()),  # +10%
    }


def aggregate(per_symbol):
    valid = [r for r in per_symbol if r and r.get("trades", 0) > 0]
    if not valid:
        return {}
    total_trades = sum(r["trades"] for r in valid)
    weighted_net = sum(r["mean_net_bps"] * r["trades"] for r in valid) / total_trades
    weighted_gross = sum(r["mean_gross_bps"] * r["trades"] for r in valid) / total_trades
    weighted_wr = sum(r["win_rate_net"] * r["trades"] for r in valid) / total_trades
    return {
        "total_trades": total_trades,
        "avg_net_bps": round(weighted_net, 2),
        "avg_gross_bps": round(weighted_gross, 2),
        "win_rate_net": round(weighted_wr, 4),
        "lottery_5pct": sum(r.get("lottery_count_gt_500bps", 0) for r in valid),
        "lottery_10pct": sum(r.get("lottery_count_gt_1000bps", 0) for r in valid),
    }


def main():
    print(f"G002 Mingogogo CH1 backtest — universe={len(SYMBOLS)} alts, TF={TF}, hold={HOLD_BARS}h, threshold={SCORE_THRESHOLD}", flush=True)
    per_symbol = []
    for sym in SYMBOLS:
        r = backtest_symbol(sym)
        per_symbol.append(r)
        if r and r.get("trades", 0):
            print(f"  {sym}: n={r['trades']} net={r['mean_net_bps']}bps WR={r['win_rate_net']*100:.1f}% lottery5%={r['lottery_count_gt_500bps']}", flush=True)
        else:
            print(f"  {sym}: SKIP (no data or no entries)", flush=True)

    agg = aggregate(per_symbol)
    result = {
        "strategy_id": "G002",
        "playbook": "PB001",
        "setup": "Mingogogo CH1 (10-indicator weighted score >= 70)",
        "mode": "batch_backtest_standalone",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "universe": SYMBOLS,
        "timeframe": TF,
        "holding_period_bars": HOLD_BARS,
        "cost_bps_round_trip": COST_BPS_RT,
        "score_threshold": SCORE_THRESHOLD,
        "weights": WEIGHTS,
        "per_symbol": per_symbol,
        "aggregate": agg,
    }

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[AGG] trades={agg.get('total_trades', 0)} net={agg.get('avg_net_bps')}bps gross={agg.get('avg_gross_bps')}bps WR={agg.get('win_rate_net', 0)*100:.1f}% lottery5%={agg.get('lottery_5pct', 0)} lottery10%={agg.get('lottery_10pct', 0)}", flush=True)
    print(f"saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
