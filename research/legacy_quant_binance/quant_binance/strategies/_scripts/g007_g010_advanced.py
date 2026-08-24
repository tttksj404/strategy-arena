"""
G007 (intra-bar TP/SL) + G010 (양방향 short 추가) — variable-1 룰 준수.

부모: G003 (production main, threshold 70 / hold 72h / universe 18)

G007: G003 + intra-bar exit logic (R 1.5/3.0 TP, ATR×1.5 SL) — exit_logic 1개 변수
G010: G003 + short side 추가 (CH1 inverted score) — direction 1개 변수

각자 단독 변형. 병합형은 G011 후속.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import (
    load_klines, compute_ch1_score, COST_BPS_RT,
    rsi, mfi, stoch_k, cci, williams_r, bbands_pct, macd_hist, adx, obv_slope, atr_pct,
    normalize, WEIGHTS,
)

UNIVERSE_18 = ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT",
               "AVAXUSDT", "SUIUSDT", "ADAUSDT", "APTUSDT", "BNBUSDT",
               "DOTUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "SOLUSDT",
               "UNIUSDT", "XRPUSDT", "BTCUSDT"]

THRESHOLD_LONG = 70
THRESHOLD_SHORT = 70  # short score 70+ = 강력매도
HOLD_BARS_MAX = 72
TP_R = 1.5    # take profit at 1.5 × initial risk
SL_R = 1.0    # stop loss at 1.0 × initial risk
ATR_LEN = 14
SL_ATR_MULT = 1.5  # initial risk = ATR × 1.5 (in price units)


def compute_ch1_short_score(df):
    """CH1 inverted: 높은 score = short bias. RSI/MFI/Stoch/CCI/W%R/BB%B 모두 invert=False
    (overbought 일수록 100). MACD/ADX/OBV/ATR 은 그대로 (방향 중립)."""
    h, l, c, v = df["high_price"], df["low_price"], df["close_price"], df["base_volume"]
    s = {}
    s["rsi"]   = normalize(rsi(c), 20, 80, invert=False)            # overbought
    s["mfi"]   = normalize(mfi(h, l, c, v), 20, 80, invert=False)
    s["stoch"] = normalize(stoch_k(h, l, c), 20, 80, invert=False)
    s["cci"]   = normalize(cci(h, l, c), -200, 200, invert=False)   # +100 이상 overbought
    s["wr"]    = normalize(williams_r(h, l, c), -100, 0, invert=False)  # -20 이상 overbought
    s["bbpct"] = normalize(bbands_pct(c), 0, 1, invert=False)        # 1 근처 upper band touch
    s["macd"]  = normalize(-macd_hist(c), -c.std()*0.01, c.std()*0.01, invert=False)  # 음수 momentum
    s["adx"]   = normalize(adx(h, l, c), 20, 50, invert=False)       # 동일 trending
    s["obv"]   = normalize(-obv_slope(c, v), -v.mean()*5, v.mean()*5, invert=False)  # OBV 하락 = 분배
    s["atr"]   = normalize(atr_pct(h, l, c), 1, 5, invert=False)     # 동일 변동성
    score = sum(s[k] * WEIGHTS[k] / 100 for k in WEIGHTS)
    return score


def simulate_intra_bar(df, entry_idx, side, entry_price, sl_distance, tp_distance):
    """진입 후 72봉 내에서 SL/TP 인터바 hit 시뮬. side: 'long'|'short'.
    Returns (exit_idx, exit_price, exit_reason)."""
    if side == "long":
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance

    end_idx = min(entry_idx + 1 + HOLD_BARS_MAX, len(df))
    for i in range(entry_idx + 1, end_idx):
        bar = df.iloc[i]
        h, l = bar["high_price"], bar["low_price"]
        if side == "long":
            if l <= sl_price:
                return i, sl_price, "SL"
            if h >= tp_price:
                return i, tp_price, "TP"
        else:
            if h >= sl_price:
                return i, sl_price, "SL"
            if l <= tp_price:
                return i, tp_price, "TP"
    last = min(entry_idx + HOLD_BARS_MAX, len(df) - 1)
    return last, df.iloc[last]["close_price"], "TIMEOUT"


def run_intra_bar_strategy(universe, threshold, side, label):
    total_n = 0
    total_net = 0.0
    total_gross = 0.0
    wins = 0
    tp_hits = 0
    sl_hits = 0
    timeouts = 0
    durations_h = []
    big_wins = 0
    per_sym = {}

    for sym in universe:
        df = load_klines(sym)
        if df is None or len(df) < 100:
            continue
        if side == "long":
            score, _ = compute_ch1_score(df)
        else:
            score = compute_ch1_short_score(df)
        df["score"] = score

        # ATR 시리즈 미리 계산
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], ATR_LEN)
        df["atr_pct"] = a
        df["atr_price"] = a / 100 * df["close_price"]

        sym_n = 0
        sym_net = 0.0
        sym_wins = 0

        i = 0
        while i < len(df) - 1:
            row = df.iloc[i]
            if pd.isna(row["score"]) or row["score"] < threshold or pd.isna(row["atr_price"]):
                i += 1
                continue
            entry_price = row["close_price"]
            sl_dist = row["atr_price"] * SL_ATR_MULT
            tp_dist = sl_dist * TP_R / SL_R   # TP 1.5R = 1.5 × SL distance
            exit_idx, exit_price, reason = simulate_intra_bar(df, i, side, entry_price, sl_dist, tp_dist)
            if side == "long":
                gross_bps = (exit_price / entry_price - 1) * 10000
            else:
                gross_bps = (1 - exit_price / entry_price) * 10000
            net_bps = gross_bps - COST_BPS_RT
            duration_h = exit_idx - i
            total_n += 1
            sym_n += 1
            total_net += net_bps
            total_gross += gross_bps
            sym_net += net_bps
            if net_bps > 0:
                wins += 1
                sym_wins += 1
            if net_bps > 1000:
                big_wins += 1
            if reason == "TP":
                tp_hits += 1
            elif reason == "SL":
                sl_hits += 1
            else:
                timeouts += 1
            durations_h.append(duration_h)
            # 다음 진입은 exit 이후부터 (포지션 1개 제한 simulation per symbol)
            i = exit_idx + 1

        if sym_n > 0:
            per_sym[sym] = {
                "n": sym_n,
                "net_bps": round(sym_net / sym_n, 2),
                "wr": round(sym_wins / sym_n, 4),
            }

    if total_n == 0:
        return None
    return {
        "label": label,
        "side": side,
        "n": total_n,
        "avg_gross_bps": round(total_gross / total_n, 2),
        "avg_net_bps": round(total_net / total_n, 2),
        "win_rate": round(wins / total_n, 4),
        "tp_pct": round(tp_hits / total_n, 4),
        "sl_pct": round(sl_hits / total_n, 4),
        "timeout_pct": round(timeouts / total_n, 4),
        "avg_duration_h": round(sum(durations_h) / total_n, 1),
        "big_winners_10pct": big_wins,
        "per_symbol_top5": dict(sorted(per_sym.items(), key=lambda x: -x[1]["n"])[:5]),
    }


def main():
    print(f"{'label':<20} {'side':>6} {'n':>5} {'gross':>8} {'net':>8} {'WR':>7} {'TP%':>5} {'SL%':>5} {'TO%':>5} {'durH':>5} {'L10%':>5}")
    results = []
    variants = [
        ("G007_intra_long",  UNIVERSE_18, THRESHOLD_LONG,  "long"),
        ("G010_short",       UNIVERSE_18, THRESHOLD_SHORT, "short"),
    ]
    for label, univ, thr, side in variants:
        r = run_intra_bar_strategy(univ, thr, side, label)
        if r is None:
            print(f"{label:<20} (no entries)")
            continue
        results.append(r)
        print(f"{label:<20} {side:>6} {r['n']:>5} {r['avg_gross_bps']:>+8.2f} {r['avg_net_bps']:>+8.2f} {r['win_rate']*100:>6.1f}% {r['tp_pct']*100:>4.1f}% {r['sl_pct']*100:>4.1f}% {r['timeout_pct']*100:>4.1f}% {r['avg_duration_h']:>5.1f} {r['big_winners_10pct']:>5}")

    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "g007_g010_results.json"
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "tp_r": TP_R, "sl_r": SL_R, "atr_len": ATR_LEN,
            "sl_atr_mult": SL_ATR_MULT, "hold_max_bars": HOLD_BARS_MAX,
            "threshold": THRESHOLD_LONG, "universe_size": len(UNIVERSE_18),
        },
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
