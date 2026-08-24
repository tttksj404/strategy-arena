"""G099 — Funding × CH1 dual-confirm (양방향). 2024 funding false positive 해결 시도."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct
from g098_funding_validation import (
    load, merge_funding, COMMON, KLINES_DIR, FUND_DIR,
    HOLD, LEV, SIZE_PCT, MAX_CONC, EQUITY, ATR_GUARD, HOLD_MS,
    split_yearly, portfolio_sim, run_9point,
)


def gather_g099(symbols, ch1_long_thr=60, ch1_short_thr=40,
                fund_long_thr=-0.0003, fund_short_thr=0.0005, hold_bars=24):
    """
    LONG: CH1 score >= ch1_long_thr AND funding <= fund_long_thr AND bullish + vol spike
    SHORT: CH1 score <= ch1_short_thr AND funding >= fund_short_thr AND bearish + vol spike
    """
    long_e, short_e = [], []
    for sym in symbols:
        df, fund = load(sym)
        if df is None or len(df) < 100: continue
        score, _ = compute_ch1_score(df)
        a = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["score"] = score; df["atr_pct"] = a
        df["bullish"] = df["close_price"] > df["open_price"]
        df["bearish"] = df["close_price"] < df["open_price"]
        df["vol_ma20"] = df["base_volume"].rolling(20).mean()
        df["vol_spike"] = df["base_volume"] > 1.3 * df["vol_ma20"]
        df["fwd_pct"] = (df["close_price"].shift(-hold_bars) / df["close_price"] - 1) * 10000
        intra_low_long = df["low_price"].rolling(window=hold_bars+1, min_periods=1).min().shift(-hold_bars)
        df["intra_low_bps_long"] = (intra_low_long / df["close_price"] - 1) * 10000
        intra_high_short = df["high_price"].rolling(window=hold_bars+1, min_periods=1).max().shift(-hold_bars)
        df["intra_low_bps_short"] = (df["close_price"] / intra_high_short - 1) * 10000

        df = merge_funding(df, fund)

        # LONG dual-confirm
        long_cond = ((df["score"] >= ch1_long_thr)
                     & (df["funding"] <= fund_long_thr)
                     & df["bullish"]
                     & df["vol_spike"]
                     & (a <= ATR_GUARD)
                     & df["fwd_pct"].notna())
        e_l = df[long_cond].copy()
        e_l["sym"] = sym; e_l["side"] = "long"
        e_l["gross_bps"] = e_l["fwd_pct"]
        e_l["net_bps"] = e_l["fwd_pct"] - 16
        e_l["intra_low_bps"] = e_l["intra_low_bps_long"]
        if len(e_l):
            long_e.append(e_l[["open_time","gross_bps","net_bps","atr_pct","intra_low_bps","funding","sym","side"]])

        # SHORT dual-confirm
        short_cond = ((df["score"] <= ch1_short_thr)
                      & (df["funding"] >= fund_short_thr)
                      & df["bearish"]
                      & df["vol_spike"]
                      & (a <= ATR_GUARD)
                      & df["fwd_pct"].notna())
        e_s = df[short_cond].copy()
        e_s["sym"] = sym; e_s["side"] = "short"
        e_s["gross_bps"] = -e_s["fwd_pct"]
        e_s["net_bps"] = -e_s["fwd_pct"] - 16
        e_s["intra_low_bps"] = e_s["intra_low_bps_short"]
        if len(e_s):
            short_e.append(e_s[["open_time","gross_bps","net_bps","atr_pct","intra_low_bps","funding","sym","side"]])

    long_df = pd.concat(long_e).sort_values("open_time").reset_index(drop=True) if long_e else pd.DataFrame()
    short_df = pd.concat(short_e).sort_values("open_time").reset_index(drop=True) if short_e else pd.DataFrame()
    return long_df, short_df


def main():
    print("=== G099 — funding × CH1 dual-confirm ===\n")

    # 변형 sweep
    variants = [
        ("v1 (ch1L60/S40 fund-3/+5)", 60, 40, -0.0003, 0.0005),
        ("v2 (ch1L70/S30 fund-3/+5)", 70, 30, -0.0003, 0.0005),
        ("v3 (ch1L60/S40 fund-5/+8)", 60, 40, -0.0005, 0.0008),
        ("v4 (ch1L80/S20 fund-3/+5)", 80, 20, -0.0003, 0.0005),
    ]
    best_combined = None; best_passed = 0
    for label, ch1_l, ch1_s, fund_l, fund_s in variants:
        long_e, short_e = gather_g099(COMMON, ch1_l, ch1_s, fund_l, fund_s)
        combined = pd.concat([long_e, short_e]).sort_values("open_time").reset_index(drop=True) if len(long_e) or len(short_e) else pd.DataFrame()
        c = run_9point(f"G099 {label}", combined)
        passed = sum(c) if isinstance(c, list) else 0
        if passed > best_passed:
            best_passed = passed; best_combined = (label, combined)

    print(f"\n\n{'='*70}\n=== Best: {best_combined[0] if best_combined else 'N/A'} {best_passed}/9 ===\n{'='*70}")


if __name__ == "__main__":
    main()
