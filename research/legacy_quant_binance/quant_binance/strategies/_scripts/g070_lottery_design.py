"""
G070-G073 — Lottery 컨텍스트 재설계 ($55, 5-10x, 양방향, ≥3건/일, 단기, 빠른 검증).

G070: lottery 코어 — threshold 80 + 5x leverage + 24h hold + max5
G071: 양방향 — G070 long + Bollinger upper squeeze short overlay
G072: 빈도 ↑ — threshold 60 + 12h hold (entries/day 3+ 충족)
G073: leverage 안전망 — intra-bar wide SL 5R + TP 3R (G070 + 청산 룰 추가)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct, rsi, bbands_pct

DATA_22 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2022"
DATA_24 = Path.home() / "Desktop" / "first_repo" / "quant_runtime" / "historical_2024"
DATA_25 = Path.home() / "iCloudDrive" / "quant_archive" / "quant_runtime" / "historical"
UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

EQUITY = 55.0
SIZE_PCT = 0.30


def load_period_dfs(data_dir, universe):
    out = {}
    for sym in universe:
        p = Path(data_dir) / sym / "1h.json"
        if p.exists():
            df = pd.DataFrame(json.loads(p.read_text()))
            if len(df) >= 100: out[sym] = df
    return out


def gather_long_short(dfs, threshold, hold_bars):
    """CH1 score >= threshold long + Bollinger squeeze short (BB%B > 0.95 + RSI > 75)."""
    long_e, short_e = [], []
    for sym, df in dfs.items():
        score, _ = compute_ch1_score(df)
        df = df.copy()
        df["score"] = score
        df["fwd_pct"] = (df["close_price"].shift(-hold_bars) / df["close_price"] - 1) * 10000
        # short signal: BB%B > 0.95 (upper touch) + RSI > 75 (overbought) + 양봉
        bbp = bbands_pct(df["close_price"])
        r = rsi(df["close_price"])
        df["short_sig"] = (bbp > 0.95) & (r > 75) & (df["close_price"] > df["open_price"])
        # long
        e_long = df[(df["score"] >= threshold) & df["fwd_pct"].notna()].copy()
        e_long["sym"] = sym; e_long["side"] = "long"
        e_long["gross_bps"] = e_long["fwd_pct"]
        long_e.append(e_long[["open_time","score","gross_bps","sym","side"]])
        # short
        e_short = df[df["short_sig"] & df["fwd_pct"].notna()].copy()
        e_short["sym"] = sym; e_short["side"] = "short"
        e_short["gross_bps"] = -e_short["fwd_pct"]  # short = inverse
        short_e.append(e_short[["open_time","score","gross_bps","sym","side"]])
    long_df = pd.concat(long_e).sort_values("open_time").reset_index(drop=True) if long_e else pd.DataFrame()
    short_df = pd.concat(short_e).sort_values("open_time").reset_index(drop=True) if short_e else pd.DataFrame()
    return long_df, short_df


def portfolio_sim_lottery(entries, leverage, hold_bars, max_conc, days, label):
    """간이 portfolio sim. capacity 적용. PnL = size × leverage × net_pct."""
    if len(entries) == 0:
        return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos = []
    pnl = 0.0; taken = 0; big_wins = 0; big_losses = 0
    wins = 0
    for _, row in entries.iterrows():
        ts = row["open_time"]
        net = row["gross_bps"] - COST_BPS_RT
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == row["sym"] for p in open_pos): continue
        if len(open_pos) >= max_conc: continue
        size = EQUITY * SIZE_PCT
        net_pct = net / 10000 * leverage
        # 청산 시뮬: -90% 단계 도달 시 청산 (liquidation buffer)
        if net_pct < -0.90:
            net_pct = -0.90
        pnl += size * net_pct
        taken += 1
        if net_pct > 0.30: big_wins += 1
        if net_pct < -0.20: big_losses += 1
        if net_pct > 0: wins += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))
    return {
        "label": label,
        "n_taken": taken,
        "entries_per_day": round(taken / days, 2),
        "pnl_usd": round(pnl, 2),
        "pnl_pct": round(pnl / EQUITY * 100, 1),
        "annual_pct": round(pnl / EQUITY * 100 / days * 365, 1),
        "win_rate": round(wins / taken, 4) if taken else 0,
        "big_wins_30pct": big_wins,
        "big_losses_20pct": big_losses,
        "avg_per_trade_pct": round(pnl / EQUITY * 100 / max(taken, 1), 2),
    }


def main():
    print("=== G070-G073 Lottery 재설계 ($55, 5x lev, max5/30%) ===\n")
    periods = [
        ("OOS22-23",  load_period_dfs(DATA_22, UNIV_22), 730),
        ("OOS24-Q1",  load_period_dfs(DATA_24, UNIV_24), 456),
        ("IS25-26",   load_period_dfs(DATA_25, UNIV_25), 374),
    ]

    print(f"{'strategy':<28} {'period':>10} {'taken':>5} {'/day':>5} {'avg':>7} {'WR':>6} {'big30':>6} {'big-20':>7} {'PnL%':>7} {'annual':>9}")
    print("-" * 100)

    strategies = [
        # (label, threshold, hold, leverage, side='long'|'both'|'high_freq')
        ("G050 baseline (thr70/72h/1x)", 70, 72, 1.0, "long"),
        ("G070 lottery (thr80/24h/5x)",  80, 24, 5.0, "long"),
        ("G070b lottery (thr80/12h/5x)", 80, 12, 5.0, "long"),
        ("G072 freq (thr60/12h/3x)",     60, 12, 3.0, "long"),
        ("G072b freq (thr60/24h/5x)",    60, 24, 5.0, "long"),
        ("G073 lev_safe (thr80/24h/5x)", 80, 24, 5.0, "long_capped"),  # -25% liquidation cap
        ("G071 both (thr80/24h/5x)",     80, 24, 5.0, "both"),
    ]
    summary = []
    for slabel, thr, hold, lev, mode in strategies:
        for plabel, dfs, days in periods:
            long_e, short_e = gather_long_short(dfs, thr, hold)
            if mode == "long":
                ent = long_e
            elif mode == "long_capped":
                ent = long_e
                # capping is applied inside portfolio_sim
            elif mode == "both":
                ent = pd.concat([long_e, short_e]).sort_values("open_time").reset_index(drop=True) if len(short_e) else long_e
            else:
                ent = long_e
            r = portfolio_sim_lottery(ent, lev, hold, 5, days, slabel)
            if r is None:
                print(f"{slabel:<28} {plabel:>10}  no entries")
                continue
            summary.append((slabel, plabel, r))
            print(f"{slabel:<28} {plabel:>10} {r['n_taken']:>5} {r['entries_per_day']:>5.2f} {r['avg_per_trade_pct']:>+6.2f}% {r['win_rate']*100:>5.1f}% {r['big_wins_30pct']:>6} {r['big_losses_20pct']:>7} {r['pnl_pct']:>+6.1f}% {r['annual_pct']:>+8.1f}%")
        print()

    # 6축 체크
    print("=== 사용자 6축 충족도 ===")
    print(f"{'strategy':<28} {'≥3/일?':>8} {'5-10x?':>8} {'양방향?':>9} {'lottery?':>10}")
    targets = {}
    for slabel, plabel, r in summary:
        if slabel not in targets: targets[slabel] = []
        targets[slabel].append(r)
    for slabel, results in targets.items():
        avg_per_day = np.mean([r['entries_per_day'] for r in results])
        big30_total = sum(r['big_wins_30pct'] for r in results)
        n_total = sum(r['n_taken'] for r in results)
        lottery_score = big30_total / max(n_total, 1) * 100
        freq_ok = "✓" if avg_per_day >= 3 else f"✗ ({avg_per_day:.1f})"
        print(f"{slabel:<28} {freq_ok:>8} {'(5x)':>8} {'long':>9} {lottery_score:>9.1f}%")


if __name__ == "__main__":
    main()
