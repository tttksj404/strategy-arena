"""
G185 walk-forward validation — G070 + size_pct 0.30 -> 0.40, capital $55 -> $100.

3-period 검증:
  OOS22-23:  730 days (historical_2022)
  OOS24-Q1:  456 days (historical_2024)
  IS25-26:   374 days (historical)

목표 검증:
  - 거래당 winner avg ≥ $30 (=$100 × 0.40 × 5x × 0.15)
  - WR ≥ 70%
  - 3-period 모두 양수 PnL
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
sys.path.insert(0, str(ROOT / "quant_binance" / "strategies" / "_scripts"))
from g002_mingogogo_ch1_backtest import compute_ch1_score, COST_BPS_RT, atr_pct  # type: ignore

DATA_22 = ROOT / "quant_runtime" / "historical_2022"
DATA_24 = ROOT / "quant_runtime" / "historical_2024"
DATA_25 = ROOT / "quant_runtime" / "historical_is25"

UNIV_22 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","MATICUSDT","NEARUSDT","UNIUSDT","XRPUSDT"]
UNIV_24 = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT","LINKUSDT","LTCUSDT","AVAXUSDT","NEARUSDT","UNIUSDT","XRPUSDT","OPUSDT","ARBUSDT","APTUSDT","PEPEUSDT","SUIUSDT"]
UNIV_25 = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT","APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]

# G185 parameters
EQUITY = 100.0
SIZE_PCT = 0.40
LEVERAGE = 5.0
HOLD_BARS = 24
THRESHOLD = 80
MAX_CONC = 5
ATR_GUARD = 8.0


def load_period_dfs(data_dir, universe):
    out = {}
    for sym in universe:
        p = Path(data_dir) / sym / "1h.json"
        if p.exists():
            data = json.loads(p.read_text())
            if isinstance(data, list):
                data.sort(key=lambda b: b["open_time"])
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)
            for c in ("open_price","high_price","low_price","close_price","base_volume","quote_volume"):
                if c in df.columns:
                    df[c] = df[c].astype(float)
            if len(df) >= 100:
                out[sym] = df
    return out


def gather_long_entries(dfs, threshold, hold_bars, atr_guard_pct):
    rows = []
    for sym, df in dfs.items():
        score, _ = compute_ch1_score(df)
        df = df.copy()
        df["score"] = score
        df["atr_pct"] = atr_pct(df["high_price"], df["low_price"], df["close_price"], 14)
        df["fwd_pct"] = (df["close_price"].shift(-hold_bars) / df["close_price"] - 1) * 10000
        mask = (df["score"] >= threshold) & (df["atr_pct"] <= atr_guard_pct) & df["fwd_pct"].notna()
        e = df[mask].copy()
        if len(e) == 0:
            continue
        e["sym"] = sym
        rows.append(e[["open_time","score","atr_pct","fwd_pct","sym"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).sort_values("open_time").reset_index(drop=True)


def portfolio_sim(entries, equity, size_pct, leverage, hold_bars, max_conc, days, label):
    if len(entries) == 0:
        return None
    HOLD_MS = hold_bars * 3600 * 1000
    open_pos = []
    pnl_usd = 0.0
    taken = 0
    wins = 0
    big_wins_30 = 0
    big_losses_20 = 0
    per_trade_pnl_usd = []
    win_pnl_usd = []
    loss_pnl_usd = []
    for _, row in entries.iterrows():
        ts = row["open_time"]
        gross = row["fwd_pct"]
        net_bps = gross - COST_BPS_RT
        # release expired positions
        open_pos = [p for p in open_pos if p[0] > ts]
        # avoid same-symbol concurrent
        if any(p[1] == row["sym"] for p in open_pos):
            continue
        if len(open_pos) >= max_conc:
            continue
        margin = equity * size_pct
        net_pct = net_bps / 10000 * leverage
        # liquidation: -90% margin cap (Bitget UM perp ~80% maintenance)
        if net_pct < -0.90:
            net_pct = -0.90
        trade_pnl = margin * net_pct
        pnl_usd += trade_pnl
        per_trade_pnl_usd.append(trade_pnl)
        taken += 1
        if net_pct > 0:
            wins += 1
            win_pnl_usd.append(trade_pnl)
        else:
            loss_pnl_usd.append(trade_pnl)
        if net_pct > 0.30:
            big_wins_30 += 1
        if net_pct < -0.20:
            big_losses_20 += 1
        open_pos.append((ts + HOLD_MS, row["sym"]))

    return {
        "label": label,
        "n_taken": taken,
        "entries_per_day": round(taken / days, 3) if days else 0,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct_of_equity": round(pnl_usd / equity * 100, 2),
        "annual_pct": round(pnl_usd / equity * 100 / days * 365, 1) if days else 0,
        "annual_pnl_usd": round(pnl_usd / days * 365, 2) if days else 0,
        "win_rate": round(wins / taken, 4) if taken else 0,
        "big_wins_30pct": big_wins_30,
        "big_losses_20pct": big_losses_20,
        "avg_per_trade_pnl_usd": round(np.mean(per_trade_pnl_usd), 2) if per_trade_pnl_usd else 0,
        "avg_winner_pnl_usd": round(np.mean(win_pnl_usd), 2) if win_pnl_usd else 0,
        "avg_loser_pnl_usd": round(np.mean(loss_pnl_usd), 2) if loss_pnl_usd else 0,
    }


def main():
    print("=" * 100)
    print(f"G185 walk-forward validation")
    print(f"  capital=${EQUITY}, size={SIZE_PCT}, lev={LEVERAGE}x, thr={THRESHOLD}, hold={HOLD_BARS}h, max_conc={MAX_CONC}, atr_guard={ATR_GUARD}%")
    print("=" * 100)

    periods = [
        ("OOS22-23",  load_period_dfs(DATA_22, UNIV_22), 730),
        ("OOS24-Q1",  load_period_dfs(DATA_24, UNIV_24), 456),
        ("IS25-26",   load_period_dfs(DATA_25, UNIV_25), 374),
    ]

    print(f"\n{'period':<10} {'n_sym':>6} {'taken':>6} {'/day':>6} {'WR%':>6} {'big30':>6} {'big-20':>7} {'avg$':>8} {'win$':>8} {'lose$':>8} {'PnL$':>10} {'PnL%':>8} {'ann%':>8} {'annPnL$':>10}")
    print("-" * 130)

    results = {}
    for plabel, dfs, days in periods:
        entries = gather_long_entries(dfs, THRESHOLD, HOLD_BARS, ATR_GUARD)
        r = portfolio_sim(entries, EQUITY, SIZE_PCT, LEVERAGE, HOLD_BARS, MAX_CONC, days, plabel)
        if r is None:
            print(f"{plabel:<10} {len(dfs):>6}  no entries")
            continue
        results[plabel] = r
        print(f"{plabel:<10} {len(dfs):>6} {r['n_taken']:>6} {r['entries_per_day']:>6.2f} {r['win_rate']*100:>5.1f}% {r['big_wins_30pct']:>6} {r['big_losses_20pct']:>7} {r['avg_per_trade_pnl_usd']:>+7.2f} {r['avg_winner_pnl_usd']:>+7.2f} {r['avg_loser_pnl_usd']:>+7.2f} {r['pnl_usd']:>+9.2f} {r['pnl_pct_of_equity']:>+7.2f}% {r['annual_pct']:>+7.1f}% {r['annual_pnl_usd']:>+9.2f}")

    # weighted avg
    if results:
        total_days = sum(days for plabel, _, days in periods if plabel in results)
        total_taken = sum(r["n_taken"] for r in results.values())
        total_pnl = sum(r["pnl_usd"] for r in results.values())
        # WR weighted by n_taken
        weighted_wr = sum(r["win_rate"] * r["n_taken"] for r in results.values()) / max(total_taken, 1)
        weighted_annual = total_pnl / EQUITY * 100 / total_days * 365 if total_days else 0
        print("-" * 130)
        print(f"{'WEIGHTED':<10} {'-':>6} {total_taken:>6} {total_taken/total_days:>6.2f} {weighted_wr*100:>5.1f}% {'-':>6} {'-':>7} {'-':>8} {'-':>8} {'-':>8} {total_pnl:>+9.2f} {total_pnl/EQUITY*100:>+7.2f}% {weighted_annual:>+7.1f}% {total_pnl/total_days*365:>+9.2f}")

        # User goal check
        print("\n" + "=" * 100)
        print("사용자 목표 충족 검증:")
        print("=" * 100)
        target_winner_usd = 30
        target_wr = 0.70
        target_pnl = 30
        avg_winner = np.mean([r["avg_winner_pnl_usd"] for r in results.values() if r["avg_winner_pnl_usd"] > 0])
        winner_ok = "✓" if avg_winner >= target_winner_usd else "✗"
        wr_ok = "✓" if weighted_wr >= target_wr else "✗"
        pnl_ok = "✓" if total_pnl/total_days*30.4 >= target_pnl else "✗"  # monthly pnl
        annual_pnl = total_pnl/total_days*365
        print(f"  거래당 winner avg ≥ ${target_winner_usd}:  ${avg_winner:.2f}  {winner_ok}")
        print(f"  WR ≥ {target_wr*100:.0f}%:                     {weighted_wr*100:.1f}%  {wr_ok}")
        print(f"  월 PnL ≥ ${target_pnl}:                ${total_pnl/total_days*30.4:.2f}/월  {pnl_ok}  (연 ${annual_pnl:.2f})")
        all_periods_positive = all(r["pnl_usd"] > 0 for r in results.values())
        print(f"  3-period 모두 양수:        {'✓' if all_periods_positive else '✗'}")

    out_path = ROOT / "quant_binance" / "strategies" / "G185_size40_100usd" / "runs" / "validation_3period.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "strategy": "G185",
        "params": {
            "equity_usd": EQUITY, "size_pct": SIZE_PCT, "leverage": LEVERAGE,
            "threshold": THRESHOLD, "hold_bars": HOLD_BARS, "max_conc": MAX_CONC, "atr_guard_pct": ATR_GUARD,
        },
        "periods": results,
    }, indent=2))
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
