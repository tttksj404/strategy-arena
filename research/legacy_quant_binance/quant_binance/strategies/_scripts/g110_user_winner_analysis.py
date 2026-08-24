"""
G110 — 사용자 본인 closed_trades 기록 모두 모아서 winner 패턴 추출.
35+ closed_trades.jsonl 파일 통합 + bitget_realized_winners.json + 분석.
"""
import json, sys, glob
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ARCHIVE = Path.home() / "iCloudDrive" / "quant_archive"

# Find all closed_trades.jsonl
def find_trade_files():
    files = []
    for p in ARCHIVE.rglob("closed_trades.jsonl"):
        files.append(p)
    return files


def load_all_trades():
    files = find_trade_files()
    print(f"Found {len(files)} closed_trades.jsonl files")
    trades = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass
    print(f"Total trades loaded: {len(trades)}")
    return trades


def main():
    trades = load_all_trades()
    if not trades:
        print("NO TRADES FOUND"); return

    df = pd.DataFrame(trades)
    print(f"Columns sample: {list(df.columns)[:20]}")

    # Key fields
    pnl_col = "realized_pnl_net_usd_estimate" if "realized_pnl_net_usd_estimate" in df.columns else "realized_pnl_usd_estimate"
    print(f"Using PnL column: {pnl_col}")

    df["is_winner"] = df[pnl_col] > 0
    print(f"\n=== 전체 ===")
    print(f"  trades: {len(df)}")
    print(f"  winners: {df['is_winner'].sum()} ({df['is_winner'].mean()*100:.1f}%)")
    print(f"  total realized PnL: ${df[pnl_col].sum():.2f}")
    print(f"  avg PnL: ${df[pnl_col].mean():.3f}")
    print(f"  median PnL: ${df[pnl_col].median():.3f}")

    # 부정적 판정 = ground truth winner
    big_winners = df[df[pnl_col] > 1.0].copy()  # net > $1
    print(f"\n=== Big winners (net > $1) — n={len(big_winners)} ===")
    if len(big_winners) > 0:
        print(f"  symbol distribution:")
        for sym, n in big_winners["symbol"].value_counts().head(10).items():
            sub = big_winners[big_winners["symbol"]==sym]
            avg = sub[pnl_col].mean()
            print(f"    {sym}: n={n} avg=${avg:.2f}")
        print(f"  side distribution:")
        for side, n in big_winners["side"].value_counts().items():
            print(f"    {side}: n={n}")
        if "holding_minutes" in big_winners.columns:
            print(f"  hold min: median={big_winners['holding_minutes'].median():.1f} mean={big_winners['holding_minutes'].mean():.1f}")
        if "entry_planned_leverage" in big_winners.columns:
            print(f"  leverage: median={big_winners['entry_planned_leverage'].median():.0f} mean={big_winners['entry_planned_leverage'].mean():.1f}")
        if "entry_predictability_score" in big_winners.columns:
            scores = big_winners["entry_predictability_score"]
            print(f"  entry_predictability_score: median={scores.median():.1f} mean={scores.mean():.1f} min={scores.min():.1f} max={scores.max():.1f}")
        if "peak_roe_percent" in big_winners.columns:
            print(f"  peak_roe: median={big_winners['peak_roe_percent'].median():.2f}% mean={big_winners['peak_roe_percent'].mean():.2f}%")
        if "exit_reason" in big_winners.columns:
            print(f"  exit_reason:")
            for r, n in big_winners["exit_reason"].value_counts().head(5).items():
                print(f"    {r}: {n}")
        if "entry_hour_utc" in big_winners.columns:
            print(f"  entry_hour_utc distribution (top 5):")
            for h, n in big_winners["entry_hour_utc"].value_counts().head(5).items():
                print(f"    UTC {h:02d}: {n}")

    # Manual-closed wins (사용자가 직접 종료한 winner — 진짜 manual 신호)
    manual_winners = df[(df["exit_reason"].str.contains("MANUAL", na=False)) & (df[pnl_col] > 0)]
    print(f"\n=== Manual-closed winners (사용자 직접 종료) — n={len(manual_winners)} ===")
    if len(manual_winners) > 0:
        print(f"  total PnL: ${manual_winners[pnl_col].sum():.2f}")
        for sym, n in manual_winners["symbol"].value_counts().head(5).items():
            sub = manual_winners[manual_winners["symbol"]==sym]
            print(f"    {sym}: n={n} avg=${sub[pnl_col].mean():.2f} avg_hold={sub['holding_minutes'].mean():.0f}min")

    # Big winners 이상 (>$5)
    huge = df[df[pnl_col] > 5.0]
    print(f"\n=== Huge winners (net > $5) — n={len(huge)} ===")
    if len(huge) > 0:
        for _, row in huge.iterrows():
            sym = row.get("symbol","?"); side = row.get("side","?")
            pnl = row[pnl_col]; hold = row.get("holding_minutes",0)
            score = row.get("entry_predictability_score",0)
            lev = row.get("entry_planned_leverage",0)
            print(f"  {sym}/{side}: pnl=${pnl:.2f} hold={hold:.0f}min lev={lev}x score={score:.1f}")

    # save aggregated for next step
    OUT = Path.home() / "Desktop" / "first_repo" / "quant_binance" / "strategies" / "_scripts" / "user_trade_history.json"
    df.to_json(OUT, orient="records", date_format="iso")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
